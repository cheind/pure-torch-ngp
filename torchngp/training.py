import copy
import dataclasses
import logging
import time
from itertools import islice

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from tqdm import tqdm

from . import geometric, rendering, sampling, scenes, volumes, plotting

_logger = logging.getLogger("torchngp")


class MultiViewDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        camera: geometric.MultiViewCamera,
        images: torch.Tensor,
        n_rays_per_view: int = None,
        random: bool = True,
        subpixel: bool = True,
    ):
        self.camera = copy.deepcopy(camera)
        self.images = images.clone()
        self.n_pixels_per_cam = camera.size.prod().item()
        if n_rays_per_view is None:
            # width of image per mini-batch
            n_rays_per_view = camera.size[0].item()
        self.n_rays_per_view = n_rays_per_view
        self.random = random
        self.subpixel = subpixel if random else False

    def __iter__(self):
        if self.random:
            return islice(
                sampling.generate_random_uv_samples(
                    camera=self.camera,
                    image=self.images,
                    n_samples_per_cam=self.n_rays_per_view,
                    subpixel=self.subpixel,
                ),
                len(self),
            )
        else:
            return sampling.generate_sequential_uv_samples(
                camera=self.camera,
                image=self.images,
                n_samples_per_cam=self.n_rays_per_view,
                n_passes=1,
            )

    def __len__(self) -> int:
        # Number of mini-batches required to match with number of total pixels
        return self.n_pixels_per_cam // self.n_rays_per_view


def create_fwd_bwd_closure(
    vol: volumes.Volume,
    renderer: rendering.RadianceRenderer,
    tsampler: sampling.RayStepSampler,
    scaler: torch.cuda.amp.GradScaler,
    n_acc_steps: int,
):
    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
    def run_fwd_bwd(
        cam: geometric.MultiViewCamera, uv: torch.Tensor, rgba: torch.Tensor
    ):
        B, N, M, C = rgba.shape
        maps = {"color", "alpha"}

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            uv = uv.permute(1, 0, 2, 3).reshape(N, B * M, 2)
            rgba = rgba.permute(1, 0, 2, 3).reshape(N, B * M, C)
            rgb, alpha = rgba[..., :3], rgba[..., 3:4]
            noise = torch.empty_like(rgb).uniform_(0.0, 1.0)
            # Dynamic noise background with alpha composition
            # Encourages the model to learn zero density in empty regions
            # Dynamic background is also combined with prediced colors, so
            # model does not have to learn randomness.
            gt_rgb_mixed = rgb * alpha + noise * (1 - alpha)

            # Predict
            pred_maps = renderer.trace_uv(vol, cam, uv, tsampler, which_maps=maps)
            pred_rgb, pred_alpha = pred_maps["color"], pred_maps["alpha"]
            # Mix
            pred_rgb_mixed = pred_rgb * pred_alpha + noise * (1 - pred_alpha)

            # Loss normalized by number of accumulation steps before
            # update
            loss = F.smooth_l1_loss(pred_rgb_mixed, gt_rgb_mixed)
            loss = loss / n_acc_steps

        # Scale the loss
        scaler.scale(loss).backward()
        return loss

    return run_fwd_bwd


@torch.no_grad()
def render_images(
    vol: volumes.Volume,
    renderer: rendering.RadianceRenderer,
    cam: geometric.MultiViewCamera,
    tsampler: sampling.RayStepSampler,
    use_amp: bool,
    n_samples_per_cam: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.amp.autocast(enabled=use_amp):
        maps = renderer.trace_maps(
            vol, cam, tsampler=tsampler, n_samples_per_cam=n_samples_per_cam
        )
        pred_rgba = torch.cat((maps["color"], maps["alpha"]), -1).permute(0, 3, 1, 2)
    return pred_rgba


def save_image(fname: str, rgba: torch.Tensor):
    grid_img = (
        (plotting.make_image_grid(rgba, checkerboard_bg=False, scale=1.0) * 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(grid_img, mode="RGBA").save(fname)


@dataclasses.dataclass
class OptimizerParams:
    lr: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-15
    decay_encoder: float = 0.0
    decay_density: float = 1e-6
    decay_color: float = 1e-6
    sched_factor: float = 0.75
    sched_patience: int = 20
    sched_minlr: float = 1e-4


@dataclasses.dataclass
class NeRFTrainerOptions:
    train_cam_idx: int = 0
    train_slice: str = None
    train_max_time: float = 60 * 3
    train_max_epochs: int = 3
    val_cam_idx: int = -1
    val_slice: str = None
    n_rays_batch: int = 2**14
    n_rays_minibatch: int = 2**14
    val_n_rays: int = int(1e6)
    val_min_loss: float = 5e-3
    n_worker: int = 4
    use_amp: bool = True
    random_uv: bool = True
    subpixel_uv: bool = True
    preload: bool = False
    optimizer: OptimizerParams = dataclasses.field(default_factory=OptimizerParams)


class NeRFTrainer:
    def __init__(
        self,
        dev: torch.device = None,
        train_opts: NeRFTrainerOptions = None,
    ):
        if dev is None:
            dev = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        _logger.info(f"Using device {dev}")
        if train_opts is None:
            train_opts = train_opts()
        self.dev = dev
        self.opts = train_opts

    def train(
        self,
        scene: scenes.Scene,
        volume: volumes.Volume,
        renderer: rendering.RadianceRenderer = None,
        tsampler: sampling.RayStepSampler = None,
    ):
        self.scene = scene
        self.volume = volume
        self.renderer = renderer or rendering.RadianceRenderer(ray_extension=1.0)
        self.tsampler = tsampler or sampling.StratifiedRayStepSampler(n_samples=128)

        # Move all relevent modules to device
        self._move_mods_to_device()

        # Locate the cameras to be used for training/validating
        train_cam, val_cam = self._find_cameras()

        # Bookkeeping
        n_acc_steps = self.opts.n_rays_batch // self.opts.n_rays_minibatch
        n_rays_per_view = int(
            self.opts.n_rays_minibatch / train_cam.n_views / self.opts.n_worker
        )
        val_interval = int(self.opts.val_n_rays / self.opts.n_rays_batch)

        # Train dataloader
        train_dl = self._create_train_dataloader(
            train_cam, n_rays_per_view, self.opts.n_worker
        )

        # Create optimizers, schedulers
        opt, sched = self._create_optimizers()

        # Setup AMP
        scaler = torch.cuda.amp.GradScaler(enabled=self.opts.use_amp)

        # Setup closure for gradient accumulation
        fwd_bwd_fn = create_fwd_bwd_closure(
            self.volume, self.renderer, self.tsampler, scaler, n_acc_steps=n_acc_steps
        )

        # Enter main loop
        pbar_postfix = {"loss": 0.0}
        self.global_step = 0
        loss_acc = 0.0
        t_started = time.time()
        for epoch in range(self.opts.train_max_epochs):
            pbar = tqdm(train_dl, mininterval=0.1)
            for uv, rgba in pbar:
                if (time.time() - t_started) > self.opts.train_max_time:
                    return
                loss = fwd_bwd_fn(train_cam, uv, rgba)
                loss_acc += loss.item()

                if (self.global_step + 1) % n_acc_steps == 0:
                    scaler.step(opt)
                    scaler.update()
                    sched.step(loss)
                    opt.zero_grad(set_to_none=True)
                    self.volume.spatial_filter.update(
                        self.volume.radiance_field,
                        global_step=self.global_step,
                    )

                    pbar_postfix["loss"] = loss_acc
                    pbar_postfix["lr"] = sched._last_lr[0]
                    loss_acc = 0.0

                if ((self.global_step + 1) % val_interval == 0) and (
                    pbar_postfix["loss"] <= self.opts.val_min_loss
                ):
                    self.validation_step(
                        val_camera=val_cam, n_rays_per_view=n_rays_per_view
                    )

                pbar.set_postfix(**pbar_postfix, refresh=False)
                self.global_step += 1

    @torch.no_grad()
    def validation_step(
        self, val_camera: geometric.MultiViewCamera, n_rays_per_view: int
    ):
        val_rgba = render_images(
            self.volume,
            self.renderer,
            val_camera,
            self.tsampler,
            self.opts.use_amp,
            n_samples_per_view=n_rays_per_view,
        )
        save_image(f"tmp/img_val_step={self.global_step}.png", val_rgba)
        # TODO:this is a different loss than in training
        # val_loss = F.mse_loss(val_rgba[:, :3], val_scene.images.to(dev)[:, :3])
        # pbar_postfix["val_loss"] = val_loss.item()

    def _create_train_dataloader(
        self, train_cam: geometric.MultiViewCamera, n_rays_per_view: int, n_worker: int
    ):
        if not self.opts.preload:
            train_cam = train_cam.clone().cpu()

        train_ds = MultiViewDataset(
            camera=train_cam,
            images=train_cam.load_images(),
            n_rays_per_view=n_rays_per_view,
            random=self.opts.random_uv,
            subpixel=self.opts.subpixel_uv,
        )

        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=n_worker,
            num_workers=n_worker,
        )
        return train_dl

    def _move_mods_to_device(self):
        for m in [self.scene, self.volume, self.renderer, self.tsampler]:
            m.to(self.dev)  # changes self.scene to be on device as well?!

    def _find_cameras(
        self,
    ) -> tuple[geometric.MultiViewCamera, geometric.MultiViewCamera]:
        train_cam = self.scene.cameras[self.opts.train_cam_idx]
        val_cam = self.scene.cameras[self.opts.val_cam_idx]
        if self.opts.train_slice is not None:
            train_cam = train_cam[self.opts.train_slice]
        if self.opts.val_slice is not None:
            val_cam = val_cam[self.opts.val_slice]

    def _create_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        nerf = self.volume.radiance_field
        opt = torch.optim.AdamW(
            [
                {
                    "params": nerf.pos_encoder.parameters(),
                    "weight_decay": self.opts.optimizer.decay_encoder,
                },
                {
                    "params": nerf.parameters(),
                    "weight_decay": self.opts.optimizer.decay_density,
                },
                {
                    "params": nerf.parameters(),
                    "weight_decay": self.opts.optimizer.decay_color,
                },
            ],
            betas=self.opts.optimizer.betas,
            eps=self.opts.optimizer.eps,
            lr=self.opts.optimizer.lr,
        )

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=self.opts.optimizer.sched_factor,
            patience=self.opts.optimizer.sched_patience,
            min_lr=self.opts.optimizer.sched_minlr,
        )

        return opt, sched
