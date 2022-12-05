import copy
import dataclasses
import logging
import time
import math
from typing import Optional, Protocol, Literal
from itertools import islice

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from pathlib import Path

from . import (
    geometric,
    rendering,
    sampling,
    volumes,
    radiance,
    config,
    functional,
)

_logger = logging.getLogger("torchngp")


class MultiViewDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        camera: geometric.MultiViewCamera,
        images: torch.Tensor,
        n_samples_per_view: Optional[int] = None,
        mode: Literal["randperm", "random", "sequential"] = "randperm",
        subpixel: bool = True,
    ):
        self.camera = camera
        self.images = images
        self.n_pixels_per_cam = camera.size.prod().item()
        if n_samples_per_view is None:
            # width of image per mini-batch
            n_samples_per_view = camera.size[0].item()  # type: ignore
        self.n_samples_per_view = n_samples_per_view
        self.mode = mode
        self.subpixel = subpixel if mode != "sequential" else False

    def __iter__(self):
        n_worker = torch.utils.data.get_worker_info().num_workers
        if self.mode == "random":
            gen = sampling.generate_random_uv_samples(
                camera=self.camera,
                image=self.images,
                n_samples_per_cam=self.n_samples_per_view,
                subpixel=self.subpixel,
            )
        elif self.mode == "randperm":
            gen = sampling.generate_randperm_uv_samples(
                camera=self.camera,
                image=self.images,
                n_samples_per_cam=self.n_samples_per_view,
                subpixel=self.subpixel,
            )
        elif self.mode == "sequential":
            gen = sampling.generate_sequential_uv_samples(
                camera=self.camera,
                image=self.images,
                n_samples_per_cam=self.n_samples_per_view,
                n_passes=1,
            )
        else:
            raise ValueError(f"Unknown sampling mode {self.mode}.")

        return islice(gen, int(math.ceil(len(self) / n_worker)))  # TODO: doc why

    def __len__(self) -> int:
        # Number of mini-batches required to match with number of total pixels
        return int(math.ceil(self.n_pixels_per_cam / self.n_samples_per_view))


def create_fwd_bwd_closure(
    vol: volumes.Volume,
    renderer: rendering.RadianceRenderer,
    tsampler: Optional[sampling.RayStepSampler],
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


class TrainingsCallback(Protocol):
    def after_training_step(self, trainer: "NeRFTrainer"):
        pass


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
class NeRFTrainer:
    resolved_cfg: str
    volume: volumes.Volume
    train_camera: geometric.MultiViewCamera
    val_camera: Optional[geometric.MultiViewCamera] = None
    train_renderer: Optional[rendering.RadianceRenderer] = None
    val_renderer: Optional[rendering.RadianceRenderer] = None
    output_dir: Path = Path("./tmp")
    train_max_time: float = 60 * 10
    n_rays_batch_log2: int = 14
    n_rays_parallel_log2: int = 14
    n_worker: int = 4
    use_amp: bool = True
    sample_uv_mode: str = "randperm"
    sample_uv_subpixel: bool = True
    preload: bool = False
    dev: Optional[torch.device] = None
    optimizer: OptimizerParams = dataclasses.field(default_factory=OptimizerParams)
    callbacks: list[TrainingsCallback] = dataclasses.field(default_factory=list)
    n_acc_steps: int = dataclasses.field(init=False)
    n_rays_per_view: int = dataclasses.field(init=False)

    def __post_init__(self):
        if self.dev is None:
            self.dev = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        if self.val_camera is None:
            self.val_camera = self.train_camera[:3]
        if self.train_renderer is None:
            self.train_renderer = rendering.RadianceRenderer()
        if self.val_renderer is None:
            self.val_renderer = self.train_renderer
        self.output_dir = Path(self.output_dir)
        self.n_rays_batch = 2**self.n_rays_batch_log2
        self.n_rays_parallel = 2**self.n_rays_parallel_log2
        self.n_acc_steps = self.n_rays_batch // self.n_rays_parallel
        _logger.info(f"Using device {self.dev}")
        _logger.info(f"Output directory set to {self.output_dir.as_posix()}")
        _logger.info(
            f"Processing {self.n_rays_batch} rays per optimizer step with a maximum of"
            f" {self.n_rays_parallel} parallel rays."
        )

    def train(self):

        # Move all relevent modules to device
        self._move_mods_to_device()

        # Train dataloader
        train_dl = self._create_train_dataloader(self.train_camera, self.n_worker)

        # Create optimizers, schedulers
        opt, sched = self._create_optimizers()

        # Setup AMP
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Setup closure for gradient accumulation
        fwd_bwd_fn = create_fwd_bwd_closure(
            self.volume, self.train_renderer, None, scaler, n_acc_steps=self.n_acc_steps
        )

        # Enter main loop
        self.pbar_postfix = {"loss": 0.0}
        self.global_step = 0
        self.current_loss = 0.0
        loss_acc = 0.0
        t_started = time.time()
        t_callbacks_elapsed = 0.0
        while True:
            with logging_redirect_tqdm():
                pbar = tqdm(total=len(train_dl), mininterval=0.1, leave=False)
                for uv, rgba in train_dl:
                    if (
                        time.time() - t_started - t_callbacks_elapsed
                    ) > self.train_max_time:
                        _logger.info("Max training time elapsed.")
                        return
                    uv = uv.to(self.dev)
                    rgba = rgba.to(self.dev)
                    loss = fwd_bwd_fn(self.train_camera, uv, rgba)
                    loss_acc += loss.item()

                    if (self.global_step + 1) % self.n_acc_steps == 0:
                        scaler.step(opt)
                        scaler.update()
                        sched.step(loss)
                        opt.zero_grad(set_to_none=True)
                        self.current_loss = loss_acc
                        self.pbar_postfix["loss"] = self.current_loss
                        self.pbar_postfix["lr"] = sched._last_lr[0]  # type: ignore
                        loss_acc = 0.0

                    t_callbacks_start = time.time()
                    for cb in self.callbacks:
                        cb.after_training_step(self)
                    t_callbacks_elapsed += time.time() - t_callbacks_start

                    pbar.set_postfix(**self.pbar_postfix, refresh=False)
                    pbar.update(1)
                    self.global_step += 1

    def _create_train_dataloader(
        self,
        train_camera: geometric.MultiViewCamera,
        n_worker: int,
    ):
        if not self.preload:
            train_camera = copy.deepcopy(train_camera).cpu()

        n_samples_per_view = int(
            self.n_rays_parallel / self.train_camera.n_views / self.n_worker
        )

        train_ds = MultiViewDataset(
            camera=train_camera,
            images=train_camera.load_images(),
            n_samples_per_view=n_samples_per_view,
            mode=self.sample_uv_mode,
            subpixel=self.sample_uv_subpixel,
        )

        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=n_worker,
            num_workers=n_worker,
        )
        return train_dl

    def _move_mods_to_device(self):
        for m in [
            self.volume,
            self.train_renderer,
            self.val_renderer,
            self.train_camera,
            self.val_camera,
        ]:
            m.to(self.dev)  # changes self.scene to be on device as well?!

    def _create_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        nerf: radiance.NeRF = self.volume.radiance_field  # type: ignore
        opt = torch.optim.AdamW(
            [
                {
                    "params": nerf.pos_encoder.parameters(),
                    "weight_decay": self.optimizer.decay_encoder,
                },
                {
                    "params": nerf.density_mlp.parameters(),
                    "weight_decay": self.optimizer.decay_density,
                },
                {
                    "params": nerf.color_mlp.parameters(),
                    "weight_decay": self.optimizer.decay_color,
                },
            ],
            betas=self.optimizer.betas,
            eps=self.optimizer.eps,
            lr=self.optimizer.lr,
        )

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=self.optimizer.sched_factor,
            patience=self.optimizer.sched_patience,
            min_lr=self.optimizer.sched_minlr,
        )

        return opt, sched


NeRFTrainerConf = config.build_conf(NeRFTrainer)


class IntervalTrainingsCallback(TrainingsCallback):
    def __init__(self, n_rays_interval_log2: int, callback: TrainingsCallback) -> None:
        self.step_interval = None
        self.n_rays_interval_log2 = n_rays_interval_log2
        self.callback = callback

    def after_training_step(self, trainer: "NeRFTrainer"):
        if self.step_interval is None:
            self.step_interval = int(
                math.ceil(
                    max(1.0, 2**self.n_rays_interval_log2 / trainer.n_rays_batch)
                )
            )
        if (trainer.global_step + 1) % self.step_interval == 0:
            self.callback(trainer)


class UpdateSpatialFilterCallback(IntervalTrainingsCallback):
    def __init__(self, n_rays_interval_log2: int) -> None:
        super().__init__(n_rays_interval_log2, callback=self)

    def __call__(self, trainer: NeRFTrainer):
        trainer.volume.spatial_filter.update(trainer.volume.radiance_field)


class ValidationCallback(IntervalTrainingsCallback):
    def __init__(
        self,
        n_rays_interval_log2: int,
        n_rays_parallel_log2: int,
        min_loss: float = 5e-3,
        with_psnr: bool = True,
    ) -> None:
        super().__init__(n_rays_interval_log2, callback=self)
        self.min_loss = min_loss
        self.n_rays_parallel = 2**n_rays_parallel_log2
        self.with_psnr = with_psnr

    @torch.no_grad()
    def __call__(self, trainer: NeRFTrainer):
        if trainer.current_loss > self.min_loss:
            return
        _logger.info(
            f"Validation pass after {trainer.global_step*trainer.n_rays_batch:,} rays"
        )
        rgba = trainer.val_renderer.trace_rgba(
            trainer.volume,
            trainer.val_camera,
            use_amp=trainer.use_amp,
            n_rays_parallel=self.n_rays_parallel,
        )
        functional.save_image(
            rgba,
            trainer.output_dir / f"img_val_step_{trainer.global_step}.png",
            individual=False,
        )

        if self.with_psnr:
            val_rgba = trainer.val_camera.load_images()
            if val_rgba.numel() > 0:
                # TODO how does alpha pixel influence this result?
                psnr, _ = functional.peak_signal_noise_ratio(rgba, val_rgba, 1.0)
                trainer.pbar_postfix["psnr[dB]"] = psnr.mean().item()
                _logger.info(f"psnr[dB]={psnr.mean().item():.2f}")
            else:
                _logger.warn("PSNR computation failed: images not found.")


class ExportCallback(IntervalTrainingsCallback):
    def __init__(
        self, n_rays_interval_log2: int, min_loss: float = 5e-3, config: str = None
    ) -> None:
        super().__init__(n_rays_interval_log2, callback=self)
        self.min_loss = min_loss

    @torch.no_grad()
    def __call__(self, trainer: NeRFTrainer):
        if trainer.current_loss > self.min_loss:
            return
        path = trainer.output_dir / f"nerf_step_{trainer.global_step}.pth"
        _logger.info(f"Model saved to {path.as_posix()}")
        torch.save(
            {
                "volume": trainer.volume.state_dict(),
                "config": trainer.resolved_cfg,
            },
            path,
        )
