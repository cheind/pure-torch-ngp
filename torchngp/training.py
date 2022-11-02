import time
from itertools import islice

import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

from torchngp import rendering, radiance, cameras, sampling


class MultiViewDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        camera: cameras.MultiViewCamera,
        images: torch.Tensor,
        n_samples_per_cam: int = None,
        random: bool = True,
        subpixel: bool = True,
    ):
        self.camera = camera
        self.images = images
        self.n_pixels_per_cam = camera.size.prod().item()
        if n_samples_per_cam is None:
            # width of image per mini-batch
            n_samples_per_cam = camera.size[0].item()
        self.n_samples_per_cam = n_samples_per_cam
        self.random = random
        self.subpixel = subpixel if random else False

    def __iter__(self):
        if self.random:
            return islice(
                sampling.generate_random_uv_samples(
                    camera=self.camera,
                    image=self.images,
                    n_samples_per_cam=self.n_samples_per_cam,
                    subpixel=self.subpixel,
                ),
                len(self),
            )
        else:
            return sampling.generate_sequential_uv_samples(
                camera=self.camera,
                image=self.images,
                n_samples_per_cam=self.n_samples_per_cam,
                n_passes=1,
            )

    def __len__(self) -> int:
        # Number of mini-batches required to match with number of total pixels
        return self.n_pixels_per_cam // self.n_samples_per_cam


def train(
    *,
    nerf: radiance.NeRF,
    train_mvs: tuple[cameras.MultiViewCamera, torch.Tensor],
    test_mvs: tuple[cameras.MultiViewCamera, torch.Tensor],
    batch_size: int,
    n_ray_step_samples: int = 40,
    lr: float = 1e-2,
    max_train_secs: float = 180,
    dev: torch.device = None,
):
    if dev is None:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nerf = nerf.train().to(dev)
    train_mvs = [x.to(dev) for x in train_mvs]
    test_mvs = [x.to(dev) for x in test_mvs]

    opt = torch.optim.AdamW(
        [
            {"params": nerf.pos_encoder.parameters(), "weight_decay": 0.0},
            {"params": nerf.density_mlp.parameters(), "weight_decay": 1e-6},
            {"params": nerf.color_mlp.parameters(), "weight_decay": 1e-6},
        ],
        betas=(0.9, 0.99),
        eps=1e-15,
        lr=lr,
    )

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.75, patience=40, min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()

    n_views = train_mvs[0].n_views
    n_worker = 4
    n_samples_per_cam = train_mvs[0].size[0].item()
    final_batch_size = max(batch_size // (n_samples_per_cam * n_views), 1)

    train_ds = MultiViewDataset(
        camera=train_mvs[0],
        images=train_mvs[1],
        n_samples_per_cam=n_samples_per_cam,
        random=True,
        subpixel=True,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=final_batch_size,
        num_workers=n_worker,
    )

    pbar_postfix = {"loss": 0.0}
    step = 0
    t_start = time.time()
    t_last_dump = time.time()
    while True:
        pbar = tqdm(train_dl, mininterval=0.1)
        for (uv, color) in pbar:
            B, N, M, C = color.shape
            # uv is (B,N,M,2) N=num cams, M=minibatch
            # uf_features is (B,N,M,C)
            t_now = time.time()
            opt.zero_grad()

            with torch.cuda.amp.autocast():
                uv = uv.permute(1, 0, 2, 3).reshape(N, B * M, 2)
                color = color.permute(1, 0, 2, 3).reshape(N, B * M, C)
                rgb, alpha = color[..., :3], color[..., 3:4]

                noise = torch.empty_like(rgb).uniform_(0.0, 1.0)
                # Dynamic noise background with alpha composition
                # Encourages the model to learn zero density in empty regions
                # Dynamic background is also combined with prediced colors, so
                # model does not have to learn randomness.
                gt_rgb_mixed = rgb * alpha + noise * (1 - alpha)

                pred_rgb, pred_alpha = rendering.render_volume_stratified(
                    nerf,
                    nerf.aabb,
                    train_mvs[0],
                    uv,
                    n_ray_t_steps=n_ray_step_samples,
                    boost_tfar=10.0,
                )
                pred_rgb_mixed = pred_rgb * pred_alpha + noise * (1 - pred_alpha)

                loss = F.mse_loss(pred_rgb_mixed, gt_rgb_mixed)

                loss = scaler.scale(loss)
                loss.backward()
                scaler.step(opt)
                scaler.update()
                # opt.step()
                sched.step(loss)
                pbar_postfix["loss"] = loss.item()
                pbar_postfix["lr"] = sched._last_lr
                step += 1
                if (t_now - t_start) > max_train_secs:
                    return

                if t_now - t_last_dump > 30:
                    from . import plotting
                    import matplotlib.pyplot as plt

                    with torch.no_grad():
                        pred_color, pred_alpha = rendering.render_camera_views(
                            test_mvs[0],
                            nerf,
                            nerf.aabb,
                            n_ray_step_samples=n_ray_step_samples,
                        )
                        pred_img = torch.cat((pred_color, pred_alpha), -1).permute(
                            0, 3, 1, 2
                        )
                        val_loss = F.mse_loss(pred_img, test_mvs[1])
                        pbar_postfix["val_loss"] = val_loss.item()
                        plotting.plot_image(pred_img, checkerboard_bg=True)
                        plt.show()

                    t_last_dump = time.time()

                pbar.set_postfix(**pbar_postfix, refresh=False)


if __name__ == "__main__":
    from . import plotting
    from .io import load_scene_from_json

    import matplotlib.pyplot as plt

    torch.multiprocessing.set_start_method("spawn")

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    camera, aabb, gt_images = load_scene_from_json(
        "data/suzanne/transforms.json", load_images=True
    )
    plotting.plot_camera(camera)
    plotting.plot_box(aabb)
    plt.gca().set_aspect("equal")
    plt.gca().autoscale()
    plt.show()

    # ds = MultiViewDataset(mvs)

    nerf_kwargs = dict(
        n_colors=3,
        n_hidden=64,
        n_encodings=2**14,
        n_levels=16,
        min_res=16,
        max_res=512,  # can now specify much larger resolutions due to hybrid approach
        max_n_dense=256**3,
        is_hdr=False,
    )
    nerf = radiance.NeRF(aabb=aabb, **nerf_kwargs).to(dev)

    train_time = 3 * 60
    train(
        nerf=nerf,
        train_mvs=(camera[:-1], gt_images[:-1]),
        test_mvs=(camera[-1:], gt_images[-1:]),
        batch_size=2**16,
        n_ray_step_samples=40,
        lr=1e-2,
        max_train_secs=train_time,
        dev=dev,
    )

    with torch.no_grad(), torch.cuda.amp.autocast():
        import numpy as np

        t = time.time()
        vol_colors, vol_sigma = radiance.rasterize_field(
            nerf, nerf.aabb, (512, 512, 512), batch_size=2**18, device=dev
        )
        vol_rgbd = torch.cat((vol_colors, vol_sigma), -1).cpu()
        print(time.time() - t)
        np.savez(
            "tmp/volume.npz",
            rgb=vol_rgbd[..., :3],
            d=vol_rgbd[..., 3],
            aabb=nerf.aabb.cpu(),
        )

    # # ax = plotting.plot_camera(mvs.cameras)
    # # ax = plotting.plot_box(mvs.aabb)
    # # ax.set_aspect("equal")
    # # ax.relim()  # make sure all the data fits
    # # ax.autoscale()
    # import matplotlib.pyplot as plt

    # # plt.show()
    # img = torch.empty((2, 4, 30, 40)).uniform_(0.0, 1.0)
    # plotting.plot_image(img, scale=0.5)
    # plt.show()

    # # dl = torch.utils.data.DataLoader(ds, batch_size=4)
    # # print(len(dl))
    # # for idx, (uv, uv_f) in enumerate(dl):
    # #     print(idx, uv.shape)
    # # print(idx)
