import time
from itertools import islice

import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

from ngp_nerf.nerf2 import rendering, radiance

from . import sampling
from .cameras import MultiViewScene
from .radiance import NeRF


class MultiViewDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        mvs: MultiViewScene,
        n_samples_per_cam: int = None,
        random: bool = True,
        subpixel: bool = True,
    ):
        self.mvs = mvs
        self.n_pixels_per_cam = self.mvs.cameras.size[0].prod().item()
        if n_samples_per_cam is None:
            # width of first cam (one row)
            n_samples_per_cam = mvs.cameras.size[0, 0].item()
        assert self.n_pixels_per_cam % n_samples_per_cam == 0
        self.n_samples_per_cam = n_samples_per_cam
        self.random = random
        self.subpixel = subpixel if random else False

    def __iter__(self):
        if self.random:
            return islice(
                sampling.generate_random_uv_samples(
                    camera=self.mvs.cameras,
                    image=self.mvs.images,
                    n_samples_per_cam=self.n_samples_per_cam,
                    subpixel=self.subpixel,
                ),
                len(self),
            )
        else:
            return sampling.generate_sequential_uv_samples(
                camera=self.mvs.cameras,
                image=self.mvs.images,
                n_samples_per_cam=self.n_samples_per_cam,
                n_passes=1,
            )

    def __len__(self) -> int:
        # Number of mini-batches required to match with number of total pixels
        return self.n_pixels_per_cam // self.n_samples_per_cam


def train(
    *,
    nerf: radiance.NeRF,
    train_mvs: MultiViewScene,
    test_mvs: MultiViewScene,
    batch_size: int,
    n_ray_step_samples: int = 40,
    lr: float = 1e-2,
    max_train_secs: float = 180,
    dev: torch.device = None,
):
    if dev is None:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nerf.train()
    train_mvs = train_mvs.to(dev)
    test_mvs = test_mvs.to(dev)

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

    n_cams = mvs.cameras.focal_length.shape[0]
    n_worker = 4
    n_samples_per_cam = mvs.cameras.size[0, 0].item()
    final_batch_size = max(batch_size // (n_samples_per_cam * n_cams), 1)

    train_ds = MultiViewDataset(
        train_mvs, n_samples_per_cam=n_samples_per_cam, random=True, subpixel=True
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
                uv = uv.permute(1, 0, 2, 3).reshape(N, B * M, 2).to(dev)
                color = color.permute(1, 0, 2, 3).reshape(N, B * M, C).to(dev)
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
                    train_mvs.cameras,
                    uv,
                    n_ray_steps=n_ray_step_samples,
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
                pbar.set_postfix(**pbar_postfix, refresh=False)
                step += 1
                if (t_now - t_start) > max_train_secs:
                    return

                if t_now - t_last_dump > 30:
                    from . import plotting
                    import matplotlib.pyplot as plt

                    with torch.no_grad():
                        color, alpha = rendering.render_camera(
                            nerf,
                            nerf.aabb,
                            mvs.cameras,
                            n_ray_step_samples=n_ray_step_samples,
                        )
                        img = torch.cat((color, alpha), -1).permute(0, 3, 1, 2)
                        plotting.plot_image(img, checkerboard_bg=False)
                        plt.show()

                    t_last_dump = time.time()

            #
            #     # TODO: merge this block with v.render_image
            #     ts = rays.sample_rays_uniformly(tnear, tfar, n_samples)
            #     xyz = o[:, None] + ts[..., None] * d[:, None]  # (B,T,3)
            #     nxyz = rays.normalize_xyz_in_aabb(xyz, aabb_min, aabb_max)

            #     pred_sample_sigma, pred_sample_color = nerf(nxyz.view(-1, 3))
            #     pred_colors, pred_transm, _ = radiance.integrate_path(
            #         pred_sample_color.view(-1, n_samples, 3),
            #         pred_sample_sigma.view(-1, n_samples),
            #         ts,
            #         tfar,
            #     )
            #     # TODO: the following is not quite correct, should be 1.0 - T(i)*alpha(i)
            #     pred_alpha = 1.0 - pred_transm[:, -1]
            #     pred_colors = pred_colors * pred_alpha[..., None] + noise * (
            #         1 - pred_alpha[..., None]
            #     )

            #     loss = F.mse_loss(pred_colors, color)
            #     if t_now - t_last_dump > 30:
            #         render_test_scenes(
            #             nerf,
            #             test_scene,
            #             dev,
            #             batch_size,
            #             n_samples * 4,
            #             t_now - t_start,
            #             show=False,
            #         )
            #         t_last_dump = t_now

            # loss = scaler.scale(loss)
            # loss.backward()
            # scaler.step(opt)
            # scaler.update()
            # # opt.step()
            # sched.step(loss)
            # pbar_postfix["loss"] = loss.item()
            # pbar_postfix["lr"] = sched._last_lr
            # pbar.set_postfix(**pbar_postfix, refresh=False)
            # step += 1
            # if (t_now - t_start) > max_train_secs:
            #     return


if __name__ == "__main__":
    from . import plotting
    from .io import load_scene_from_json

    import matplotlib.pyplot as plt

    torch.multiprocessing.set_start_method("spawn")

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mvs = load_scene_from_json("data/suzanne/transforms.json", load_images=True)
    plotting.plot_camera(mvs.cameras)
    plotting.plot_box(mvs.aabb)
    plt.show()
    mvs = mvs.to(dev)

    print(mvs.aabb)
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
    nerf = NeRF(aabb=mvs.aabb, **nerf_kwargs).to(dev)

    train_time = 60 * 3
    train(
        nerf=nerf,
        train_mvs=mvs,
        test_mvs=mvs,
        batch_size=2**16,
        n_ray_step_samples=40,
        lr=1e-2,
        max_train_secs=train_time,
        dev=dev,
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
