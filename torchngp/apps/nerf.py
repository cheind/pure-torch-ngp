import logging
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import io, radiance, rendering, geometric, sampling, training, filtering


def train(
    *,
    nerf: radiance.NeRF,
    aabb: torch.Tensor,
    train_mvs: tuple[geometric.MultiViewCamera, torch.Tensor],
    test_mvs: tuple[geometric.MultiViewCamera, torch.Tensor],
    n_rays_batch: int,
    n_rays_mini_batch: int,
    lr: float = 1e-2,
    max_train_secs: float = 180,
    dev: torch.device = None,
    preload: bool = True,
):
    if dev is None:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nerf = nerf.train().to(dev)
    aabb = aabb.to(dev)
    if preload:
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
        opt, mode="min", factor=0.75, patience=20, min_lr=1e-4
    )

    # compute the number of samples per cam
    n_worker = 4
    n_views = train_mvs[0].n_views
    n_acc_steps = n_rays_batch // n_rays_mini_batch
    n_samples_per_cam = int(n_rays_mini_batch / n_views / n_worker)
    val_interval = int(1e6 / n_rays_batch)

    # n_samples_per_cam = train_mvs[0].size[0].item()
    # final_batch_size = max(batch_size // (n_samples_per_cam * n_views), 1)

    train_ds = training.MultiViewDataset(
        camera=train_mvs[0],
        images=train_mvs[1],
        n_samples_per_cam=n_samples_per_cam,
        random=True,
        subpixel=True,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=n_worker,
        num_workers=n_worker,
    )

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # accel = rendering.OccupancyGridFilter(nerf, dev=dev)
    accel = filtering.OccupancyGridFilter(
        res=64, update_interval=8, stochastic_test=True
    ).to(dev)
    renderer = rendering.RadianceRenderer(aabb, accel).to(dev)
    tsampler = sampling.StratifiedRayStepSampler(n_samples=256)
    fwd_bwd_fn = training.create_fwd_bwd_closure(
        nerf, renderer, tsampler, scaler, n_acc_steps=n_acc_steps
    )

    pbar_postfix = {"loss": 0.0}
    t_start = time.time()

    while True:
        pbar = tqdm(train_dl, mininterval=0.1)
        loss_acc = 0.0
        for idx, (uv, rgba) in enumerate(pbar):
            t_now = time.time()
            uv = uv.to(dev)
            rgba = rgba.to(dev)

            loss = fwd_bwd_fn(train_mvs[0].to(dev), uv, rgba)
            loss_acc += loss.item()

            if (idx + 1) % n_acc_steps == 0:
                scaler.step(opt)
                scaler.update()
                sched.step(loss)
                opt.zero_grad(set_to_none=True)

                pbar_postfix["loss"] = loss_acc
                pbar_postfix["lr"] = sched._last_lr[0]
                loss_acc = 0.0

            if (t_now - t_start) > max_train_secs:
                return

            if (idx + 1) % val_interval == 0:
                render_val_cams = test_mvs[0].to(dev)
                val_rgba = training.render_images(
                    nerf,
                    renderer,
                    render_val_cams,
                    tsampler,
                    use_amp=use_amp,
                    n_samples_per_cam=n_rays_mini_batch // render_val_cams.n_views,
                )
                training.save_image(
                    f"tmp/img_val_step={idx}_elapsed={int(t_now - t_start):03d}.png",
                    val_rgba,
                )
                render_train_cams = train_mvs[0][:2].to(dev)
                train_rgba = training.render_images(
                    nerf,
                    renderer,
                    render_train_cams,
                    tsampler,
                    use_amp=use_amp,
                    n_samples_per_cam=n_rays_mini_batch // render_train_cams.n_views,
                )
                training.save_image(
                    f"tmp/img_train_step={idx}_elapsed={int(t_now - t_start):03d}.png",
                    train_rgba,
                )
                # TODO:this is a different loss than in training
                val_loss = F.mse_loss(val_rgba[:, :3], test_mvs[1].to(dev)[:, :3])
                pbar_postfix["val_loss"] = val_loss.item()

            accel.update(nerf, idx)
            pbar.set_postfix(**pbar_postfix, refresh=False)


def main():
    torch.multiprocessing.set_start_method("spawn")

    logging.basicConfig(level=logging.INFO)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # camera, aabb, gt_images = io.load_scene_from_json(
    #     "./data/suzanne/transforms.json", load_images=True
    # )
    # train_mvs = camera[:-2], gt_images[:-2]
    # val_mvs = camera[-2:], gt_images[-2:]

    camera_train, aabb, gt_images_train = io.load_scene_from_json(
        "./data/lego/transforms_train.json", load_images=True
    )
    camera_val, _, gt_images_val = io.load_scene_from_json(
        "./data/lego/transforms_val.json", load_images=True
    )
    train_mvs = (camera_train, gt_images_train)
    val_mvs = (camera_val[:3], gt_images_val[:3])

    nerf_kwargs = dict(
        n_colors=3,
        n_hidden=64,
        n_encodings=2**18,
        n_levels=16,
        n_color_cond=16,
        min_res=32,
        max_res=512,  # can now specify much larger resolutions due to hybrid approach
        max_n_dense=256**3,
        is_hdr=False,
    )
    nerf = radiance.NeRF(**nerf_kwargs).to(dev)

    train_time = 10 * 3600
    train(
        nerf=nerf,
        aabb=aabb,
        train_mvs=train_mvs,
        test_mvs=val_mvs,
        n_rays_batch=2**14,
        n_rays_mini_batch=2**14,
        lr=1e-2,
        max_train_secs=train_time,
        preload=False,
        dev=dev,
    )


if __name__ == "__main__":
    main()
