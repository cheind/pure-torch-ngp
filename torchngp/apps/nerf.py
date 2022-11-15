import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import io, radiance, rendering, geometric, sampling, training, filtering

import hydra
from omegaconf import DictConfig, OmegaConf

_logger = logging.getLogger("nerf")


def train(
    *,
    nerf: radiance.NeRF,
    aabb: torch.Tensor,
    train_mvs: tuple[geometric.MultiViewCamera, torch.Tensor],
    test_mvs: tuple[geometric.MultiViewCamera, torch.Tensor],
    accel: filtering.SpatialFilter,
    tsampler: sampling.RayStepSampler,
    renderer: rendering.RadianceRenderer,
    cfg: DictConfig,
    dev: torch.device,
):
    n_rays_batch = cfg.train.n_rays_batch
    n_rays_minibatch = cfg.train.n_rays_minibatch
    n_worker = cfg.train.n_worker
    n_views = train_mvs[0].n_views
    n_acc_steps = n_rays_batch // n_rays_minibatch
    n_samples_per_cam = int(n_rays_minibatch / n_views / n_worker)
    val_interval = int(cfg.train.val_interval / n_rays_batch)

    if cfg.train.preload:
        train_mvs = [x.to(dev) for x in train_mvs]
        test_mvs = [x.to(dev) for x in test_mvs]

    opt = torch.optim.AdamW(
        [
            {
                "params": nerf.pos_encoder.parameters(),
                "weight_decay": cfg.train.opt.decay_encoder,
            },
            {
                "params": nerf.density_mlp.parameters(),
                "weight_decay": cfg.train.opt.decay_density,
            },
            {
                "params": nerf.color_mlp.parameters(),
                "weight_decay": cfg.train.opt.decay_color,
            },
        ],
        betas=cfg.train.opt.betas,
        eps=cfg.train.opt.eps,
        lr=cfg.train.opt.lr,
    )

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.75,
        patience=cfg.train.sched.patience,
        min_lr=cfg.train.sched.min_lr,
    )

    train_ds = training.MultiViewDataset(
        camera=train_mvs[0],
        images=train_mvs[1],
        n_samples_per_cam=n_samples_per_cam,
        random=cfg.train.random_uv,
        subpixel=cfg.train.subpixel_uv,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=n_worker,
        num_workers=n_worker,
    )

    use_amp = cfg.train.use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    fwd_bwd_fn = training.create_fwd_bwd_closure(
        nerf, renderer, tsampler, scaler, n_acc_steps=n_acc_steps
    )

    pbar_postfix = {"loss": 0.0}
    t_start = time.time()
    global_step = 0
    loss_acc = 0.0

    while True:
        pbar = tqdm(train_dl, mininterval=0.1)
        for uv, rgba in pbar:
            t_now = time.time()
            uv = uv.to(dev)
            rgba = rgba.to(dev)

            loss = fwd_bwd_fn(train_mvs[0].to(dev), uv, rgba)
            loss_acc += loss.item()

            if (global_step + 1) % n_acc_steps == 0:
                scaler.step(opt)
                scaler.update()
                sched.step(loss)
                opt.zero_grad(set_to_none=True)

                pbar_postfix["loss"] = loss_acc
                pbar_postfix["lr"] = sched._last_lr[0]
                loss_acc = 0.0

            if ((global_step + 1) % val_interval == 0) and (
                pbar_postfix["loss"] <= cfg.train.val_min_loss
            ):
                render_val_cams = test_mvs[0].to(dev)
                val_rgba = training.render_images(
                    nerf,
                    renderer,
                    render_val_cams,
                    tsampler,
                    use_amp=use_amp,
                    n_samples_per_cam=n_rays_minibatch // render_val_cams.n_views,
                )
                training.save_image(
                    f"tmp/img_val_step={global_step}_elapsed={int(t_now - t_start):03d}.png",
                    val_rgba,
                )
                render_train_cams = train_mvs[0][:2].to(dev)
                train_rgba = training.render_images(
                    nerf,
                    renderer,
                    render_train_cams,
                    tsampler,
                    use_amp=use_amp,
                    n_samples_per_cam=n_rays_minibatch // render_train_cams.n_views,
                )
                training.save_image(
                    f"tmp/img_train_step={global_step}_elapsed={int(t_now - t_start):03d}.png",
                    train_rgba,
                )
                # TODO:this is a different loss than in training
                val_loss = F.mse_loss(val_rgba[:, :3], test_mvs[1].to(dev)[:, :3])
                pbar_postfix["val_loss"] = val_loss.item()

            accel.update(nerf, global_step)
            pbar.set_postfix(**pbar_postfix, refresh=False)
            global_step += 1


def load_dataset(path: str, slicestr: str):
    path = Path(hydra.utils.to_absolute_path(path))
    assert path.exists(), f"Failed to find {path}"
    camera, aabb, images = io.load_scene_from_json(path)
    if slicestr is not None:
        s = _string_to_slice(slicestr)
        camera = camera[s]
        images = images[s]
    return camera, aabb, images


def main(cfg: DictConfig) -> None:
    torch.multiprocessing.set_start_method("spawn")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.getLevelName(cfg.log_level.upper()),
    )

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    camera_train, aabb, images_train = load_dataset(
        cfg.dataset.train.path, cfg.dataset.train.slice
    )

    camera_val, _, images_val = load_dataset(
        cfg.dataset.val.path, cfg.dataset.val.slice
    )

    _logger.info(f"Loaded {camera_train.n_views} train views")
    _logger.info(f"Loaded {camera_val.n_views} val views")

    nerf = radiance.NeRF(**cfg.nerf).to(dev)
    accel = filtering.OccupancyGridFilter(
        res=cfg.occupancy.res,
        update_interval=cfg.occupancy.update_interval,
        stochastic_test=cfg.occupancy.stochastic_test,
        density_initial=cfg.occupancy.density_initial,
    ).to(dev)
    tsampler = sampling.StratifiedRayStepSampler(n_samples=cfg.step_sampler.n_samples)
    renderer = rendering.RadianceRenderer(aabb, accel).to(dev)

    train(
        nerf=nerf,
        aabb=aabb,
        train_mvs=(camera_train, images_train),
        test_mvs=(camera_val, images_val),
        accel=accel,
        tsampler=tsampler,
        renderer=renderer,
        cfg=cfg,
        dev=dev,
    )

    print(OmegaConf.to_yaml(cfg))


def _string_to_slice(sstr):
    # https://stackoverflow.com/questions/43089907/using-a-string-to-define-numpy-array-slice
    return tuple(
        slice(*(int(i) if i else None for i in part.strip().split(":")))
        for part in sstr.strip("[]").split(",")
    )


if __name__ == "__main__":
    from pathlib import Path

    hydra.main(
        version_base=None, config_path=Path.cwd() / "cfgs" / "nerf", config_name="setup"
    )(main)()


# def main():

#     logging.basicConfig(level=logging.INFO)
#     dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     # camera, aabb, gt_images = io.load_scene_from_json(
#     #     "./data/suzanne/transforms.json", load_images=True
#     # )
#     # train_mvs = camera[:-2], gt_images[:-2]
#     # val_mvs = camera[-2:], gt_images[-2:]

#     camera_train, aabb, gt_images_train = io.load_scene_from_json(
#         "./data/lego/transforms_train.json", load_images=True
#     )
#     camera_val, _, gt_images_val = io.load_scene_from_json(
#         "./data/lego/transforms_val.json", load_images=True
#     )
#     train_mvs = (camera_train, gt_images_train)
#     val_mvs = (camera_val[:3], gt_images_val[:3])

#     nerf_kwargs = dict(
#         n_colors=3,
#         n_hidden=64,
#         n_encodings=2**18,
#         n_levels=16,
#         n_color_cond=16,
#         min_res=32,
#         max_res=512,  # can now specify much larger resolutions due to hybrid approach
#         max_n_dense=256**3,
#         is_hdr=False,
#     )
#     nerf = radiance.NeRF(**nerf_kwargs).to(dev)

#     train_time = 10 * 3600
#     train(
#         nerf=nerf,
#         aabb=aabb,
#         train_mvs=train_mvs,
#         test_mvs=val_mvs,
#         n_rays_batch=2**14,
#         n_rays_mini_batch=2**14,
#         lr=1e-2,
#         max_train_secs=train_time,
#         preload=False,
#         dev=dev,
#     )


# if __name__ == "__main__":
#     main()
