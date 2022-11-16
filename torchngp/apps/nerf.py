import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import io, radiance, rendering, sampling, training, filtering


_logger = logging.getLogger("nerf")

logging.getLogger("PIL").setLevel(logging.WARNING)


def train(cfg: DictConfig):

    torch.multiprocessing.set_start_method("spawn")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.getLevelName(cfg.train.log_level.upper()),
    )
    _logger.debug("YAML Config")
    _logger.debug(OmegaConf.to_yaml(cfg))

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _logger.info(f"Using device {dev}")

    # Load data
    assert "train" in cfg.datasets
    train_scene: io.MultiViewScene = hydra.utils.instantiate(
        cfg.datasets.train,
    )

    if "val" in cfg.datasets:
        val_scene: io.MultiViewScene = hydra.utils.instantiate(
            cfg.datasets.val,
        )
    else:
        _logger.info("No validation dataset specified.")
        val_scene = train_scene

    if cfg.train.preload:
        train_scene = train_scene.to(dev)
        val_scene = val_scene.to(dev)

    if cfg.setup.aabb is not None:
        _logger.info("Overridding AABB from transforms.json with setup.aabb")
        aabb = torch.tensor([cfg.setup.aabb.minc, cfg.setup.aabb.maxc]).float()
    else:
        aabb = train_scene.aabb

    _logger.info(f"Loaded {train_scene.camera.n_views} train views")
    _logger.info(f"Loaded {val_scene.camera.n_views} val views")
    _logger.info(f"AABB minc={aabb[0]}, maxc={aabb[1]}")

    # Create objects
    nerf: radiance.NeRF = (
        hydra.utils.instantiate(
            cfg.setup.nerf,
        )
        .train()
        .to(dev)
    )
    sfilter: filtering.SpatialFilter = (
        hydra.utils.instantiate(
            cfg.setup.spatial_filter,
        )
        .train()
        .to(dev)
    )
    tsampler: sampling.RayStepSampler = hydra.utils.instantiate(
        cfg.setup.ray_tsampler,
    )
    renderer = rendering.RadianceRenderer(aabb, sfilter).to(dev)

    # Determine batch size

    n_rays_batch = cfg.train.n_rays_batch
    n_rays_minibatch = cfg.train.n_rays_minibatch
    n_worker = cfg.train.n_worker
    n_views = train_scene.camera.n_views
    n_acc_steps = n_rays_batch // n_rays_minibatch
    n_samples_per_cam = int(n_rays_minibatch / n_views / n_worker)
    val_interval = int(cfg.train.val_interval / n_rays_batch)

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
        factor=cfg.train.sched.factor,
        patience=cfg.train.sched.patience,
        min_lr=cfg.train.sched.min_lr,
    )

    train_ds = training.MultiViewDataset(
        camera=train_scene.camera,
        images=train_scene.images,
        n_samples_per_cam=n_samples_per_cam,
        random=cfg.train.random_uv,
        subpixel=cfg.train.subpixel_uv,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=n_worker,
        num_workers=n_worker,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)
    fwd_bwd_fn = training.create_fwd_bwd_closure(
        nerf, renderer, tsampler, scaler, n_acc_steps=n_acc_steps
    )

    pbar_postfix = {"loss": 0.0}
    t_start = time.time()
    global_step = 0
    loss_acc = 0.0

    train_camera_dev = train_scene.camera.to(dev)
    val_camera_dev = val_scene.camera.to(dev)

    while True:
        pbar = tqdm(train_dl, mininterval=0.1)
        for uv, rgba in pbar:
            t_now = time.time()
            uv = uv.to(dev)
            rgba = rgba.to(dev)

            loss = fwd_bwd_fn(train_camera_dev, uv, rgba)
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
                val_rgba = training.render_images(
                    nerf,
                    renderer,
                    val_camera_dev,
                    tsampler,
                    use_amp=cfg.train.use_amp,
                    n_samples_per_cam=n_rays_minibatch // val_camera_dev.n_views,
                )
                training.save_image(
                    f"tmp/img_val_step={global_step}_elapsed={int(t_now - t_start):03d}.png",
                    val_rgba,
                )
                render_train_cams = train_camera_dev[:2].to(dev)
                train_rgba = training.render_images(
                    nerf,
                    renderer,
                    render_train_cams,
                    tsampler,
                    use_amp=cfg.train.use_amp,
                    n_samples_per_cam=n_rays_minibatch // render_train_cams.n_views,
                )
                training.save_image(
                    f"tmp/img_train_step={global_step}_elapsed={int(t_now - t_start):03d}.png",
                    train_rgba,
                )
                # TODO:this is a different loss than in training
                val_loss = F.mse_loss(val_rgba[:, :3], val_scene.images.to(dev)[:, :3])
                pbar_postfix["val_loss"] = val_loss.item()

            sfilter.update(nerf, global_step)
            pbar.set_postfix(**pbar_postfix, refresh=False)
            global_step += 1


if __name__ == "__main__":
    hydra.main(
        version_base=None,
        config_path=Path.cwd() / "cfgs" / "nerf",
        config_name="train",
    )(train)()
