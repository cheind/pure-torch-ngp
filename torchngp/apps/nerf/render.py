import logging
import hydra
import numpy as np
import io
import torch
from pathlib import Path
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config, load_from_yaml, ZenField as zf
from omegaconf import MISSING, DictConfig, OmegaConf

from torchngp import (
    functional,
    volumes,
    rendering,
    training,
    sampling,
    geometric,
    plotting,
)

_logger = logging.getLogger("torchngp")
logging.getLogger("PIL").setLevel(logging.WARNING)


RenderTaskConf = make_config(
    ckpt=zf(Path, MISSING), output_dir="${hydra:runtime.output_dir}"
)
cs = ConfigStore.instance()
cs.store(name="render_task", node=RenderTaskConf)


@torch.no_grad()
@hydra.main(version_base="1.2", config_path="../../../cfgs/", config_name="render_task")
def render_task(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    _logger.info(f"Loading model from {cfg.ckpt}")
    ckpt_data = torch.load(cfg.ckpt)
    train_cfg = load_from_yaml(io.StringIO(ckpt_data["config"]))

    dev = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    vol: volumes.Volume = instantiate(train_cfg.volume)
    vol.load_state_dict(ckpt_data["volume"])
    vol.to(dev).eval()
    rnd = rendering.RadianceRenderer(
        tsampler=sampling.StratifiedRayStepSampler(n_samples=512)
    )
    N = 20
    poses = functional.spherical_pose(
        theta=torch.linspace(0, 2 * np.pi, N + 1),
        phi=torch.full((N + 1,), np.radians(70)),
        radius=torch.full((N + 1,), 6.0),
    )
    rvec = poses[:-1, :3, :3]
    rvec = functional.so3_log(rvec)
    tvec = poses[:-1, :3, 3]

    cam = geometric.MultiViewCamera(
        focal_length=(1000, 1000),
        principal_point=(255.5, 255.5),
        size=(512, 512),
        rvec=rvec,
        tvec=tvec,
    ).cuda()

    axmin, axmax = tvec.min(), tvec.max()

    ax = plotting.plot_box(vol.aabb)
    plotting.plot_camera(cam, ax)
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.set_zlim(axmin, axmax)
    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.set_aspect("equal")

    import matplotlib.pyplot as plt

    plt.show()

    rgba = training.render_images(
        vol, rnd, cam, None, use_amp=True, n_samples_per_view=cam.size[0]
    )
    _logger.info(f"Saving results to {cfg.output_dir}")
    training.save_image(Path(cfg.output_dir) / "render.png", rgba)

    # python -m torchngp.apps.nerf.render +ckpt=/home/cheind@profactor.local/dev/torch-instant-ngp/outputs/2022-12-01_14-28-40-run/nerf_step_9215.pth


if __name__ == "__main__":
    render_task()
