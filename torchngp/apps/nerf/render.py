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
    ckpt=zf(str, MISSING),
    output_dir="${hydra:runtime.output_dir}",
    camera=geometric.MultiViewCameraConf(
        focal_length=(1000, 1000),
        principal_point=(255.5, 255.5),
        size=(512, 512),
        poses=geometric.SphericalPosesConf(
            20,
            theta_range=(0, 2 * np.pi),
            phi_range=(70 / 180 * np.pi, 70 / 180 * np.pi),
            radius_range=(6.0, 6.0),
            center=(0.0, 0.0, 0.0),
            inclusive=False,
        ),
    ),
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
    cam: geometric.MultiViewCamera = instantiate(cfg.camera).cuda()

    ax = plotting.plot_world(vol.aabb, cam)

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
