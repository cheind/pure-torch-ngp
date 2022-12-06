import logging
import hydra
import numpy as np
import io
import torch
import dataclasses

from pathlib import Path
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config, load_from_yaml, ZenField as zf, builds
from omegaconf import MISSING, DictConfig, OmegaConf

from torchngp import (
    volumes,
    rendering,
    training,
    geometric,
    plotting,
    functional,
)
from torchngp.functional import uv_sampling

_logger = logging.getLogger("torchngp")
logging.getLogger("PIL").setLevel(logging.WARNING)


@dataclasses.dataclass
class OutputOptions:
    dir: str = "${hydra:runtime.output_dir}"
    render_setup: bool = True
    transparent: bool = True
    grid: bool = False
    fname: str = "image_{idx:03d}.png"


OutputOptionsConf = builds(OutputOptions, populate_full_signature=True)

RenderTaskConf = make_config(
    ckpt=zf(str, MISSING),
    output=OutputOptionsConf(),
    n_rays_parallel_log2=zf(int, 14),
    poses=geometric.SphericalPosesConf(
        20,
        theta_range=(0, 2 * np.pi),
        phi_range=(70 / 180 * np.pi, 70 / 180 * np.pi),
        radius_range=(6.0, 6.0),
        center=(0.0, 0.0, 0.0),
        inclusive=False,
    ),
    camera=geometric.MultiViewCameraConf(
        focal_length=(1000, 1000),
        principal_point=(255.5, 255.5),
        size=(512, 512),
        poses="${poses}",
    ),
    renderer=rendering.RadianceRendererConf(
        tsampler=uv_sampling.StratifiedRayStepSamplerConf(512)
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
    rnd: rendering.RadianceRenderer = instantiate(cfg.renderer).to(dev)
    cam: geometric.MultiViewCamera = instantiate(cfg.camera).to(dev)

    ax = plotting.plot_world(vol.aabb, cam)
    # plt.show()

    _logger.info(f"Rendering, saving results to {cfg.output.dir}")
    rgba = rnd.trace_rgba(
        vol, cam, use_amp=True, n_rays_parallel=2**cfg.n_rays_parallel_log2
    )
    if not cfg.output.transparent:
        rgba = functional.compose_image_alpha(rgba, 1.0)

    functional.save_image(
        rgba,
        str(Path(cfg.output.dir) / cfg.output.fname),
        individual=not cfg.output.grid,
    )

    # python -m torchngp.apps.nerf.render +ckpt=/home/cheind@profactor.local/dev/torch-instant-ngp/outputs/2022-12-01/15-48-06/nerf_step_6143.pth poses.n_poses=10 output.grid=False output.transparent=False
    # ffmpeg -i /home/cheind@profactor.local/dev/torch-instant-ngp/outputs/2022-12-02/11-44-35/image_%03d.png -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 10 output.gif


if __name__ == "__main__":
    render_task()
