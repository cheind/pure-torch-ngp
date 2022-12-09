import logging
import hydra
import numpy as np
import io
import torch

from pathlib import Path
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config, load_from_yaml, ZenField as zf
from omegaconf import MISSING, DictConfig, OmegaConf

from torchngp import modules
from torchngp import functional
from torchngp import config
from torchngp import helpers
from torchngp import plotting

_logger = logging.getLogger("torchngp")
logging.getLogger("PIL").setLevel(logging.WARNING)


SphericalPosesConf = config.build_conf(helpers.spherical_poses)

RenderTaskConf = make_config(
    ckpt=zf(str, MISSING),
    out_dir=zf(str, "${hydra:runtime.output_dir}"),
    rgba_fname="rgba_{idx:03d}.png",
    depth_fname="depth_{idx:03d}.png",
    rgba_transparent=False,
    depth_dynamic_range=False,
    as_grid=False,
    n_rays_parallel_log2=zf(int, 14),
    poses=SphericalPosesConf(
        20,
        theta_range=(0, 2 * np.pi),
        phi_range=(70 / 180 * np.pi, 70 / 180 * np.pi),
        radius_range=(6.0, 6.0),
        center=(0.0, 0.0, 0.0),
        inclusive=False,
    ),
    camera=modules.MultiViewCameraConf(
        focal_length=(1000, 1000),
        principal_point=(255.5, 255.5),
        size=(512, 512),
        poses="${poses}",
        tnear=0,
        tfar=10,
    ),
    renderer=modules.RadianceRendererConf(
        tsampler=modules.StratifiedRayStepSamplerConf(512)
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
    vol: modules.Volume = instantiate(train_cfg.volume)
    vol.load_state_dict(ckpt_data["volume"])
    vol.to(dev).eval()
    rnd: modules.RadianceRenderer = instantiate(cfg.renderer).to(dev)
    cam: modules.MultiViewCamera = instantiate(cfg.camera).to(dev)

    # ax = plotting.plot_world(vol.aabb, cam)
    # plt.show()

    _logger.info(f"Rendering, saving results to {cfg.out_dir}")
    rgbad = rnd.trace(
        vol, cam, use_amp=True, n_rays_parallel=2**cfg.n_rays_parallel_log2
    )
    rgba = rgbad[:, :4]
    depth = rgbad[:, 4:5]
    _logger.info(
        f"Min depth {depth.min().item():.3f}, max depth {depth.max().item():.3f}"
    )
    if cfg.depth_dynamic_range:
        depth = functional.scale_depth(depth, depth.min(), depth.max())
    else:
        depth = functional.scale_depth(depth, cam.tnear, cam.tfar)

    if not cfg.rgba_transparent:
        rgba = functional.compose_image_alpha(rgba, 1.0)

    functional.save_image(
        rgba,
        str(Path(cfg.out_dir) / cfg.rgba_fname),
        individual=not cfg.as_grid,
    )

    functional.save_image(
        depth,
        str(Path(cfg.out_dir) / cfg.depth_fname),
        individual=not cfg.as_grid,
    )

    # python -m torchngp.apps.nerf.render ckpt=/home/cheind@profactor.local/dev/torch-instant-ngp/outputs/2022-12-09/10-47-28/nerf_step_4095.pth poses.n_poses=60 as_grid=False rgba_transparent=False depth_dynamic_range=True
    # ffmpeg -i /home/cheind@profactor.local/dev/torch-instant-ngp/outputs/2022-12-02/11-44-35/rgba_%03d.png -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 10 output.gif


if __name__ == "__main__":
    render_task()
