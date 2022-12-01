import logging
import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config, to_yaml
from omegaconf import MISSING, DictConfig

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


NerfRenderConf = make_config(volume=MISSING, weights=MISSING)
cs = ConfigStore.instance()
cs.store(name="nerf_render", node=NerfRenderConf)


@torch.no_grad()
@hydra.main(version_base="1.2", config_path=None, config_name="nerf")
def render_task(cfg: DictConfig):
    print(to_yaml(cfg))

    vol: volumes.Volume = instantiate(cfg.volume)
    data = torch.load(cfg.weights)
    vol.load_state_dict(data["volume"])
    vol.cuda()
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
    training.save_image("test.png", rgba)

    # python -m torchngp.apps.nerf.render --config-path ~/dev/torch-instant-ngp/outputs/2022-12-01/09-14-17/.hydra/ --config-name config.yaml +weights="outputs/2022-12-01/09-14-17/nerf_step\=639.pth"


if __name__ == "__main__":
    render_task()
