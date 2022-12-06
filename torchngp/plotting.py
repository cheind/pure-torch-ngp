import numpy as np
import pytransform3d.camera as pc
import pytransform3d.plot_utils as pu
import pytransform3d.transformations as pt
import torch

from . import modules


def plot_camera(cam: modules.MultiViewCamera, ax=None, **kwargs):
    """Plot camera objects in 3D."""
    if ax is None:
        ax = pu.make_3d_axis(unit="m", ax_s=1.0)
    N = cam.n_views
    E = cam.E.detach().cpu().numpy()
    K = cam.K.detach().cpu().numpy()
    size = cam.size.detach().cpu().numpy()

    for idx in range(N):
        try:
            transform_kwargs = {"linewidth": 0.25, "name": str(idx), **kwargs}
            camera_kwargs = {"linewidth": 0.25, **kwargs}
            pt.plot_transform(A2B=E[idx], s=0.5, ax=ax, **transform_kwargs)
            pc.plot_camera(
                ax=ax,
                cam2world=E[idx],
                M=K,
                sensor_size=size,
                virtual_image_distance=1.0,
                **camera_kwargs,
            )
        except ValueError as e:
            print("failed to display transform " + str(idx) + " " + str(e))
    return ax


def plot_box(aabb: torch.Tensor, ax=None, **kwargs):
    """Plot box in 3D."""
    if ax is None:
        ax = pu.make_3d_axis(unit="m", ax_s=1.0)
    aabb = aabb.detach().cpu().numpy()
    e = np.eye(4)
    e[:3, 3] = (aabb[0] + aabb[1]) * 0.5

    pu.plot_box(ax=ax, size=(aabb[1] - aabb[0]), A2B=e, **kwargs)
    return ax


def plot_world(aabb: torch.Tensor, cam: modules.MultiViewCamera, ax=None):
    axmin, axmax = cam.tvec.min().item(), cam.tvec.max().item()
    axmin = min(axmin, aabb.min().item())
    axmax = max(axmax, aabb.max().item())

    if ax is None:
        ax = pu.make_3d_axis(unit="m", ax_s=1.0)

    plot_box(aabb, ax)
    plot_camera(cam, ax)
    ax.set_xlim(axmin, axmax)
    ax.set_ylim(axmin, axmax)
    ax.set_zlim(axmin, axmax)
    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.set_aspect("equal")
    return ax
