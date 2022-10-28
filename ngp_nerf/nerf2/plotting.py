import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as pu

from torchvision.utils import make_grid


from .cameras import MultiViewCamera


def plot_camera(cam: MultiViewCamera, ax=None, **kwargs):
    """Plot camera objects in 3D."""
    if ax is None:
        ax = pu.make_3d_axis(unit="m", ax_s=1.0)
    N = cam.n_views
    E = cam.E.detach().cpu().numpy()
    K = cam.K.detach().cpu().numpy()
    size = cam.size.detach().cpu().numpy()

    for idx in range(N):
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


def _checkerboard(shape: tuple[int, ...], k: int = None) -> torch.Tensor:
    """Returns a checkerboar background.
    See https://stackoverflow.com/questions/72874737
    """
    # nearest h,w multiple of k
    k = k or max(max(shape) // 20, 1)
    H = shape[0] + (k - shape[0] % k)
    W = shape[1] + (k - shape[1] % k)
    indices = torch.stack(
        torch.meshgrid(torch.arange(H // k), torch.arange(W // k), indexing="ij")
    )
    base = indices.sum(dim=0) % 2
    x = base.repeat_interleave(k, 0).repeat_interleave(k, 1)
    return x[: shape[0], : shape[1]]


def plot_image(
    img: torch.Tensor,
    checkerboard_bg: bool = True,
    scale: float = 1.0,
    ax=None,
):
    H, W = img.shape[-2:]

    if img.shape[1] == 4:
        img = img.detach().clone()
        rgb = img[:, :3]
        alpha = img[:, 3:4]
        if checkerboard_bg:
            bg = _checkerboard((H, W)).expand_as(rgb).to(rgb.device)
        else:
            bg = torch.zeros_like(rgb)

        rgb[:] = rgb * alpha + (1 - alpha) * bg

    if scale != 1.0:
        img = F.interpolate(
            img,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )

    grid = make_grid(img)
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(grid[:3].permute(1, 2, 0).detach().cpu().numpy())
    return ax
