"""Low-level image helpers."""

from typing import Union
from PIL import Image
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


def checkerboard_image(
    shape: tuple[int, int, int, int],
    k: int = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Generates a checkerboard image.

    Useful as background for transparent images.
    Adapted from https://stackoverflow.com/questions/72874737

    Params:
        shape: (N,C,H,W) shape to generate
        k: size of square
        dtype: data type
        device: compute device

    Returns:
        rgba: image filled with checkerboard pattern

    """
    assert shape[1] in [1, 3, 4]
    # nearest h,w multiple of k
    k = k or max(max(shape[-2:]) // 20, 1)
    H = shape[2] + (k - shape[2] % k)
    W = shape[3] + (k - shape[3] % k)
    indices = torch.stack(
        torch.meshgrid(
            torch.arange(H // k, dtype=dtype, device=device),
            torch.arange(W // k, dtype=dtype, device=device),
            indexing="ij",
        )
    )
    base = indices.sum(dim=0) % 2
    x = base.repeat_interleave(k, 0).repeat_interleave(k, 1)
    x = x[: shape[2], : shape[3]]

    if shape[1] in [1, 3]:
        x = x.unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], -1, -1).to(dtype)
    else:
        x = x.unsqueeze(0).unsqueeze(0).expand(shape[0], 3, -1, -1).to(dtype)
        x = torch.cat(
            (
                x,
                torch.ones(
                    (shape[0], 1, shape[2], shape[3]),
                    device=device,
                    dtype=dtype,
                ),
            ),
            1,
        )
    return x


def constant_image(
    shape: tuple[int, int, int, int],
    c: Union[torch.Tensor, tuple[float, float, float, float]],
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Returns an image of constant channel values."""
    c = torch.as_tensor(c, dtype=dtype, device=device)
    return c.view(1, -1, 1, 1).expand(*shape)


def compose_image_alpha(rgba: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Performs alpha composition on inputs"""
    alpha = rgba[:, 3:4]
    c = rgba[:, :3] * alpha + (1 - alpha) * rgb
    return torch.cat((c, c.new_ones(rgba.shape[0], 1, rgba.shape[2], rgba.shape[3])), 1)


def scale_image(
    rgba: torch.Tensor, scale: float, mode: str = "bilinear"
) -> torch.Tensor:
    """Scale image by factor."""
    return F.interpolate(
        rgba,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def create_image_grid(rgba: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """Convert batch images to grid"""
    return make_grid(rgba, padding=padding).unsqueeze(0)


def save_image(rgba: torch.Tensor, outpath: str, individual: bool = False):
    """Save images."""
    if not individual:
        rgba = create_image_grid(rgba, padding=0)
    rgba = (rgba * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for idx, img in enumerate(rgba):
        outp = str(outpath).format(idx=idx)
        Image.fromarray(img, mode="RGBA").save(outp)


def load_image(
    paths: list[Path],
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Load images associated with this camera."""
    assert len(paths) > 0
    loaded = []
    for path in paths:
        img = Image.open(path).convert("RGBA")
        img = (
            torch.tensor(np.asarray(img), dtype=dtype, device=device).permute(2, 0, 1)
            / 255.0
        )
        loaded.append(img)
    return torch.stack(loaded, 0)


__all__ = [
    "checkerboard_image",
    "constant_image",
    "scale_image",
    "create_image_grid",
    "save_image",
    "load_image",
    "compose_image_alpha",
]
