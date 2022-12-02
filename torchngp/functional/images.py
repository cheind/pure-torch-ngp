"""Low-level image helpers."""

from typing import Union
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


def checkerboard_image(
    shape: tuple[int, int, int, int],
    k: int = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Returns a checkerboard background image.
    See https://stackoverflow.com/questions/72874737
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
) -> torch.Tensor:
    c = torch.as_tensor(c)
    return c.view(1, -1, 1, 1).expand(*shape)


def scale_image(
    rgba: torch.Tensor, scale: float, mode: str = "bilinear"
) -> torch.Tensor:
    return F.interpolate(
        rgba,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def create_image_grid(rgba: torch.Tensor, padding: int = 2) -> torch.Tensor:
    return make_grid(rgba, padding=padding).unsqueeze(0)


def save_image(rgba: torch.Tensor, outpath: str, individual: bool = False):
    if not individual:
        rgba = create_image_grid(rgba, padding=0)
    rgba = (rgba * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for idx, img in enumerate(rgba):
        outp = outpath.format(idx=idx)
        Image.fromarray(img, mode="RGBA").save(outp)


__all__ = [
    "checkerboard_image",
    "constant_image",
    "scale_image",
    "create_image_grid",
    "save_image",
]
