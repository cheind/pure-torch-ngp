import torch
from typing import Callable

from . import pixels

"""Protocol of a spatial radiance field"""
RadianceField = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def integrate_path(
    color: torch.Tensor, sigma: torch.Tensor, ts: torch.Tensor, tfar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrates the color integral for radiance fields for multiple rays.

    This implementation is a batched version of equations (1,3) in [1].

    Params:
        color: (B,T,C) for each ray b and each sample t a color sample c
        sigma: (B,T) volume density for each ray b and each sample t
        ts: (B,T) parametric ray (b) sampling positions t
        tfar: (B,) ray exit values

    Returns:
        color: (B,C) final colors for each ray
        transmittance: (B,T) accumulated transmittance for each ray/sample
        alpha: (B,T) alpha transparency values for ray/sample position

    References:
        [1] NeRF: Representing Scenes as
        Neural Radiance Fields for View Synthesis
        https://arxiv.org/pdf/2003.08934.pdf
    """

    B, T, C = color.shape

    # deltae[:,i] is the segment length between tsample[:,i+1] and tsample[:,i]
    # TODO: the path difference for the last sample is zero, leads
    # to last color being ignored.
    delta = ts[:, 1:] - ts[:, :-1]  # (B,T-1)
    delta = torch.cat((delta, tfar[:, None]), -1)  # (B,T)

    # Alpha values
    sigma_mul_delta = sigma * delta
    alpha = 1.0 - (-sigma_mul_delta).exp()

    # Accumulated transmittance - this is an exclusive cumsum
    acc_transm = sigma_mul_delta.cumsum(-1).roll(1, -1)
    acc_transm[:, 0] = 0
    acc_transm = (-acc_transm).exp()

    final_colors = (acc_transm[..., None] * alpha[..., None] * color).sum(1)

    return final_colors, acc_transm, alpha


def rasterize_field(
    nerf: RadianceField,
    shape: tuple[int, ...],
    dev: torch.device,
    batch_size: int = 2**16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates the radiance field at rasterized grid locations.

    Params:
        nerf: radiance field
        shape: desired resolution of rasterization in each dimension
        dev: computation device
        batch_size: max grid locations per batch

    Returns:
        sigma: density values with shape `shape`
        rgb: color values with shape `shape + (3,)`
    """
    xyz = pixels.generate_grid_coords(shape)
    nxyz = pixels.normalize_coords(xyz).view(-1, 3)

    rgbs = []
    densities = []
    for batch in nxyz.split(batch_size):
        d, rgb = nerf(batch.to(dev))
        rgbs.append(rgb)
        densities.append(d)
    rgbs = torch.cat(rgbs, 0).view(*shape, 3)
    densities = torch.cat(densities, 0).view(*shape)
    return densities, rgbs
