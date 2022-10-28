from typing import Callable

import torch
import torch.nn

from . import geometric
from . import encoding

"""Protocol of a spatial radiance field.

A spatial radiance field takes positions and returns color and densities values
for each location.
"""
RadianceField = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def integrate_path(
    color: torch.Tensor, sigma: torch.Tensor, ts: torch.Tensor, tfar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrates the color integral for multiple rays.

    This implementation is a batched version of equations (1,3) in [1].

    Params:
        color: (T,N,...,C) color samples C for (N,...) rays at steps (T,).
        sigma: (T,N,...,1) volume density for each ray (N,...) and step (T,).
        ts: (T,N,...,1) parametric ray step parameters
        tfar: (N,...,1) ray end values

    Returns:
        color: (N,...,C) final colors for each ray
        sample_transmittance: (T,N,...,1) accumulated transmittance
            for each ray and step
        sample_alpha: (T,N,...,1) alpha transparency values for ray and step

    References:
        [1] NeRF: Representing Scenes as
        Neural Radiance Fields for View Synthesis
        https://arxiv.org/pdf/2003.08934.pdf
    """

    # delta[i] is defined as the segment lengths between ts[i+1] and ts[i]
    delta = ts[1:] - ts[:-1]  # (T-1,N,...,1)
    delta = torch.cat((delta, tfar.unsqueeze(0)), 0)  # (T,N,...,1)

    # Alpha values (T,N,...,1)
    sigma_mul_delta = sigma * delta
    alpha = 1.0 - (-sigma_mul_delta).exp()

    # Accumulated transmittance - this is an exclusive cumsum
    # (T,N,...,1)
    acc_transm = sigma_mul_delta.cumsum(0).roll(1, 0)
    acc_transm[0] = 0
    acc_transm = (-acc_transm).exp()

    final_colors = (acc_transm * alpha * color).sum(0)

    return final_colors, acc_transm, alpha


def rasterize_field(
    rf: RadianceField,
    aabb: torch.Tensor,
    resolution: tuple[int, int, int],
    batch_size: int = 2**16,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates the radiance field at rasterized grid locations.

    Note, we rasterize at voxel centers.

    Params:
        nerf: radiance field
        shape: desired resolution of rasterization in each dimension
        dev: computation device
        batch_size: max grid locations per batch

    Returns:
        color: color values with shape `shape + (C,)` and 'xy' indexing
        sigma: density values with shape `shape + (1,)` and 'xy' indexing
    """
    xyz = geometric.make_grid(
        resolution,
        indexing="xy",
        device=device,
        dtype=dtype,
    )
    span = aabb[1] - aabb[0]
    res = aabb.new_tensor(resolution[::-1])
    xyz = aabb[0].expand_as(xyz) + (xyz + 0.5) * (span / res).expand_as(xyz)

    color_parts = []
    sigma_parts = []
    for batch in xyz.split(batch_size):
        rgb, d = rf(batch)
        color_parts.append(rgb)
        sigma_parts.append(d)
    C = color_parts[0].shape[-1]
    color = torch.cat(color_parts, 0).view(resolution + (C,))
    sigma = torch.cat(sigma_parts, 0).view(resolution + (1,))
    return color, sigma


class NeRF(torch.nn.Module):
    """Neural radiance field module.

    Currently supports only spatial features and not view dependent ones.
    """

    def __init__(
        self,
        aabb: torch.Tensor = None,
        n_colors: int = 3,
        n_hidden: int = 64,
        n_encodings: int = 2**12,
        n_levels: int = 16,
        min_res: int = 32,
        max_res: int = 256,
        max_n_dense: int = 256**3,
        is_hdr: bool = False,
    ) -> None:
        super().__init__()
        if aabb is None:
            aabb = torch.tensor([[-1.0] * 3, [1.0] * 3])
        self.register_buffer("aabb", aabb)
        self.aabb: torch.Tensor
        self.is_hdr = is_hdr
        self.pos_encoder = encoding.MultiLevelHybridHashEncoding(
            n_encodings=n_encodings,
            n_input_dims=3,
            n_embed_dims=2,
            n_levels=n_levels,
            min_res=min_res,
            max_res=max_res,
            max_n_dense=max_n_dense,
        )
        n_enc_features = self.pos_encoder.n_levels * self.pos_encoder.n_embed_dims
        # 1-layer hidden density mlp
        self.density_mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_enc_features, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 16),  # 0 density in log-space
        )

        # 2-layer hidden color mlp
        self.color_mlp = torch.nn.Sequential(
            torch.nn.Linear(16, n_hidden),  # TODO: add aux-dims
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_colors),
        )

    def forward(self, x):
        """Predict density and colors for sample positions.

        Params:
            x: (N,...,3) sampling positions in world space

        Returns:
            colors: (N,...,C) predicted color values [0,+inf] (when hdr)
                [0,1] otherwise
            sigma: (N,...,1) prediced density values [0,+inf]
        """
        batch_shape = x.shape[:-1]

        # We query the hash using normalized box coordinates
        xn = geometric.convert_world_to_box_normalized(x, self.aabb)
        # Current encoding implementation does not support more
        # than one batch dimension
        xn_flat = xn.view(-1, 3)
        h = self.pos_encoder(xn_flat)

        # Use the spatial features to estimate density
        d = self.density_mlp(h)

        # Use spatial features + aux. dims to estimate color
        c = self.color_mlp(d)  # TODO: concat aux. dims

        # Transform from linear range to output range
        color = c.exp() if self.is_hdr else torch.sigmoid(c)
        sigma = d[:, 0:1].exp()

        # Reshape to match input batch dims
        color = color.view(batch_shape + (-1,))
        sigma = sigma.view(batch_shape + (1,))

        return color, sigma
