from typing import Callable

import torch
import torch.nn

from . import geo
from . import encoding

"""Protocol of a spatial radiance field"""
RadianceField = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def integrate_path(
    color: torch.Tensor, sigma: torch.Tensor, ts: torch.Tensor, tfar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrates the color integral for multiple rays.

    This implementation is a batched version of equations (1,3) in [1].

    Params:
        color: (N,...,T,C) color samples C for (N,...) rays at steps (T,).
        sigma: (N,...,T,1) volume density for each ray (N,...) and step (T,).
        ts: (N,...,T,1) parametric ray step parameters
        tfar: (N,...,1) ray end values

    Returns:
        color: (N,...,C) final colors for each ray
        transmittance: (N,...,T,1) accumulated transmittance for each ray and step
        alpha: (N,...,T,1) alpha transparency values for ray and step

    References:
        [1] NeRF: Representing Scenes as
        Neural Radiance Fields for View Synthesis
        https://arxiv.org/pdf/2003.08934.pdf
    """

    # delta (N,...,T)
    # delta[...,i,0] is defined as the segment length
    # between ts[...,i+1,0] and ts[...,i,0]
    delta = ts[..., 1:, :] - ts[..., :-1, :]  # (N,...,T-1,1)
    delta = torch.cat((delta, tfar.unsqueeze(-2)), -2)  # (N,...,T,1)

    # Alpha values (N,...,T,1)
    sigma_mul_delta = sigma * delta
    alpha = 1.0 - (-sigma_mul_delta).exp()

    # Accumulated transmittance - this is an exclusive cumsum
    # (N,...,T,1)
    acc_transm = sigma_mul_delta.cumsum(-2).roll(1, -2)
    acc_transm[..., 0, 0] = 0
    acc_transm = (-acc_transm).exp()

    final_colors = (acc_transm * alpha * color).sum(-2)

    return final_colors, acc_transm, alpha


class NeRF(torch.nn.Module):
    def __init__(
        self,
        n_colors: int = 3,
        n_hidden: int = 64,
        n_encodings: int = 2**12,
        n_levels: int = 16,
        min_res: int = 32,
        max_res: int = 256,
        max_n_dense: int = 256**3,
        aabb: torch.Tensor = None,
        is_hdr: bool = False,
    ) -> None:
        super().__init__()
        if aabb is None:
            aabb = torch.tensor([[-1.0] * 3, [1.0] * 3])
        self.register_buffer("aabb", aabb)
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
        xn = geo.convert_world_to_box_normalized(x, self.aabb)
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
