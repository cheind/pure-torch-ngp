from typing import Optional
import torch

from .. import config
from . import protocols
from . import encoding


class NeRF(torch.nn.Module, protocols.RadianceField):
    """Neural radiance field module.

    Currently supports only spatial features and not view dependent ones.
    """

    def __init__(
        self,
        n_colors: int = 3,
        n_hidden: int = 64,
        n_encodings_log2: int = 16,
        n_levels: int = 16,
        n_color_cond: int = 15,
        min_res: int = 32,
        max_res: int = 512,
        max_res_dense: int = 256,
        is_hdr: bool = False,
    ) -> None:
        super().__init__()
        self.is_hdr = is_hdr
        self.n_color_cond_dims = n_color_cond
        self.n_color_dims = n_colors
        self.n_density_dims = 1
        self.pos_encoder = encoding.MultiLevelHybridHashEncoding(
            n_encodings=2**n_encodings_log2,
            n_input_dims=3,
            n_embed_dims=2,
            n_levels=n_levels,
            min_res=min_res,
            max_res=max_res,
            max_n_dense=max_res_dense**3,
        )
        n_enc_features = self.pos_encoder.n_levels * self.pos_encoder.n_embed_dims
        # 1-layer hidden density mlp
        self.density_mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_enc_features, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 16),  # 0 index is density in log-space
        )

        # 2-layer hidden color mlp
        self.color_mlp = torch.nn.Sequential(
            torch.nn.Linear(16 + n_color_cond, n_hidden),  # TODO: add aux-dims
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_colors),
        )

        torch.nn.init.constant_(self.density_mlp[-1].bias[:1], -1.0)

        # print(sum(param.numel() for param in self.parameters()))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return features from positions.

        Params:
            x: (N,...,3) positions in NDC [-1,1]

        Returns:
            f: (N,...,16) feature values for each sample point
        """
        batch_shape = x.shape[:-1]

        # Compute encoder features and pass them trough density mlp.
        x_flat = x.view(-1, 3)
        h = self.pos_encoder(x_flat)
        d = self.density_mlp(h)

        return d.view(batch_shape + (16,))

    def decode_density(self, f: torch.Tensor) -> torch.Tensor:
        """Return density estimates from encoded features.

        Params:
            f: (N,...,16) feature values for each sample point

        Returns:
            d: (N,...,1) color values in ranges [0..1]
        """

        # We consider the first feature dimension to be the
        # log-density estimate
        # print(f[..., 0].mean())
        return f[..., 0 : self.n_density_dims].exp()

    def decode_color(
        self, f: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return color estimates from encoded features

        Params:
            f: (N,..., 16) feature values for each sample point
            cond: (N,...,n_color_cond_dims) conditioning values for each
                sample point.

        Returns:
            colors: (N,...,n_colors) color values in range [0,+inf] (when hdr)
                [0,1] otherwise.

        """
        batch_shape = f.shape[:-1]

        # Concat with condition
        if cond is not None:
            f = torch.cat((f, cond), -1)

        # Raw color prediction
        c = self.color_mlp(f).view(batch_shape + (self.n_color_dims,))

        # Reparametrize
        color = c.exp() if self.is_hdr else torch.sigmoid(c)

        return color

    def forward(self, xyz_ndc, color_cond: Optional[torch.Tensor] = None):
        """Predict density and color values for sample positions.

        Params:
            xyz_ndc: (N,...,3) sampling positions in normalized device coords
                [-1,1]
            color_cond: (N,...,n_color_cond_dims) optional color
                condititioning values

        Returns:
            color: (N,...,n_colors) predicted color values [0,+inf] (when hdr)
                [0,1] otherwise
            sigma: (N,...,1) prediced density values [0,+inf]
        """
        f = self.encode(xyz_ndc)
        sigma = self.decode_density(f)
        color = self.decode_color(f, cond=color_cond)
        return color, sigma


NeRFConf = config.build_conf(NeRF)

__all__ = [
    "NeRF",
    "NeRFConf",
]
