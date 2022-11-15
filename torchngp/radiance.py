from typing import Optional, Protocol

import torch

from . import functional
from . import encoding


class RadianceField(Protocol):
    """Protocol of a spatial radiance field.

    A spatial radiance field takes normalized locations plus optional
    conditioning values and returns color and densities values for
    each location."""

    n_color_cond_dims: int
    n_color_dims: int
    n_density_dims: int

    def __call__(
        self, xyz: torch.Tensor, color_cond: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        """Return features from positions.

        Params:
            xyz_ndc: (N,...,3) normalized device locations [-1,1]

        Returns:
            f: (N,...,16) feature values for each sample point
        """
        ...

    def decode_density(self, f: torch.Tensor) -> torch.Tensor:
        """Return density estimates from encoded features.

        Params:
            f: (N,...,16) feature values for each sample point

        Returns:
            d: (N,...,1) color values in ranges [0..1]
        """
        ...

    def decode_color(
        self, f: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return color estimates from encoded features

        Params:
            f: (N,..., 16) feature values for each sample point
            cond: (N,...,n_color_cond) conditioning values for each
                sample point.

        Returns:
            colors: (N,...,n_colors) color values in range [0,+inf] (when hdr)
                [0,1] otherwise.
        """
        ...


def rasterize_field(
    rf: RadianceField,
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
    ijk = functional.make_grid(
        resolution,
        indexing="xy",
        device=device,
        dtype=dtype,
    )
    res = ijk.new_tensor(resolution[::-1])
    nxyz = (ijk + 0.5) * 2 / res - 1.0

    color_parts = []
    sigma_parts = []
    for batch in nxyz.split(batch_size, dim=0):
        rgb, d = rf(batch)
        color_parts.append(rgb)
        sigma_parts.append(d)
    C = color_parts[0].shape[-1]
    color = torch.cat(color_parts, 0).view(resolution + (C,))
    sigma = torch.cat(sigma_parts, 0).view(resolution + (1,))
    return color, sigma


class NeRF(torch.nn.Module, RadianceField):
    """Neural radiance field module.

    Currently supports only spatial features and not view dependent ones.
    """

    def __init__(
        self,
        n_colors: int = 3,
        n_hidden: int = 64,
        n_encodings: int = 2**12,
        n_levels: int = 16,
        n_color_cond: int = 0,
        min_res: int = 32,
        max_res: int = 256,
        max_n_dense: int = 256**3,
        is_hdr: bool = False,
    ) -> None:
        super().__init__()
        self.is_hdr = is_hdr
        self.n_color_cond_dims = n_color_cond
        self.n_color_dims = n_colors
        self.n_density_dims = 1
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

    def encode(self, xyz_ndc: torch.Tensor) -> torch.Tensor:
        """Return features from positions.

        Params:
            xyz: (N,...,3) positions in world space

        Returns:
            f: (N,...,16) feature values for each sample point
        """
        batch_shape = xyz_ndc.shape[:-1]

        # Compute encoder features and pass them trough density mlp.
        xn_flat = xyz_ndc.view(-1, 3)
        h = self.pos_encoder(xn_flat)
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
            cond: (N,...,n_color_cond) conditioning values for each
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
            color_cond: (N,...,n_cond_color_dims) optional color
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
