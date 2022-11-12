from typing import Union, Optional, Protocol

import torch
import torch.nn
import torch.nn.functional as F

from . import geometric
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


def integrate_timesteps(
    sigma: torch.Tensor,
    ts: torch.Tensor,
    dnorm: torch.Tensor,
    tfinal: Union[float, torch.Tensor] = 1e7,
):
    """Computes the timestep weights for multiple rays.

    This implementation is a batched version of equations (1,3) in [1]. In particular
    it computes the weights for timestep `i` as follows
        weights(i)  = T(i)*alpha(i)
                    = exp(-log_T(i-1))*(1-(ts(i+1)-ts(i))*sigma(i))
                    = exp(-log_T(i-1)) - exp(-log_T(i))
    with
        log_T(i) = cumsum([0,(ts(1)-ts(0))*sigma(0),...,(ts(-1)-ts(-2))*sigma(-1))])

    Params:
        sigma: (T,N,...,1) volume density for each ray (N,...) and timestep (T,).
        ts: (T,N,...,1) ray timestep values.
        dnorm: (N,...,1) ray direction norms. Used for conversion of timestep units
            to real world units.
        tfinal: float or (N,...,1) tensor of one beyond last timestep values.
    Returns:
        weights: (T,N,...,1) weights for each timestep.

    References:
        [1] NeRF: Representing Scenes as
        Neural Radiance Fields for View Synthesis
        https://arxiv.org/pdf/2003.08934.pdf
    """

    eps = torch.finfo(ts.dtype).eps
    batch_shape = ts.shape[1:]

    # delta[i] is defined as the segment lengths between ts[i+1] and ts[i]
    # we add a tfinal value far beyond reach which boosts the last element in
    # integration.
    if not torch.is_tensor(tfinal):
        tfinal = ts.new_tensor(tfinal).expand(batch_shape)
    ts = torch.cat((ts, tfinal.unsqueeze(0)), 0)
    delta = ts[1:] - ts[:-1]  # (T+1,N,...,1) -> (T,N,...,1)

    # Account for un-normalized ray direction lengths.
    # TODO: add note about z-stepping and depthmaps
    delta = delta * dnorm.unsqueeze(0)

    # Eps is necessary because in testing sigma is often inf and if delta
    # is zero then 0.0*float('inf')=nan
    sigma_mul_delta = sigma * (delta + eps)  # (T,N,...,1)

    # We construct a full-cumsum which has the following form
    # full_cumsum([1,2,3]) = [0,1,3,6]
    sigma_delta = F.pad(
        sigma_mul_delta,
        (0,) * 2 * (sigma.dim() - 1) + (1, 0),
        mode="constant",
        value=0.0,
    )
    log_transmittance = -sigma_delta.cumsum(0)  # (T+1,N,...,1)
    transmittance = log_transmittance.exp()

    # T(i)*alpha(i) when multiplied equals the following
    weights = transmittance[:-1] - transmittance[1:]  # (T,N,...,1)

    return weights


def color_map(
    ts_color: torch.Tensor, ts_weights: torch.Tensor, per_timestep: bool = False
) -> torch.Tensor:
    """Computes the RGB color map.

    Params:
        ts_color: (T,N,...,C) color samples
        ts_weights: (T,N,...,1) integration weights
        per_timestep: When true, returns the colors
            as they accumulate over time.

    Returns:
        color_map: (N,...,C) final colors or (T,N,...,C) when per_timestep
            is enabled.
    """
    color = ts_color * ts_weights
    return color.cumsum(0) if per_timestep else color.sum(0)


def alpha_map(ts_weights: torch.Tensor, per_timestep: bool = False) -> torch.Tensor:
    """Computes the alpha map.

    Params:
        ts_weights: (T,N,...,1) integration weights
        per_timestep: When true, returns the alpha
            as its accumulate over time.

    Returns:
        alpha_map: (N,...,1) or (T,N,...,1) alpha values in range [0,1].
            High values indicate opaque regions, wheras low values indicate
            regions of large amounts of transmittance.
    """
    ts_acc = ts_weights.cumsum(0) if per_timestep else ts_weights.sum(0)
    return ts_acc


def depth_map(
    ts: torch.Tensor, ts_weights: torch.Tensor, per_timestep: bool = False
) -> torch.Tensor:
    """Estimates a depth map.

    For this method to work correctly, the ray directions are not supposed
    to be normalized. This ensures that ts are actually steps in z-direction
    of the camera for all rays.

    Params:
        ts: (T,N,...,1) ray timestep values
        ts_weights: (T,N,...,1) integration weights
        per_timestep: When true, returns the colors
            as they accumulate over time.

    Returns:
        depth_map: (N,...,1) or (T,N,...,1) depth map
    """
    depth = ts_weights * ts
    return depth.cumsum(0) if per_timestep else depth.sum(0)


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
    ijk = geometric.make_grid(
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
