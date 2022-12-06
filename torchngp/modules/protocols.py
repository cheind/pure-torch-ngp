from typing import Protocol, Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .ray_bundle import RayBundle
    from .volume import Volume


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


class RayStepSampler(Protocol):
    """Protocol for sampling z/timestep values along rays.

    Note that ray directions are not normalized.
    """

    def __call__(self, rays: "RayBundle", vol: Optional["Volume"]) -> torch.Tensor:
        """Sample timestep/z values along rays.

        Params:
            rays: (N,...) bundle of rays
            vol: radiance volume

        Returns:
            ts: (T,N,...,1) timestep samples for each ray.
        """
        ...


class SpatialFilter(Protocol):
    """Protocol for a spatial rendering filter.

    A spatial rendering accelerator takes spatial positions in NDC format
    and returns a mask of samples worthwile considering.
    """

    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        """Test given NDC locations.

        Params:
            xyz_ndc: (N,...,3) tensor of normalized [-1,1] coordinates

        Returns:
            mask: (N,...) boolean mask of the samples to be processed further
        """
        ...

    def update(self, rf: RadianceField):
        """Update this accelerator."""
        ...


__all__ = [
    "RadianceField",
    "RayStepSampler",
    "SpatialFilter",
]
