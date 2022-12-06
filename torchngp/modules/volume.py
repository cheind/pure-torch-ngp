from typing import Optional
import torch

from .. import functional
from .. import config
from . import protocols

from .nerf import NeRFConf
from .spatial_filters import BoundsFilter
from .spatial_filters import OccupancyGridFilterConf


class Volume(torch.nn.Module):
    """Represents a physical volume space."""

    def __init__(
        self,
        aabb: torch.Tensor,
        radiance_field: protocols.RadianceField,
        spatial_filter: Optional[protocols.SpatialFilter] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb.float())
        self.aabb: torch.Tensor
        self.radiance_field = radiance_field
        self.spatial_filter = spatial_filter or BoundsFilter()

    def sample(
        self,
        xyz: torch.Tensor,
        ynm: Optional[torch.Tensor] = None,
        return_color: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample volume at ray locations

        This is a convenience method that takes world rays and samples
        the underlying radiance field taking spatial acceleration structures
        into account.

        Params:
            xyz: (N,...,3) world points
            ynm: (N,...,16) view direction encodings. Required only when
                return_color is True.
            return_color: When set, returns color samples in addition
                to density values.

        Returns:
            density: (N,...,1) density samples
            color: (N,...,C) color samples when ynm is specified and
                return_color is set. None otherwise.
        """
        assert (
            not return_color or ynm is not None
        ), "Color sampling requires viewdir encoding."

        batch_shape = xyz.shape[:-1]
        out_density = xyz.new_zeros(batch_shape + (self.radiance_field.n_density_dims,))
        if return_color:
            out_color = xyz.new_zeros(batch_shape + (self.radiance_field.n_color_dims,))

        # Convert to NDC (T,N,...,3)
        xyz_ndc = functional.convert_world_to_box_normalized(xyz, self.aabb)

        # Invoke spatial filter (T,N,...)
        mask = self.spatial_filter.test(xyz_ndc)

        # Compute features for active elements
        f = self.radiance_field.encode(xyz_ndc[mask])

        # Compute densities
        density = self.radiance_field.decode_density(f)
        out_density[mask] = density.to(density.dtype)

        # Compute optional colors
        if return_color:
            color = self.radiance_field.decode_color(f, cond=ynm[mask])
            out_color[mask] = color.to(out_color.dtype)
            return out_density, out_color
        else:
            return out_density, None


VolumeConf = config.build_conf(
    Volume,
    aabb=config.Vecs3Conf([(-1.0,) * 3, (1.0,) * 3]),
    radiance_field=NeRFConf(),
    spatial_filter=OccupancyGridFilterConf(),
)

__all__ = [
    "Volume",
    "VolumeConf",
]
