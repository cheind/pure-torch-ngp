import torch
from typing import Optional

from . import filtering
from . import functional
from . import config
from . import radiance


class Volume(torch.nn.Module):
    """Represents a physical volume space."""

    def __init__(
        self,
        aabb: torch.Tensor,
        radiance_field: radiance.RadianceField,
        spatial_filter: Optional[filtering.SpatialFilter] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb.float())
        self.aabb: torch.Tensor
        self.radiance_field = radiance_field
        self.spatial_filter = spatial_filter or filtering.BoundsFilter()

    def to_ndc(self, xyz: torch.Tensor) -> torch.Tensor:
        return functional.convert_world_to_box_normalized(xyz, self.aabb)


VolumeConf = config.build_conf(
    Volume,
    aabb=config.Vecs3Conf([(-1.0,) * 3, (1.0,) * 3]),
    radiance_field=radiance.NeRFConf(),
    spatial_filter=filtering.OccupancyGridFilterConf(),
)
