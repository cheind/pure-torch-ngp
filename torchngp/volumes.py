import torch
from typing import Optional
from .radiance import RadianceField
from .filtering import SpatialFilter, BoundsFilter
from . import functional


class Volume(torch.nn.Module):
    def __init__(
        self,
        aabb: torch.Tensor,
        radiance_field: RadianceField,
        spatial_filter: Optional[SpatialFilter] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb.float())
        self.aabb: torch.Tensor
        self.radiance_field = radiance_field
        self.spatial_filter = spatial_filter or BoundsFilter()

    def to_ndc(self, xyz: torch.Tensor) -> torch.Tensor:
        return functional.convert_world_to_box_normalized(xyz, self.aabb)
