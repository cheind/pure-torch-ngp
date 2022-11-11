from typing import Protocol
import torch

from .radiance import RadianceField
from . import geometric


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

    def update(self, global_step: int):
        """Update this accelerator."""
        ...


class BoundsFilter(SpatialFilter):
    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        mask = ((xyz_ndc >= -1.0) & (xyz_ndc <= 1.0)).all(-1)
        return mask

    def update(self, global_step: int):
        pass


class OccupancyGridFilter(BoundsFilter):
    def __init__(
        self,
        rf: RadianceField,
        res: int = 64,
        update_interval: int = 16,
        decay: float = 0.7,
        density_initial=0.02,
        density_threshold=0.01,
        noise_scale: float = None,
        selection_rate=0.25,
        dev: torch.device = None,
    ) -> None:
        super().__init__()
        self.rf = rf
        self.res = res
        self.update_interval = update_interval
        self.decay = decay
        self.density_initial = density_initial
        self.density_threshold = density_threshold
        self.selection_rate = selection_rate
        if noise_scale is None:
            noise_scale = 0.9
        self.noise_scale = noise_scale
        self.dev = dev
        self.grid = torch.full((res, res, res), density_initial, device=dev)
        self.grid_mask = self.grid > density_threshold

    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        mask = super().test(xyz_ndc)

        ijk = (xyz_ndc + 1) * self.res * 0.5 - 0.5
        ijk = torch.round(ijk).clamp(0, self.res - 1).long()

        ijk_valid = ijk[mask]
        ijk_mask = self.grid_mask[ijk_valid[:, 2], ijk_valid[:, 1], ijk_valid[:, 0]]

        out_mask = mask.clone()
        out_mask[mask] &= ijk_mask

        return out_mask

    @torch.no_grad()
    def update(self, global_step: int):
        if (global_step + 1) % self.update_interval > 0:
            return

        self.grid *= self.decay

        if self.selection_rate < 1.0:
            M = int(self.selection_rate * self.res**3)
            ijk = torch.randint(0, self.res, size=(M, 3), device=self.dev)
        else:
            ijk = geometric.make_grid(
                (self.res, self.res, self.res),
                indexing="xy",
                device=self.dev,
                dtype=torch.long,
            ).view(-1, 3)

        noise = torch.rand_like(ijk, dtype=torch.float) - 0.5
        noise *= self.noise_scale
        xyz = ijk + noise
        xyz_ndc = (xyz + 0.5) * 2 / self.res - 1.0

        f = self.rf.encode(xyz_ndc)
        d = self.rf.decode_density(f).squeeze(-1)
        cur = self.grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
        new = torch.maximum(d, cur)
        self.grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = new
        self.grid_mask = self.grid > self.density_threshold
