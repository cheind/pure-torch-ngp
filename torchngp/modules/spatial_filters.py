import torch

from .. import config
from .. import functional
from . import protocols


class BoundsFilter(torch.nn.Module, protocols.SpatialFilter):
    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        mask = ((xyz_ndc >= -1.0) & (xyz_ndc <= 1.0)).all(-1)
        return mask

    def update(self, rf: protocols.RadianceField):
        del rf
        pass


BoundsFilterConf = config.build_conf(BoundsFilter)


class OccupancyGridFilter(BoundsFilter, torch.nn.Module):
    def __init__(
        self,
        res: int = 64,
        density_initial=0.02,
        density_threshold=0.01,
        stochastic_test: bool = True,
        update_decay: float = 0.7,
        update_noise_scale: float = None,
        update_selection_rate=0.25,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.res = res
        self.update_decay = update_decay
        self.density_initial = density_initial
        self.density_threshold = density_threshold
        self.update_selection_rate = update_selection_rate
        self.stochastic_test = stochastic_test
        if update_noise_scale is None:
            update_noise_scale = 0.9
        self.update_noise_scale = update_noise_scale
        self.register_buffer("grid", torch.full((res, res, res), density_initial))
        self.grid: torch.Tensor

    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        mask = super().test(xyz_ndc)

        ijk = (xyz_ndc + 1) * self.res * 0.5 - 0.5
        ijk = torch.round(ijk).clamp(0, self.res - 1).long()

        d = self.grid[ijk[..., 2], ijk[..., 1], ijk[..., 0]]
        d_mask = d > self.density_threshold
        if self.stochastic_test:
            d_stoch_mask = torch.bernoulli(1 - (-(d + 1e-4)).exp()).bool()
            d_mask |= d_stoch_mask

        return mask & d_mask

    @torch.no_grad()
    def update(self, rf: protocols.RadianceField):
        self.grid *= self.update_decay

        if self.update_selection_rate < 1.0:
            M = int(self.update_selection_rate * self.res**3)
            ijk = torch.randint(0, self.res, size=(M, 3), device=self.grid.device)
        else:
            ijk = functional.make_grid(
                (self.res, self.res, self.res),
                indexing="xy",
                device=self.grid.device,
                dtype=torch.long,
            ).view(-1, 3)

        noise = torch.rand_like(ijk, dtype=torch.float) - 0.5
        noise *= self.update_noise_scale
        xyz = ijk + noise
        xyz_ndc = (xyz + 0.5) * 2 / self.res - 1.0

        f = rf.encode(xyz_ndc)
        d = rf.decode_density(f).squeeze(-1)
        cur = self.grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
        new = torch.maximum(d, cur)
        self.grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = new


OccupancyGridFilterConf = config.build_conf(OccupancyGridFilter)

__all__ = [
    "BoundsFilter",
    "BoundsFilterConf",
    "OccupancyGridFilter",
    "OccupancyGridFilterConf",
]
