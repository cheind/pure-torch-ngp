from typing import Optional
import torch


from .. import config
from .. import functional
from . import protocols

from .ray_bundle import RayBundle
from .volume import Volume


class StratifiedRayStepSampler(torch.nn.Module, protocols.RayStepSampler):
    def __init__(self, n_samples: int = 256, noise_scale: float = 1.0) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.noise_scale = noise_scale

    def __call__(self, rays: RayBundle, vol: Optional[Volume]) -> torch.Tensor:
        del vol
        return functional.sample_ray_step_stratified(
            rays.tnear, rays.tfar, self.n_samples, noise_scale=self.noise_scale
        )


StratifiedRayStepSamplerConf = config.build_conf(StratifiedRayStepSampler)


class InformedRayStepSampler(torch.nn.Module, protocols.RayStepSampler):
    def __init__(self, n_samples: int = 256, n_coarse_samples: int = 32) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_coarse_samples = n_coarse_samples

    @torch.no_grad()
    def __call__(self, rays: RayBundle, vol: Optional[Volume]) -> torch.Tensor:
        ts_coarse = functional.batch_linspace(
            rays.tnear, rays.tfar, self.n_coarse_samples
        )  # (M,N,...,1)
        density, _ = vol.sample(rays(ts_coarse), ynm=None, return_color=False)
        coarse_weights = functional.integrate_timesteps(density, ts_coarse, rays.dnorm)
        ts_informed = functional.sample_ray_step_informed(
            ts=ts_coarse,
            tnear=rays.tnear,
            tfar=rays.tfar,
            weights=coarse_weights,
            n_samples=self.n_samples,
        )
        return ts_informed


InformedRayStepSamplerConf = config.build_conf(InformedRayStepSampler)

__all__ = [
    "StratifiedRayStepSampler",
    "StratifiedRayStepSamplerConf",
    "InformedRayStepSampler",
    "InformedRayStepSamplerConf",
]
