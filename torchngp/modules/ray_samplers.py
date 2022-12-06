from typing import TYPE_CHECKING
import torch


from .. import config
from .. import functional
from . import protocols

if TYPE_CHECKING:
    from .ray_bundle import RayBundle


class StratifiedRayStepSampler(torch.nn.Module, protocols.RayStepSampler):
    def __init__(self, n_samples: int = 256) -> None:
        super().__init__()
        self.n_samples = n_samples

    def __call__(self, rays: "RayBundle") -> torch.Tensor:
        return functional.sample_ray_step_stratified(
            rays.tnear, rays.tfar, self.n_samples
        )


StratifiedRayStepSamplerConf = config.build_conf(StratifiedRayStepSampler)

__all__ = [
    "StratifiedRayStepSampler",
    "StratifiedRayStepSamplerConf",
]
