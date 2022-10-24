import torch
import torch.nn
import dataclasses
from typing import Optional, Iterator


@dataclasses.dataclass
class PixelSample:
    uv: torch.Tensor
    features: Optional[torch.Tensor]


def generate_random_pixel_samples(
    shape: tuple[int, ...],
    features: Optional[torch.Tensor] = None,
    n_samples: Optional[int] = None,
) -> Iterator[PixelSample]:
    pass
