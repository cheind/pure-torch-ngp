import dataclasses

import torch

from .cameras import MultiViewCamera
from . import functional


@dataclasses.dataclass
class RayBundle:
    """A collection of rays."""

    o: torch.Tensor  # (N,...,3)
    d: torch.Tensor  # (N,...,3)
    tnear: torch.Tensor  # (N,...,1)
    tfar: torch.Tensor  # (N,...,1)
    dnorm: torch.Tensor  # (N,...,1)

    @staticmethod
    def make_world_rays(cam: MultiViewCamera, uv: torch.Tensor) -> "RayBundle":
        """Returns a new RayBundle from uv coordinates."""
        o, d, tnear, tfar = functional.make_world_rays(
            uv,
            cam.focal_length,
            cam.principal_point,
            cam.R,
            cam.T,
            tnear=cam.tnear,
            tfar=cam.tfar,
        )
        dnorm = torch.norm(d, p=2, dim=-1, keepdim=True)

        return RayBundle(o, d, tnear, tfar, dnorm)

    def __call__(self, ts: torch.Tensor) -> torch.Tensor:
        """Evaluate rays at given time steps.

        Params:
            ts: (N,...,1) or more general (T,...,N,...,1) time steps

        Returns:
            xyz: (N,...,3) / (T,...,N,...,3) locations
        """
        return functional.evaluate_ray(self.o, self.d, ts)

    def filter_by_mask(self, mask: torch.BoolTensor) -> "RayBundle":
        """Filter rays by boolean mask.

        Params:
            mask: (N,...) tensor

        Returns
            rays: filtered ray bundle with flattened dimensions
        """
        return RayBundle(
            o=self.o[mask],
            d=self.d[mask],
            tnear=self.tnear[mask],
            tfar=self.tfar[mask],
            dnorm=self.dnorm[mask],
        )

    def update_bounds(self, tnear: torch.Tensor, tfar: torch.Tensor) -> "RayBundle":
        """Updates the bounds of this ray bundle.

        Params:
            tnear: (N,...,1) near time step values
            tfar: (N,...,1) far time step values

        Returns:
            rays: updated ray bundle sharing tensors
        """
        tnear = torch.max(tnear, self.tnear)
        tfar = torch.min(tfar, self.tfar)
        return RayBundle(o=self.o, d=self.d, tnear=tnear, tfar=tfar, dnorm=self.dnorm)

    def intersect_aabb(self, box: torch.Tensor) -> "RayBundle":
        """Ray/box intersection.

        Params:
            box: (2,3) min/max corner of aabb

        Returns:
            rays: ray bundle with updated bounds

        Adapted from
        https://github.com/evanw/webgl-path-tracing/blob/master/webgl-path-tracing.js
        """

        tnear, tfar = functional.intersect_ray_aabb(
            self.o, self.d, self.tnear, self.tfar, box
        )
        return self.update_bounds(tnear, tfar)

    def active_mask(self) -> torch.BoolTensor:
        """Returns a mask of active rays.

        Active rays have a positive time step range.

        Returns:
            mask: (N,...) tensor of active rays
        """
        return (self.tnear < self.tfar).squeeze(-1)
