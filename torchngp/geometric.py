import dataclasses
from typing import Union, Optional
from pathlib import Path
from PIL import Image


import torch
import torch.nn
import numpy as np


from . import functional
from . import config


class MultiViewCamera(torch.nn.Module):
    """Perspective camera with multiple poses.

    All camera models treat pixels as squares with pixel centers
    corresponding to integer pixel coordinates. That is, a pixel
    (u,v) extends from (u+0.5,v+0.5) to `(u+0.5,v+0.5)`.

    In addition we generalize to `N` poses by batching along
    the first dimension for attributes `R` and `T`.
    """

    def __init__(
        self,
        focal_length: tuple[float, float],
        principal_point: tuple[float, float],
        size: tuple[int, int],
        rvec: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        tvec: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        poses: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        image_paths: Optional[list[str]] = None,
        tnear: float = 0.0,
        tfar: float = 10.0,
    ) -> None:
        super().__init__()
        focal_length = torch.as_tensor(focal_length).view(2).float()
        principal_point = torch.as_tensor(principal_point).view(2).float()
        size = torch.as_tensor(size).view(2).int()
        tnear = torch.as_tensor(tnear).view(1).float()
        tfar = torch.as_tensor(tfar).view(1).float()

        rvec_given = rvec is not None
        tvec_given = tvec is not None
        pos_given = poses is not None

        if not (
            (not rvec_given and not tvec_given and pos_given)
            or (rvec_given and tvec_given and not pos_given)
        ):
            raise ValueError(
                "Either specify (rvec,tvec) or poses but not both or neither"
            )
        if rvec_given:
            # rvec, tvec specified
            if isinstance(rvec, list):
                rvec = torch.stack(rvec, 0)
            if isinstance(tvec, list):
                tvec = torch.stack(tvec, 0)
        else:
            # poses specified
            if isinstance(poses, list):
                poses = torch.stack(poses, 0)
            rvec = poses[:, :3, :3]
            rvec = functional.so3_log(rvec)
            tvec = poses[:, :3, 3]
        rvec = rvec.view(-1, 3).float()
        tvec = tvec.view(-1, 3, 1).float()

        self.register_buffer("focal_length", focal_length)
        self.register_buffer("principal_point", principal_point)
        self.register_buffer("size", size)
        self.register_buffer("tnear", tnear)
        self.register_buffer("tfar", tfar)
        self.register_buffer("rvec", rvec)
        self.register_buffer("tvec", tvec)
        if image_paths is not None:
            image_paths = np.array(
                image_paths, dtype=object
            )  # to support advanced indexing
        self.image_paths = image_paths

        self.focal_length: torch.Tensor
        self.principal_point: torch.Tensor
        self.size: torch.IntTensor
        self.tnear: torch.Tensor
        self.tfar: torch.Tensor
        self.rvec: torch.Tensor
        self.tvec: torch.Tensor

    def __getitem__(self, index) -> "MultiViewCamera":
        """Slice a subset of camera poses."""
        return MultiViewCamera(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            size=self.size,
            rvec=self.rvec[index],
            tvec=self.tvec[index],
            image_paths=(
                self.image_paths[index] if self.image_paths is not None else None
            ),
            tnear=self.tnear,
            tfar=self.tfar,
        )

    @property
    def K(self):
        """Return the 3x3 camera intrinsic matrix"""
        K = self.rvec.new_zeros((3, 3))
        K[0, 0] = self.focal_length[0]
        K[1, 1] = self.focal_length[1]
        K[0, 2] = self.principal_point[0]
        K[1, 2] = self.principal_point[1]
        K[2, 2] = 1
        return K

    @property
    def E(self) -> torch.Tensor:
        """Return the (N,4,4) extrinsic pose matrices."""
        N = self.n_views
        t = self.rvec.new_zeros((N, 4, 4))
        t[:, :3, :3] = functional.so3_exp(self.rvec)
        t[:, :3, 3:4] = self.tvec
        t[:, -1, -1] = 1
        return t

    @property
    def R(self) -> torch.Tensor:
        """Return the (N,3,3) rotation matrices."""
        return functional.so3_exp(self.rvec)

    @property
    def n_views(self):
        """Return the number of views."""
        return self.R.shape[0]

    def make_uv_grid(self):
        """Generates uv-pixel grid coordinates.

        Returns:
            uv: (N,H,W,2) tensor of grid coordinates using
                'xy' indexing.
        """
        N = self.n_views
        dev = self.focal_length.device
        dtype = self.focal_length.dtype
        uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(self.size[0], dtype=dtype, device=dev),
                    torch.arange(self.size[1], dtype=dtype, device=dev),
                    indexing="xy",
                ),
                -1,
            )
            .unsqueeze(0)
            .expand(N, -1, -1, -1)
        )
        return uv

    def load_images(self, base_path: Path = None) -> torch.Tensor:
        """Load images associated with this camera."""
        if base_path is None:
            base_path = Path.cwd()
        if self.image_paths is None or len(self.image_paths) == 0:
            return self.rvec.new_empty((0, self.size[1], self.size[2], 4))
        loaded = []
        for path in self.image_paths:
            img = Image.open(path).convert("RGBA")
            img = self.rvec.new_tensor(np.asarray(img)).float().permute(2, 0, 1) / 255.0
            loaded.append(img)
        return torch.stack(loaded, 0)

    def extra_repr(self):
        out = ""
        out += f"focal_length={self.focal_length}, "
        out += f"size={self.size}, "
        out += f"n_poses={self.n_views}"
        return out


MultiViewCameraConf = config.build_conf(MultiViewCamera)


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
            cam.tvec,
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
