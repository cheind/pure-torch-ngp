from lib2to3.pytree import Base
import torch
import torch.nn
import torch.nn.functional as F
import dataclasses
from typing import Optional, Iterator, Iterable


class BaseCamera(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_length: torch.Tensor
        self.principal_point: torch.Tensor
        self.R: torch.Tensor
        self.T: torch.Tensor
        self.size: torch.Tensor


class Camera(BaseCamera):
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        R: torch.tensor = None,
        T: torch.tensor = None,
    ) -> None:
        super().__init__()
        if R is None:
            R = torch.eye(3)
        if T is None:
            T = torch.zeros(3)
        self.register_buffer("focal_length", torch.Tensor([fx, fy]))
        self.register_buffer("principal_point", torch.Tensor([cx, cy]))
        self.register_buffer("size", torch.Tensor([width, height]))
        self.register_buffer("R", R)
        self.register_buffer("T", T)


class CameraBatch(BaseCamera):
    def __init__(self, cams: list[Camera]) -> None:
        super().__init__()
        self.register_buffer(
            "focal_length", torch.stack([c.focal_length for c in cams], 0)
        )
        self.register_buffer(
            "principal_point", torch.stack([c.principal_point for c in cams], 0)
        )
        self.register_buffer("size", torch.stack([c.size for c in cams], 0))
        self.register_buffer("R", torch.stack([c.R for c in cams]))
        self.register_buffer("T", torch.stack([c.T for c in cams]))


@dataclasses.dataclass
class RayBatch:
    pixels_uv: torch.Tensor
    ray_origins: torch.Tensor
    ray_dirs: torch.Tensor
    ray_tnear: torch.Tensor
    ray_tfar: torch.Tensor
    features: Optional[torch.Tensor]


def sample_random_rays(
    cams: CameraBatch,
    features: Optional[torch.Tensor],
    n_rays_per_cam: int = None,
    subpixel: bool = True,
    normalize_dir: bool = False,
):
    N = cams.focal_length.shape[0]
    n_rays_per_cam = n_rays_per_cam or cams.size[0, 0]
    M = n_rays_per_cam
    dev = cams.focal_length.device
    dtype = cams.focal_length.dtype

    uv = torch.rand(size=(N, M, 2), dtype=dtype, device=dev) * cams.size[:, None, :]
    if not subpixel:
        uv = torch.round(uv)  # may create duplicates
    xy = (uv - cams.principal_point[:, None, :]) / cams.focal_length[:, None, :]
    xyz = torch.cat((xy, xy.new_ones(N, M, 1)), -1)
    xyz = (cams.R.unsqueeze(1) @ xyz.unsqueeze(-1)).squeeze(-1) + cams.T[:, None, :]
    if normalize_dir:
        xyz = F.normalize(xyz, p=2, dim=-1)

    # f_uv = None
    # if features is not None:
    #     if not subpixel:
    #         uv_int = uv.long()
    #         # f_uv = features.permute(0,2,3,1)[]

    ray_dirs = xyz  # normalize necessary?
    ray_origins = cams.T[:, None, :].expand_as(xyz)
    print(xyz.shape, ray_origins.shape)


if __name__ == "__main__":
    c = Camera(fx=500, fy=500, cx=160, cy=120, width=320, height=240)
    cb = CameraBatch([c, c])

    sample_random_rays(cb, 20, subpixel=False)


# def generate_random_pixel_samples(
#     cams: list[PerspectiveCamera],
#     features: Optional[list[torch.Tensor]] = None,
#     n_samples: Optional[int] = None,
#     only_pixelcenters: bool = False,
# ) -> Iterator[PixelBatch]:
#     n_cams = len(cams)
#     n_samples = n_samples or cams[0].hw[0] * cams[0].hw[1]
#     dev = cams[0].focal_length.device

#     uv = torch.empty((n_samples, 2), device=dev)
#     while True:
#         cam_idx = torch.rand

#     pass


if __name__ == "__main__":
    pass
