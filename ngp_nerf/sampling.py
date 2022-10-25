from lib2to3.pytree import Base
from numpy import isscalar
import torch
import torch.nn
import torch.nn.functional as F
import dataclasses
from typing import Optional, Iterator, Iterable, Union


class BaseCamera(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_length: torch.Tensor
        self.principal_point: torch.Tensor
        self.R: torch.Tensor
        self.T: torch.Tensor
        self.size: torch.Tensor
        self.tnear: torch.Tensor
        self.tfar: torch.Tensor

    def uv_grid(self):
        N = self.focal_length.shape[0]
        dev = self.focal_length.device
        dtype = self.focal_length.dtype
        uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(self.size[0, 0], dtype=dtype, device=dev),
                    torch.arange(self.size[0, 1], dtype=dtype, device=dev),
                    indexing="xy",
                ),
                -1,
            )
            .unsqueeze(0)
            .expand(N, -1, -1, -1)
        )
        return uv

    def unproject_uv(
        self, uv: torch.Tensor, depth: Union[float, torch.Tensor] = 1.0
    ) -> torch.Tensor:
        # uv is (N,...,2)
        N = self.focal_length.shape[0]
        mbatch = uv.shape[1:-1]
        mbatch_ones = (1,) * len(mbatch)

        if not torch.is_tensor(depth):
            depth = uv.new_tensor(depth)

        depth = depth.expand((N,) + mbatch + (1,))
        pp = self.principal_point.view((N,) + mbatch_ones + (2,))
        fl = self.focal_length.view((N,) + mbatch_ones + (2,))

        xy = (uv - pp) / fl
        xyz = torch.cat((xy, depth), -1)
        return xyz

    def world_rays(
        self, uv: torch.Tensor = None, normalize_dirs: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # uv is (N,...,2), result is (N,...,3)
        if uv is None:
            uv = self.uv_grid()
        N = self.focal_length.shape[0]
        mbatch = uv.shape[1:-1]
        mbatch_ones = (1,) * len(mbatch)

        rot = self.R.view((N,) + mbatch_ones + (3, 3))
        trans = self.T.view((N,) + mbatch_ones + (3,))
        xyz = self.unproject_uv(uv, depth=1.0)
        ray_dir = (rot @ xyz.unsqueeze(-1)).squeeze(-1) + trans

        if normalize_dirs:
            F.normalize(ray_dir, p=2, dim=-1)

        ray_origin = trans.expand_as(ray_dir)

        return ray_origin, ray_dir


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
        self.register_buffer("focal_length", torch.tensor([[fx, fy]]).float())
        self.register_buffer("principal_point", torch.tensor([[cx, cy]]).float())
        self.register_buffer("size", torch.tensor([[width, height]]).int())
        self.register_buffer("R", R.unsqueeze(0).float())
        self.register_buffer("T", T.unsqueeze(0).float())


class CameraBatch(BaseCamera):
    def __init__(self, cams: list[Camera]) -> None:
        super().__init__()
        self.register_buffer(
            "focal_length", torch.cat([c.focal_length for c in cams], 0)
        )
        self.register_buffer(
            "principal_point", torch.cat([c.principal_point for c in cams], 0)
        )
        self.register_buffer("size", torch.cat([c.size for c in cams], 0))
        self.register_buffer("R", torch.cat([c.R for c in cams]))
        self.register_buffer("T", torch.cat([c.T for c in cams]))


@dataclasses.dataclass
class RayBatch:
    pixels_uv: torch.Tensor
    ray_origins: torch.Tensor
    ray_dirs: torch.Tensor
    ray_tnear: torch.Tensor
    ray_tfar: torch.Tensor
    features: Optional[torch.Tensor]


def generate_random_uv_samples(
    camera: BaseCamera,
    image: torch.Tensor = None,
    n_samples_per_cam: int = None,
    subpixel: bool = True,
) -> Iterator[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    N = camera.focal_length.shape[0]
    M = n_samples_per_cam or camera.size[0, 0].item()
    dev = camera.focal_length.device
    dtype = camera.focal_length.dtype

    rand_01 = torch.empty((N, M, 2), dtype=dtype, device=dev)

    while True:
        # The following code samples within the valid image area. We treat
        # pixels as squares and integer pixel coords as centers of pixels.
        rand_01.uniform_()
        uv = rand_01 * (camera.size[:, None, :] + 1) - 0.5

        if not subpixel:
            # The folloing may create duplicate pixel coords
            uv = torch.round(uv)

        # TODO: if we want to support radial distortions, we need
        # to forward distort uvs here.

        feature_uv = None
        if image is not None:
            feature_uv = _sample_features_uv(
                camera_images=image, camera_uvs=uv, subpixel=subpixel
            )
        yield uv, feature_uv


def generate_sequential_uv_samples(
    camera: BaseCamera,
    image: torch.Tensor = None,
    n_samples_per_cam: int = None,
    n_passes: int = 1,
) -> Iterator[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    # TODO: assumes same image size currently
    N = camera.focal_length.shape[0]
    M = n_samples_per_cam or camera.size[0, 0].item()

    uv_grid = camera.uv_grid().view(N, -1, 2)
    for _ in range(n_passes):
        for uv in uv_grid.split(M, 1):
            feature_uv = None
            if image is not None:
                feature_uv = _sample_features_uv(
                    camera_images=image, camera_uvs=uv, subpixel=False
                )
            yield uv, feature_uv


def _sample_features_uv(
    camera_images: torch.Tensor, camera_uvs: torch.Tensor, subpixel: bool
) -> torch.Tensor:
    N, M = camera_uvs.shape[:2]
    C = camera_images.shape[1]
    mode = "bilinear" if subpixel else "nearest"
    features_uv = (
        F.grid_sample(
            camera_images,
            camera_uvs.view(N, M, 1, 2),
            mode=mode,
            padding_mode="border",
            align_corners=False,
        )
        .view(N, C, M)
        .permute(0, 2, 1)
    )
    return features_uv


def sample_random_rays(
    cams: CameraBatch,
    features: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
    n_rays_per_cam: int = None,
    subpixel: bool = True,
    normalize_dir: bool = False,
):
    N = cams.focal_length.shape[0]
    n_rays_per_cam = n_rays_per_cam or cams.size[0, 0].item()
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

    # sample_random_rays(cb, 20, subpixel=False)
    features = torch.rand(2, 6, 240, 320)
    uv, uv_features = next(iter(generate_random_uv_samples(cb, features)))
    print(uv.shape, uv_features.shape)

    uv, uv_features = next(iter(generate_sequential_uv_samples(cb, features)))
    print(uv.shape, uv_features.shape)

    xyz = cb.unproject_uv(cb.uv_grid(), depth=1.0)

    cb.world_rays(uv=cb.uv_grid())


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
