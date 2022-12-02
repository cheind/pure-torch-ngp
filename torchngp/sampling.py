from typing import Iterator, Optional, Protocol
import torch
import torch.nn.functional as F


from . import geometric
from . import functional
from . import config


def generate_random_uv_samples(
    camera: geometric.MultiViewCamera,
    image: Optional[torch.Tensor] = None,
    n_samples_per_cam: Optional[int] = None,
    subpixel: bool = True,
) -> Iterator[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Generate random pixel samples.
    This methods acts as an infinite generator that samples
    `n_samples_per_cam` pixel coordinates per camera randomly
    from the image plane.
    Params:
        camera: a batch of N perspective cameras
        image: (N,C,H,W) image tensor with C feature channels
        n_samples_per_cam: number of samples from each camera to
            draw. If not specified, defaults to W.
        subpixel: whether subpixel coordinates are allowed.
    Generates:
        uv: (N, n_samples_per_cam, 2) sampled coordinates
        uv_feature (N, n_samples_per_cam, C) samples features
    """
    N = camera.n_views
    M = n_samples_per_cam or camera.size[0].item()

    uvn = camera.R.new_empty((N, M, 2))
    # we actually sample from (-1, +1) to avoid issues
    # when rounding to subpixel coords. -0.5 would get mapped to -1.
    bounds = 1.0 - 1e-7

    while True:
        # The following code samples within the valid image area. We treat
        # pixels as squares and integer pixel coords as centers of pixels.
        uvn.uniform_(-bounds, bounds)
        uv = (uvn + 1.0) * camera.size[None, None, :] * 0.5 - 0.5

        if not subpixel:
            # The folloing may create duplicate pixel coords
            uv = uv.round()

        # TODO: if we want to support radial distortions, we need
        # to forward distort uvs here.

        feature_uv = None
        if image is not None:
            feature_uv = _sample_features_uv(
                camera_images=image, camera_uvs=uvn, subpixel=subpixel
            )
        yield uv, feature_uv


def generate_sequential_uv_samples(
    camera: geometric.MultiViewCamera,
    image: Optional[torch.Tensor] = None,
    n_samples_per_cam: Optional[int] = None,
    n_passes: int = 1,
) -> Iterator[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Generate sequential pixel samples.
    This methods acts as a generator, generating `n_samples_per_cam`
    pixel grid samples per step.
    Params:
        camera: a batch of N perspective cameras
        image: (N,C,H,W) image tensor with C feature channels.
        n_samples_per_cam: number of samples from each camera to
            draw. If not specified, defaults to W.
        n_passes: number of total passes over all pixels
    Generates:
        uv: (N, n_samples_per_cam, 2) sampled coordinates (row-major)
        uv_feature (N, n_samples_per_cam, C) samples features (row-major)
    """
    # TODO: assumes same image size currently
    N = camera.n_views
    M = n_samples_per_cam or camera.size[0].item()

    uv_grid = camera.make_uv_grid()
    uvn_grid = (uv_grid + 0.5) * 2 / camera.size.view(1, 1, 1, 2) - 1.0
    for _ in range(n_passes):
        for uv, uvn in zip(
            uv_grid.view(N, -1, 2).split(M, 1),
            uvn_grid.view(N, -1, 2).split(M, 1),
        ):
            feature_uv = None
            if image is not None:
                feature_uv = _sample_features_uv(
                    camera_images=image, camera_uvs=uvn, subpixel=False
                )
            yield uv, feature_uv


class RayStepSampler(Protocol):
    """Protocol for sampling timestep values along rays.

    Note that ray directions are not normalized.

    """

    def __call__(self, rays: geometric.RayBundle) -> torch.Tensor:
        """Sample timestep values

        Params:
            rays: (N,...) bundle of rays

        Returns:
            ts: (T,N,...,1) timestep samples for each ray.
        """
        ...


class StratifiedRayStepSampler(torch.nn.Module, RayStepSampler):
    def __init__(self, n_samples: int = 128) -> None:
        super().__init__()
        self.n_samples = n_samples

    def __call__(self, rays: geometric.RayBundle) -> torch.Tensor:
        return functional.sample_ray_step_stratified(
            rays.tnear, rays.tfar, self.n_samples
        )


StratifiedRayStepSamplerConf = config.build_conf(StratifiedRayStepSampler)


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
