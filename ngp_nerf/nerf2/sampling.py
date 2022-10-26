from typing import Iterator, Optional
import torch
import torch.nn.functional as F

from .cameras import BaseCamera


def generate_random_uv_samples(
    camera: BaseCamera,
    image: torch.Tensor = None,
    n_samples_per_cam: int = None,
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
        uv: (N, n_samples_per_cam, 2) sampled coordinates
        uv_feature (N, n_samples_per_cam, C) samples features
    """
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
