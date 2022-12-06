from typing import Iterator, Optional, Union
import torch
import torch.nn.functional as F

from .geometric import make_multiview_grid


def generate_random_uv_samples(
    uv_size: Union[tuple[int, int], torch.IntTensor],
    n_views: int,
    image: Optional[torch.Tensor] = None,
    n_samples_per_view: Optional[int] = None,
    subpixel: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Iterator[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Generate random pixel samples.
    This methods acts as an infinite generator that samples
    `n_samples_per_cam` pixel coordinates per camera randomly
    from the image plane.
    Params:
        shape: a batch of N perspective cameras
        image: (N,C,H,W) image tensor with C feature channels
        n_samples_per_cam: number of samples from each camera to
            draw. If not specified, defaults to W.
        subpixel: whether subpixel coordinates are allowed.
    Generates:
        uv: (N, n_samples_per_cam, 2) sampled coordinates
        uv_feature (N, n_samples_per_cam, C) samples features
    """
    N = n_views
    M = n_samples_per_view or uv_size[0]
    dtype = dtype or torch.float32
    if torch.is_tensor(uv_size):
        uv_size = tuple(uv_size.tolist())

    uvn = torch.empty((N, M, 2), dtype=dtype, device=device)
    uv_sizef = uvn.new_tensor(uv_size)
    # we actually sample from (-1, +1) to avoid issues
    # when rounding to subpixel coords. -0.5 would get mapped to -1.
    bounds = 1.0 - 1e-7

    while True:
        # The following code samples within the valid image area. We treat
        # pixels as squares and integer pixel coords as centers of pixels.
        uvn.uniform_(-bounds, bounds)
        uv = (uvn + 1.0) * uv_sizef[None, None, :] * 0.5 - 0.5

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


def generate_randperm_uv_samples(
    uv_size: Union[tuple[int, int], torch.IntTensor],
    n_views: int,
    image: Optional[torch.Tensor] = None,
    n_samples_per_view: Optional[int] = None,
    subpixel: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Iterator[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Generate random pixel samples.

    This method is similar to `generate_random_uv_samples`, but
    ensures that every pixel is visited once before randomizing
    the access again. The method `generate_random_uv_samples` is
    more of Monte Carlo style in that it might generate quite a
    few duplicates.

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
    N = n_views
    M = n_samples_per_view or uv_size[0]
    if torch.is_tensor(uv_size):
        uv_size = tuple(uv_size.tolist())

    dtype = dtype or torch.float32

    uvgrid = make_multiview_grid(
        n_views, uv_size, device=device, dtype=dtype
    )  # (N,H,W,2)
    uvgrid = uvgrid.view(N, -1, 2)  # (N,L,2)
    uv_sizef = uvgrid.new_tensor(uv_size)
    L = uvgrid.shape[1]

    if subpixel:
        noise = uvgrid.new_empty((N, M, 2), dtype=dtype)
        bounds = (1 - 1e-7) * 0.5

    pos = 0
    # batched version of randperm (N,L)
    ids = torch.argsort(torch.rand(N, L, device=uvgrid.device), dim=-1)
    while True:
        # Reshuffle if necessary
        if (pos + M) > L:
            ids = torch.argsort(torch.rand(N, L, device=uvgrid.device), dim=-1)
            pos = 0

        # Get next M random ids
        rand_ids = ids[:, pos : pos + M]  # (N,M)
        # Map to uv coords
        uv = []
        for i in range(N):
            uv.append(uvgrid[i, rand_ids[i, :]])
        uv = torch.stack(uv, 0)

        # Add noise (+/- 0.5)
        if subpixel:
            noise.uniform_(-bounds, bounds)
            uv = uv + noise

        # TODO: if we want to support radial distortions, we need
        # to forward distort uvs here.

        # Compute color features
        feature_uv = None
        if image is not None:
            uvn = (uv + 0.5) * 2 / uv_sizef[None, None, :] - 1.0
            feature_uv = _sample_features_uv(
                camera_images=image, camera_uvs=uvn, subpixel=subpixel
            )

        yield uv, feature_uv
        pos += M


def generate_sequential_uv_samples(
    uv_size: Union[tuple[int, int], torch.Tensor],
    n_views: int,
    image: Optional[torch.Tensor] = None,
    n_samples_per_view: Optional[int] = None,
    n_passes: int = 1,
    device: torch.device = None,
    dtype: torch.dtype = None,
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
    N = n_views
    M = n_samples_per_view or uv_size[0]
    dtype = dtype or torch.float32
    if torch.is_tensor(uv_size):
        uv_size = tuple(uv_size.tolist())

    uv_grid = make_multiview_grid(
        n_views, uv_size, device=device, dtype=dtype
    )  # (N,H,W,2)
    uv_sizef = uv_grid.new_tensor(uv_size)

    uvn_grid = (uv_grid + 0.5) * 2 / uv_sizef.view(1, 1, 1, 2) - 1.0
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


__all__ = [
    "generate_random_uv_samples",
    "generate_randperm_uv_samples",
    "generate_sequential_uv_samples",
]
