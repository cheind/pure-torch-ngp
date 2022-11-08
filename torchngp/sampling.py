from typing import Iterator, Optional
from pyparsing import line
import torch
import torch.nn.functional as F
import torch.distributions as D

from .cameras import MultiViewCamera


def generate_random_uv_samples(
    camera: MultiViewCamera,
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
    camera: MultiViewCamera,
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


def sample_ray_step_stratified(
    ray_tnear: torch.Tensor, ray_tfar: torch.Tensor, n_samples: int
) -> torch.Tensor:
    """Creates stratified ray step random samples between tnear/tfar.

    The returned samples per ray are guaranteed to be sorted
    in step ascending order.

    Params:
        ray_tnear: (N,...,1) ray start
        ray_tfar: (N,...,1) ray ends
        n_bins: number of strata

    Returns:
        tsamples: (n_bins,N,...,1)

    Based on:
        NeRF: Representing Scenes as
        Neural Radiance Fields for View Synthesis
        https://arxiv.org/pdf/2003.08934.pdf
        https://en.wikipedia.org/wiki/Stratified_sampling
    """
    dev = ray_tnear.device
    dtype = ray_tnear.dtype
    batch_shape = ray_tnear.shape
    batch_ones = (1,) * len(batch_shape)

    # The shape of uniform samples is chosen so that the same stratified samples
    # will be generated for a single call with input shape (N,...) or consecutive
    # calls with mini-batches of N when the initial random state matches. This is
    # mostly required for testing purposes.
    u = (
        ray_tnear.new_empty(ray_tnear.shape + (n_samples,))
        .uniform_(0.0, 1.0)
        .movedim(-1, 0)
    )  # (b,N,...,1)
    ifrac = torch.arange(n_samples, dtype=dtype, device=dev) / n_samples
    td = (ray_tfar - ray_tnear).unsqueeze(0)  # (1,N,...,1)
    tnear_bins = (
        ray_tnear.unsqueeze(0) + ifrac.view((n_samples,) + batch_ones) * td
    )  # (b,N,...,1)
    ts = tnear_bins + (td / n_samples) * u
    if ((ts[1:] - ts[:-1]) < 0.0).any():
        print("ts violation", (ts[1:] - ts[:-1]).min(), td.min())
    return ts


def sample_ray_step_informed(
    ts: torch.Tensor,
    tnear: torch.Tensor,
    tfar: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    e: torch.Tensor = None,
) -> torch.Tensor:
    """(Re)samples ray steps from a per-ray probability distribution
    estimated by a discrete set of weights.

    The returned samples per ray are guaranteed to be sorted in ascending
    order along the ray.

    This method makes use of inverse transformation sampling, which states
    that one can generate samples from `f(t)` by reparametrizing uniform
    `[0,1]` samples `u` via the inverse CDF `t=F^-1(u))`.

    In this implementation, we first estimate a piecewise constant CDF from
    the PMF of the given sample weights. Since we need continuous samples `t`,
    we need to assume a non-constant CDF between any two bins. We select a
    uniform distribution. In terms of the CDF this transforms the piecewise
    constant CDF to a piecewise linear CDF.

    To generate a sample `t`, we first sample u from `[0,1]` uniformly. We implement
    `F^-1(u)` by first searching for the CDF bin indices (t-low, t-high) `u` belongs
    to. Then we compute the resulting sample t by solving a linear equation
    `at+bu+c=0` for t. The linear equation found as the line connecting the two points
    `(t-low,cdf-low)`and `(t-high,cdf-high)`.

    To ensure ordered samples along the ray step direction, we first ensure
    that our random uniform samples `u` are ordered via sampling/integrating
    from an exponential distribution. Since the CDF is monotonically increasing,
    and `u` is ordered, the calculated t-samples will also be ordered. Quick warning
    for future reference: the order will be broken if one locates the CDF bin,
    but then samples random uniform variable from that bin: for multiple `u` mapping
    to the same bin, resulting `t` would not be ordered in general.

    Params:
        ts: (T,N,...,1) input ray step values
        tnear: (N,...,1) ray start values
        tfar: (N,...,1) ray end values
        weights: (T,N,...,1) weight for each ray step value in [0,+inf]
        n_samples: number of samples to generate

    Returns
        ts_informed: (n_samples,N,...,1) samples following the
            weight distribution
    """
    T = ts.shape[0]

    # For computational reasons we shuffle the T dimension to last
    ts = ts.squeeze(-1).movedim(0, -1)  # (N,...,T)
    weights = weights.squeeze(-1).movedim(0, -1)  # (N,...,T)

    # Create PMF over weights per ray
    pmf = weights / weights.sum(-1, keepdim=True)  # (N,...,T)
    # Create CDF for inverse uniform sampling
    cdf = pmf.cumsum(dim=-1)  # (N,...,T)
    cdf[..., -1] = 1.0

    # Piecewise linear functions of CDFs between consecutive
    # ts/cdf sample points. Using tools from perspective geometry
    # to construct lines
    xyone = torch.stack(
        (
            ts,
            cdf,
            cdf.new_tensor(1.0).expand(ts.shape),
        ),
        -1,
    )
    lines = torch.cross(
        xyone[..., 1:, :],
        xyone[..., :-1, :],
        dim=-1,
    )  # (N,...,T-1,3), at+bu+c=0

    # Generate n_samples+1 sorted uniform samples per batch
    # See https://cs.stackexchange.com/a/50553/154714
    if e is None:
        e: torch.Tensor = D.Exponential(ts.new_tensor(1.0)).sample(
            ts.shape[:-1] + (n_samples + 1,)
        )
    u = e.cumsum(-1)
    u = u[..., :-1] / u[..., -1:]  # last one is not valid, # (N,...,n_samples)

    # Invoke the inverse CDF: x=CDF^-1(u) by locating the left-edge of the bin
    # u belongs to. This gives us also the linear segment which we solve for t given u
    # t = -(bu+c)/a
    low = torch.searchsorted(cdf, u, side="right") - 1  # (N,...,n_samples)
    low = low.clamp(0, T - 2)  # we have 1 less piecwise lines than input samples
    low = low.unsqueeze(-1).expand(low.shape + (3,))  # (N,...,n_samples,3)
    uline = torch.gather(lines, dim=-2, index=low)  # (N,...,n_samples,3)
    t = -(uline[..., 1] * u + uline[..., 2]) / (uline[..., 0])  # (N,...,n_samples)
    t = t.clamp(tnear, tfar).movedim(-1, 0).unsqueeze(-1).contiguous()  # (T,N,...,1)

    return t


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
