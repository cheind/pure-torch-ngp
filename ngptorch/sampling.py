from typing import Iterator, Optional
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
    ray_tnear: torch.Tensor, ray_tfar: torch.Tensor, n_bins: int
) -> torch.Tensor:
    """Creates stratified ray step samples between tnear/tfar.

    The returned samples per ray are guaranteed to be
    sorted in ascending order.

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

    u = ray_tnear.new_empty((n_bins,) + ray_tnear.shape).uniform_(
        0.0, 1.0
    )  # (b,N,...,1)
    ifrac = torch.arange(n_bins, dtype=dtype, device=dev) / n_bins
    td = (ray_tfar - ray_tnear).unsqueeze(0)  # (1,N,...,1)
    tnear_bins = (
        ray_tnear.unsqueeze(0) + ifrac.view((n_bins,) + batch_ones) * td
    )  # (b,N,...,1)
    ts = tnear_bins + (td / n_bins) * u
    return ts


def sample_ray_step_informed(
    ts: torch.Tensor,
    tnear: torch.Tensor,
    tfar: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """

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

    # For computational reasons we shuffle the T dimension to last
    ts = ts.movedim(0, -1)  # (N,...,1,T)
    weights = weights.movedim(0, -1)  # (N,...,1,T)

    print("ts", ts.shape)

    # Create PMF over weights per ray
    pmf = weights / weights.sum(-1, keepdim=True)  # (N,...,1,T)
    # Create CDF for inverse uniform sampling
    cdf = pmf.cumsum(dim=-1)  # (N,...,1,T)
    # For boundary constraints, we add -eps to front and 1+eps to end of CDF.
    # These correspond to sample positions tnear and tfar
    eps = 1e-5
    cdf = torch.cat(
        (
            torch.tensor([-eps]).expand(cdf.shape[:-1] + (1,)),
            cdf,
            torch.tensor([1 + eps]).expand(cdf.shape[:-1] + (1,)),
        ),
        -1,
    )  # (N,...,1,T+2)
    print(cdf.shape)
    ts = torch.cat((tnear.unsqueeze(-1), ts, tfar.unsqueeze(-1)), -1)  # (N,...,1,T+2)

    # Generate n_samples+1 sorted uniform samples per batch
    # See https://cs.stackexchange.com/a/50553/154714
    e: torch.Tensor = D.Exponential(ts.new_tensor(1.0)).sample(
        ts.shape[:-1] + (n_samples + 1,)
    )
    u = e.cumsum(-1)
    print(u.shape)
    u = u[..., :-1] / u[..., -1:]
    print("u", u.shape)
    # u = u[..., :-1]  # last one is not valid, # (N,...,1,T)

    # Invoke the inverse CDF: x=CDF^-1(u) by locating the left-edge of the bin
    # u belongs to
    low = torch.searchsorted(cdf, u, side="right") - 1
    print(low.shape)

    # fall into (add 2 pseudo values front and back corresponding to tnear, tfar)
    # into cdf and also concat with ts.
    # sorted_sequence = torch.tensor([0-1e-5, 0.0, 0.5, 0.8, 1.0, 1.0+1e-5])
    # low =torch.searchsorted(sorted_sequence, torch.tensor([0.0]), side="right") - 1
    # then final = uniform(0,1)*(ts[low+1]-ts[low]) + ts[low], assuming that we
    # have linear ramps between two support points of cdf, instead of flat region.
    # note, the above idea with uniform sample in low-high is nice but would break
    # sorted order if two samples fall into the same bin. so instead, assume linear
    # function between low and high and compute intersection. Use homogeneous coordinates
    # and cross
    # In [2]: a = torch.tensor([7,2,1.])
    # In [3]: b = torch.tensor([3., -2, 1.])
    # In [4]: line = torch.cross(a,b) -> 4,-4,-20 = 4x - 4y -20 = 0
    # then to find the intersection, turn u into a horizontal line
    # uline = torch.cross(torch.tensor([0.0,u,1.0]), torch.tensor([1.0,u,1.0]))
    # uline == torch.tensor([0.0, 1.0, -u])
    # x = torch.cross(line, uline)
    # x = x / x[-1]
    # note, torch.searchsorted needs the innermost dimension sorted, so we need
    # to go to (N,...,1,T) using x.transpose(0,-1)


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


if __name__ == "__main__":
    print("here")
    ts = torch.rand(20, 2, 1) * 5 + 0.5
    tnear = ts.min(0)[0] - 1.0
    tfar = ts.max(0)[0] + 1.0
    weights = torch.rand(20, 2, 1)

    sample_ray_step_informed(ts, tnear, tfar, weights, 10)
