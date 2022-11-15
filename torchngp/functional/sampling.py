import torch
import torch.distributions as D


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
    return ts


def sample_ray_fixed_step_stratified(
    ray_tnear: torch.Tensor, stepsize: float, n_samples: int, noise_scale: float = None
) -> torch.Tensor:
    """Creates stratified ray step random samples between tnear/tfar.

    The returned samples per ray are guaranteed to be sorted
    in step ascending order.

    Params:
        ray_tnear: (N,...,1) ray start
        stepsize: step size
        n_samples: number of steps

    Returns:
        tsamples: (n_bins,N,...,1)

    Based on:
        NeRF: Representing Scenes as
        Neural Radiance Fields for View Synthesis
        https://arxiv.org/pdf/2003.08934.pdf
        https://en.wikipedia.org/wiki/Stratified_sampling
    """
    S = ray_tnear.shape
    dev = ray_tnear.device
    dtype = ray_tnear.dtype
    eps = torch.finfo(dtype).eps
    half_step = stepsize * 0.5

    ts = torch.arange(0, stepsize * n_samples, stepsize, device=dev, dtype=dtype)
    ts = ts + half_step
    if noise_scale is None:
        noise_scale = half_step - eps
    assert noise_scale < half_step, "Decrease noise scale or lose sample order"

    u = ray_tnear.new_empty((ts.shape[0],) + S).uniform_(-noise_scale, noise_scale)
    ts = ts.view((ts.shape[0],) + (1,) * len(S)) + ray_tnear + u
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
    eps = torch.finfo(ts.dtype).eps

    # For computational reasons we shuffle the T dimension to last
    ts = ts.squeeze(-1).movedim(0, -1)  # (N,...,T)
    weights = weights.squeeze(-1).movedim(0, -1)  # (N,...,T)

    # Create PMF over weights per ray
    pmf = weights / (weights.sum(-1, keepdim=True) + eps)  # (N,...,T)
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


__all__ = [
    "sample_ray_step_stratified",
    "sample_ray_fixed_step_stratified",
    "sample_ray_step_informed",
]
