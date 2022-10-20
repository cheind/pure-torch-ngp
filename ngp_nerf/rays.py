import torch


def intersect_rays_aabb(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched ray/box intersection using slab method.

    Params:
        origins: (B,3) ray origins
        dirs: (B,3) ray directions
        box_min: (3,) min corner of aabb
        box_max: (3,) max corner of aabb

    Returns:
        tnear: (B,) tnear for each ray. Ray missed box if tnear > tfar
        tfar: (B,) tfar for each ray. Ray missed box if tnear > tfar

    Adapted from
    https://github.com/evanw/webgl-path-tracing/blob/master/webgl-path-tracing.js
    """

    tmin = (box_min.unsqueeze(0) - origins) / dirs
    tmax = (box_max.unsqueeze(0) - origins) / dirs
    t1 = torch.min(tmin, tmax)
    t2 = torch.max(tmin, tmax)

    tnear = t1.max(dim=1)[0]
    tfar = t2.min(dim=1)[0]
    return tnear, tfar


def sample_rays_uniformly(
    tnear: torch.Tensor, tfar: torch.Tensor, n_bins: int
) -> torch.Tensor:
    """Samples rays uniformly in bins between tnear/tfar.

    The returned samples per ray are guaranteed to be
    sorted in ascending order.

    Params:
        tnear: (B,) tensor
        tfar: (B,) tensor

    Returns:
        samples: (B,n_bins,n_samples)


    Based on
    NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis
    https://arxiv.org/pdf/2003.08934.pdf
    """
    B = tnear.shape[0]
    td = tfar - tnear
    # TODO: check inclusive bounds of uniform
    u = tnear.new_empty((B, n_bins)).uniform_(0.0, 1.0)
    ifrac = torch.arange(n_bins, dtype=tnear.dtype, device=tnear.device) / n_bins
    tnear_bins = tnear[:, None] + ifrac[None, :] * td[:, None]
    ts = tnear_bins + (td[:, None] / n_bins) * u
    return ts
