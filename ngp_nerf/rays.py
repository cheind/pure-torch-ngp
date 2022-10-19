import torch


def intersect_ray_aabb(
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
