import torch

from .cameras import BaseCamera


def world_ray_from_pixel(
    cam: BaseCamera, uv: torch.Tensor = None, normalize_dirs: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns camera rays for each pixel specified in world the frame.

    Depending on the parameter `normalize_dirs` the semantics of tnear,tfar
    changes. When normalize_dirs is false,  tnear and tfar can be interpreted
    as distance to camera parallel to image plane.

    Params:
        cam: CameraBatch of size N
        uv: (N,...,2) uv coordinates to generate rays for. If not specified,
            defaults to all grid coordiantes.
        normalize_dirs: wether to normalize ray directions to unit length

    Returns:
        ray_origin: (N,...,3) ray origins
        ray_dir: (N,...,3) ray directions
        ray_tnear: (N,...,1) ray start
        ray_tfar: (N,...,1) ray end
    """
    if uv is None:
        uv = cam.uv_grid()
    N = cam.focal_length.shape[0]
    mbatch = uv.shape[1:-1]
    mbatch_ones = (1,) * len(mbatch)

    rot = cam.R.view((N,) + mbatch_ones + (3, 3))
    trans = cam.T.view((N,) + mbatch_ones + (3,))
    xyz = cam.unproject_uv(uv, depth=1.0)
    ray_dir = (rot @ xyz.unsqueeze(-1)).squeeze(-1)

    s = 1.0
    if normalize_dirs:
        s = 1.0 / torch.norm(ray_dir, p=2, dim=-1, keepdim=True)
        ray_dir *= s

    ray_origin = trans.expand_as(ray_dir)
    ray_tnear = cam.tnear.view((N,) + mbatch_ones + (1,)).expand(-1, *mbatch, -1) * s
    ray_tfar = cam.tfar.view((N,) + mbatch_ones + (1,)).expand(-1, *mbatch, -1) * s

    return ray_origin, ray_dir, ray_tnear, ray_tfar


@torch.jit.script
def evaluate_ray(
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    ray_t: torch.Tensor,
) -> torch.Tensor:
    """Evaluate rays at a specific parameter.

    Params:
        ray_origin: (N,...,3) ray origins
        ray_dir: (N,...,3) ray directions
        ray_t: (N,...,1) or in general (T,...,N,...,1) ray eval parameters

    Returns:
        xyz: (N,...,3) or (T,...,N,...,3) locations
    """
    new_dims = (1,) * (ray_t.dim() - ray_origin.dim())
    o = ray_origin.view(new_dims + ray_origin.shape)
    d = ray_dir.view(new_dims + ray_dir.shape)
    return o + ray_t * d


def intersect_ray_aabb(
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    ray_tnear: torch.Tensor,
    ray_tfar: torch.Tensor,
    box: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched ray/box intersection using slab method.

    Params:
        ray_origin: (N,...,3) ray origins
        ray_dir: (N,...,3) ray directions
        ray_tnear: (N,...,1) ray starts
        ray_tfar: (N,...,1) ray ends
        box: (2,3) min/max corner of aabb

    Returns:
        ray_tnear: (N,...,1) updated ray starts. Ray missed box if tnear > tfar
        ray_tfar: (N,...,1) updated ray ends. Ray missed box if tnear > tfar

    Adapted from
    https://github.com/evanw/webgl-path-tracing/blob/master/webgl-path-tracing.js
    """
    tmin = (box[0].expand_as(ray_origin) - ray_origin) / ray_dir
    tmax = (box[1].expand_as(ray_origin) - ray_origin) / ray_dir

    t1 = torch.min(tmin, tmax)
    t2 = torch.max(tmin, tmax)

    tnear = t1.max(dim=-1, keepdim=True)[0]
    tfar = t2.min(dim=-1, keepdim=True)[0]

    # Respect initial bounds
    tnear = torch.max(tnear, ray_tnear)
    tfar = torch.min(tfar, ray_tfar)

    return tnear, tfar


def convert_world_to_box_normalized(
    xyz: torch.Tensor, box: torch.Tensor
) -> torch.Tensor:
    """Convert world points to normalized box coordinates [-1,+1].

    Params:
        xyz: (N,...,3) points
        box: (2,3) tensor containing min/max box corner in world coordinates

    Returns:
        nxyz: (N,...,3) normalized coordinates
    """
    span = box[1] - box[0]
    return (xyz - box[0].expand_as(xyz)) / span.expand_as(xyz) * 2.0 - 1.0
