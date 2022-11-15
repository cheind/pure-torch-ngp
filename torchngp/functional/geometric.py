import torch
from typing import Union


def make_grid(
    shape: tuple[int, ...],
    indexing: str = "xy",
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.LongTensor:
    """Generalized mesh-grid routine.

    torch.meshgrid `indexing='xy'` only works for 2 dimensions and switches to 'ij'
    for more than two dimensions. This method is consistent for any number of dims.

    Params:
        shape: shape of grid to generate coordinates for
        indexing: order of coordinates in last-dimension
        device: device to put it on
        dtype: dtype to return

    Returns:
        coords: (shape,)+(dims,) tensor
    """
    ranges = [torch.arange(r, device=device, dtype=dtype) for r in shape]
    coords = torch.stack(torch.meshgrid(*ranges, indexing="ij"), -1)
    if indexing == "xy":
        coords = torch.index_select(
            coords, -1, torch.arange(len(shape), device=device).flip(0)
        )
    return coords


def make_multiview_grid(
    n_views: int,
    size: tuple[int, int],
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    """Generates uv-pixel grid coordinates for n-views.

    Params:
        shape: shape of grid to generate coordinates for
        indexing: order of coordinates in last-dimension
        device: device to put it on
        dtype: dtype to return


    Returns:
        uv: (N,H,W,2) tensor of grid coordinates using
            'xy' indexing.
    """
    uv = (
        make_grid(
            shape=size[::-1],
            indexing="xy",
            device=device,
            dtype=dtype,
        )
        .unsqueeze(0)
        .expand(n_views, -1, -1, -1)
    )
    return uv


def normalize_uv(
    x: torch.Tensor, spatial_shape: tuple[int, ...], indexing: str = "xy"
) -> torch.Tensor:
    """Transforms UV pixel coordinates to spatial [-1,+1] range

    We treat pixels as squares with centers at integer coordinates. This method
    converts UV coordinates in the image plane to normalized UV coordinates. That
    is a coordinate of (-0.5,-0.5) gets mapped to (-1,-1) and (W-1+0.5,H-1+0.5) to
    (+1,+1). The generated coordinates with `align_corners=False` setting of
    `torch.nn.functional.grid_sample`.

    This method generalizes straight forward to more than 2 dimensions.

    Params:
        x: (N,...,K) unnormalized coordinates in K dimensions
        spatial_shape: K-tuple of dimensions
        indexing: how the coordinate dimension (last) is indexed.

    Returns:
        xn: normalized coordinates
    """

    if indexing == "xy":
        spatial_shape = spatial_shape[::-1]

    sizes = x.new_tensor(spatial_shape).expand_as(x)

    # suitable for align_corners=False
    return (x + 0.5) * 2 / sizes - 1.0


def denormalize_uv(
    xn: torch.Tensor, spatial_shape: tuple[int, ...], indexing: str = "xy"
) -> torch.Tensor:
    """De-normalize pixel coordinates from [-1,+1] to 0..spatial_shape-1.

    Reverse operation to `normalize_uv`.

    Params:
        xn: (N,...,K) normalized coordinates in K dimensions
        spatial_shape: K-tuple of dimensions
        indexing: how the coordinate dimension (last) is indexed.

    Returns:
        x: de-normalized coordinates
    """

    if indexing == "xy":
        spatial_shape = spatial_shape[::-1]

    sizes = xn.new_tensor(spatial_shape).expand_as(xn)

    return (xn + 1) * sizes * 0.5 - 0.5


def unproject_uv(
    uv: torch.Tensor,
    focal_length: torch.Tensor,
    principal_point: torch.Tensor,
    depth: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """Unprojects uv-pixel coordinates to view space.

    Params:
        cam: Camera
        uv: (N,...,2) uv coordinates to unproject. If not specified, defaults
            to all grid coordiantes.
        focal_length: (2,) tensor containing focal length in x,y direction
        principal_point: (2,) tensor containing principal point in x,y
        depth: scalar or any shape broadcastable to (N,...,1) representing
            the depths of unprojected pixels.

    Returns:
        xyz: (N,...,3) tensor of coordinates.
    """
    N = uv.shape[0]
    mbatch = uv.shape[:-1]

    if not torch.is_tensor(depth):
        depth = uv.new_tensor(depth)

    depth = depth.expand(mbatch + (1,))
    pp = principal_point.expand_as(uv)
    fl = focal_length.expand_as(uv)

    xy = (uv - pp) / fl * depth
    xyz = torch.cat((xy, depth), -1)
    return xyz


def make_world_rays(
    uv: torch.Tensor,
    focal_length: torch.Tensor,
    principal_point: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    tnear: float = 0.0,
    tfar: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns camera rays for each pixel specified in world the frame.

    Depending on the parameter `normalize_dirs` the semantics of tnear,tfar
    changes. When normalize_dirs is false,  tnear and tfar can be interpreted
    as distance to camera parallel to image plane.

    Params:
        uv: (N,...,2) uv coordinates for N views to generate rays for.
        focal_length: (2,) tensor containing focal length in x,y direction
        principal_point: (2,) tensor containing principal point in x,y
        R: (N,3,3) rotation matrices for each view N
        T: (N,3,1) translation vectors for each view N

    Returns:
        ray_origin: (N,...,3) ray origins
        ray_dir: (N,...,3) ray directions
        ray_tnear: (N,...,1) ray start
        ray_tfar: (N,...,1) ray end
    """
    N = uv.shape[0]
    mbatch = uv.shape[1:-1]
    mbatch_ones = (1,) * len(mbatch)

    if not torch.is_tensor(tnear):
        tnear = uv.new_tensor(tnear)
    if not torch.is_tensor(tfar):
        tfar = uv.new_tensor(tfar)

    rot = R.view((N,) + mbatch_ones + (3, 3))
    trans = T.view((N,) + mbatch_ones + (3,))
    xyz = unproject_uv(uv, focal_length, principal_point, depth=1.0)

    # eye rays to world
    ray_dir = (rot @ xyz.unsqueeze(-1)).squeeze(-1)

    ray_origin = trans.expand_as(ray_dir)

    ray_tnear = tnear.expand(N, *mbatch, 1)
    ray_tfar = tfar.expand(N, *mbatch, 1)

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


__all__ = [
    "make_grid",
    "make_multiview_grid",
    "make_world_rays",
    "evaluate_ray",
    "intersect_ray_aabb",
    "convert_world_to_box_normalized",
    "normalize_uv",
]
