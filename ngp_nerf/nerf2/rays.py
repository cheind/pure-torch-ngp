import torch
import torch.nn.functional as F

from .cameras import BaseCamera


def world_rays(
    cam: BaseCamera, uv: torch.Tensor = None, normalize_dirs: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
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

    if normalize_dirs:
        F.normalize(ray_dir, p=2, dim=-1)

    ray_origin = trans.expand_as(ray_dir)

    return ray_origin, ray_dir
