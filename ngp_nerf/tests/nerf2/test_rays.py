import torch
from torch.testing import assert_close


from ngp_nerf.nerf2 import cameras, rays
from ngp_nerf.linalg import dehom


def test_world_rays():
    H, W = 5, 10
    cam = cameras.Camera(
        fx=2.0,
        fy=2.0,
        cx=(W + 1) / 2 - 1,
        cy=(H + 1) / 2 - 1,
        width=W,
        height=H,
        R=torch.eye(3),
        T=torch.tensor([1.0, 2.0, 3.0]),
    )
    cb = cameras.CameraBatch([cam, cam])

    ray_origin, ray_dir = rays.world_rays(cb, cb.uv_grid(), normalize_dirs=True)
    assert ray_origin.shape == (2, H, W, 3)
    assert ray_dir.shape == (2, H, W, 3)
