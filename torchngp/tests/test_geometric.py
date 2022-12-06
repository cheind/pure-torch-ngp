import torch
from torch.testing import assert_close

from torchngp import modules
from torchngp import functional


def test_world_rays_shape():
    H, W = 5, 10
    cam = modules.MultiViewCamera(
        focal_length=[2.0, 2.0],
        principal_point=[(W + 1) / 2 - 1, (H + 1) / 2 - 1],
        size=[W, H],
        rvec=[torch.zeros(3), torch.zeros(3)],
        tvec=[torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])],
        tnear=0,
        tfar=10,
    )

    rays = modules.RayBundle.make_world_rays(cam, cam.make_uv_grid())

    assert rays.o.shape == (2, H, W, 3)
    assert rays.d.shape == (2, H, W, 3)
    assert rays.tnear.shape == (2, H, W, 1)
    assert rays.tfar.shape == (2, H, W, 1)


def test_world_rays_origins_directions():

    H, W = 3, 3
    cam = modules.MultiViewCamera(
        focal_length=[2.0, 2.0],
        principal_point=[1.0, 1.0],
        size=[W, H],
        rvec=functional.so3_log(
            functional.rotation_matrix(
                torch.tensor([0.0, 0.0, 1.0]), torch.tensor(torch.pi)
            )
        ),
        tvec=torch.tensor([1.0, 2.0, 3.0]),
        tnear=0,
        tfar=100,
    )
    cam = cam[[0, 0]]  # twice the same cam

    rays = modules.RayBundle.make_world_rays(cam, cam.make_uv_grid())
    assert_close(rays.tnear, torch.tensor([0.0]).expand_as(rays.tnear))
    assert_close(rays.tfar, torch.tensor([100.0]).expand_as(rays.tnear))

    center_dir = rays.d[:, 1, 1, :]  # ray through princ. point should match z-dir of R.

    assert_close(rays.o, cam.tvec.view(-1, 1, 1, 3).expand_as(rays.o))
    assert_close(center_dir, cam.R[..., 2])


def test_ray_aabb_intersection():
    aabb = torch.Tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

    o = torch.tensor([[2.0, 0.0, 0.0]])
    d = torch.tensor([[-1.0, 0.0, 0.0]])
    tnear_initial = torch.tensor([0.0])
    tfar_initial = torch.tensor([10.0])

    tnear, tfar = functional.intersect_ray_aabb(o, d, tnear_initial, tfar_initial, aabb)
    assert_close(tnear, torch.tensor([[1.0]]))
    assert_close(tfar, torch.tensor([[3.0]]))

    # invert direction, because of initial tnear=0, result won't allow -3 as tnear
    tnear, tfar = functional.intersect_ray_aabb(
        o, -d, tnear_initial, tfar_initial, aabb
    )
    assert_close(tnear, torch.tensor([[0.0]]))
    assert_close(tfar, torch.tensor([[-1.0]]))

    # however with updated initial bounds we see the hit
    tnear, tfar = functional.intersect_ray_aabb(
        o, -d, torch.tensor([-10.0]), tfar_initial, aabb
    )
    assert_close(tnear, torch.tensor([[-3.0]]))
    assert_close(tfar, torch.tensor([[-1.0]]))

    # miss
    tnear, tfar = functional.intersect_ray_aabb(
        o, torch.tensor([[0.0, 0.0, 1.0]]), tnear_initial, tfar_initial, aabb
    )
    assert (tnear > tfar).all()

    # Inside -> Outside rays
    o = torch.tensor([[0.0, 0.0, 0.0]]).expand(100, -1)  # inside
    d = torch.randn(100, 3)
    tnear_initial = torch.tensor([[0.0]]).expand(100, -1)
    tfar_initial = torch.tensor([[10.0]]).expand(100, -1)

    tnear, tfar = functional.intersect_ray_aabb(o, d, tnear_initial, tfar_initial, aabb)
    assert tnear.shape == (100, 1)
    assert tfar.shape == (100, 1)
    assert (tnear < tfar).all()  # all intersect
    assert (tnear == 0).all()
    assert (tfar > 0).all()

    isect_near = o + tnear * d
    isect_far = o + tfar * d

    # at least one coordinate should be a +/- 1
    assert (isect_near.abs() - 1.0 < 1e-5).any(1).sum() == 100
    assert (isect_far.abs() - 1.0 < 1e-5).any(1).sum() == 100

    # Test with multiple batch dims
    o = torch.tensor([[0.0, 0.0, 0.0]]).expand(100, 3).view(5, 20, 3)
    d = torch.randn(100, 3).view(5, 20, 3)
    tnear_initial = torch.tensor([[0.0]]).expand(100, -1).view(5, 20, 1)
    tfar_initial = torch.tensor([[10.0]]).expand(100, -1).view(5, 20, 1)

    tnear, tfar = functional.intersect_ray_aabb(o, d, tnear_initial, tfar_initial, aabb)
    assert tnear.shape == (5, 20, 1)
    assert tfar.shape == (5, 20, 1)
    assert (tnear < tfar).all()  # all intersect
    assert (tnear == 0).all()
    assert (tfar > 0).all()

    isect_near = o + tnear * d
    isect_far = o + tfar * d

    # at least one coordinate should be a +/- 1
    assert (isect_near.abs() - 1.0 < 1e-5).any(-1).sum() == 100
    assert (isect_far.abs() - 1.0 < 1e-5).any(-1).sum() == 100


def test_convert_world_to_box_normalized():
    aabb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    t = torch.tensor([[10, 10, 10]])

    # xy layout in meshgrid only works for 2 dims
    xyz = torch.stack(
        torch.meshgrid(
            torch.arange(2), torch.arange(2), torch.arange(2), indexing="ij"
        ),
        -1,
    ).float()
    xyz = torch.index_select(xyz, -1, torch.arange(3).flip(0))
    xyz *= torch.tensor([1.0, 2.0, 3.0]).view(1, 1, 1, 3)
    xyz += t.view(1, 1, 1, 3)

    nxyz = functional.convert_world_to_box_normalized(xyz, aabb + t.view(1, 3))

    # Note: shape layout (D,H,W) but indexing is (w,h,d)
    assert_close(nxyz[0, 0, 0], torch.tensor([-1.0, -1.0, -1.0]))
    assert_close(nxyz[0, 0, 1], torch.tensor([1.0, -1.0, -1.0]))
    assert_close(nxyz[0, 1, 0], torch.tensor([-1.0, 1.0, -1.0]))
    assert_close(nxyz[0, 1, 1], torch.tensor([1.0, 1.0, -1.0]))
    assert_close(nxyz[1, 0, 0], torch.tensor([-1.0, -1.0, 1.0]))
    assert_close(nxyz[1, 0, 1], torch.tensor([1.0, -1.0, 1.0]))
    assert_close(nxyz[1, 1, 0], torch.tensor([-1.0, 1.0, 1.0]))
    assert_close(nxyz[1, 1, 1], torch.tensor([1.0, 1.0, 1.0]))


def test_ray_evaluate():
    o = torch.tensor([[2.0, 0.5, 0.0]]).expand(10, 3)
    d = torch.tensor([[-1.0, 0.0, 0.0]]).expand(10, 3)
    t = torch.linspace(0, 2.0, 10).unsqueeze(-1)

    x = functional.evaluate_ray(o, d, t)  # x is (10,3)
    assert_close(x[..., 0], torch.linspace(2.0, 0.0, 10))
    assert_close(x[..., 1], torch.tensor([0.5]).expand(10))
    assert_close(x[..., 2], torch.tensor([0.0]).expand(10))

    t = torch.randn(30, 20, 10, 1)
    x = functional.evaluate_ray(o, d, t)  # x is (30,20,10,3)
    assert x.shape == (30, 20, 10, 3)


def test_make_multiview_grid():
    grid = functional.make_multiview_grid(4, (5, 3), dtype=int)
    assert grid.shape == (4, 3, 5, 2)

    assert_close(grid[0, 0, 0], torch.tensor((0, 0)))
    assert_close(grid[0, -1, -1], torch.tensor((4, 2)))
    assert_close(grid[0, -1, 0], torch.tensor((0, 2)))
    assert_close(grid[0, 0, -1], torch.tensor((4, 0)))

    assert_close(grid[2, 0, 0], torch.tensor((0, 0)))
    assert_close(grid[2, -1, -1], torch.tensor((4, 2)))
    assert_close(grid[2, -1, 0], torch.tensor((0, 2)))
    assert_close(grid[2, 0, -1], torch.tensor((4, 0)))

    grid = functional.make_multiview_grid(4, (5, 3), dtype=torch.float32)
    assert grid.shape == (4, 3, 5, 2)

    assert_close(grid[0, 0, 0], torch.tensor((0.0, 0)))
    assert_close(grid[0, -1, -1], torch.tensor((4.0, 2)))
    assert_close(grid[0, -1, 0], torch.tensor((0.0, 2)))
    assert_close(grid[0, 0, -1], torch.tensor((4.0, 0)))

    assert_close(grid[2, 0, 0], torch.tensor((0.0, 0)))
    assert_close(grid[2, -1, -1], torch.tensor((4.0, 2)))
    assert_close(grid[2, -1, 0], torch.tensor((0.0, 2)))
    assert_close(grid[2, 0, -1], torch.tensor((4.0, 0)))
