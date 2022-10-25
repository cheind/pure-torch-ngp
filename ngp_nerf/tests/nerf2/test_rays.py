import torch
from torch.testing import assert_close


from ngp_nerf.nerf2 import cameras, geo
from ngp_nerf.linalg import rotation_matrix


def test_world_rays_shape():
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

    ray_origin, ray_dir, ray_tnear, ray_tfar = geo.world_ray(
        cb, cb.uv_grid(), normalize_dirs=True
    )
    assert ray_origin.shape == (2, H, W, 3)
    assert ray_dir.shape == (2, H, W, 3)
    assert ray_tnear.shape == (2, H, W, 1)
    assert ray_tfar.shape == (2, H, W, 1)


def test_world_rays_origins_directions():
    H, W = 3, 3
    cam = cameras.Camera(
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=1.0,
        width=W,
        height=H,
        R=rotation_matrix(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(torch.pi)),
        T=torch.tensor([1.0, 2.0, 3.0]),
    )
    cb = cameras.CameraBatch([cam, cam])

    ray_origin, ray_dir, ray_tnear, ray_tfar = geo.world_ray(
        cb, cb.uv_grid(), normalize_dirs=False
    )
    assert_close(ray_tnear, torch.tensor([0.0]).expand_as(ray_tnear))
    assert_close(ray_tfar, torch.tensor([100.0]).expand_as(ray_tfar))

    center_dir = ray_dir[
        :, 1, 1, :
    ]  # ray through princ. point should match z-dir of R.

    assert_close(ray_origin, cb.T.view(-1, 1, 1, 3).expand_as(ray_origin))
    assert_close(center_dir, cb.R[..., 2])


def test_ray_aabb_intersection():
    aabb = torch.Tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

    o = torch.tensor([[2.0, 0.0, 0.0]])
    d = torch.tensor([[-1.0, 0.0, 0.0]])
    tnear_initial = torch.tensor([0.0])
    tfar_initial = torch.tensor([10.0])

    tnear, tfar = geo.intersect_ray_aabb(o, d, tnear_initial, tfar_initial, aabb)
    assert_close(tnear, torch.tensor([[1.0]]))
    assert_close(tfar, torch.tensor([[3.0]]))

    # invert direction, because of initial tnear=0, result won't allow -3 as tnear
    tnear, tfar = geo.intersect_ray_aabb(o, -d, tnear_initial, tfar_initial, aabb)
    assert_close(tnear, torch.tensor([[0.0]]))
    assert_close(tfar, torch.tensor([[-1.0]]))

    # however with updated initial bounds we see the hit
    tnear, tfar = geo.intersect_ray_aabb(
        o, -d, torch.tensor([-10.0]), tfar_initial, aabb
    )
    assert_close(tnear, torch.tensor([[-3.0]]))
    assert_close(tfar, torch.tensor([[-1.0]]))

    # miss
    tnear, tfar = geo.intersect_ray_aabb(
        o, torch.tensor([[0.0, 0.0, 1.0]]), tnear_initial, tfar_initial, aabb
    )
    assert (tnear > tfar).all()

    # Inside -> Outside rays
    o = torch.tensor([[0.0, 0.0, 0.0]]).expand(100, -1)  # inside
    d = torch.randn(100, 3)
    tnear_initial = torch.tensor([[0.0]]).expand(100, -1)
    tfar_initial = torch.tensor([[10.0]]).expand(100, -1)

    tnear, tfar = geo.intersect_ray_aabb(o, d, tnear_initial, tfar_initial, aabb)
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

    tnear, tfar = geo.intersect_ray_aabb(o, d, tnear_initial, tfar_initial, aabb)
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


def test_sample_ray_step_stratified():
    tnear = torch.tensor([0.0, 10.0]).repeat_interleave(100).view(2, 100, 1)
    tfar = torch.tensor([1.0, 15.0]).repeat_interleave(100).view(2, 100, 1)
    ts = geo.sample_ray_step_stratified(tnear, tfar, n_bins=2)

    assert ts.shape == (2, 100, 2)

    assert ((ts[0, :, 0] >= 0.0) & (ts[0, :, 0] <= 0.5)).all()
    assert ((ts[0, :, 1] >= 0.5) & (ts[0, :, 1] <= 1.0)).all()
    assert ((ts[1, :, 0] >= 10.0) & (ts[1, :, 0] <= 12.5)).all()
    assert ((ts[1, :, 1] >= 12.5) & (ts[1, :, 1] <= 15.0)).all()
