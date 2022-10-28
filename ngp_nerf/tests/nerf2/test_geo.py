import torch
from torch.testing import assert_close


from ngp_nerf.nerf2 import cameras, geo
from ngp_nerf.linalg import rotation_matrix, dehom


def test_uv_unproject():
    H, W = 5, 10
    cam = cameras.MultiViewCamera(
        focal_length=[2.0, 2.0],
        principal_point=[(W + 1) / 2 - 1, (H + 1) / 2 - 1],
        size=[W, H],
        R=[torch.eye(3), torch.eye(3)],
        T=[torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])],
        tnear=0,
        tfar=10,
    )

    # referenced 3D points
    xyz_ref = torch.cat(
        (torch.empty((2, 10, 2)).uniform_(-10.0, 10.0), torch.ones((2, 10, 1)) * 2), -1
    )
    uv_ref = dehom((cam.K.view(1, 1, 3, 3) @ xyz_ref.unsqueeze(-1)).squeeze(-1))

    xyz_un = geo.unproject_uv(cam, uv=uv_ref[..., :2], depth=2)
    assert_close(xyz_un, xyz_ref)


def test_world_rays_shape():
    H, W = 5, 10
    cam = cameras.MultiViewCamera(
        focal_length=[2.0, 2.0],
        principal_point=[(W + 1) / 2 - 1, (H + 1) / 2 - 1],
        size=[W, H],
        R=[torch.eye(3), torch.eye(3)],
        T=[torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])],
        tnear=0,
        tfar=10,
    )

    ray_origin, ray_dir, ray_tnear, ray_tfar = geo.world_ray_from_pixel(
        cam, cam.make_uv_grid(), normalize_dirs=True
    )
    assert ray_origin.shape == (2, H, W, 3)
    assert ray_dir.shape == (2, H, W, 3)
    assert ray_tnear.shape == (2, H, W, 1)
    assert ray_tfar.shape == (2, H, W, 1)


def test_world_rays_origins_directions():

    H, W = 3, 3
    cam = cameras.MultiViewCamera(
        focal_length=[2.0, 2.0],
        principal_point=[1.0, 1.0],
        size=[W, H],
        R=rotation_matrix(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(torch.pi)),
        T=torch.tensor([1.0, 2.0, 3.0]),
        tnear=0,
        tfar=100,
    )
    cam = cam[[0, 0]]
    ray_origin, ray_dir, ray_tnear, ray_tfar = geo.world_ray_from_pixel(
        cam, cam.make_uv_grid(), normalize_dirs=False
    )
    assert_close(ray_tnear, torch.tensor([0.0]).expand_as(ray_tnear))
    assert_close(ray_tfar, torch.tensor([100.0]).expand_as(ray_tfar))

    center_dir = ray_dir[
        :, 1, 1, :
    ]  # ray through princ. point should match z-dir of R.

    assert_close(ray_origin, cam.T.view(-1, 1, 1, 3).expand_as(ray_origin))
    assert_close(center_dir, cam.R[..., 2])


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

    nxyz = geo.convert_world_to_box_normalized(xyz, aabb + t.view(1, 3))

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

    x = geo.evaluate_ray(o, d, t)  # x is (10,3)
    assert_close(x[..., 0], torch.linspace(2.0, 0.0, 10))
    assert_close(x[..., 1], torch.tensor([0.5]).expand(10))
    assert_close(x[..., 2], torch.tensor([0.0]).expand(10))

    t = torch.randn(30, 20, 10, 1)
    x = geo.evaluate_ray(o, d, t)  # x is (30,20,10,3)
    assert x.shape == (30, 20, 10, 3)
