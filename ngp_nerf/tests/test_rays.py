import torch
from torch.testing import assert_close
from ngp_nerf import rays


def test_ray_aabb_intersection():
    aabb_min = torch.Tensor([-1.0, -1.0, -1.0])
    aabb_max = torch.Tensor([1.0, 1.0, 1.0])

    o = torch.tensor([[2.0, 0.0, 0.0]])
    d = torch.tensor([[-1.0, 0.0, 0.0]])

    tnear, tfar = rays.intersect_ray_aabb(o, d, aabb_min, aabb_max)
    assert_close(tnear, torch.tensor([1.0]))
    assert_close(tfar, torch.tensor([3.0]))

    tnear, tfar = rays.intersect_ray_aabb(o, -d, aabb_min, aabb_max)
    assert_close(tnear, torch.tensor([-3.0]))
    assert_close(tfar, torch.tensor([-1.0]))

    tnear, tfar = rays.intersect_ray_aabb(
        o, torch.tensor([[0.0, 0.0, 1.0]]), aabb_min, aabb_max
    )
    assert (tnear > tfar).all()

    o = torch.tensor([[0.0, 0.0, 0.0]])  # inside
    d = torch.randn(100, 3)

    tnear, tfar = rays.intersect_ray_aabb(o, d, aabb_min, aabb_max)
    assert tnear.shape == (100,)
    assert tfar.shape == (100,)
    assert (tnear < tfar).all()  # all intersect
    assert (tnear < 0).all()
    assert (tfar > 0).all()

    isect_near = o + tnear[:, None] * d
    isect_far = o + tfar[:, None] * d

    # at least one coordinate should be a +/- 1
    assert (isect_near.abs() - 1.0 < 1e-5).any(1).sum() == 100
    assert (isect_far.abs() - 1.0 < 1e-5).any(1).sum() == 100
