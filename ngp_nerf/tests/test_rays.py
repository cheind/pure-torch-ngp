import torch
from torch.testing import assert_close
from ngp_nerf import rays


def test_ray_aabb_intersection():
    aabb_min = torch.Tensor([-1.0, -1.0, -1.0])
    aabb_max = torch.Tensor([1.0, 1.0, 1.0])

    o = torch.tensor([[2.0, 0.0, 0.0]])
    d = torch.tensor([[-1.0, 0.0, 0.0]])

    tnear, tfar = rays.intersect_rays_aabb(o, d, aabb_min, aabb_max)
    assert_close(tnear, torch.tensor([1.0]))
    assert_close(tfar, torch.tensor([3.0]))

    tnear, tfar = rays.intersect_rays_aabb(o, -d, aabb_min, aabb_max)
    assert_close(tnear, torch.tensor([-3.0]))
    assert_close(tfar, torch.tensor([-1.0]))

    tnear, tfar = rays.intersect_rays_aabb(
        o, torch.tensor([[0.0, 0.0, 1.0]]), aabb_min, aabb_max
    )
    assert (tnear > tfar).all()

    o = torch.tensor([[0.0, 0.0, 0.0]])  # inside
    d = torch.randn(100, 3)

    tnear, tfar = rays.intersect_rays_aabb(o, d, aabb_min, aabb_max)
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


def test_sample_rays_uniformly():
    tnear = torch.tensor([0.0, 10.0]).repeat_interleave(100)
    tfar = torch.tensor([1.0, 15.0]).repeat_interleave(100)
    ts = rays.sample_rays_uniformly(
        tnear,
        tfar,
        n_bins=2,
    )

    assert ((ts[:100, 0] >= 0.0) & (ts[:100, 0] <= 0.5)).all()
    assert ((ts[:100, 1] >= 0.5) & (ts[:100, 1] <= 1.0)).all()
    assert ((ts[100:, 0] >= 10.0) & (ts[100:, 0] <= 12.5)).all()
    assert ((ts[100:, 1] >= 12.5) & (ts[100:, 1] <= 15.0)).all()
