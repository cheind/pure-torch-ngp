import torch
from torch.testing import assert_close

from ngp_nerf.nerf2 import sampling
from ngp_nerf.nerf2 import cameras

from ngp_nerf import linalg


def test_sample_ray_step_stratified():
    tnear = torch.tensor([0.0, 10.0]).repeat_interleave(100).view(2, 100, 1)
    tfar = torch.tensor([1.0, 15.0]).repeat_interleave(100).view(2, 100, 1)
    ts = sampling.sample_ray_step_stratified(tnear, tfar, n_bins=2)

    assert ts.shape == (2, 2, 100, 1)

    ts = ts.squeeze(-1)

    assert ((ts[0, 0] >= 0.0) & (ts[0, 0] <= 0.5)).all()
    assert ((ts[1, 0] >= 0.5) & (ts[1, 0] <= 1.0)).all()
    assert ((ts[0, 1] >= 10.0) & (ts[0, 1] <= 12.5)).all()
    assert ((ts[1, 1] >= 12.5) & (ts[1, 1] <= 15.0)).all()


def test_generate_sequential_uv_samples():
    H, W = 3, 4
    cam = cameras.Camera(
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=1.0,
        width=W,
        height=H,
    )

    # Individual camera
    # Channel first image as input as it is common in PyTorch
    img = torch.randn(1, 1, 3, 4)  # nchw
    gen = sampling.generate_sequential_uv_samples(
        cam, image=img, n_samples_per_cam=4, n_passes=1
    )

    grid = cam.uv_grid()

    uv, f = zip(*list(gen))
    assert len(uv) == 3
    assert len(f) == 3

    assert_close(uv[0], grid[:, 0, :])
    assert_close(uv[1], grid[:, 1, :])
    assert_close(uv[2], grid[:, 2, :])

    # channels last image as it will be used in nerf
    imgp = img.permute(0, 2, 3, 1)
    assert_close(f[0], imgp[:, 0])
    assert_close(f[1], imgp[:, 1])
    assert_close(f[2], imgp[:, 2])

    # Same but with batched cameras
    cams = cameras.CameraBatch([cam, cam])
    grid = cams.uv_grid()
    img = torch.randn(2, 1, 3, 4)  # nchw
    imgp = img.permute(0, 2, 3, 1)
    gen = sampling.generate_sequential_uv_samples(
        cams, image=img, n_samples_per_cam=4, n_passes=1
    )
    uv, f = zip(*list(gen))
    assert len(uv) == 3
    assert len(f) == 3
    assert_close(uv[0], grid[:, 0, :])
    assert_close(uv[1], grid[:, 1, :])
    assert_close(uv[2], grid[:, 2, :])


def test_generate_random_uv_samples():
    H, W = 5, 5
    cam = cameras.Camera(
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=1.0,
        width=W,
        height=H,
    )
    img = torch.randn(1, 1, 3, 4)
    gen = sampling.generate_random_uv_samples(
        cam, image=img, n_samples_per_cam=4, subpixel=True
    )

    counter = torch.zeros((1, H, W))

    from itertools import islice

    for uv, f in islice(gen, 1000):
        uvc = torch.round(uv).long().view(-1, 2)
        counter[0, uvc[:, 1], uvc[:, 0]] += 1  # problematic for duplicate coords

    print(counter.sum())

    exp_freq = (1000 * 4) / (H * W)
    err = (counter - exp_freq).square().mean().sqrt()
    print(err)

    print(counter)
