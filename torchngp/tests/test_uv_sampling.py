import pytest
import torch
from torch.testing import assert_close

from torchngp.functional import uv_sampling
from torchngp.functional import geometric


def test_generate_sequential_uv_samples():
    # Single camera
    # Channel first image as input as it is common in PyTorch

    H, W = 3, 4
    img = torch.randn(1, 1, 3, 4)  # nchw
    gen = uv_sampling.generate_sequential_uv_samples(
        uv_size=(W, H), n_views=1, image=img, n_samples_per_view=W, n_passes=1
    )

    grid = geometric.make_multiview_grid(1, (W, H), dtype=torch.float32)

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
    img = torch.randn(2, 1, 3, 4)  # nchw
    imgp = img.permute(0, 2, 3, 1)
    gen = uv_sampling.generate_sequential_uv_samples(
        uv_size=(W, H), n_views=2, image=img, n_samples_per_view=W, n_passes=1
    )
    grid = geometric.make_multiview_grid(2, (W, H), dtype=torch.float32)
    uv, f = zip(*list(gen))
    assert len(uv) == 3
    assert len(f) == 3
    assert_close(uv[0], grid[:, 0, :])
    assert_close(uv[1], grid[:, 1, :])
    assert_close(uv[2], grid[:, 2, :])


def test_generate_random_uv_samples():
    H, W = 5, 5

    imgs = torch.randn(2, 1, 5, 5).expand(2, 1, 5, 5)
    N = 1000
    M = 4
    gen = uv_sampling.generate_random_uv_samples(
        (W, H), 2, image=imgs, n_samples_per_view=M, subpixel=True
    )

    from itertools import islice
    import torch.nn.functional as F

    counter = torch.zeros((H, W))
    for uv, f in islice(gen, N):
        uvn = (uv + 0.5) * 2 / torch.tensor([[W, H]]) - 1.0
        f_ref = (
            F.grid_sample(
                imgs,
                uvn.unsqueeze(-2),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(-1)
            .permute(0, 2, 1)
        )
        assert (f_ref - f).abs().sum() < 1e-5
        uvc = torch.round(uv).long().view(-1, 2)
        coords, freq = torch.unique(uvc, dim=0, return_counts=True)
        counter[coords[:, 1], coords[:, 0]] += freq

    # Test for uniform dist
    exp_freq = (imgs.shape[0] * N * M) / (H * W)
    err = (counter - exp_freq).square().mean().sqrt()
    rel_error = err / counter.sum()
    assert rel_error < 0.01


def test_generate_randperm_uv_samples():
    H, W = 5, 5

    imgs = torch.randn(2, 1, 5, 5).expand(2, 1, 5, 5)
    N = 1000
    M = 4
    gen = uv_sampling.generate_randperm_uv_samples(
        (W, H), n_views=2, image=imgs, n_samples_per_view=M, subpixel=True
    )

    from itertools import islice
    import torch.nn.functional as F

    counter = torch.zeros((H, W))
    for uv, f in islice(gen, N):
        uvn = (uv + 0.5) * 2 / torch.tensor([[W, H]]) - 1.0
        f_ref = (
            F.grid_sample(
                imgs,
                uvn.unsqueeze(-2),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(-1)
            .permute(0, 2, 1)
        )
        assert (f_ref - f).abs().sum() < 1e-5
        uvc = uv.round().long().view(-1, 2)
        coords, freq = torch.unique(uvc, dim=0, return_counts=True)
        counter[coords[:, 1], coords[:, 0]] += freq

    # Test for uniform dist
    exp_freq = (imgs.shape[0] * N * M) / (H * W)
    err = (counter - exp_freq).square().mean().sqrt()
    rel_error = err / counter.sum()
    assert rel_error < 0.01
