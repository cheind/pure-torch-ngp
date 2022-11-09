import pytest
import torch
from torch.testing import assert_close

from torchngp import sampling, cameras


def test_sample_ray_step_stratified():
    tnear = torch.tensor([0.0, 10.0]).repeat_interleave(100).view(2, 100, 1)
    tfar = torch.tensor([1.0, 15.0]).repeat_interleave(100).view(2, 100, 1)
    ts = sampling.sample_ray_step_stratified(tnear, tfar, n_samples=2)
    assert ts.shape == (2, 2, 100, 1)

    # Ensure ordered
    assert (ts[1:] - ts[:-1] >= 0).all()

    ts = ts.squeeze(-1)

    assert ((ts[0, 0] >= 0.0) & (ts[0, 0] <= 0.5)).all()
    assert ((ts[1, 0] >= 0.5) & (ts[1, 0] <= 1.0)).all()
    assert ((ts[0, 1] >= 10.0) & (ts[0, 1] <= 12.5)).all()
    assert ((ts[1, 1] >= 12.5) & (ts[1, 1] <= 15.0)).all()

    # Ensure ordered samples
    ts = sampling.sample_ray_step_stratified(tnear, tfar, n_samples=100)
    assert (ts[1:] - ts[:-1] >= 0).all()


def test_sample_ray_step_stratified_repeated_same_as_once():
    torch.random.manual_seed(123)

    ts_once = sampling.sample_ray_step_stratified(
        torch.tensor(0.0).view(1, 1).expand(100, 1),
        torch.tensor(1.0).view(1, 1).expand(100, 1),
        n_samples=100,
    )

    torch.random.manual_seed(123)
    ts_repeated = [
        sampling.sample_ray_step_stratified(
            torch.tensor(0.0).view(1, 1).expand(10, 1),
            torch.tensor(1.0).view(1, 1).expand(10, 1),
            n_samples=100,
        )
        for _ in range(10)
    ]
    ts_repeated = torch.cat(ts_repeated, 1)
    assert_close(ts_once, ts_repeated)


def test_sample_ray_step_informed():

    torch.random.manual_seed(456)
    B = 10
    Ts = 100
    Ti = 100
    tnear = torch.tensor([[0.0]]).expand(B, 1)
    tfar = torch.tensor([[10.0]]).expand(B, 1)

    def compute_weights(ts: torch.Tensor):
        # unnormalized bimodal distribution with sharp peaks
        pi1 = 0.25 * (-((ts - 5.0) ** 2) / 0.5).exp()
        pi2 = 0.75 * (-((ts - 8.0) ** 2) / 0.5).exp()
        return pi1 + pi2

    ts = sampling.sample_ray_step_stratified(tnear, tfar, Ts)
    assert (ts[1:] - ts[:-1] > 0).all()
    weights = compute_weights(ts)

    ts_new = sampling.sample_ray_step_informed(
        ts, tnear, tfar, weights=weights, n_samples=Ti
    )
    weights_new = compute_weights(ts_new)

    # Assert shape
    assert ts_new.shape == (Ti, B, 1)

    # Assert ordered
    assert (ts_new[1:, 2] - ts_new[:-1, 2] >= 0).all()

    # Assert that likelihood of samples increases
    ll = (weights + 1e-5).log().sum()
    ll_new = (weights_new + 1e-5).log().sum()
    assert ll / ll_new > 2.0


@pytest.mark.parametrize(
    "fname",
    ["resample_input_20.pkl", "resample_input_508.pkl", "resample_input_636.pkl"],
)
def test_sample_ray_step_informed_errors(fname):
    from pathlib import Path

    torch.use_deterministic_algorithms(True)
    d = torch.load(
        str(Path("data/testdata/sample_informed") / fname), map_location="cpu"
    )
    ts = sampling.sample_ray_step_informed(**d)
    assert not ((ts[1:] - ts[:-1]) < 0.0).any()
    assert ((ts >= d["tnear"]) & (ts <= d["tfar"])).all()


def test_generate_sequential_uv_samples():
    H, W = 3, 4
    cam = cameras.MultiViewCamera(
        focal_length=[2.0, 2.0],
        principal_point=[1.0, 1.0],
        size=[W, H],
        R=torch.eye(3),
        T=torch.zeros(3),
        tnear=0.0,
        tfar=10.0,
    )

    # Individual camera
    # Channel first image as input as it is common in PyTorch
    img = torch.randn(1, 1, 3, 4)  # nchw
    gen = sampling.generate_sequential_uv_samples(
        cam, image=img, n_samples_per_cam=4, n_passes=1
    )

    grid = cam.make_uv_grid()

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
    cam = cam[[0, 0]]
    grid = cam.make_uv_grid()
    img = torch.randn(2, 1, 3, 4)  # nchw
    imgp = img.permute(0, 2, 3, 1)
    gen = sampling.generate_sequential_uv_samples(
        cam, image=img, n_samples_per_cam=4, n_passes=1
    )
    uv, f = zip(*list(gen))
    assert len(uv) == 3
    assert len(f) == 3
    assert_close(uv[0], grid[:, 0, :])
    assert_close(uv[1], grid[:, 1, :])
    assert_close(uv[2], grid[:, 2, :])


def test_generate_random_uv_samples():
    H, W = 5, 5
    cam = cameras.MultiViewCamera(
        focal_length=[2.0, 2.0],
        principal_point=[1.0, 1.0],
        size=[W, H],
        R=[torch.eye(3), torch.eye(3)],
        T=[torch.zeros(3), torch.zeros(3)],
        tnear=0.0,
        tfar=10.0,
    )

    imgs = torch.randn(2, 1, 5, 5).expand(2, 1, 5, 5)
    N = 1000
    M = 4
    gen = sampling.generate_random_uv_samples(
        cam, image=imgs, n_samples_per_cam=M, subpixel=True
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
