import pytest
import torch
from torch.testing import assert_close

from torchngp.functional import z_sampling


def test_sample_ray_step_stratified():
    tnear = torch.tensor([0.0, 10.0]).repeat_interleave(100).view(2, 100, 1)
    tfar = torch.tensor([1.0, 15.0]).repeat_interleave(100).view(2, 100, 1)
    ts = z_sampling.sample_ray_step_stratified(tnear, tfar, n_samples=2)
    assert ts.shape == (2, 2, 100, 1)

    # Ensure ordered
    assert (ts[1:] - ts[:-1] >= 0).all()

    ts = ts.squeeze(-1)

    assert ((ts[0, 0] >= 0.0) & (ts[0, 0] <= 0.5)).all()
    assert ((ts[1, 0] >= 0.5) & (ts[1, 0] <= 1.0)).all()
    assert ((ts[0, 1] >= 10.0) & (ts[0, 1] <= 12.5)).all()
    assert ((ts[1, 1] >= 12.5) & (ts[1, 1] <= 15.0)).all()

    # Ensure ordered samples
    ts = z_sampling.sample_ray_step_stratified(tnear, tfar, n_samples=100)
    assert (ts[1:] - ts[:-1] >= 0).all()


def test_sample_ray_fixed_step_stratified():
    tnear = torch.tensor([0.0, 10.0]).repeat_interleave(100).view(2, 100, 1)
    # disable noise
    ts = z_sampling.sample_ray_fixed_step_stratified(
        tnear, stepsize=0.5, n_samples=2, noise_scale=0
    )
    assert ts.shape == (2, 2, 100, 1)
    assert (ts[1:] - ts[:-1] >= 0).all()

    assert_close(ts[:, 0], torch.tensor([0.25, 0.75]).view(2, 1, 1).expand(-1, 100, -1))
    assert_close(
        ts[:, 1], torch.tensor([10.25, 10.75]).view(2, 1, 1).expand(-1, 100, -1)
    )

    # enable noise
    tnear = torch.tensor([0.0]).repeat_interleave(10000).view(10000, 1)
    ts = z_sampling.sample_ray_fixed_step_stratified(
        tnear, stepsize=0.5, n_samples=32, noise_scale=None
    )
    assert ts.shape == (32, 10000, 1)
    assert (ts[1:] - ts[:-1] >= 0).all()

    # import matplotlib.pyplot as plt

    # # enable noise
    # tnear = torch.tensor([0.0]).view(1, 1)
    # ts = sampling.sample_ray_fixed_step_stratified(
    #     tnear, stepsize=0.5, n_samples=32, noise_scale=None
    # )
    # plt.plot(ts[:, 0], torch.tensor([1.0]).expand_as(ts[:, 0]))
    # plt.scatter(ts[:, 0], torch.tensor([1.0]).expand_as(ts[:, 0]))

    # ts = sampling.sample_ray_fixed_step_stratified(
    #     tnear, stepsize=0.5, n_samples=32, noise_scale=1e-1
    # )
    # plt.plot(ts[:, 0], torch.tensor([2.0]).expand_as(ts[:, 0]))
    # plt.scatter(ts[:, 0], torch.tensor([2.0]).expand_as(ts[:, 0]))
    # plt.show()


def test_sample_ray_step_stratified_repeated_same_as_once():
    torch.random.manual_seed(123)

    ts_once = z_sampling.sample_ray_step_stratified(
        torch.tensor(0.0).view(1, 1).expand(100, 1),
        torch.tensor(1.0).view(1, 1).expand(100, 1),
        n_samples=100,
        noise_scale=0,
    )

    torch.random.manual_seed(123)
    ts_repeated = [
        z_sampling.sample_ray_step_stratified(
            torch.tensor(0.0).view(1, 1).expand(10, 1),
            torch.tensor(1.0).view(1, 1).expand(10, 1),
            n_samples=100,
            noise_scale=0,
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

    ts = z_sampling.sample_ray_step_stratified(tnear, tfar, Ts)
    assert (ts[1:] - ts[:-1] > 0).all()
    weights = compute_weights(ts)

    ts_new = z_sampling.sample_ray_step_informed(
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
    ts = z_sampling.sample_ray_step_informed(**d)
    assert not ((ts[1:] - ts[:-1]) < 0.0).any()
    assert ((ts >= d["tnear"]) & (ts <= d["tfar"])).all()
