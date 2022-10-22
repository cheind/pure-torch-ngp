import torch
import torch.nn.functional as F
from ngp_nerf.interpolation import _bilinear_interpolate


def test_bilinear_interpolation():
    H, W = 200, 320
    img = torch.randn((2, H, W))
    x = torch.empty((1000, 2)).uniform_(-1.0, 1.0)

    y = (
        F.grid_sample(
            img.unsqueeze(0),
            x.view(1, -1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        .view(2, 1000)
        .permute(1, 0)
        .clone()
    )

    # we call the testing function instead of
    # compute_bilinear_params directly
    yhat = _bilinear_interpolate(img, x)
    assert (y - yhat).abs().max() < 1e-4


def test_trilinear_interpolation():
    D, H, W = 100, 200, 320
    img = torch.randn((2, D, H, W))
    x = torch.empty((1000, 3)).uniform_(-1.0, 1.0)

    y = (
        F.grid_sample(
            img.unsqueeze(0),
            x.view(1, -1, 1, 1, 3),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        .view(2, 1000)
        .permute(1, 0)
        .clone()
    )

    # we call the testing function instead of
    # compute_bilinear_params directly
    yhat = _bilinear_interpolate(img, x)
    assert (y - yhat).abs().max() < 1e-4
