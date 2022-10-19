import torch
from torch.testing import assert_close

from ngp_nerf import linalg


def test_hom_dehom():
    x = torch.randn(100, 2)
    x3d = linalg.hom(x)
    assert x3d.shape == (100, 3)
    assert_close(x3d[:, :2], x)
    assert_close(x3d[:, 2], torch.ones(100))

    x2d = linalg.dehom(x3d)
    assert x2d.shape == (100, 2)
    assert_close(x2d, x)

    x3d = linalg.hom(x, 2.0)
    x2d = linalg.dehom(x3d)
    assert_close(x2d, x * 0.5)
