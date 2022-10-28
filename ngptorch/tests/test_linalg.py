import torch
from torch.testing import assert_close

from ngptorch import linalg


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


def test_rotation_matrix():

    R = linalg.rotation_matrix(
        torch.eye(3),
        torch.tensor([torch.pi, torch.pi, torch.pi]),
    )
    assert_close(
        R[0], torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    )
    assert_close(
        R[1], torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    )
    assert_close(
        R[2], torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    )
