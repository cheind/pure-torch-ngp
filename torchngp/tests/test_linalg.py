import torch
from torch.testing import assert_close

from torchngp.functional import linalg


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
    print(R[0])
    assert_close(
        R[0], torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    )
    assert_close(
        R[1], torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    )
    assert_close(
        R[2], torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    )


def test_rotation_vector():
    R = linalg.rotation_matrix(
        torch.eye(3),
        torch.tensor([0.0, torch.pi / 2, -torch.pi / 2]),
    )
    u, theta = linalg.rotation_vector(R)
    assert_close(u, torch.eye(3))
    assert_close(theta, torch.tensor([0.0, torch.pi / 2, -torch.pi / 2]))

    # Random in float64
    torch.random.manual_seed(123)
    batch = (2, 5, 10, 50)
    u_gt = torch.randn(batch + (3,), dtype=torch.float64)
    u_gt = u_gt / torch.linalg.vector_norm(u_gt, ord=2, dim=-1, keepdim=True)
    theta_gt = torch.empty(batch, dtype=torch.float64).uniform_(-torch.pi, torch.pi)
    R_gt = linalg.rotation_matrix(u_gt, theta_gt)

    u, theta = linalg.rotation_vector(R_gt)
    # Vectors might actually be flipped and signs of theta changed, so we revert
    # back to rot matrices for comparison
    R = linalg.rotation_matrix(u, theta)
    assert (R_gt - R).abs().max() < 1e-8
