import torch
from torch.testing import assert_close

from ngp_nerf import cameras
from ngp_nerf.linalg import dehom


def test_image_points():
    shape = (200, 300)
    K = torch.tensor(
        [
            [100, 0, 150],
            [0, 100, 100],
            [0, 0, 1.0],
        ]
    )
    ipoints = cameras.image_points(K, shape)
    assert_close(ipoints[100, 150], torch.tensor([0, 0, 1.0]))

    pixels = dehom(ipoints @ K.T)
    assert_close(pixels.view(-1, 2).min(0)[0], torch.tensor([0.0, 0.0]))
    assert_close(pixels.view(-1, 2).max(0)[0], torch.tensor([299.0, 199.0]))


def test_eye_rays():
    shape = (200, 300)
    K = torch.tensor(
        [
            [100, 0, 150],
            [0, 100, 100],
            [0, 0, 1.0],
        ]
    )
    ipoints = cameras.image_points(K, shape)
    assert_close(ipoints[100, 150], torch.tensor([0, 0, 1.0]))

    pixels = dehom(ipoints @ K.T)
    assert_close(pixels.view(-1, 2).min(0)[0], torch.tensor([0.0, 0.0]))
    assert_close(pixels.view(-1, 2).max(0)[0], torch.tensor([299.0, 199.0]))
