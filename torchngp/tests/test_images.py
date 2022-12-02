import pytest
import torch
import matplotlib.pyplot as plt
from torch.testing import assert_close


from torchngp.functional import images


def test_checkerboard_image():
    rgba = images.checkerboard_image((10, 4, 20, 30))
    assert rgba.shape == (10, 4, 20, 30)
    assert rgba.max() == 1
    assert rgba.min() == 0
    assert_close(rgba[:, 3], torch.ones_like(rgba[:, 3]))
    # plt.imshow(rgba.permute(0, 2, 3, 1)[0])
    # plt.show()


def test_scale_image():
    rgba = torch.rand(10, 4, 20, 30)
    scaled = images.scale_image(rgba, 0.5)
    assert scaled.shape == (10, 4, 10, 15)
    assert_close(rgba.mean(), scaled.mean())


def test_constant_image():
    rgba = images.constant_image((10, 4, 20, 30), (1.0, 2.0, 3.0, 4.0))
    assert rgba.shape == (10, 4, 20, 30)
    assert_close(rgba[:, 0], torch.ones_like(rgba[:, 0]))
    assert_close(rgba[:, 1], torch.ones_like(rgba[:, 1]) * 2)
    assert_close(rgba[:, 2], torch.ones_like(rgba[:, 2]) * 3)
    assert_close(rgba[:, 3], torch.ones_like(rgba[:, 3]) * 4)


def test_image_grid():
    rgba = torch.rand(10, 4, 20, 30)
    grid = images.create_image_grid(rgba, padding=0)
    assert grid.shape == (1, 4, 40, 240)  # 2 rows
