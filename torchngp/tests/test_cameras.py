import pytest
import torch
from torch.testing import assert_close


from torchngp import modules


def test_camera_shapes():
    H, W = 5, 10
    cam = modules.MultiViewCamera(
        focal_length=[2.0, 2.0],
        principal_point=[(W + 1) / 2 - 1, (H + 1) / 2 - 1],
        size=[W, H],
        rvec=torch.zeros(3),
        tvec=torch.tensor([1.0, 2.0, 3.0]),
        image_paths=["dummy.png"],
        tnear=0,
        tfar=10,
    )

    assert_close(cam.focal_length, torch.tensor([2.0, 2.0]))
    assert_close(cam.principal_point, torch.tensor([4.5, 2.0]))
    assert_close(cam.size, torch.tensor([W, H]).int())
    assert_close(cam.R, torch.eye(3).unsqueeze(0))
    assert_close(cam.tvec, torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1))
    assert_close(cam.tnear, torch.tensor([0.0]))
    assert_close(cam.tfar, torch.tensor([10.0]))
    assert cam.image_paths == ["dummy.png"]

    cam = modules.MultiViewCamera(
        focal_length=[2.0, 2.0],
        principal_point=[(W + 1) / 2 - 1, (H + 1) / 2 - 1],
        size=[W, H],
        rvec=[torch.zeros(3), torch.zeros(3)],
        tvec=[torch.zeros(3), torch.ones(3)],
        tnear=0,
        tfar=10,
    )

    assert_close(cam.focal_length, torch.tensor([2.0, 2.0]))
    assert_close(cam.principal_point, torch.tensor([4.5, 2.0]))
    assert_close(cam.size, torch.tensor([W, H]).int())
    assert_close(cam.tnear, torch.tensor([0.0]))
    assert_close(cam.tfar, torch.tensor([10.0]))
    assert_close(cam.R, torch.eye(3).expand(2, 3, 3))
    assert_close(cam.tvec, torch.stack((torch.zeros(3), torch.ones(3))).view(2, 3, 1))


def test_camera_poses():
    H, W = 5, 10
    from functools import partial

    cam_partial = partial(
        modules.MultiViewCamera,
        focal_length=[2.0, 2.0],
        principal_point=[(W + 1) / 2 - 1, (H + 1) / 2 - 1],
        size=[W, H],
        tnear=0,
        tfar=10,
    )

    cam = cam_partial(
        rvec=[torch.zeros(3), torch.zeros(3)],
        tvec=[torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])],
    )

    cam = cam_partial(
        poses=[torch.eye(4)] * 2,
    )

    with pytest.raises(ValueError):
        cam_partial()  # no poses

    with pytest.raises(ValueError):
        cam_partial(rvec=[torch.zeros(3)])

    with pytest.raises(ValueError):
        cam_partial(tvec=[torch.zeros(3)])

    with pytest.raises(ValueError):
        cam_partial(tvec=[torch.zeros(3)], poses=[torch.eye(4)])


def test_camera_grid():
    H, W = 5, 10
    cam = modules.MultiViewCamera(
        focal_length=[2.0, 2.0],
        principal_point=[(W + 1) / 2 - 1, (H + 1) / 2 - 1],
        size=[W, H],
        rvec=[torch.zeros(3), torch.zeros(3)],
        tvec=[torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])],
        tnear=0,
        tfar=10,
    )

    uv_grid = cam.make_uv_grid()
    assert uv_grid.shape == (2, H, W, 2)
    assert_close(uv_grid[:, 0, 1], torch.Tensor([1.0, 0.0]).expand(2, -1))
    assert_close(uv_grid[:, 1, 0], torch.Tensor([0.0, 1.0]).expand(2, -1))
    assert_close(uv_grid[:, -1, 0], torch.Tensor([0.0, H - 1]).expand(2, -1))
    assert_close(uv_grid[:, 0, -1], torch.Tensor([W - 1.0, 0.0]).expand(2, -1))
