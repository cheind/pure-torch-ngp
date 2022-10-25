import torch
from torch.testing import assert_close


from ngp_nerf.nerf2 import cameras
from ngp_nerf.linalg import dehom


def test_camera_shapes():
    H, W = 5, 10
    cam = cameras.Camera(
        fx=2.0,
        fy=2.0,
        cx=(W + 1) / 2 - 1,
        cy=(H + 1) / 2 - 1,
        width=W,
        height=H,
        R=torch.eye(3),
        T=torch.tensor([1.0, 2.0, 3.0]),
    )

    assert_close(cam.focal_length, torch.tensor([[2.0, 2.0]]))
    assert_close(cam.principal_point, torch.tensor([[4.5, 2.0]]))
    assert_close(cam.size, torch.tensor([[W, H]]).int())
    assert_close(cam.R, torch.eye(3).unsqueeze(0))
    assert_close(cam.T, torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0))

    cb = cameras.CameraBatch([cam, cam])
    assert_close(cb.focal_length, torch.tensor([2.0, 2.0]).expand(2, -1))
    assert_close(cb.principal_point, torch.tensor([4.5, 2.0]).expand(2, -1))
    assert_close(cb.size, torch.tensor([W, H]).int().expand(2, -1))
    assert_close(cb.R, torch.eye(3).expand(2, -1, -1))
    assert_close(cb.T, torch.tensor([1.0, 2.0, 3.0]).expand(2, -1))


def test_camera_grid():
    H, W = 5, 10
    cam = cameras.Camera(
        fx=2.0,
        fy=2.0,
        cx=(W + 1) / 2 - 1,
        cy=(H + 1) / 2 - 1,
        width=W,
        height=H,
        R=torch.eye(3),
        T=torch.tensor([1.0, 2.0, 3.0]),
    )
    cb = cameras.CameraBatch([cam, cam])

    uv_grid = cb.uv_grid()
    assert uv_grid.shape == (2, H, W, 2)
    assert_close(uv_grid[:, 0, 1], torch.Tensor([1.0, 0.0]).expand(2, -1))
    assert_close(uv_grid[:, 1, 0], torch.Tensor([0.0, 1.0]).expand(2, -1))
    assert_close(uv_grid[:, -1, 0], torch.Tensor([0.0, H - 1]).expand(2, -1))
    assert_close(uv_grid[:, 0, -1], torch.Tensor([W - 1.0, 0.0]).expand(2, -1))


def test_camera_unproject():
    H, W = 5, 10
    cam = cameras.Camera(
        fx=2.0,
        fy=2.0,
        cx=(W + 1) / 2 - 1,
        cy=(H + 1) / 2 - 1,
        width=W,
        height=H,
        R=torch.eye(3),
        T=torch.tensor([1.0, 2.0, 3.0]),
    )
    cb = cameras.CameraBatch([cam, cam])

    # referenced 3D points
    xyz_ref = torch.cat(
        (torch.empty((2, 10, 2)).uniform_(-10.0, 10.0), torch.ones((2, 10, 1)) * 2), -1
    )
    uv_ref = dehom((cb.K.unsqueeze(1) @ xyz_ref.unsqueeze(-1)).squeeze(-1))

    xyz_un = cb.unproject_uv(uv=uv_ref[..., :2], depth=2)
    assert_close(xyz_un, xyz_ref)
