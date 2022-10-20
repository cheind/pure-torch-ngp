import torch
from torch.testing import assert_close

from ngp_nerf.scene import View


def test_view():
    H, W = 1080, 1920
    K = torch.tensor(
        [
            [2.6667e03, 0.0000e00, 9.6000e02],
            [0.0000e00, 2.6667e03, 5.4000e02],
            [0.0000e00, 0.0000e00, 1.0000e00],
        ]
    )
    T = torch.tensor(
        [
            [0.9333, 0.3408, -0.1133, 0.8244],
            [0.3591, -0.8856, 0.2944, -2.1425],
            [0.0000, -0.3155, -0.9489, 6.9052],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )
    v = View(K, T, (H, W))
    o, d = v.view_rays()
    assert_close(o, torch.zeros((3))[None, None, :])
    assert_close(d[H // 2, W // 2], torch.tensor([0.0, 0.0, 1.0]))

    o, d = v.world_rays()
    assert_close(o, v.t_world_cam[:3, 3][None, None, :])
    assert_close(d[H // 2, W // 2], v.t_world_cam[:3, 2])
