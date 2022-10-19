import torch
import torch.nn.functional as F

from ngp_nerf import pixels


def test_grid_sample():
    img = torch.randn(1, 1, 4, 2, 8)  # (N,C,D,H,W)
    D, H, W = img.shape[-3:]
    xyz = pixels.generate_grid_coords((D, H, W), indexing="xy")
    assert xyz.shape == (D, H, W, 3)
    assert all(xyz[2, 1, 3] == torch.tensor([3, 1, 2]))

    nxyz = pixels.normalize_coords(xyz, indexing="xy")
    dxyz = pixels.denormalize_coords(nxyz, indexing="xy")
    assert (dxyz == xyz).all()

    imgs = F.grid_sample(img, nxyz.unsqueeze(0), mode="bilinear", align_corners=False)
    assert (img - imgs).abs().sum() == 0.0
