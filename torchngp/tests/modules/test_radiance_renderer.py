import torch
from torch.testing import assert_close
import matplotlib.pyplot as plt

from torchngp import modules
from torchngp.functional import uv_sampling

from ..functional.test_radiance import ColorGradientRadianceField


def test_render_volume_stratified():
    aabb = torch.tensor([[0.0] * 3, [1.0] * 3])
    rf = ColorGradientRadianceField(
        surface_pos=0.2 * 2 - 1,
        surface_dim=2,
        density_scale=1e1,  # soft density scale
        cmap="jet",
    )
    vol = modules.Volume(aabb, rf)

    cam = modules.MultiViewCamera(
        focal_length=[50.0, 50.0],
        principal_point=[15.0, 15.0],
        size=[31, 31],
        rvec=torch.zeros(3),
        tvec=torch.tensor([0.5, 0.5, -1.0]),
        tnear=0.0,
        tfar=10.0,
    )

    rdr = modules.RadianceRenderer(tfar_scale=1)
    which_maps = {"color", "alpha"}
    torch.random.manual_seed(123)
    tsampler = modules.StratifiedRayStepSampler(n_samples=2048, noise_scale=0)
    result = rdr.trace_uv(vol, cam, cam.make_uv_grid(), tsampler, which_maps=which_maps)
    img = torch.cat((result["color"], result["alpha"]), -1)

    torch.random.manual_seed(123)
    result = rdr.trace_uv(vol, cam, cam.make_uv_grid(), tsampler, which_maps=which_maps)
    img2 = torch.cat((result["color"], result["alpha"]), -1)
    # assert_close(img, img2)

    # _, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    # axs[0].imshow(img.squeeze(0))
    # axs[1].imshow(img2.squeeze(0))
    # axs[2].imshow((img - img2).abs().sum(-1).squeeze(0))
    # plt.show()

    # TODO: test this

    color_parts = []
    alpha_parts = []

    torch.random.manual_seed(123)
    tsampler = modules.StratifiedRayStepSampler(n_samples=2048, noise_scale=0)
    for uv, _ in uv_sampling.generate_sequential_uv_samples(cam.size, cam.n_views):
        maps = rdr.trace_uv(vol, cam, uv, tsampler, which_maps=which_maps)
        color_parts.append(maps["color"])
        alpha_parts.append(maps["alpha"])

    W, H = cam.size
    color = torch.cat(color_parts, 1).view(1, H, W, 3)
    alpha = torch.cat(alpha_parts, 1).view(1, H, W, 1)
    img3 = torch.cat((color, alpha), -1)

    # _, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    # axs[0].imshow(img.squeeze(0))
    # axs[1].imshow(img2.squeeze(0))
    # axs[2].imshow((img - img2).abs().sum(-1).squeeze(0))
    # plt.show()

    assert_close(
        img, img3, atol=1e-5, rtol=1e-5
    )  # TODO: when normalize_dirs=False/True gives different results
