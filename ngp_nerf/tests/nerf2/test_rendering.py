import torch
from torch.testing import assert_close
import matplotlib as mpl

from ngp_nerf.nerf2 import geo, radiance, cameras, rendering

from .test_radiance import ColorGradientRadianceField


def test_render_volume_stratified():
    aabb = torch.Tensor([[0.0] * 3, [1.0] * 3])
    rf = ColorGradientRadianceField(
        aabb=aabb,
        surface_pos=0.5,
        surface_dim=2,
        density_scale=1e1,  # hard surface boundary
        cmap="hsv",
    )

    cam = cameras.MultiViewCamera(
        focal_length=[50.0, 50.0],
        principal_point=[15.0, 15.0],
        size=[31, 31],
        R=torch.eye(3),
        T=torch.Tensor([0.5, 0.5, -1.0]),
        tnear=0.0,
        tfar=10.0,
    )

    torch.random.manual_seed(123)
    color, alpha = rendering.render_volume_stratified(
        rf, aabb, cam, cam.make_uv_grid(), n_ray_steps=200
    )
    img = torch.cat((color, alpha), -1)

    import matplotlib.pyplot as plt

    plt.imshow(img.squeeze(0))
    plt.show()

    color_parts = []
    alpha_parts = []
    from ngp_nerf.nerf2.sampling import generate_sequential_uv_samples

    torch.random.manual_seed(123)
    for uv, _ in generate_sequential_uv_samples(cam):
        color, alpha = rendering.render_volume_stratified(
            rf, aabb, cam, uv, n_ray_steps=200
        )
        color_parts.append(color)
        alpha_parts.append(alpha)

    W, H = cam.size
    color = torch.cat(color_parts, 1).view(1, H, W, 3)
    alpha = torch.cat(alpha_parts, 1).view(1, H, W, 1)
    img2 = torch.cat((color, alpha), -1)
    print((img - img2).abs().max())
    plt.imshow(img.squeeze(0))
    plt.show()
