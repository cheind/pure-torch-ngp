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
        density_scale=10000.0,  # hard surface boundary
        cmap="hsv",
    )

    cam = cameras.Camera(
        fx=20.0,
        fy=20.0,
        cx=15,
        cy=15,
        width=31,
        height=31,
        T=torch.Tensor([0.5, 0.5, -1.0]),
    )

    color, alpha = rendering.render_volume_stratified(
        rf, aabb, cam, cam.uv_grid(), n_ray_steps=100
    )
    img = torch.cat((color, alpha), -1)

    import matplotlib.pyplot as plt

    plt.imshow(img.squeeze(0))
    plt.show()
