import torch
from torch.testing import assert_close
import matplotlib as mpl

from ngp_nerf.nerf2 import radiance


def test_radiance_integrate_path():
    o = torch.tensor([[0.0, 0.0, 0.0]])
    d = torch.tensor([[1.0, 0.0, 0.0]])

    # ts = rays.sample_rays_uniformly(tnear, tfar, 100)
    ts = torch.linspace(0, 1, 100)[None, :]
    xyz = o[:, None] + ts[..., None] * d[:, None]  # (B,T,3)

    # Estimate colors and density values at sample positions
    def sample_colors_density(xyz, plane_o, plane_n):
        color = torch.tensor(
            mpl.colormaps["gray"](xyz[..., 0].numpy())
        ).float()  # (B,T,4)
        density = (
            (
                (xyz - plane_o[None, None, :]).unsqueeze(-2)
                @ plane_n[None, None, :, None]
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        mask = density < 0
        density[mask] = 0.0
        return color, density, mask

    # Test hard transit in density
    plane_n = torch.tensor([1.0, 0.0, 0.0])
    plane_o = torch.tensor([0.2, 0.0, 0.0])

    color, density, mask = sample_colors_density(xyz, plane_o, plane_n)
    density[~mask] = torch.inf  # hard
    final_colors, transmittance, alpha = radiance.integrate_path(
        color[..., :3], density, ts, torch.tensor([[1.0]])
    )
    assert_close(final_colors, torch.tensor([[plane_o[0], plane_o[0], plane_o[0]]]))

    # Test soft transit and move plane
    plane_n = torch.tensor([1.0, 0.0, 0.0])
    plane_o = torch.tensor([0.5, 0.0, 0.0])
    color, density, mask = sample_colors_density(xyz, plane_o, plane_n)
    density[~mask] = density[~mask] * 1000  # softer
    final_colors, transmittance, alpha = radiance.integrate_path(
        color[..., :3], density, ts, torch.tensor([[1.0]])
    )
    assert ((final_colors > 0.5) & (final_colors < 0.6)).all()


@torch.no_grad()
def test_radiance_nerf_module():
    nerf = radiance.NeRF(
        3,
        n_hidden=16,
        n_encodings=2**8,
        n_levels=4,
        min_res=16,
        max_res=64,
        aabb=torch.Tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        is_hdr=False,
    )

    d, rgb = nerf(torch.randn(10, 5, 20, 3))
    assert d.shape == (10, 5, 20, 1)
    assert rgb.shape == (10, 5, 20, 3)