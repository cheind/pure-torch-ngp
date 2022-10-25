import torch
from torch.testing import assert_close
import matplotlib as mpl

from ngp_nerf.nerf2 import geo, radiance


class ColorGradientRadianceField(radiance.RadianceField):
    """A test radiance field with a color gradient in x-dir and a planar surface."""

    def __init__(
        self,
        aabb: torch.Tensor,
        surface_pos: float = 0.2,  # absolute x-pos
        density_scale: float = 1.0,
        cmap: str = "gray",
    ):
        self.aabb = aabb
        self.cmap = mpl.colormaps[cmap]
        self.surface_pos = (surface_pos - aabb[0, 0]) / (aabb[1, 0] - aabb[0, 0])
        self.density_scale = density_scale

    def __call__(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = xyz.dtype
        # For cmap we need [0..1] coords
        nxyz = geo.convert_world_to_box_normalized(xyz, self.aabb) * 0.5 + 0.5

        # (N,...,3)
        colors = torch.tensor(self.cmap(nxyz[..., 0].cpu().numpy()))[..., :3]
        colors = colors.to(dtype)

        # (N,...,1)
        # Density dependes on the
        density = nxyz[..., 0] - self.surface_pos

        mask = density < 0
        density[mask] = 0.0
        density[~mask] *= self.density_scale

        return colors, density


def test_radiance_integrate_path():
    o = torch.tensor([[0.0, 0.0, 0.0]])
    d = torch.tensor([[1.0, 0.0, 0.0]])

    # ts = rays.sample_rays_uniformly(tnear, tfar, 100)
    ts = torch.linspace(0, 1, 100)[None, :]
    xyz = o[:, None] + ts[..., None] * d[:, None]  # (B,T,3)

    rf = ColorGradientRadianceField(
        aabb=torch.Tensor([[0.0] * 3, [1.0] * 3]),
        surface_pos=0.2,
        density_scale=float("inf"),  # hard surface boundary
        cmap="gray",
    )

    color, density = rf(xyz)
    final_colors, transmittance, alpha = radiance.integrate_path(
        color, density, ts, torch.tensor([[1.0]])
    )
    assert_close(final_colors, torch.tensor([[0.2, 0.2, 0.2]]))

    # Test soft transit and move plane
    rf = ColorGradientRadianceField(
        aabb=torch.Tensor([[0.0] * 3, [1.0] * 3]),
        surface_pos=0.5,
        density_scale=1000.0,  # soft surface boundary
        cmap="gray",
    )
    color, density = rf(xyz)
    final_colors, transmittance, alpha = radiance.integrate_path(
        color, density, ts, torch.tensor([[1.0]])
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

    rgb, d = nerf(torch.randn(10, 5, 20, 3))
    assert d.shape == (10, 5, 20, 1)
    assert rgb.shape == (10, 5, 20, 3)
