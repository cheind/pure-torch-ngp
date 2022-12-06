import torch
import numpy as np

from . import functional
from . import config
from . import modules


def spherical_poses(
    n_poses: int = 20,
    theta_range: tuple[float, float] = (0, 2 * np.pi),
    phi_range: tuple[float, float] = (70 / 180 * np.pi, 70 / 180 * np.pi),
    radius_range: tuple[float, float] = (6.0, 6.0),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    inclusive: bool = False,
) -> torch.Tensor:
    """Returns spherical camera poses from the given parameter ranges.

    Params
        n_poses: number of total poses
        theta_range: azimuth angle range in radians
        phi_range: elevation angle range in radians
        radius_range: radius range
        inclusive: Whether to be exclusive or inclusive on ranges
    """
    N = n_poses if inclusive else n_poses + 1
    poses = functional.spherical_pose(
        theta=torch.linspace(theta_range[0], theta_range[1], N),
        phi=torch.linspace(phi_range[0], phi_range[1], N),
        radius=torch.linspace(radius_range[0], radius_range[1], N),
        center=torch.tensor(center).unsqueeze(0).expand(N, 3),
    )
    return poses[:n_poses]


SphericalPosesConf = config.build_conf(spherical_poses)


def rasterize_field(
    rf: modules.RadianceField,
    resolution: tuple[int, int, int],
    batch_size: int = 2**16,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates the radiance field at rasterized grid locations.

    Note, we rasterize at voxel centers.

    Params:
        nerf: radiance field
        shape: desired resolution of rasterization in each dimension
        dev: computation device
        batch_size: max grid locations per batch

    Returns:
        color: color values with shape `shape + (C,)` and 'xy' indexing
        sigma: density values with shape `shape + (1,)` and 'xy' indexing
    """
    ijk = functional.make_grid(
        resolution,
        indexing="xy",
        device=device,
        dtype=dtype,
    )
    res = ijk.new_tensor(resolution[::-1])
    nxyz = (ijk + 0.5) * 2 / res - 1.0

    color_parts = []
    sigma_parts = []
    for batch in nxyz.split(batch_size, dim=0):
        rgb, d = rf(batch)
        color_parts.append(rgb)
        sigma_parts.append(d)
    C = color_parts[0].shape[-1]
    color = torch.cat(color_parts, 0).view(resolution + (C,))
    sigma = torch.cat(sigma_parts, 0).view(resolution + (1,))
    return color, sigma
