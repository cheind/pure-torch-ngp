import math
import torch
import dataclasses
from typing import Protocol

from . import radiance
from . import cameras
from . import sampling
from . import geometric
from .harmonics import rsh_cart_3


def _strided_windows(n: int, batch: int, stride: int = 1):
    for i in range(0, n, stride):
        j = min(i + batch, n)
        if (j - i) > 2:  # last one not needed
            yield i, j


class SpatialFilter(Protocol):
    """Protocol for a spatial rendering filter.

    A spatial rendering accelerator takes spatial positions in NDC format
    and returns a mask of samples worthwile considering.
    """

    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        """Test given NDC locations.

        Params:
            xyz_ndc: (N,...,3) tensor of normalized [-1,1] coordinates

        Returns:
            mask: (N,...) boolean mask of the samples to be processed further
        """
        ...

    def update(self):
        """Update this accelerator."""
        ...


class BoundsFilter(SpatialFilter):
    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        mask = ((xyz_ndc > -1.0) & (xyz_ndc < 1.0)).all(-1)
        return mask

    def update(self):
        pass


@dataclasses.dataclass
class RayData:
    o: torch.Tensor
    d: torch.Tensor
    tnear: torch.Tensor
    tfar: torch.Tensor


class RadianceRenderer:
    def __init__(
        self,
        radiance_field: radiance.RadianceField,
        aabb: torch.Tensor,
        cam: cameras.MultiViewCamera,
        filter: SpatialFilter = None,
    ) -> None:
        self.radiance_field = radiance_field
        self.aabb = aabb
        self.cam = cam
        self.with_harmonics = radiance_field.n_color_cond_dims > 0
        self.filter = filter or BoundsFilter()

    def render_uv(
        self, uv: torch.Tensor, ray_td: float = None, ray_tbatch: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ray_td is None:
            ray_td = torch.norm(self.aabb[1] - self.aabb[0]) / 256

        batch_shape = uv.shape[:-1]

        # Output (N,...,C), (N,...,1)
        out_color = uv.new_zeros(batch_shape + (3,))
        out_log_transm = uv.new_zeros(batch_shape + (1,))

        with torch.no_grad():
            # Generate world rays (N,...,[1,3])
            rays = self._compute_rays(uv)

            # Ray-active mask (N,...)
            ray_lengths = rays.tfar - rays.tnear
            ray_mask = (ray_lengths > ray_td * 0.5).squeeze(-1)
            ray_term_th = math.log(1e-4)

            # Pre-generate all timesteps
            ray_ts = self._sample_ts(rays, ray_lengths=ray_lengths, ray_td=ray_td)

        for ti, tj in _strided_windows(ray_ts.shape[0], ray_tbatch, ray_tbatch - 1):

            o = rays.o[ray_mask]
            d = rays.d[ray_mask]
            ts = ray_ts[ti:tj, ray_mask]

            color, density = self._query_radiance_field(o, d, ts[:-1])

            # Integrate colors along rays
            part_color, part_log_transm = radiance.integrate_path(
                color,
                density,
                ts,
                prev_log_transmittance=out_log_transm[ray_mask].detach(),
            )

            out_color[ray_mask] += part_color[-1].to(out_color.dtype)
            out_log_transm[ray_mask] += part_log_transm[-2].to(out_log_transm.dtype)

            # Update ray active mask
            ray_mask = ray_mask & (out_log_transm.detach() > ray_term_th).squeeze(-1)
            ray_mask = ray_mask & (ray_ts[tj - 1] <= rays.tfar).squeeze(-1)

            if not ray_mask.any():
                break

        out_color = out_color
        out_alpha = 1 - out_log_transm.exp()

        return out_color, out_alpha

    def _query_radiance_field(self, o, d, ts):
        """Query the radiance field for the given points `xyz=o + d*ts`"""
        batch_shape = ts.shape[:-1]
        out_color = ts.new_zeros(batch_shape + (self.radiance_field.n_color_dims,))
        out_density = ts.new_zeros(batch_shape + (self.radiance_field.n_density_dims,))

        # Evaluate world points (T,N,...,3)
        xyz = geometric.evaluate_ray(o, d, ts)

        ray_ynm = None
        if self.with_harmonics:
            ray_ynm = rsh_cart_3(d).unsqueeze(0).expand(ts.shape[0], -1, -1)

        # Convert to ndc (T,N,...,3)
        xyz_ndc = geometric.convert_world_to_box_normalized(xyz, self.aabb)

        # Filter (T,N,...)
        mask = self.filter.test(xyz_ndc)

        # (V,3)
        xyz_ndc_masked = xyz_ndc[mask]
        if self.with_harmonics:
            ray_ynm = ray_ynm[mask]

        # Predict for filtered locations
        color, density = self.radiance_field(xyz_ndc_masked, color_cond=ray_ynm)

        out_color[mask] = color.to(out_color.dtype)
        out_density[mask] = density.to(density.dtype)
        return out_color, out_density

    def _compute_rays(self, uv: torch.Tensor) -> RayData:
        # Get world rays (N,...,3)
        ray_origin, ray_dir, ray_tnear, ray_tfar = geometric.world_ray_from_pixel(
            self.cam, uv, normalize_dirs=True
        )

        # Intersect rays with AABB
        ray_tnear, ray_tfar = geometric.intersect_ray_aabb(
            ray_origin, ray_dir, ray_tnear, ray_tfar, self.aabb
        )

        return RayData(o=ray_origin, d=ray_dir, tnear=ray_tnear, tfar=ray_tfar)

    def _sample_ts(
        self, rays: RayData, ray_lengths: torch.Tensor, ray_td: float
    ) -> torch.Tensor:
        max_length = max(ray_lengths.max().item(), 0.0)
        n_samples = int(max_length / ray_td)

        return sampling.sample_ray_fixed_step_stratified(
            rays.tnear, stepsize=ray_td, n_samples=n_samples
        )


def render_camera_views(
    cam: cameras.MultiViewCamera,
    radiance_field: radiance.RadianceField,
    aabb: torch.Tensor,
    n_ray_t_steps: int = 100,
    boost_tfar: float = 1.0,
    resample: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:

    gen = sampling.generate_sequential_uv_samples(
        camera=cam, image=None, n_samples_per_cam=None
    )

    rnd = RadianceRenderer(radiance_field, aabb, cam)

    color_parts = []
    alpha_parts = []
    for uv, _ in gen:
        color, alpha = rnd.render_uv(uv)
        color_parts.append(color)
        alpha_parts.append(alpha)

    N = cam.n_views
    C = color_parts[0].shape[-1]
    W, H = cam.size
    color = torch.cat(color_parts, 1).view(N, H, W, C)
    alpha = torch.cat(alpha_parts, 1).view(N, H, W, 1)
    return color, alpha
