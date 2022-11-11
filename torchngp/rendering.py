import torch

from . import radiance
from . import cameras
from . import sampling
from . import geometric
from . import filtering
from .harmonics import rsh_cart_3


class RadianceRenderer:
    def __init__(
        self,
        radiance_field: radiance.RadianceField,
        aabb: torch.Tensor,
        filter: filtering.SpatialFilter = None,
    ) -> None:
        self.radiance_field = radiance_field
        self.aabb = aabb
        self.with_harmonics = radiance_field.n_color_cond_dims > 0
        self.filter = filter or filtering.BoundsFilter()

    def render_uv(
        self,
        cam: cameras.MultiViewCamera,
        uv: torch.Tensor,
        ray_td: float = None,
        booster: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ray_td is None:
            ray_td = torch.norm(self.aabb[1] - self.aabb[0]) / 1024

        batch_shape = uv.shape[:-1]

        rays = geometric.RayBundle.create_world_rays(cam, uv)
        rays = rays.intersect_aabb(self.aabb)
        active_mask = rays.active_mask()
        active_rays = rays.filter_by_mask(active_mask)

        ts = sampling.sample_ray_step_stratified(
            active_rays.tnear, active_rays.tfar, n_samples=128
        )  # TODO: in sample consider that we are not having normalized dirs

        sample_color, sample_density = self._query_radiance_field(active_rays, ts)

        # Integrate colors along rays
        int_color, int_logtransm = radiance.integrate_path(
            sample_color,
            sample_density,
            torch.cat((ts, booster * active_rays.tfar.unsqueeze(0)), 0),
        )

        # Output (N,...,C), (N,...,1)
        out_color = sample_color.new_zeros(batch_shape + (int_color.shape[-1],))
        out_alpha = sample_density.new_zeros(batch_shape + (1,))

        out_color[active_mask] = int_color[-1]
        out_alpha[active_mask] = 1 - int_logtransm[-2].exp()

        return out_color, out_alpha

    def _query_radiance_field(self, rays: geometric.RayBundle, ts: torch.Tensor):
        """Query the radiance field for the given points `xyz=o + d*ts`"""
        batch_shape = ts.shape[:-1]
        out_color = ts.new_zeros(batch_shape + (self.radiance_field.n_color_dims,))
        out_density = ts.new_zeros(batch_shape + (self.radiance_field.n_density_dims,))

        # Evaluate world points (T,N,...,3)
        xyz = rays(ts)

        ray_ynm = None
        if self.with_harmonics:
            ray_ynm = rsh_cart_3(rays.d).unsqueeze(0).expand(ts.shape[0], -1, -1)

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

    def _sample_ts(self, rays: geometric.RayBundle) -> torch.Tensor:
        # max_length = max(ray_lengths.max().item(), 0.0)
        # n_samples = int(max_length / ray_td)

        return sampling.sample_ray_step_stratified(rays.tnear, rays.tfar, n_samples=128)

    def render_camera_views(
        self,
        cam: cameras.MultiViewCamera,
        n_samples_per_cam: int = None,
        booster: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        gen = sampling.generate_sequential_uv_samples(
            camera=cam, image=None, n_samples_per_cam=n_samples_per_cam
        )

        color_parts = []
        alpha_parts = []
        for uv, _ in gen:
            color, alpha = self.render_uv(cam, uv, booster=booster)
            color_parts.append(color)
            alpha_parts.append(alpha)

        N = cam.n_views
        C = color_parts[0].shape[-1]
        W, H = cam.size
        color = torch.cat(color_parts, 1).view(N, H, W, C)
        alpha = torch.cat(alpha_parts, 1).view(N, H, W, 1)
        return color, alpha
