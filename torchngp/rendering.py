import torch

from . import radiance
from . import cameras
from . import sampling
from . import geometric
from . import filtering
from .harmonics import rsh_cart_3


class RadianceRenderer(torch.nn.Module):
    def __init__(
        self,
        aabb: torch.Tensor,
        filter: filtering.SpatialFilter = None,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.filter = filter or filtering.BoundsFilter()
        self.aabb: torch.Tensor

    def render_uv(
        self,
        rf: radiance.RadianceField,
        cam: cameras.MultiViewCamera,
        uv: torch.Tensor,
        ray_td: float = None,
        booster: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ray_td is None:
            ray_td = torch.norm(self.aabb[1] - self.aabb[0]) / 1024

        # Output (N,...,C), (N,...,1)
        batch_shape = uv.shape[:-1]
        out_color = uv.new_zeros(batch_shape + (rf.n_color_dims,))
        out_alpha = uv.new_zeros(batch_shape + (1,))

        rays = geometric.RayBundle.create_world_rays(cam, uv)
        rays = rays.intersect_aabb(self.aabb)
        active_mask = rays.active_mask()
        active_rays = rays.filter_by_mask(active_mask)

        if active_rays.d.numel() == 0:
            return out_color, out_alpha

        ts = sampling.sample_ray_step_stratified(
            active_rays.tnear, active_rays.tfar, n_samples=512
        )  # TODO: in sample consider that we are not having normalized dirs

        ts_color, ts_density = self._query_radiance_field(rf, active_rays, ts)

        # Integrate colors along rays
        ts_weights = radiance.integrate_timesteps(
            ts_density, ts, active_rays.dnorm, tfinal=active_rays.tfar + booster
        )
        ts_final_color = radiance.color_map(ts_color, ts_weights)
        ts_final_alpha = radiance.alpha_map(ts_weights)

        out_color[active_mask] = ts_final_color
        out_alpha[active_mask] = ts_final_alpha

        return out_color, out_alpha

    def _query_radiance_field(
        self, rf: radiance.RadianceField, rays: geometric.RayBundle, ts: torch.Tensor
    ):
        """Query the radiance field for the given points `xyz=o + d*ts`"""
        batch_shape = ts.shape[:-1]
        out_color = ts.new_zeros(batch_shape + (rf.n_color_dims,))
        out_density = ts.new_zeros(batch_shape + (rf.n_density_dims,))

        # Evaluate world points (T,N,...,3)
        xyz = rays(ts)

        ray_ynm = None
        with_harmonics = rf.n_color_cond_dims > 0
        if with_harmonics:
            dn = rays.d / rays.dnorm
            ray_ynm = rsh_cart_3(dn).unsqueeze(0).expand(ts.shape[0], -1, -1)

        # Convert to ndc (T,N,...,3)
        xyz_ndc = geometric.convert_world_to_box_normalized(xyz, self.aabb)

        # Filter (T,N,...)
        mask = self.filter.test(xyz_ndc)

        # (V,3)
        xyz_ndc_masked = xyz_ndc[mask]
        if with_harmonics:
            ray_ynm = ray_ynm[mask]

        # Predict
        color, density = rf(xyz_ndc_masked, color_cond=ray_ynm)

        out_color[mask] = color.to(out_color.dtype)
        out_density[mask] = density.to(density.dtype)
        return out_color, out_density

    def render_camera_views(
        self,
        rf: radiance.RadianceField,
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
            color, alpha = self.render_uv(rf, cam, uv, booster=booster)
            color_parts.append(color)
            alpha_parts.append(alpha)

        N = cam.n_views
        C = color_parts[0].shape[-1]
        W, H = cam.size
        color = torch.cat(color_parts, 1).view(N, H, W, C)
        alpha = torch.cat(alpha_parts, 1).view(N, H, W, 1)
        return color, alpha
