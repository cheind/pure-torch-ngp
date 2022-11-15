from typing import Literal, Optional
from collections import defaultdict
import torch

from . import radiance
from . import cameras
from . import sampling
from . import geometric
from . import filtering
from .harmonics import rsh_cart_3

MAPKEY = Literal["color", "depth", "alpha"]


class RadianceRenderer(torch.nn.Module):
    def __init__(
        self,
        aabb: torch.Tensor,
        filter: filtering.SpatialFilter = None,
    ) -> None:
        super().__init__()
        self.filter = filter or filtering.BoundsFilter()
        self.register_buffer("aabb", aabb)
        self.aabb: torch.Tensor

    def trace_uv(
        self,
        rf: radiance.RadianceField,
        cam: cameras.MultiViewCamera,
        uv: torch.Tensor,
        tsampler: sampling.RayStepSampler,
        which_maps: Optional[set[MAPKEY]] = None,
        booster: float = 1.0,
    ) -> dict[MAPKEY, Optional[torch.Tensor]]:
        # if ray_td is None:
        #     ray_td = torch.norm(self.aabb[1] - self.aabb[0]) / 1024
        if which_maps is None:
            which_maps = {"color", "alpha"}

        # Output alloc
        bshape = uv.shape[:-1]
        result = defaultdict(None)
        if "color" in which_maps:
            result["color"] = uv.new_zeros(bshape + (rf.n_color_dims,))
        if "alpha" in which_maps:
            result["alpha"] = uv.new_zeros(bshape + (1,))
        if "depth" in which_maps:
            result["depth"] = uv.new_zeros(bshape + (1,))

        rays = geometric.RayBundle.create_world_rays(cam, uv)
        rays = rays.intersect_aabb(self.aabb)
        active_mask = rays.active_mask()
        active_rays = rays.filter_by_mask(active_mask)

        if active_rays.d.numel() == 0:
            return result

        # Sample along rays
        ts = tsampler(active_rays)

        # Query radiance field at sample locations.
        ts_color, ts_density = self._query_radiance_field(
            rf,
            active_rays,
            ts,
        )

        # Compute integration weights
        ts_weights = radiance.integrate_timesteps(
            ts_density, ts, active_rays.dnorm, tfinal=active_rays.tfar + booster
        )

        # Compute result maps
        if "color" in which_maps:
            result["color"][active_mask] = radiance.color_map(ts_color, ts_weights)
        if "alpha" in which_maps:
            result["alpha"][active_mask] = radiance.alpha_map(ts_weights)
        if "depth" in which_maps:
            result["depth"][active_mask] = radiance.depth_map(ts, ts_weights)

        return result

    def trace_maps(
        self,
        rf: radiance.RadianceField,
        cam: cameras.MultiViewCamera,
        tsampler: sampling.RayStepSampler,
        which_maps: Optional[set[MAPKEY]] = None,
        n_samples_per_cam: int = None,
        booster: float = 1.0,
    ) -> dict[MAPKEY, Optional[torch.Tensor]]:
        if which_maps is None:
            which_maps = {"color", "alpha"}

        gen = sampling.generate_sequential_uv_samples(
            camera=cam, image=None, n_samples_per_cam=n_samples_per_cam
        )

        parts = []
        for uv, _ in gen:
            result = self.trace_uv(
                rf=rf,
                cam=cam,
                uv=uv,
                tsampler=tsampler,
                which_maps=which_maps,
                booster=booster,
            )
            parts.append(result)

        N = cam.n_views
        W, H = cam.size

        result = {
            k: torch.cat([p[k] for p in parts], 1).view(N, H, W, -1) for k in which_maps
        }
        return result

    def _query_radiance_field(
        self,
        rf: radiance.RadianceField,
        rays: geometric.RayBundle,
        ts: torch.Tensor,
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
