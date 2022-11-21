from typing import Literal, Optional
from collections import defaultdict
import torch

from . import radiance
from . import sampling
from . import geometric
from . import filtering
from . import volumes
from . import functional

MAPKEY = Literal["color", "depth", "alpha"]


class RadianceRenderer(torch.nn.Module):
    def __init__(
        self,
        tsampler: sampling.RayStepSampler = None,
        ray_extension: float = 2.0,
    ) -> None:
        super().__init__()
        self.ray_extension = ray_extension
        self.tsampler = tsampler or sampling.StratifiedRayStepSampler(256)

    def trace_uv(
        self,
        vol: volumes.Volume,
        cam: geometric.MultiViewCamera,
        uv: torch.Tensor,
        tsampler: Optional[sampling.RayStepSampler] = None,
        which_maps: Optional[set[MAPKEY]] = None,
    ) -> dict[MAPKEY, Optional[torch.Tensor]]:
        # if ray_td is None:
        #     ray_td = torch.norm(self.aabb[1] - self.aabb[0]) / 1024
        if which_maps is None:
            which_maps = {"color", "alpha"}
        tsampler = tsampler or self.tsampler

        # Output alloc
        bshape = uv.shape[:-1]
        result = defaultdict(None)
        if "color" in which_maps:
            result["color"] = uv.new_zeros(bshape + (vol.radiance_field.n_color_dims,))
        if "alpha" in which_maps:
            result["alpha"] = uv.new_zeros(bshape + (1,))
        if "depth" in which_maps:
            result["depth"] = uv.new_zeros(bshape + (1,))

        rays = geometric.RayBundle.make_world_rays(cam, uv)
        rays = rays.intersect_aabb(vol.aabb)
        active_mask = rays.active_mask()
        active_rays = rays.filter_by_mask(active_mask)

        if active_rays.d.numel() == 0:
            return result

        # Sample along rays
        ts = tsampler(active_rays)

        # Query radiance field at sample locations.
        ts_color, ts_density = self._query_radiance_field(
            vol,
            active_rays,
            ts,
        )

        # Compute integration weights
        ts_weights = functional.integrate_timesteps(
            ts_density,
            ts,
            active_rays.dnorm,
            tfinal=active_rays.tfar + self.ray_extension,
        )

        # Compute result maps
        if "color" in which_maps:
            result["color"][active_mask] = functional.color_map(ts_color, ts_weights)
        if "alpha" in which_maps:
            result["alpha"][active_mask] = functional.alpha_map(ts_weights)
        if "depth" in which_maps:
            result["depth"][active_mask] = functional.depth_map(ts, ts_weights)

        return result

    def trace_maps(
        self,
        vol: volumes.Volume,
        cam: geometric.MultiViewCamera,
        tsampler: Optional[sampling.RayStepSampler],
        which_maps: Optional[set[MAPKEY]] = None,
        n_samples_per_cam: Optional[int] = None,
    ) -> dict[MAPKEY, Optional[torch.Tensor]]:
        if which_maps is None:
            which_maps = {"color", "alpha"}
        tsampler = tsampler or self.tsampler

        gen = sampling.generate_sequential_uv_samples(
            camera=cam, image=None, n_samples_per_cam=n_samples_per_cam
        )

        parts = []
        for uv, _ in gen:
            result = self.trace_uv(
                vol=vol,
                cam=cam,
                uv=uv,
                tsampler=tsampler,
                which_maps=which_maps,
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
        vol: volumes.Volume,
        rays: geometric.RayBundle,
        ts: torch.Tensor,
    ):
        """Query the radiance field for the given points `xyz=o + d*ts`"""
        batch_shape = ts.shape[:-1]
        out_color = ts.new_zeros(batch_shape + (vol.radiance_field.n_color_dims,))
        out_density = ts.new_zeros(batch_shape + (vol.radiance_field.n_density_dims,))

        # Evaluate world points (T,N,...,3)
        xyz = rays(ts)

        ray_ynm = None
        with_harmonics = vol.radiance_field.n_color_cond_dims > 0
        if with_harmonics:
            dn = rays.d / rays.dnorm
            ray_ynm = functional.rsh_cart_3(dn).unsqueeze(0).expand(ts.shape[0], -1, -1)

        # Convert to ndc (T,N,...,3)
        xyz_ndc = vol.to_ndc(xyz)

        # Filter (T,N,...)
        mask = vol.spatial_filter.test(xyz_ndc)

        # (V,3)
        xyz_ndc_masked = xyz_ndc[mask]
        if with_harmonics:
            ray_ynm = ray_ynm[mask]

        # Predict
        color, density = vol.radiance_field(xyz_ndc_masked, color_cond=ray_ynm)

        out_color[mask] = color.to(out_color.dtype)
        out_density[mask] = density.to(density.dtype)
        return out_color, out_density
