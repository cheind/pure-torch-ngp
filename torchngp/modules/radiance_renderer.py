from typing import Literal, Optional
from collections import defaultdict

import torch

from .. import functional
from .. import config
from . import protocols
from .ray_samplers import StratifiedRayStepSampler
from .ray_samplers import StratifiedRayStepSamplerConf
from .volume import Volume
from .camera import MultiViewCamera
from .ray_bundle import RayBundle

MAPKEY = Literal["color", "depth", "alpha"]


class RadianceRenderer(torch.nn.Module):
    def __init__(
        self,
        tsampler: protocols.RayStepSampler = None,
        ray_ext_factor: float = 10.0,
    ) -> None:
        super().__init__()
        self.ray_ext_factor = ray_ext_factor
        self.tsampler = tsampler or StratifiedRayStepSampler()

    def trace_uv(
        self,
        vol: Volume,
        cam: MultiViewCamera,
        uv: torch.Tensor,
        tsampler: Optional[protocols.RayStepSampler] = None,
        which_maps: Optional[set[MAPKEY]] = None,
    ) -> dict[MAPKEY, Optional[torch.Tensor]]:
        """Render various radiance properties for specific pixel coordinates

        Params:
            vol: volume holding radiance information
            cam: camera and pose information to be rendered
            uv: (N,...,2) uv coordinates for N views to render
            tsampler: optional ray step sampling strategy to be used
            which_maps: set of map names to be rendered

        Returns:
            maps: dictionary from map name to (N,...,C) tensor with C depending
                on the map type.
        """
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

        rays = RayBundle.make_world_rays(cam, uv)
        rays = rays.intersect_aabb(vol.aabb)
        active_mask = rays.active_mask()
        active_rays = rays.filter_by_mask(active_mask)

        if active_rays.d.numel() == 0:
            return result

        # Sample along rays
        ts = tsampler(active_rays, vol=vol)

        # Evaluate ray locations
        xyz = active_rays(ts)

        # Query radiance field at sample locations
        if "color" in which_maps:
            ynm = active_rays.encode_raydir()
            # ynm (N,...,16) -> (T,N,...,16)
            ynm = ynm.unsqueeze(0).expand(ts.shape[0], *ynm.shape)
            ts_density, ts_color = vol.sample(
                xyz,
                ynm=ynm,
                return_color=True,
            )
        else:
            ts_density, ts_color = vol.sample(
                xyz,
                ynm=None,
                return_color=False,
            )

        # Compute integration weights
        ts_weights = functional.integrate_timesteps(
            ts_density,
            ts,
            active_rays.dnorm,
            tfinal=active_rays.tfar * self.ray_ext_factor,
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
        vol: Volume,
        cam: MultiViewCamera,
        tsampler: Optional[protocols.RayStepSampler] = None,
        which_maps: Optional[set[MAPKEY]] = None,
        n_rays_parallel: int = 2**13,
    ) -> dict[MAPKEY, Optional[torch.Tensor]]:
        """Densly render various radiance properties.

        Params:
            vol: volume holding radiance information
            cam: camera and pose information to be rendered
            tsampler: optional ray step sampling strategy to be used
            which_maps: set of maps to be rendered
            n_rays_parallel: maximum number of parallel rays to process with C depending
                on the map type.

        Returns:
            maps: dictionary from map name to (N,H,W,C) tensor
        """
        if which_maps is None:
            which_maps = {"color", "alpha"}
        tsampler = tsampler or self.tsampler

        gen = functional.generate_sequential_uv_samples(
            uv_size=cam.size,
            n_views=cam.n_views,
            image=None,
            n_samples_per_view=n_rays_parallel // cam.n_views,
            dtype=cam.focal_length.dtype,
            device=cam.focal_length.device,
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

    def trace_rgba(
        self,
        vol: Volume,
        cam: MultiViewCamera,
        tsampler: Optional[protocols.RayStepSampler] = None,
        use_amp: bool = True,
        n_rays_parallel: int = 2**13,
    ) -> torch.Tensor:
        """Render RGBA images.

        This is a high-level routine best used in validation/testing. See
        `trace_maps` for more control.

        Params:
            vol: volume holding radiance information
            cam: camera and pose information to be rendered
            tsampler: optional ray step sampling strategy to be used
            use_amp: enable/disample amp
            n_rays_parallel: maximum number of parallel rays to process

        Returns:
            rgba: (N,4,H,W) batch of images in [0,1] range
        """
        with torch.cuda.amp.autocast(enabled=use_amp):
            maps = self.trace_maps(
                vol,
                cam,
                tsampler=tsampler,
                which_maps={"color", "alpha"},
                n_rays_parallel=n_rays_parallel,
            )
            pred_rgba = torch.cat((maps["color"], maps["alpha"]), -1).permute(
                0, 3, 1, 2
            )
        return pred_rgba


RadianceRendererConf = config.build_conf(
    RadianceRenderer,
    tsampler=StratifiedRayStepSamplerConf(),
)

__all__ = ["RadianceRenderer", "RadianceRendererConf"]
