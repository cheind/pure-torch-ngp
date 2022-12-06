from typing import Protocol, Optional, Union, Literal
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import torch
import numpy as np

from . import functional
from . import config
from . import encoding


class RadianceField(Protocol):
    """Protocol of a spatial radiance field.

    A spatial radiance field takes normalized locations plus optional
    conditioning values and returns color and densities values for
    each location."""

    n_color_cond_dims: int
    n_color_dims: int
    n_density_dims: int

    def __call__(
        self, xyz: torch.Tensor, color_cond: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        """Return features from positions.

        Params:
            xyz_ndc: (N,...,3) normalized device locations [-1,1]

        Returns:
            f: (N,...,16) feature values for each sample point
        """
        ...

    def decode_density(self, f: torch.Tensor) -> torch.Tensor:
        """Return density estimates from encoded features.

        Params:
            f: (N,...,16) feature values for each sample point

        Returns:
            d: (N,...,1) color values in ranges [0..1]
        """
        ...

    def decode_color(
        self, f: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return color estimates from encoded features

        Params:
            f: (N,..., 16) feature values for each sample point
            cond: (N,...,n_color_cond) conditioning values for each
                sample point.

        Returns:
            colors: (N,...,n_colors) color values in range [0,+inf] (when hdr)
                [0,1] otherwise.
        """
        ...


class RayStepSampler(Protocol):
    """Protocol for sampling timestep values along rays.

    Note that ray directions are not normalized.
    """

    def __call__(self, rays: "RayBundle") -> torch.Tensor:
        """Sample timestep values

        Params:
            rays: (N,...) bundle of rays

        Returns:
            ts: (T,N,...,1) timestep samples for each ray.
        """
        ...


class StratifiedRayStepSampler(torch.nn.Module, RayStepSampler):
    def __init__(self, n_samples: int = 128) -> None:
        super().__init__()
        self.n_samples = n_samples

    def __call__(self, rays: "RayBundle") -> torch.Tensor:
        return functional.sample_ray_step_stratified(
            rays.tnear, rays.tfar, self.n_samples
        )


StratifiedRayStepSamplerConf = config.build_conf(StratifiedRayStepSampler)


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

    def update(self, rf: RadianceField):
        """Update this accelerator."""
        ...


class MultiViewCamera(torch.nn.Module):
    """Perspective camera with multiple poses.

    All camera models treat pixels as squares with pixel centers
    corresponding to integer pixel coordinates. That is, a pixel
    (u,v) extends from (u+0.5,v+0.5) to `(u+0.5,v+0.5)`.

    In addition we generalize to `N` poses by batching along
    the first dimension for attributes `R` and `T`.
    """

    def __init__(
        self,
        focal_length: tuple[float, float],
        principal_point: tuple[float, float],
        size: tuple[int, int],
        rvec: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        tvec: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        poses: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        image_paths: Optional[list[str]] = None,
        tnear: float = 0.0,
        tfar: float = 10.0,
    ) -> None:
        super().__init__()
        focal_length = torch.as_tensor(focal_length).view(2).float()
        principal_point = torch.as_tensor(principal_point).view(2).float()
        size = torch.as_tensor(size).view(2).int()
        tnear = torch.as_tensor(tnear).view(1).float()
        tfar = torch.as_tensor(tfar).view(1).float()

        rvec_given = rvec is not None
        tvec_given = tvec is not None
        pos_given = poses is not None

        if not (
            (not rvec_given and not tvec_given and pos_given)
            or (rvec_given and tvec_given and not pos_given)
        ):
            raise ValueError(
                "Either specify (rvec,tvec) or poses but not both or neither"
            )
        if rvec_given:
            # rvec, tvec specified
            if isinstance(rvec, list):
                rvec = torch.stack(rvec, 0)
            if isinstance(tvec, list):
                tvec = torch.stack(tvec, 0)
        else:
            # poses specified
            if isinstance(poses, list):
                poses = torch.stack(poses, 0)
            rvec = poses[:, :3, :3]
            rvec = functional.so3_log(rvec)
            tvec = poses[:, :3, 3]
        rvec = rvec.view(-1, 3).float()
        tvec = tvec.view(-1, 3, 1).float()

        self.register_buffer("focal_length", focal_length)
        self.register_buffer("principal_point", principal_point)
        self.register_buffer("size", size)
        self.register_buffer("tnear", tnear)
        self.register_buffer("tfar", tfar)
        self.register_buffer("rvec", rvec)
        self.register_buffer("tvec", tvec)
        if image_paths is not None:
            image_paths = np.array(
                image_paths, dtype=object
            )  # to support advanced indexing
        self.image_paths = image_paths

        self.focal_length: torch.Tensor
        self.principal_point: torch.Tensor
        self.size: torch.IntTensor
        self.tnear: torch.Tensor
        self.tfar: torch.Tensor
        self.rvec: torch.Tensor
        self.tvec: torch.Tensor

    def __getitem__(self, index) -> "MultiViewCamera":
        """Slice a subset of camera poses."""
        return MultiViewCamera(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            size=self.size,
            rvec=self.rvec[index],
            tvec=self.tvec[index],
            image_paths=(
                self.image_paths[index] if self.image_paths is not None else None
            ),
            tnear=self.tnear,
            tfar=self.tfar,
        )

    @property
    def K(self):
        """Return the 3x3 camera intrinsic matrix"""
        K = self.rvec.new_zeros((3, 3))
        K[0, 0] = self.focal_length[0]
        K[1, 1] = self.focal_length[1]
        K[0, 2] = self.principal_point[0]
        K[1, 2] = self.principal_point[1]
        K[2, 2] = 1
        return K

    @property
    def E(self) -> torch.Tensor:
        """Return the (N,4,4) extrinsic pose matrices."""
        N = self.n_views
        t = self.rvec.new_zeros((N, 4, 4))
        t[:, :3, :3] = functional.so3_exp(self.rvec)
        t[:, :3, 3:4] = self.tvec
        t[:, -1, -1] = 1
        return t

    @property
    def R(self) -> torch.Tensor:
        """Return the (N,3,3) rotation matrices."""
        return functional.so3_exp(self.rvec)

    @property
    def n_views(self):
        """Return the number of views."""
        return self.R.shape[0]

    def make_uv_grid(self):
        """Generates uv-pixel grid coordinates.

        Returns:
            uv: (N,H,W,2) tensor of grid coordinates using
                'xy' indexing.
        """
        N = self.n_views
        dev = self.focal_length.device
        dtype = self.focal_length.dtype
        uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(self.size[0], dtype=dtype, device=dev),
                    torch.arange(self.size[1], dtype=dtype, device=dev),
                    indexing="xy",
                ),
                -1,
            )
            .unsqueeze(0)
            .expand(N, -1, -1, -1)
        )
        return uv

    def load_images(self, base_path: Path = None) -> torch.Tensor:
        """Load images associated with this camera."""

        if self.image_paths is None or len(self.image_paths) == 0:
            imgs = self.rvec.new_empty((0, self.size[1], self.size[2], 4))
        else:
            if base_path is None:
                base_path = Path.cwd()
            paths = [base_path / p for p in self.image_paths]

            imgs = functional.load_image(
                paths, dtype=self.rvec.dtype, device=self.rvec.device
            )
        return imgs

    def extra_repr(self):
        out = ""
        out += f"focal_length={self.focal_length}, "
        out += f"size={self.size}, "
        out += f"n_poses={self.n_views}"
        return out


MultiViewCameraConf = config.build_conf(MultiViewCamera)


@dataclass
class RayBundle:
    """A collection of world rays."""

    o: torch.Tensor  # (N,...,3)
    d: torch.Tensor  # (N,...,3)
    tnear: torch.Tensor  # (N,...,1)
    tfar: torch.Tensor  # (N,...,1)
    dnorm: torch.Tensor  # (N,...,1)

    @staticmethod
    def make_world_rays(cam: MultiViewCamera, uv: torch.Tensor) -> "RayBundle":
        """Returns a new RayBundle from uv coordinates."""
        o, d, tnear, tfar = functional.make_world_rays(
            uv,
            cam.focal_length,
            cam.principal_point,
            cam.R,
            cam.tvec,
            tnear=cam.tnear,
            tfar=cam.tfar,
        )
        dnorm = torch.norm(d, p=2, dim=-1, keepdim=True)

        return RayBundle(o, d, tnear, tfar, dnorm)

    def __call__(self, ts: torch.Tensor) -> torch.Tensor:
        """Evaluate rays at given time steps.

        Params:
            ts: (N,...,1) or more general (T,...,N,...,1) time steps

        Returns:
            xyz: (N,...,3) / (T,...,N,...,3) locations
        """
        return functional.evaluate_ray(self.o, self.d, ts)

    def filter_by_mask(self, mask: torch.BoolTensor) -> "RayBundle":
        """Filter rays by boolean mask.

        Params:
            mask: (N,...) tensor

        Returns
            rays: filtered ray bundle with flattened dimensions
        """
        return RayBundle(
            o=self.o[mask],
            d=self.d[mask],
            tnear=self.tnear[mask],
            tfar=self.tfar[mask],
            dnorm=self.dnorm[mask],
        )

    def update_bounds(self, tnear: torch.Tensor, tfar: torch.Tensor) -> "RayBundle":
        """Updates the bounds of this ray bundle.

        Params:
            tnear: (N,...,1) near time step values
            tfar: (N,...,1) far time step values

        Returns:
            rays: updated ray bundle sharing tensors
        """
        tnear = torch.max(tnear, self.tnear)
        tfar = torch.min(tfar, self.tfar)
        return RayBundle(o=self.o, d=self.d, tnear=tnear, tfar=tfar, dnorm=self.dnorm)

    def intersect_aabb(self, box: torch.Tensor) -> "RayBundle":
        """Ray/box intersection.

        Params:
            box: (2,3) min/max corner of aabb

        Returns:
            rays: ray bundle with updated bounds

        Adapted from
        https://github.com/evanw/webgl-path-tracing/blob/master/webgl-path-tracing.js
        """

        tnear, tfar = functional.intersect_ray_aabb(
            self.o, self.d, self.tnear, self.tfar, box
        )
        return self.update_bounds(tnear, tfar)

    def active_mask(self) -> torch.BoolTensor:
        """Returns a mask of active rays.

        Active rays have a positive time step range.

        Returns:
            mask: (N,...) tensor of active rays
        """
        return (self.tnear < self.tfar).squeeze(-1)

    def encode_raydir(self):
        """Encodes the ray directions using spherical harmonics projection.

        Returns:
            ynm: (N,...,16) spherical harmonics of order 0,1,2,3.
        """
        dn = self.d / self.dnorm
        return functional.rsh_cart_3(dn)


class BoundsFilter(torch.nn.Module, SpatialFilter):
    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        mask = ((xyz_ndc >= -1.0) & (xyz_ndc <= 1.0)).all(-1)
        return mask

    def update(self, rf: RadianceField):
        del rf
        pass


BoundsFilterConf = config.build_conf(BoundsFilter)


class OccupancyGridFilter(BoundsFilter, torch.nn.Module):
    def __init__(
        self,
        res: int = 64,
        density_initial=0.02,
        density_threshold=0.01,
        stochastic_test: bool = True,
        update_decay: float = 0.7,
        update_noise_scale: float = None,
        update_selection_rate=0.25,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.res = res
        self.update_decay = update_decay
        self.density_initial = density_initial
        self.density_threshold = density_threshold
        self.update_selection_rate = update_selection_rate
        self.stochastic_test = stochastic_test
        if update_noise_scale is None:
            update_noise_scale = 0.9
        self.update_noise_scale = update_noise_scale
        self.register_buffer("grid", torch.full((res, res, res), density_initial))
        self.grid: torch.Tensor

    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        mask = super().test(xyz_ndc)

        ijk = (xyz_ndc + 1) * self.res * 0.5 - 0.5
        ijk = torch.round(ijk).clamp(0, self.res - 1).long()

        d = self.grid[ijk[..., 2], ijk[..., 1], ijk[..., 0]]
        d_mask = d > self.density_threshold
        if self.stochastic_test:
            d_stoch_mask = torch.bernoulli(1 - (-(d + 1e-4)).exp()).bool()
            d_mask |= d_stoch_mask

        return mask & d_mask

    @torch.no_grad()
    def update(self, rf: RadianceField):
        self.grid *= self.update_decay

        if self.update_selection_rate < 1.0:
            M = int(self.update_selection_rate * self.res**3)
            ijk = torch.randint(0, self.res, size=(M, 3), device=self.grid.device)
        else:
            ijk = functional.make_grid(
                (self.res, self.res, self.res),
                indexing="xy",
                device=self.grid.device,
                dtype=torch.long,
            ).view(-1, 3)

        noise = torch.rand_like(ijk, dtype=torch.float) - 0.5
        noise *= self.update_noise_scale
        xyz = ijk + noise
        xyz_ndc = (xyz + 0.5) * 2 / self.res - 1.0

        f = rf.encode(xyz_ndc)
        d = rf.decode_density(f).squeeze(-1)
        cur = self.grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
        new = torch.maximum(d, cur)
        self.grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = new


OccupancyGridFilterConf = config.build_conf(OccupancyGridFilter)


class NeRF(torch.nn.Module, RadianceField):
    """Neural radiance field module.

    Currently supports only spatial features and not view dependent ones.
    """

    def __init__(
        self,
        n_colors: int = 3,
        n_hidden: int = 64,
        n_encodings_log2: int = 16,
        n_levels: int = 16,
        n_color_cond: int = 16,
        min_res: int = 32,
        max_res: int = 512,
        max_res_dense: int = 256,
        is_hdr: bool = False,
    ) -> None:
        super().__init__()
        self.is_hdr = is_hdr
        self.n_color_cond_dims = n_color_cond
        self.n_color_dims = n_colors
        self.n_density_dims = 1
        self.pos_encoder = encoding.MultiLevelHybridHashEncoding(
            n_encodings=2**n_encodings_log2,
            n_input_dims=3,
            n_embed_dims=2,
            n_levels=n_levels,
            min_res=min_res,
            max_res=max_res,
            max_n_dense=max_res_dense**3,
        )
        n_enc_features = self.pos_encoder.n_levels * self.pos_encoder.n_embed_dims
        # 1-layer hidden density mlp
        self.density_mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_enc_features, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 16),  # 0 index is density in log-space
        )

        # 2-layer hidden color mlp
        self.color_mlp = torch.nn.Sequential(
            torch.nn.Linear(16 + n_color_cond, n_hidden),  # TODO: add aux-dims
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_colors),
        )

        torch.nn.init.constant_(self.density_mlp[-1].bias[:1], -1.0)

        # print(sum(param.numel() for param in self.parameters()))

    def encode(self, xyz_ndc: torch.Tensor) -> torch.Tensor:
        """Return features from positions.

        Params:
            xyz: (N,...,3) positions in world space

        Returns:
            f: (N,...,16) feature values for each sample point
        """
        batch_shape = xyz_ndc.shape[:-1]

        # Compute encoder features and pass them trough density mlp.
        xn_flat = xyz_ndc.view(-1, 3)
        h = self.pos_encoder(xn_flat)
        d = self.density_mlp(h)

        return d.view(batch_shape + (16,))

    def decode_density(self, f: torch.Tensor) -> torch.Tensor:
        """Return density estimates from encoded features.

        Params:
            f: (N,...,16) feature values for each sample point

        Returns:
            d: (N,...,1) color values in ranges [0..1]
        """

        # We consider the first feature dimension to be the
        # log-density estimate
        # print(f[..., 0].mean())
        return f[..., 0 : self.n_density_dims].exp()

    def decode_color(
        self, f: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return color estimates from encoded features

        Params:
            f: (N,..., 16) feature values for each sample point
            cond: (N,...,n_color_cond) conditioning values for each
                sample point.

        Returns:
            colors: (N,...,n_colors) color values in range [0,+inf] (when hdr)
                [0,1] otherwise.

        """
        batch_shape = f.shape[:-1]

        # Concat with condition
        if cond is not None:
            f = torch.cat((f, cond), -1)

        # Raw color prediction
        c = self.color_mlp(f).view(batch_shape + (self.n_color_dims,))

        # Reparametrize
        color = c.exp() if self.is_hdr else torch.sigmoid(c)

        return color

    def forward(self, xyz_ndc, color_cond: Optional[torch.Tensor] = None):
        """Predict density and color values for sample positions.

        Params:
            xyz_ndc: (N,...,3) sampling positions in normalized device coords
                [-1,1]
            color_cond: (N,...,n_cond_color_dims) optional color
                condititioning values

        Returns:
            color: (N,...,n_colors) predicted color values [0,+inf] (when hdr)
                [0,1] otherwise
            sigma: (N,...,1) prediced density values [0,+inf]
        """
        f = self.encode(xyz_ndc)
        sigma = self.decode_density(f)
        color = self.decode_color(f, cond=color_cond)
        return color, sigma


NeRFConf = config.build_conf(NeRF)


class Volume(torch.nn.Module):
    """Represents a physical volume space."""

    def __init__(
        self,
        aabb: torch.Tensor,
        radiance_field: RadianceField,
        spatial_filter: Optional[SpatialFilter] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb.float())
        self.aabb: torch.Tensor
        self.radiance_field = radiance_field
        self.spatial_filter = spatial_filter or BoundsFilter()

    def sample(
        self,
        xyz: torch.Tensor,
        ynm: Optional[torch.Tensor] = None,
        return_color: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample volume at ray locations

        This is a convenience method that takes world rays and samples
        the underlying radiance field taking spatial acceleration structures
        into account.

        Params:
            xyz: (N,...,3) world points
            ynm: (N,...,16) view direction encodings. Required only when
                return_color is True.
            return_color: When set, returns color samples in addition
                to density values.

        Returns:
            density: (N,...,1) density samples
            color: (N,...,C) color samples when ynm is specified and
                return_color is set. None otherwise.
        """
        assert (
            not return_color or ynm is not None
        ), "Color sampling requires viewdir encoding."

        batch_shape = xyz.shape[:-1]
        out_density = xyz.new_zeros(batch_shape + (self.radiance_field.n_density_dims,))
        if return_color:
            out_color = xyz.new_zeros(batch_shape + (self.radiance_field.n_color_dims,))

        # Convert to NDC (T,N,...,3)
        xyz_ndc = functional.convert_world_to_box_normalized(xyz, self.aabb)

        # Invoke spatial filter (T,N,...)
        mask = self.spatial_filter.test(xyz_ndc)

        # Compute features for active elements
        f = self.radiance_field.encode(xyz_ndc[mask])

        # Compute densities
        density = self.radiance_field.decode_density(f)
        out_density[mask] = density.to(density.dtype)

        # Compute optional colors
        if return_color:
            color = self.radiance_field.decode_color(f, cond=ynm[mask])
            out_color[mask] = color.to(out_color.dtype)
            return out_density, out_color
        else:
            return out_density, None


VolumeConf = config.build_conf(
    Volume,
    aabb=config.Vecs3Conf([(-1.0,) * 3, (1.0,) * 3]),
    radiance_field=NeRFConf(),
    spatial_filter=OccupancyGridFilterConf(),
)

MAPKEY = Literal["color", "depth", "alpha"]


class RadianceRenderer(torch.nn.Module):
    def __init__(
        self,
        tsampler: RayStepSampler = None,
        ray_ext_factor: float = 10.0,
    ) -> None:
        super().__init__()
        self.ray_ext_factor = ray_ext_factor
        self.tsampler = tsampler or StratifiedRayStepSampler(256)

    def trace_uv(
        self,
        vol: Volume,
        cam: MultiViewCamera,
        uv: torch.Tensor,
        tsampler: Optional[RayStepSampler] = None,
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
        ts = tsampler(active_rays)

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
        tsampler: Optional[RayStepSampler] = None,
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
        tsampler: Optional[RayStepSampler] = None,
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
    tsampler=StratifiedRayStepSamplerConf(256),
)
