import math
import torch

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


def render_radiance_field(
    radiance_field: radiance.RadianceField,
    aabb: torch.Tensor,
    cam: cameras.MultiViewCamera,
    uv: torch.Tensor,
    n_ray_t_steps: int = 100,
    boost_tfar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:

    batch_shape = uv.shape[:-1]
    with_harmonics = radiance_field.n_color_cond > 0

    # Get world rays corresponding to pixels (N,...,3)
    ray_origin, ray_dir, ray_tnear, ray_tfar = geometric.world_ray_from_pixel(
        cam, uv, normalize_dirs=with_harmonics
    )

    # Intersect rays with AABB
    ray_tnear, ray_tfar = geometric.intersect_ray_aabb(
        ray_origin, ray_dir, ray_tnear, ray_tfar, aabb
    )

    # Determine set of rays that actually intersect (N,...)
    ray_hit = (ray_tnear < ray_tfar).squeeze(-1)
    ray_origin = ray_origin[ray_hit]
    ray_dir = ray_dir[ray_hit]
    ray_tnear = ray_tnear[ray_hit]
    ray_tfar = ray_tfar[ray_hit]

    # Sample ray steps (T,V,1)
    ray_ts = sampling.sample_ray_step_stratified(
        ray_tnear, ray_tfar, n_samples=n_ray_t_steps
    )

    ray_ynm = None
    if with_harmonics:
        ray_ynm = (
            rsh_cart_3(ray_dir)
            .unsqueeze(0)
            .expand(ray_ts.shape[0], -1, -1)
            .contiguous()
        )

    # Evaluate world points (T,V,3)
    xyz = geometric.evaluate_ray(ray_origin, ray_dir, ray_ts)

    # Predict radiance properties
    color, sigma = radiance_field(xyz, color_cond=ray_ynm)

    # Integrate colors along rays
    integ_color, integ_log_transm = radiance.integrate_path(
        color, sigma, torch.cat((ray_ts, boost_tfar * ray_tfar.unsqueeze(0)), 0)
    )

    final_alpha = 1.0 - integ_log_transm[-2].exp()
    final_color = integ_color[-1]

    out_color = color.new_zeros(batch_shape + color.shape[-1:])
    out_alpha = sigma.new_zeros(batch_shape + (1,))

    out_color[ray_hit] = final_color.to(out_color.dtype)
    out_alpha[ray_hit] = final_alpha.to(out_alpha.dtype)

    return out_color, out_alpha


def render_radiance_field_time_batches(
    radiance_field: radiance.RadianceField,
    aabb: torch.Tensor,
    cam: cameras.MultiViewCamera,
    uv: torch.Tensor,
    n_ray_t_steps: int = 100,
    boost_tfar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:

    batch_shape = uv.shape[:-1]

    # Output (N,...,C), (N,...,1)
    integ_color = uv.new_zeros(batch_shape + (3,))
    integ_log_transm = uv.new_zeros(batch_shape + (1,))

    # Get world rays (N,...,3)
    ray_origin, ray_dir, ray_tnear, ray_tfar = geometric.world_ray_from_pixel(
        cam, uv, normalize_dirs=False
    )

    # Intersect rays with AABB
    ray_tnear, ray_tfar = geometric.intersect_ray_aabb(
        ray_origin, ray_dir, ray_tnear, ray_tfar, aabb
    )

    # Sample ray steps (T,N,...,1) and padded (T+1,N,...,1)
    ray_ts = sampling.sample_ray_step_stratified(
        ray_tnear, ray_tfar, n_samples=n_ray_t_steps
    )
    ray_ts = torch.cat((ray_ts, boost_tfar * ray_tfar.unsqueeze(0)), 0)

    # Ray-active mask (N,...)
    ray_mask = (ray_tnear < ray_tfar).squeeze(-1)
    log_transm_th = math.log(1e-5)

    for ti, tj in _strided_windows(ray_ts.shape[0], 20, 20 - 1):

        o = ray_origin[ray_mask]
        d = ray_dir[ray_mask]
        ts = ray_ts[ti:tj, ray_mask]

        # Evaluate world points (T,V,3)
        xyz = geometric.evaluate_ray(o, d, ts[:-1])

        # Predict radiance properties
        sample_color, sample_sigma = radiance_field(xyz)

        # Integrate colors along rays
        part_color, part_log_transm = radiance.integrate_path(
            sample_color,
            sample_sigma,
            ts,
            prev_log_transmittance=integ_log_transm[ray_mask],
        )

        integ_color[ray_mask] += part_color[-1].to(integ_color.dtype)
        integ_log_transm[ray_mask] += part_log_transm[-2].to(integ_log_transm.dtype)

        # Update ray active mask
        ray_mask = ray_mask & (integ_log_transm > log_transm_th).detach().squeeze(-1)

    out_color = integ_color
    out_alpha = 1 - integ_log_transm.exp()

    return out_color, out_alpha


def render_camera_views(
    cam: cameras.MultiViewCamera,
    radiance_field: radiance.RadianceField,
    aabb: torch.Tensor,
    n_ray_step_samples: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:

    gen = sampling.generate_sequential_uv_samples(
        camera=cam, image=None, n_samples_per_cam=None
    )

    color_parts = []
    alpha_parts = []
    for uv, _ in gen:
        color, alpha = render_radiance_field(
            radiance_field, aabb, cam, uv, n_ray_t_steps=n_ray_step_samples
        )
        color_parts.append(color)
        alpha_parts.append(alpha)

    N = cam.n_views
    C = color_parts[0].shape[-1]
    W, H = cam.size
    color = torch.cat(color_parts, 1).view(N, H, W, C)
    alpha = torch.cat(alpha_parts, 1).view(N, H, W, 1)
    return color, alpha
