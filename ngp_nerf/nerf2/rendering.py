import torch

from . import radiance
from . import cameras
from . import sampling
from . import geo


def render_volume_stratified(
    radiance_field: radiance.RadianceField,
    aabb: torch.Tensor,
    cam: cameras.BaseCamera,
    uv: torch.Tensor,
    n_ray_steps: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:

    batch_shape = uv.shape[:-1]

    # Get world rays corresponding to pixels (N,...,3)
    ray_origin, ray_dir, ray_tnear, ray_tfar = geo.world_ray_from_pixel(
        cam, uv, normalize_dirs=False
    )

    # Intersect rays with AABB
    ray_tnear, ray_tfar = geo.intersect_ray_aabb(
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
        ray_tnear, ray_tfar, n_bins=n_ray_steps
    )

    # Evaluate world points (T,V,3)
    xyz = geo.evaluate_ray(ray_origin, ray_dir, ray_ts)

    # Predict radiance properties
    color, sigma = radiance_field(xyz)

    # Integrate color (T,N,...,C) -> (N,...,C), others are per-sample
    final_color, sample_transm, sample_alpha = radiance.integrate_path(
        color, sigma, ray_ts, ray_tfar
    )

    # TODO: the following is not quite correct, should be 1.0 - T(i)*alpha(i)
    final_alpha = 1.0 - sample_transm[-1]

    out_color = color.new_zeros(batch_shape + color.shape[-1:])
    out_alpha = sigma.new_zeros(batch_shape + (1,))

    out_color[ray_hit] = final_color.to(out_color.dtype)
    out_alpha[ray_hit] = final_alpha.to(out_alpha.dtype)

    return out_color, out_alpha


def render_camera(
    radiance_field: radiance.RadianceField,
    aabb: torch.Tensor,
    cam: cameras.BaseCamera,
    n_ray_step_samples: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:

    gen = sampling.generate_sequential_uv_samples(
        camera=cam, image=None, n_samples_per_cam=None
    )

    color_parts = []
    alpha_parts = []
    for uv, _ in gen:
        color, alpha = render_volume_stratified(
            radiance_field, aabb, cam, uv, n_ray_steps=n_ray_step_samples
        )
        color_parts.append(color)
        alpha_parts.append(alpha)

    N = cam.size.shape[0]
    C = color_parts[0].shape[-1]
    H, W = cam.size[0, 1], cam.size[0, 0]
    color = torch.cat(color_parts, 1).view(N, H, W, C)
    alpha = torch.cat(alpha_parts, 1).view(N, H, W, 1)
    return color, alpha
