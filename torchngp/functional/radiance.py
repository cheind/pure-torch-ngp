from typing import Union

import torch
import torch.nn.functional as F


def integrate_timesteps(
    sigma: torch.Tensor,
    ts: torch.Tensor,
    dnorm: torch.Tensor,
    tfinal: Union[float, torch.Tensor] = 1e7,
):
    """Computes the timestep weights for multiple rays.

    This implementation is a batched version of equations (1,3) in [1]. In particular
    it computes the weights for timestep `i` as follows
        weights(i)  = T(i)*alpha(i)
                    = exp(-log_T(i-1))*(1-(ts(i+1)-ts(i))*sigma(i))
                    = exp(-log_T(i-1)) - exp(-log_T(i))
    with
        log_T(i) = cumsum([0,(ts(1)-ts(0))*sigma(0),...,(ts(-1)-ts(-2))*sigma(-1))])

    Params:
        sigma: (T,N,...,1) volume density for each ray (N,...) and timestep (T,).
        ts: (T,N,...,1) ray timestep values.
        dnorm: (N,...,1) ray direction norms. Used for conversion of timestep units
            to real world units.
        tfinal: float or (N,...,1) tensor of one beyond last timestep values.
    Returns:
        weights: (T,N,...,1) weights for each timestep.

    References:
        [1] NeRF: Representing Scenes as
        Neural Radiance Fields for View Synthesis
        https://arxiv.org/pdf/2003.08934.pdf
    """

    eps = torch.finfo(ts.dtype).eps
    batch_shape = ts.shape[1:]

    # delta[i] is defined as the segment lengths between ts[i+1] and ts[i]
    # we add a tfinal value far beyond reach which boosts the last element in
    # integration.
    if not torch.is_tensor(tfinal):
        tfinal = ts.new_tensor(tfinal).expand(batch_shape)
    ts = torch.cat((ts, tfinal.unsqueeze(0)), 0)
    delta = ts[1:] - ts[:-1]  # (T+1,N,...,1) -> (T,N,...,1)

    # Account for un-normalized ray direction lengths.
    # TODO: add note about z-stepping and depthmaps
    delta = delta * dnorm.unsqueeze(0)

    # Eps is necessary because in testing sigma is often inf and if delta
    # is zero then 0.0*float('inf')=nan
    sigma_mul_delta = sigma * (delta + eps)  # (T,N,...,1)

    # We construct a full-cumsum which has the following form
    # full_cumsum([1,2,3]) = [0,1,3,6]
    sigma_delta = F.pad(
        sigma_mul_delta,
        (0,) * 2 * (sigma.dim() - 1) + (1, 0),
        mode="constant",
        value=0.0,
    )
    log_transmittance = -sigma_delta.cumsum(0)  # (T+1,N,...,1)
    transmittance = log_transmittance.exp()

    # T(i)*alpha(i) when multiplied equals the following
    weights = transmittance[:-1] - transmittance[1:]  # (T,N,...,1)

    return weights


def color_map(
    ts_color: torch.Tensor, ts_weights: torch.Tensor, per_timestep: bool = False
) -> torch.Tensor:
    """Computes the RGB color map.

    Params:
        ts_color: (T,N,...,C) color samples
        ts_weights: (T,N,...,1) integration weights
        per_timestep: When true, returns the colors
            as they accumulate over time.

    Returns:
        color_map: (N,...,C) final colors or (T,N,...,C) when per_timestep
            is enabled.
    """
    color = ts_color * ts_weights
    return color.cumsum(0) if per_timestep else color.sum(0)


def alpha_map(ts_weights: torch.Tensor, per_timestep: bool = False) -> torch.Tensor:
    """Computes the alpha map.

    Params:
        ts_weights: (T,N,...,1) integration weights
        per_timestep: When true, returns the alpha
            as its accumulate over time.

    Returns:
        alpha_map: (N,...,1) or (T,N,...,1) alpha values in range [0,1].
            High values indicate opaque regions, wheras low values indicate
            regions of large amounts of transmittance.
    """
    ts_acc = ts_weights.cumsum(0) if per_timestep else ts_weights.sum(0)
    return ts_acc


def depth_map(
    ts: torch.Tensor, ts_weights: torch.Tensor, per_timestep: bool = False
) -> torch.Tensor:
    """Estimates a depth map.

    For this method to work correctly, the ray directions are not supposed
    to be normalized. This ensures that ts are actually steps in z-direction
    of the camera for all rays.

    Params:
        ts: (T,N,...,1) ray timestep values
        ts_weights: (T,N,...,1) integration weights
        per_timestep: When true, returns the colors
            as they accumulate over time.

    Returns:
        depth_map: (N,...,1) or (T,N,...,1) depth map
    """
    depth = ts_weights * ts
    return depth.cumsum(0) if per_timestep else depth.sum(0)


__all__ = ["integrate_timesteps", "color_map", "depth_map", "alpha_map"]
