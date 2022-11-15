import torch


def hom(x: torch.Tensor, v: float = 1.0):
    """Returns v as homogeneous vectors by inserting one more element into
    the last axis."""
    return torch.cat((x, x.new_full(x.shape[:-1] + (1,), v)), -1)


def dehom(x: torch.Tensor):
    """Makes homogeneous vectors inhomogenious by dividing by the
    last element in the last axis."""
    return x[..., :-1] / x[..., None, -1]


def rotation_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Returns rotation matrices from axis and angle.

    This uses Rodrigues formula which matches the exponential map in SO(3).
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Params:
        axis: (N,...,3) tensor of unit rotation axes
        theta: (N,...) tensor of rotation angles in radians

    Returns:
        R: (N,...,3,3) rotation matrices
    """
    dev = axis.device
    dtype = axis.dtype

    batch_shape = axis.shape[:-1]

    # Skew symmetric matrices
    S = axis.new_zeros(batch_shape + (3, 3))
    S[..., 0, 1] = -axis[..., 2]
    S[..., 0, 2] = axis[..., 1]
    S[..., 1, 0] = axis[..., 2]
    S[..., 1, 2] = -axis[..., 0]
    S[..., 2, 0] = -axis[..., 1]
    S[..., 2, 1] = axis[..., 0]

    R = torch.eye(3, dtype=dtype, device=dev).expand_as(S).contiguous()
    R += torch.sin(theta).view(batch_shape + (1, 1)) * S
    R += (1.0 - torch.cos(theta)).view(batch_shape + (1, 1)) * (S @ S)
    return R


__all__ = ["hom", "dehom", "rotation_matrix"]
