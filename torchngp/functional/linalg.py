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
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula and
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

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

    cosa = torch.cos(theta).view(batch_shape + (1, 1))
    sina = torch.sin(theta).view(batch_shape + (1, 1))
    R = cosa * torch.eye(3, dtype=dtype, device=dev).expand_as(S).contiguous()
    R += sina * S
    R += (1 - cosa) * (axis.unsqueeze(-1) @ axis.unsqueeze(-2))

    return R


def rotation_vector(R: torch.Tensor) -> torch.Tensor:
    """Returns the Rodrigues parametrization for the given rotation matrices.

    Params:
        R: (N,...,3,3) rotation matrices

    Returns:
        u: (N,...,3) unit length rotation axes
        theta (N,...) angles of rotation

    References:
     - See last page of https://courses.cs.duke.edu/fall13/compsci527/notes/rodrigues.pdf
     - Gohlke https://github.com/malcolmreynolds/transformations/blob/af825900e981ec232ee0fc4955e592cf3ffe3cd2/transformations/transformations.py
    """
    # Most robust approach is to look for eigenvector with eval=1 as the direction.
    # There is only one real eigenvector close to one.
    R_dtype = R.dtype
    R = R.to(torch.float64)
    bshape = R.shape[:-2]
    e, E = torch.linalg.eig(R.transpose(-2, -1))  # (N,...,3) and (N,...,3,3)
    th = (abs(torch.real(e) - 1) < 1e-5) & (abs(torch.imag(e)) < 1e-5)
    assert th.any(-1).all(), "No unit eigenvector corresponding to eigenvalue 1"
    i = torch.argmax(th.float(), -1)
    i = i.unsqueeze(-1).unsqueeze(-1).expand((-1,) * len(bshape) + (3, -1))
    u = torch.take_along_dim(torch.real(E), i, dim=-1).squeeze(-1)

    # trace(R) = 1 + 2*cos(theta)
    cosa = (torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    # theta only up to sign
    sina = torch.empty_like(cosa)
    mask = abs(u[..., 2]) > 1e-8
    sina[mask] = (
        R[mask][:, 1, 0] + (cosa[mask] - 1.0) * u[mask][:, 0] * u[mask][:, 1]
    ) / u[mask][:, 2]
    mask = abs(u[..., 1]) > 1e-8
    sina[mask] = (
        R[mask][:, 0, 2] + (cosa[mask] - 1.0) * u[mask][:, 0] * u[mask][:, 2]
    ) / u[mask][:, 1]
    mask = abs(u[..., 0]) > 1e-8
    sina[mask] = (
        R[mask][:, 2, 1] + (cosa[mask] - 1.0) * u[mask][:, 1] * u[mask][:, 2]
    ) / u[mask][:, 0]
    theta = torch.atan2(sina, cosa)

    return u.to(R_dtype), theta.to(R_dtype)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    axis, theta = rotation_vector(R)
    return axis * theta.unsqueeze(-1)


def so3_exp(r: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(r, dim=-1)
    axis = torch.zeros_like(r)
    axis[..., 0] = 1.0
    mask = theta > 1e-7
    axis[mask] = r[mask] / theta[mask][..., None]
    R = rotation_matrix(axis, theta)
    return R


__all__ = [
    "hom",
    "dehom",
    "rotation_matrix",
    "rotation_vector",
    "so3_log",
    "so3_exp",
]
