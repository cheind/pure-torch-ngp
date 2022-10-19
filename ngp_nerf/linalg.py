import torch


def hom(x: torch.Tensor, v: float = 1.0):
    """Returns v as homogeneous vectors by inserting one more element into
    the last axis."""
    return torch.cat((x, x.new_full(x.shape[:-1] + (1,), v)), -1)


def dehom(x: torch.Tensor):
    """Makes homogeneous vectors inhomogenious by dividing by the
    last element in the last axis."""
    return x[..., :-1] / x[..., None, -1]
