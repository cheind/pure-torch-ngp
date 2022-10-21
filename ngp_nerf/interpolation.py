import torch


def linear_interpolate_info(
    x: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.LongStorage, torch.Tensor, torch.BoolTensor]:
    """Computes bilinear/trilinear interpolation parameter

    Params:
        x: (B,C) points with C=2 or C=3
        shape: (H,W) or (D,H,W) of interpolation grid size

    Returns:
        c: (B,4) or (B,8) corner indices for each x
        w: (B,4) or (B,8) weights per corner (sum to one)
        m: (B) mask of valid points x.
    """
    B, C = x.shape
    if C == 2:
        c, w, m = _bilinear_interpolate(x, shape)
    elif C == 3:
        c, w, m = _trilinear_interpolate(x, shape)
    else:
        raise NotImplementedError
    return c, w, m


def _bilinear_interpolate(
    x: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.LongStorage, torch.Tensor, torch.BoolTensor]:
    o = x.new_tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
    xl = x.floor()
    xf = x - xl

    # Compute corners
    c = (xl.long().unsqueeze(1) + o[None, ...]).contiguous()

    # Compute mask
    m = ((c >= 0) & (c < c.new_tensor(shape).unsqueeze(0))).all(-1)  # B,4

    # Compute weights
    one_min = 1.0 - xf
    w11 = one_min[:,0] * one_min[:,1]
    w12 = one_min[:,0] * xf[:, 1]
    w21 = xf[:, 0] * one_min[:,1]
    w22 = xf[:, 0] * xf[:, 1]
    w = torch.stack((w11, w12, w21, w22), 1)  # B,4

    return c, w, m


def _trilinear_interpolate(
    x: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.LongStorage, torch.Tensor, torch.BoolTensor]:
    o = x.new_tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.long,
    )

    xl = x.floor()
    xf = x - xl

    # Compute corners
    c = (xl.long().unsqueeze(1) + o[None, ...]).contiguous()

    # Compute mask
    m = ((c >= 0) & (c < c.new_tensor(shape).unsqueeze(0))).all(-1)  # B,8

    # Compute weights
    one_min = 1 - xf
    w000 = one_min[:, 0] * one_min[:, 1] * one_min[:, 2]
    w001 = one_min[:, 0] * one_min[:, 1] * xf[:, 2]
    w010 = one_min[:, 0] * xf[:, 1] * one_min[:, 2]
    w011 = one_min[:, 0] * xf[:, 1] * xf[:, 2]
    w100 = xf[:, 0] * one_min[:, 1] * one_min[:, 2]
    w101 = xf[:, 0] * one_min[:, 1] * xf[:, 2]
    w110 = xf[:, 0] * xf[:, 1] * one_min[:, 2]
    w111 = xf[:, 0] * xf[:, 1] * xf[:, 2]

    w = torch.stack((w000, w001, w010, w011, w100, w101, w110, w111), 1)  # B,8

    return c, w, m
