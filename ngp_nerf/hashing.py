import torch


def xor_index_hashing(x: torch.LongTensor, n_indices: int) -> torch.LongTensor:
    # See https://matthias-research.github.io/pages/publications/tetraederCollision.pdf
    N = x.dim() - 1
    Q = x.shape[-1]
    pi = torch.tensor([1, 2654435761, 805459861], dtype=x.dtype, device=x.device)[:Q]
    pi = pi.view(*([1] * N), Q)

    xpi_int = x * pi
    indices = xpi_int[..., 0]
    for d in range(1, Q):
        indices = torch.bitwise_xor(indices, xpi_int[..., d])
    return indices % n_indices


def ravel_index(
    x: torch.LongTensor, shape: tuple[int, ...], n_indices: int
) -> torch.LongTensor:
    Q = x.shape[-1]
    assert Q in [2, 3], "Only implemented for 2D and 3D"
    if Q == 2:
        indices = x[..., 0] + x[..., 1] * shape[0]
    elif Q == 3:
        indices = x[..., 0] + x[..., 1] * shape[0] + x[..., 2] * (shape[0] * shape[1])
    return indices % n_indices
