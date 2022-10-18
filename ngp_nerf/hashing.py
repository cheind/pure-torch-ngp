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
