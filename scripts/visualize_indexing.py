import matplotlib.pyplot as plt
import numpy as np
import torch
from ngp_nerf import hashing


def main():
    T = 512
    res_range = torch.arange(128)
    index_coords = torch.stack(
        torch.meshgrid([res_range] * 2, indexing="ij"), -1
    ).long()

    # uniform distribution of hash values desired
    indices = hashing.xor_index_hashing(index_coords, T)
    plt.bar(*np.unique(indices, return_counts=True))
    plt.show()


if __name__ == "__main__":
    main()
