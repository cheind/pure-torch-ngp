import matplotlib.pyplot as plt
import numpy as np
import torch
from ngptorch import hashing


def main():
    T = 4096
    res_range = torch.arange(32)
    index_coords = torch.stack(
        torch.meshgrid([res_range] * 2, indexing="ij"), -1
    ).long()

    # uniform distribution of hash values desired
    indices = hashing.ravel_index(index_coords, (32, 32), T)
    plt.bar(*np.unique(indices, return_counts=True))
    plt.show()


if __name__ == "__main__":
    main()
