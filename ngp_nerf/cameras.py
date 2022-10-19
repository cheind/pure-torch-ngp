import torch

from .pixels import generate_grid_coords
from .linalg import hom


def image_points(K: torch.Tensor, shape: tuple[int, int]):
    """Returns 3D points at z=1 for each image pixel.

    Params:
        K: (3,3) intrinsic camera matrix
        shape: H,W of image

    Returns:
        points: (HW,3) points (x,y,z)
    """
    invK = torch.inverse(K)
    pixels = generate_grid_coords(shape, indexing="xy").float()
    return hom(pixels, 1.0) @ invK.T


# Alias: in camera frame image-points corresponds to unnormalized
# view directions.
view_dirs = image_points
