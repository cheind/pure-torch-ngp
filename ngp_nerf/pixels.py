import torch


def generate_grid_coords(
    shape: tuple[int, ...], indexing: str = "xy"
) -> torch.LongTensor:

    ranges = [torch.arange(r) for r in shape]
    coords = torch.stack(torch.meshgrid(*ranges, indexing="ij"), -1)
    if indexing == "xy":
        coords = torch.index_select(coords, -1, torch.arange(len(shape)).flip(0))
    return coords


def normalize_coords(x: torch.Tensor, indexing: str = "xy") -> torch.Tensor:

    spatial = x.shape[:-1]
    if indexing == "xy":
        spatial = spatial[::-1]

    sizes = x.new_tensor(spatial).expand_as(x)

    # suitable for align_corners=False
    return (x + 0.5) * 2 / sizes - 1.0


def denormalize_coords(x: torch.Tensor, indexing: str = "xy") -> torch.Tensor:

    spatial = x.shape[:-1]
    if indexing == "xy":
        spatial = spatial[::-1]

    sizes = x.new_tensor(spatial).expand_as(x)

    # suitable for align_corners=False
    return (x + 1) * sizes * 0.5 - 0.5
