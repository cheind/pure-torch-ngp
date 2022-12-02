import torch
from hydra_zen import make_custom_builds_fn

build_conf = make_custom_builds_fn(populate_full_signature=True)


def vecs3_to_tensor(v: list[tuple[float, float, float]]) -> torch.Tensor:
    return torch.tensor(v).float()


Vecs3Conf = build_conf(vecs3_to_tensor)
