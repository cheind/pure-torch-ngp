import torch

from .geometric import MultiViewCamera


class Scene(torch.nn.Module):
    def __init__(
        self,
        cams: list[MultiViewCamera],
        aabb: torch.Tensor,
    ) -> None:
        super().__init__()
        self.cameras = torch.nn.ModuleList(cams)
        self.register_buffer("aabb", aabb)
        self.aabb: torch.Tensor
