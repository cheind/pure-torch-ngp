import torch

from .geometric import MultiViewCamera
from .volumes import Volume


class Scene(torch.nn.Module):
    def __init__(self, cams: list[MultiViewCamera], volume: Volume) -> None:
        super().__init__()
        self.cameras = torch.nn.ModuleList(cams)
        self.volume = volume
