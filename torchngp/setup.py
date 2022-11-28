import torch

from . import geometric
from . import volumes
from . import rendering
from . import config


class Setup(torch.nn.Module):
    def __init__(
        self,
        volume: volumes.Volume,
        cameras: dict[geometric.MultiViewCamera],
        renderer: dict[rendering.RadianceRenderer],
    ) -> None:
        super().__init__()
        self.cameras = torch.nn.ModuleDict(cameras)
        self.volume = volume
        self.renderer = torch.nn.ModuleDict(renderer)


SetupConf = config.build_conf(Setup)
