import logging
import json
import math
from typing import Union
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from hydra_zen import make_custom_builds_fn

from .filtering import BoundsFilter, OccupancyGridFilter
from .geometric import MultiViewCamera
from .radiance import NeRF
from .scenes import Scene
from .volumes import Volume


builds = make_custom_builds_fn(populate_full_signature=True)


def vecs3_to_tensor(v: list[tuple[float, float, float]]) -> torch.Tensor:
    return torch.tensor(v).float()


Vecs3Conf = builds(vecs3_to_tensor)

MultiViewCameraConf = builds(
    MultiViewCamera, rvec=Vecs3Conf([(0.0,) * 3]), tvec=Vecs3Conf([(0.0,) * 3])
)
NeRFConf = builds(NeRF)
OccupancyGridFilterConf = builds(OccupancyGridFilter)
BoundsFilterConf = builds(BoundsFilter)
VolumeConf = builds(
    Volume,
    aabb=Vecs3Conf([(-1.0,) * 3, (1.0,) * 3]),
    radiance_field=NeRFConf(),
)
SceneConf = builds(Scene, cameras=[], aabb=Vecs3Conf([(-1.0,) * 3, (1.0,) * 3]))
