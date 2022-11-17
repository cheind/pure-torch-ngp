import torch
from typing import Union
import hydra
from hydra.core.config_store import ConfigStore

from hydra_zen import instantiate, make_config, make_custom_builds_fn, to_yaml


builds = make_custom_builds_fn(populate_full_signature=True)

Float3 = tuple[float, float, float]

# Floats3x3 = Union[
#     tuple[float, float, float, float, float, float, float, float, float],
#     torch.FloatTensor,
# ]
# Floats3 = Union[tuple[float, float, float], torch.FloatTensor]
# Floats2 = Union[tuple[float, float], torch.FloatTensor]


# class Camera(torch.nn.Module):
#     def __init__(self, focal_length: Floats2, rots: list[Floats3x3], ts: list[Floats3]):
#         super().__init__()
#         self.focal_length = torch.as_tensor(focal_length, dtype=torch.float32)
#         self.R = torch.stack(rots)
#         self.ts = torch.stack(ts)


# class World(torch.nn.Module):
#     def __init__(self, cams: list[Camera]):
#         super().__init__()
#         self.cams = torch.nn.ModuleList(cams)


# CameraConf = builds(Camera)

# WorldConf = builds(World)
# worldcfg = WorldConf(
#     cams=[
#         CameraConf(
#             focal_length=(1.0, 2.0),
#             rots=[torch.eye(3).tolist()],
#             ts=[torch.ones(3).tolist()],
#         )
#     ]
# )
# print(to_yaml(worldcfg))

# world = instantiate(worldcfg)


# class Camera(torch.nn.Module):
#     def __init__(self, focal_length: float, size: tuple[int, int]) -> None:
#         super().__init__()
#         self.register_buffer("focal_length", torch.tensor([focal_length, focal_length]))
#         self.register_buffer("size", torch.as_tensor(size))

#     def extra_repr(self) -> str:
#         out = ""
#         out += f"focal_length={self.focal_length}, "
#         out += f"size={self.size}"
#         return out


# class RadianceField(torch.nn.Module):
#     def __init__(self, res: int = 64) -> None:
#         super().__init__()
#         self.res = res


# class SpatialFilter(torch.nn.Module):
#     def __init__(self, res: int = 32) -> None:
#         super().__init__()
#         self.res = res


# class BoxTransform(torch.nn.Module):
#     def __init__(
#         self, minc: tuple[float, float, float], maxc: tuple[float, float, float]
#     ) -> None:
#         super().__init__()
#         self.register_buffer("aabb", torch.as_tensor([minc, maxc]))


# class Volume(torch.nn.Module):
#     def __init__(
#         self,
#         rf: RadianceField,
#         filter: SpatialFilter,
#         boxt: BoxTransform,
#     ) -> None:
#         super().__init__()

#         self.rf = rf
#         self.filter = filter
#         self.box_transform = boxt


# class Scene(torch.nn.Module):
#     def __init__(self, cams: dict[str, Camera], volume: Volume) -> None:
#         super().__init__()
#         self.cameras = torch.nn.ModuleDict(cams)
#         self.volume = volume


# CameraConf = builds(Camera, populate_full_signature=True)
# RadianceFieldConf = builds(RadianceField, populate_full_signature=True)
# SpatialFilterConf = builds(SpatialFilter, populate_full_signature=True)
# BoxTransformConf = builds(BoxTransform, populate_full_signature=True)
# VolConf = builds(
#     Volume,
#     rf=RadianceFieldConf(),
#     filter=SpatialFilterConf(),
#     boxt=BoxTransformConf((-1.0,) * 3, (1.0,) * 3),
#     populate_full_signature=True,
# )

# SceneConfig = builds(
#     Scene,
#     cams={"cam_train": CameraConf(1.0, (100, 200))},
#     volume=VolConf(),
#     populate_full_signature=True,
# )
# print(to_yaml(SceneConfig))

from torchngp.conf import SceneConfig, CameraConf

scenecfg = SceneConfig(cams={"cam_val": CameraConf(1.0, (100, 200))})
print(to_yaml(scenecfg))
scene = instantiate(scenecfg)
print(scene)

# print(to_yaml(SceneConfig))

# cams = {"cam_train": Camera(1.0, (100, 200)), "cam_val": Camera(0.5, (100, 200))}
# vol = Volume(RadianceField(), SpatialFilter(), BoxTransform((-1.0,) * 3, (1.0,) * 3))
# scene = Scene(cams, vol)

# SceneConfig = builds(scene, populate_full_signature=True)
# print(to_yaml(SceneConfig))

# # def make_poses(poses: list[dict[str, Float3]]) -> torch.Tensor:
# #     ts = []
# #     for p in poses:
# #         t = torch.eye(4)
# #         t[:3, -1] = torch.as_tensor(p["tvec"])
# #         ts.append(t)
# #     return torch.stack(ts, 0)


# PosesConf = builds(make_poses, populate_full_signature=True)
# posescfg = PosesConf(
#     poses=[
#         {"rvec": (0.0, 0.0, 0.0), "tvec": (1.0, 2.0, 3.0)},
#         {"rvec": (0.0, 0.0, 0.0), "tvec": (1.0, 2.0, 3.0)},
#         {"rvec": (0.0, 0.0, 0.0), "tvec": (1.0, 2.0, 3.0)},
#     ]
# )
# print(to_yaml(posescfg))

# myposes = instantiate(posescfg, _convert_="partial")
# print(myposes)


# Scene:
#     cameras['train']
#     volume.aabb
#     volume.radiance_field
#     volume.spatial_filter

# RadianceRenderer:
#     tsampler

#     rnd(cam, poses, uv, volume)


# # print(world)

# # @hydra.main(config_path=None, config_name="my_app")
# # def task_function(cfg: Config):

# #     player = instantiate(cfg.player)  # an instance of `Character`

# #     print(player)

# #     with open("player_log.txt", "w") as f:
# #         f.write("Game session log:\n")
# #         f.write(f"Player: {player}\n")

# #     return player


# # if __name__ == "__main__":
# #     task_function()
