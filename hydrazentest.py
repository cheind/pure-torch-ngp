import torch

from hydra_zen import instantiate, to_yaml


from hydra_zen import save_as_yaml, load_from_yaml, make_config
from torchngp.config import (
    SceneConf,
    MultiViewCameraConf,
    Vecs3Conf,
    VolumeConf,
    NeRFTrainerConf,
    RadianceRendererConf,
    StratifiedRayStepSamplerConf,
)

from torchngp import io

scenecfg = SceneConf(
    cameras=[
        MultiViewCameraConf(
            focal_length=(1.0, 1.0),
            principal_point=(0.0, 0.0),
            size=(100, 100),
            rvec=Vecs3Conf([(0.1, 0.0, 0.0)]),
            tvec=Vecs3Conf([(1.0, 2.0, 3.0)]),
            image_paths=["data/lenna.png"],
        ),
        MultiViewCameraConf(
            focal_length=(1.0, 1.0),
            principal_point=(0.0, 0.0),
            size=(100, 100),
            rvec=Vecs3Conf([(0.0, 0.1, 0.2)]),
            tvec=Vecs3Conf([(1.0, 2.0, 3.0)]),
            image_paths=["data/lenna.png"],
        ),
    ],
)
print(to_yaml(scenecfg))
cams = instantiate(scenecfg.cameras)
print(cams[0].E)
print(cams[0].load_images().shape)
# scene = instantiate(scenecfg)

# # print(scene.cameras["cam_val"].T)
# # print(list(scene.volume.rf.parameters()))

# torch.save(scene.state_dict(), "test.pth")
# save_as_yaml(scenecfg, "test.yaml")


# scenecfg = load_from_yaml("test.yaml")
# scene: Scene = instantiate(scenecfg)
# scene.load_state_dict(torch.load("test.pth"))

# # print(scene)
# print(list(scene.parameters()))

# scenecfg = io.load_scene_from_json(
#     ["data/suzanne/transforms.json", "data/trivial/transforms.json"]
# )
# print(to_yaml(scenecfg))

# Conf = make_config(scene=scenecfg_loaded, volume=VolumeConf(aabb="${scene.aabb}"))
# cfg = Conf()
# print(to_yaml(cfg))

# inst = instantiate(cfg)
# print(inst.scene.aabb, inst.volume.aabb)

# cams = instantiate(scenecfg_loaded.cams, _convert_="all")
# print(type(cams))
# cams[0].load_images()

# https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.instantiate.html
# for aabb?

# trainercfg = NeRFTrainerConf(scene=SceneConf(), volume=VolumeConf())


from torchngp import io
from hydra_zen import make_custom_builds_fn

builds = make_custom_builds_fn(populate_full_signature=True)
LoadSceneFromJsonConf = builds(io.load_scene_from_json)

NeRFConf = make_config(
    scene=io.load_scene_from_json("data/suzanne/transforms.json"),
    volume=VolumeConf(aabb="${scene.aabb}"),
    renderer=RadianceRendererConf(),
    tsampler=StratifiedRayStepSamplerConf(),
    trainer=NeRFTrainerConf(),
)

nconf = NeRFConf()
print(to_yaml(nconf))
inst = instantiate(nconf)
