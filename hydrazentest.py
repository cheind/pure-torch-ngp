import torch

from hydra_zen import instantiate, to_yaml


from hydra_zen import save_as_yaml, load_from_yaml
from torchngp.config import (
    SceneConf,
    MultiViewCameraConf,
    Vecs3Conf,
    VolumeConf,
    load_transforms_json,
)

scenecfg = SceneConf(
    cams=[
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
    volume=VolumeConf(),
)
print(to_yaml(scenecfg))
cams = instantiate(scenecfg.cams)
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

scenecfg_loaded = load_transforms_json("data/suzanne/transforms.json")
print(to_yaml(scenecfg_loaded))

cams = instantiate(scenecfg_loaded.cams, _convert_="all")
print(type(cams))
cams[0].load_images()
