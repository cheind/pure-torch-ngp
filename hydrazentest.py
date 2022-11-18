import torch

from hydra_zen import instantiate, to_yaml


from hydra_zen import save_as_yaml, load_from_yaml
from torchngp.conf import SceneConfig, CameraConf, Scene, PosesConf, Pose, VolumeConf

scenecfg = SceneConfig(
    cams={
        "cam_val": CameraConf(
            1.0,
            (100, 200),
            T=PosesConf([Pose(0.1, 0.0, 0.0, 1, 2, 3), Pose(0.1, 0.0, 0.0, 4, 5, 6)]),
        )
    },
    volume=VolumeConf(),
)
print(to_yaml(scenecfg))
scene = instantiate(scenecfg)

# print(scene.cameras["cam_val"].T)
# print(list(scene.volume.rf.parameters()))

torch.save(scene.state_dict(), "test.pth")
save_as_yaml(scenecfg, "test.yaml")


scenecfg = load_from_yaml("test.yaml")
scene: Scene = instantiate(scenecfg)
scene.load_state_dict(torch.load("test.pth"))

# print(scene)
print(list(scene.parameters()))
