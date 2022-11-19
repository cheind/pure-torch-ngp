import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config, make_custom_builds_fn, to_yaml
from omegaconf import DictConfig

from torchngp import config, scenes, training, io

_logger = logging.getLogger("torchngp")
logging.getLogger("PIL").setLevel(logging.WARNING)


builds = make_custom_builds_fn(populate_full_signature=True)
LoadSceneFromJsonConf = builds(io.load_scene_from_json)

TrainAppConfig = make_config(
    # scene=io.load_scene_from_json("data/suzanne/transforms.json", pose_to_cv=True),
    scene=LoadSceneFromJsonConf(path="data/trivial/transforms.json"),
    volume=config.VolumeConf(aabb="${scene.aabb}"),
    renderer=config.RadianceRendererConf(),
    tsampler=config.StratifiedRayStepSamplerConf(),
    trainer=config.NeRFTrainerConf(
        train_opts=config.NeRFTrainerOptionsConf(
            output_dir="${hydra:runtime.output_dir}"
        )
    ),
)

# 1) Register our config with Hydra's config store
cs = ConfigStore.instance()
cs.store(name="train", node=TrainAppConfig)


def classname(cls):
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name


@hydra.main(version_base="1.2", config_path=None, config_name="train")
def train(cfg: DictConfig):
    _logger.debug("Training config")
    _logger.debug(to_yaml(cfg))
    if cfg.scene._target_ != classname(scenes.Scene):
        cfg.scene = instantiate(cfg.scene)

    inst = instantiate(cfg)

    trainer: training.NeRFTrainer = inst.trainer
    trainer.train(inst.scene, inst.volume, inst.renderer, inst.tsampler)


if __name__ == "__main__":
    import sys

    sys.argv.append(
        f"hydra.searchpath=[file://{Path.cwd().as_posix()},"
        f"file://{Path.cwd().as_posix()}/cfgs/nerf]"
    )
    train()
