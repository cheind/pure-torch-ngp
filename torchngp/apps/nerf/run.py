import logging
import copy
from typing import Any
from pathlib import Path

import hydra
from hydra.experimental.callback import Callback as HydraCallback
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config, make_custom_builds_fn, to_yaml
from omegaconf import DictConfig, OmegaConf, read_write


from .helpers import NerfAppConfig
from torchngp import config, scenes, training, io

_logger = logging.getLogger("torchngp")
logging.getLogger("PIL").setLevel(logging.WARNING)


# 1) Register our config with Hydra's config store
cs = ConfigStore.instance()
cs.store(name="nerf", node=NerfAppConfig)


@hydra.main(version_base="1.2", config_path=None, config_name="nerf")
def run_nerf_task(cfg: DictConfig):
    _logger.debug("Training config")
    _logger.debug("\n" + to_yaml(cfg))
    if cfg.scene._target_ != _get_qual_classname(scenes.Scene):
        cfg.scene = instantiate(cfg.scene)

    # Save resolved config
    OmegaConf.save(
        cfg,
        Path(HydraConfig.get().runtime.output_dir) / "resolved.yaml",
        resolve=True,
    )

    # need 'all' or dataclass trainer is not instantiated
    inst = instantiate(cfg, _convert_="all")

    trainer: training.NeRFTrainer = inst["trainer"]
    trainer.train(inst["scene"], inst["volume"], inst["renderer"], inst["tsampler"])


def _get_qual_classname(cls):
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name


if __name__ == "__main__":
    run_nerf_task()
