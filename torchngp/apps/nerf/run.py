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
from omegaconf import MISSING


from torchngp import config, scenes, training, io

_logger = logging.getLogger("torchngp")
logging.getLogger("PIL").setLevel(logging.WARNING)

NerfTrainConf = make_config(trainer=MISSING)


# 1) Register our config with Hydra's config store
cs = ConfigStore.instance()
cs.store(name="nerf_train", node=NerfTrainConf)


@hydra.main(version_base="1.2", config_path=None, config_name="nerf")
def run_nerf_train_task(cfg: DictConfig):
    _logger.debug("Training config")
    _logger.debug("\n" + to_yaml(cfg))

    OmegaConf.resolve(cfg)
    trainer: training.NeRFTrainer = instantiate(cfg.trainer)
    trainer.train()


if __name__ == "__main__":
    run_nerf_train_task()
