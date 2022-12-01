import logging

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config, to_yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf import MISSING

from torchngp import training

_logger = logging.getLogger("torchngp")
logging.getLogger("PIL").setLevel(logging.WARNING)

NerfTrainConf = make_config(trainer=MISSING)
cs = ConfigStore.instance()
cs.store(name="nerf_run", node=NerfTrainConf)


@hydra.main(version_base="1.2", config_path=None, config_name="nerf")
def run_task(cfg: DictConfig):
    _logger.info(f"Running NeRF on {cfg.data.name}")
    _logger.debug("Training config")
    _logger.debug("\n" + to_yaml(cfg))

    OmegaConf.resolve(cfg)
    trainer: training.NeRFTrainer = instantiate(cfg.trainer)
    trainer.train()


if __name__ == "__main__":
    run_task()
