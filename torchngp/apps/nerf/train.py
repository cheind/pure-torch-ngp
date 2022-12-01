import logging

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, make_config, to_yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf import MISSING


_logger = logging.getLogger("torchngp")
logging.getLogger("PIL").setLevel(logging.WARNING)

TrainTaskConf = make_config(trainer=MISSING)
cs = ConfigStore.instance()
cs.store(name="train_task", node=TrainTaskConf)


@hydra.main(version_base="1.2", config_path="../../../cfgs/", config_name="train")
def train_task(cfg: DictConfig):
    _logger.info(f"Running NeRF on {cfg.data.name}")
    _logger.debug("Training config")
    _logger.debug("\n" + to_yaml(cfg))

    OmegaConf.resolve(cfg)
    trainer_partial = instantiate(cfg.trainer)
    trainer = trainer_partial(resolved_cfg=to_yaml(cfg))
    trainer.train()


if __name__ == "__main__":
    train_task()
