import logging
import hydra

from hydra.core.config_store import ConfigStore
from hydra_zen import make_config
from omegaconf import OmegaConf, DictConfig

from . import helpers

_logger = logging.getLogger("torchngp")

CreateConfigApp = make_config(
    output_path="./cfgs/nerf/nerf.yaml", scene_path="data/trivial/transforms.json"
)

cs = ConfigStore.instance()
cs.store(name="create_config", node=CreateConfigApp)


@hydra.main(version_base="1.2", config_path=None, config_name="create_config")
def main(cfg: DictConfig):

    nerf_cfg = helpers.NerfAppConfig(
        scene=helpers.LoadSceneFromJsonConf(path=cfg.scene_path),
    )

    OmegaConf.save(nerf_cfg, cfg.output_path)
    _logger.info(f"Saved config to {cfg.output_path}")


if __name__ == "__main__":
    main()
