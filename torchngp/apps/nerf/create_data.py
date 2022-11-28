import logging

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, to_yaml, instantiate
from omegaconf import DictConfig, OmegaConf

from torchngp import config
from torchngp import io

from . import helpers

_logger = logging.getLogger("torchngp")

CreateConfigApp = make_config(output_path="./cfgs/nerf/data.yaml")

cs = ConfigStore.instance()
cs.store(name="create_data", node=CreateConfigApp)


@hydra.main(version_base="1.2", config_path=None, config_name="create_data")
def main(cfg: DictConfig):

    cfg = OmegaConf.create(
        {
            "aabb": io.AabbFromJsonConf(path="data/suzanne/transforms.json"),
            "cameras": {
                "train_camera": io.CamFromJsonConf(
                    path="data/suzanne/transforms.json", slice=":-3"
                ),
                "val_camera": io.CamFromJsonConf(
                    path="data/suzanne/transforms.json", slice="-3:"
                ),
            },
        }
    )
    print(to_yaml(instantiate(cfg)))

    # cfg = Conf()
    # print(to_yaml(instantiate(cfg)))

    # nerf_cfg = helpers.NerfAppConfig(
    #     scene=helpers.LoadSceneFromJsonConf(path=cfg.scene_path),
    # )

    # OmegaConf.save(nerf_cfg, cfg.output_path)
    # _logger.info(f"Saved config to {cfg.output_path}")


if __name__ == "__main__":
    main()
