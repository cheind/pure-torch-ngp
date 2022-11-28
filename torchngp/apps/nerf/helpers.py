# from hydra_zen import make_config, make_custom_builds_fn
# from omegaconf import MISSING

# from torchngp import io, config

# builds = make_custom_builds_fn(populate_full_signature=True)
# LoadSceneFromJsonConf = builds(io.load_scene_from_json)

# NerfAppConfig = make_config(
#     scene=MISSING,
#     volume=config.VolumeConf(aabb="${scene.aabb}"),
#     train_renderer=config.RadianceRendererConf(
#         tsampler=config.StratifiedRayStepSamplerConf(128)
#     ),
#     val_renderer=config.RadianceRendererConf(
#         tsampler=config.StratifiedRayStepSamplerConf(512)
#     ),
#     trainer=config.NeRFTrainerConf(output_dir="${hydra:runtime.output_dir}"),
# )


# # scene=io.load_scene_from_json("data/suzanne/transforms.json", pose_to_cv=True),
