import pytest
import torch
import matplotlib.pyplot as plt
from torch.testing import assert_close

from torchngp import config, io, geometric, plotting
from hydra_zen import instantiate


def test_ray_box_intersect_real():
    scenecfg = io.load_scene_from_json("data/lego/transforms_train.json")
    scene = instantiate(scenecfg)
    cam = scene.cameras[0]
    sel = cam[4:5]
    rays = geometric.RayBundle.make_world_rays(sel, sel.make_uv_grid())
    newrays = rays.intersect_aabb(scene.aabb)
    print(scene.aabb)
    arays = newrays.filter_by_mask(newrays.active_mask())
    print(newrays.active_mask().numel(), newrays.active_mask().sum())
    xyz = arays(arays.tfar)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plotting.plot_camera(sel, ax=ax)
    plotting.plot_box(scene.aabb, ax=ax)
    ax.scatter(xyz[::4, 0], xyz[::4, 1], xyz[::4, 2], s=5)
    plt.show()
