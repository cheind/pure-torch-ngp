import time
import logging
import copy
from itertools import islice
from PIL import Image

import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

from torchngp import rendering, radiance, cameras, sampling, plotting, filtering
import matplotlib.pyplot as plt


class MultiViewDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        camera: cameras.MultiViewCamera,
        images: torch.Tensor,
        n_samples_per_cam: int = None,
        random: bool = True,
        subpixel: bool = True,
    ):
        self.camera = copy.deepcopy(camera)
        self.images = images.clone()
        self.n_pixels_per_cam = camera.size.prod().item()
        if n_samples_per_cam is None:
            # width of image per mini-batch
            n_samples_per_cam = camera.size[0].item()
        self.n_samples_per_cam = n_samples_per_cam
        self.random = random
        self.subpixel = subpixel if random else False

    def __iter__(self):
        if self.random:
            return islice(
                sampling.generate_random_uv_samples(
                    camera=self.camera,
                    image=self.images,
                    n_samples_per_cam=self.n_samples_per_cam,
                    subpixel=self.subpixel,
                ),
                len(self),
            )
        else:
            return sampling.generate_sequential_uv_samples(
                camera=self.camera,
                image=self.images,
                n_samples_per_cam=self.n_samples_per_cam,
                n_passes=1,
            )

    def __len__(self) -> int:
        # Number of mini-batches required to match with number of total pixels
        return self.n_pixels_per_cam // self.n_samples_per_cam


def make_run_fwd_bwd(
    rf: radiance.RadianceField,
    renderer: rendering.RadianceRenderer,
    scaler: torch.cuda.amp.GradScaler,
    n_acc_steps: int,
):
    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
    def run_fwd_bwd(cam: cameras.MultiViewCamera, uv: torch.Tensor, rgba: torch.Tensor):
        B, N, M, C = rgba.shape

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            uv = uv.permute(1, 0, 2, 3).reshape(N, B * M, 2)
            rgba = rgba.permute(1, 0, 2, 3).reshape(N, B * M, C)
            rgb, alpha = rgba[..., :3], rgba[..., 3:4]
            noise = torch.empty_like(rgb).uniform_(0.0, 1.0)
            # Dynamic noise background with alpha composition
            # Encourages the model to learn zero density in empty regions
            # Dynamic background is also combined with prediced colors, so
            # model does not have to learn randomness.
            gt_rgb_mixed = rgb * alpha + noise * (1 - alpha)

            # Predict
            pred_rgb, pred_alpha = renderer.render_uv(rf, cam, uv, booster=2.0)
            # Mix
            pred_rgb_mixed = pred_rgb * pred_alpha + noise * (1 - pred_alpha)

            # Loss normalized by number of accumulation steps before
            # update
            loss = F.smooth_l1_loss(pred_rgb_mixed, gt_rgb_mixed)
            loss = loss / n_acc_steps

        # Scale the loss
        scaler.scale(loss).backward()
        return loss

    return run_fwd_bwd


@torch.no_grad()
def render_images(
    rf: radiance.RadianceField,
    renderer: rendering.RadianceRenderer,
    cam: cameras.MultiViewCamera,
    use_amp: bool,
    n_samples_per_cam: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.amp.autocast(enabled=use_amp):
        pred_color, pred_alpha = renderer.render_camera_views(
            rf, cam, n_samples_per_cam=n_samples_per_cam, booster=1
        )
        pred_rgba = torch.cat((pred_color, pred_alpha), -1).permute(0, 3, 1, 2)
    return pred_rgba


def save_image(fname: str, rgba: torch.Tensor):
    grid_img = (
        (plotting.make_image_grid(rgba, checkerboard_bg=False, scale=1.0) * 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(grid_img, mode="RGBA").save(fname)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from .io import load_scene_from_json

    torch.multiprocessing.set_start_method("spawn")

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # camera_train, aabb, gt_images_train = load_scene_from_json(
    #     "./data/lego/transforms_train.json", load_images=True
    # )
    # camera_val, _, gt_images_val = load_scene_from_json(
    #     "./data/lego/transforms_val.json", load_images=True
    # )
    # train_mvs = (camera_train, gt_images_train)
    # val_mvs = (camera_val[:3], gt_images_val[:3])

    camera, aabb, gt_images = load_scene_from_json(
        "./data/suzanne/transforms.json", load_images=True
    )
    train_mvs = camera[:-2], gt_images[:-2]
    val_mvs = camera[-2:], gt_images[-2:]

    # camera, aabb, gt_images = load_scene_from_json(
    #     "./data/trivial/transforms.json", load_images=True
    # )

    # train_mvs = camera, gt_images
    # val_mvs = camera, gt_images

    # train_mvs = camera[:-2], gt_images[:-2]
    # val_mvs = camera[-2:], gt_images[-2:]

    # plotting.plot_camera(train_mvs[0])
    # plotting.plot_box(aabb)
    # plt.gca().set_aspect("equal")
    # plt.gca().autoscale()
    # plt.show()

    # ds = MultiViewDataset(mvs)

    nerf_kwargs = dict(
        n_colors=3,
        n_hidden=64,
        n_encodings=2**18,
        n_levels=16,
        n_color_cond=16,
        min_res=32,
        max_res=512,  # can now specify much larger resolutions due to hybrid approach
        max_n_dense=256**3,
        is_hdr=False,
    )
    nerf = radiance.NeRF(**nerf_kwargs).to(dev)

    train_time = 10 * 3600
    train(
        nerf=nerf,
        aabb=aabb,
        train_mvs=train_mvs,
        test_mvs=val_mvs,
        n_rays_batch=2**14,
        n_rays_mini_batch=2**14,
        lr=1e-2,
        max_train_secs=train_time,
        preload=False,
        dev=dev,
    )

    # with torch.no_grad(), torch.cuda.amp.autocast():
    #     import numpy as np

    #     t = time.time()
    #     vol_colors, vol_sigma = radiance.rasterize_field(
    #         nerf, nerf.aabb, (512, 512, 512), batch_size=2**18, device=dev
    #     )
    #     vol_rgbd = torch.cat((vol_colors, vol_sigma), -1).cpu()
    #     print(time.time() - t)
    #     np.savez(
    #         "tmp/volume.npz",
    #         rgb=vol_rgbd[..., :3],
    #         d=vol_rgbd[..., 3],
    #         aabb=nerf.aabb.cpu(),
    #     )

    # # ax = plotting.plot_camera(mvs.cameras)
    # # ax = plotting.plot_box(mvs.aabb)
    # # ax.set_aspect("equal")
    # # ax.relim()  # make sure all the data fits
    # # ax.autoscale()
    # import matplotlib.pyplot as plt

    # # plt.show()
    # img = torch.empty((2, 4, 30, 40)).uniform_(0.0, 1.0)
    # plotting.plot_image(img, scale=0.5)
    # plt.show()

    # # dl = torch.utils.data.DataLoader(ds, batch_size=4)
    # # print(len(dl))
    # # for idx, (uv, uv_f) in enumerate(dl):
    # #     print(idx, uv.shape)
    # # print(idx)
