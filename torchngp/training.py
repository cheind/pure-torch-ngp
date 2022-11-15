import copy
from itertools import islice
from PIL import Image

import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F

from torchngp import rendering, radiance, geometric, sampling, plotting


class MultiViewDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        camera: geometric.MultiViewCamera,
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


def create_fwd_bwd_closure(
    rf: radiance.RadianceField,
    renderer: rendering.RadianceRenderer,
    tsampler: sampling.RayStepSampler,
    scaler: torch.cuda.amp.GradScaler,
    n_acc_steps: int,
):
    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
    def run_fwd_bwd(
        cam: geometric.MultiViewCamera, uv: torch.Tensor, rgba: torch.Tensor
    ):
        B, N, M, C = rgba.shape
        maps = {"color", "alpha"}

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
            pred_maps = renderer.trace_uv(rf, cam, uv, tsampler, which_maps=maps)
            pred_rgb, pred_alpha = pred_maps["color"], pred_maps["alpha"]
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
    cam: geometric.MultiViewCamera,
    tsampler: sampling.RayStepSampler,
    use_amp: bool,
    n_samples_per_cam: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.amp.autocast(enabled=use_amp):
        maps = renderer.trace_maps(
            rf, cam, tsampler=tsampler, n_samples_per_cam=n_samples_per_cam
        )
        pred_rgba = torch.cat((maps["color"], maps["alpha"]), -1).permute(0, 3, 1, 2)
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
