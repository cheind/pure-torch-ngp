"""Reconstruct image from random sampling.

Shows reconstruction behaviors from different
sampling strategies.

randperm sampling generates fewer duplications
and after the same number of samples, reconstructs
the input image better.

Note, the current impl. of reconstruction 
might access duplicate memory locations in place,
leading to undefined results. This effect is noticable
in the reconstruction image. An alternative
reconstruction without this problem is provided 
(commented), but is much slower.
"""

from torchngp import functional
from torchngp import modules
from torchngp.modules.encoding import _bilinear_params_2d
import torch
from itertools import islice


def reconstruct(
    image_rgba,
    method=functional.generate_random_uv_samples,
    M=512,
    batches=512,
    subpixel=True,
):
    H, W = image_rgba.shape[-2:]
    cam = modules.MultiViewCamera((1, 1), (0, 0), size=(W, H), poses=[torch.eye(4)])
    gen = method(
        cam.size, cam.n_views, image_rgba, n_samples_per_view=M, subpixel=subpixel
    )
    rgba_rec = torch.zeros(
        (H + 2, W + 2, 4)
    )  # slightly larger to account for outside corners
    rgba_w = torch.zeros((H + 2, W + 2), dtype=torch.float64)
    for uv, rgba in islice(gen, batches):
        rgba = rgba[0]
        uv = uv[0]
        corners, weights, mask = _bilinear_params_2d(uv, shape=(H, W))
        # Fast, but collisions leading to noticably visual artefacts
        mask = mask.any(-1)
        c = corners[mask]
        w = weights[mask]
        rgba = rgba[mask]
        rgba_rec[c[..., 1] + 1, c[..., 0] + 1] += rgba[:, None, :] * w[..., None]
        rgba_w[c[..., 1] + 1, c[..., 0] + 1] += w

        # Slow but correct
        # for c, w, m, color in zip(corners, weights, mask, rgba):
        #     if m.all():
        #         rgba_rec[c[:, 1]+1, c[:, 0]+1] += color[None] * w[:, None]
        #         rgba_w[c[:, 1]+1, c[:, 0]+1] += w

    mask = rgba_w > 0
    rgba_rec[mask] /= rgba_w[mask].unsqueeze(-1)
    rgba_rec = rgba_rec[1:-1, 1:-1]
    return rgba_rec


def main():

    image_rgba = functional.load_image(["data/lenna.png"])

    import matplotlib.pyplot as plt

    cfg = dict(M=512, batches=512, subpixel=True)

    torch.manual_seed(123)
    rec1 = reconstruct(
        image_rgba, method=functional.uv_sampling.generate_randperm_uv_samples, **cfg
    )
    torch.manual_seed(123)
    rec2 = reconstruct(
        image_rgba, method=functional.uv_sampling.generate_random_uv_samples, **cfg
    )

    gt_img = image_rgba[0].permute(1, 2, 0)

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    axs[0, 0].imshow(gt_img)
    axs[0, 0].set_title("original")
    axs[0, 1].imshow(rec1)
    axs[0, 1].set_title("rec randperm")
    axs[0, 2].imshow((gt_img - rec1).abs().sum(-1), vmin=0, vmax=1)
    axs[0, 2].set_title("error randperm")
    axs[1, 0].imshow(gt_img)
    axs[1, 0].set_title("original")
    axs[1, 1].imshow(rec2)
    axs[1, 1].set_title("rec randmc")
    axs[1, 2].imshow((gt_img - rec2).abs().sum(-1), vmin=0, vmax=1)
    axs[1, 2].set_title("error randmc")
    plt.show()


if __name__ == "__main__":
    main()
