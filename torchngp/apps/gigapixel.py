"""GigaPixel image compression

Learns to compress a single large image using 2D multi-level
hash encoding as described in the paper [1]. The compressed image
is given by a set of hash embedding vectors and a small
(in parameters) multi layer perceptron network. Given only
image positions, the original image can be reconstructed.

It has been shown that for the reconstruction of fine details of large
images, no more than 4% of the possible degrees of freedom of the
original image (#params / #pixels) are required.

This implementation is closely aligned to the proposed method in
[1], but might fail for true giga pixel images due to memory constraints.
Adding support would require to implement out-of-memory image manipulation
routines.

References:
[1] https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim

from torchngp import functional

from PIL import Image
from tqdm import tqdm

from torchngp import functional
from torchngp import modules


class CompressionModule(torch.nn.Module):
    def __init__(
        self,
        n_out: int,
        n_hidden: int,
        n_encodings: int,
        n_levels: int,
        min_res: int,
        max_res: int,
        max_n_dense: int,
    ) -> None:
        super().__init__()
        self.encoder = modules.MultiLevelHybridHashEncoding(
            n_encodings=n_encodings,
            n_input_dims=2,
            n_embed_dims=2,
            n_levels=n_levels,
            min_res=min_res,
            max_res=max_res,
            max_n_dense=max_n_dense,
        )
        n_features = self.encoder.n_levels * self.encoder.n_embed_dims

        self.mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_out),
        )

    def forward(self, xn):
        f = self.encoder(xn)
        return self.mlp(f)


def compute_dof_rate(model, img):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_pixels = np.prod(img.shape[1:])
    return n_params / n_pixels


@torch.no_grad()
def render_image(net, ncoords, image_shape, mean, std, batch_size) -> torch.Tensor:
    C, H, W = image_shape
    parts = []
    for batch in ncoords.reshape(-1, 2).split(batch_size):
        parts.append(net(batch))
    img = torch.cat(parts, 0)
    img = img.view((H, W, C)).permute(2, 0, 1)
    img = torch.clip(img * std + mean, 0, 1)
    return img


def main():
    parser = argparse.ArgumentParser(
        description=str(__doc__), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-epochs", type=int, help="Number of image epochs", default=100
    )
    parser.add_argument(
        "--batch-size", type=int, help="Locations per batch", default=2**17
    )
    parser.add_argument("--lr", type=float, help="Initial learning rate", default=1e-2)
    parser.add_argument(
        "--num-hidden", type=int, help="Number of hidden units", default=64
    )
    parser.add_argument(
        "--num-encodings",
        type=int,
        help="Number of encodings",
        default=2**14,
    )
    parser.add_argument("image", help="Image to compress")
    args = parser.parse_args()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Read image
    img = np.asarray(Image.open(args.image))
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    img = img.to(dev)
    coords = functional.make_grid(img.shape[1:], indexing="xy", device=dev)
    ncoords = functional.normalize_uv(coords, coords.shape[:-1], indexing="xy")

    # Compute image stats
    mean, std = img.mean((1, 2), keepdim=True), img.std((1, 2), keepdim=True)
    n_pixels = np.prod(img.shape[1:])

    # Normalize image
    nimg = (img - mean) / std

    # Setup module
    net = CompressionModule(
        n_out=img.shape[0],
        n_hidden=args.num_hidden,
        n_encodings=args.num_encodings,
        n_levels=16,
        min_res=16,
        max_res=max(img.shape[2:]) // 2,
        max_n_dense=sys.maxsize,
    ).to(dev)
    dofs = compute_dof_rate(net, img)
    print(f"DoFs of input: {dofs*100:.2f}%")

    # Setup optimizer and scheduler
    opt = torch.optim.AdamW(
        [
            {"params": net.encoder.parameters(), "weight_decay": 0.0},
            {"params": net.mlp.parameters(), "weight_decay": 1e-6},
        ],
        betas=(0.9, 0.99),
        eps=1e-15,
        lr=args.lr,
    )

    # Estimation of total steps such that each pixel is selected num_epochs
    # times.
    n_steps_per_epoch = max(n_pixels // args.batch_size, 1)
    n_steps = int(args.num_epochs * n_steps_per_epoch)

    # Using superconvergence (deviates from the paper)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-2, total_steps=n_steps)

    # Train
    pbar = tqdm(range(n_steps), mininterval=0.1)
    pbar_postfix = {"loss": 0.0}
    ncoords_flat = ncoords.reshape(-1, 2)
    nimg_flat = nimg.view(nimg.shape[0], -1)
    for epoch in range(args.num_epochs):
        ids = torch.randperm(ncoords_flat.shape[0], device=ncoords_flat.device)
        for batch in ids.split(args.batch_size):
            batch_ncoords = ncoords_flat[batch]
            batch_colors = nimg_flat[:, batch]
            x_colors = net(batch_ncoords)
            opt.zero_grad()
            loss = F.mse_loss(x_colors, batch_colors.permute(1, 0))
            loss.backward()
            opt.step()
            sched.step()
            pbar_postfix["loss"] = loss.item()
            pbar.set_postfix(**pbar_postfix, refresh=False)
            pbar.update()
        recimg = render_image(net, ncoords, nimg.shape, mean, std, args.batch_size)
        psnr, _ = functional.peak_signal_noise_ratio(
            img.unsqueeze(0), recimg.unsqueeze(0), 1.0
        )
        # sched.step(psnr.mean().item())
        pbar_postfix["lr"] = max(sched._last_lr)
        pbar_postfix["psnr[dB]"] = psnr.mean().item()

        if epoch % 20 == 0:
            recimg = recimg.permute(1, 2, 0).cpu() * 255
            Image.fromarray(recimg.to(torch.uint8).numpy()).save(
                f"tmp/out_{epoch:04d}.png"
            )

    imgrec = render_image(net, ncoords, nimg.shape, mean, std, args.batch_size)
    err = (imgrec - img).abs().sum(0)

    fig, axs = plt.subplots(1, 3, figsize=(16, 9), sharex=True, sharey=True)
    fig.suptitle(f"Nerf2D dof={dofs*100:.2f}")
    axs[0].imshow(img.permute(1, 2, 0).cpu())
    axs[0].set_title("Input")
    axs[1].imshow(imgrec.permute(1, 2, 0).cpu())
    axs[1].set_title("Reconstruction")
    axs[2].imshow(err.cpu())
    axs[2].set_title(f"Absolute Error mu={err.mean():2f}, std={err.std():.2f}")
    fig.savefig("tmp/uncompressed.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
