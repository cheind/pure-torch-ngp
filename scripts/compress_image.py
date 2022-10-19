import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim

from ngp_nerf import pixels, metrics
from ngp_nerf.encoding import MultiLevelHashEncoding

from PIL import Image
from tqdm import tqdm


class CompressionModule(torch.nn.Module):
    def __init__(
        self,
        n_out: int,
        n_hidden: int,
        n_encodings: int,
        n_levels: int,
        min_res: int,
        max_res: int,
    ) -> None:
        super().__init__()
        self.encoder = MultiLevelHashEncoding(
            n_encodings=n_encodings,
            n_input_dims=2,
            n_embed_dims=2,
            n_levels=n_levels,
            min_res=min_res,
            max_res=max_res,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-epochs", type=int, help="Number of image epochs", default=100
    )
    parser.add_argument(
        "--batch-size", type=int, help="Locations per batch", default=2**18
    )
    parser.add_argument("--lr", type=float, help="Initial learning rate", default=1e-2)
    parser.add_argument(
        "--num-hidden", type=int, help="Number of hidden units", default=64
    )
    parser.add_argument(
        "--num-encodings-exp",
        type=int,
        help="Exponent of number of encodings = 2**this",
        default=2**16,
    )
    parser.add_argument("image", help="Image to compress")
    args = parser.parse_args()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Read image
    img = np.asarray(
        Image.open(
            args.image,
        )
    )
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    img = img.to(dev)
    coords = pixels.generate_grid_coords(img.shape[1:], indexing="xy").to(dev)
    ncoords = pixels.normalize_coords(coords, indexing="xy").to(dev)

    # Compute image stats
    mean, std = img.mean((1, 2), keepdim=True), img.std((1, 2), keepdim=True)
    H, W = img.shape[1:]

    # Normalize image
    nimg = (img - mean) / std

    # Setup module
    net = CompressionModule(
        n_out=img.shape[0],
        n_hidden=args.num_hidden,
        n_encodings=2**args.num_encodings_exp,
        n_levels=16,
        min_res=16,
        max_res=max(img.shape[2:]) // 2,
    ).to(dev)
    dofs = compute_dof_rate(net, img)
    print(f"DoFs of input: {dofs*100:.2f}%")

    # Setup optimizer and scheduler
    opt = torch.optim.Adam(
        [
            {"params": net.encoder.parameters(), "weight_decay": 0.0},
            {"params": net.mlp.parameters(), "weight_decay": 1e-6},
        ],
        betas=(0.9, 0.99),
        eps=1e-15,
        lr=args.lr,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=20, factor=0.1, min_lr=1e-5, threshold=0.1
    )

    # Estimation of total steps such that each pixel is selected num_epochs
    # times.
    n_steps_per_epoch = max(np.prod(img.shape[1:]) // args.batch_size, 1)
    n_steps = args.num_epochs * n_steps_per_epoch

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
            pbar_postfix["loss"] = loss.item()
            pbar.set_postfix(**pbar_postfix, refresh=False)
            pbar.update()
        recimg = render_image(net, ncoords, nimg.shape, mean, std, args.batch_size)
        psnr, _ = metrics.peak_signal_noise_ratio(
            img.unsqueeze(0), recimg.unsqueeze(0), 1.0
        )
        sched.step(psnr.mean().item())
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
    axs[1].set_title("Reconstruction 1")
    axs[2].imshow(err.cpu())
    axs[2].set_title(f"Absolute Error mu={err.mean():2f}, std={err.std():.2f}")
    fig.savefig("tmp/uncompressed.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
