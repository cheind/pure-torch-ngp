import torch
import torch.nn
import torch.optim
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from ngp_nerf.encoding import MultiLevelHashEncoding
from ngp_nerf import pixels


class Nerf2D(torch.nn.Module):
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
            torch.nn.Linear(n_hidden, n_out),
            torch.nn.Sigmoid(),
        )

    def forward(self, xn):
        f = self.encoder(xn)
        return self.mlp(f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_compression(model, img):
    params_bytes = count_parameters(model) * 4
    img_bytes = np.prod(img.shape)
    return img_bytes / params_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Image to compress")
    args = parser.parse_args()

    img = np.asarray(
        Image.open(
            args.image,
        )
    )
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    img = img.cuda()
    H, W = img.shape[1:]

    net = Nerf2D(
        n_out=img.shape[0],
        n_hidden=16,
        n_encodings=2**10,
        n_levels=16,
        min_res=8,
        max_res=max(img.shape[:2]),
    ).cuda()
    cf = compute_compression(net, img)
    print(f"Compression factor: {cf:.2f}x")

    coords = pixels.generate_grid_coords(img.shape[1:], indexing="xy").cuda()
    ncoords = pixels.normalize_coords(coords, indexing="xy").cuda()

    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    a = img.new_ones(img.shape[1:]) * 0.6

    for step in range(2000):
        mask = torch.bernoulli(a).bool()
        x_colors = img[:, mask]
        x_ncoords = ncoords[mask]
        y_colors = net(x_ncoords)
        opt.zero_grad()
        loss = F.mse_loss(y_colors, x_colors.permute(1, 0))
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(loss.item())

    with torch.no_grad():
        y_colors = net(ncoords.reshape(-1, 2))
        err = (y_colors.view((H, W, 3)) - img.permute(1, 2, 0)).abs().sum(-1)
    fig, axs = plt.subplots(1, 3, figsize=(16, 9), sharex=True, sharey=True)
    fig.suptitle(f"Nerf2D compression={cf:.2f}")
    axs[0].imshow(img.permute(1, 2, 0).cpu())
    axs[0].set_title("Input")
    axs[1].imshow(y_colors.view((H, W, 3)).cpu())
    axs[1].set_title("Reconstruction")
    axs[2].imshow(err.cpu())
    axs[2].set_title(f"Absolute Error mu={err.mean():2f}, std={err.std():.2f}")
    fig.savefig("result.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
