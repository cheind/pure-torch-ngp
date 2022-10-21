import enum
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim

from ngp_nerf import rays, radiance

from ..encoding import MultiLevelHashEncoding
from ..scene import MultiViewScene, MultiViewDataset


class NeRF(torch.nn.Module):
    def __init__(
        self,
        n_colors: int,
        n_hidden: int,
        n_encodings: int,
        n_levels: int,
        min_res: int,
        max_res: int,
        is_hdr: bool = False,
    ) -> None:
        super().__init__()

        self.pos_encoder = MultiLevelHashEncoding(
            n_encodings=n_encodings,
            n_input_dims=3,
            n_embed_dims=2,
            n_levels=n_levels,
            min_res=min_res,
            max_res=max_res,
        )
        n_enc_features = self.pos_encoder.n_levels * self.pos_encoder.n_embed_dims
        self.is_hdr = is_hdr
        # 1-layer hidden density mlp
        self.density_mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_enc_features, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 16),  # 0 density in log-space
        )

        # 2-layer hidden color mlp
        self.color_mlp = torch.nn.Sequential(
            torch.nn.Linear(16, n_hidden),  # TODO: add aux-dims
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_colors),
        )

    def forward(self, x):
        """Predict density and colors for sample positions.

        Params:
            x: (B,3) sampling positions in normalized [-1,+1] range

        Returns:
            sigma: (B,) estimated density values
            colors: (B,C) estimated color values
        """

        h = self.pos_encoder(x)
        d = self.density_mlp(h)
        c = self.color_mlp(d)  # TODO: concat aux. dims
        c = c.exp() if self.is_hdr else torch.sigmoid(c)
        sigma = d[:, 0].exp()

        return sigma, c


def train(
    nerf: torch.nn.Module,
    scene: MultiViewScene,
    dev: torch.device,
    batch_size: int,
    n_epochs: int,
    n_samples: int,
    lr: float = 1e-2,
):

    opt = torch.optim.AdamW(
        [
            {"params": nerf.pos_encoder.parameters(), "weight_decay": 0.0},
            {"params": nerf.density_mlp.parameters(), "weight_decay": 1e-6},
            {"params": nerf.color_mlp.parameters(), "weight_decay": 1e-6},
        ],
        betas=(0.9, 0.99),
        eps=1e-15,
        lr=lr,
    )

    # Set all colors corresponding to alpha to a random value
    # images = data["colors"]
    # masks = data["masks"]
    # # TODO when using random background pixels, ensure that start color matches random color
    # images[~masks] = images[~masks].uniform_(0.0, 1.0)

    # import matplotlib.pyplot as plt

    # plt.imshow(images[0].cpu())
    # plt.show()

    # Exclude all rays that are invalid (miss aabb)
    # tnear = data["tnear"]
    # tfar = data["tfar"]
    # validmask = tnear < tfar
    # validmask[10] = False
    # # validmask[-1] = False  # don't use in training
    # tnear = tnear[validmask].contiguous()
    # tfar = tfar[validmask].contiguous()
    # origins = data["origins"][validmask].contiguous()
    # dirs = data["directions"][validmask].contiguous()
    # colors = images[validmask].contiguous()
    # masks = masks[validmask].contiguous()
    # aabb_min = scene.aabb_minc.to(dev)
    # aabb_max = scene.aabb_maxc.to(dev)
    # base_colors = colors.clone()
    # base_colors[masks] = 0.0

    # n_pixels = colors.shape[0]
    # n_steps = int(n_epochs * n_pixels / batch_size)

    # sched = torch.optim.lr_scheduler.OneCycleLR(
    #     opt, max_lr=5e-1, total_steps=n_steps
    # )

    train_ds = MultiViewDataset(scene=scene)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=10,
        sampler=train_ds.create_mini_batch_sampler(batch_size // 10, random=True),
        num_workers=4,
    )

    from tqdm import tqdm

    aabb_min, aabb_max = scene.aabb_minc.to(dev), scene.aabb_maxc.to(dev)

    pbar_postfix = {"loss": 0.0}
    step = 0
    for epoch in range(n_epochs):
        pbar = tqdm(train_dl, mininterval=0.1)
        for batch in pbar:
            # Collapse mini-batch dim and upload
            batch = [b.view(-1, *b.shape[2:]) for b in batch]
            batch = [b.to(dev) for b in batch]

            color, alpha, o, d, tnear, tfar = batch
            noise = torch.empty_like(color).uniform_(0.0, 1.0)
            color = color * alpha[..., None] + noise * (1 - alpha[..., None])

            ts = rays.sample_rays_uniformly(tnear, tfar, n_samples)
            xyz = o[:, None] + ts[..., None] * d[:, None]  # (B,T,3)
            nxyz = rays.normalize_xyz_in_aabb(xyz, aabb_min, aabb_max)

            pred_sample_sigma, pred_sample_color = nerf(nxyz.view(-1, 3))
            pred_colors, pred_transm, _ = radiance.integrate_path(
                pred_sample_color.view(-1, n_samples, 3),
                pred_sample_sigma.view(-1, n_samples),
                ts,
                tfar,
            )
            pred_alpha = 1.0 - pred_transm[:, -1]
            pred_colors = pred_colors * pred_alpha[..., None] + noise * (
                1 - pred_alpha[..., None]
            )

            opt.zero_grad()
            loss = F.mse_loss(pred_colors, color)
            loss.backward()
            opt.step()
            # sched.step()
            pbar_postfix["loss"] = loss.item()
            pbar.set_postfix(**pbar_postfix, refresh=False)
            step += 1

            if (step % 100) == 0:
                v = scene.views[0].to(dev)
                img = v.render_image(
                    nerf,
                    aabb_min,
                    aabb_max,
                    batch_size=batch_size,
                    n_samples=100,
                )
                fig, ax = v.plot_image(img)
                fig.savefig(f"tmp/nerf_{step:04d}.png")
                import matplotlib.pyplot as plt

                plt.show()


if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scene = MultiViewScene()
    scene.load_from_json("data/suzanne/transforms.json")
    nerf = NeRF(
        3,
        n_hidden=64,
        n_encodings=2**14,
        n_levels=32,
        min_res=16,
        max_res=256,
        is_hdr=False,
    ).to(dev)
    train(nerf, scene, dev, batch_size=2**14, n_epochs=20, n_samples=100, lr=1e-3)
