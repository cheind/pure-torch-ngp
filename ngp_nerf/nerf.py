import torch
import torch.nn
import torch.nn.functional as F
import torch.optim

from ngp_nerf import rays, radiance

from .encoding import MultiLevelHashEncoding
from .scene import MultiViewScene


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
        self,
        scene: MultiViewScene,
        dev: torch.device,
        batch_size: int,
        n_epochs: int,
        n_samples: int,
        lr: float = 1e-2,
    ):
        self.to(dev)
        data = scene.compute_train_info()

        opt = torch.optim.AdamW(
            [
                {"params": self.pos_encoder.parameters(), "weight_decay": 0.0},
                {"params": self.density_mlp.parameters(), "weight_decay": 1e-6},
                {"params": self.color_mlp.parameters(), "weight_decay": 1e-6},
            ],
            betas=(0.9, 0.99),
            eps=1e-15,
            lr=lr,
        )

        # Set all colors corresponding to alpha to a random value
        images = data["colors"]
        masks = data["masks"]
        # TODO when using random background pixels, ensure that start color matches random color
        images[~masks] = images[~masks].uniform_(0.0, 1.0)

        # import matplotlib.pyplot as plt

        # plt.imshow(images[0].cpu())
        # plt.show()

        # Exclude all rays that are invalid (miss aabb)
        tnear = data["tnear"]
        tfar = data["tfar"]
        validmask = tnear < tfar
        validmask[10] = False
        # validmask[-1] = False  # don't use in training
        tnear = tnear[validmask].contiguous()
        tfar = tfar[validmask].contiguous()
        origins = data["origins"][validmask].contiguous()
        dirs = data["directions"][validmask].contiguous()
        colors = images[validmask].contiguous()
        masks = masks[validmask].contiguous()
        aabb_min = scene.aabb_minc.to(dev)
        aabb_max = scene.aabb_maxc.to(dev)
        base_colors = colors.clone()
        base_colors[masks] = 0.0

        n_pixels = colors.shape[0]
        n_steps = int(n_epochs * n_pixels / batch_size)

        # sched = torch.optim.lr_scheduler.OneCycleLR(
        #     opt, max_lr=5e-1, total_steps=n_steps
        # )

        from tqdm import tqdm

        pbar = tqdm(range(n_steps), mininterval=0.1)
        pbar_postfix = {"loss": 0.0}

        for epoch in range(n_epochs):
            ids = torch.randperm(n_pixels, device=dev)
            for idx, batch in enumerate(ids.split(batch_size)):
                B = batch.shape[0]
                batch_colors = colors[batch].to(dev)
                batch_origins = origins[batch].to(dev)
                batch_dirs = dirs[batch].to(dev)
                batch_tnear = tnear[batch].to(dev)
                batch_tfar = tfar[batch].to(dev)
                batch_basecolors = base_colors[batch].to(dev)

                ts = rays.sample_rays_uniformly(batch_tnear, batch_tfar, n_samples)

                xyz = (
                    batch_origins[:, None] + ts[..., None] * batch_dirs[:, None]
                )  # (B,T,3)
                nxyz = rays.normalize_xyz_in_aabb(xyz, aabb_min, aabb_max)

                pred_sample_sigma, pred_sample_color = self(nxyz.view(-1, 3))
                pred_colors, _, _ = radiance.integrate_path(
                    pred_sample_color.view(B, n_samples, 3),
                    pred_sample_sigma.view(B, n_samples),
                    ts,
                    batch_tfar,
                )
                pred_colors = pred_colors + batch_basecolors

                opt.zero_grad()
                loss = F.mse_loss(pred_colors, batch_colors)
                loss.backward()
                opt.step()
                # sched.step()
                pbar_postfix["loss"] = loss.item()
                pbar.set_postfix(**pbar_postfix, refresh=False)
                pbar.update()

                if idx % 200 == 0:
                    self.render_image(
                        data["origins"][10].to(dev),
                        data["directions"][10].to(dev),
                        aabb_min,
                        aabb_max,
                        idx,
                        batch_size=batch_size,
                        n_samples=100,
                    )
                    self.render_image(
                        data["origins"][12].to(dev),
                        data["directions"][12].to(dev),
                        aabb_min,
                        aabb_max,
                        idx + 1,
                        batch_size=batch_size,
                        n_samples=100,
                    )

    @torch.no_grad()
    def render_image(
        self,
        origins: torch.Tensor,
        dirs: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        step: int,
        batch_size: int,
        n_samples: int = 100,
    ):
        H, W = origins.shape[:2]

        origins = origins.view(-1, 3)
        dirs = dirs.view(-1, 3)

        tnear, tfar = rays.intersect_rays_aabb(
            origins,
            dirs,
            aabb_min,
            aabb_max,
        )

        valid = tnear < tfar
        tnear = tnear[valid]
        tfar = tfar[valid]
        origins = origins[valid]
        dirs = dirs[valid]

        ts = rays.sample_rays_uniformly(tnear, tfar, n_samples)
        xyz = origins[:, None] + ts[..., None] * dirs[:, None]  # (B,T,3)
        nxyz = rays.normalize_xyz_in_aabb(xyz, aabb_min, aabb_max)

        parts = []
        for batch_ids in torch.arange(dirs.shape[0]).split(batch_size):
            B = len(batch_ids)
            pred_sample_sigma, pred_sample_color = self(nxyz[batch_ids].view(-1, 3))
            pred_colors, pred_transm, _ = radiance.integrate_path(
                pred_sample_color.view(B, n_samples, 3),
                pred_sample_sigma.view(B, n_samples),
                ts[batch_ids],
                tfar[batch_ids],
            )
            parts.append(pred_colors)

        pred_colors = torch.cat(parts, 0)
        img = origins.new_zeros((H, W, 3))
        img.view(-1, 3)[valid] = pred_colors

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(img.cpu())
        # ax[1].imshow(pred_transm.cpu())
        fig.savefig(f"tmp/nerf_{step:04d}.png")
        plt.close(fig)
        pass


if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scene = MultiViewScene()
    scene.load_from_json("data/suzanne/transforms.json")
    nerf = NeRF(
        3,
        n_hidden=64,
        n_encodings=2**14,
        n_levels=64,
        min_res=16,
        max_res=256,
        is_hdr=False,
    )
    nerf.train(scene, dev, batch_size=2**12, n_epochs=20, n_samples=100, lr=1e-3)
