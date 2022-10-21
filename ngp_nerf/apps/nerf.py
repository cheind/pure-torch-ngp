import math
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim
import time

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
    train_scene: MultiViewScene,
    test_scene: MultiViewScene,
    dev: torch.device,
    batch_size: int,
    n_samples: int,
    lr: float = 1e-2,
    max_train_secs: float = 180,
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

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.75, patience=40, min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()

    train_ds = MultiViewDataset(scene=train_scene)
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
    t_start = time.time()
    t_last_dump = time.time()
    for _ in range(1000):
        pbar = tqdm(train_dl, mininterval=0.1)
        for batch in pbar:
            t_now = time.time()
            opt.zero_grad()

            # Collapse mini-batch dim and upload
            batch = [b.view(-1, *b.shape[2:]) for b in batch]
            batch = [b.to(dev) for b in batch]

            color, alpha, o, d, tnear, tfar = batch
            noise = torch.empty_like(color).uniform_(0.0, 1.0)
            # Dynamic noise background with alpha composition
            # Encourages the model to learn zero density in empty regions
            # Dynamic background is also combined with prediced colors, so
            # model does not have to learn randomness.
            color = color * alpha[..., None] + noise * (1 - alpha[..., None])

            with torch.cuda.amp.autocast():
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

                loss = F.mse_loss(pred_colors, color)
                if t_now - t_last_dump > 30:
                    render_test_scenes(
                        nerf,
                        test_scene,
                        dev,
                        batch_size,
                        n_samples * 4,
                        t_now - t_start,
                        show=False,
                    )
                    t_last_dump = t_now

            loss = scaler.scale(loss)
            loss.backward()
            scaler.step(opt)
            scaler.update()
            # opt.step()
            sched.step(loss)
            pbar_postfix["loss"] = loss.item()
            pbar_postfix["lr"] = sched._last_lr
            pbar.set_postfix(**pbar_postfix, refresh=False)
            step += 1
            if (t_now - t_start) > max_train_secs:
                return


@torch.no_grad()
def render_test_scenes(
    nerf: radiance.RadianceField,
    test_scene: MultiViewScene,
    dev: torch.device,
    batch_size: int,
    n_samples: int = 100,
    t_elapsed: int = 0.0,
    show: bool = False,
):
    import matplotlib.pyplot as plt

    N = len(test_scene.views)
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    ncols = 2
    nrows = int(math.ceil(N / ncols))
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows)

    for vidx, v in enumerate(test_scene.views):
        ax = fig.add_subplot(spec[vidx // ncols, vidx % ncols])
        v = v.to(dev)
        img = v.render_image(
            nerf,
            test_scene.aabb_minc.to(dev),
            test_scene.aabb_maxc.to(dev),
            batch_size=batch_size,
            n_samples=n_samples,
        )
        v.plot_image(img, ax=ax)
    fig.savefig(f"tmp/nerf_secs{int(t_elapsed):03d}.png")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scene = MultiViewScene()
    scene.load_from_json("data/suzanne/transforms.json")
    # scene.render_world()
    nerf = NeRF(
        3,
        n_hidden=64,
        n_encodings=2**14,
        n_levels=32,
        min_res=16,
        max_res=256,
        is_hdr=False,
    ).to(dev)

    train_scene, test_scene = scene.split(0.85)

    for v in test_scene.views:
        v.K[0, 0] /= 2
        v.K[1, 1] /= 2
        v.K[0, 2] = 512 / 2
        v.K[1, 2] = 512 / 2
        v.image_spatial_shape = (512, 512)

    # train_scene.render_world()
    # test_scene.render_world()

    train_time = 60 * 3
    train(
        nerf,
        train_scene,
        test_scene,
        dev,
        batch_size=2**16,
        n_samples=40,
        lr=1e-2,
        max_train_secs=train_time,
    )
    render_test_scenes(
        nerf,
        test_scene,
        dev,
        batch_size=2**16,
        n_samples=100,
        t_elapsed=train_time,
        show=True,
    )
