from operator import index
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import json
import numpy as np
from pathlib import Path
from PIL import Image

from . import cameras, linalg, rays, radiance


class View(torch.nn.Module):
    # Uses HWC format!
    def __init__(
        self,
        K: torch.Tensor,
        t_world_cam: torch.Tensor,
        image_spatial_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        assert len(image_spatial_shape) == 2, "Only (H,W) required"
        self.register_buffer("K", K)
        self.register_buffer("t_world_cam", t_world_cam)
        self.K: torch.Tensor
        self.t_world_cam: torch.Tensor
        self.image_spatial_shape = image_spatial_shape

    def view_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns rays in view local frame.

        Returns:
            o: (1,1,3) origins ray directions
            d: (H,W,3) normalized ray directions
        """
        o = self.K.new_zeros(3)[None, None, :]
        d = cameras.image_points(self.K, self.image_spatial_shape)
        d = F.normalize(d, p=2, dim=-1)
        return o, d

    def world_rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns rays in world frame.

        Returns:
            o: (1,1,3) origins ray directions
            d: (H,W,3) normalized ray directions
        """
        o, d = self.view_rays()
        o = (linalg.hom(o, 1) @ self.t_world_cam.T)[..., :3]
        d = (linalg.hom(d, 0) @ self.t_world_cam.T)[..., :3]
        return o, d

    @torch.no_grad()
    def render_image(
        self,
        radiance_field: radiance.RadianceField,
        radiance_aabb_minc: torch.Tensor,
        radiance_aabb_maxc: torch.Tensor,
        batch_size: int = 2**16,
        n_samples: int = 100,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        """Performs a volumentric rendering of this perspective.

        The rendering is performed by integrating color values
        along each ray. The returned image will be RGBA, where
        RGB is integrated and A is defined as (1.0 - ttfar),
        with ttfar being the probability of being transmissive
        at ray/radiance field exit.

        Returns:
            out: (4,H,W) RGBA image in [0..1] range. Note, we have C
                in the last dimension.
        """

        # Compute world rays
        origins, dirs = self.world_rays()
        origins = origins.expand_as(dirs).view(-1, 3)
        dirs = dirs.view(-1, 3)

        # Find limits of intersection with radiance field bounds
        tnear, tfar = rays.intersect_rays_aabb(
            origins, dirs, radiance_aabb_minc, radiance_aabb_maxc
        )

        # Determine which rays intersect the radiance field and limit
        # computation to those
        hit = tnear < tfar
        tnear = tnear[hit]
        tfar = tfar[hit]
        origins = origins[hit]
        dirs = dirs[hit]

        n_hit = dirs.shape[0]
        if out is None:
            out = self.K.new_zeros(self.image_spatial_shape + (4,))  # RGBA image

        # Reconstruct image in parts
        color_parts = []
        alpha_parts = []
        for batch_ids in torch.arange(n_hit, device=self.K.device).split(batch_size):
            # Sample T ray positions uniform randomly
            ts = rays.sample_rays_uniformly(
                tnear[batch_ids], tfar[batch_ids], n_samples
            )
            # Evaluate the rays (B,T,3)
            xyz = origins[batch_ids, None] + ts[..., None] * dirs[batch_ids, None]
            # Normalize range of world points to [-1,+1]
            nxyz = rays.normalize_xyz_in_aabb(
                xyz, radiance_aabb_minc, radiance_aabb_maxc
            )

            pred_sample_sigma, pred_sample_color = radiance_field(nxyz.view(-1, 3))
            pred_colors, pred_transm, _ = radiance.integrate_path(
                pred_sample_color.view(-1, n_samples, 3),
                pred_sample_sigma.view(-1, n_samples),
                ts,
                tfar[batch_ids],
            )
            color_parts.append(pred_colors)
            # TODO: not quite correct- should be 1.0 - T(i)*alpha(i)
            alpha_parts.append(1.0 - pred_transm[:, -1])

        colors = torch.cat(color_parts, 0)
        alphas = torch.cat(alpha_parts, 0)

        outflat = out.view(-1, 4)
        outflat[hit, :3] = colors
        outflat[hit, 3] = alphas

        return out.permute(2, 0, 1)

    def plot_image(
        self,
        img: torch.Tensor,
        use_alpha: bool = True,
        native_size: bool = False,
        checkerboard_bg: bool = True,
        ax=None,
    ):
        import matplotlib.pyplot as plt

        H, W = img.shape[-2:]
        C = 4 if use_alpha else 3

        if ax is None:
            figsize = plt.figaspect(H / W)  # uses mpl.rcParams['figure.figsize'][1]
            dpi = W // figsize[0] if native_size else None
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            if native_size:
                fig.subplots_adjust(0, 0, 1, 1)
        else:
            fig = plt.gcf()
        if use_alpha and checkerboard_bg:
            ax.imshow(self._checkerboard((H, W), k=max(H, W) // 100), cmap="gray")
        ax.imshow(img[:C].permute(1, 2, 0).cpu())

        return fig, ax

    def _checkerboard(self, shape: tuple[int, ...], k: int) -> torch.Tensor:
        """See https://stackoverflow.com/questions/72874737"""
        # nearest h,w multiple of k
        H = shape[0] + shape[0] % k
        W = shape[1] + shape[1] % k
        indices = torch.stack(
            torch.meshgrid(torch.arange(H // k), torch.arange(W // k), indexing="ij")
        )
        base = indices.sum(dim=0) % 2
        x = base.repeat_interleave(k, 0).repeat_interleave(k, 1)
        return x[: shape[0], : shape[1]]


class MultiViewScene:
    def __init__(self) -> None:
        self.views: list[View] = []
        self.images: list[torch.Tensor] = []
        self.image_alphas: list[torch.Tensor] = []
        self.aabb_minc: torch.Tensor = -torch.ones((3,))
        self.aabb_maxc: torch.Tensor = torch.ones((3,))

    def add_view(self, v: View):
        self.views.append(v)

    def split(self, ratio: float) -> tuple["MultiViewScene", "MultiViewScene"]:
        """Split scene views for generating train/val datasets"""
        ids = torch.randperm(len(self.views))
        left_part = int(len(self.views) * ratio)
        if left_part == len(self.views):
            left_part -= 1
        left = MultiViewScene()
        left.aabb_minc = self.aabb_minc.clone()
        left.aabb_maxc = self.aabb_maxc.clone()
        left.images = [self.images[i] for i in ids[:left_part]]
        left.image_alphas = [self.image_alphas[i] for i in ids[:left_part]]
        left.views = [self.views[i] for i in ids[:left_part]]

        right = MultiViewScene()
        right.aabb_minc = self.aabb_minc.clone()
        right.aabb_maxc = self.aabb_maxc.clone()
        right.images = [self.images[i] for i in ids[left_part:]]
        right.image_alphas = [self.image_alphas[i] for i in ids[left_part:]]
        right.views = [self.views[i] for i in ids[left_part:]]

        return left, right

    def load_from_json(self, path: str, device: torch.device = None):
        """Loads scene information from nvidia compatible transforms.json"""
        self.views.clear()
        self.images.clear()
        self.image_alphas.clear()

        path = Path(path)
        assert path.is_file()
        with open(path, "r") as f:
            data = json.load(f)

        K = torch.eye(3)
        K[0, 0] = data["fl_x"]
        K[1, 1] = data["fl_y"]
        K[0, 2] = data["cx"]
        K[1, 2] = data["cy"]

        self.aabb_minc = -torch.ones((3,)) * data["aabb_scale"] * 0.5
        self.aabb_maxc = torch.ones((3,)) * data["aabb_scale"] * 0.5

        for frame in data["frames"]:
            # Handle image
            imgpath = path.parent / frame["file_path"]
            if not imgpath.is_file():
                print(f"Skipping {str(imgpath)}, image not found.")
                continue
            img = Image.open(imgpath)
            img = torch.tensor(np.asarray(img)).float().permute(2, 0, 1) / 255.0
            C, H, W = img.shape
            if C == 4:
                alpha = img[-1]
            else:
                alpha = img.new_ones((H, W), dtype=float)
            self.images.append(img[:3].contiguous())
            self.image_alphas.append(alpha.contiguous())

            # Correct non-orthonormal rotations. That's the case for some
            # frames of the fox-dataset.
            t = torch.tensor(frame["transform_matrix"]).to(torch.float64)
            if (torch.det(t[:3, :3]) - 1.0) > 1e-6:
                print(f"Correcting rotation matrix for {str(imgpath)}")
                res = torch.svd(t[:3, :3])
                u, s, v = res
                u = F.normalize(u, p=2, dim=0)
                v = F.normalize(v, p=2, dim=0)
                s[:] = 1.0
                rot = u @ torch.diag(s) @ v.T
                t[:3, :3] = rot

            # Handle extrinsics (convert from OpenGL to OpenCV camera
            # model: i.e look towards positive z)
            flip = torch.eye(4)
            flip[1, 1] = -1
            flip[2, 2] = -1

            t = t.float() @ flip

            view = View(K.clone(), t, (H, W)).to(device)
            self.views.append(view)

    def render_world(self):
        import matplotlib.pyplot as plt
        import pytransform3d.camera as pc
        import pytransform3d.transformations as pt
        from pytransform3d.plot_utils import plot_box

        ax = pt.plot_transform(s=0.5)
        for v in self.views:

            ax = pt.plot_transform(A2B=v.t_world_cam.cpu().numpy(), s=0.5)
            pc.plot_camera(
                ax,
                cam2world=v.t_world_cam.cpu().numpy(),
                M=v.K.cpu().numpy(),
                sensor_size=v.image_spatial_shape[::-1],
                virtual_image_distance=1.0,
                linewidth=0.25,
            )
        plot_box(ax, (self.aabb_maxc - self.aabb_minc).numpy())
        ax.set_aspect("equal")
        ax.relim()  # make sure all the data fits
        ax.autoscale()
        plt.show()

    def __repr__(self):
        return (
            f"MultiViewScene(nviews={len(self.images)}, shape={self.images[0].shape})"
        )


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, scene: MultiViewScene, view_indices: list[int] = None):
        self.scene = scene
        self._prepare_elements(scene, view_indices)

    def _prepare_elements(self, scene: MultiViewScene, view_indices: list[int]):
        if view_indices is None:
            view_indices = list(range(len(scene.views)))

        images = [scene.images[i] for i in view_indices]
        alphas = [scene.image_alphas[i] for i in view_indices]
        images = torch.stack(images, 0).permute(0, 2, 3, 1).reshape(-1, 3)
        alphas = torch.stack(alphas, 0).view(-1)
        origin, dir, tnear, tfar = [], [], [], []
        for idx in view_indices:
            v = scene.views[idx]
            o, d = v.world_rays()
            o = o.expand_as(d)
            d = d.view(-1, 3)
            o = o.view(-1, 3)
            tn, tf = rays.intersect_rays_aabb(
                o,
                d,
                scene.aabb_minc,
                scene.aabb_maxc,
            )
            origin.append(o)
            dir.append(d)
            tnear.append(tn)
            tfar.append(tf)
        tnear = torch.cat(tnear, 0)
        tfar = torch.cat(tfar, 0)
        origin = torch.cat(origin, 0)
        dir = torch.cat(dir, 0)

        # Select relevant
        self.thit = tnear < tfar
        self.images = images[self.thit].contiguous().float()
        self.alphas = alphas[self.thit].contiguous().float()
        self.origin = origin[self.thit].contiguous()
        self.dir = dir[self.thit].contiguous()
        self.tfar = tfar[self.thit].contiguous()
        self.tnear = tnear[self.thit].contiguous()

    def __getitem__(self, idx):
        """Return elements for all indices in idx."""
        return (
            self.images[idx],
            self.alphas[idx],
            self.origin[idx],
            self.dir[idx],
            self.tnear[idx],
            self.tfar[idx],
        )

    def __len__(self):
        return self.images.shape[0]

    def create_mini_batch_sampler(
        self, batch_size: int, random: bool = True
    ) -> torch.utils.data.Sampler:
        """Returns a sampler that queries a mini-batch of indices
        from the dataset at once.

        When used in conjuncation with a dataloader the resulting
        dimensions will be (B,M,...) where B is the batch size of
        the dataload and M is the mini-batch size.
        """

        sampler = (
            torch.utils.data.sampler.RandomSampler(self)
            if random
            else torch.utils.data.sampler.SequentialSampler(self)
        )
        return torch.utils.data.sampler.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True,
        )


if __name__ == "__main__":
    scene = MultiViewScene()
    scene.load_from_json("data/fox/transforms.json")
    scene.render_world()

    # # scene.render_setup()
    # scene.compute_train_info()
    # print(scene)

    # v = View(scene.K, scene.t_world_cam[0], scene.images[0].shape[1:])
    # o, d = v.world_rays()
    # print(o, v.t_world_cam[:3, 3])
    # print(d.shape)
    # print(repr(v.K))
    # print(repr(v.t_world_cam))

    # v = View(K,T,image_shape(...))
    # o,d = v.world_rays()
    # o,d = v.view_rays()
    # v.render_image(...) -> img
    # v.save_image(...)

    # tnear, tfar, valid = scene.intersect_rays_aabb(o,d)
    # ts = ray.sample(tnear,tfar,n_samples)
    # xyz = ray.evaluate(o,d,ts)

    # v.to_world(o,d)

    # v.tnear
    # v.tfar
