from re import S
from typing import Callable
import torch
import torch.nn
import torch.nn.functional as F
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
            out: (H,W,4) RGBA image in [0..1] range. Note, we have C
                in the last dimension.
        """

        # Compute world rays
        origins, dirs = self.world_rays()
        origins = origins.view(-1, 3)
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
            alpha_parts.append(1.0 - pred_transm[:, -1])

        colors = torch.cat(color_parts, 0)
        alphas = torch.cat(alpha_parts, 0)

        outflat = out.view(-1, 4)
        outflat[hit, :3] = colors
        outflat[hit, 3] = alphas

        return out


class MultiViewScene:
    def __init__(self) -> None:
        self.images: list[torch.Tensor] = []
        self.image_masks: list[torch.Tensor] = []
        self.K = torch.eye(3)
        self.t_world_cam: list[torch.Tensor] = []
        self.aabb_minc: torch.Tensor = -torch.ones((3,))
        self.aabb_maxc: torch.Tensor = torch.ones((3,))

    def load_from_json(self, path: str):
        """Loads scene information from nvidia compatible transforms.json"""
        path = Path(path)
        assert path.is_file()
        with open(path, "r") as f:
            data = json.load(f)

        self.K[0, 0] = data["fl_x"]
        self.K[1, 1] = data["fl_y"]
        self.K[0, 2] = data["cx"]
        self.K[1, 2] = data["cy"]

        self.aabb_minc = -torch.ones((3,)) * data["aabb_scale"] * 0.5
        self.aabb_maxc = torch.ones((3,)) * data["aabb_scale"] * 0.5

        for frame in data["frames"]:
            # Handle image
            imgpath = path.parent / frame["file_path"]
            if not imgpath.is_file():
                print(f"Skipping {str(imgpath)}")
                continue
            assert imgpath.is_file()
            img = Image.open(imgpath)
            img = torch.tensor(np.asarray(img)).float().permute(2, 0, 1) / 255.0
            C, H, W = img.shape
            if C == 4:
                mask = img[-1] > 0.0
            else:
                mask = img.new_ones((H, W), dtype=bool)
            self.images.append(img[:3].contiguous())
            self.image_masks.append(mask.clone())

            # Handle extrinsics (convert from OpenGL to OpenCV camera
            # model: i.e look towards positive z)
            flip = torch.eye(4)
            flip[1, 1] = -1
            flip[2, 2] = -1
            t = torch.tensor(frame["transform_matrix"])
            t = t @ flip
            self.t_world_cam.append(t)

    def compute_train_info(self) -> dict[str, torch.Tensor]:
        """Returns world rays + color information for all views.

        Returns:
            A dictionary with the following keys
                K: (3,3) camera intrinsics
                extrinsics: (B,4,4) cam in world matrices
                origins: (B,H,W,3) camera origins in world
                directions: (B,H,W,3) normalized ray directions in world
                tnear: (B,H,W) ray tnear
                tfar: (B,H,W) ray tfar
                colors: (B,H,W,C) color images
                masks: (B,H,W) color alpha masks
        """
        ret = {}
        ret["colors"] = torch.stack(self.images, 0).permute(0, 2, 3, 1)
        ret["masks"] = torch.stack(self.image_masks, 0)
        ret["extrinsics"] = torch.stack(self.t_world_cam, 0)
        N, H, W, C = ret["colors"].shape

        ret["origins"] = (
            ret["extrinsics"][:, :3, 3].unsqueeze(1).unsqueeze(1).tile(1, H, W, 1)
        )

        # TODO: do this batched
        dirs = []
        cpts = cameras.image_points(self.K, self.images[0].shape[1:])
        crays = F.normalize(cpts, p=2, dim=-1)
        for i in range(N):
            # note, dehom here leads to infs
            dirs.append((linalg.hom(crays, 0) @ self.t_world_cam[i].T)[..., :3])
        dirs = torch.stack(dirs, 0)
        ret["directions"] = dirs

        tnear, tfar = rays.intersect_rays_aabb(
            ret["origins"].view(-1, 3),
            ret["directions"].view(-1, 3),
            self.aabb_minc,
            self.aabb_maxc,
        )

        ret["tnear"] = tnear.view(N, H, W)
        ret["tfar"] = tfar.view(N, H, W)

        # should match:
        # print(rays[0, H // 2, W // 2], self.t_world_cam[0][:3, 2])

        return ret

    def render_setup(self):
        import matplotlib.pyplot as plt
        import pytransform3d.camera as pc
        import pytransform3d.transformations as pt
        from pytransform3d.plot_utils import plot_box

        ax = pt.plot_transform(s=0.5)
        for i in range(len(self.images)):
            H, W = self.images[i].shape[1:]

            ax = pt.plot_transform(A2B=self.t_world_cam[i].numpy(), s=0.5)
            pc.plot_camera(
                ax,
                cam2world=self.t_world_cam[i].numpy(),
                M=self.K.numpy(),
                sensor_size=(W, H),
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


if __name__ == "__main__":
    scene = MultiViewScene()
    scene.load_from_json("data/suzanne/transforms.json")
    # # scene.render_setup()
    # scene.compute_train_info()
    # print(scene)

    v = View(scene.K, scene.t_world_cam[0], scene.images[0].shape[1:])
    o, d = v.world_rays()
    print(o, v.t_world_cam[:3, 3])
    print(d.shape)
    print(repr(v.K))
    print(repr(v.t_world_cam))

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
