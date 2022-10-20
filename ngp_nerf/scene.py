import torch
import torch.nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from PIL import Image

from . import cameras, linalg


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
            assert imgpath.is_file()
            img = Image.open(imgpath)
            img = torch.tensor(np.asarray(img)).float().permute(2, 0, 1)
            C, H, W = img.shape
            if C == 4:
                mask = img[-1] > 0.0
            else:
                mask = img.new_ones((H, W), dtype=bool)
            self.images.append(img[:3].contiguous())
            self.image_masks.append(mask)

            # Handle extrinsics (convert from OpenGL to OpenCV camera
            # model: i.e look towards positive z)
            flip = torch.eye(4)
            flip[1, 1] = -1
            flip[2, 2] = -1
            t = torch.tensor(frame["transform_matrix"])
            t = t @ flip
            self.t_world_cam.append(t)

    def compute_world_rays(self) -> dict[str, torch.Tensor]:
        """Returns world rays + color information for all views.

        Returns:
            A dictionary with the following keys
                K: (3,3) camera intrinsics
                extrinsics: (B,4,4) cam in world matrices
                origins: (B,3) camera origins in world
                directions: (B,H,W,3) normalized ray directions in world
                colors: (B,C,H,W) color images
                masks: (B,H,W) color alpha masks
        """
        ret = {}
        ret["extrinsics"] = torch.stack(self.t_world_cam, 0)
        ret["origins"] = ret["extrinsics"][:, :3, 3].contiguous()
        ret["colors"] = torch.stack(self.images, 0)
        ret["masks"] = torch.stack(self.image_masks, 0)

        N, C, H, W = ret["colors"].shape

        # TODO: do this batched
        rays = []
        cpts = cameras.image_points(self.K, self.images[0].shape[1:])
        crays = F.normalize(cpts, p=2, dim=-1)
        for i in range(N):
            # note, dehom here leads to infs
            rays.append((linalg.hom(crays, 0) @ self.t_world_cam[i].T)[..., :3])
        rays = torch.stack(rays, 0)
        ret["directions"] = rays

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
    # scene.render_setup()
    scene.compute_world_rays()
    print(scene)


# scene = load_nerf_scene("...")
# scene.imgs(N, H, W, 3)
# scene.img_masks(N, H, W, 1)
# scene.img_mean, img_std(
#     4,
# )
# scene.nimgs
# scene.K
# scene.extrinsics
# scene.aabb_minc
# scene.aabb_maxc
# scene.ray_origins(N, 3)
# scene.ray_dirs(N, H, W, 3)
# scene.ray_colors(N, H, W, 3)
