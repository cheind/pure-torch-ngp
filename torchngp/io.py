import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .cameras import MultiViewCamera

_logger = logging.getLogger(__name__)


def load_scene_from_json(
    path: str,
    load_images: bool = True,
    device: torch.device = None,
    dtype=torch.float32,
) -> tuple[MultiViewCamera, torch.Tensor, Optional[torch.Tensor]]:
    """Loads scene information from nvidia compatible transforms.json

    Params:
        path: path to `transforms.json` file
        load_images: whether we should load ground-truth images
        device: return data on this device
        dtype: return data using this dtype

    Returns:
        camera: multi-view camera parameters
        aabb: (2,3) tensor containing min/max corner of world aabb
        images: (N,H,W,4) optional RGBA images normalized to [0..1] range

    See:
        https://github.com/NVlabs/instant-ngp/blob/54aba7cfbeaf6a60f29469a9938485bebeba24c3/docs/nerf_dataset_tips.md
        https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi
        https://github.com/bmild/nerf#generating-poses-for-your-own-scenes
    """

    path = Path(path)
    assert path.is_file()
    with open(path, "r") as f:
        data = json.load(f)

    scale = data.get("scale", 0.33)
    aabb_scale = data.get("aabb_scale", 1.0)
    offset = torch.tensor(data.get("offset", 0.5)).to(device).to(dtype)

    if "aabb" not in data:
        aabb = (
            torch.stack(
                (
                    -torch.ones((3,)) * (aabb_scale / scale) * 0.5,
                    torch.ones((3,)) * (aabb_scale / scale) * 0.5,
                ),
                0,
            )
            .to(device)
            .to(dtype)
        ) + offset
    else:
        aabb = torch.tensor(data["aabb"]).to(device).to(dtype)

    print("AABB", aabb)

    Rs = []
    Ts = []
    view_images = []
    for frame in data["frames"]:
        # Handle image
        if load_images:
            imgpath = path.parent / frame["file_path"]
            if not imgpath.is_file():
                print(f"Skipping {str(imgpath)}, image not found.")
                continue
            img = Image.open(imgpath).convert("RGBA")
            img = torch.tensor(np.asarray(img)).to(dtype).permute(2, 0, 1) / 255.0
            view_images.append(img)

        # Handle extrinsics. Correct non-orthonormal rotations.
        # That's the case for some frames of the fox-dataset.
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

        # Cconvert from OpenGL to OpenCV camera
        # model: i.e look towards positive z
        flip = torch.eye(4, dtype=torch.float64)
        flip[1, 1] = -1
        flip[2, 2] = -1

        t = (t @ flip).to(dtype)
        Rs.append(t[:3, :3])
        Ts.append(t[:3, 3])

    camera = MultiViewCamera(
        focal_length=[data["fl_x"], data["fl_y"]],
        principal_point=[data["cx"], data["cy"]],
        size=[data["w"], data["h"]],
        tnear=0.0,
        tfar=10,
        R=Rs,
        T=Ts,
    )

    images = None
    if load_images:
        images = torch.stack(view_images, 0).to(device)

    return camera, aabb, images


if __name__ == "__main__":
    camera, aabb, images = load_scene_from_json("data/suzanne/transforms.json")
    camera, aabb, images = load_scene_from_json("data/fox/transforms.json")
