import json
from pathlib import Path
from typing import Optional


import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


from .cameras import Camera, CameraBatch


def load_scene_from_json(
    path: str,
    load_images: bool = True,
    device: torch.device = None,
    dtype=torch.float32,
) -> tuple[CameraBatch, torch.Tensor, Optional[torch.Tensor]]:
    """Loads scene information from nvidia compatible transforms.json"""

    path = Path(path)
    assert path.is_file()
    with open(path, "r") as f:
        data = json.load(f)

    aabb = (
        torch.stack(
            (
                -torch.ones((3,)) * data["aabb_scale"] * 0.5,
                torch.ones((3,)) * data["aabb_scale"] * 0.5,
            ),
            0,
        )
        .to(device)
        .to(dtype)
    )

    camera_common_kwargs = {
        "fx": data["fl_x"],
        "fy": data["fl_y"],
        "cx": data["cx"],
        "cy": data["cy"],
        "width": data["w"],
        "height": data["h"],
    }

    cams = []
    images = []
    for frame in data["frames"]:
        # Handle image
        if load_images:
            imgpath = path.parent / frame["file_path"]
            if not imgpath.is_file():
                print(f"Skipping {str(imgpath)}, image not found.")
                continue
            img = Image.open(imgpath).convert("RGBA")
            img = torch.tensor(np.asarray(img)).to(dtype).permute(2, 0, 1) / 255.0
            images.append(img)

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
        cams.append(
            Camera(
                R=t[:3, :3],
                T=t[:3, 3],
                tnear=0.0,
                tfar=10.0,
                **camera_common_kwargs,
            )
        )

    cams = CameraBatch(cams).to(device)
    if load_images:
        images = torch.stack(images, 0).to(device)
    else:
        images = None
    return cams, aabb, images


if __name__ == "__main__":
    cams, aabb, images = load_scene_from_json("data/suzanne/transforms.json")
