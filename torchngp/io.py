import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import geometric
from . import functional

_logger = logging.getLogger(__name__)


def load_scene_from_json(
    path: str,
    load_images: bool = True,
    device: torch.device = None,
    dtype=torch.float32,
) -> tuple[geometric.MultiViewCamera, torch.Tensor, Optional[torch.Tensor]]:
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
        _logger.debug(
            "Key 'aabb' not found. Assuming legacy NeRF format with origin at 0.5 and"
            " side length 1."
        )
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

    Rs = []
    Ts = []
    view_images = []
    image_paths = []
    n_skipped = 0
    n_fixed = 0
    for frame in data["frames"]:
        # Handle image
        if load_images:
            imgpath = path.parent / frame["file_path"]
            if imgpath.suffix == "":
                # Original nerf does not specify image suffix
                imgpath = imgpath.with_suffix(".png")
            if not imgpath.is_file():
                _logger.debug(f"Failed to find {str(imgpath)}, skipping.")
                n_skipped += 1
                continue

            img = Image.open(imgpath).convert("RGBA")
            img = torch.tensor(np.asarray(img)).to(dtype).permute(2, 0, 1) / 255.0
            view_images.append(img)
            image_paths.append(frame["file_path"])

        # Handle extrinsics. Correct non-orthonormal rotations.
        # That's the case for some frames of the fox-dataset.
        t = torch.tensor(frame["transform_matrix"]).to(torch.float64)
        if (torch.det(t[:3, :3]) - 1.0) > 1e-5:
            _logger.debug(f"Pose for {str(imgpath)} not ortho-normalized, correcting.")
            n_fixed += 1
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

    H, W = view_images[0].shape[-2:]
    if "fl_x" not in data or "fl_y" not in data:
        assert "camera_angle_x" in data
        fl_x = 0.5 * W / math.tan(0.5 * data["camera_angle_x"])
        fl_y = fl_x  # square pixels
    else:
        fl_x, fl_y = data["fl_x"], data["fl_y"]
    if "cx" not in data or "cy" not in data:
        cx = (W + 1) * 0.5
        cy = (H + 1) * 0.5
    else:
        cx = data["cx"]
        cy = data["cy"]

    camera = geometric.MultiViewCamera(
        focal_length=[fl_x, fl_y],
        principal_point=[cx, cy],
        size=[W, H],
        tnear=0.0,
        tfar=20,
        rvec=functional.so3_log(torch.stack(Rs)),
        tvec=torch.stack(Ts, 0),
        image_paths=image_paths,
    )

    images = None
    if load_images:
        images = torch.stack(view_images, 0).to(device)

    _logger.debug(
        f"Imported {camera.n_views} poses from '{str(path)}', skipped"
        f" {n_skipped} poses and fixed {n_fixed} poses. Bounds set to {aabb}."
    )

    return camera, aabb, images


class MultiViewScene(torch.nn.Module):
    def __init__(
        self, path: Path, load_images: bool = True, slice: Optional[str] = None
    ) -> None:
        super().__init__()
        camera, aabb, images = load_scene_from_json(path, load_images=load_images)
        if slice is not None:
            s = _string_to_slice(slice)
            camera = camera[s]
            if images is not None:
                images = images[s]

        self.camera = camera
        self.register_buffer("images", images)
        self.register_buffer("aabb", aabb)
        self.images: Optional[torch.Tensor]
        self.aabb: torch.Tensor


def _string_to_slice(sstr):
    # https://stackoverflow.com/questions/43089907/using-a-string-to-define-numpy-array-slice
    return tuple(
        slice(*(int(i) if i else None for i in part.strip().split(":")))
        for part in sstr.strip("[]").split(",")
    )
