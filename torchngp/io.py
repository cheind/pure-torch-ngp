import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from . import modules

_logger = logging.getLogger("torchngp")


def _load_json(path: str) -> dict:
    path = Path(path)
    assert path.is_file(), f"Path {path} does not exist."
    with open(path, "r") as f:
        data = json.load(f)
    return data


def cam_from_json(path: str, slice: str = None) -> modules.MultiViewCamera:
    """Loads camera configuration information from a transforms.json file.

    Params:
        path: path to `transforms.json` file

    Returns:
        cam: camera

    See:
        https://github.com/NVlabs/instant-ngp/blob/54aba7cfbeaf6a60f29469a9938485bebeba24c3/docs/nerf_dataset_tips.md
        https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi
        https://github.com/bmild/nerf#generating-poses-for-your-own-scenes
    """
    path = Path(path)
    data = _load_json(path)

    poses = []
    image_paths = []
    n_skipped = 0
    n_fixed = 0
    for frame in data["frames"]:
        # Check for image
        imgpath = path.parent / frame["file_path"]
        if imgpath.suffix == "":
            # Original nerf does not specify image suffix
            imgpath = imgpath.with_suffix(".png")
        if not imgpath.is_file():
            _logger.debug(f"Failed to find {str(imgpath)}, skipping.")
            n_skipped += 1
            continue
        image_paths.append(imgpath.as_posix())

        # Handle extrinsics. Correct non-orthonormal rotations.
        # That's the case for some frames of the fox-dataset.
        t = torch.tensor(frame["transform_matrix"], dtype=torch.float64)
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

        t = t @ flip
        poses.append(t)

    # Handle camera params
    if "h" not in data or "w" not in data:
        assert len(image_paths) > 0, "Failed to deduce camera size."
        img = Image.open(image_paths[0])
        H, W = img.height, img.width
        del img
    else:
        H, W = int(data.get("h", 0)), int(data.get("w", 0))

    if "fl_x" not in data or "fl_y" not in data:
        assert "camera_angle_x" in data, "Failed to deduce camera focal length."
        fl_x = 0.5 * W / math.tan(0.5 * data["camera_angle_x"])
        fl_y = fl_x  # square pixels
    else:
        fl_x, fl_y = data["fl_x"], data["fl_y"]
    if "cx" not in data or "cy" not in data:
        # TODO: why does W * 0.5 yield a higher PSNR?
        # especially on blender datasets.
        cx = (W + 1) * 0.5
        cy = (H + 1) * 0.5
    else:
        cx = data["cx"]
        cy = data["cy"]

    poses = torch.stack(poses, 0)
    image_paths = np.array(image_paths, dtype=object)  # to support advanced indexing

    if slice is not None:
        poses = eval("poses[" + slice + "]")
        image_paths = eval("image_paths[" + slice + "]")

    _logger.info(
        f"Imported {len(image_paths)} poses from '{str(path)}', skipped"
        f" {n_skipped} poses and fixed {n_fixed} poses."
    )

    return modules.MultiViewCamera(
        focal_length=(fl_x, fl_y),
        principal_point=(cx, cy),
        size=(W, H),
        poses=poses,
        image_paths=image_paths.tolist(),
    )


def aabb_from_json(path: str) -> torch.Tensor:
    """Loads AABB configuration information from a transforms.json file.

    Params:
        path: path to `transforms.json` file

    Returns:
        aabb: min/max corner of box

    See:
        https://github.com/NVlabs/instant-ngp/blob/54aba7cfbeaf6a60f29469a9938485bebeba24c3/docs/nerf_dataset_tips.md
        https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi
        https://github.com/bmild/nerf#generating-poses-for-your-own-scenes
    """
    path = Path(path)
    data = _load_json(path)

    scale = data.get("scale", 0.4)
    aabb_scale = data.get("aabb_scale", 1.0)
    offset = data.get("offset", 0.0)

    if "aabb" not in data:
        _logger.debug(
            "Key 'aabb' not found in transforms.json file. Assuming legacy NeRF format"
        )
        aabb = (
            torch.stack(
                (
                    -torch.ones((3,)) * (aabb_scale / scale) * 0.5,
                    torch.ones((3,)) * (aabb_scale / scale) * 0.5,
                ),
                0,
            )
        ) + torch.as_tensor(offset)
    else:
        aabb = torch.tensor(data["aabb"])

    with np.printoptions(precision=3):
        _logger.info(
            f"Imported bounds {aabb[0].numpy()}, {aabb[1].numpy()} from '{str(path)}'."
        )
    return aabb


AabbFromJsonConf = config.build_conf(aabb_from_json)
CamFromJsonConf = config.build_conf(cam_from_json)
