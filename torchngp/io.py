import json
import logging
import math
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image

from . import config, functional

_logger = logging.getLogger("torchngp")


def load_scene_from_json(
    path: Union[str, list[str]], pose_to_cv: bool = True
) -> config.SceneConf:
    """Loads scene information from one or more NeRF transforms.json files.

    Params:
        paths: path or dictionary of paths to `transforms.json` files
        pose_to_cv: Enable conversion from OpenGL to OpenCV camera frame.

    Returns:
        scenecfg: scene configuration

    See:
        https://github.com/NVlabs/instant-ngp/blob/54aba7cfbeaf6a60f29469a9938485bebeba24c3/docs/nerf_dataset_tips.md
        https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi
        https://github.com/bmild/nerf#generating-poses-for-your-own-scenes
    """
    if isinstance(path, (str, Path)):
        path = [path]

    cfgs = [_load_transform_json(p, pose_to_cv=pose_to_cv) for p in path]
    cams = []
    for c in cfgs:
        cams.extend(c.cameras)
    aabb = cfgs[0].aabb
    return config.SceneConf(cameras=cams, aabb=aabb)


def _load_transform_json(path: str, pose_to_cv: bool) -> config.SceneConf:

    path = Path(path)
    assert path.is_file(), f"Path {path} does not exist."
    with open(path, "r") as f:
        data = json.load(f)

    scale = data.get("scale", 0.33)
    aabb_scale = data.get("aabb_scale", 1.0)
    offset = data.get("offset", 0.0)

    if "aabb" not in data:
        _logger.debug(
            "Key 'aabb' not found in transforms.json file. Assuming legacy NeRF format"
            " with origin at 0.5 and side length 1."
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

    Rs = []
    Ts = []
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
        if pose_to_cv:
            flip = torch.eye(4, dtype=torch.float64)
            flip[1, 1] = -1
            flip[2, 2] = -1

            t = t @ flip
        Rs.append(t[:3, :3])
        Ts.append(t[:3, 3])

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
        cx = (W + 1) * 0.5
        cy = (H + 1) * 0.5
    else:
        cx = data["cx"]
        cy = data["cy"]

    rvec = functional.so3_log(torch.stack(Rs, 0))
    tvec = torch.stack(Ts, 0)

    camcfg = config.MultiViewCameraConf(
        focal_length=(fl_x, fl_y),
        principal_point=(cx, cy),
        size=(W, H),
        rvec=config.Vecs3Conf([tuple(r.tolist()) for r in rvec]),
        tvec=config.Vecs3Conf([tuple(t.tolist()) for t in tvec]),
        image_paths=image_paths,
    )

    aabbcfg = config.Vecs3Conf([tuple(aabb[0].tolist()), tuple(aabb[1].tolist())])

    _logger.debug(
        f"Imported {len(image_paths)} poses from '{str(path)}', skipped"
        f" {n_skipped} poses and fixed {n_fixed} poses. Bounds set to {aabb}."
    )

    return config.SceneConf(cameras=[camcfg], aabb=aabbcfg)
