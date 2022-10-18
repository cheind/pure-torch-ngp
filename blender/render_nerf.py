import bpy
import numpy as np
from mathutils import Vector, Euler

CAMERA_COLLECTION_NAME = "NERF Cameras"


def add_cameras(origins, target, reference_camera):
    cam_collection = bpy.data.collections.new(CAMERA_COLLECTION_NAME)
    bpy.context.scene.collection.children.link(cam_collection)
    for idx, origin in enumerate(origins):
        camera_obj = reference_camera.copy()
        camera_obj.data = camera_obj.data.copy()
        camera_obj.name = f"NERF Camera.{idx+1:0>3}"
        camera_obj.location = Vector(origin)

        ttc = camera_obj.constraints.new(type="TRACK_TO")
        ttc.target = target
        ttc.track_axis = "TRACK_NEGATIVE_Z"
        ttc.up_axis = "UP_Y"

        bpy.ops.object.select_all(action="DESELECT")
        cam_collection.objects.link(camera_obj)
        camera_obj.select_set(True)
        bpy.ops.object.visual_transform_apply()
    return cam_collection


def equidistant_points_on_sphere(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.stack((x, y, z), -1)


def add_timeline_markers(camera_collection):
    for idx, cam in enumerate(camera_collection.objects):
        marker_name = f"M_{cam.name}"
        marker = bpy.context.scene.timeline_markers.new(marker_name, frame=idx + 1)
        marker.camera = cam
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(camera_collection.objects)


def get_intrinsic_params(camera):
    camdata = camera.data

    def get_sensor_size(sensor_fit, sensor_x, sensor_y):
        if sensor_fit == "VERTICAL":
            return sensor_y
        return sensor_x

    def get_sensor_fit(sensor_fit, size_x, size_y):
        if sensor_fit == "AUTO":
            if size_x >= size_y:
                return "HORIZONTAL"
            else:
                return "VERTICAL"
        return sensor_fit

    if camdata.type != "PERSP":
        raise ValueError("Non-perspective cameras not supported")

    f_mm = camdata.lens
    scale = bpy.context.scene.render.resolution_percentage * 1e-2
    resolution_x_in_px = scale * bpy.context.scene.render.resolution_x
    resolution_y_in_px = scale * bpy.context.scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(
        camdata.sensor_fit, camdata.sensor_width, camdata.sensor_height
    )
    sensor_fit = get_sensor_fit(
        camdata.sensor_fit,
        bpy.context.scene.render.pixel_aspect_x * resolution_x_in_px,
        bpy.context.scene.render.pixel_aspect_y * resolution_y_in_px,
    )

    pixel_aspect_ratio = (
        bpy.context.scene.render.pixel_aspect_y
        / bpy.context.scene.render.pixel_aspect_x
    )
    if sensor_fit == "HORIZONTAL":
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camdata.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camdata.shift_y * view_fac_in_px / pixel_aspect_ratio

    K = np.array([[s_u, 0.0, u_0], [0.0, s_v, v_0], [0.0, 0.0, 1.0]])
    size = (resolution_x_in_px, resolution_y_in_px)
    fov = (
        2 * np.arctan(size[0] / (2 * K[0, 0])),
        2 * np.arctan(size[1] / (2 * K[1, 1])),
    )

    return K, size, fov


def get_extrinsic_matrix(camera, to_cv=True):
    cm = camera.matrix_world.copy()
    if to_cv:
        cm = cm @ Euler((-np.pi, 0, 0), "XYZ").to_matrix().to_4x4()
    return np.asarray(cm)


def bbox(obj):
    crns = np.array([b for b in obj.bound_box])
    return crns.min(0), crns.max(0)


def after_render_complete(scene):
    from pathlib import Path
    import json

    camera_collection = scene.collection.children[CAMERA_COLLECTION_NAME]

    # instant-ngp only supports a single camera intrinsics
    K, size, fov = get_intrinsic_params(camera_collection.objects[0])

    data = {
        "camera_angle_x": fov[0],
        "camera_angle_y": fov[1],
        "fl_x": K[0, 0],
        "fl_y": K[1, 1],
        "cx": K[0, 2],
        "cy": K[1, 2],
        "w": size[0],
        "h": size[1],
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "aabb_scale": 4.0,
    }

    outdir = None
    frames = []
    for idx, cam in enumerate(camera_collection.objects):
        path = bpy.context.scene.render.frame_path(frame=idx + 1)
        path = Path(path)
        frame = {
            "file_path": path.name,
            "sharpness": 30.0,  # should be the variance of the Laplacian
            "transform_matrix": get_extrinsic_matrix(cam, to_cv=False).tolist(),
        }
        frames.append(frame)
        outdir = Path(path).parent

    data["frames"] = frames
    outpath = outdir / "transforms.json"
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {str(outpath)}")


def main():
    import sys
    import argparse

    if "--" in sys.argv:
        script_args = sys.argv[sys.argv.index("--") + 1 :]
    else:
        script_args = []

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--target", help="Target object name to focus on", default="Cube"
    )
    parser.add_argument(
        "-c",
        "--camera",
        help="Reference camera name to copy lens properties from",
        default="Camera",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        help="Radial distance of cameras from origin",
        default=None,
    )
    parser.add_argument(
        "-n", "--numviews", type=int, help="Number of views to generate", default=20
    )
    args = parser.parse_args(script_args)

    if args.target not in bpy.data.objects:
        raise ValueError(f"Target {args.target} not found in scene")
    target = bpy.data.objects[args.target]
    minc, maxc = bbox(target)

    if args.camera not in bpy.data.objects:
        raise ValueError(f"Camera {args.camera} not found in scene")
    camera = bpy.data.objects[args.camera]
    if not camera.type == "CAMERA":
        raise ValueError(f"Camera {args.camera} is not of type Camera")

    if args.radius is None:
        args.radius = np.linalg.norm(maxc - minc) * 2.0
        print(f"Setting radius to {args.radius}")

    center = (maxc + minc) * 0.5
    origins = equidistant_points_on_sphere(args.numviews) * args.radius + center
    camcoll = add_cameras(origins, target, camera)
    add_timeline_markers(camcoll)

    bpy.app.handlers.render_complete.append(after_render_complete)


if __name__ == "__main__":
    main()


# camera_origins = generate_points_on_sphere(10)
# print(len(camera_origins))
# print(bpy.data.scenes["Scene"].render.filepath)
# camera_collection = add_cameras(camera_origins*10.0, target_name='Suzanne')
# add_timeline_markers(camera_collection)

# note order of arguments!!
# blender nerf.blend -b -o c:\tmp1 --python render_nerf.py -- myarg=1
# with rendering
# blender nerf.blend -o c:\tmp\ -b --python render_nerf.py -a -- -t Suzanne -r 10
# without rendering
# blender nerf.blend -o c:\tmp\ -b --python render_nerf.py -- -t Suzanne -r 10
# blender nerf.blend -o c:\tmp\suzanne\image#### --python render_nerf.py -b -a -- -t Suzanne -n 20
