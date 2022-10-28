from itertools import cycle

import numpy as np

from vispy import app, scene, io
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform

canvas = scene.SceneCanvas(keys="interactive", size=(800, 600), show=True)
canvas.measure_fps()
view = canvas.central_widget.add_view()


# Read volume
data = np.load("tmp/volume.npz")
vol_density = data["d"]
vol_aabb = data["aabb"]


# Create the volume visuals, only one is visible
volume = scene.visuals.Volume(
    vol=vol_density,
    parent=view.scene,
    threshold=0.0,
    method="translucent",
    clim=(0.1, 2.0),
    interpolation="linear",
    relative_step_size=0.2,
)

# volume.transform = scene.STTransform(translate=-(vol_aabb[1] - vol_aabb[0]) * 0.5)

fov = 60.0
cam = scene.cameras.TurntableCamera(parent=view.scene, fov=fov, name="Arcball")
view.camera = cam  # Select turntable at first
cam.reset()

axis = scene.visuals.XYZAxis(parent=view)
s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
affine = s.as_matrix()
axis.transform = affine

# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """


# Setup colormap iterators
opaque_cmaps = cycle(get_colormaps())
translucent_cmaps = cycle([TransGrays(), TransFire()])
opaque_cmap = next(opaque_cmaps)
translucent_cmap = next(translucent_cmaps)

# interp_methods = cycle(volume1.interpolation_methods)
# interp = next(interp_methods)
# print(volume1.interpolation_methods)

volume.cmap = TransGrays()


# Implement key presses
# @canvas.events.key_press.connect
# def on_key_press(event):
#     global opaque_cmap, translucent_cmap
#     if event.text == "1":
#         cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
#         view.camera = cam_toggle.get(view.camera, cam2)
#         print(view.camera.name + " camera")
#         if view.camera is cam2:
#             axis.visible = True
#         else:
#             axis.visible = False
#     elif event.text == "2":
#         methods = ["mip", "translucent", "iso", "additive"]
#         method = methods[(methods.index(volume1.method) + 1) % 4]
#         print("Volume render method: %s" % method)
#         cmap = opaque_cmap if method in ["mip", "iso"] else translucent_cmap
#         volume1.method = method
#         volume1.cmap = cmap
#     elif event.text == "3":
#         volume1.visible = not volume1.visible
#     elif event.text == "4":
#         if volume1.method in ["mip", "iso"]:
#             cmap = opaque_cmap = next(opaque_cmaps)
#         else:
#             print("here")
#             cmap = translucent_cmap = next(translucent_cmaps)
#         volume1.cmap = cmap
#     elif event.text == "5":
#         interp = next(interp_methods)
#         volume1.interpolation = interp
#         print(f"Interpolation method: {interp}")
#     elif event.text == "0":
#         cam1.set_range()
#         cam3.set_range()
#     elif event.text != "" and event.text in "[]":
#         s = -0.025 if event.text == "[" else 0.025
#         volume1.threshold += s
#         th = volume1.threshold
#         print("Isosurface threshold: %0.3f" % th)


# for testing performance
# @canvas.connect
# def on_draw(ev):
# canvas.update()

if __name__ == "__main__":
    print(__doc__)
    app.run()
    # needed PyQt, vispy, OpenGL
