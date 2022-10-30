# torchngp

This repository contains an unofficial implementation of _Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_ using pure PyTorch functionality.

Currently the tasks learning Neural Radiance Fields and Gigapixel image approximation are implemented.

## NeRF Features

-   Support for multi-level hash encodings using a hybrid dense/sparse approach
-   Support for loading `transforms.json` scene file format
-   Support for alpha channels in RGBA input images
-   Support for training with dynamic noise backgrounds to encourage zero density learning
-   Added functionality to render scene setup
-   Added functionality to render novel viewpoints
-   Added volume rasterization and export
-   Added (unfinished) script to raytrace exported volume

## Blender Features

-   Automatically generate camera views for scene
-   Export scene to `transforms.json` for NeRF training

## Runtime

This implementation is roughly an order of magnitude slower than the original implementation. That is, it runs in minutes what takes the original implementation only seconds. In particular we haven't implemented acceleration structures to speed up learning and we miss an integrated real-time viewer.
