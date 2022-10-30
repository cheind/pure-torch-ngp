# torchngp

This repository contains an inofficial implementation of

> "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" by
> Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller @
> ACM Transactions on Graphics (SIGGRAPH 2022)

based strictly on PyTorch functionality. Currently, the tasks of learning Neural Radiance Fields and approximating Gigapixel images are implemented.

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

## Performance considerations

This implementation is roughly an order of magnitude slower than the original implementation. That is, it takes a minute what takes the original implementation only a few seconds. In particular, we haven't implemented acceleration structures to speed up learning and we miss an integrated real-time viewer.
