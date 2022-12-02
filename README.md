# pure-torch-ngp

This repository contains an unofficial implementation of

> "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" by
> Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller @
> ACM Transactions on Graphics (SIGGRAPH 2022)

based strictly on PyTorch only functionality.

Currently, the tasks of learning Neural Radiance Fields and approximating Gigapixel images are available. The image below shows three novel viewpoints of a Lego model. The underlying NeRF model was trained on a synthetic dataset for about 2 mins using a _NVIDIA GeForce 3090 Ti_.

![](etc/lego_val.png?raw=true)

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

## More results

![](etc/lego.gif)![](etc/suzanne.gif)

## Performance considerations

This implementation is roughly an order of magnitude slower than the original implementation. That is, it takes a minute what takes the original implementation only a few seconds. In additiona, we miss an integrated real-time viewer.
