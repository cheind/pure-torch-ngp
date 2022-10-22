import torch
import numpy as np

from .nerf import NeRF
from ngp_nerf import radiance


if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nerf_kwargs, state_dict, aabb = torch.load("tmp/nerf.pth")
    nerf = NeRF(**nerf_kwargs)
    nerf.load_state_dict(state_dict)
    nerf = nerf.to(dev).eval()

    with torch.no_grad():
        d, rgb = radiance.rasterize_field(
            nerf, (256, 256, 256), dev=dev, batch_size=2**16
        )
        rgbd = torch.cat((rgb, d.unsqueeze(-1)), -1).cpu()

    from pathlib import Path
    from PIL import Image

    outdir = Path("tmp/nerf_slices/")
    outdir.mkdir(exist_ok=True, parents=False)
    for idx, slice in enumerate(rgbd):
        slice = (slice * 255).to(torch.uint8).numpy()
        img = Image.fromarray(slice, mode="RGBA")
        img.save(outdir / f"slice_{idx:04d}.png")

    rgbd = rgbd.cpu()
    np.savez("tmp/volume.npz", rgb=rgbd[..., :3], d=rgbd[..., 3])
