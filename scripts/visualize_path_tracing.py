import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from ngptorch import rays, radiance


def main():
    o = torch.tensor([[-1.0, 0.0, 0.0]])
    d = torch.tensor([[1.0, 0.0, 0.0]])

    plane_n = torch.tensor([1.0, 0.0, 0.0])
    plane_o = torch.tensor([0.2, 0.0, 0.0])

    tnear = torch.tensor([1.0])
    tfar = torch.tensor([2.0])

    ts = rays.sample_rays_uniformly(tnear, tfar, 100)

    # Estimate colors and density values at sample positions
    xyz = o[:, None] + ts[..., None] * d[:, None]  # (B,T,3)
    color = torch.tensor(mpl.colormaps["hsv"](xyz[..., 0].numpy()))  # (B,T,4)
    density = (
        ((xyz - plane_o[None, None, :]).unsqueeze(-2) @ plane_n[None, None, :, None])
        .squeeze(-1)
        .squeeze(-1)
    )
    mask = density < 0
    density[mask] = 0.0
    density[~mask] *= 100

    final_colors, transmittance, alpha = radiance.integrate_path(
        color[..., :3], density, ts, tfar
    )
    plt.vlines(
        plane_o[0], 0.0, 1.0, colors="k", linestyles="--", zorder=3, label="plane"
    )
    plt.plot(ts[0] - tnear[0], transmittance[0], c="gray", label="transmittance")
    plt.plot(
        ts[0] - tnear[0], density[0] / density.max(), c="g", label="relative density"
    )

    plt.imshow(
        final_colors.view(1, 1, 3),
        extent=(ts[0, 0] - tnear[0], ts[0, 0] - tnear[0] + 0.1, 0.5, 0.6),
        label="final color",
    )
    plt.scatter(
        ts[0] - tnear[0],
        transmittance[0],
        c=color[0],
        s=10,
        zorder=2,
        label="sampled colors",
    )
    plt.xlabel("x")
    plt.ylabel("transmittance")
    plt.gca().set_aspect("equal")
    plt.xlim(-0.1, 1.1)
    plt.ylim(0.0, 1.2)
    plt.legend(loc="upper center", ncols=2)
    plt.show()


if __name__ == "__main__":
    main()
