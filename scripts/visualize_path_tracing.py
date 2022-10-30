import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from torchngp import radiance, geometric, sampling


def plot_density_scale(ds, show=True):
    o = torch.tensor([[-1.0, 0.0, 0.0]])
    d = torch.tensor([[1.0, 0.0, 0.0]])

    plane_n = torch.tensor([1.0, 0.0, 0.0])
    plane_o = torch.tensor([0.5, 0.0, 0.0])

    tnear = torch.tensor([[1.0]])
    tfar = torch.tensor([[2.0]])

    ts = sampling.sample_ray_step_stratified(tnear, tfar, 50)

    # Estimate colors and density values at sample positions
    xyz = geometric.evaluate_ray(o, d, ts)  # (T,B,3)
    color = torch.tensor(mpl.colormaps["jet"](xyz[..., 0].numpy()))  # (T,B,4)
    density = (
        (xyz - plane_o[None, None, :]).unsqueeze(-2) @ plane_n[None, None, :, None]
    ).squeeze(-1)
    mask = density < 0
    density[mask] = 0.0
    density[~mask] *= ds

    final_colors, transmittance, alpha = radiance.integrate_path(
        color[..., :3], density, ts, tfar
    )
    print(final_colors)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.text(0.35, 0.9, f"density scale factor {ds:.2f}")
    ax.vlines(
        plane_o[0], 0.0, 1.0, colors="k", linestyles="--", zorder=3, label="plane"
    )
    ax.plot(
        ts[:, 0] - tnear[0, 0], transmittance[:, 0], c="gray", label="transmittance"
    )
    ax.plot(
        ts[:, 0] - tnear[0, 0],
        density[:, 0] / density.max(),
        c="g",
        label="relative density",
    )

    ax.imshow(
        final_colors.view(1, 1, 3),
        extent=(ts[0, 0, 0] - tnear[0, 0], ts[0, 0, 0] - tnear[0, 0] + 0.1, 0.5, 0.6),
    )
    plt.text(0.01, 0.45, "final color")
    ax.scatter(
        ts[:, 0] - tnear[0, 0],
        transmittance[:, 0],
        c=color[:, 0],
        s=10,
        zorder=2,
        label="sampled colors",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("transmittance/density")
    # ax.set_aspect("equal")
    plt.xlim(-0.1, 1.1)
    plt.ylim(0.0, 1.2)
    plt.suptitle("Path tracing")

    plt.legend(loc="upper center", ncols=2)
    plt.savefig(f"etc/path_tracing_{ds}.png", dpi=300)

    return fig


def main():
    fig = plot_density_scale(10)
    fig2 = plot_density_scale(100)

    plt.show()


if __name__ == "__main__":
    main()
