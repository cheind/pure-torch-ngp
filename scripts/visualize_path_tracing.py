import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from torchngp import radiance, geometric, sampling


def plot_density_scale(ds, show=True):
    o = torch.tensor([[-1.0, 0.0, 0.0]])
    d = torch.tensor([[1.0, 0.0, 0.0]])

    plane_n = torch.tensor([1.0, 0.0, 0.0])
    plane_o = torch.tensor([0.3, 0.0, 0.0])

    tnear = torch.tensor([[1.0]])
    tfar = torch.tensor([[2.0]])

    ts = sampling.sample_ray_step_stratified(tnear, tfar, 50)
    xyz = geometric.evaluate_ray(o, d, ts)  # (T,B,3)

    color = torch.tensor(mpl.colormaps["jet"](xyz[..., 0].numpy()))  # (T,B,4)
    density = (
        (xyz - plane_o[None, None, :]).unsqueeze(-2) @ plane_n[None, None, :, None]
    ).squeeze(-1)
    mask = density < 0
    density[mask] = 0.0
    density[~mask] *= ds

    out_color, out_transm, out_alpha = radiance.integrate_path(
        color[..., :3], density, torch.cat((ts, tfar.unsqueeze(0)), 0)
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.text(0.35, 0.9, f"density scale factor {ds:.1f}")
    ax.vlines(
        plane_o[0], -0.1, 1.3, colors="k", linestyles="--", zorder=3, label="plane"
    )

    shifted_ts = ts[:, 0] - tnear[0, 0]
    ax.plot(shifted_ts, out_transm[:, 0], c="gray", label="transmittance")
    ax.plot(
        shifted_ts,
        density[:, 0],
        c="g",
        label="density",
    )

    ax.imshow(
        out_color[-1].view(1, 1, 3),
        extent=(ts[0, 0, 0] - tnear[0, 0], ts[0, 0, 0] - tnear[0, 0] + 0.1, 0.5, 0.6),
    )
    plt.text(0.01, 0.45, "final color")
    ax.scatter(
        shifted_ts,
        out_transm[:, 0],
        c=color[:, 0],
        s=20,
        zorder=2,
        label="sampled colors",
    )
    ax.scatter(
        shifted_ts,
        out_transm[:, 0] + 0.04,
        c=out_color[:, 0],
        s=20,
        marker="s",
        zorder=2,
        label="intgrated colors",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("transmittance/density")
    # ax.set_aspect("equal")
    plt.xlim(-0.1, 1.1)
    plt.ylim(0.0, 1.2)
    plt.suptitle("Path tracing")

    plt.legend(loc="upper right", ncols=2)
    plt.savefig(f"etc/path_tracing_{ds}.png", dpi=300)

    return fig


def main():
    fig = plot_density_scale(10)
    fig2 = plot_density_scale(100)
    fig3 = plot_density_scale(float("inf"))

    plt.show()


if __name__ == "__main__":
    main()
