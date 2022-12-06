from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from torchngp import functional

o = torch.tensor([[-1.0, 0.0, 0.0]])
d = torch.tensor([[1.0, 0.0, 0.0]])
dnorm = torch.tensor([[1.0]])

plane_n = torch.tensor([1.0, 0.0, 0.0])
plane_o = torch.tensor([0.3, 0.0, 0.0])


def sample_vol(ts, density_mode: Literal["constant", "linear"], c: float):
    xyz = functional.evaluate_ray(o, d, ts)  # (T,B,3)

    color = torch.tensor(mpl.colormaps["jet"](xyz[..., 0].numpy()))  # (T,B,4)
    density = (
        (xyz - plane_o[None, None, :]).unsqueeze(-2) @ plane_n[None, None, :, None]
    ).squeeze(-1)
    mask = density < 0
    density[mask] = 0.0
    if density_mode == "constant":
        density[~mask] = c
    else:
        density[~mask] *= c

    return color, density


def plot_ray(
    density_mode: Literal["constant", "linear"],
    c: float,
    sampling_mode: Literal["linear", "stratified", "informed"],
):

    tnear = torch.tensor([[1.0]])
    tfar = torch.tensor([[2.0]])

    if sampling_mode == "linear":
        ts = torch.linspace(tnear.item(), tfar.item(), 50).reshape(50, 1, 1)
    elif sampling_mode == "stratified":
        ts = functional.sample_ray_step_stratified(tnear, tfar, 50)
    elif sampling_mode == "informed":
        ts = functional.sample_ray_step_stratified(tnear, tfar, 20)
        _, density = sample_vol(ts, density_mode=density_mode, c=c)
        ts_weights = functional.integrate_timesteps(
            density, ts, dnorm, tfinal=tfar + 1e-2
        )
        ts = functional.sample_ray_step_informed(
            ts, tnear, tfar, ts_weights, n_samples=50
        )

    color, density = sample_vol(ts, density_mode=density_mode, c=c)
    ts_weights = functional.integrate_timesteps(density, ts, dnorm, tfinal=tfar + 1e-2)
    out_color = functional.color_map(color[..., :3], ts_weights, per_timestep=True)
    out_transm = 1.0 - functional.alpha_map(ts_weights, per_timestep=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.text(
        0.25,
        0.9,
        f"density_mode={density_mode}, c={c:.1f}, sampling_mode={sampling_mode}",
    )
    ax.vlines(
        plane_o[0], -0.1, 1.3, colors="k", linestyles="--", zorder=3, label="plane"
    )

    shifted_ts = ts[:, 0] - tnear[0, 0]
    ax.plot(shifted_ts, out_transm[:, 0], c="gray", label=r"$T(t)\alpha(t)$")
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
        label="sample colors",
    )
    ax.scatter(
        shifted_ts,
        out_transm[:, 0] + 0.04,
        c=out_color[:, 0],
        s=20,
        marker="s",
        zorder=2,
        label="integrated colors",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("transmittance|density")
    # ax.set_aspect("equal")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.2)
    plt.suptitle("Path tracing")

    plt.legend(loc="upper right", ncols=2)
    plt.savefig(f"etc/path_tracing_{sampling_mode}_{density_mode}_{c}.png", dpi=300)

    return fig


def main():
    _ = plot_ray(density_mode="constant", c=10.0, sampling_mode="linear")
    _ = plot_ray(density_mode="constant", c=10.0, sampling_mode="stratified")
    _ = plot_ray(density_mode="constant", c=10.0, sampling_mode="informed")
    # _ = plot_ray(density_mode="linear", c=1.0, sampling_mode="linear")
    # _ = plot_ray(density_mode="linear", c=10.0, sampling_mode="linear")
    # _ = plot_ray(density_mode="linear", c=10.0, sampling_mode="stratified")
    # _ = plot_ray(density_mode="linear", c=10.0, sampling_mode="informed")
    # _ = plot_ray(density_mode="linear", c=float("inf"), sampling_mode="linear")
    plt.show()


if __name__ == "__main__":
    main()
