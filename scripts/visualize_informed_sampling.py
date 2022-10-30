import torch
import matplotlib.pyplot as plt

from torchngp import sampling


def compute_weights(ts: torch.Tensor):
    # unnormalized bimodal distribution
    pi1 = 0.25 * (-((ts - 5.0) ** 2) / 2).exp()
    pi2 = 0.75 * (-((ts - 8.0) ** 2) / 2).exp()
    return pi1 + pi2


def main():
    B = 1
    Ts = 20
    Ti = 100
    tnear = torch.tensor([[0.0]]).expand(B, 1)
    tfar = torch.tensor([[10.0]]).expand(B, 1)

    # Uninformed samples and weights
    ts = sampling.sample_ray_step_stratified(tnear, tfar, Ts)
    # Lets assume weights peak around 5.0

    weights = compute_weights(ts)

    # Informed samples
    ts_new = sampling.sample_ray_step_informed(
        ts, tnear, tfar, weights=weights, n_samples=Ti
    )
    weights_new = compute_weights(ts_new)
    assert (ts_new[1:, 0] - ts_new[:-1, 0] >= 0).all()

    # Plot (transparency shows time (sequence order))

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=(4, 1),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

    ax0.scatter(
        ts.view(-1),
        weights.view(-1),
        s=40,
        zorder=2,
        color="gray",
        label=f"stratified #{Ts}",
        marker="+",
    )
    alphas = torch.linspace(0.1, 1, Ti)
    ax0.scatter(
        ts_new.view(-1),
        weights_new.view(-1),
        s=40,
        alpha=alphas,
        color="C1",
        label=f"informed #{Ti}",
    )

    ax0.legend(loc="upper right")
    for lh in ax0.legend().legendHandles:
        lh.set_alpha(1)
    ax0.set_ylabel("weight")
    ax1.hist(ts.view(-1), range=(0, 10), bins=20, density=True, alpha=0.8, color="gray")
    ax1.hist(
        ts_new.view(-1), range=(0, 10), bins=20, density=True, alpha=0.8, color="C1"
    )
    ax0.tick_params(axis="x", labelbottom=False)
    ax1.set_ylabel("density")
    ax1.set_xlabel("t")
    fig.suptitle("Stratified vs. informed sampling")
    fig.savefig("etc/stratified_vs_informed.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
