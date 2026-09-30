"""Theorem-1 bound vs. horizon for the lagged RCRL update, across lag values k.

With a release-mode finalization lag of k windows, the guarantee becomes

    (1/T) Σ_t ℓ_t  ≤  α + (1 + (2k+1) γ B) / (γ T),      B = max(α, 1-α)

k=0 recovers the no-lag bound α + (1+γB)/(γT). Larger k -> a looser (higher) bound: the
delayed threshold update costs guarantee tightness. This script draws the bound lines only
(no simulation) so the k-dependence is visible. No title; large fonts.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--ks", nargs="+", type=int, default=[0, 1, 2, 3, 5, 10])
    p.add_argument("--T", type=int, default=1800)
    p.add_argument("--tmin", type=int, default=1)
    p.add_argument("--ymax", type=float, default=0.30)
    p.add_argument("--out", default="outputs/sample/figures/11_bound_lag/theorem_bound_lag.png")
    args = p.parse_args()

    alpha, gamma = args.alpha, args.gamma
    B = max(alpha, 1.0 - alpha)
    t = np.arange(args.tmin, args.T + 1)

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#bbbbbb", "axes.linewidth": 1.2,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 1.0,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 20, "axes.labelsize": 24, "xtick.labelsize": 19,
        "ytick.labelsize": 19, "legend.frameon": True, "legend.fontsize": 18,
        "legend.framealpha": 0.9, "legend.edgecolor": "#cccccc",
    })
    fig, ax = plt.subplots(figsize=(11, 7))

    # blue (k=0, no lag) -> red (large k)
    cmap = plt.get_cmap("viridis")
    kmax = max(args.ks) or 1
    for k in args.ks:
        bound = alpha + (1.0 + (2 * k + 1) * gamma * B) / (gamma * t)
        color = cmap(0.08 + 0.84 * (k / kmax))
        lw, ls, z = (3.6, "-", 6) if k == 0 else (2.6, "-", 4)
        label = "k=0 (no lag)" if k == 0 else f"k={k}"
        ax.plot(t, bound, color=color, lw=lw, ls=ls, zorder=z, label=label)

    ax.axhline(alpha, color="#666666", lw=2.0, ls="--", zorder=3,
               label=fr"$\alpha$={alpha:g}")
    ax.set_xlabel("shielded window  T")
    ax.set_ylabel("Theorem-1 bound")
    ax.set_ylim(0, args.ymax)
    ax.set_xlim(0, args.T)
    ax.legend(loc="upper right", ncol=2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[bound] wrote {args.out}")

    # print the tail (T) values for reference
    print(f"[bound] alpha={alpha} gamma={gamma} B={B}")
    for k in args.ks:
        b = alpha + (1.0 + (2 * k + 1) * gamma * B) / (gamma * args.T)
        print(f"  k={k:<3} bound(T={args.T}) = {b:.5f}")


if __name__ == "__main__":
    main()
