"""Plot the epsilon sweep: settle, utilisation, QoS, and saturation vs initial epsilon."""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/epsilon_sweep/epsilon_summary.csv")
    p.add_argument("--out", default="outputs/sample/figures/3_rl_behavior/epsilon_sweep.png")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    def col(k):
        return [float(r[k]) if r.get(k) not in (None, "", "-") else None for r in rows]
    eps = col("epsilon")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax1, ax2), (ax3, ax4) = axes

    def errbar(ax, y, sd, label, color):
        sdv = col(sd) if sd else None
        ax.errorbar(eps, y, yerr=sdv, marker="o", capsize=3, label=label, color=color)

    errbar(ax1, [v / 1000 for v in col("settle_ms")],
           None, "settle (s, mean)", "tab:red")
    ysd = col("settle_ms_sd")
    ax1.errorbar(eps, [v / 1000 for v in col("settle_ms")],
                 yerr=[v / 1000 for v in ysd], fmt="none", capsize=3, color="tab:red", alpha=0.5)
    ax1.set_title("Settle time (lower better) — U-shaped")
    ax1.set_ylabel("settle time (s)")

    errbar(ax2, col("ss_util"), "ss_util_sd", "steady util", "tab:blue")
    ax2b = ax2.twinx()
    ax2b.plot(eps, col("ss_qos"), marker="s", color="tab:green", label="steady QoS")
    ax2b.set_ylabel("steady QoS", color="tab:green")
    ax2.set_title("Steady-state utilisation & QoS")
    ax2.set_ylabel("steady util", color="tab:blue")

    ax3.errorbar(eps, col("sat_all"), yerr=col("sat_all_sd"), marker="o", capsize=3,
                 label="all", color="tab:purple")
    ax3.plot(eps, col("sat_warmup"), marker="^", ls="--", label="warm-up (0-200)", color="tab:orange")
    ax3.plot(eps, col("sat_steady"), marker="v", ls=":", label="steady (last 25%)", color="tab:brown")
    ax3.set_title("Utilisation-saturation (spike) rate")
    ax3.set_ylabel("fraction of windows at util>=0.999")
    ax3.legend(fontsize=8)

    ax4.errorbar(eps, col("realized_miss"), yerr=col("realized_miss_sd"), marker="o",
                 capsize=3, label="realized miss", color="tab:red")
    ax4.plot(eps, col("fallback_rate"), marker="s", label="fallback rate", color="tab:cyan")
    ax4.axhline(0.1, ls="--", color="k", alpha=0.5, label="alpha=0.1")
    ax4.set_title("Safety: miss & fallback (guarantee holds at all eps)")
    ax4.set_ylabel("rate")
    ax4.legend(fontsize=8)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlabel("initial epsilon")
        ax.grid(alpha=0.3)
    # highlight chosen default
    for ax in (ax1, ax2, ax3, ax4):
        ax.axvline(0.3, color="green", ls="-", alpha=0.25, lw=6)

    fig.suptitle("Initial-epsilon sweep (static, 3-seed mean; green band = chosen 0.3)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
