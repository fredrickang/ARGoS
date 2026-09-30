"""Plot the window-size vs. lag trade-off from run_window_size_sweep.py.

Aggregates over seeds (mean +/- std) and draws metrics vs. window size W (log x):
  * lag k = max ceil(D_i/W)              -- grows as W shrinks
  * settle time (ms = windows*W)         -- ADAPTATION cost: grows with W
  * steady-state λ oscillation, realized miss / guarantee slack  -- FEEDBACK-LAG cost:
                                             grows as W shrinks (staler shield feedback)
  * tracking error, jitter, QoS
Plus a normalized composite (adaptation cost + feedback cost) to expose a sweet-spot W.
No titles; large fonts.
"""
import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_A, C_F = "#0072B2", "#D55E00"


def _load(path):
    by_w = defaultdict(list)
    for r in csv.DictReader(open(path)):
        def f(k):
            v = r.get(k)
            return float(v) if v not in (None, "", "None") else np.nan
        by_w[int(r["W"])].append({
            "lag": f("lag"), "settle_ms": f("settle_ms"), "settle_windows": f("settle_windows"),
            "track_err": f("track_err"), "ustd": f("ustd"), "qos": f("qos"),
            "realized_miss": f("realized_miss"), "guar_margin": f("guar_margin"),
            "lambda_std": f("lambda_std"), "fallback_rate": f("fallback_rate"),
        })
    Ws = sorted(by_w)
    agg = {}
    for k in ["lag", "settle_ms", "settle_windows", "track_err", "ustd", "qos",
              "realized_miss", "guar_margin", "lambda_std", "fallback_rate"]:
        mean = np.array([np.nanmean([d[k] for d in by_w[w]]) for w in Ws])
        std = np.array([np.nanstd([d[k] for d in by_w[w]]) for w in Ws])
        agg[k] = (mean, std)
    return np.array(Ws, dtype=float), agg


def _style(fs=20):
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#bbbbbb", "axes.linewidth": 1.2,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 1.0,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": fs, "axes.labelsize": fs + 3, "xtick.labelsize": fs - 3,
        "ytick.labelsize": fs - 3, "legend.frameon": False, "legend.fontsize": fs - 3,
    })


def _panel(ax, W, mean, std, color, ylabel, tag):
    ax.plot(W, mean, "o-", color=color, lw=2.6, ms=8, zorder=3)
    ax.fill_between(W, mean - std, mean + std, color=color, alpha=0.15, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("window size  W (ms)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(W)
    ax.set_xticklabels([f"{int(w)}" for w in W], rotation=45, fontsize=13)
    ax.text(0.03, 0.95, tag, transform=ax.transAxes, ha="left", va="top",
            fontsize=15, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ddd", alpha=0.85))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/window_size_sweep/window_size_results.csv")
    p.add_argument("--out-dir", default="outputs/sample/figures/12_window_size")
    args = p.parse_args()
    W, agg = _load(args.csv)
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)

    # ---- (A) metric grid
    _style(18)
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    specs = [
        ("lag", "#333333", "lag  k = max⌈D/W⌉", "lag (windows)"),
        ("settle_ms", C_A, "settle time (ms)  ↑ adaptation cost (large W)", "settle time (ms)"),
        ("ustd", C_F, "jitter / instability  ↑ feedback-lag cost (small W)", "steady util std"),
        ("track_err", "#009E73", "tracking error", "|mean util − ref|"),
        ("qos", "#56B4E9", "steady QoS", "avg QoS"),
        ("realized_miss", "#CC79A7", "realized miss ratio", "realized miss"),
    ]
    for ax, (key, col, tag, ylab) in zip(axes.flat, specs):
        m, s = agg[key]
        _panel(ax, W, m, s, col, ylab, tag)
    fig.tight_layout()
    fig.savefig(od / "metrics_grid.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[wsweep-plot] wrote {od/'metrics_grid.png'}")

    # ---- (B) trade-off / sweet-spot: normalized adaptation vs feedback cost + composite
    def norm(x):
        x = np.array(x, dtype=float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    adapt = norm(agg["settle_ms"][0])          # grows with W (slow real-time adaptation)
    feedback = norm(agg["ustd"][0])             # grows as W shrinks (laggy shield -> instability)
    composite = adapt + feedback
    _style(20)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(W, adapt, "o-", color=C_A, lw=2.8, ms=9, label="adaptation cost (settle time), norm.")
    ax.plot(W, feedback, "s-", color=C_F, lw=2.8, ms=9, label="feedback-lag cost (jitter), norm.")
    ax.plot(W, composite, "D-", color="#333333", lw=3.4, ms=10, label="composite (sum)")
    wi = int(np.nanargmin(composite))
    ax.axvline(W[wi], color="#666666", ls=":", lw=2.0, zorder=1)
    ax.annotate(f"sweet spot ≈ {int(W[wi])} ms", xy=(W[wi], composite[wi]),
                xytext=(W[wi] * 1.15, composite[wi] + 0.25), fontsize=17,
                arrowprops=dict(arrowstyle="->", color="#666666"))
    ax.set_xscale("log"); ax.set_xlabel("window size  W (ms)")
    ax.set_ylabel("normalized cost")
    ax.set_xticks(W); ax.set_xticklabels([f"{int(w)}" for w in W], rotation=45)
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(od / "tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[wsweep-plot] wrote {od/'tradeoff.png'}")


if __name__ == "__main__":
    main()
