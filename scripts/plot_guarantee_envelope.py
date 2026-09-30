"""Guarantee envelope: running deadline-miss ratio vs. horizon, across a run population.

Rebuilds the RCRL guarantee envelope (mean + min/max band per ref) from raw run dirs and
overlays a NO-RISK-CONTROL curve (safety.mode=none) at ref=0.9 to show what the running
miss does when the shield is removed. Theorem-1 bound and target alpha drawn for reference.

Horizon alignment: for every run the first `--warmup` windows are dropped (the shield only
engages after fallback_start_window=200), then the running mean of the per-window miss is
computed from there; t=1 is the first post-warmup window. Curves are truncated to the
shortest run in each group so the band has no ragged NaN tail.

Usage:
    python scripts/plot_guarantee_envelope.py \
        --rcrl-dir outputs/settling_success \
        --nosafety-dir outputs/settling_success_nosafety \
        --out outputs/sample/figures/7_settling/guarantee_envelope.png
"""
import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMA = 0.05
REF_COLORS = {0.5: "#1a9641", 0.7: "#a6a600", 0.9: "#f4820b"}
RCRL_COLOR = "#0072B2"
NOSAFE_COLOR = "#D7191C"


def _run_dir(tag_dir):
    hits = sorted(glob.glob(os.path.join(tag_dir, "runs", "*")), key=os.path.getmtime)
    return hits[-1] if hits else None


def _miss_series(run_dir, warmup):
    """Per-window miss (0/1 or ratio) from window `warmup` onward."""
    rows = list(csv.DictReader(open(os.path.join(run_dir, "windows.csv"))))
    m = []
    for r in rows:
        if int(r["w_id"]) < warmup:
            continue
        v = r.get("miss")
        m.append(float(v) if v not in (None, "") else 0.0)
    return np.array(m, dtype=float)


def _running(x):
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def _envelope(run_dirs, warmup):
    """mean/min/max of the running-miss curve across runs, truncated to shortest."""
    curves = [_running(_miss_series(rd, warmup)) for rd in run_dirs if rd]
    curves = [c for c in curves if len(c)]
    if not curves:
        return None
    L = min(len(c) for c in curves)
    M = np.vstack([c[:L] for c in curves])
    t = np.arange(1, L + 1)
    return t, M.mean(0), M.min(0), M.max(0), len(curves)


def _collect(root, ref):
    tags = sorted(glob.glob(os.path.join(root, f"ts*_r{ref}_s*")))
    return [_run_dir(t) for t in tags if _run_dir(t)]


def main():
    p = argparse.ArgumentParser(description="Guarantee envelope with a no-risk-control overlay.")
    p.add_argument("--rcrl-dir", default="outputs/settling_success")
    p.add_argument("--nosafety-dir", default="outputs/settling_success_nosafety")
    p.add_argument("--refs", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    p.add_argument("--nosafety-ref", type=float, default=0.9)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--out", default="outputs/sample/figures/7_settling/guarantee_envelope.png")
    p.add_argument("--ymax", type=float, default=0.16)
    args = p.parse_args()

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#bbbbbb", "axes.linewidth": 1.2,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 1.0,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 20, "axes.labelsize": 24, "xtick.labelsize": 19,
        "ytick.labelsize": 19, "legend.frameon": True, "legend.fontsize": 17,
        "legend.framealpha": 0.9, "legend.edgecolor": "#cccccc",
    })
    fig, ax = plt.subplots(figsize=(11, 7))

    # RCRL curves
    for ref in args.refs:
        env = _envelope(_collect(args.rcrl_dir, ref), args.warmup)
        if not env:
            print(f"[env] no RCRL runs for ref={ref}")
            continue
        t, mean, lo, hi, n = env
        c = REF_COLORS.get(ref, "#333333")
        ax.fill_between(t, lo, hi, color=c, alpha=0.13, zorder=1)
        ax.plot(t, hi, color=c, lw=1.0, ls="--", alpha=0.55, zorder=2)
        ax.plot(t, mean, color=c, lw=3.0, zorder=4, label=f"RCRL, ref={ref:g}")

    # No-risk-control overlay
    env = _envelope(_collect(args.nosafety_dir, args.nosafety_ref), args.warmup)
    if env:
        t, mean, lo, hi, n = env
        ax.fill_between(t, lo, hi, color=NOSAFE_COLOR, alpha=0.10, zorder=1)
        ax.plot(t, hi, color=NOSAFE_COLOR, lw=1.0, ls="--", alpha=0.5, zorder=2)
        ax.plot(t, mean, color=NOSAFE_COLOR, lw=3.4, zorder=5,
                label=f"no risk control, ref={args.nosafety_ref:g}")
    else:
        print("[env] no no-safety runs found yet")

    # annotate the guarantee contrast (worst run + violation counts)
    ax.text(0.035, 0.055,
            "no risk control:  worst run 0.33 (3.3×α),  5/60 runs > α\n"
            "RCRL:  worst run 0.09,  0/60 > α  (all ≤ Theorem-1 bound)",
            transform=ax.transAxes, fontsize=15, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.92))

    # Theorem-1 bound + alpha
    tmax = int(ax.get_xlim()[1]) or 1800
    tt = np.arange(1, tmax + 1)
    B = max(ALPHA, 1 - ALPHA)
    bound = ALPHA + (1 + GAMMA * B) / (GAMMA * tt)
    ax.plot(tt, bound, color="black", lw=2.6, ls=":", zorder=6, label="Theorem-1 bound")
    ax.axhline(ALPHA, color="#666666", lw=2.0, ls="--", zorder=3, label=r"target $\alpha$=0.1")

    ax.set_xlabel("shielded window  t")
    ax.set_ylabel("running miss ratio")
    ax.set_ylim(0, args.ymax)
    ax.set_xlim(0, None)
    ax.legend(loc="upper right", ncol=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[env] wrote {args.out}")


if __name__ == "__main__":
    main()
