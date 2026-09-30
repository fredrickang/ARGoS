"""Render each PATROL result panel as its own file — no titles, large fonts.

Splits the 6-panel single-run figure (utilization tracking, deadline-miss guarantee,
per-window DMR, reward, RCRL lambda, average QoS) into six standalone PNGs suitable
for slides/papers: titles removed, font sizes enlarged, one axis per file.

Usage:
    python scripts/plot_panels.py --run <run_dir> --out-dir <dir> [--prefix p_]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patrol.sim.recorder import load_windows

C = {
    "act": "#0072B2", "ref": "#555555", "miss": "#D55E00", "bound": "#E69F00",
    "reward": "#009E73", "qos": "#56B4E9", "fallback": "#CC79A7", "env": "#999999",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#bbbbbb", "axes.linewidth": 1.2,
        "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 1.0,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        # enlarged fonts for standalone panels
        "font.size": 20, "axes.titlesize": 22,
        "axes.labelsize": 24, "xtick.labelsize": 19, "ytick.labelsize": 19,
        "legend.frameon": False, "legend.fontsize": 19,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "axes.labelcolor": "#222222", "text.color": "#222222",
    })


def _col(rows, key):
    return np.array([r.get(key) if r.get(key) is not None else np.nan for r in rows], dtype=float)


def _running_mean(x):
    x = np.asarray(x, dtype=float)
    return np.cumsum(np.nan_to_num(x)) / np.arange(1, len(x) + 1)


def _rolling(x, w=15):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    if len(x) < 2:
        return x
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode="same")


def _load_run(run_dir):
    run_dir = Path(run_dir)
    rows = load_windows(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    return rows, summary, meta


def _mark_fallbacks(ax, w, fb, label=True):
    idx = np.where(np.nan_to_num(fb) > 0)[0]
    if not len(idx):
        return
    ax.plot(w[idx], np.full(len(idx), 0.015), marker="|", linestyle="none",
            transform=ax.get_xaxis_transform(), color=C["fallback"],
            ms=11, mew=1.6, alpha=0.6, clip_on=False,
            label=f"fallback ({len(idx)})" if label else None)


def _save(fig, ax, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[panel] wrote {path}")


def main():
    p = argparse.ArgumentParser(description="Per-panel PATROL result figures (no title, big font).")
    p.add_argument("--run", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--figsize", nargs=2, type=float, default=[8.5, 5.5])
    args = p.parse_args()

    rows, summary, meta = _load_run(args.run)
    if not rows:
        raise SystemExit(f"No windows.csv rows in {args.run}")
    _style()
    os.makedirs(args.out_dir, exist_ok=True)
    fs = tuple(args.figsize)
    pre = args.prefix

    w = _col(rows, "w_id")
    fb = _col(rows, "safe_fallback")
    rcrl = summary.get("rcrl", {})

    def newfig():
        fig, ax = plt.subplots(figsize=fs)
        return fig, ax

    # (1) Utilization tracking
    fig, ax = newfig()
    ax.plot(w, _col(rows, "act_util"), color=C["act"], lw=2.2, label="actual util")
    ax.plot(w, _col(rows, "ref_util"), color=C["ref"], lw=2.0, ls="--", label="reference util")
    _mark_fallbacks(ax, w, fb)
    ax.set_ylabel("CPU utilization"); ax.set_xlabel("window")
    ax.set_ylim(0, max(1.05, np.nanmax(_col(rows, "act_util")) * 1.05))
    ax.legend(loc="lower right")
    _save(fig, ax, os.path.join(args.out_dir, f"{pre}utilization.png"))

    # (2) Deadline-miss guarantee
    fig, ax = newfig()
    miss = _col(rows, "miss")
    sa = _col(rows, "shield_active")
    shielded = np.where(np.nan_to_num(sa) > 0)[0] if np.isfinite(sa).any() else np.arange(len(w))
    if len(shielded) and len(shielded) < len(w):
        wsh = w[shielded]
        ax.axvline(wsh[0], color=C["fallback"], ls="-", lw=1.6, alpha=0.7, label="shield on")
    else:
        wsh = w
    run_miss = _running_mean(miss[shielded]) if len(shielded) else _running_mean(miss)
    ax.plot(wsh, run_miss, color=C["miss"], lw=2.4, label=r"running miss ratio")
    alpha = rcrl.get("target_alpha"); gamma = rcrl.get("gamma")
    if alpha is not None:
        ax.axhline(alpha, color=C["ref"], ls="--", lw=2.0, label=fr"target $\alpha$={alpha:g}")
        if gamma:
            B = max(alpha, 1.0 - alpha)
            t = np.arange(1, len(wsh) + 1)
            bound = alpha + (1.0 + gamma * B) / (gamma * t)
            ax.plot(wsh, bound, color=C["bound"], ls=":", lw=2.4,
                    label=r"Theorem-1 bound")
            ax.fill_between(wsh, bound, 1.0, color=C["bound"], alpha=0.05, zorder=0)
    ax.set_ylabel("miss ratio"); ax.set_xlabel("window")
    ax.set_ylim(0, min(1.0, max(0.2, (alpha or 0.1) * 3)))
    ax.legend(loc="upper right")
    _save(fig, ax, os.path.join(args.out_dir, f"{pre}guarantee.png"))

    # (3) Per-window DMR
    fig, ax = newfig()
    ax.fill_between(w, _col(rows, "dmr"), color=C["miss"], alpha=0.25, step="mid")
    ax.plot(w, _col(rows, "dmr"), color=C["miss"], lw=1.4, drawstyle="steps-mid")
    mi = np.where(miss > 0)[0]
    if len(mi):
        ax.scatter(w[mi], _col(rows, "dmr")[mi], s=28, color=C["miss"], zorder=3,
                   label=f"miss windows ({len(mi)})")
        ax.legend(loc="upper right")
    _mark_fallbacks(ax, w, fb, label=False)
    ax.set_ylabel("DMR"); ax.set_xlabel("window")
    _save(fig, ax, os.path.join(args.out_dir, f"{pre}dmr.png"))

    # (4) Reward
    fig, ax = newfig()
    ax.plot(w, _col(rows, "reward"), color=C["reward"], lw=1.0, alpha=0.32, label="reward")
    ax.plot(w, _rolling(_col(rows, "reward")), color=C["reward"], lw=2.4, label="reward (rolling mean)")
    ax.set_ylabel("reward"); ax.set_xlabel("window")
    ax.legend(loc="lower right")
    _save(fig, ax, os.path.join(args.out_dir, f"{pre}reward.png"))

    # (5) RCRL lambda
    fig, ax = newfig()
    lam = _col(rows, "rcrl_lambda")
    ax.plot(w, lam, color=C["act"], lw=2.2, label=r"threshold $\lambda_t$")
    ax.axhline(0.0, color=C["ref"], ls="--", lw=1.6, label=r"$\lambda=0$ (accept-all)")
    fbi = np.where(np.nan_to_num(fb) > 0)[0]
    if len(fbi):
        ax.scatter(w[fbi], lam[fbi], s=24, color=C["fallback"], zorder=3,
                   label=f"fallback ({len(fbi)})")
    ax.set_ylabel(r"$\lambda$"); ax.set_xlabel("window")
    ax.legend(loc="upper right")
    _save(fig, ax, os.path.join(args.out_dir, f"{pre}lambda.png"))

    # (6) Average QoS
    fig, ax = newfig()
    ax.plot(w, _col(rows, "avg_qos"), color=C["qos"], lw=2.2, label="avg QoS / job")
    ax.plot(w, _rolling(_col(rows, "avg_qos")), color=C["act"], lw=2.0, alpha=0.8, label="rolling mean")
    _mark_fallbacks(ax, w, fb, label=False)
    ax.set_ylabel("QoS"); ax.set_xlabel("window")
    ax.legend(loc="lower right")
    _save(fig, ax, os.path.join(args.out_dir, f"{pre}qos.png"))


if __name__ == "__main__":
    main()
