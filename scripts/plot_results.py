"""Visualize PATROL run results from the per-run recorder output.

Reads a run directory (``windows.csv`` + ``summary.json`` + ``meta.json``) and renders
a multi-panel figure: utilization tracking, the deadline-miss guarantee (running miss
ratio vs. target α and the Theorem-1 bound), per-window DMR, reward, the RCRL λ
trajectory, and average QoS.

Usage:
    # single run
    python scripts/plot_results.py --run outputs/.../runs/<run_id>
    # compare safety modes (e.g. an ablation output dir, or explicit run dirs)
    python scripts/plot_results.py --ablation-dir outputs/safety_ablation
    python scripts/plot_results.py --compare <run_dir_a> <run_dir_b> ...

Design: one y-axis per panel (no dual-axis); Okabe-Ito colorblind-safe palette in a
fixed role order; recessive grid; legend whenever >= 2 series share a panel.
"""

import argparse
import glob
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

# Okabe-Ito colorblind-safe palette, assigned by role (fixed order, never cycled).
C = {
    "act": "#0072B2",      # blue     - actual utilization / lambda
    "ref": "#555555",      # gray     - reference / target lines
    "miss": "#D55E00",     # vermilion- miss ratio
    "bound": "#E69F00",    # orange   - theorem bound
    "reward": "#009E73",   # green    - reward
    "qos": "#56B4E9",      # sky      - QoS
    "fallback": "#CC79A7", # purple   - fallback markers
    "env": "#999999",      # gray     - env-change markers
}
# Fixed color per safety mode for comparison plots.
MODE_COLORS = {"rcrl": "#0072B2", "conformal": "#E69F00",
               "rt_analytic": "#009E73", "none": "#D55E00"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": 9,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "axes.labelcolor": "#333333", "text.color": "#333333",
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
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def _load_run(run_dir):
    run_dir = Path(run_dir)
    rows = load_windows(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    return rows, summary, meta


def _env_boundaries(summary):
    segs = summary.get("settle_by_segment") or []
    return [s["window_start"] for s in segs[1:] if "window_start" in s]


def _mark_env(ax, boundaries):
    for i, b in enumerate(boundaries):
        ax.axvline(b, color=C["env"], ls=(0, (2, 3)), lw=1.0,
                   label="env change" if i == 0 else None, zorder=1)


def _mark_changes(ax, changes, label=True):
    """Vertical markers + labels for scheduled dynamic changes (ref-util / task-set)."""
    for i, ch in enumerate(changes or []):
        x = ch.get("window")
        if x is None:
            continue
        ax.axvline(x, color=C["env"], ls=(0, (4, 2)), lw=1.1, zorder=1,
                   label="dynamic change" if (label and i == 0) else None)


def _mark_fallbacks(ax, w, fb, label=True):
    """Rug of ticks along the bottom marking windows where the safe-fallback fired."""
    idx = np.where(np.nan_to_num(fb) > 0)[0]
    if not len(idx):
        return
    ax.plot(w[idx], np.full(len(idx), 0.015), marker="|", linestyle="none",
            transform=ax.get_xaxis_transform(), color=C["fallback"],
            ms=7, mew=1.1, alpha=0.55, clip_on=False,
            label=f"fallback ({len(idx)})" if label else None)


# --------------------------------------------------------------------------- single run

def plot_run(run_dir, out_path=None):
    rows, summary, meta = _load_run(run_dir)
    if not rows:
        raise SystemExit(f"No windows.csv rows in {run_dir}")
    _style()

    w = _col(rows, "w_id")
    fb = _col(rows, "safe_fallback")
    boundaries = _env_boundaries(summary)
    changes = summary.get("dynamic_changes", [])
    rcrl = summary.get("rcrl", {})
    guard_mode = (meta.get("safety_mode") or rcrl.get("mode") or "").lower()
    is_rcrl = guard_mode == "rcrl"

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    (axU, axG), (axD, axR), (axL, axQ) = axes

    # (1) Utilization tracking
    axU.plot(w, _col(rows, "act_util"), color=C["act"], lw=1.6, label="actual util")
    axU.plot(w, _col(rows, "ref_util"), color=C["ref"], lw=1.4, ls="--", label="reference util")
    _mark_env(axU, boundaries)
    _mark_changes(axU, changes)
    _mark_fallbacks(axU, w, fb)
    axU.set_title("Utilization tracking")
    axU.set_ylabel("CPU utilization"); axU.set_xlabel("window")
    axU.set_ylim(0, max(1.05, np.nanmax(_col(rows, "act_util")) * 1.05))
    axU.legend(loc="lower right")

    # (2) Guarantee: running miss ratio vs alpha vs Theorem-1 bound.
    # Measured over SHIELDED windows only — the guarantee doesn't cover an unshielded warmup.
    miss = _col(rows, "miss")
    sa = _col(rows, "shield_active")
    shielded = np.where(np.nan_to_num(sa) > 0)[0] if np.isfinite(sa).any() else np.arange(len(w))
    if len(shielded) and len(shielded) < len(w):
        wsh = w[shielded]
        axG.axvline(wsh[0], color=C["fallback"], ls="-", lw=1.2, alpha=0.7, label="shield on")
    else:
        wsh = w
    run_miss = _running_mean(miss[shielded]) if len(shielded) else _running_mean(miss)
    axG.plot(wsh, run_miss, color=C["miss"], lw=1.8, label=r"running miss ratio $\frac{1}{t}\sum\ell$")
    alpha = rcrl.get("target_alpha")
    gamma = rcrl.get("gamma")
    if alpha is not None:
        axG.axhline(alpha, color=C["ref"], ls="--", lw=1.4, label=fr"target $\alpha$={alpha:g}")
        if gamma:
            B = max(alpha, 1.0 - alpha)
            t = np.arange(1, len(wsh) + 1)
            bound = alpha + (1.0 + gamma * B) / (gamma * t)
            axG.plot(wsh, bound, color=C["bound"], ls=":", lw=1.6,
                     label=r"Theorem-1 bound $\alpha+\frac{1+\gamma B}{\gamma t}$")
            axG.fill_between(wsh, bound, 1.0, color=C["bound"], alpha=0.05, zorder=0)
    _mark_env(axG, boundaries)
    _mark_changes(axG, changes, label=False)
    axG.set_title("Deadline-miss guarantee")
    axG.set_ylabel("miss ratio"); axG.set_xlabel("window")
    axG.set_ylim(0, min(1.0, max(0.2, (alpha or 0.1) * 3)))
    axG.legend(loc="upper right")

    # (3) Per-window deadline-miss ratio
    axD.fill_between(w, _col(rows, "dmr"), color=C["miss"], alpha=0.25, step="mid")
    axD.plot(w, _col(rows, "dmr"), color=C["miss"], lw=1.0, drawstyle="steps-mid")
    mi = np.where(miss > 0)[0]
    if len(mi):
        axD.scatter(w[mi], _col(rows, "dmr")[mi], s=14, color=C["miss"], zorder=3,
                    label=f"miss windows ({len(mi)})")
        axD.legend(loc="upper right")
    _mark_env(axD, boundaries)
    _mark_changes(axD, changes, label=False)
    _mark_fallbacks(axD, w, fb, label=False)
    axD.set_title("Per-window deadline-miss ratio")
    axD.set_ylabel("DMR"); axD.set_xlabel("window")

    # (4) Reward
    axR.plot(w, _col(rows, "reward"), color=C["reward"], lw=0.8, alpha=0.35, label="reward")
    axR.plot(w, _rolling(_col(rows, "reward")), color=C["reward"], lw=1.8,
             label="reward (rolling mean)")
    _mark_env(axR, boundaries)
    _mark_changes(axR, changes, label=False)
    axR.set_title("Reward"); axR.set_ylabel("reward"); axR.set_xlabel("window")
    axR.legend(loc="lower right")

    # (5) RCRL lambda trajectory + fallback markers
    lam = _col(rows, "rcrl_lambda")
    if is_rcrl and np.isfinite(lam).any():
        axL.plot(w, lam, color=C["act"], lw=1.6, label=r"threshold $\lambda_t$")
        axL.axhline(0.0, color=C["ref"], ls="--", lw=1.0, label=r"$\lambda=0$ (accept-all)")
        fb = _col(rows, "safe_fallback")
        fbi = np.where(fb > 0)[0]
        if len(fbi):
            axL.scatter(w[fbi], lam[fbi], s=12, color=C["fallback"], zorder=3,
                        label=f"fallback ({len(fbi)})")
        _mark_env(axL, boundaries)
        _mark_changes(axL, changes, label=False)
        axL.set_title(r"RCRL adaptive threshold $\lambda$")
        axL.set_ylabel(r"$\lambda$"); axL.set_xlabel("window")
        axL.legend(loc="upper right")
    else:
        axL.set_title(r"RCRL threshold $\lambda$ (n/a for this mode)")
        axL.text(0.5, 0.5, f"guard mode: {guard_mode or 'n/a'}", ha="center", va="center",
                 transform=axL.transAxes, color="#999999")

    # (6) Average QoS
    axQ.plot(w, _col(rows, "avg_qos"), color=C["qos"], lw=1.6, label="avg QoS / job")
    axQ.plot(w, _rolling(_col(rows, "avg_qos")), color=C["act"], lw=1.4, alpha=0.8,
             label="rolling mean")
    _mark_env(axQ, boundaries)
    _mark_changes(axQ, changes, label=False)
    _mark_fallbacks(axQ, w, fb, label=False)
    axQ.set_title("Average QoS level")
    axQ.set_ylabel("QoS"); axQ.set_xlabel("window"); axQ.legend(loc="lower right")

    # Title with headline metrics
    ov = summary.get("overall", {})
    bits = [f"{meta.get('ctrl_mode','?')}/{guard_mode or '-'}",
            f"dyn={meta.get('dynamic_mode','?')}",
            f"util={ov.get('avg_act_util', float('nan')):.3f}"]
    if rcrl:
        bits += [f"miss={rcrl.get('realized_miss_rate', float('nan')):.3f}",
                 f"α={rcrl.get('target_alpha','-')}",
                 f"guarantee={'OK' if rcrl.get('guarantee_satisfied') else '—'}"]
    fig.suptitle("PATROL run — " + "   ".join(bits), y=0.995, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    out_path = out_path or os.path.join(run_dir, "results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------- comparison

def _discover_runs(ablation_dir):
    return sorted(glob.glob(os.path.join(ablation_dir, "**", "runs", "*"), recursive=True) +
                  glob.glob(os.path.join(ablation_dir, "runs", "*")))


def plot_compare(run_dirs, out_path):
    _style()
    runs = []
    for rd in run_dirs:
        rows, summary, meta = _load_run(rd)
        if rows:
            runs.append((meta.get("safety_mode", Path(rd).name), rows, summary, meta))
    if not runs:
        raise SystemExit("No valid runs to compare.")
    # order by known mode order, then name
    order = list(MODE_COLORS.keys())
    runs.sort(key=lambda r: (order.index(r[0]) if r[0] in order else 99, r[0]))
    labels = [r[0] for r in runs]
    colors = [MODE_COLORS.get(m, C["act"]) for m in labels]

    fig, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(14, 9))

    def bar(ax, values, title, ylabel, hline=None, hlabel=None):
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, width=0.6, zorder=2)
        for xi, v in zip(x, values):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
        finite = [v for v in list(values) + ([hline] if hline is not None else []) if v is not None and np.isfinite(v)]
        if finite:
            ax.set_ylim(0, max(finite) * 1.18)  # headroom for value labels + legend
        if hline is not None:
            ax.axhline(hline, color=C["ref"], ls="--", lw=1.4, label=hlabel)
            ax.legend(loc="upper left")  # tallest bars sit right; keep legend clear of them
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(title); ax.set_ylabel(ylabel)

    def sget(s, *path, default=np.nan):
        cur = s
        for p in path:
            cur = cur.get(p, {}) if isinstance(cur, dict) else {}
        return cur if isinstance(cur, (int, float)) else default

    alphas = [r[2].get("rcrl", {}).get("target_alpha") for r in runs]
    alpha = next((a for a in alphas if a is not None), None)
    bar(a0, [sget(r[2], "rcrl", "realized_miss_rate") for r in runs],
        "Realized deadline-miss ratio", "miss ratio",
        hline=alpha, hlabel=fr"target $\alpha$={alpha:g}" if alpha else None)
    bar(a1, [sget(r[2], "overall", "avg_act_util") for r in runs],
        "Average utilization", "CPU utilization",
        hline=runs[0][2].get("overall", {}).get("start_ref_util"), hlabel="reference")
    bar(a2, [r[2].get("avg_qos_controllable") or np.nan for r in runs],
        "Average QoS (controllable tasks)", "QoS")

    # (4) overlaid running miss ratio
    for (m, rows, summary, meta), col in zip(runs, colors):
        w = _col(rows, "w_id"); rm = _running_mean(_col(rows, "miss"))
        a3.plot(w, rm, color=col, lw=1.7, label=m)
    if alpha is not None:
        a3.axhline(alpha, color=C["ref"], ls="--", lw=1.4, label=fr"$\alpha$={alpha:g}")
    a3.set_title("Running miss ratio over time"); a3.set_ylabel("miss ratio")
    a3.set_xlabel("window"); a3.legend(loc="upper right")

    fig.suptitle("Safety-mode comparison", y=0.995, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Visualize PATROL run results.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="A single run directory (contains windows.csv).")
    g.add_argument("--compare", nargs="+", help="Multiple run directories to compare.")
    g.add_argument("--ablation-dir", help="Discover runs under an ablation output dir.")
    p.add_argument("--out", help="Output PNG path.")
    args = p.parse_args()

    if args.run:
        plot_run(args.run, args.out)
    else:
        runs = args.compare or _discover_runs(args.ablation_dir)
        if not runs:
            raise SystemExit("No runs found.")
        out = args.out or "results_comparison.png"
        plot_compare(runs, out)


if __name__ == "__main__":
    main()
