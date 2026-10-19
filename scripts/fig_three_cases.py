#!/usr/bin/env python
"""Three-case comparison figures on a single task set.

Runs ARGoS on ONE fixed task set (deterministic, seed 22) under:
  * Case 1  — valid profiles, analytical score, alpha=0, mode-change  -> hard guarantee
  * Case 2  — invalid profiles, analytical score, WCET underestimated by 10/30/50/70/90%
  * Case 3  — unavailable profiles, learned classifier
All with the shield active from window 0 (no unshielded warmup backlog). Produces three
figures — utilization, per-window DMR, and cumulative-average DMR — each overlaying all
cases. Output: paper_results/figures/.

Usage:  python scripts/fig_three_cases.py            # run sims + plot
        python scripts/fig_three_cases.py --plot-only # re-plot from existing runs
"""

import argparse
import copy
import csv
import glob
import os
import subprocess
import sys

import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    print("matplotlib required:", exc); sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CFG = os.path.join(ROOT, "configs", "run_sim.yaml")
FIG_DIR = os.path.join(ROOT, "paper_results", "figures")
DURATION_MS = 2_500_000

# Set by main(): 'paper' = Eq.1 (QoS gated on no-miss); 'v2' = legacy util-tracking reward.
REWARD = "paper"


def out_root():
    name = "fig3cases" if REWARD == "paper" else f"fig3cases_{REWARD}"
    return os.path.join(ROOT, "outputs", name)


def fig_path(key):
    pre = "" if REWARD == "paper" else f"{REWARD}_"
    return os.path.join(FIG_DIR, f"three_cases_{pre}{key}.png")

UNDEREST = [0.1, 0.5, 0.9]                      # Case-2 WCET underestimation levels
ALPHA = 0.1                                     # target for Cases 2 & 3

# --- colors: Case 1 / Case 3 are distinct hues; Case 2 is ONE hue, light->dark by invalidity.
C1_COLOR = "#0B7A75"   # teal  (hard guarantee)
C3_COLOR = "#6A3D9A"   # purple (learned)
C2_RAMP = ["#FEB24C", "#FC4E2A", "#B10026"]  # YlOrRd 10% / 50% / 90%


def cases():
    """(tag, label, color, safety-overrides) for every curve, in legend order."""
    out = [("case1", "Case 1 (valid, hard)", C1_COLOR, dict(
        score="analytical", profile="valid", target_miss_alpha=0.0,
        lambda_init=0.0, mode_change=True))]
    for u, col in zip(UNDEREST, C2_RAMP):
        out.append((f"case2_u{int(u*100)}", f"Case 2 ({int(u*100)}% underest.)", col, dict(
            score="analytical", profile="invalid", profile_underest=u,
            target_miss_alpha=ALPHA, lambda_init=0.5, mode_change=False)))
    out.append(("case3", "Case 3 (learned)", C3_COLOR, dict(
        score="learned", target_miss_alpha=ALPHA, lambda_init=0.5, mode_change=False)))
    return out


def build_cfg(safety_over, out_dir):
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["simulation"]["duration_ms"] = DURATION_MS
    cfg.setdefault("output", {})["dir"] = out_dir
    cfg["controller"]["dynamic_mode"] = 0
    cfg["controller"].setdefault("rl", {})["reward"] = REWARD
    s = cfg["controller"].setdefault("safety", {})
    s["fallback_start_window"] = 0          # shield from window 0 (no backlog warmup)
    s["loss_mode"] = "releasable"
    s.update(safety_over)
    return cfg


def run_case(tag, safety_over):
    out_dir = os.path.join(out_root(), tag)
    os.makedirs(out_dir, exist_ok=True)
    cfg = build_cfg(safety_over, out_dir)
    cfg_path = os.path.join(out_dir, "config.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    print(f"[run] {tag} ...", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), cfg_path],
                   cwd=ROOT, check=True)


def latest_windows_csv(tag):
    hits = sorted(glob.glob(os.path.join(out_root(), tag, "runs", "*", "windows.csv")),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def load_series(tag):
    """Return dict with t (seconds), util, dmr, avg_dmr (cumulative mean)."""
    path = latest_windows_csv(tag)
    if not path:
        raise FileNotFoundError(f"no windows.csv for {tag}")
    t, util, dmr = [], [], []
    for r in csv.DictReader(open(path)):
        try:
            t.append(float(r["t_end"]) / 1000.0)   # ms -> s
            util.append(float(r["act_util"]))
            dmr.append(float(r["dmr"]))
        except (KeyError, ValueError):
            continue
    avg, s = [], 0.0
    for i, d in enumerate(dmr):
        s += d
        avg.append(s / (i + 1))
    return {"t": t, "util": util, "dmr": dmr, "avg_dmr": avg}


def _smooth(y, w=9):
    """Centered moving average (readability for overlaid noisy series)."""
    if w <= 1 or len(y) < w:
        return y
    out, half = [], w // 2
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out


def _style(ax):
    ax.grid(True, ls="--", alpha=0.35, lw=0.6)
    ax.set_xlabel("time (s)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _pct(seg, p):
    """Linear-interpolated p-th percentile of seg (p in 0..100)."""
    s = sorted(seg)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _rollband(y, w=21, env_smooth=11, plo=10, phi=90):
    """Rolling mean + a SMOOTH percentile band (10th–90th by default). A min/max band is
    dominated by rare spikes on skewed data (e.g. DMR: mostly 0 with occasional 1.0), which
    balloons the upper edge and leaves the mean looking stranded at the bottom. The 10–90
    percentile reflects where most windows actually sit, so the mean sits naturally inside the
    band; edges are moving-average-smoothed for the shaded-band look. Lower edge clipped at 0."""
    n, half = len(y), w // 2
    m, lo, hi = [], [], []
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        seg = y[a:b]
        m.append(sum(seg) / len(seg))
        lo.append(_pct(seg, plo)); hi.append(_pct(seg, phi))
    lo = [max(0.0, v) for v in _smooth(lo, env_smooth)]
    hi = _smooth(hi, env_smooth)
    return m, lo, hi


def plot():
    os.makedirs(FIG_DIR, exist_ok=True)
    specs = cases()
    data = {tag: load_series(tag) for tag, *_ in specs}
    rlabel = "Eq.1 QoS reward" if REWARD == "paper" else f"{REWARD} util-tracking reward"

    # ---- small multiples: one panel per case, rolling-mean line + min/max range band ----
    def small_multiples(key, metric_title, ylabel, hline=None, ylim=None):
        n = len(specs)
        fig, axes = plt.subplots(n, 1, figsize=(9, 1.55 * n), sharex=True, sharey=True)
        for ax, (tag, label, color, _) in zip(axes, specs):
            d = data[tag]
            m, lo, hi = _rollband(d[key])
            ax.fill_between(d["t"], lo, hi, color=color, alpha=0.22, lw=0)    # smooth min/max band
            ax.plot(d["t"], m, color=color, lw=1.7)                           # mean line
            if hline is not None:
                ax.axhline(hline, color="black", ls="--", lw=0.8, alpha=0.7)
            if ylim:
                ax.set_ylim(*ylim)
            ax.text(0.012, 0.86, label, transform=ax.transAxes, fontsize=8,
                    color=color, fontweight="bold", va="top")
            ax.grid(True, ls="--", alpha=0.3, lw=0.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[-1].set_xlabel("time (s)")
        fig.supylabel(ylabel, fontsize=10)
        fig.suptitle(f"{metric_title} — {rlabel}  (line = 21-window mean, band = 10–90th percentile)",
                     fontsize=11, y=0.998)
        fig.tight_layout(rect=[0.015, 0, 1, 0.99])
        p = fig_path(key)
        fig.savefig(p, dpi=140); plt.close(fig)
        print("saved", p)

    small_multiples("util", "Utilization", "actual utilization", ylim=(0, 1.05))
    small_multiples("dmr", "Deadline-miss ratio (per window)", "DMR", hline=ALPHA, ylim=(0, 1.05))

    # ---- cumulative average DMR: overlaid (already smooth; cross-case comparison) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, label, color, _ in specs:
        ax.plot(data[tag]["t"], data[tag]["avg_dmr"], color=color, lw=1.6, label=label)
    ax.axhline(ALPHA, color="black", ls="--", lw=1.0, alpha=0.8)
    ax.text(ax.get_xlim()[1], ALPHA, f"  α={ALPHA}", va="center", fontsize=8)
    ax.set_title(f"Average deadline-miss ratio (cumulative) — {rlabel}", fontsize=11)
    ax.set_ylabel("mean DMR"); ax.set_xlabel("time (s)")
    ax.grid(True, ls="--", alpha=0.3, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout(); fig.savefig(fig_path("avg_dmr"), dpi=140); plt.close(fig)
    print("saved", fig_path("avg_dmr"))

    # ---- combined overview: MEAN lines only (no raw jitter), 3 stacked panels ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    combo = [("util", "Utilization", "actual util", None),
             ("dmr", "Deadline-miss ratio (per window)", "DMR", ALPHA),
             ("avg_dmr", "Average deadline-miss ratio (cumulative)", "mean DMR", ALPHA)]
    for ax, (key, title, ylab, hl) in zip(axes, combo):
        for tag, label, color, _ in specs:
            y = _rollband(data[tag][key])[0] if key != "avg_dmr" else data[tag][key]
            ax.plot(data[tag]["t"], y, color=color, lw=1.5, label=label)
        if hl is not None:
            ax.axhline(hl, color="black", ls="--", lw=0.9, alpha=0.8)
            ax.text(ax.get_xlim()[1], hl, f"  α={hl}", va="center", fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylab)
        ax.grid(True, ls="--", alpha=0.3, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"ARGoS three-case comparison — {rlabel}  (mean lines)", fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(fig_path("combined"), dpi=140); plt.close(fig)
    print("saved", fig_path("combined"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--reward", default="paper", choices=["paper", "v2", "decoupled_sum"],
                    help="RL reward: paper=Eq.1 (default), v2=legacy util-tracking")
    args = ap.parse_args()
    global REWARD
    REWARD = args.reward
    if not args.plot_only:
        for tag, _lbl, _col, over in cases():
            run_case(tag, over)
    plot()


if __name__ == "__main__":
    main()
