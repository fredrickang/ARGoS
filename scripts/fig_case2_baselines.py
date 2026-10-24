#!/usr/bin/env python
"""Case-2 baseline comparison on the SAME task set as fig_three_cases.

Case 2 = invalid profiles (WCET underestimated by 50%). Three controllers:
  * argos  — RL + SafetyGuard (analytical risk, adaptive threshold + fallback). Bounds M_k <= alpha.
  * noval  — RL with SafetyGuard DISABLED (safety.mode=none): decisions applied as-is, no
             validation/fallback. Shows RL alone => misses are not bounded.
  * fcs    — FCS feedback controller trusting the 50%-underestimated profile. Over-allocates
             => overload. Shows the analytical baseline failing under invalid profiles.

Same task set (taskset_1, seed 22), fixed W=10 s, 2000 windows. x-axis = window index.
Output: paper_results/figures/case2_*.png and paper_results/figures/excel_data/case2_*.csv.

Usage:  python scripts/fig_case2_baselines.py              # run + plot + export
        python scripts/fig_case2_baselines.py --plot-only  # re-plot from existing runs
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CFG = os.path.join(ROOT, "configs", "run_sim.yaml")
FIG_DIR = os.path.join(ROOT, "paper_results", "figures")
OUT_ROOT = os.path.join(ROOT, "outputs", "case2_baselines")
DATA_DIR = os.path.join(FIG_DIR, "excel_data")
W_MS = 10_000
N_WINDOWS = 2000
DURATION_MS = W_MS * (N_WINDOWS + 1)       # +1 so the N-th window completes (sim-end off-by-one)
ALPHA = 0.1
UNDEREST = 0.5                              # Case-2 WCET underestimation
GAMMA, RMIN, LAM1 = 0.05, 0.0, 0.5         # ARGoS risk-controller params (for the Theorem-1 bound)

# (tag, label, color, controller-overrides). controller-overrides merged into cfg['controller'].
VARIANTS = [
    ("argos", "ARGoS (RL + SafetyGuard)", "#0B7A75", dict(
        mode="rl",
        safety=dict(mode="rcrl", score="analytical", profile="invalid", profile_underest=UNDEREST,
                    target_miss_alpha=ALPHA, lambda_init=LAM1, rmin=0.0, mode_change=False,
                    fallback_start_window=0, loss_mode="releasable"))),
    ("noval", "No validation (RL, QoS-only reward)", "#D7191C", dict(
        mode="rl", reward="qos_ungated",     # pure QoS max, no miss gating -> keeps overloading
        safety=dict(mode="none", loss_mode="releasable"))),
    ("fcs", "FCS (invalid profile, 50%)", "#B35806", dict(
        mode="fcs", profile_underest=UNDEREST)),
    ("fcs_rand", "FCS (random profile / Case 3)", "#5E3C99", dict(
        mode="fcs", profile_random=True)),
]


def build_cfg(over, out_dir):
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["simulation"]["duration_ms"] = DURATION_MS
    cfg.setdefault("window", {})["size_ms"] = W_MS
    cfg.setdefault("output", {})["dir"] = out_dir
    cfg["controller"]["dynamic_mode"] = 0
    cfg["controller"].setdefault("adaptive_window", {})["enabled"] = False
    cfg["controller"].setdefault("rl", {})["reward"] = "paper"
    # merge overrides. 'reward' routes into controller.rl; 'safety' replaces the block wholesale
    # (clean per-variant setup); everything else is set on controller directly.
    for k, v in over.items():
        if k == "reward":
            cfg["controller"].setdefault("rl", {})["reward"] = v
        else:
            cfg["controller"][k] = v
    return cfg


def run_variant(tag, over):
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    cfg = build_cfg(over, out_dir)
    cfg_path = os.path.join(out_dir, "config.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    print(f"[run] {tag} ...", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), cfg_path], cwd=ROOT, check=True)


def load_series(tag):
    hits = sorted(glob.glob(os.path.join(OUT_ROOT, tag, "runs", "*", "windows.csv")),
                  key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(f"no windows.csv for {tag}")
    t, util, dmr, qos = [], [], [], []
    for r in csv.DictReader(open(hits[-1])):
        try:
            t.append(int(float(r["w_id"])))
            util.append(float(r["act_util"]))
            qos.append(float(r.get("avg_qos") or 0))
            nrel = float(r.get("n_job_releasable") or 0)
            mk = (float(r.get("n_miss_jobs") or 0) / nrel) if nrel > 0 else float(r.get("dmr") or 0)
            dmr.append(min(1.0, mk))
        except (KeyError, ValueError):
            continue
    avg, s = [], 0.0
    for i, d in enumerate(dmr):
        s += d; avg.append(s / (i + 1))
    return {"t": t, "util": util, "qos": qos, "dmr": dmr, "avg_dmr": avg}


def _smooth(y, w=21):
    if w <= 1 or len(y) < w:
        return y
    out, half = [], w // 2
    for i in range(len(y)):
        a, b = max(0, i - half), min(len(y), i + half + 1)
        out.append(sum(y[a:b]) / (b - a))
    return out


def theorem_bound(n):
    B = max(ALPHA, 1 - ALPHA)
    return [ALPHA + (LAM1 - RMIN + GAMMA * B) / (GAMMA * (k + 1)) for k in range(n)]


def plot_and_export():
    os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
    data = {tag: load_series(tag) for tag, *_ in VARIANTS}

    # ---- combined 4-panel: util / avg QoS / per-window DMR / cumulative avg DMR ----
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    panels = [("util", "Utilization", "util", None),
              ("qos", "Average QoS level (delivered service quality)", "avg QoS", None),
              ("dmr", "Deadline-miss ratio M_k (per window)", "M_k", ALPHA),
              ("avg_dmr", "Average deadline-miss ratio (cumulative)", "mean M_k", ALPHA)]
    for ax, (key, title, ylab, hl) in zip(axes, panels):
        for tag, label, color, _ in VARIANTS:
            y = _smooth(data[tag][key]) if key != "avg_dmr" else data[tag][key]
            ax.plot(data[tag]["t"], y, color=color, lw=1.6, label=label)
        if key == "avg_dmr":
            t = data["argos"]["t"]
            ax.plot(t, theorem_bound(len(t)), color="0.35", ls=":", lw=1.4,
                    label="ARGoS Theorem-1 bound")
            ax.set_ylim(0, 1.0)
        if hl is not None:
            ax.axhline(hl, color="black", ls="--", lw=0.9, alpha=0.7)
            ax.text(ax.get_xlim()[1], hl, f"  α={hl}", va="center", fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylab)
        ax.grid(True, ls="--", alpha=0.3, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("window index")
    fig.suptitle("ARGoS vs baselines (Case 2 invalid + Case 3 random profile)", fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    p = os.path.join(FIG_DIR, "case2_baselines.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)

    # ---- standalone avg-DMR (the headline: ARGoS bounded, baselines not) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, label, color, _ in VARIANTS:
        ax.plot(data[tag]["t"], data[tag]["avg_dmr"], color=color, lw=1.8, label=label)
    t = data["argos"]["t"]
    ax.plot(t, theorem_bound(len(t)), color="0.35", ls=":", lw=1.4, label="ARGoS Theorem-1 bound")
    ax.axhline(ALPHA, color="black", ls="--", lw=1.0, alpha=0.7)
    ax.text(ax.get_xlim()[1], ALPHA, f"  α={ALPHA}", va="center", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Cumulative deadline-miss ratio — ARGoS vs baselines", fontsize=11)
    ax.set_ylabel("mean M_k"); ax.set_xlabel("window index")
    ax.grid(True, ls="--", alpha=0.3, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "case2_avg_dmr.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)

    # ---- raw CSVs (window + per-variant), one per metric ----
    n = min(len(data[tag]["t"]) for tag, *_ in VARIANTS)
    win = data["argos"]["t"][:n]
    for key, fname in (("util", "case2_raw_util"), ("qos", "case2_raw_qos"),
                       ("dmr", "case2_raw_dmr"), ("avg_dmr", "case2_raw_avg_dmr")):
        path = os.path.join(DATA_DIR, f"{fname}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["window"] + [tag for tag, *_ in VARIANTS])
            for i in range(n):
                w.writerow([win[i]] + [round(data[tag][key][i], 6) for tag, *_ in VARIANTS])
        print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        for tag, _l, _c, over in VARIANTS:
            run_variant(tag, over)
    plot_and_export()


if __name__ == "__main__":
    main()
