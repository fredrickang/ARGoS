#!/usr/bin/env python
"""Three-case comparison figures on a single task set.

Runs ARGoS on ONE fixed task set (deterministic, seed 22) under:
  * Case 1  — valid profiles, analytical score, alpha=0, mode-change  -> hard guarantee
  * Case 2  — invalid profiles, analytical score, WCET underestimated by 50%
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
# Fixed window W = 10 s, N = 2000 windows (adaptive window OFF so every case shares the same
# window axis). x-axis is the window index throughout. Duration is N+1 windows: the timer event
# at exactly N*W coincides with the simulation end and does not fire, so N*W would log only N-1
# windows; (N+1)*W makes the N-th window complete, giving exactly N logged windows (w_id 0..N-1).
W_MS = 10_000
N_WINDOWS = 2000
DURATION_MS = W_MS * (N_WINDOWS + 1)

# Set by main(): 'paper' = Eq.1 (QoS gated on no-miss); 'v2' = legacy util-tracking reward.
REWARD = "paper"


def out_root():
    name = "fig3cases" if REWARD == "paper" else f"fig3cases_{REWARD}"
    return os.path.join(ROOT, "outputs", name)


def fig_path(key):
    pre = "" if REWARD == "paper" else f"{REWARD}_"
    return os.path.join(FIG_DIR, f"three_cases_{pre}{key}.png")

UNDEREST = [0.5]                                # Case-2 WCET underestimation level(s)
ALPHA = 0.1                                     # target for Cases 2 & 3

# --- colors: one distinct hue per case.
C1_COLOR = "#0B7A75"   # teal   (Case 1, hard guarantee)
C3_COLOR = "#6A3D9A"   # purple (Case 3, learned)
C2_RAMP = ["#FC4E2A"]  # orange-red (Case 2)


# Risk framing. Case 1 (analytical, valid) uses the analytical threshold lambda_init=1
# (accept U<=1). Cases 2 & 3 share lambda_init=0.5.
LAM_ANALYTIC = 1.0
LAM_SOFT = 0.5


def cases():
    """(tag, label, color, safety-overrides) for every curve, in legend order."""
    out = [("case1", "Case 1 (valid, hard)", C1_COLOR, dict(
        score="analytical", profile="valid", target_miss_alpha=0.0,
        lambda_init=LAM_ANALYTIC, mode_change=True))]
    for u, col in zip(UNDEREST, C2_RAMP):
        out.append((f"case2_u{int(u*100)}", f"Case 2 ({int(u*100)}% underest.)", col, dict(
            score="analytical", profile="invalid", profile_underest=u,
            target_miss_alpha=ALPHA, lambda_init=LAM_SOFT, mode_change=False)))
    out.append(("case3", "Case 3 (learned)", C3_COLOR, dict(
        score="learned", target_miss_alpha=ALPHA, lambda_init=LAM_SOFT, mode_change=False)))
    return out


def build_cfg(safety_over, out_dir):
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["simulation"]["duration_ms"] = DURATION_MS
    cfg.setdefault("window", {})["size_ms"] = W_MS           # fixed 10 s window
    cfg.setdefault("output", {})["dir"] = out_dir
    cfg["controller"]["dynamic_mode"] = 0
    cfg["controller"].setdefault("adaptive_window", {})["enabled"] = False  # fixed W, same axis
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
    """Return dict with t (WINDOW index), util, dmr, avg_dmr (cumulative mean).

    The x-axis is the window index (w_id). 'dmr' is the per-window miss ratio
    M_k = missed / max-releasable jobs (paper Eq. 1) — the quantity Theorem 1 bounds —
    from n_miss_jobs / n_job_releasable when available, else the recorded dmr.
    """
    path = latest_windows_csv(tag)
    if not path:
        raise FileNotFoundError(f"no windows.csv for {tag}")
    t, util, dmr = [], [], []
    for r in csv.DictReader(open(path)):
        try:
            t.append(int(float(r["w_id"])))        # x-axis = window index
            util.append(float(r["act_util"]))
            nrel = float(r.get("n_job_releasable") or 0)
            if nrel > 0:
                mk = float(r.get("n_miss_jobs") or 0) / nrel
            else:
                mk = float(r.get("dmr") or 0)
            dmr.append(min(1.0, mk))
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
    ax.set_xlabel("window index")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _rollband(y, w=21, env_smooth=15):
    """Rolling mean + a SMOOTH min/max envelope band. The raw rolling min/max plateaus at
    extremes (boxy on spiky 0/1 data), so the lo/hi envelopes are themselves smoothed with a
    moving average — the band then reflects the min/max range but hugs the mean smoothly
    (the shaded-confidence-band look). Lower edge clipped at 0."""
    n, half = len(y), w // 2
    m, lo, hi = [], [], []
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        seg = y[a:b]
        m.append(sum(seg) / len(seg)); lo.append(min(seg)); hi.append(max(seg))
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
        axes[-1].set_xlabel("window index")
        fig.supylabel(ylabel, fontsize=10)
        fig.suptitle(f"{metric_title} — {rlabel}  (line = 21-window mean, band = smoothed min–max)",
                     fontsize=11, y=0.998)
        fig.tight_layout(rect=[0.015, 0, 1, 0.99])
        p = fig_path(key)
        fig.savefig(p, dpi=140); plt.close(fig)
        print("saved", p)

    small_multiples("util", "Utilization", "actual utilization", ylim=(0, 1.05))
    small_multiples("dmr", "Deadline-miss ratio (per window)", "DMR", hline=ALPHA, ylim=(0, 1.05))

    # ---- cumulative average DMR + Theorem-1 bound ----
    # Theorem 1 (Eq.12, risk framing): (1/K) Σ ℓ_k ≤ α + (λ₁ − r_min + γB)/(γK),
    # B = max(α,1-α), r_min=0, γ=0.05. Decays like 1/K toward α; the constant depends on λ₁, so
    # analytical cases (λ₁=1) and the learned case (λ₁=0.5) have different envelopes.
    GAMMA, RMIN = 0.05, 0.0

    def theorem_bound(tvals, alpha, lam1):
        B = max(alpha, 1.0 - alpha)
        return [alpha + (lam1 - RMIN + GAMMA * B) / (GAMMA * (i + 1)) for i in range(len(tvals))]

    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, label, color, _ in specs:
        ax.plot(data[tag]["t"], data[tag]["avg_dmr"], color=color, lw=1.6, label=label)
    # bound envelopes (dashed): Cases 2 & 3 share (α=0.1, λ₁=0.5); Case 1 is (α=0, λ₁=1).
    t23 = data["case3"]["t"]
    ax.plot(t23, theorem_bound(t23, 0.1, LAM_SOFT), color="0.35", ls=":", lw=1.6,
            label="bound: Cases 2 & 3 (α=0.1, λ₁=0.5)")
    t00 = data["case1"]["t"]
    ax.plot(t00, theorem_bound(t00, 0.0, LAM_ANALYTIC), color=C1_COLOR, ls=":", lw=1.4,
            label="bound: Case 1 (α=0)")
    ax.axhline(ALPHA, color="black", ls="--", lw=1.0, alpha=0.6)
    ax.text(ax.get_xlim()[1], ALPHA, f"  α={ALPHA}", va="center", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Average deadline-miss ratio (cumulative) vs Theorem-1 bound — {rlabel}",
                 fontsize=11)
    ax.set_ylabel("mean DMR"); ax.set_xlabel("window index")
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
        if key == "avg_dmr":   # overlay Theorem-1 bound envelopes (depend on α and λ₁)
            ax.plot(t23, theorem_bound(t23, 0.1, LAM_SOFT), color="0.35", ls=":", lw=1.5,
                    label="bound: Cases 2 & 3 (λ₁=0.5)")
            ax.plot(t00, theorem_bound(t00, 0.0, LAM_ANALYTIC), color=C1_COLOR, ls=":", lw=1.3,
                    label="bound: Case 1 (α=0)")
            ax.set_ylim(0, 1.0)
        if hl is not None:
            ax.axhline(hl, color="black", ls="--", lw=0.9, alpha=0.6)
            ax.text(ax.get_xlim()[1], hl, f"  α={hl}", va="center", fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylab)
        ax.grid(True, ls="--", alpha=0.3, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    axes[-1].set_xlabel("window index")
    fig.suptitle(f"ARGoS three-case comparison — {rlabel}  (mean lines)", fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(fig_path("combined"), dpi=140); plt.close(fig)
    print("saved", fig_path("combined"))


def export_csv():
    """Write Excel-ready CSVs on a common time grid (cases have different window sizes, so each
    series is linearly interpolated onto one shared time axis). For util/dmr each case gets
    (lo, band=hi-lo, mean) so the band is built with the stacked-area trick (plot lo with
    No-fill, band with a translucent fill, mean as a line). avg_dmr gets per-case cumulative
    curves plus the three Theorem-1 bounds."""
    out_dir = os.path.join(FIG_DIR, "excel_data")
    os.makedirs(out_dir, exist_ok=True)
    specs = cases()
    data = {tag: load_series(tag) for tag, *_ in specs}
    # x-axis = window index. Fixed W + equal duration => every case has the same window count;
    # align directly by index (truncate to the shortest, just in case of a 1-window edge diff).
    n = min(len(data[tag]["t"]) for tag, *_ in specs)
    win = data[specs[0][0]]["t"][:n]
    pre = "" if REWARD == "paper" else f"{REWARD}_"

    # band CSVs (util, dmr): window + per-case lo / band / mean
    for key in ("util", "dmr"):
        header, cols = ["window"], {}
        for tag, *_ in specs:
            m, lo, hi = _rollband(data[tag][key])
            cols[tag] = (lo, hi, m)
            header += [f"{tag}_lo", f"{tag}_band", f"{tag}_mean"]
        path = os.path.join(out_dir, f"{pre}band_{key}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header)
            for i in range(n):
                row = [win[i]]
                for tag, *_ in specs:
                    lo, hi, m = cols[tag]
                    row += [round(lo[i], 5), round(hi[i] - lo[i], 5), round(m[i], 5)]
                w.writerow(row)
        print("saved", path)

    # avg_dmr CSV: window + per-case cumulative + the three Theorem-1 bounds (Eq. 12).
    # Bound at window K: alpha + (lambda_1 - r_min + gamma*B)/(gamma*K).
    GAMMA, RMIN = 0.05, 0.0
    def bound(alpha, lam1):
        B = max(alpha, 1 - alpha)
        return [alpha + (lam1 - RMIN + GAMMA * B) / (GAMMA * (k + 1)) for k in range(n)]
    b23, b1 = bound(0.1, LAM_SOFT), bound(0.0, LAM_ANALYTIC)
    path = os.path.join(out_dir, f"{pre}avg_dmr.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window"] + [f"{tag}_avg" for tag, *_ in specs] +
                   ["bound_cases23_lam05", "bound_case1_a0"])
        for i in range(n):
            row = [win[i]] + [round(data[tag]["avg_dmr"][i], 5) for tag, *_ in specs]
            row += [round(b23[i], 5), round(b1[i], 5)]
            w.writerow(row)
    print("saved", path)

    # RAW per-window values (no smoothing / no band): one file per metric, columns = cases.
    for key, fname in (("util", "raw_util"), ("dmr", "raw_dmr"), ("avg_dmr", "raw_avg_dmr")):
        path = os.path.join(out_dir, f"{pre}{fname}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["window"] + [tag for tag, *_ in specs])
            for i in range(n):
                w.writerow([win[i]] + [round(data[tag][key][i], 6) for tag, *_ in specs])
        print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--export", action="store_true",
                    help="also write Excel-ready CSVs (lo/band/mean per case) to figures/excel_data/")
    ap.add_argument("--reward", default="paper", choices=["paper", "v2", "decoupled_sum"],
                    help="RL reward: paper=Eq.1 (default), v2=legacy util-tracking")
    args = ap.parse_args()
    global REWARD
    REWARD = args.reward
    if not args.plot_only:
        for tag, _lbl, _col, over in cases():
            run_case(tag, over)
    plot()
    if args.export:
        export_csv()


if __name__ == "__main__":
    main()
