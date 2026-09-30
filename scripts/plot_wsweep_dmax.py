"""Does the window-size sweet spot track the deadline scale D_max?

Reads several window-size sweeps run at different task period ranges (different D_max) and
overlays their normalized composite cost (adaptation = settle time; feedback lag = jitter)
against W / D_max. If the sweet spot ~ D_max, all curves bottom out near W/D_max = 1.
A second panel plots sweet-spot W vs D_max (expected ~ the line W = D_max).
No titles; large fonts.
"""
import argparse, csv, glob, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00"]


def _agg(path):
    by_w = defaultdict(list)
    dmax = None
    for r in csv.DictReader(open(path)):
        def f(k):
            v = r.get(k)
            return float(v) if v not in (None, "", "None") else np.nan
        by_w[int(r["W"])].append((f("settle_ms"), f("ustd"), f("lag"), f("track_err")))
        if r.get("dmax") not in (None, "", "None"):
            dmax = float(r["dmax"])
    Ws = np.array(sorted(by_w), dtype=float)
    settle = np.array([np.nanmean([d[0] for d in by_w[w]]) for w in Ws])
    ustd = np.array([np.nanmean([d[1] for d in by_w[w]]) for w in Ws])
    lag = np.array([np.nanmean([d[2] for d in by_w[w]]) for w in Ws])
    track = np.array([np.nanmean([d[3] for d in by_w[w]]) for w in Ws])
    return Ws, settle, ustd, lag, dmax, track


def _norm(x):
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+",
                   default=sorted(glob.glob("outputs/wsweep_dmax/d*")))
    p.add_argument("--out-dir", default="outputs/sample/figures/13_wsweep_dmax")
    p.add_argument("--tau", type=float, default=0.05, help="tracking-error knee threshold")
    args = p.parse_args()
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)

    TAU = args.tau
    groups = []
    for d in args.dirs:
        csvp = os.path.join(d, "window_size_results.csv")
        if not os.path.exists(csvp):
            print(f"[dmax] missing {csvp}"); continue
        Ws, settle, ustd, lag, dmax, track = _agg(csvp)
        # knee = smallest window that reaches good tracking (|util-ref| <= tau): the point
        # where the release-lag stops destabilising control. Robust, unlike a normalized argmin.
        good = np.where(track <= TAU)[0]
        knee_W = Ws[good[0]] if len(good) else Ws[-1]
        groups.append(dict(Ws=Ws, settle=settle, ustd=ustd, lag=lag, dmax=dmax,
                           track=track, knee_W=knee_W))
        print(f"[dmax] D_max={dmax:.0f}  knee_W(track<= {TAU})={knee_W:.0f}  (W/D_max={knee_W/dmax:.2f})")

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#bbbbbb", "axes.linewidth": 1.2,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 1.0,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 19, "axes.labelsize": 22, "xtick.labelsize": 16,
        "ytick.labelsize": 16, "legend.frameon": True, "legend.fontsize": 16,
        "legend.framealpha": 0.9, "legend.edgecolor": "#ccc",
    })

    # (A) tracking error vs W/D_max -- the feedback-lag penalty collapses near W ~ D_max,
    # regardless of the deadline scale -> the control-quality knee tracks D_max.
    fig, ax = plt.subplots(figsize=(11, 7))
    for g, c in zip(groups, COLORS):
        x = g["Ws"] / g["dmax"]
        ax.plot(x, g["track"], "o-", color=c, lw=2.8, ms=9, label=f"D_max≈{g['dmax']:.0f} ms")
        ax.plot(g["knee_W"] / g["dmax"], TAU, "*", color=c, ms=22, mec="black", mew=0.8, zorder=5)
    ax.axhline(TAU, color="#666", ls="--", lw=1.8, zorder=1, label=f"track thresh {TAU:g}")
    ax.axvline(1.0, color="#999", ls=":", lw=2.0, zorder=1, label="W = D_max")
    ax.set_xscale("log")
    ax.set_xlabel("W / D_max"); ax.set_ylabel("tracking error  |mean util − ref|")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(od / "trackerr_vs_WoverDmax.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[dmax] wrote {od/'trackerr_vs_WoverDmax.png'}")

    # (B) knee window (smallest W with good tracking) vs D_max -- expect ~ line W = D_max
    fig, ax = plt.subplots(figsize=(9, 7))
    dm = np.array([g["dmax"] for g in groups])
    sw = np.array([g["knee_W"] for g in groups])
    lo, hi = min(dm.min(), sw.min()) * 0.6, max(dm.max(), sw.max()) * 1.5
    ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=2.0, label="W = D_max")
    for g, c in zip(groups, COLORS):
        ax.plot(g["dmax"], g["knee_W"], "o", color=c, ms=17, mec="black", mew=0.8,
                label=f"D_max≈{g['dmax']:.0f} → W*≈{g['knee_W']:.0f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("D_max (ms)"); ax.set_ylabel("control-quality knee window  W* (ms)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(od / "knee_vs_Dmax.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[dmax] wrote {od/'knee_vs_Dmax.png'}")


if __name__ == "__main__":
    main()
