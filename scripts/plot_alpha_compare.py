"""ref=0.9, tight alpha probe: RCRL (safety ON) vs no-risk-control (safety OFF).

Left: running deadline-miss envelope (mean + min/max band) for both, with the
Theorem-1 bound and target alpha at the tightened value. Right: the utilisation /
QoS / shield-activity price of the tighter guarantee (RCRL alpha=0.1, alpha=0.01,
no-control).
"""
import argparse, csv, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMA = 0.05
C_ON, C_OFF = "#0072B2", "#D7191C"


def _rundir(tag):
    h = sorted(glob.glob(os.path.join(tag, "runs", "*")), key=os.path.getmtime)
    return h[-1] if h else None


def _miss(run, warm=200):
    rows = list(csv.DictReader(open(os.path.join(run, "windows.csv"))))
    return np.array([float(r["miss"] or 0) for r in rows if int(r["w_id"]) >= warm])


def _steady(run):
    rows = list(csv.DictReader(open(os.path.join(run, "windows.csv"))))
    n = len(rows)
    st = [r for r in rows if int(r["w_id"]) >= n - 500]
    return (np.mean([float(r["act_util"]) for r in st]),
            np.mean([float(r["avg_qos"]) for r in st]),
            np.mean([float(r.get("safe_fallback") or 0) for r in rows]))


def _runs(root, seeds=(1,)):
    out = []
    for t in sorted(glob.glob(os.path.join(root, "ts*_r0.9_s*"))):
        if not any(t.endswith(f"_s{s}") for s in seeds):
            continue
        r = _rundir(t)
        if r:
            out.append(r)
    return out


def _env(runs):
    curves = [np.cumsum(m) / np.arange(1, len(m) + 1) for m in (_miss(r) for r in runs) if len(m)]
    L = min(len(c) for c in curves)
    M = np.vstack([c[:L] for c in curves])
    return np.arange(1, L + 1), M.mean(0), M.min(0), M.max(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--on-dir", default="outputs/alpha001_ref09")
    p.add_argument("--off-dir", default="outputs/settling_success_nosafety")
    p.add_argument("--ref-dir", default="outputs/settling_success")  # alpha=0.1 baseline
    p.add_argument("--out", default="outputs/sample/figures/7_settling/alpha001_ref09.png")
    args = p.parse_args()

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#bbbbbb", "axes.linewidth": 1.2,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 1.0,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 18, "axes.labelsize": 21, "xtick.labelsize": 16,
        "ytick.labelsize": 16, "legend.frameon": True, "legend.fontsize": 15,
        "legend.framealpha": 0.92, "legend.edgecolor": "#cccccc",
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.45, 1]})

    # ---- (A) running-miss envelope
    for runs, c, lab in [(_runs(args.off_dir), C_OFF, "no risk control"),
                         (_runs(args.on_dir), C_ON, fr"RCRL, $\alpha$={args.alpha:g}")]:
        t, mean, lo, hi = _env(runs)
        axA.fill_between(t, lo, hi, color=c, alpha=0.12, zorder=1)
        axA.plot(t, hi, color=c, lw=1.0, ls="--", alpha=0.5, zorder=2)
        axA.plot(t, mean, color=c, lw=3.2, zorder=4, label=f"{lab}  (mean)")
    tt = np.arange(1, int(axA.get_xlim()[1]) + 1)
    B = max(args.alpha, 1 - args.alpha)
    axA.plot(tt, args.alpha + (1 + GAMMA * B) / (GAMMA * tt), color="black", lw=2.4, ls=":",
             zorder=6, label="Theorem-1 bound")
    axA.axhline(args.alpha, color="#666666", lw=1.8, ls="--", zorder=3,
                label=fr"target $\alpha$={args.alpha:g}")
    axA.set_xlabel("shielded window  t"); axA.set_ylabel("running miss ratio")
    axA.set_ylim(0, 0.10); axA.set_xlim(0, None)
    axA.legend(loc="upper right")

    # ---- (B) price of the guarantee: util / QoS / fallback
    conds = [("no control", args.off_dir, C_OFF),
             (r"RCRL $\alpha$=0.1", args.ref_dir, "#8aa"),
             (r"RCRL $\alpha$=0.01", args.on_dir, C_ON)]
    util, qos, fbk = [], [], []
    for _, root, _c in conds:
        vals = np.array([_steady(r) for r in _runs(root)])
        util.append(vals[:, 0].mean()); qos.append(vals[:, 1].mean()); fbk.append(vals[:, 2].mean())
    x = np.arange(len(conds)); w = 0.6
    labels = [c[0] for c in conds]; cols = [c[2] for c in conds]
    # normalise QoS(0-9) and fallback(0-1) onto the util axis via twin? keep simple: grouped
    metrics = [("steady util", util, 1.0), ("avg QoS /9", [q / 9 for q in qos], 1.0),
               ("fallback rate", fbk, 1.0)]
    gw = w / len(metrics)
    hatches = ["", "//", ".."]
    for mi, (mname, vals, _s) in enumerate(metrics):
        axB.bar(x + (mi - 1) * gw, vals, gw, color=cols, alpha=0.9, hatch=hatches[mi],
                edgecolor="white", label=mname, zorder=2)
        for xi, v in zip(x + (mi - 1) * gw, vals):
            axB.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=15)
    axB.set_ylim(0, 1.05); axB.set_ylabel("value (util, QoS/9, fallback)")
    axB.legend(loc="upper center", ncol=3, fontsize=13)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[alpha-cmp] wrote {args.out}")


if __name__ == "__main__":
    main()
