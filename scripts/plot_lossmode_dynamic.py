"""Compare RCRL loss modes (binary vs fraction) on the dynamic-change suite.

Reads matched run pairs (same scenario, loss_mode=binary vs fraction) and produces:
  * util-tracking overlay grid (one panel per scenario, both modes + ref + change marks)
  * avg-QoS overlay grid
  * a summary bar grid (steady util, QoS/9, fallback rate, and the *binary* miss-window
    rate recomputed identically for both modes so safety is compared on equal footing).

No suptitles; large fonts; scenario name as a compact in-panel tag.
"""
import argparse, csv, glob, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BIN, C_FRAC, C_REF = "#0072B2", "#D55E00", "#555555"
SCEN_ORDER = ["ref_down", "ref_up", "add", "remove", "replace", "spaced"]


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#bbbbbb", "axes.linewidth": 1.1,
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.9,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 16, "axes.labelsize": 18, "xtick.labelsize": 14,
        "ytick.labelsize": 14, "legend.frameon": False, "legend.fontsize": 15,
    })


def _run(root, scen):
    d = os.path.join(root, scen)
    h = sorted(glob.glob(os.path.join(d, "runs", "*")), key=os.path.getmtime)
    return h[-1] if h else None


def _col(run, key):
    rows = list(csv.DictReader(open(os.path.join(run, "windows.csv"))))
    w = np.array([int(r["w_id"]) for r in rows])
    v = np.array([float(r.get(key) or 0) for r in rows])
    return w, v, rows


def _changes(run):
    d = json.load(open(os.path.join(run, "summary.json")))
    return [c.get("window") for c in (d.get("dynamic_changes") or []) if c.get("window") is not None]


def _tag(ax, txt):
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=15, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.85))


def _grid_single(root, key, ylabel, out, *, color, mode_label, ref_line=False, ylim=None):
    """One loss mode, 2x3 scenario grid, single line per panel (no overlay)."""
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    for ax, scen in zip(axes.flat, SCEN_ORDER):
        r = _run(root, scen)
        if not r:
            ax.set_visible(False); continue
        w, v, _ = _col(r, key)
        if ref_line:
            _, ref, _ = _col(r, "ref_util")
            ax.plot(w, ref, color=C_REF, lw=1.8, ls="--", label="ref", zorder=2)
        ax.plot(w, v, color=color, lw=1.4, alpha=0.95, label=mode_label, zorder=3)
        for ch in _changes(r):
            ax.axvline(ch, color="#999999", ls=(0, (4, 3)), lw=1.2, zorder=1)
        _tag(ax, scen)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("window"); ax.set_ylabel(ylabel)
    axes.flat[0].legend(loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[cmp] wrote {out}")


def _grid_guarantee(root, out, *, color, mode_label):
    """Running realized miss (the integrated loss) vs the Theorem-1 bound, per scenario.

    The running mean is over SHIELDED windows and uses the SAME loss the controller
    integrated: 1[dmr>0] for binary, dmr for fraction. x-axis is the actual window id so
    the change markers line up; the bound at the k-th shielded window uses t=k.
    """
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    frac = (mode_label == "fraction")
    for ax, scen in zip(axes.flat, SCEN_ORDER):
        r = _run(root, scen)
        if not r:
            ax.set_visible(False); continue
        rows = list(csv.DictReader(open(os.path.join(r, "windows.csv"))))
        d = json.load(open(os.path.join(r, "summary.json")))
        rc = d.get("rcrl", {})
        alpha = rc.get("target_alpha", 0.1); gamma = rc.get("gamma", 0.05)
        sh = [row for row in rows if str(row.get("shield_active")).lower() in ("true", "1")]
        base = sh if sh else rows
        w = np.array([int(row["w_id"]) for row in base])
        dmr = np.array([float(row.get("dmr") or 0) for row in base])
        loss = dmr if frac else (dmr > 0).astype(float)
        run_miss = np.cumsum(loss) / np.arange(1, len(loss) + 1)
        t = np.arange(1, len(loss) + 1)
        B = max(alpha, 1 - alpha)
        bound = alpha + (1 + gamma * B) / (gamma * t)
        ax.fill_between(w, bound, 1.0, color="#E69F00", alpha=0.05, zorder=0)
        ax.plot(w, bound, color="black", lw=2.0, ls=":", label="Theorem-1 bound", zorder=4)
        ax.axhline(alpha, color=C_REF, lw=1.6, ls="--", label=fr"$\alpha$={alpha:g}", zorder=3)
        ax.plot(w, run_miss, color=color, lw=2.0, label=f"running miss ({mode_label})", zorder=5)
        for ch in _changes(r):
            ax.axvline(ch, color="#999999", ls=(0, (4, 3)), lw=1.2, zorder=1)
        _tag(ax, scen)
        ax.set_ylim(0, min(1.0, max(0.25, alpha * 3)))
        ax.set_xlabel("window"); ax.set_ylabel("running miss ratio")
    axes.flat[0].legend(loc="upper right", ncol=1, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[cmp] wrote {out}")


def _steady_stats(run, warm_tail=500):
    rows = list(csv.DictReader(open(os.path.join(run, "windows.csv"))))
    n = len(rows)
    st = [r for r in rows if int(r["w_id"]) >= n - warm_tail]
    util = np.mean([float(r["act_util"]) for r in st])
    qos = np.mean([float(r["avg_qos"]) for r in st])
    fb = np.mean([float(r.get("safe_fallback") or 0) for r in rows])
    # binary miss-window rate over SHIELDED windows, recomputed identically for both modes
    sh = [r for r in rows if str(r.get("shield_active")).lower() in ("true", "1")]
    base = sh if sh else rows
    missrate = np.mean([1.0 if float(r.get("dmr") or 0) > 0 else 0.0 for r in base])
    return util, qos, fb, missrate


def _summary(binroot, fracroot, out):
    _style()
    util_b, qos_b, fb_b, mr_b, util_f, qos_f, fb_f, mr_f = ([] for _ in range(8))
    scen_ok = []
    for scen in SCEN_ORDER:
        rb, rf = _run(binroot, scen), _run(fracroot, scen)
        if not rb or not rf:
            continue
        ub, qb, fbb, mrb = _steady_stats(rb)
        uf, qf, fbf, mrf = _steady_stats(rf)
        scen_ok.append(scen)
        util_b.append(ub); qos_b.append(qb / 9); fb_b.append(fbb); mr_b.append(mrb)
        util_f.append(uf); qos_f.append(qf / 9); fb_f.append(fbf); mr_f.append(mrf)
    metrics = [("steady util", util_b, util_f),
               ("avg QoS / 9", qos_b, qos_f),
               ("fallback rate", fb_b, fb_f),
               ("miss-window rate", mr_b, mr_f)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x = np.arange(len(scen_ok)); w = 0.38
    for ax, (name, vb, vf) in zip(axes.flat, metrics):
        ax.bar(x - w / 2, vb, w, color=C_BIN, label="binary", zorder=2)
        ax.bar(x + w / 2, vf, w, color=C_FRAC, label="fraction", zorder=2)
        for xi, v in zip(x - w / 2, vb):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
        for xi, v in zip(x + w / 2, vf):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
        _tag(ax, name)
        ax.set_xticks(x); ax.set_xticklabels(scen_ok, rotation=20, fontsize=13)
        top = max(max(vb), max(vf))
        ax.set_ylim(0, top * 1.25 + 1e-6)
    axes.flat[0].legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[cmp] wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin-dir", default="outputs/dyn_loss/binary")
    p.add_argument("--frac-dir", default="outputs/dyn_loss/fraction")
    p.add_argument("--out-dir", default="outputs/sample/figures/9_dynamic_lossmode")
    args = p.parse_args()
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    modes = [("binary", args.bin_dir, C_BIN), ("fraction", args.frac_dir, C_FRAC)]
    for mlabel, root, col in modes:
        _grid_single(root, "act_util", "CPU utilization",
                     str(od / f"util_tracking_{mlabel}.png"),
                     color=col, mode_label=mlabel, ref_line=True, ylim=(0, 1.05))
        _grid_single(root, "avg_qos", "avg QoS",
                     str(od / f"qos_{mlabel}.png"), color=col, mode_label=mlabel)
        _grid_single(root, "dmr", "per-window DMR",
                     str(od / f"dmr_{mlabel}.png"), color=col, mode_label=mlabel, ylim=(0, None))
        _grid_guarantee(root, str(od / f"guarantee_{mlabel}.png"), color=col, mode_label=mlabel)
    _summary(args.bin_dir, args.frac_dir, str(od / "summary_bars.png"))


if __name__ == "__main__":
    main()
