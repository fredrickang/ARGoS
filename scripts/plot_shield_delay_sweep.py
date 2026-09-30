"""Compare shield-start delays: utilisation, cumulative miss, and fallback over time."""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_windows(run_dir):
    with open(os.path.join(run_dir, "windows.csv"), newline="") as f:
        return list(csv.DictReader(f))


def latest_run(out_dir):
    hits = sorted(glob.glob(os.path.join(out_dir, "runs", "*")), key=os.path.getmtime)
    return hits[-1] if hits else None


def is_true(v):
    return v in ("1", "True", "true")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/shield_delay_sweep")
    p.add_argument("--delays", nargs="+", type=int, default=[0, 100, 200, 300, 500])
    p.add_argument("--out", default="outputs/sample/figures/3_rl_behavior/shield_delay_sweep.png")
    p.add_argument("--roll", type=int, default=50)
    args = p.parse_args()

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    ax_u, ax_m, ax_f = axes

    for delay in args.delays:
        run_dir = latest_run(os.path.join(args.root, f"delay{delay}"))
        if not run_dir:
            continue
        rows = load_windows(run_dir)
        w = [int(r["w_id"]) for r in rows]
        util = [float(r["act_util"]) for r in rows]
        miss = [1.0 if is_true(r.get("miss")) else 0.0 for r in rows]
        fb = [1.0 if is_true(r.get("safe_fallback")) else 0.0 for r in rows]

        # rolling means
        def roll(x):
            k = args.roll
            out, s = [], 0.0
            for i, v in enumerate(x):
                s += v
                if i >= k:
                    s -= x[i - k]
                out.append(s / min(i + 1, k))
            return out

        cum_miss = [sum(miss[:i + 1]) for i in range(len(miss))]
        lbl = f"start={delay}"
        ax_u.plot(w, roll(util), label=lbl, lw=1.2)
        ax_m.plot(w, cum_miss, label=lbl, lw=1.2)
        ax_f.plot(w, roll(fb), label=lbl, lw=1.2)
        ax_f.axvline(delay, color=ax_f.lines[-1].get_color(), ls=":", lw=0.8, alpha=0.5)

    ax_u.set_ylabel(f"act_util (roll{args.roll})")
    ax_u.set_title("Shield-start-delay sweep (static env, 2000 windows)")
    ax_m.set_ylabel("cumulative miss windows")
    ax_f.set_ylabel(f"fallback rate (roll{args.roll})")
    ax_f.set_xlabel("window")
    for ax in axes:
        ax.legend(fontsize=8, ncol=5)
        ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
