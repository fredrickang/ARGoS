"""Initial-epsilon sweep: find the exploration rate that learns best with fewest spikes.

The DQN decays epsilon only on explore steps (dqn.py), so from any start it reaches ~0
within the first ~50-100 windows; the *initial* epsilon therefore governs how much the
agent explores during the warm-up (the main source of utilisation-saturation spikes) and
how good a policy it settles into. This sweeps initial epsilon on a static env and reports
learning quality (settle, steady-state util/QoS), the spike cost (saturation rate, split
warm-up vs steady), and safety (realized miss vs the Theorem-1 bound).

Example:
    python scripts/run_epsilon_sweep.py --epsilons 0.0 0.1 0.3 0.5 0.7 0.9 --workers 6
"""

import argparse
import copy
import csv
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_config(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def build_run_config(base, *, epsilon, decay, out_dir, duration_ms, seed=None):
    cfg = copy.deepcopy(base)
    cfg["controller"]["mode"] = "rl"
    cfg["controller"]["dynamic_mode"] = 0  # static: isolate learning quality
    rl = dict(cfg["controller"].get("rl") or {})
    rl["epsilon"] = float(epsilon)
    if decay is not None:
        rl["epsilon_decay"] = float(decay)
    rl["epsilon_reset"] = 0.0  # new default; irrelevant in a static run but keep consistent
    if seed is not None:
        rl["seed"] = int(seed)
    cfg["controller"]["rl"] = rl
    cfg.setdefault("output", {})["dir"] = str(out_dir)
    if duration_ms is not None:
        cfg["simulation"]["duration_ms"] = int(duration_ms)
    return cfg


def run_one(args):
    run_py, cfg_path, workdir = args
    subprocess.run([sys.executable, str(run_py), str(cfg_path)], cwd=workdir, check=True)
    return cfg_path


def latest_run(out_dir):
    hits = sorted(glob.glob(os.path.join(out_dir, "runs", "*")), key=os.path.getmtime)
    return hits[-1] if hits else None


def is_true(v):
    return v in ("1", "True", "true")


def sat_rate(rows, a, b):
    seg = [r for r in rows if a <= int(r["w_id"]) < b]
    if not seg:
        return None
    return sum(1 for r in seg if float(r["act_util"]) >= 0.999) / len(seg)


def collect_row(eps, out_dir, warmup_end=200, tail_frac=0.25):
    row = {"epsilon": eps, "out_dir": out_dir}
    run_dir = latest_run(out_dir)
    if not run_dir:
        row["status"] = "MISSING"
        return row
    d = json.load(open(os.path.join(run_dir, "summary.json")))
    r = d.get("rcrl", {})
    o = d.get("overall", {})
    settle = d.get("settle", {})
    rows = list(csv.DictReader(open(os.path.join(run_dir, "windows.csv"))))
    n = len(rows)
    tail_start = int(n * (1.0 - tail_frac))
    ss = [rows[i] for i in range(tail_start, n)]
    ss_util = sum(float(x["act_util"]) for x in ss) / len(ss) if ss else None
    ss_qos = sum(float(x["avg_qos"]) for x in ss) / len(ss) if ss else None
    row.update({
        "status": "ok",
        "realized_miss": r.get("realized_miss_rate"),
        "bound": r.get("theorem_bound"),
        "guar_ok": r.get("guarantee_satisfied"),
        "fallback_rate": r.get("fallback_rate"),
        "settle_ms": settle.get("settle_time_ms"),
        "avg_util": o.get("avg_act_util"),
        "ss_util": ss_util,
        "ss_qos": ss_qos,
        "sat_all": sat_rate(rows, 0, n),
        "sat_warmup": sat_rate(rows, 0, warmup_end),
        "sat_steady": sat_rate(rows, tail_start, n),
    })
    return row


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return "-" if v is None else str(v)


def aggregate(eps, rows):
    """Mean over seed replicates for one epsilon."""
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"epsilon": eps, "status": "MISSING", "n": 0}
    keys = ["realized_miss", "fallback_rate", "settle_ms", "avg_util", "ss_util",
            "ss_qos", "sat_all", "sat_warmup", "sat_steady"]
    agg = {"epsilon": eps, "status": "ok", "n": len(ok),
           "guar_ok": all(r.get("guar_ok") for r in ok)}
    import statistics
    for k in keys:
        vals = [r[k] for r in ok if r.get(k) is not None]
        agg[k] = statistics.fmean(vals) if vals else None
        agg[k + "_sd"] = (statistics.pstdev(vals) if len(vals) > 1 else 0.0) if vals else None
    return agg


def pick_best(aggs):
    ok = [a for a in aggs if a.get("status") == "ok" and a.get("guar_ok")]
    if not ok:
        return None
    # reward high steady util + QoS, penalise steady-state spikes (warm-up spikes are
    # a transient cost, weighted lightly)
    def score(a):
        return ((a.get("ss_util") or 0) + 0.05 * (a.get("ss_qos") or 0)
                - 1.5 * (a.get("sat_steady") or 0) - 0.15 * (a.get("sat_warmup") or 0))
    return max(ok, key=score)


def main():
    p = argparse.ArgumentParser(description="Initial-epsilon sweep (static env).")
    p.add_argument("--config", default="./configs/run_sim.yaml")
    p.add_argument("--epsilons", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--seeds", nargs="+", type=int, default=[22, 7, 99],
                   help="RNG seeds to average over (replicates).")
    p.add_argument("--decay", type=float, default=None, help="Override epsilon_decay (else base).")
    p.add_argument("--duration-ms", type=int, default=10000000)
    p.add_argument("--out-dir", default="outputs/epsilon_sweep")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    workdir = Path(__file__).resolve().parents[1]
    run_py = workdir / "run.py"
    base = load_config(args.config)
    out_root = Path(args.out_dir)
    cfg_root = out_root / "_configs"

    jobs, plan = [], []  # plan: (eps, seed, out_dir)
    for eps in args.epsilons:
        for seed in args.seeds:
            tag = f"eps{eps}_s{seed}"
            out_dir = out_root / tag
            plan.append((eps, seed, str(out_dir)))
            if args.skip_existing and latest_run(str(out_dir)):
                continue
            cfg = build_run_config(base, epsilon=eps, decay=args.decay,
                                   out_dir=out_dir, duration_ms=args.duration_ms, seed=seed)
            cfg_path = cfg_root / f"{tag}.yaml"
            write_config(cfg_path, cfg)
            jobs.append((run_py, cfg_path, workdir))

    print(f"[eps] {len(jobs)} run(s) to execute "
          f"({len(args.epsilons)} eps x {len(args.seeds)} seeds)")
    if jobs:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_one, j) for j in jobs]
                for f in as_completed(futs):
                    f.result()
        else:
            for j in jobs:
                run_one(j)

    # per-run rows, then aggregate over seeds per epsilon
    per_run = {}
    for eps, seed, out_dir in plan:
        per_run.setdefault(eps, []).append(collect_row(eps, out_dir))
    aggs = [aggregate(eps, per_run[eps]) for eps in args.epsilons]

    cols = ["epsilon", "n", "realized_miss", "guar_ok", "fallback_rate", "settle_ms",
            "avg_util", "ss_util", "ss_qos", "sat_all", "sat_warmup", "sat_steady", "status"]
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "epsilon_summary.csv"
    with open(csv_path, "w", newline="") as f:
        allcols = cols + [c + "_sd" for c in ["realized_miss", "settle_ms", "ss_util",
                                              "ss_qos", "sat_all", "sat_warmup", "sat_steady"]]
        w = csv.DictWriter(f, fieldnames=allcols, extrasaction="ignore")
        w.writeheader()
        for a in aggs:
            w.writerow(a)

    widths = {c: max(len(c), *(len(fmt(a.get(c))) for a in aggs)) for c in cols}
    print("\n(mean over seeds; sat_* = utilisation-saturation rate)")
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for a in aggs:
        print(" | ".join(fmt(a.get(c)).ljust(widths[c]) for c in cols))

    best = pick_best(aggs)
    print(f"\n[eps] wrote {csv_path}")
    if best:
        print(f"[eps] BEST epsilon = {best['epsilon']}  "
              f"(ss_util={fmt(best.get('ss_util'))}+/-{fmt(best.get('ss_util_sd'))}, "
              f"ss_qos={fmt(best.get('ss_qos'))}, "
              f"sat_steady={fmt(best.get('sat_steady'))}, "
              f"settle_ms={fmt(best.get('settle_ms'))})")
    else:
        print("[eps] no run satisfied the guarantee.")


if __name__ == "__main__":
    main()
