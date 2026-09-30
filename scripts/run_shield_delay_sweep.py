"""Shield-start-delay sweep: pick the default ``fallback_start_window``.

Delaying the RCRL shield lets the RL agent learn unshielded first, so by the time
the shield engages the greedy policy is closer to self-sufficient (lower steady-state
fallback rate, higher utilisation). Too long a delay wastes horizon under an unguarded
policy and accrues warm-up misses. This sweep runs a static env at several delays and
reports both the Theorem-1 safety accounting (over shielded windows) and the
performance / self-sufficiency signal so we can choose the default.

Example:
    python scripts/run_shield_delay_sweep.py --delays 0 100 200 300 500 \
        --duration-ms 10000000 --workers 5
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


def build_run_config(base, *, start_window, out_dir, duration_ms):
    cfg = copy.deepcopy(base)
    cfg["controller"]["mode"] = "rl"
    cfg["controller"]["dynamic_mode"] = 0  # static env: isolate the delay effect
    safety = dict(cfg["controller"].get("safety") or {})
    safety["mode"] = "rcrl"
    safety["fallback_enabled"] = True
    safety["fallback_start_window"] = int(start_window)
    cfg["controller"]["safety"] = safety
    cfg.setdefault("output", {})["dir"] = str(out_dir)
    if duration_ms is not None:
        cfg["simulation"]["duration_ms"] = int(duration_ms)
    return cfg


def run_one(args):
    run_py, cfg_path, workdir = args
    subprocess.run([sys.executable, str(run_py), str(cfg_path)], cwd=workdir, check=True)
    return cfg_path


def find_run_dir(out_dir):
    hits = sorted(glob.glob(os.path.join(out_dir, "runs", "*")),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def read_windows(run_dir):
    path = os.path.join(run_dir, "windows.csv")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def tail_stats(rows, frac=0.25):
    """Steady-state signal from the last ``frac`` of the run."""
    if not rows:
        return {}
    n = len(rows)
    tail = rows[int(n * (1.0 - frac)):]
    def favg(key):
        vals = [float(r[key]) for r in tail if r.get(key) not in (None, "")]
        return sum(vals) / len(vals) if vals else None
    fb = [1.0 if r.get("safe_fallback") in ("1", "True", "true") else 0.0 for r in tail]
    miss = [1.0 if r.get("miss") in ("1", "True", "true") else 0.0 for r in tail]
    return {
        "ss_util": favg("act_util"),
        "ss_qos": favg("avg_qos"),
        "ss_fallback_rate": sum(fb) / len(fb) if fb else None,
        "ss_miss_rate": sum(miss) / len(miss) if miss else None,
    }


def warmup_stats(rows):
    """Misses accrued while the shield was disengaged (shield_active == 0)."""
    def is_true(v):
        return v in ("1", "True", "true")
    warm = [r for r in rows if not is_true(r.get("shield_active"))]
    n = len(warm)
    misses = sum(1 for r in warm if is_true(r.get("miss")))
    return {"warmup_windows": n,
            "warmup_miss_windows": misses,
            "warmup_miss_rate": (misses / n) if n else None}


def collect_row(delay, out_dir):
    row = {"start_window": delay, "out_dir": out_dir}
    run_dir = find_run_dir(out_dir)
    if not run_dir:
        row["status"] = "MISSING"
        return row
    summ = os.path.join(run_dir, "summary.json")
    d = json.load(open(summ)) if os.path.exists(summ) else {}
    rcrl = d.get("rcrl", {})
    overall = d.get("overall", {})
    rows = read_windows(run_dir)
    row.update({
        "status": "ok",
        "windows": rcrl.get("total_windows", overall.get("total_windows")),
        "realized_miss": rcrl.get("realized_miss_rate"),
        "overall_miss": rcrl.get("overall_miss_rate"),
        "alpha": rcrl.get("target_alpha"),
        "bound": rcrl.get("theorem_bound"),
        "guar_ok": rcrl.get("guarantee_satisfied"),
        "fallback_rate": rcrl.get("fallback_rate"),
        "lambda_final": rcrl.get("lambda_final"),
        "avg_util": overall.get("avg_act_util"),
    })
    row.update(tail_stats(rows))
    row.update(warmup_stats(rows))
    return row


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return "-" if v is None else str(v)


def pick_best(rows):
    """Best = safety holds, then most self-sufficient (low steady fallback) at high util."""
    ok = [r for r in rows if r.get("status") == "ok" and r.get("guar_ok")]
    if not ok:
        return None
    # score: reward low steady-state fallback + high util, penalise warm-up misses
    def score(r):
        ss_fb = r.get("ss_fallback_rate") or 1.0
        ss_util = r.get("ss_util") or 0.0
        warm = r.get("warmup_miss_rate") or 0.0
        return ss_util - ss_fb - 0.5 * warm
    return max(ok, key=score)


def main():
    p = argparse.ArgumentParser(description="Shield-start-delay sweep.")
    p.add_argument("--config", default="./configs/run_sim.yaml")
    p.add_argument("--delays", nargs="+", type=int, default=[0, 100, 200, 300, 500])
    p.add_argument("--duration-ms", type=int, default=10000000)  # 2000 windows @ 5000ms
    p.add_argument("--out-dir", default="outputs/shield_delay_sweep")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    workdir = Path(__file__).resolve().parents[1]
    run_py = workdir / "run.py"
    base = load_config(args.config)
    out_root = Path(args.out_dir)
    cfg_root = out_root / "_configs"

    jobs, plan = [], []
    for delay in args.delays:
        tag = f"delay{delay}"
        out_dir = out_root / tag
        plan.append((delay, str(out_dir)))
        if args.skip_existing and find_run_dir(str(out_dir)):
            continue
        cfg = build_run_config(base, start_window=delay, out_dir=out_dir,
                               duration_ms=args.duration_ms)
        cfg_path = cfg_root / f"{tag}.yaml"
        write_config(cfg_path, cfg)
        jobs.append((run_py, cfg_path, workdir))

    print(f"[sweep] {len(jobs)} run(s) to execute, {len(plan)} delay(s) total")
    if jobs:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_one, j) for j in jobs]
                for f in as_completed(futs):
                    f.result()
        else:
            for j in jobs:
                run_one(j)

    rows = [collect_row(delay, out_dir) for delay, out_dir in plan]

    cols = ["start_window", "windows", "realized_miss", "overall_miss", "alpha",
            "bound", "guar_ok", "fallback_rate", "ss_util", "ss_fallback_rate",
            "ss_miss_rate", "warmup_miss_rate", "avg_util", "status"]
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "shield_delay_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["out_dir"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    print()
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols))

    best = pick_best(rows)
    print(f"\n[sweep] wrote {csv_path}")
    if best:
        print(f"[sweep] BEST delay = {best['start_window']}  "
              f"(ss_util={fmt(best.get('ss_util'))}, "
              f"ss_fallback={fmt(best.get('ss_fallback_rate'))}, "
              f"guarantee={best.get('guar_ok')}, "
              f"warmup_miss={fmt(best.get('warmup_miss_rate'))})")
    else:
        print("[sweep] no run satisfied the guarantee; inspect table above.")


if __name__ == "__main__":
    main()
