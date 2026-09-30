"""Dynamic-environment test suite for the delayed-shield (start=200) default.

Runs each canonical env-change type as a single change well after the shield has
engaged and the agent has settled, plus a spaced multi-change stress run. For each
run it reports the Theorem-1 safety accounting (over shielded windows) and the
per-segment re-settle time, so we can confirm the guarantee holds *through* changes
and the agent re-adapts.

Scenarios (2000 windows, change(s) after the delay-200 warm-up + settle):
    ref_down : ref_util 0.9 -> 0.7  @1000   (target relaxes)
    ref_up   : ref_util 0.9 -> 0.95 @1000   (target tightens)
    add      : +2 tasks             @1000   (load rises)
    remove   : -2 tasks             @1000   (load falls)
    replace  : replace 2 tasks      @1000   (churn, load ~constant)
    spaced   : ref/add/remove/replace at 400/800/1200/1600 (multi-change stress)

Example:
    python scripts/run_dynamic_experiments.py --workers 6
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


# Scenarios assume the run STARTS at ref_util=0.7 (configs/run_sim.yaml) and change from there.
SCENARIOS = {
    "ref_down": [{"at_window": 1000, "kind": "ref_util", "value": 0.5}],   # 0.7 -> 0.5 (relax)
    "ref_up":   [{"at_window": 1000, "kind": "ref_util", "value": 0.9}],   # 0.7 -> 0.9 (tighten)
    "add":      [{"at_window": 1000, "kind": "taskset", "op": "add", "count": 2, "util": 0.15}],
    "remove":   [{"at_window": 1000, "kind": "taskset", "op": "remove", "count": 2}],
    "replace":  [{"at_window": 1000, "kind": "taskset", "op": "replace", "count": 2, "util": 0.15}],
    "spaced": [
        {"at_window": 400,  "kind": "ref_util", "value": 0.5},
        {"at_window": 800,  "kind": "taskset", "op": "add", "count": 2, "util": 0.15},
        {"at_window": 1200, "kind": "taskset", "op": "remove", "count": 2},
        {"at_window": 1600, "kind": "taskset", "op": "replace", "count": 2, "util": 0.15},
    ],
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_config(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def build_run_config(base, *, changes, out_dir, duration_ms, adaptation, pool_extra,
                     epsilon_reset=None, loss_mode=None, alpha=None):
    cfg = copy.deepcopy(base)
    cfg["controller"]["mode"] = "rl"
    if epsilon_reset is not None:
        cfg["controller"].setdefault("rl", {})["epsilon_reset"] = float(epsilon_reset)
    # Schedule-driven changes take precedence; dynamic_mode is ignored when changes present,
    # but set to a nonzero value so nothing falls back to the static path.
    cfg["controller"]["dynamic_mode"] = 1
    safety = dict(cfg["controller"].get("safety") or {})
    safety["mode"] = "rcrl"
    safety["fallback_enabled"] = True
    # inherit the new default (200) from base unless base was tampered with
    safety.setdefault("fallback_start_window", 200)
    if loss_mode is not None:
        safety["loss_mode"] = str(loss_mode)
    if alpha is not None:
        safety["target_miss_alpha"] = float(alpha)
    cfg["controller"]["safety"] = safety
    dyn = dict(cfg.get("dynamic_config") or {})
    dyn["changes"] = changes
    dyn["adaptation"] = adaptation
    # Only reserve dormant pool tasks for scenarios that actually add/replace. Extra dormant
    # tasks enlarge the RL action space (14 vs 10) and, via the larger untrained network,
    # inflate the unguarded warm-up saturation ~8x -- so ref/remove runs get pool_extra=0.
    needed = max([int(e.get("count", 0)) for e in changes
                  if e.get("kind") == "taskset" and e.get("op") in ("add", "replace")]
                 + [0])
    dyn["pool_extra"] = needed if pool_extra is None else pool_extra
    dyn["dynamic_window"] = False
    cfg["dynamic_config"] = dyn
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


def collect_row(name, out_dir):
    row = {"scenario": name, "out_dir": out_dir}
    run_dir = latest_run(out_dir)
    if not run_dir:
        row["status"] = "MISSING"
        return row
    d = json.load(open(os.path.join(run_dir, "summary.json")))
    r = d.get("rcrl", {})
    o = d.get("overall", {})
    segs = d.get("settle_by_segment", []) or []
    # settle time of the segment AFTER the (first) change
    post = None
    if len(segs) >= 2:
        post = segs[-1].get("settle", {}).get("settle_time_ms")
    row.update({
        "status": "ok",
        "run_dir": run_dir,
        "windows": r.get("total_windows"),
        "realized_miss": r.get("realized_miss_rate"),
        "overall_miss": r.get("overall_miss_rate"),
        "alpha": r.get("target_alpha"),
        "bound": r.get("theorem_bound"),
        "guar_ok": r.get("guarantee_satisfied"),
        "fallback_rate": r.get("fallback_rate"),
        "avg_util": o.get("avg_act_util"),
        "avg_qos": o.get("avg_qos"),
        "post_change_settle_ms": post,
        "n_changes": len(d.get("dynamic_changes", []) or []),
    })
    return row


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return "-" if v is None else str(v)


def main():
    p = argparse.ArgumentParser(description="Dynamic-env test suite (delayed shield).")
    p.add_argument("--config", default="./configs/run_sim.yaml")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()))
    p.add_argument("--duration-ms", type=int, default=10000000)  # 2000 windows
    p.add_argument("--adaptation", default="reset_only")
    p.add_argument("--epsilon-reset", type=float, default=None,
                   help="Override controller.rl.epsilon_reset (post-change exploration bump).")
    p.add_argument("--loss-mode", default=None, choices=["binary", "fraction"],
                   help="Override controller.safety.loss_mode (RCRL loss: window indicator vs DMR).")
    p.add_argument("--alpha", type=float, default=None,
                   help="Override controller.safety.target_miss_alpha.")
    p.add_argument("--pool-extra", type=int, default=None,
                   help="Dormant pool size. Default: per-scenario (only add/replace need it).")
    p.add_argument("--out-dir", default="outputs/dynamic_experiments")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--fig-dir", default="outputs/sample/figures/5_dynamic",
                   help="Directory for output figures (use a new dir to keep existing PNGs).")
    p.add_argument("--fig-prefix", default="dyn_delay200_",
                   help="Filename prefix for figures.")
    args = p.parse_args()

    workdir = Path(__file__).resolve().parents[1]
    run_py = workdir / "run.py"
    base = load_config(args.config)
    out_root = Path(args.out_dir)
    cfg_root = out_root / "_configs"

    jobs, plan = [], []
    for name in args.scenarios:
        changes = SCENARIOS[name]
        out_dir = out_root / name
        plan.append((name, str(out_dir)))
        if args.skip_existing and latest_run(str(out_dir)):
            continue
        cfg = build_run_config(base, changes=changes, out_dir=out_dir,
                               duration_ms=args.duration_ms,
                               adaptation=args.adaptation, pool_extra=args.pool_extra,
                               epsilon_reset=args.epsilon_reset, loss_mode=args.loss_mode,
                               alpha=args.alpha)
        cfg_path = cfg_root / f"{name}.yaml"
        write_config(cfg_path, cfg)
        jobs.append((run_py, cfg_path, workdir))

    print(f"[dyn] {len(jobs)} run(s) to execute, {len(plan)} scenario(s) total")
    if jobs:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_one, j) for j in jobs]
                for f in as_completed(futs):
                    f.result()
        else:
            for j in jobs:
                run_one(j)

    rows = [collect_row(name, out_dir) for name, out_dir in plan]

    cols = ["scenario", "windows", "n_changes", "realized_miss", "alpha", "bound",
            "guar_ok", "overall_miss", "fallback_rate", "avg_util", "avg_qos",
            "post_change_settle_ms", "status"]
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "dynamic_summary.csv"
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
    print(f"\n[dyn] wrote {csv_path}")

    if args.plot:
        fig_dir = workdir / args.fig_dir
        fig_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r.get("status") != "ok":
                continue
            out_png = fig_dir / f"{args.fig_prefix}{r['scenario']}.png"
            subprocess.run([sys.executable, str(workdir / "scripts/plot_results.py"),
                            "--run", r["run_dir"], "--out", str(out_png)],
                           cwd=workdir, check=False)
            print(f"[dyn] plotted {out_png}")

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_guar = sum(1 for r in rows if r.get("guar_ok"))
    print(f"\n[dyn] {n_guar}/{n_ok} scenarios satisfy the Theorem-1 guarantee.")


if __name__ == "__main__":
    main()
