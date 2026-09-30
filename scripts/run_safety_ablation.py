"""Safety-guard ablation: sweep controller.safety.mode x dynamic_mode.

Runs the cross-product of safety modes (rcrl / conformal / rt_analytic / none) and
dynamic modes (0..3), each into its own output dir, then collates the per-run summary
JSON — in particular the ``rcrl`` telemetry block (realized miss rate vs. the Theorem-1
bound) — into a comparison table (printed + CSV).

Example:
    python scripts/run_safety_ablation.py --duration-ms 5000000 \
        --modes rcrl conformal none --dynamic-modes 0 1 --workers 4
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


def build_run_config(base, *, mode, dynamic_mode, out_dir, duration_ms):
    cfg = copy.deepcopy(base)
    cfg["controller"]["mode"] = "rl"
    cfg["controller"]["dynamic_mode"] = dynamic_mode
    safety = dict(cfg["controller"].get("safety") or {})
    safety["mode"] = mode
    cfg["controller"]["safety"] = safety
    cfg.setdefault("output", {})["dir"] = str(out_dir)
    if duration_ms is not None:
        cfg["simulation"]["duration_ms"] = int(duration_ms)
    return cfg


def run_one(args):
    run_py, cfg_path, workdir = args
    subprocess.run([sys.executable, str(run_py), str(cfg_path)], cwd=workdir, check=True)
    return cfg_path


def find_summary(out_dir):
    # Prefer the per-run recorder summary (outputs/<dir>/runs/<run_id>/summary.json);
    # fall back to the legacy fat summary at the dir root.
    hits = sorted(glob.glob(os.path.join(out_dir, "runs", "*", "summary.json")),
                  key=os.path.getmtime)
    if not hits:
        hits = sorted(glob.glob(os.path.join(out_dir, "*_summary.json")),
                      key=os.path.getmtime)
    return hits[-1] if hits else None


def collect_row(mode, dynamic_mode, summary_path):
    row = {"mode": mode, "dynamic_mode": dynamic_mode, "summary": summary_path}
    if not summary_path or not os.path.exists(summary_path):
        row["status"] = "MISSING"
        return row
    d = json.load(open(summary_path))
    overall = d.get("overall", {})
    rcrl = d.get("rcrl", {})
    settle = d.get("settle", {})
    row.update({
        "status": "ok",
        "windows": rcrl.get("total_windows", overall.get("total_windows")),
        "realized_miss_rate": rcrl.get("realized_miss_rate"),
        "target_alpha": rcrl.get("target_alpha"),
        "theorem_bound": rcrl.get("theorem_bound"),
        "guarantee_ok": rcrl.get("guarantee_satisfied"),
        "fallback_rate": rcrl.get("fallback_rate"),
        "lambda_final": rcrl.get("lambda_final"),
        "avg_act_util": overall.get("avg_act_util"),
        "avg_dmr": overall.get("avg_dmr"),
        "avg_qos_ctrl": d.get("avg_qos_controllable"),
        "settle_time_ms": settle.get("settle_time_ms"),
    })
    return row


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return "-" if v is None else str(v)


def main():
    p = argparse.ArgumentParser(description="Safety-guard ablation sweep.")
    p.add_argument("--config", default="./configs/run_sim.yaml", help="Base config.")
    p.add_argument("--modes", nargs="+",
                   default=["rcrl", "conformal", "rt_analytic", "none"],
                   help="controller.safety.mode values to sweep.")
    p.add_argument("--dynamic-modes", nargs="+", type=int, default=[0],
                   help="controller.dynamic_mode values to sweep.")
    p.add_argument("--duration-ms", type=int, default=None,
                   help="Override simulation.duration_ms (default: base config).")
    p.add_argument("--out-dir", default="outputs/safety_ablation", help="Root output dir.")
    p.add_argument("--workers", type=int, default=1, help="Parallel runs.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a combo if its summary JSON already exists.")
    args = p.parse_args()

    workdir = Path(__file__).resolve().parents[1]
    run_py = workdir / "run.py"
    base = load_config(args.config)
    out_root = Path(args.out_dir)
    cfg_root = out_root / "_configs"

    jobs, plan = [], []
    for mode in args.modes:
        for dyn in args.dynamic_modes:
            tag = f"{mode}_dyn{dyn}"
            out_dir = out_root / tag
            plan.append((mode, dyn, str(out_dir)))
            if args.skip_existing and find_summary(str(out_dir)):
                continue
            cfg = build_run_config(base, mode=mode, dynamic_mode=dyn,
                                   out_dir=out_dir, duration_ms=args.duration_ms)
            cfg_path = cfg_root / f"{tag}.yaml"
            write_config(cfg_path, cfg)
            jobs.append((run_py, cfg_path, workdir))

    print(f"[ablation] {len(jobs)} run(s) to execute, {len(plan)} combo(s) total")
    if jobs:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_one, j) for j in jobs]
                for f in as_completed(futs):
                    f.result()
        else:
            for j in jobs:
                run_one(j)

    rows = [collect_row(mode, dyn, find_summary(out_dir)) for mode, dyn, out_dir in plan]

    cols = ["mode", "dynamic_mode", "windows", "realized_miss_rate", "target_alpha",
            "theorem_bound", "guarantee_ok", "fallback_rate", "avg_act_util",
            "avg_qos_ctrl", "settle_time_ms", "status"]
    csv_path = out_root / "ablation_summary.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["summary"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    print()
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols))
    print(f"\n[ablation] wrote {csv_path}")


if __name__ == "__main__":
    main()
