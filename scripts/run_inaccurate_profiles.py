#!/usr/bin/env python
"""Motivational experiment — Inaccurate Profiles.

Sweep the cosine similarity between the FCS controller's *estimated* execution
profile and the ground-truth profile, and measure the settling success ratio
(fraction of task sets whose utilization settles within +-5% of the target for
five consecutive windows) at each similarity level.

Run from the project root:
    python scripts/run_inaccurate_profiles.py --draws 3 --workers 8
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

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_job_config(base, taskset_path, out_dir, similarity, seed, qos_level, duration, cov, ref_util, metric="cosine"):
    cfg = copy.deepcopy(base)
    cfg["simulation"] = {"duration_ms": int(duration)}
    cfg["taskset"] = {"path": str(taskset_path)}
    cfg.setdefault("qos", {})["level"] = int(qos_level)
    cfg["qos"]["policy"] = cfg.get("qos", {}).get("policy", "linear")
    cfg.setdefault("window", {})["size_ms"] = cfg.get("window", {}).get("size_ms", 5000)
    cfg["exec_time"] = {"model": "gaussian", "cov": float(cov), "mu": 0, "sigma": float(cov)}
    cfg["output"] = {"dir": str(out_dir)}
    ctrl = cfg.setdefault("controller", {})
    ctrl.update({"mode": "fcs", "dynamic_mode": 0, "ref_util": float(ref_util),
                 "ref_dmr": 0.01, "enable_profile_fallback": False, "save_rl_samples": False})
    fcs = cfg.setdefault("fcs_ctrl_param", {})
    fcs.update({"similarity_metric": str(metric), "similarity": float(similarity),
                "similarity_seed": int(seed), "kp_m": 0.37, "kp_u": 0.37, "Ga": 1.0})
    return cfg


def run_one(job):
    cfg, cfg_path, out_dir = job["cfg"], job["cfg_path"], job["out_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    try:
        subprocess.run([sys.executable, "run.py", str(cfg_path)], cwd=str(ROOT),
                       check=True, capture_output=True, timeout=1200)
    except subprocess.CalledProcessError as e:
        return {**job["meta"], "settled": None, "error": e.stderr.decode()[-500:]}
    files = glob.glob(str(Path(out_dir) / "*_summary.json"))
    if not files:
        return {**job["meta"], "settled": None, "error": "no summary"}
    summary = json.load(open(files[0]))
    settle = summary.get("settle", {})
    settle_ms = settle.get("settle_time_ms")
    return {
        **job["meta"],
        "settled": 1 if settle_ms is not None else 0,
        "settle_time_ms": settle_ms,
        "avg_util": summary.get("overall", {}).get("avg_act_util"),
        "avg_qos": summary.get("overall", {}).get("avg_qos"),
        "realized_similarity": summary.get("realized_similarity"),
        "error": "",
    }


def main():
    p = argparse.ArgumentParser(description="Inaccurate-profiles settling-success sweep.")
    p.add_argument("--taskset-file", default="tasksets/tasksets_u2.5_n10_c20.txt")
    p.add_argument("--sims", default="1.0,0.95,0.90,0.85,0.80",
                   help="Comma-separated cosine-similarity levels.")
    p.add_argument("--draws", type=int, default=3, help="Approx-profile draws per (set, sim<1).")
    p.add_argument("--qos-level", type=int, default=5)
    p.add_argument("--duration", type=int, default=2000000,
                   help="Sim horizon (ms). Needs >~1.5M so the accurate baseline settles.")
    p.add_argument("--cov", type=float, default=0.1, help="Execution-time coefficient of variation.")
    p.add_argument("--ref-util", type=float, default=0.9)
    p.add_argument("--metric", default="cosine", choices=["cosine", "mape"],
                   help="Profile-similarity metric used to generate inaccurate profiles.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--base-config", default="configs/run_sim.yaml")
    p.add_argument("--out-dir", default="outputs/exp_inaccurate_profiles")
    args = p.parse_args()

    base = load_yaml(ROOT / args.base_config)
    sims = [float(s) for s in args.sims.split(",") if s.strip()]
    taskset_lines = [ln.strip() for ln in open(ROOT / args.taskset_file) if ln.strip()]
    out_root = ROOT / args.out_dir
    cfg_dir = out_root / "configs"
    ts_dir = out_root / "tasksets"
    runs_dir = out_root / "runs"
    for d in (cfg_dir, ts_dir, runs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # One temp taskset file per set (single line), reused across sims/draws.
    set_paths = []
    for si, line in enumerate(taskset_lines, start=1):
        tp = ts_dir / f"set{si:02d}.txt"
        tp.write_text(line + "\n", encoding="utf-8")
        set_paths.append(tp)

    jobs = []
    for si, tp in enumerate(set_paths, start=1):
        for s in sims:
            draws = [0] if s >= 1.0 else range(args.draws)  # s=1.0 is deterministic
            for d in draws:
                tag = f"set{si:02d}_sim{s:.2f}_d{d}"
                out_dir = runs_dir / tag
                cfg = build_job_config(base, tp, out_dir, s, seed=1000 * si + d,
                                       qos_level=args.qos_level, duration=args.duration,
                                       cov=args.cov, ref_util=args.ref_util, metric=args.metric)
                jobs.append({
                    "cfg": cfg, "cfg_path": cfg_dir / f"{tag}.yaml", "out_dir": out_dir,
                    "meta": {"set_idx": si, "similarity": s, "draw": d},
                })

    print(f"Running {len(jobs)} simulations ({len(set_paths)} sets x {len(sims)} sims, "
          f"draws={args.draws}, cov={args.cov}, dur={args.duration}) on {args.workers} workers...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), start=1):
            results.append(fut.result())
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done")

    # Per-run CSV.
    runs_csv = out_root / "runs.csv"
    cols = ["set_idx", "similarity", "draw", "settled", "settle_time_ms",
            "avg_util", "avg_qos", "realized_similarity", "error"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(results, key=lambda r: (r["similarity"], r["set_idx"], r["draw"])):
            w.writerow({k: r.get(k) for k in cols})

    # Aggregate: settling success ratio per similarity.
    agg = {}
    for r in results:
        agg.setdefault(r["similarity"], []).append(r)
    ssr_csv = out_root / "ssr.csv"
    rows = []
    for s in sorted(agg, reverse=True):
        rs = agg[s]
        valid = [r for r in rs if r["settled"] is not None]
        settled = sum(r["settled"] for r in valid)
        ssr = settled / len(valid) if valid else float("nan")
        real = [r["realized_similarity"] for r in valid if r["realized_similarity"] is not None]
        mean_real = sum(real) / len(real) if real else s
        rows.append({"similarity": s, "ssr": ssr, "n": len(valid),
                     "settled": settled, "mean_realized_sim": round(mean_real, 4)})
    with open(ssr_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["similarity", "ssr", "n", "settled", "mean_realized_sim"])
        w.writeheader()
        w.writerows(rows)

    print("\nsimilarity  SSR    settled/n   mean_realized_cos")
    for r in rows:
        print(f"  {r['similarity']:.2f}      {r['ssr']:.3f}   {r['settled']}/{r['n']}"
              f"        {r['mean_realized_sim']:.3f}")

    # Plot SSR vs similarity.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["similarity"] for r in rows][::-1]
        ys = [r["ssr"] for r in rows][::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker="o", color="tab:blue")
        ax.set_xlabel("Cosine similarity (estimated vs. ground-truth profile)")
        ax.set_ylabel("Settling success ratio")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(f"FCS settling vs. profile accuracy (target util={args.ref_util})")
        fig.tight_layout()
        png = out_root / "ssr_vs_similarity.png"
        fig.savefig(png, dpi=150)
        print(f"\nplot -> {png}")
    except Exception as exc:
        print(f"plot skipped: {exc}")
    print(f"per-run csv -> {runs_csv}\nssr csv     -> {ssr_csv}")


if __name__ == "__main__":
    main()
