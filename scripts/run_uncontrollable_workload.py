#!/usr/bin/env python
"""Motivational experiment — Partial profiles / uncontrollable workload (matrix).

10-task sets (total util 2.5 at highest QoS). The last `num_uncontrollable` tasks of a
set are no-profile tasks that run at a fixed QoS and together generate a fixed
"uncontrollable workload" (swept 0.0..1.0). FCS adjusts only the controllable tasks.

Two controllable-capacity cases:
  - natural : controllable = the set's remaining tasks as-is  (C_max = natural sum, C_min = 0)
  - fixed   : controllable rescaled to a fixed capacity range [C_min, C_max]

Sweeps num_uncontrollable in {3,5,7}. Reports settling success ratio vs uncontrollable util.

    python scripts/run_uncontrollable_workload.py --workers 64
"""

import argparse
import copy
import csv
import glob
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_job_config(base, taskset_path, out_dir, unc_util, num_unc, qos_level,
                     duration, cov, ref_util, cmax, min_frac):
    cfg = copy.deepcopy(base)
    cfg["simulation"] = {"duration_ms": int(duration)}
    cfg["taskset"] = {"path": str(taskset_path)}
    cfg.setdefault("qos", {})["level"] = int(qos_level)
    cfg["qos"]["policy"] = cfg.get("qos", {}).get("policy", "linear")
    cfg["qos"]["min_frac"] = float(min_frac)
    cfg["exec_time"] = {"model": "gaussian", "cov": float(cov), "mu": 0, "sigma": float(cov)}
    cfg["output"] = {"dir": str(out_dir)}
    cfg["uncontrollable"] = {"num_tasks": int(num_unc), "util": float(unc_util)}
    cfg["controllable"] = {"cmax": (float(cmax) if cmax else None)}
    ctrl = cfg.setdefault("controller", {})
    ctrl.update({"mode": "fcs", "dynamic_mode": 0, "ref_util": float(ref_util),
                 "ref_dmr": 0.01, "enable_profile_fallback": False, "save_rl_samples": False})
    fcs = cfg.setdefault("fcs_ctrl_param", {})
    fcs.update({"kp_m": 0.37, "kp_u": 0.37, "Ga": 1.0})
    fcs.pop("similarity_metric", None); fcs.pop("similarity", None)
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
        return {**job["meta"], "settled": None, "error": e.stderr.decode()[-300:]}
    files = glob.glob(str(Path(out_dir) / "*_summary.json"))
    if not files:
        return {**job["meta"], "settled": None, "error": "no summary"}
    summary = json.load(open(files[0]))
    settle_ms = summary.get("settle", {}).get("settle_time_ms")
    return {**job["meta"], "settled": 1 if settle_ms is not None else 0,
            "avg_util": summary.get("overall", {}).get("avg_act_util"),
            "qos_ctrl": summary.get("avg_qos_controllable"),
            "qos_ctrl_settled": summary.get("avg_qos_controllable_post_settle"),
            "error": ""}


def main():
    p = argparse.ArgumentParser(description="Uncontrollable-workload matrix sweep.")
    p.add_argument("--taskset-file", default="tasksets/uexp_u2.5_n10_c20.txt")
    p.add_argument("--utils", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--num-uncontrollable", default="3,5,7", help="Comma list of uncontrollable counts.")
    p.add_argument("--cases", default="natural,fixed", help="Comma list of: natural, fixed.")
    p.add_argument("--cmax", type=float, default=0.8, help="Fixed-case controllable C_max.")
    p.add_argument("--cmin", type=float, default=0.2, help="Fixed-case controllable C_min.")
    p.add_argument("--qos-level", type=int, default=5)
    p.add_argument("--duration", type=int, default=2000000)
    p.add_argument("--cov", type=float, default=0.1)
    p.add_argument("--ref-util", type=float, default=0.9)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--base-config", default="configs/run_sim.yaml")
    p.add_argument("--out-dir", default="outputs/exp_uncontrollable_matrix")
    args = p.parse_args()

    base = load_yaml(ROOT / args.base_config)
    utils = [float(u) for u in args.utils.split(",") if u.strip() != ""]
    num_uncs = [int(x) for x in args.num_uncontrollable.split(",") if x.strip()]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    lines = [ln.strip() for ln in open(ROOT / args.taskset_file) if ln.strip()]
    out_root = ROOT / args.out_dir
    cfg_dir, ts_dir, runs_dir = out_root / "configs", out_root / "tasksets", out_root / "runs"
    for d in (cfg_dir, ts_dir, runs_dir):
        d.mkdir(parents=True, exist_ok=True)

    set_paths = []
    for si, line in enumerate(lines, start=1):
        tp = ts_dir / f"set{si:02d}.txt"
        tp.write_text(line + "\n", encoding="utf-8")
        set_paths.append(tp)

    def case_params(case):
        if case == "fixed":
            return args.cmax, (args.cmin / args.cmax if args.cmax else 0.0)
        return None, 0.0  # natural: C_max from taskset, C_min = 0

    jobs = []
    for case in cases:
        cmax, min_frac = case_params(case)
        for num_unc in num_uncs:
            for si, tp in enumerate(set_paths, start=1):
                for u in utils:
                    tag = f"{case}_N{num_unc}_set{si:02d}_u{u:.2f}"
                    out_dir = runs_dir / tag
                    cfg = build_job_config(base, tp, out_dir, u, num_unc, args.qos_level,
                                           args.duration, args.cov, args.ref_util, cmax, min_frac)
                    jobs.append({"cfg": cfg, "cfg_path": cfg_dir / f"{tag}.yaml", "out_dir": out_dir,
                                 "meta": {"case": case, "num_unc": num_unc, "set_idx": si, "unc_util": u}})

    print(f"Running {len(jobs)} sims "
          f"({len(cases)} cases x {len(num_uncs)} num_unc x {len(set_paths)} sets x {len(utils)} utils) "
          f"on {args.workers} workers...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), start=1):
            results.append(fut.result())
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done")

    # runs.csv
    with open(out_root / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "num_unc", "set_idx", "unc_util", "settled", "avg_util", "error"])
        w.writeheader()
        for r in sorted(results, key=lambda r: (r["case"], r["num_unc"], r["unc_util"], r["set_idx"])):
            w.writerow({k: r.get(k) for k in ["case", "num_unc", "set_idx", "unc_util", "settled", "avg_util", "error"]})

    # aggregate SSR per (case, num_unc, util)
    agg = {}
    for r in results:
        agg.setdefault((r["case"], r["num_unc"], r["unc_util"]), []).append(r)
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    rows = []
    for (case, num_unc, u) in sorted(agg):
        rs = [r for r in agg[(case, num_unc, u)] if r["settled"] is not None]
        settled = sum(r["settled"] for r in rs)
        # controllable avg QoS: prefer settled runs' steady-state value, else all-window value
        qos_settled = _mean([r.get("qos_ctrl_settled") for r in rs if r["settled"] == 1])
        rows.append({"case": case, "num_unc": num_unc, "unc_util": u,
                     "ssr": settled / len(rs) if rs else float("nan"), "n": len(rs), "settled": settled,
                     "avg_qos_ctrl": _mean([r.get("qos_ctrl") for r in rs]),
                     "avg_qos_ctrl_settled": qos_settled})
    with open(out_root / "ssr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "num_unc", "unc_util", "ssr", "n", "settled",
                                          "avg_qos_ctrl", "avg_qos_ctrl_settled"])
        w.writeheader(); w.writerows(rows)

    # print tables
    def _cell(case, num_unc, u, key, fmt):
        m = [r for r in rows if r["case"] == case and r["num_unc"] == num_unc and abs(r["unc_util"] - u) < 1e-9]
        v = m[0].get(key) if m else None
        return fmt(v) if v is not None else " - "

    for case in cases:
        print(f"\n=== case: {case} — SSR ===")
        print("num_unc  " + "  ".join(f"u{u:.1f}" for u in utils))
        for num_unc in num_uncs:
            cells = [_cell(case, num_unc, u, "ssr", lambda v: f"{v:.2f}") for u in utils]
            print(f"  N={num_unc}    " + "  ".join(f"{c:>4}" for c in cells))
        print(f"--- case: {case} — controllable avg QoS (settled runs) ---")
        for num_unc in num_uncs:
            cells = [_cell(case, num_unc, u, "avg_qos_ctrl_settled", lambda v: f"{v:.1f}") for u in utils]
            print(f"  N={num_unc}    " + "  ".join(f"{c:>4}" for c in cells))

    # plot: one subplot per case, 3 curves (num_unc)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(cases), figsize=(6.2 * len(cases), 4.3), sharey=True, squeeze=False)
        colors = {3: "tab:blue", 5: "tab:green", 7: "tab:red"}
        for ax, case in zip(axes[0], cases):
            for num_unc in num_uncs:
                pts = sorted([(r["unc_util"], r["ssr"]) for r in rows if r["case"] == case and r["num_unc"] == num_unc])
                ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o",
                        color=colors.get(num_unc), label=f"{num_unc} uncontrollable")
            title = case + (f"  (C_min={args.cmin}, C_max={args.cmax})" if case == "fixed" else "  (C_min=0, C_max=natural)")
            ax.set_title(title); ax.set_xlabel("Uncontrollable workload utilization")
            ax.set_ylim(-0.02, 1.02); ax.grid(True, ls="--", alpha=0.4); ax.legend()
        axes[0][0].set_ylabel("Settling success ratio")
        fig.tight_layout(); fig.savefig(out_root / "ssr_matrix.png", dpi=150)
        print(f"\nplot -> {out_root / 'ssr_matrix.png'}")
    except Exception as exc:
        print(f"plot skipped: {exc}")


if __name__ == "__main__":
    main()
