"""Static-environment settling success rate over arbitrary tasksets.

Design (locked with user):
  * Population: N same-size (10-task) tasksets drawn spread across difficulty (total max-util).
  * Factors: taskset x static ref_util in {0.5,0.7,0.9} x seed in {1,2,3}.  Each run = static
    (dynamic_mode 0), 2000 windows, current locked config.
  * A run SETTLES SUCCESSFULLY iff, over the steady-state window (last 500 windows), ALL hold:
      1. tracking : |mean(act_util) - ref| <= tau_track   (default 0.05)
      2. stability: std(act_util)        <= tau_jitter     (default 0.05)
      3. safety   : guarantee_satisfied  (realized_miss <= Theorem-1 bound)
  * Reported: overall success rate; breakdown by ref and by difficulty bin; per-criterion
    failure counts; settle-time and steady-QoS distributions.

Example:
    python scripts/run_static_settling_success.py --n-tasksets 20 --seeds 1 2 3 \
        --refs 0.5 0.7 0.9 --workers 6
"""

import argparse
import copy
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_tasksets(source, out_dir, n):
    """Extract N single-line tasksets spread across the file (loader needs one line/file)."""
    lines = [ln.strip() for ln in open(source) if ln.strip()]
    # sort by total util so the stride spans the difficulty range evenly
    lines.sort(key=lambda l: sum(float(x) for x in l.split(",")))
    idx = [round(i * (len(lines) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for k, i in enumerate(idx):
        util = sum(float(x) for x in lines[i].split(","))
        p = os.path.join(out_dir, f"ts_{k:02d}_u{util:.2f}.txt")
        with open(p, "w") as f:
            f.write(lines[i] + "\n")
        out.append((p, util))
    return out


def build_cfg(base, *, taskset_path, ref, seed, out_dir, duration_ms):
    cfg = copy.deepcopy(base)
    cfg["controller"]["mode"] = "rl"
    cfg["controller"]["dynamic_mode"] = 0  # static
    cfg["controller"]["ref_util"] = float(ref)
    cfg["taskset"]["path"] = taskset_path
    cfg["controller"].setdefault("rl", {})["seed"] = int(seed)
    cfg.setdefault("output", {})["dir"] = out_dir
    cfg["simulation"]["duration_ms"] = int(duration_ms)
    return cfg


def run_one(args):
    run_py, cfg_path, workdir = args
    subprocess.run([sys.executable, str(run_py), str(cfg_path)], cwd=workdir,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cfg_path


def latest_run(out_dir):
    hits = sorted(glob.glob(os.path.join(out_dir, "runs", "*")), key=os.path.getmtime)
    return hits[-1] if hits else None


def evaluate(out_dir, ref, tau_track, tau_jitter, steady_windows):
    rd = latest_run(out_dir)
    if not rd:
        return {"status": "MISSING"}
    d = json.load(open(os.path.join(rd, "summary.json")))
    rows = list(csv.DictReader(open(os.path.join(rd, "windows.csv"))))
    n = len(rows)
    steady = [r for r in rows if int(r["w_id"]) >= n - steady_windows]
    u = [float(r["act_util"]) for r in steady]
    q = [float(r["avg_qos"]) for r in steady]
    umean = statistics.fmean(u)
    ustd = statistics.pstdev(u)
    guar = bool(d.get("rcrl", {}).get("guarantee_satisfied"))
    track_ok = abs(umean - ref) <= tau_track
    stable_ok = ustd <= tau_jitter
    success = track_ok and stable_ok and guar
    settle = d.get("settle", {}).get("settle_time_ms")
    return {
        "status": "ok", "success": success,
        "track_ok": track_ok, "stable_ok": stable_ok, "guar_ok": guar,
        "umean": umean, "ustd": ustd, "track_err": abs(umean - ref),
        "qos": statistics.fmean(q), "settle_s": (settle / 1000 if settle else None),
        "realized_miss": d.get("rcrl", {}).get("realized_miss_rate"),
    }


def pct(x, n):
    return f"{100*x/n:.1f}%" if n else "-"


def main():
    p = argparse.ArgumentParser(description="Static settling success rate over tasksets.")
    p.add_argument("--config", default="./configs/run_sim.yaml")
    p.add_argument("--source", default="tasksets/tasksets_u2.0_to_3.0_n10_c5_all.txt")
    p.add_argument("--n-tasksets", type=int, default=20)
    p.add_argument("--refs", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--tau-track", type=float, default=0.05)
    p.add_argument("--tau-jitter", type=float, default=0.05)
    p.add_argument("--steady-windows", type=int, default=500)
    p.add_argument("--duration-ms", type=int, default=10000000)  # 2000 windows
    p.add_argument("--out-dir", default="outputs/settling_success")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    workdir = Path(__file__).resolve().parents[1]
    run_py = workdir / "run.py"
    base = load_config(args.config)
    out_root = Path(args.out_dir)
    ts = extract_tasksets(args.source, str(out_root / "_tasksets"), args.n_tasksets)

    plan, jobs = [], []
    for ti, (tpath, util) in enumerate(ts):
        for ref in args.refs:
            for seed in args.seeds:
                tag = f"ts{ti:02d}_r{ref}_s{seed}"
                od = str(out_root / tag)
                plan.append({"ti": ti, "util": util, "ref": ref, "seed": seed, "out": od})
                cfg = build_cfg(base, taskset_path=tpath, ref=ref, seed=seed,
                                out_dir=od, duration_ms=args.duration_ms)
                cp = out_root / "_cfgs" / f"{tag}.yaml"
                cp.parent.mkdir(parents=True, exist_ok=True)
                with open(cp, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                jobs.append((run_py, str(cp), workdir))

    print(f"[settle] {len(ts)} tasksets (util {ts[0][1]:.2f}..{ts[-1][1]:.2f}) x "
          f"{len(args.refs)} refs x {len(args.seeds)} seeds = {len(plan)} runs")
    print(f"[settle] success = |util-ref|<={args.tau_track} AND std<={args.tau_jitter} AND guarantee")
    if args.dry_run:
        print("[settle] dry-run; not executing.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            f.result()
            if (i + 1) % 20 == 0:
                print(f"[settle] {i+1}/{len(jobs)} runs done")

    # evaluate
    rows = []
    for pl in plan:
        ev = evaluate(pl["out"], pl["ref"], args.tau_track, args.tau_jitter, args.steady_windows)
        rows.append({**pl, **ev})

    ok = [r for r in rows if r.get("status") == "ok"]
    n = len(ok)
    nsucc = sum(1 for r in ok if r["success"])
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "settling_results.csv", "w", newline="") as f:
        cols = ["ti", "util", "ref", "seed", "success", "track_ok", "stable_ok", "guar_ok",
                "umean", "ustd", "track_err", "qos", "settle_s", "realized_miss", "status"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n===== SETTLING SUCCESS RATE =====")
    print(f"Overall: {nsucc}/{n} = {pct(nsucc, n)}")
    print(f"\nBy ref:")
    for ref in args.refs:
        sub = [r for r in ok if r["ref"] == ref]
        s = sum(1 for r in sub if r["success"])
        print(f"  ref={ref}: {s}/{len(sub)} = {pct(s, len(sub))}")
    print(f"\nBy difficulty (total max-util bin):")
    for lo, hi in [(1.9, 2.34), (2.34, 2.67), (2.67, 3.1)]:
        sub = [r for r in ok if lo <= r["util"] < hi]
        s = sum(1 for r in sub if r["success"])
        print(f"  util[{lo:.2f},{hi:.2f}): {s}/{len(sub)} = {pct(s, len(sub))}")
    print(f"\nPer-criterion failures (of {n}):")
    print(f"  tracking fail : {sum(1 for r in ok if not r['track_ok'])}")
    print(f"  stability fail: {sum(1 for r in ok if not r['stable_ok'])}")
    print(f"  guarantee fail: {sum(1 for r in ok if not r['guar_ok'])}")
    fails = [r for r in ok if not r["success"]]
    if fails:
        print(f"\nSample failures (worst tracking):")
        for r in sorted(fails, key=lambda r: -r["track_err"])[:6]:
            print(f"  ts{r['ti']:02d} util={r['util']:.2f} ref={r['ref']} seed={r['seed']}: "
                  f"umean={r['umean']:.3f} std={r['ustd']:.3f} guar={r['guar_ok']}")
    print(f"\n[settle] wrote {out_root/'settling_results.csv'}")


if __name__ == "__main__":
    main()
