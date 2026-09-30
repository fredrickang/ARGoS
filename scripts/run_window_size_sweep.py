"""Window-size sweep: adaptation speed vs. release-mode feedback lag trade-off.

For release-mode DMR the RCRL threshold update lags by k = max_i ceil(D_i / W) windows.
Shrinking the window W:
  * speeds QoS adaptation (more control steps per unit time), but
  * grows the lag k -> the shield's λ reacts later (staler feedback).
Enlarging W does the opposite (k -> 1, fresh feedback, but coarse/slow adaptation).
This sweeps W (fixed window COUNT so learning steps are comparable) and records, per W:
lag k, cold-start settle time (windows and ms = windows*W), steady tracking error, jitter,
QoS, realized miss / guarantee margin, fallback rate, and steady-state λ oscillation.

    python scripts/run_window_size_sweep.py --workers 6
"""
import argparse, copy, csv, glob, json, math, os, subprocess, sys, statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import yaml

WORK = Path(__file__).resolve().parents[1]
RUN = WORK / "run.py"
sys.path.insert(0, str(WORK))


def taskset_periods(taskset_path, base_cfg, period_range=(500, 1000)):
    """Deterministic seed-22 task periods (= relative deadlines) for a taskset, matching what
    run.py generates (it calls random.seed(22) before task_generator). Lets lag = max ceil(D/W)
    be computed exactly for any taskset / period range."""
    import random
    from patrol.sim import config as C
    random.seed(22)
    q = base_cfg.get("qos", {})
    tasks = C.task_generator(path=str(taskset_path),
                             qos_level=int(q.get("level", 10)),
                             qos_policy=str(q.get("policy", "linear")),
                             period_range=tuple(period_range))
    return [t.period for t in (tasks or [])]


def run_one(cfg_path):
    subprocess.run([sys.executable, str(RUN), str(cfg_path)], cwd=WORK,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cfg_path


def latest_run(out_dir):
    hits = sorted(glob.glob(os.path.join(out_dir, "runs", "*")), key=os.path.getmtime)
    return hits[-1] if hits else None


def _settle_windows(util, ref, tau=0.05, roll=15):
    """First window after which the rolling-mean util stays within +/-tau of ref for good
    (= 1 + last window where it is outside). n (never settles) if it never stabilises."""
    n = len(util)
    if n == 0:
        return None
    k = min(roll, n)
    kern = [1.0 / k] * k
    rmean = [sum(util[max(0, i - k + 1):i + 1]) / len(util[max(0, i - k + 1):i + 1]) for i in range(n)]
    last_bad = -1
    for i, v in enumerate(rmean):
        if abs(v - ref) > tau:
            last_bad = i
    return last_bad + 1  # 0 if already in-band from the start; n-1.. if late


def collect(out_dir, ref, W, n_windows, periods, steady=500):
    rd = latest_run(out_dir)
    if not rd:
        return None
    d = json.load(open(os.path.join(rd, "summary.json")))
    rows = list(csv.DictReader(open(os.path.join(rd, "windows.csv"))))
    n = len(rows)
    util_all = [float(r["act_util"]) for r in rows]
    st = [r for r in rows if int(r["w_id"]) >= n - steady]
    u = [float(r["act_util"]) for r in st]
    q = [float(r["avg_qos"]) for r in st]
    lam = [float(r["rcrl_lambda"]) for r in st if r.get("rcrl_lambda") not in (None, "", "None")]
    rc = d.get("rcrl", {})
    settle_windows = _settle_windows(util_all, ref)
    settle_ms = settle_windows * W if settle_windows is not None else None
    lag = max(math.ceil(p / W) for p in periods)
    bound = rc.get("theorem_bound")
    realized = rc.get("realized_miss_rate")
    return {
        "W": W, "lag": lag, "ref": ref,
        "settle_ms": settle_ms,
        "settle_windows": settle_windows,
        "umean": statistics.fmean(u), "track_err": abs(statistics.fmean(u) - ref),
        "ustd": statistics.pstdev(u), "qos": statistics.fmean(q),
        "realized_miss": realized, "bound": bound,
        "guar_margin": (bound - realized) if (bound is not None and realized is not None) else None,
        "guar_ok": rc.get("guarantee_satisfied"),
        "fallback_rate": rc.get("fallback_rate"),
        "lambda_std": statistics.pstdev(lam) if len(lam) > 1 else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./configs/run_sim.yaml")
    p.add_argument("--taskset", default="./tasksets/taskset_1.txt")
    p.add_argument("--ref", type=float, default=0.7)
    p.add_argument("--windows", nargs="+", type=int,
                   default=[200, 333, 500, 750, 1000, 1500, 2000, 3500, 5000])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--n-windows", type=int, default=1500)
    p.add_argument("--period-range", nargs=2, type=int, default=[500, 1000],
                   help="Task period (=deadline) range; controls D_max and thus the lag.")
    p.add_argument("--out-dir", default="outputs/window_size_sweep")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    base = yaml.safe_load(open(WORK / args.config))
    out_root = WORK / args.out_dir
    (out_root / "_cfgs").mkdir(parents=True, exist_ok=True)

    pr = tuple(args.period_range)
    periods = taskset_periods(args.taskset, base, pr)
    dmax = max(periods) if periods else None
    print(f"[wsweep] taskset={args.taskset}  D_max={dmax}  periods={periods}")

    plan, jobs = [], []
    for W in args.windows:
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg["controller"]["mode"] = "rl"
            cfg["controller"]["dynamic_mode"] = 0
            cfg["controller"]["ref_util"] = args.ref
            cfg["controller"].setdefault("rl", {})["seed"] = seed
            cfg["controller"].setdefault("safety", {})["mode"] = "rcrl"
            cfg["controller"]["safety"]["target_miss_alpha"] = 0.1
            cfg["controller"]["safety"]["dmr_membership"] = "release"  # auto-lag = max ceil(D/W)
            cfg["controller"]["safety"]["fallback_start_window"] = 200  # fixed # warm-up windows
            cfg["taskset"]["path"] = args.taskset
            cfg["taskset"]["period_range"] = list(pr)
            cfg["window"]["size_ms"] = int(W)
            cfg["simulation"]["duration_ms"] = int(args.n_windows * W)
            tag = f"W{W}_s{seed}"
            od = str(out_root / tag)
            cfg.setdefault("output", {})["dir"] = od
            cp = out_root / "_cfgs" / f"{tag}.yaml"
            yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
            plan.append({"W": W, "seed": seed, "out": od})
            jobs.append(str(cp))

    print(f"[wsweep] {len(args.windows)} window sizes x {len(args.seeds)} seeds = {len(jobs)} runs "
          f"({args.n_windows} windows each, ref={args.ref}, release-mode auto-lag)")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            f.result()
            if (i + 1) % 5 == 0:
                print(f"[wsweep] {i+1}/{len(jobs)} done")

    rows = []
    for pl in plan:
        r = collect(pl["out"], args.ref, pl["W"], args.n_windows, periods)
        if r:
            r["seed"] = pl["seed"]
            r["dmax"] = dmax
            rows.append(r)
    cols = ["W", "lag", "seed", "ref", "dmax", "settle_ms", "settle_windows", "umean", "track_err",
            "ustd", "qos", "realized_miss", "bound", "guar_margin", "guar_ok",
            "fallback_rate", "lambda_std"]
    with open(out_root / "window_size_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[wsweep] wrote {out_root/'window_size_results.csv'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
