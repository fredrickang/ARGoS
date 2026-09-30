"""Run RCRL at a chosen target alpha over a small taskset population (fast probe).

Reuses the SAME 20 tasksets under outputs/settling_success/_tasksets. Single seed by
default -> ~20 runs. ref_util and alpha are CLI args; everything else is the locked config.

    python scripts/run_rcrl_alpha.py --alpha 0.01 --ref 0.9 --seeds 1 --out outputs/alpha001_ref09
"""
import argparse, copy, glob, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import yaml

WORK = Path(__file__).resolve().parents[1]
RUN = WORK / "run.py"
TS_DIR = WORK / "outputs" / "settling_success" / "_tasksets"


def run_one(cfg_path):
    subprocess.run([sys.executable, str(RUN), str(cfg_path)], cwd=WORK,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cfg_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--ref", type=float, default=0.9)
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--out", default="outputs/alpha001_ref09")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    OUT = WORK / args.out
    base = yaml.safe_load(open(WORK / "configs" / "run_sim.yaml"))
    tasksets = sorted(glob.glob(str(TS_DIR / "ts_*.txt")))
    (OUT / "_cfgs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for ti, tpath in enumerate(tasksets):
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg["controller"]["mode"] = "rl"
            cfg["controller"]["dynamic_mode"] = 0
            cfg["controller"]["ref_util"] = args.ref
            cfg["controller"].setdefault("rl", {})["seed"] = seed
            cfg["controller"].setdefault("safety", {})["mode"] = "rcrl"
            cfg["controller"]["safety"]["target_miss_alpha"] = args.alpha
            cfg["taskset"]["path"] = tpath
            tag = f"ts{ti:02d}_r{args.ref}_s{seed}"
            cfg.setdefault("output", {})["dir"] = str(OUT / tag)
            cfg["simulation"]["duration_ms"] = 10000000  # 2000 windows
            cp = OUT / "_cfgs" / f"{tag}.yaml"
            yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
            jobs.append(str(cp))

    print(f"[alpha] {len(tasksets)} tasksets x {len(args.seeds)} seeds = {len(jobs)} runs "
          f"(ref={args.ref}, alpha={args.alpha}, mode=rcrl)")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            f.result()
            if (i + 1) % 5 == 0:
                print(f"[alpha] {i+1}/{len(jobs)} done")
    print("[alpha] all runs complete")


if __name__ == "__main__":
    main()
