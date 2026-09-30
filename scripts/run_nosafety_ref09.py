"""Run the settling-success population at ref=0.9 with risk control OFF (safety.mode=none).

Reuses the SAME 20 tasksets already extracted under outputs/settling_success/_tasksets
(so it's directly comparable to the RCRL guarantee envelope). Static, 2000 windows,
3 seeds. RL agent still targets ref_util=0.9, but no shield enforces the miss bound.
"""
import copy, glob, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import yaml

WORK = Path(__file__).resolve().parents[1]
RUN = WORK / "run.py"
OUT = WORK / "outputs" / "settling_success_nosafety"
TS_DIR = WORK / "outputs" / "settling_success" / "_tasksets"
SEEDS = [1, 2, 3]
REF = 0.9


def run_one(cfg_path):
    subprocess.run([sys.executable, str(RUN), str(cfg_path)], cwd=WORK,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cfg_path


def main():
    base = yaml.safe_load(open(WORK / "configs" / "run_sim.yaml"))
    tasksets = sorted(glob.glob(str(TS_DIR / "ts_*.txt")))
    (OUT / "_cfgs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for ti, tpath in enumerate(tasksets):
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["controller"]["mode"] = "rl"
            cfg["controller"]["dynamic_mode"] = 0
            cfg["controller"]["ref_util"] = REF
            cfg["controller"].setdefault("rl", {})["seed"] = seed
            cfg["controller"].setdefault("safety", {})["mode"] = "none"  # risk control OFF
            cfg["taskset"]["path"] = tpath
            tag = f"ts{ti:02d}_r{REF}_s{seed}"
            cfg.setdefault("output", {})["dir"] = str(OUT / tag)
            cfg["simulation"]["duration_ms"] = 10000000  # 2000 windows
            cp = OUT / "_cfgs" / f"{tag}.yaml"
            yaml.safe_dump(cfg, open(cp, "w"), sort_keys=False)
            jobs.append(str(cp))

    print(f"[nosafety] {len(tasksets)} tasksets x {len(SEEDS)} seeds = {len(jobs)} runs (ref={REF}, mode=none)")
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            f.result()
            if (i + 1) % 10 == 0:
                print(f"[nosafety] {i+1}/{len(jobs)} done")
    print("[nosafety] all runs complete")


if __name__ == "__main__":
    main()
