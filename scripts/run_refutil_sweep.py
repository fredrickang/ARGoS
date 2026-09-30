import argparse
import copy
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_config(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def load_taskset_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def ref_util_values(start, end, step):
    count = int(round((end - start) / step))
    return [round(start + i * step, 1) for i in range(count + 1)]


def run_sim(run_sim_path, cfg_path, workdir, env=None):
    cmd = [sys.executable, str(run_sim_path), str(cfg_path)]
    subprocess.run(cmd, cwd=workdir, check=True, env=env)


def run_job(args):
    run_sim_path, cfg_path, workdir, env = args
    run_sim(run_sim_path, cfg_path, workdir, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run ref_util sweep for task sets.")
    parser.add_argument(
        "--config",
        default="./configs/run_sim.yaml",
        help="Base config path.",
    )
    parser.add_argument(
        "--taskset-file",
        required=True,
        help="Taskset file path (each line is a taskset).",
    )
    parser.add_argument(
        "--out-config-dir",
        default="./outputs/sweep_configs",
        help="Directory to write generated configs.",
    )
    parser.add_argument(
        "--out-dir",
        default="./outputs/exp_result",
        help="Base directory to store run outputs.",
    )
    parser.add_argument(
        "--exp-name",
        default="refutil_sweep",
        help="Experiment name used under exp_result/exp_<name>.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start target_util value.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=1.0,
        help="End target_util value.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Step size for target_util.",
    )
    parser.add_argument(
        "--ref-util",
        type=float,
        default=0.9,
        help="Fixed controller ref_util value.",
    )
    parser.add_argument(
        "--workers",
        default="1",
        help="Number of parallel workers or 'max' for all cores.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for CUDA_VISIBLE_DEVICES (e.g., 0,1).",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    taskset_path = Path(args.taskset_file)
    if not taskset_path.exists():
        raise SystemExit(f"Taskset file not found: {taskset_path}")

    env_base = os.environ.copy()
    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [g.strip() for g in str(args.gpus).split(",") if g.strip()]

    run_sim_path = Path("run.py")
    workdir = run_sim_path.parent

    exp_root = Path(args.out_dir) / f"exp_{args.exp_name}"

    taskset_lines = load_taskset_lines(taskset_path)
    if not taskset_lines:
        raise SystemExit(f"No tasksets found in {taskset_path}")

    target_utils = ref_util_values(args.start, args.end, args.step)

    # Ensure the temp-config directory exists before writing tasksets/configs into it.
    Path(args.out_config_dir).mkdir(parents=True, exist_ok=True)

    jobs = []
    temp_tasksets = []

    for idx, line in enumerate(taskset_lines, start=1):
        per_taskset_dir = f"{taskset_path.stem}_set{idx}"
        tmp_taskset = Path(args.out_config_dir) / f"{taskset_path.stem}_set{idx}.txt"
        tmp_taskset.write_text(line + "\n", encoding="utf-8")
        temp_tasksets.append(tmp_taskset)

        for target_util in target_utils:
            cfg = copy.deepcopy(base_cfg)
            cfg.setdefault("taskset", {})["path"] = str(tmp_taskset)
            cfg.setdefault("controller", {})["ref_util"] = float(args.ref_util)
            cfg.setdefault("dynamic_config", {})["target_util"] = float(target_util)

            output_dir = exp_root / taskset_path.stem / per_taskset_dir / f"target_{target_util:.1f}"
            cfg["output"] = {"dir": str(output_dir)}

            cfg_name = f"{per_taskset_dir}_target{target_util:.1f}.yaml"
            cfg_path = Path(args.out_config_dir) / cfg_name
            write_config(cfg_path, cfg)

            jobs.append((run_sim_path, cfg_path, workdir))

    total_runs = len(jobs)
    if str(args.workers).lower() == "max":
        max_workers = os.cpu_count() or 1
    else:
        max_workers = int(args.workers)

    if max_workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            run_sim_path, cfg_path, workdir = job
            env = env_base.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[(idx - 1) % len(gpu_ids)]
            print(f"[{idx}/{total_runs}] Running {cfg_path.name} gpu={env.get('CUDA_VISIBLE_DEVICES', 'all')}", flush=True)
            run_sim(run_sim_path, cfg_path, workdir, env=env)
    else:
        print(f"Running {total_runs} jobs with {max_workers} workers.")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, (run_sim_path, cfg_path, workdir) in enumerate(jobs, start=1):
                env = env_base.copy()
                if gpu_ids:
                    env["CUDA_VISIBLE_DEVICES"] = gpu_ids[(idx - 1) % len(gpu_ids)]
                futures.append(executor.submit(run_job, (run_sim_path, cfg_path, workdir, env)))
            if tqdm is None:
                for fut in as_completed(futures):
                    fut.result()
            else:
                for fut in tqdm(as_completed(futures), total=len(futures), desc="runs"):
                    fut.result()

    for path in temp_tasksets:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


if __name__ == "__main__":
    main()
