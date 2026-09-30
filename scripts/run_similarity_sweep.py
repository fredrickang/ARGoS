import argparse
import copy
import subprocess
import sys
import os
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


def iter_taskset_files(taskset_dir):
    return sorted(Path(taskset_dir).glob("*.txt"))


def load_taskset_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def similarity_values(metric, step=0.1):
    metric = metric.lower()
    if metric == "cosine":
        return [round(0.9 + i * 0.01, 2) for i in range(11)]
    count = int(round(1.0 / step))
    return [round(i * step, 1) for i in range(count + 1)]

def ref_util_values():
    return [round(0.1 * i, 1) for i in range(1, 10)]


def run_sim(run_sim_path, cfg_path, workdir, env=None):
    cmd = [sys.executable, str(run_sim_path), str(cfg_path)]
    subprocess.run(cmd, cwd=workdir, check=True, env=env)


def run_job(args):
    run_sim_path, cfg_path, workdir, env = args
    run_sim(run_sim_path, cfg_path, workdir, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run similarity sweep for task sets.")
    parser.add_argument(
        "--config",
        default="./configs/run_sim.yaml",
        help="Base config path.",
    )
    parser.add_argument(
        "--taskset-file",
        default=None,
        help="Taskset file path (each line is a taskset).",
    )
    parser.add_argument(
        "--taskset-dir",
        default="./tasksets",
        help="Directory with taskset files (used when --taskset-file is not set).",
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
        default="similarity_sweep",
        help="Experiment name used under exp_result/exp_<name>.",
    )
    parser.add_argument(
        "--metrics",
        default="cosine,mape",
        help="Comma-separated similarity metrics.",
    )
    parser.add_argument(
        "--single-metric",
        default=None,
        help="Run only one similarity metric (e.g., cosine).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Similarity step size between 0 and 1.",
    )
    parser.add_argument(
        "--single-similarity",
        type=float,
        default=None,
        help="Run only one similarity value.",
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
    parser.add_argument(
        "--single-ref-util",
        type=float,
        default=None,
        help="Run only one ref_util value.",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    env_base = os.environ.copy()
    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [g.strip() for g in str(args.gpus).split(",") if g.strip()]
    if args.taskset_file:
        taskset_files = [Path(args.taskset_file)]
    else:
        taskset_files = iter_taskset_files(args.taskset_dir)
        if not taskset_files:
            raise SystemExit(f"No taskset files found in {args.taskset_dir}")

    if args.single_metric:
        metrics = [args.single_metric.strip()]
    else:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    run_sim_path = Path("run.py")
    workdir = run_sim_path.parent

    ctrl_mode = base_cfg.get("controller", {}).get("mode", "fcs").lower()
    fcs_cfg_base = base_cfg.get("fcs_ctrl_param", {})
    base_metric = fcs_cfg_base.get("similarity_metric", "cosine")
    base_similarity = fcs_cfg_base.get("similarity", 0.1)

    jobs = []

    exp_root = Path(args.out_dir) / f"exp_{args.exp_name}"

    # Ensure the temp-config directory exists before writing tasksets/configs into it.
    Path(args.out_config_dir).mkdir(parents=True, exist_ok=True)

    temp_tasksets = []

    for taskset_path in taskset_files:
        taskset_config_name = taskset_path.stem
        taskset_lines = load_taskset_lines(taskset_path)
        for idx, line in enumerate(taskset_lines, start=1):
            per_taskset_dir = f"{taskset_path.stem}_set{idx}"
            tmp_taskset = Path(args.out_config_dir) / f"{taskset_path.stem}_set{idx}.txt"
            tmp_taskset.write_text(line + "\n", encoding="utf-8")
            temp_tasksets.append(tmp_taskset)
            if ctrl_mode == "rl":
                metrics_to_use = [base_metric]
            else:
                metrics_to_use = metrics
            for metric in metrics_to_use:
                if ctrl_mode == "rl":
                    sims = [float(base_similarity)]
                elif args.single_similarity is not None:
                    sims = [float(args.single_similarity)]
                else:
                    sims = similarity_values(metric, args.step)
                for sim in sims:
                    if args.single_ref_util is not None:
                        ref_utils = [float(args.single_ref_util)]
                    else:
                        ref_utils = ref_util_values()
                    for ref_util in ref_utils:
                        cfg = copy.deepcopy(base_cfg)
                        cfg.setdefault("taskset", {})["path"] = str(tmp_taskset)
                        cfg.setdefault("controller", {})["ref_util"] = float(ref_util)
                        fcs_cfg = cfg.setdefault("fcs_ctrl_param", {})
                        fcs_cfg["similarity_metric"] = metric
                        fcs_cfg["similarity"] = float(sim)
                        output_dir = exp_root / taskset_config_name / per_taskset_dir / metric / f"sim_{sim:.2f}" / f"ref_{ref_util:.1f}"
                        cfg["output"] = {"dir": str(output_dir)}

                        cfg_name = f"{per_taskset_dir}_{metric}_sim{sim:.2f}_ref{ref_util:.1f}.yaml"
                        cfg_path = Path(args.out_config_dir) / cfg_name
                        write_config(cfg_path, cfg)

                        jobs.append((run_sim_path, cfg_path, workdir, per_taskset_dir, metric, sim, ref_util))

    total_runs = len(jobs)
    if str(args.workers).lower() == "max":
        max_workers = os.cpu_count() or 1
    else:
        max_workers = int(args.workers)

    if max_workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            _, cfg_path, _, taskset_name, metric, sim, ref_util = job
            env = env_base.copy()
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[(idx - 1) % len(gpu_ids)]
            print(
                f"[{idx}/{total_runs}] Running taskset={taskset_name} metric={metric} similarity={sim:.2f} ref_util={ref_util:.1f} gpu={env.get('CUDA_VISIBLE_DEVICES', 'all')}",
                flush=True,
            )
            run_sim(run_sim_path, cfg_path, workdir, env=env)
        return

    print(f"Running {total_runs} jobs with {max_workers} workers.")
    for idx, (_, _, _, taskset_name, metric, sim, ref_util) in enumerate(jobs, start=1):
        gpu = gpu_ids[(idx - 1) % len(gpu_ids)] if gpu_ids else "all"
        print(f"Queued taskset={taskset_name} metric={metric} similarity={sim:.2f} ref_util={ref_util:.1f} gpu={gpu}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, (run_sim_path, cfg_path, workdir, _, _, _, _) in enumerate(jobs, start=1):
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
