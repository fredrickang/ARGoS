import argparse
import copy
import os
import sys
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_job(python_exe: str, cfg_path: str, gpu_id: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    subprocess.run([python_exe, "run.py", cfg_path], check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RL mode over taskset files with varying seeds."
    )
    parser.add_argument(
        "--config",
        default="./configs/run_sim.yaml",
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--taskset-dir",
        default="./tasksets",
        help="Directory containing taskset files.",
    )
    parser.add_argument(
        "--pattern",
        default="tasksets_u*_n10_c5.txt",
        help="Glob pattern for taskset files.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per taskset (seed varies per run).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed value (inclusive).",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="Python executable to run simulations.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids to use.",
    )
    parser.add_argument(
        "--exp-root",
        default="./outputs/exp_result",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--exp-group",
        default="experiment",
        help="Top-level experiment folder name under exp-root.",
    )
    parser.add_argument(
        "--exp-name",
        default="setting1",
        help="Setting folder name under exp-root/exp-group.",
    )
    parser.add_argument(
        "--fallback-start-sweep",
        action="store_true",
        help="Sweep controller.profile_fallback_start_window over a range.",
    )
    parser.add_argument(
        "--fallback-start-begin",
        type=int,
        default=100,
        help="Fallback-start sweep begin (inclusive).",
    )
    parser.add_argument(
        "--fallback-start-end",
        type=int,
        default=1000,
        help="Fallback-start sweep end (inclusive).",
    )
    parser.add_argument(
        "--fallback-start-step",
        type=int,
        default=100,
        help="Fallback-start sweep step.",
    )
    args = parser.parse_args()

    base_cfg = load_config(Path(args.config))
    base_cfg.setdefault("controller", {})
    base_cfg["controller"]["mode"] = "rl"

    taskset_dir = Path(args.taskset_dir)
    taskset_files = sorted(taskset_dir.glob(args.pattern))
    if not taskset_files:
        raise SystemExit(f"No taskset files match: {taskset_dir / args.pattern}")

    if args.fallback_start_sweep:
        fallback_values = list(
            range(
                args.fallback_start_begin,
                args.fallback_start_end + 1,
                args.fallback_start_step,
            )
        )
    else:
        fallback_values = [
            base_cfg.get("controller", {}).get("profile_fallback_start_window", 500)
        ]

    jobs = []
    run_idx = 0
    output_dirs = []
    for fallback_start in fallback_values:
        if args.fallback_start_sweep:
            exp_name = f"{args.exp_name}_pfw{fallback_start}"
        else:
            exp_name = args.exp_name
        exp_dir = Path(args.exp_root) / args.exp_group / exp_name
        output_dir = exp_dir / "window_logs"
        tmp_dir = exp_dir / "tmp_rl_cfgs"
        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(exist_ok=True)
        output_dirs.append(output_dir)

        for taskset_path in taskset_files:
            lines = [ln.strip() for ln in taskset_path.read_text().splitlines() if ln.strip()]
            if not lines:
                continue
            for line_idx, line in enumerate(lines, start=1):
                taskset_line_path = tmp_dir / f"{taskset_path.stem}_line{line_idx}.txt"
                taskset_line_path.write_text(line + "\n", encoding="utf-8")
                for i in range(args.runs):
                    seed = args.seed_start + i
                    cfg = copy.deepcopy(base_cfg)
                    cfg["taskset"]["path"] = str(taskset_line_path)
                    cfg.setdefault("controller", {})["profile_seed"] = seed
                    cfg.setdefault("controller", {})["profile_fallback_start_window"] = fallback_start
                    cfg.setdefault("output", {})["dir"] = str(output_dir)
                    tmp_cfg = tmp_dir / (
                        f"rl_{taskset_path.stem}_line{line_idx}_seed{seed}_pfw{fallback_start}.yaml"
                    )
                    save_config(cfg, tmp_cfg)
                    run_idx += 1
                    jobs.append(str(tmp_cfg))

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided.")

    total = len(jobs)
    completed = 0

    def _progress(done: int, total: int) -> None:
        width = 30
        filled = int(width * done / total) if total else width
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r[{bar}] {done}/{total}")
        sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=len(gpus)) as exe:
        futures = []
        for idx, cfg_path in enumerate(jobs):
            gpu_id = gpus[idx % len(gpus)]
            futures.append(exe.submit(_run_job, args.python, cfg_path, gpu_id))
        _progress(0, total)
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            _progress(completed, total)
    sys.stdout.write("\n")
    for output_dir in sorted(set(output_dirs)):
        print(f"saved summaries/results under {output_dir}")


if __name__ == "__main__":
    main()
