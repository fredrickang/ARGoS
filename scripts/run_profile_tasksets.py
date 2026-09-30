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


def _run_job(python_exe: str, cfg_path: str, taskset_name: str, seed: int, idx: int) -> None:
    subprocess.run([python_exe, "run.py", cfg_path], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run profile mode over taskset files with varying seeds."
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
        default=10,
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
        "--jobs",
        type=int,
        default=None,
        help="Maximum parallel jobs (defaults to CPU count).",
    )
    args = parser.parse_args()

    base_cfg = load_config(Path(args.config))
    base_cfg.setdefault("controller", {})
    base_cfg["controller"]["mode"] = "profile"

    taskset_dir = Path(args.taskset_dir)
    taskset_files = sorted(taskset_dir.glob(args.pattern))
    if not taskset_files:
        raise SystemExit(f"No taskset files match: {taskset_dir / args.pattern}")

    tmp_dir = taskset_dir / "tmp_profile_cfgs"
    tmp_dir.mkdir(exist_ok=True)

    jobs = []
    run_idx = 0
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
                cfg["controller"]["profile_seed"] = seed
                tmp_cfg = tmp_dir / f"profile_{taskset_path.stem}_line{line_idx}_seed{seed}.yaml"
                save_config(cfg, tmp_cfg)
                run_idx += 1
                jobs.append((str(tmp_cfg), taskset_path.name, seed, run_idx))

    max_workers = args.jobs or os.cpu_count() or 1
    total = len(jobs)
    completed = 0

    def _progress(done: int, total: int) -> None:
        width = 30
        filled = int(width * done / total) if total else width
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r[{bar}] {done}/{total}")
        sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = [
            exe.submit(_run_job, args.python, cfg_path, taskset_name, seed, idx)
            for cfg_path, taskset_name, seed, idx in jobs
        ]
        _progress(0, total)
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            _progress(completed, total)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
