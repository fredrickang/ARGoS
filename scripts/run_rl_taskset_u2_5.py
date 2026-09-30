import argparse
import copy
from pathlib import Path
import subprocess

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RL scheduler on tasksets_u2.5_n10_c5.txt (each line)."
    )
    parser.add_argument(
        "--config",
        default="./configs/run_sim.yaml",
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--taskset-file",
        default="./tasksets/tasksets_u2.5_n10_c5.txt",
        help="Taskset file to run (multi-line).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per taskset line.",
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
    args = parser.parse_args()

    base_cfg = load_config(Path(args.config))
    base_cfg.setdefault("controller", {})
    base_cfg["controller"]["mode"] = "rl"

    taskset_path = Path(args.taskset_file)
    lines = [ln.strip() for ln in taskset_path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"No tasksets in {taskset_path}")

    tmp_dir = taskset_path.parent / "tmp_rl_cfgs"
    tmp_dir.mkdir(exist_ok=True)

    run_idx = 0
    for line_idx, line in enumerate(lines, start=1):
        taskset_line_path = tmp_dir / f"{taskset_path.stem}_line{line_idx}.txt"
        taskset_line_path.write_text(line + "\n", encoding="utf-8")
        for i in range(args.runs):
            seed = args.seed_start + i
            cfg = copy.deepcopy(base_cfg)
            cfg["taskset"]["path"] = str(taskset_line_path)
            cfg.setdefault("controller", {})["profile_seed"] = seed
            tmp_cfg = tmp_dir / f"rl_{taskset_path.stem}_line{line_idx}_seed{seed}.yaml"
            save_config(cfg, tmp_cfg)
            run_idx += 1
            print(f"[{run_idx}] line={line_idx} seed={seed} cfg={tmp_cfg}")
            subprocess.run([args.python, "run.py", str(tmp_cfg)], check=True)


if __name__ == "__main__":
    main()
