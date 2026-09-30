"""Pretrain a general warm-start policy across MANY tasksets x MANY ref-utils.

Why: a checkpoint trained on a single (taskset, ref=0.7) is a narrow prior -- warm-starting
from it fails to adapt when the env moves to an unseen regime (e.g. ref->0.9 got stuck).
Because the RL state already includes ref_util, one network can learn a ref-CONDITIONED
policy; exposing it to a range of refs AND a range of tasksets (same task count = same net
dims) yields a policy that transfers.

How: sequential checkpoint chaining. For each (pass, taskset): load the running checkpoint,
run a training episode that cycles ref_util across the run, save the checkpoint back. Multiple
passes + interleaved tasksets reduce catastrophic forgetting. All tasksets must have the same
task count (network dims are fixed); we extract N diverse same-size tasksets from a source file.

Example:
    python scripts/pretrain_multi_taskset.py --n-tasksets 10 --passes 2 \
        --out outputs/pretrain/general_policy.pth
"""

import argparse
import copy
import glob
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_tasksets(source, out_dir, n, stride=None):
    """Extract N single-line taskset files (the loader only supports one line per file)."""
    lines = [ln.strip() for ln in open(source) if ln.strip()]
    if stride is None:
        stride = max(1, len(lines) // n)
    picked = lines[::stride][:n]
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, ln in enumerate(picked):
        p = os.path.join(out_dir, f"ts_{i:02d}.txt")
        with open(p, "w") as f:
            f.write(ln + "\n")
        paths.append(p)
    return paths


def ref_cycle_changes(refs, seg_windows, n_segs):
    """ref_util change events cycling through `refs` every seg_windows."""
    evs = []
    for k in range(1, n_segs):
        evs.append({"at_window": k * seg_windows,
                    "kind": "ref_util", "value": float(refs[k % len(refs)])})
    return evs


def build_train_config(base, *, taskset_path, start_ref, changes, ckpt, duration_ms, out_dir):
    cfg = copy.deepcopy(base)
    cfg["controller"]["mode"] = "rl"
    cfg["controller"]["dynamic_mode"] = 1
    cfg["controller"]["ref_util"] = float(start_ref)
    cfg["taskset"]["path"] = taskset_path
    rl = dict(cfg["controller"].get("rl") or {})
    # Training exploration: sustained (do NOT anneal to 0) so the net keeps exploring the whole
    # ref range on every taskset; strong re-explore on each ref change.
    rl["epsilon"] = 0.3
    rl["epsilon_decay_per_window"] = False
    rl["epsilon_reset"] = 0.5
    rl["save_model"] = True
    rl["save_path"] = ckpt
    if os.path.exists(ckpt):
        rl["load_path"] = ckpt          # continue from the running checkpoint (chaining)
    rl.pop("seed", None)
    cfg["controller"]["rl"] = rl
    dyn = dict(cfg.get("dynamic_config") or {})
    dyn["changes"] = changes
    dyn["adaptation"] = "reset_only"
    dyn["pool_extra"] = 0
    dyn["dynamic_window"] = False
    cfg["dynamic_config"] = dyn
    cfg.setdefault("output", {})["dir"] = out_dir
    cfg["simulation"]["duration_ms"] = int(duration_ms)
    return cfg


def main():
    p = argparse.ArgumentParser(description="Multi-taskset x multi-ref warm-start pretraining.")
    p.add_argument("--config", default="./configs/run_sim.yaml")
    p.add_argument("--source", default="tasksets/tasksets_u2.0_to_3.0_n10_c5_all.txt",
                   help="Source file of same-size tasksets (one per line).")
    p.add_argument("--n-tasksets", type=int, default=10)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--refs", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--seg-windows", type=int, default=300, help="Windows per ref segment.")
    p.add_argument("--segs", type=int, default=7, help="Ref segments per taskset episode.")
    p.add_argument("--out", default="outputs/pretrain/general_policy.pth")
    p.add_argument("--workdir-out", default="outputs/pretrain")
    args = p.parse_args()

    workdir = Path(__file__).resolve().parents[1]
    run_py = workdir / "run.py"
    base = load_config(args.config)
    ckpt = str(Path(args.out).resolve())
    Path(args.workdir_out).mkdir(parents=True, exist_ok=True)
    if os.path.exists(ckpt):
        os.remove(ckpt)  # fresh start

    ts_paths = extract_tasksets(args.source, os.path.join(args.workdir_out, "_tasksets"),
                                args.n_tasksets)
    duration_ms = args.seg_windows * args.segs * 5000  # w_sz assumed 5000ms
    print(f"[pretrain] {len(ts_paths)} tasksets x {args.passes} passes "
          f"= {len(ts_paths)*args.passes} episodes, refs={args.refs}")

    episode = 0
    for pass_i in range(args.passes):
        for ti, ts in enumerate(ts_paths):
            episode += 1
            start_ref = args.refs[episode % len(args.refs)]
            changes = ref_cycle_changes(args.refs, args.seg_windows, args.segs)
            out_dir = os.path.join(args.workdir_out, f"p{pass_i}_ts{ti:02d}")
            cfg = build_train_config(base, taskset_path=ts, start_ref=start_ref,
                                     changes=changes, ckpt=ckpt, duration_ms=duration_ms,
                                     out_dir=out_dir)
            cfg_path = os.path.join(args.workdir_out, f"_cfg_p{pass_i}_ts{ti:02d}.yaml")
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"[pretrain] episode {episode}/{len(ts_paths)*args.passes} "
                  f"pass={pass_i} taskset={os.path.basename(ts)} start_ref={start_ref}")
            subprocess.run([sys.executable, str(run_py), cfg_path], cwd=workdir, check=True)

    print(f"\n[pretrain] DONE. General policy -> {ckpt}")
    print(f"[pretrain] warm-start with:  controller.rl.load_path: {ckpt}")


if __name__ == "__main__":
    main()
