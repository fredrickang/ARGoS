import argparse
import glob
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path


UTIL_RE = re.compile(r"tasksets_u([0-9.]+)_n")
SEED_RE = re.compile(r"_seed(\d+)")


def _parse_group(path: str) -> str:
    name = Path(path).name
    util_m = UTIL_RE.search(name)
    seed_m = SEED_RE.search(name)
    util = util_m.group(1) if util_m else "unknown"
    seed = seed_m.group(1) if seed_m else "unknown"
    return f"u{util}_seed{seed}"


def _allocate_targets(counts: dict[str, int], total: int) -> dict[str, int]:
    targets = {k: 0 for k in counts}
    remaining = total
    items = sorted(counts.items(), key=lambda x: x[1])
    for i, (k, cnt) in enumerate(items):
        groups_left = len(items) - i
        if groups_left <= 0:
            break
        share = remaining // groups_left
        take = min(cnt, share)
        targets[k] = take
        remaining -= take
    if remaining > 0:
        for k, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            if remaining <= 0:
                break
            room = cnt - targets[k]
            if room <= 0:
                continue
            add = min(room, remaining)
            targets[k] += add
            remaining -= add
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance dataset by util×seed, keeping all y_miss=0 and sampling y_miss=1."
    )
    parser.add_argument(
        "--input-dir",
        default="./dataset",
        help="Directory containing *.pkl files.",
    )
    parser.add_argument(
        "--output",
        default="./dataset/balanced.pkl",
        help="Output pickle path.",
    )
    parser.add_argument(
        "--train-out",
        default="./dataset/train.pkl",
        help="Train output pickle path.",
    )
    parser.add_argument(
        "--val-out",
        default="./dataset/val.pkl",
        help="Validation output pickle path.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed for sampling.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    files = sorted(glob.glob(str(Path(args.input_dir) / "*.pkl")))
    if not files:
        raise SystemExit(f"No .pkl files found in {args.input_dir}")

    counts_1 = defaultdict(int)
    y0_samples = []
    total_0 = 0

    for p in files:
        group = _parse_group(p)
        with open(p, "rb") as f:
            data = pickle.load(f)
        for item in data:
            if item.get("y_miss") == 1:
                counts_1[group] += 1
            else:
                y0_samples.append(item)
                total_0 += 1

    if total_0 == 0:
        raise SystemExit("No y_miss=0 samples found.")
    if not counts_1:
        raise SystemExit("No y_miss=1 samples found.")

    targets = _allocate_targets(counts_1, total_0)
    taken = defaultdict(int)
    reservoirs = {k: [] for k in targets}

    for p in files:
        group = _parse_group(p)
        target = targets.get(group, 0)
        if target <= 0:
            continue
        with open(p, "rb") as f:
            data = pickle.load(f)
        for item in data:
            if item.get("y_miss") != 1:
                continue
            taken[group] += 1
            if len(reservoirs[group]) < target:
                reservoirs[group].append(item)
            else:
                j = rng.randrange(taken[group])
                if j < target:
                    reservoirs[group][j] = item

    y1_samples = []
    for group, samples in reservoirs.items():
        if len(samples) < targets[group]:
            raise SystemExit(
                f"Group {group} has {len(samples)} samples, expected {targets[group]}"
            )
        y1_samples.extend(samples)

    balanced = y0_samples + y1_samples
    rng.shuffle(balanced)

    with open(args.output, "wb") as f:
        pickle.dump(balanced, f)

    val_count = int(len(balanced) * args.val_ratio)
    val_set = balanced[:val_count]
    train_set = balanced[val_count:]
    with open(args.train_out, "wb") as f:
        pickle.dump(train_set, f)
    with open(args.val_out, "wb") as f:
        pickle.dump(val_set, f)

    print(
        f"Saved {len(balanced)} samples to {args.output} "
        f"(y0={len(y0_samples)}, y1={len(y1_samples)}, groups={len(targets)})"
    )
    print(
        f"Train/val split: {len(train_set)}/{len(val_set)} "
        f"-> {args.train_out}, {args.val_out}"
    )


if __name__ == "__main__":
    main()
