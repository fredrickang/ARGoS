import argparse
import glob
import pickle
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample the profiling samples in ./dataset/profile and split into train/val."
    )
    parser.add_argument(
        "--pattern",
        default="profile/*.pkl",
        help="Glob pattern for input files (in this directory).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Total samples to draw (default: all).",
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
        help="Sampling RNG seed.",
    )
    parser.add_argument(
        "--train-out",
        default="train.pkl",
        help="Train output filename.",
    )
    parser.add_argument(
        "--val-out",
        default="val.pkl",
        help="Validation output filename.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    files = sorted(glob.glob(str(base_dir / args.pattern)))
    if not files:
        raise SystemExit(f"No files match pattern: {args.pattern}")

    data = []
    for p in files:
        with open(p, "rb") as f:
            data.extend(pickle.load(f))

    if not data:
        raise SystemExit("No samples loaded.")

    rng = random.Random(args.seed)
    if args.count is None or args.count >= len(data):
        sample = data
    else:
        sample = rng.sample(data, args.count)

    rng.shuffle(sample)
    val_count = int(len(sample) * args.val_ratio)
    val_set = sample[:val_count]
    train_set = sample[val_count:]

    train_path = base_dir / args.train_out
    val_path = base_dir / args.val_out
    with open(train_path, "wb") as f:
        pickle.dump(train_set, f)
    with open(val_path, "wb") as f:
        pickle.dump(val_set, f)

    print(f"train {len(train_set)} -> {train_path}")
    print(f"val {len(val_set)} -> {val_path}")


if __name__ == "__main__":
    main()
