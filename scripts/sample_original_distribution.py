import argparse
import glob
import pickle
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample data while keeping original class distribution."
    )
    parser.add_argument(
        "--input-dir",
        default="./dataset",
        help="Directory containing *.pkl files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pkl",
        help="Glob pattern for target files.",
    )
    parser.add_argument(
        "--output",
        default="./dataset/sampled.pkl",
        help="Output pickle path.",
    )
    parser.add_argument(
        "--train-out",
        default="./dataset/sampled_train.pkl",
        help="Train output pickle path.",
    )
    parser.add_argument(
        "--val-out",
        default="./dataset/sampled_val.pkl",
        help="Validation output pickle path.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Total samples to draw (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Sampling RNG seed.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    files = sorted(glob.glob(str(Path(args.input_dir) / args.pattern)))
    if not files:
        raise SystemExit(f"No .pkl files found in {args.input_dir}")

    data = []
    for p in files:
        with open(p, "rb") as f:
            data.extend(pickle.load(f))

    if not data:
        raise SystemExit("No samples loaded.")

    if args.count is None or args.count >= len(data):
        sample = data
    else:
        sample = rng.sample(data, args.count)

    with open(args.output, "wb") as f:
        pickle.dump(sample, f)

    rng.shuffle(sample)
    val_count = int(len(sample) * args.val_ratio)
    val_set = sample[:val_count]
    train_set = sample[val_count:]
    with open(args.train_out, "wb") as f:
        pickle.dump(train_set, f)
    with open(args.val_out, "wb") as f:
        pickle.dump(val_set, f)

    print(f"saved {len(sample)} samples to {args.output}")
    print(
        f"train/val split: {len(train_set)}/{len(val_set)} "
        f"-> {args.train_out}, {args.val_out}"
    )


if __name__ == "__main__":
    main()
