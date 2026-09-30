import argparse
import os
import random


def uunifast_discard(n_tasks, total_util, n_sets):
    sets = []
    while len(sets) < n_sets:
        # Classic UUniFast algorithm:
        utilizations = []
        sum_u = total_util
        for i in range(1, n_tasks):
            next_sum_u = sum_u * (random.random() ** (1.0 / (n_tasks - i)))
            utilizations.append(sum_u - next_sum_u)
            sum_u = next_sum_u
        utilizations.append(sum_u)

        # If no task utilization exceeds 1:
        if all(u <= 1 for u in utilizations):
            sets.append(utilizations)

    return sets


def format_util(u, decimals=3):
    text = f"{u:.{decimals}f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def main():
    parser = argparse.ArgumentParser(description="Generate taskset configs using UUniFast-Discard.")
    parser.add_argument("--total-util", type=float, required=True, help="Target total utilization.")
    parser.add_argument("--tasks", type=int, required=True, help="Number of tasks per taskset.")
    parser.add_argument("--count", type=int, required=True, help="Number of tasksets to generate.")
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for taskset files.",
    )
    parser.add_argument(
        "--out-file",
        default=None,
        help="Output filename. Defaults to a name derived from the config.",
    )
    parser.add_argument(
        "--max-util",
        type=float,
        default=1.0,
        help="Maximum utilization per task.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for utilization values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.total_util <= 0:
        raise SystemExit("total-util must be positive")
    if args.tasks <= 0:
        raise SystemExit("tasks must be positive")
    if args.count <= 0:
        raise SystemExit("count must be positive")

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.out_file is None:
        filename = f"tasksets_u{format_util(args.total_util, 2)}_n{args.tasks}_c{args.count}.txt"
    else:
        filename = args.out_file
    path = os.path.join(args.out_dir, filename)

    lines = []
    tasksets = uunifast_discard(args.tasks, args.total_util, args.count)
    for utils in tasksets:
        values = ",".join(format_util(u, args.decimals) for u in utils)
        lines.append(values)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
