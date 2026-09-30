import argparse
import os
import random


def uunifast_discard(n_tasks, total_util, n_sets, max_util=1.0, max_tries=100000):
    sets = []
    while len(sets) < n_sets:
        utilizations = []
        sum_u = total_util
        for i in range(1, n_tasks):
            next_sum_u = sum_u * (random.random() ** (1.0 / (n_tasks - i)))
            utilizations.append(sum_u - next_sum_u)
            sum_u = next_sum_u
        utilizations.append(sum_u)

        if all(u <= max_util for u in utilizations):
            sets.append(utilizations)
    return sets


def format_util(u, decimals=3):
    text = f"{u:.{decimals}f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def main():
    parser = argparse.ArgumentParser(description="Generate tasksets with a sub-utilization group.")
    parser.add_argument("--total-util", type=float, required=True, help="Target total utilization.")
    parser.add_argument("--sub-total-util", type=float, required=True, help="Target utilization for sub-tasks.")
    parser.add_argument("--tasks", type=int, required=True, help="Total number of tasks.")
    parser.add_argument("--sub-tasks", type=int, required=True, help="Number of sub-tasks.")
    parser.add_argument("--count", type=int, required=True, help="Number of tasksets to generate.")
    parser.add_argument(
        "--out-dir",
        default="./tasksets",
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
    if args.sub_total_util < 0:
        raise SystemExit("sub-total-util must be non-negative")
    if args.sub_total_util > args.total_util:
        raise SystemExit("sub-total-util must be <= total-util")
    if args.tasks <= 0:
        raise SystemExit("tasks must be positive")
    if args.sub_tasks <= 0:
        raise SystemExit("sub-tasks must be positive")
    if args.sub_tasks >= args.tasks:
        raise SystemExit("sub-tasks must be less than tasks")
    if args.count <= 0:
        raise SystemExit("count must be positive")

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.out_file is None:
        filename = (
            f"tasksets_u{format_util(args.total_util, 2)}_"
            f"sub{format_util(args.sub_total_util, 2)}_"
            f"n{args.tasks}_s{args.sub_tasks}_c{args.count}.txt"
        )
    else:
        filename = args.out_file
    path = os.path.join(args.out_dir, filename)

    lines = []
    remaining_util = args.total_util - args.sub_total_util

    sub_sets = uunifast_discard(args.sub_tasks, args.sub_total_util, args.count, max_util=args.max_util)
    main_sets = uunifast_discard(args.tasks - args.sub_tasks, remaining_util, args.count, max_util=args.max_util)

    for sub_utils, main_utils in zip(sub_sets, main_sets):
        utils = sub_utils + main_utils
        values = ",".join(format_util(u, args.decimals) for u in utils)
        lines.append(values)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
