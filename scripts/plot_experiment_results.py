import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


UTIL_RE = re.compile(r"tasksets_u([0-9]+(?:\.[0-9]+)?)_n")
SETTING_LABELS = {
    "setting1": "w/o conformal prediction",
    "setting2": "w/ conformal prediction",
}


def parse_util(path: Path) -> float:
    m = UTIL_RE.search(path.name)
    if not m:
        raise ValueError(f"failed to parse utilization from {path}")
    return float(m.group(1))


def display_label(setting: str) -> str:
    return SETTING_LABELS.get(setting, setting)


def load_setting(
    setting_dir: Path,
    miss_window_start: int,
    miss_window_total: int,
) -> dict[float, dict[str, float]]:
    summary_files = sorted((setting_dir / "window_logs").glob("*_summary.json"))
    if not summary_files:
        raise SystemExit(f"no summary files found under {setting_dir / 'window_logs'}")

    per_util = defaultdict(list)
    for path in summary_files:
        util = parse_util(path)
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        settle = summary.get("settle", {})
        miss_window_ids = summary.get("miss_window_ids", [])
        per_util[util].append(
            {
                "settled": 1.0 if settle.get("settle_time_ms") is not None else 0.0,
                "miss_windows_ratio_after_start": (
                    float(sum(1 for w in miss_window_ids if int(w) >= miss_window_start))
                    / max(1, miss_window_total)
                ),
            }
        )

    stats = {}
    for util, rows in sorted(per_util.items()):
        n = len(rows)
        settled_ratio = sum(r["settled"] for r in rows) / max(1, n)
        miss_windows_mean = sum(r["miss_windows_ratio_after_start"] for r in rows) / max(1, n)
        stats[util] = {
            "count": n,
            "settled_ratio": settled_ratio,
            "miss_windows_ratio_mean": miss_windows_mean,
        }
    return stats


def plot_results(
    exp_dir: Path,
    settings: list[str],
    miss_window_start: int,
    miss_window_total: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded = {}
    all_utils = set()
    for setting in settings:
        stats = load_setting(exp_dir / setting, miss_window_start, miss_window_total)
        loaded[setting] = stats
        all_utils.update(stats.keys())

    utils = sorted(all_utils)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for setting in settings:
        stats = loaded[setting]
        ys = [stats.get(u, {}).get("settled_ratio", float("nan")) for u in utils]
        ax.plot(utils, ys, marker="o", linewidth=1.5, label=display_label(setting))
    ax.set_xlabel("Utilization")
    ax.set_ylabel("Settled Taskset Ratio")
    ax.set_xticks(utils)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "settled_ratio_vs_util.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for setting in settings:
        stats = loaded[setting]
        ys = [stats.get(u, {}).get("miss_windows_ratio_mean", float("nan")) for u in utils]
        ax.plot(utils, ys, marker="o", linewidth=1.5, label=display_label(setting))
    ax.set_xlabel("Utilization")
    ax.set_ylabel(f"Miss Window Ratio (window_id >= {miss_window_start})")
    ax.set_xticks(utils)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"miss_window_ratio_from_{miss_window_start}_vs_util.png", dpi=160)
    plt.close(fig)

    print(f"saved plots under {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot experiment summary comparisons.")
    parser.add_argument(
        "--exp-dir",
        default="./outputs/exp_result/my_experiment",
        help="Experiment directory containing setting subdirectories.",
    )
    parser.add_argument(
        "--settings",
        default="setting1,setting2",
        help="Comma-separated setting directory names to compare.",
    )
    parser.add_argument(
        "--miss-window-start",
        type=int,
        default=500,
        help="Only count deadline-miss windows whose window_id is at least this value.",
    )
    parser.add_argument(
        "--miss-window-total",
        type=int,
        default=1500,
        help="Total number of windows in the counting interval, used as denominator.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    if not settings:
        raise SystemExit("no settings provided")

    plot_results(exp_dir, settings, args.miss_window_start, args.miss_window_total)


if __name__ == "__main__":
    main()
