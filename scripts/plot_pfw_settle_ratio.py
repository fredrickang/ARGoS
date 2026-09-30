import argparse
import json
import re
from pathlib import Path


PFW_RE = re.compile(r"_pfw(\d+)$")
UTIL_RE = re.compile(r"tasksets_u([0-9]+(?:\.[0-9]+)?)_n")


def parse_pfw(setting_dir: Path) -> int:
    m = PFW_RE.search(setting_dir.name)
    if not m:
        raise ValueError(f"failed to parse pfw from {setting_dir}")
    return int(m.group(1))


def parse_util(summary_path: Path) -> float:
    m = UTIL_RE.search(summary_path.name)
    if not m:
        raise ValueError(f"failed to parse util from {summary_path}")
    return float(m.group(1))


def load_settle_ratio(setting_dir: Path, target_util: float) -> tuple[int, float, float, int]:
    window_dir = setting_dir / "window_logs"
    summary_files = sorted(window_dir.glob("*_summary.json"))
    if not summary_files:
        raise ValueError(f"no summary files under {window_dir}")

    selected = [p for p in summary_files if abs(parse_util(p) - target_util) < 1e-9]
    if not selected:
        raise ValueError(f"no util={target_util} summaries under {window_dir}")

    settled = 0
    no_miss_window_count_sum = 0.0
    for path in selected:
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("settle", {}).get("settle_time_ms") is not None:
            settled += 1
        total_windows = int(summary.get("overall", {}).get("total_windows", 0))
        miss_windows = len(summary.get("miss_window_ids", []))
        no_miss_window_count_sum += float(max(0, total_windows - miss_windows))

    ratio = settled / len(selected)
    no_miss_window_count_mean = no_miss_window_count_sum / len(selected)
    return parse_pfw(setting_dir), ratio, no_miss_window_count_mean, len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot settled-taskset ratio vs profile_fallback_start_window."
    )
    parser.add_argument(
        "--exp-dir",
        default="./outputs/exp_result/my_experiment",
        help="Experiment directory containing setting*_pfw* folders.",
    )
    parser.add_argument(
        "--setting-prefix",
        default="setting3",
        help="Prefix of swept setting folders, e.g. setting3 -> setting3_pfw100.",
    )
    parser.add_argument(
        "--target-util",
        type=float,
        default=2.5,
        help="Target utilization to filter summaries.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    setting_dirs = sorted(
        p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith(f"{args.setting_prefix}_pfw")
    )
    if not setting_dirs:
        raise SystemExit(f"no setting folders matching {args.setting_prefix}_pfw* under {exp_dir}")

    rows = []
    for setting_dir in setting_dirs:
        pfw, ratio, no_miss_window_count_mean, n = load_settle_ratio(setting_dir, args.target_util)
        rows.append((pfw, ratio, no_miss_window_count_mean, n))

    rows.sort(key=lambda x: x[0])
    x = [r[0] for r in rows]
    y = [r[1] for r in rows]
    y_no_miss = [r[2] for r in rows]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    line1 = ax.plot(x, y, marker="o", linewidth=1.8, color="tab:blue", label="Settled Ratio")
    line2 = ax2.plot(
        x,
        y_no_miss,
        marker="s",
        linewidth=1.8,
        color="tab:red",
        linestyle="--",
        label="Avg # No-Miss Windows",
    )

    ax.set_xlabel("profile_fallback_start_window")
    ax.set_ylabel("Settled Taskset Ratio", color="tab:blue")
    ax2.set_ylabel("Avg # No-Miss Windows", color="tab:red")
    ax.set_title(f"Settle Ratio / No-Miss Windows vs PFW (util={args.target_util})")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()

    out_path = plots_dir / f"{args.setting_prefix}_u{args.target_util}_settled_ratio_vs_pfw.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"saved plot to {out_path}")
    for pfw, ratio, no_miss_window_count_mean, n in rows:
        print(
            f"pfw={pfw} settled_ratio={ratio:.4f} "
            f"avg_no_miss_windows={no_miss_window_count_mean:.2f} n={n}"
        )


if __name__ == "__main__":
    main()
