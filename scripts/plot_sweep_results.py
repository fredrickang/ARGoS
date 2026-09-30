import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_summary_files(base_dir):
    return list(Path(base_dir).rglob("*_summary.json"))


def parse_similarity(path):
    # Expect .../sim_0.9/... or .../sim_0.90/...
    for part in path.parts:
        if part.startswith("sim_"):
            try:
                return float(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def parse_metric(path):
    # Expect .../<metric>/sim_x/...
    parts = list(path.parts)
    for i, part in enumerate(parts):
        if part.startswith("sim_") and i > 0:
            return parts[i - 1]
    return None


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(values):
    return sum(values) / len(values) if values else None


def collect_metrics(summary):
    settle = summary.get("settle", {})
    return {
        "settle_time_ms": settle.get("settle_time_ms"),
        "post_settle_util_std": settle.get("post_settle_util_std"),
        "post_settle_util_avg": settle.get("post_settle_util_avg"),
    }


def plot_metric(ax, x, y, label, ylabel):
    ax.plot(x, y, marker="o", linewidth=1, label=label)
    ax.set_xlabel("Similarity")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)


def similarity_axis(metric):
    if metric == "cosine":
        return [round(0.9 + i * 0.01, 2) for i in range(11)]
    return [round(i * 0.1, 1) for i in range(11)]


def parse_metric_from_name(name):
    name = name.lower()
    if "mape" in name:
        return "mape"
    if "cosine" in name:
        return "cosine"
    return "cosine"


def parse_ref_util(path):
    for part in path.parts:
        if part.startswith("ref_"):
            try:
                return float(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot sweep results from summary JSON files.")
    parser.add_argument(
        "--base-dir",
        default="./outputs/exp_result/exp_similarity_sweep",
        help="Base output directory containing sweep results.",
    )
    parser.add_argument(
        "--out-dir",
        default="./outputs/exp_result/exp_similarity_sweep/plots",
        help="Directory to write plots.",
    )
    args = parser.parse_args()

    summaries = find_summary_files(args.base_dir)
    if not summaries:
        raise SystemExit(f"No summary JSON files found under {args.base_dir}")

    fcs_data = {}
    # fcs_data[metric][similarity] = list of metrics dict
    rl_data = {}
    # rl_data[metric] = list of metrics dict
    for path in summaries:
        if "rl" in path.parts:
            continue
        ref_util = parse_ref_util(path)
        if ref_util is None or abs(ref_util - 0.9) > 1e-9:
            continue
        similarity = parse_similarity(path)
        if similarity is None:
            continue
        metric = parse_metric(path)
        if metric is None:
            continue
        summary = load_summary(path)
        metrics = collect_metrics(summary)
        fcs_data.setdefault(metric, {}).setdefault(similarity, []).append(metrics)

    rl_summaries = list(Path(args.base_dir).rglob("rl/*_summary.json"))
    for path in rl_summaries:
        summary = load_summary(path)
        metric = parse_metric_from_name(path.name)
        metrics = collect_metrics(summary)
        rl_data.setdefault(metric, []).append(metrics)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, fcs_map in fcs_data.items():
        sims = sorted(fcs_map.keys())
        if metric == "cosine":
            sims = [s for s in sims if abs(s - 0.1) > 1e-9]
        if not sims:
            continue

        def _series(sim_map):
            settled_pct = []
            for sim in sims:
                records = sim_map.get(sim, [])
                settled = [r for r in records if r["settle_time_ms"] is not None]
                total = len(records)
                settled_pct.append((len(settled) / total * 100.0) if total else None)
            return settled_pct

        fcs_series = _series(fcs_map)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        if fcs_series:
            plot_metric(ax, sims, fcs_series, "fcs_settled_pct", "Settled Tasksets (%)")

        rl_records = rl_data.get(metric, [])
        if not rl_records and rl_data:
            rl_records = rl_data.get("cosine") or next(iter(rl_data.values()), [])
        if rl_records:
            rl_settled = [r for r in rl_records if r["settle_time_ms"] is not None]
            rl_total = len(rl_records)
            rl_settled_pct = (len(rl_settled) / rl_total * 100.0) if rl_total else None
            if rl_settled_pct is not None:
                ax.plot(sims, [rl_settled_pct] * len(sims), marker="o", linewidth=1, color="tab:orange", label="rl_settled_pct")
        ax.legend()
        ax.set_xlabel("Similarity")
        ax.set_ylabel("Settled Tasksets (%)")
        ax.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(f"{metric} Similarity Sweep - Settled Tasksets", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        out_path = out_dir / f"{metric}_fcs_sweep.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)



if __name__ == "__main__":
    main()
