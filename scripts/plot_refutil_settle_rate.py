#!/usr/bin/env python3
"""
Plot settle ratio by target value for refutil sweep experiments.

Definition:
- For each taskset/target directory, a taskset is considered "settled"
  if the last segment in summary["settle_by_segment"] has a non-null
  "settle" object with a non-null "settle_time_ms".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_target_dir(name: str) -> float | None:
    if not name.startswith("target_"):
        return None
    try:
        return float(name.split("_", 1)[1])
    except ValueError:
        return None


def _is_settled_in_last_segment(summary: dict) -> bool:
    segments = summary.get("settle_by_segment")
    if not segments or not isinstance(segments, list):
        return False
    last_seg = segments[-1]
    if not isinstance(last_seg, dict):
        return False
    settle = last_seg.get("settle")
    if not isinstance(settle, dict):
        return False
    return settle.get("settle_time_ms") is not None


def _scan_target_dir(target_dir: Path) -> Tuple[bool, int]:
    """Return (settled, num_summary_files)."""
    summaries = list(target_dir.glob("*_summary.json"))
    if not summaries:
        return False, 0
    for path in summaries:
        try:
            with path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue
        if _is_settled_in_last_segment(summary):
            return True, len(summaries)
    return False, len(summaries)


def _collect_results(base_dir: Path) -> Dict[float, Dict[str, int]]:
    results: Dict[float, Dict[str, int]] = {}
    for target_dir in base_dir.rglob("target_*"):
        if not target_dir.is_dir():
            continue
        target_val = _parse_target_dir(target_dir.name)
        if target_val is None:
            continue
        settled, num_summaries = _scan_target_dir(target_dir)
        if num_summaries == 0:
            continue
        info = results.setdefault(target_val, {"total": 0, "settled": 0})
        info["total"] += 1
        if settled:
            info["settled"] += 1
    return results


def _write_csv(results: Dict[float, Dict[str, int]], out_path: Path) -> None:
    lines = ["target,total_tasksets,settled_tasksets,settled_percent"]
    for target in sorted(results.keys()):
        total = results[target]["total"]
        settled = results[target]["settled"]
        percent = (settled / total * 100.0) if total else 0.0
        lines.append(f"{target},{total},{settled},{percent}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(results: Dict[float, Dict[str, int]], out_path: Path) -> None:
    targets = sorted(results.keys())
    if not targets:
        raise SystemExit("No target directories with summary files found.")
    percents = []
    for t in targets:
        total = results[t]["total"]
        settled = results[t]["settled"]
        percents.append((settled / total * 100.0) if total else 0.0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(targets, percents, marker="o", linewidth=1.2)
    ax.set_xlabel("target")
    ax.set_ylabel("settled_percent")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Settled Taskset Ratio by Target")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot settled taskset ratio by target for refutil sweep."
    )
    parser.add_argument(
        "--base-dir",
        default="./outputs/exp_result/exp_refutil_sweep",
        help="Base directory for refutil sweep results.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plot/CSV (default: base_dir/plots).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"base-dir not found: {base_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _collect_results(base_dir)
    if not results:
        raise SystemExit("No results found. Check base-dir and summary files.")

    csv_path = out_dir / "settled_ratio_by_target.csv"
    _write_csv(results, csv_path)

    plot_path = out_dir / "settled_ratio_by_target.png"
    _plot(results, plot_path)

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {plot_path}")


if __name__ == "__main__":
    main()
