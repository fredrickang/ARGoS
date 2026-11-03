"""Durable logging of a scheduler run: the per-job trace and the per-window aggregates,
written as CSV for later analysis or plotting."""
from __future__ import annotations

import csv
from pathlib import Path

from ..scheduler.edf import RunTrace


def export_trace_csv(trace: RunTrace, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "job_id", "level", "release_ms", "deadline_ms",
                    "start_ms", "finish_ms", "exec_ms", "status"])
        for r in trace.records:
            w.writerow([r.task_id, r.job_id, r.level,
                        f"{r.release_ms:.3f}", f"{r.deadline_ms:.3f}",
                        "" if r.start_ms is None else f"{r.start_ms:.3f}",
                        "" if r.finish_ms is None else f"{r.finish_ms:.3f}",
                        f"{r.exec_ms:.3f}", r.status])
    return path


def export_windows_csv(trace: RunTrace, W: float, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "start_ms", "end_ms", "utilization", "released", "misses", "dmr"])
        for win in trace.windows(W):
            w.writerow([win.index, f"{win.start_ms:.1f}", f"{win.end_ms:.1f}",
                        f"{win.utilization:.4f}", win.released, win.misses, f"{win.dmr:.4f}"])
    return path
