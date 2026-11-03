"""Demo: schedule several periodic image-processing tasks on one shared worker under EDF.

Each task = a camera stream with its own period and QoS level (resolution). The scheduler
dispatches frame-jobs earliest-deadline-first and reports on-time / late / dropped frames
and worker utilization.

    python scripts/run_scheduler.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argos_va.pipeline.model import ImagePipeline
from argos_va.runtime.report import export_trace_csv, export_windows_csv
from argos_va.runtime.viz import plot_run
from argos_va.scheduler.edf import EdfScheduler
from argos_va.scheduler.model import Task
from argos_va.streams.source import SyntheticSource

DURATION_MS = 2000.0
WINDOW_MS = 200.0
RESULTS = Path(__file__).resolve().parents[1] / "results"


def build_tasks(device: str) -> list[Task]:
    src = SyntheticSource(native_hw=(1080, 1920), device=device, seed=0)
    # (period_ms, qos_level)  -- level indexes the pipeline resolution ladder
    specs = [
        (40.0, 3),    # 25 fps @ 640x360  (~11 ms)
        (66.0, 4),    # 15 fps @ 960x540  (~25 ms)
        (100.0, 2),   # 10 fps @ 480x270  (~7 ms)
        (50.0, 3),    # 20 fps @ 640x360  (~11 ms)
    ]
    return [Task(task_id=i, period_ms=p, source=src, qos_level=q, offset_ms=0.0)
            for i, (p, q) in enumerate(specs)]


def main() -> None:
    pipeline = ImagePipeline()
    print(f"device={pipeline.device}  ladder={pipeline.ladder}")
    pipeline.warmup()

    tasks = build_tasks(pipeline.device)
    sched = EdfScheduler(tasks, pipeline)
    trace = sched.run(DURATION_MS)

    print(f"\nrun: {DURATION_MS:.0f} ms   utilization={trace.utilization():.3f}")
    s = trace.summary()
    print(f"overall: released={s['released']} ontime={s['ontime']} "
          f"late={s['late']} dropped={s['dropped']} DMR={s['dmr']:.3f}")

    print(f"\n{'task':<8}{'period':>8}{'qos':>5}{'res':>11}"
          f"{'rel':>6}{'ontime':>8}{'miss':>6}{'DMR':>7}")
    pt = trace.per_task()
    for t in tasks:
        hw = pipeline.resolution(t.qos_level)
        res = f"{hw[1]}x{hw[0]}" if hw else "drop"
        d = pt.get(t.task_id, {"released": 0, "ontime": 0, "misses": 0, "dmr": 0.0})
        print(f"{t.name:<8}{t.period_ms:>7.0f}m{t.qos_level:>5}{res:>11}"
              f"{d['released']:>6}{d['ontime']:>8}{d['misses']:>6}{d['dmr']:>7.3f}")

    trace_csv = export_trace_csv(trace, RESULTS / "trace.csv")
    win_csv = export_windows_csv(trace, WINDOW_MS, RESULTS / "windows.csv")
    fig = plot_run(trace, tasks, WINDOW_MS, RESULTS / "schedule.png",
                   ladder=pipeline.ladder, title="ARGoS-VA — EDF schedule (fixed QoS)")
    print(f"\nlogged: {trace_csv}\n        {win_csv}\nplot:   {fig}")


if __name__ == "__main__":
    main()
