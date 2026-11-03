"""ARGoS in the loop: the RL agent chooses per-stream QoS each window and SafetyGuard
validates every decision, keeping the long-run deadline-miss loss below the target α.

    python scripts/run_argos.py

Produces results/argos_schedule.png (schedule + per-window util/DMR with the α line) and
results/argos_guarantee.png (running-avg miss loss vs α and the Theorem-1 bound).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argos_va.controller.argos import ArgosController
from argos_va.pipeline.model import ImagePipeline
from argos_va.runtime.report import export_trace_csv, export_windows_csv
from argos_va.runtime.viz import plot_guarantee, plot_run
from argos_va.scheduler.edf import EdfScheduler
from argos_va.scheduler.model import Task
from argos_va.streams.source import SyntheticSource

DURATION_MS = 40000.0
WINDOW_MS = 500.0
ALPHA = 0.10
RESULTS = Path(__file__).resolve().parents[1] / "results"


def build_tasks(device: str) -> list[Task]:
    src = SyntheticSource(native_hw=(1080, 1920), device=device, seed=0)
    periods = [40.0, 50.0, 66.0, 100.0, 33.0]   # 5 streams, mixed frame rates
    return [Task(task_id=i, period_ms=p, source=src, qos_level=1) for i, p in enumerate(periods)]


def main() -> None:
    pipeline = ImagePipeline()
    pipeline.warmup()
    tasks = build_tasks(pipeline.device)

    controller = ArgosController(tasks, num_levels=pipeline.num_levels(),
                                 alpha=ALPHA, gamma=0.05, u_ref=0.9,
                                 save_path=str(RESULTS / "models" / "argos_dqn.pth"))
    sched = EdfScheduler(tasks, pipeline)
    print(f"device={pipeline.device}  streams={len(tasks)}  levels={pipeline.num_levels()}  "
          f"W={WINDOW_MS:.0f}ms  α={ALPHA}")
    trace = sched.run_windowed(DURATION_MS, WINDOW_MS, controller)

    s = trace.summary()
    K = len(controller.loss_history)
    long_run = sum(controller.loss_history) / K if K else 0.0
    print(f"\nwindows={K}  overall DMR={s['dmr']:.4f}  utilization={trace.utilization():.3f}")
    print(f"long-run avg miss loss = {long_run:.4f}   (target α = {ALPHA})   "
          f"{'WITHIN BOUND' if long_run <= ALPHA else 'ABOVE'}")
    print(f"final λ={controller.lambda_history[-1]:.3f}  "
          f"fallbacks fired in {len(controller.fallback_windows)} windows")

    export_trace_csv(trace, RESULTS / "argos_trace.csv")
    export_windows_csv(trace, WINDOW_MS, RESULTS / "argos_windows.csv")
    p1 = plot_run(trace, tasks, WINDOW_MS, RESULTS / "argos_schedule.png",
                  ladder=pipeline.ladder, alpha=ALPHA,
                  title="ARGoS-VA — RL QoS under SafetyGuard")
    p2 = plot_guarantee(controller, RESULTS / "argos_guarantee.png")
    print(f"\nplots: {p1}\n       {p2}")


if __name__ == "__main__":
    main()
