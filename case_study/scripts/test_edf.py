"""Deterministic checks on the EDF dispatcher, using a fake fixed-cost pipeline so results
do not depend on GPU timing.

    python scripts/test_edf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argos_va.pipeline.model import InferResult
from argos_va.scheduler.edf import EdfScheduler
from argos_va.scheduler.model import Task
from argos_va.streams.source import FrameSource


class DummySource(FrameSource):
    def read(self):
        return None


class FixedCostPipeline:
    """exec_ms = cost_by_level[level], deterministic. No real compute."""
    def __init__(self, cost_by_level):
        self.cost_by_level = cost_by_level
        self.ladder = [None] + [(1, 1)] * len(cost_by_level)

    def resolution(self, level):
        return self.ladder[level]

    def process(self, frame, level):
        return InferResult(level=level, hw=(1, 1), exec_ms=float(self.cost_by_level[level]))


def check(name, cond):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_edf_order():
    """Two tasks release at t=0; the one with the earlier deadline (shorter period) must run
    first even though both are ready simultaneously."""
    pipe = FixedCostPipeline({1: 5.0})
    src = DummySource()
    tasks = [
        Task(task_id=0, period_ms=100.0, source=src, qos_level=1),  # deadline 100
        Task(task_id=1, period_ms=30.0, source=src, qos_level=1),   # deadline 30 -> first
    ]
    trace = EdfScheduler(tasks, pipe).run(20.0)
    first = trace.records[0]
    check("earlier-deadline task dispatched first", first.task_id == 1)
    check("both start at t=0 region", trace.records[0].start_ms == 0.0)
    check("second job starts after first finishes", trace.records[1].start_ms == 5.0)


def test_miss_and_drop():
    """Overload: a 40ms job on a 30ms period cannot keep up -> misses and drops."""
    pipe = FixedCostPipeline({1: 40.0})
    tasks = [Task(task_id=0, period_ms=30.0, source=DummySource(), qos_level=1)]
    trace = EdfScheduler(tasks, pipe).run(200.0)
    s = trace.summary()
    check("some frames missed under overload", s["misses"] > 0)
    check("released == ontime + misses", s["released"] == s["ontime"] + s["misses"])


def test_qos0_releases_nothing():
    """A task pinned to QoS 0 (drop) releases no job and cannot miss -> d_safe component."""
    pipe = FixedCostPipeline({1: 5.0})
    tasks = [Task(task_id=0, period_ms=30.0, source=DummySource(), qos_level=0)]
    trace = EdfScheduler(tasks, pipe).run(200.0)
    check("QoS-0 task releases no jobs", trace.summary()["released"] == 0)


def test_utilization():
    """One 10ms job every 50ms over 1000ms -> 20 jobs, util ~= 200/1000 = 0.2."""
    pipe = FixedCostPipeline({1: 10.0})
    tasks = [Task(task_id=0, period_ms=50.0, source=DummySource(), qos_level=1)]
    trace = EdfScheduler(tasks, pipe).run(1000.0)
    check("utilization ~= 0.2", abs(trace.utilization() - 0.2) < 1e-9)
    check("20 jobs released", trace.summary()["released"] == 20)


if __name__ == "__main__":
    for fn in [test_edf_order, test_miss_and_drop, test_qos0_releases_nothing, test_utilization]:
        print(fn.__name__)
        fn()
    print("\nall checks passed")
