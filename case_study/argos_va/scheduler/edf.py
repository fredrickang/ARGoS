"""Non-preemptive EDF dispatcher on a single shared worker.

The shared compute worker (one GPU here) processes one frame-job at a time. Jobs are
dispatched earliest-deadline-first among those released and not yet handled. Frame-jobs run
to completion (non-preemptive): a running inference is never interrupted mid-frame, which
matches a real analytics pipeline and keeps measured execution times clean.

Time is event-driven: a virtual clock advances by each job's *measured* execution time, so
arrival order and EDF decisions are exactly reproducible while the compute cost is real.

Overload policy: a job whose deadline has already passed at dispatch is dropped (a late
result is useless to the consumer) and counted as a miss, rather than wasting the worker on
it. A task at QoS level 0 releases no job (a dropped stream cannot miss a deadline) -- this
is the per-task component of the paper's known-safe decision d_safe.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from ..pipeline.model import ImagePipeline
from .model import Job, JobRecord, Task, WindowObs


@dataclass
class WindowStat:
    index: int
    start_ms: float
    end_ms: float
    busy_ms: float = 0.0
    released: int = 0
    misses: int = 0

    @property
    def utilization(self) -> float:
        span = self.end_ms - self.start_ms
        return self.busy_ms / span if span else 0.0

    @property
    def dmr(self) -> float:
        return self.misses / self.released if self.released else 0.0


@dataclass
class RunTrace:
    records: list[JobRecord] = field(default_factory=list)
    duration_ms: float = 0.0
    busy_ms: float = 0.0

    def utilization(self) -> float:
        return self.busy_ms / self.duration_ms if self.duration_ms else 0.0

    def windows(self, W: float) -> list[WindowStat]:
        """Aggregate the trace into fixed windows of length W. Busy time is split across
        window boundaries by overlap; a frame is released into its release window and, if it
        misses, counted in the window its deadline falls in (matching the paper's per-window
        loss accounting)."""
        n = max(1, math.ceil(self.duration_ms / W))
        wins = [WindowStat(i, i * W, min((i + 1) * W, self.duration_ms)) for i in range(n)]
        for r in self.records:
            rb = min(int(r.release_ms / W), n - 1)
            wins[rb].released += 1
            if r.status != "ontime":
                db = min(int(r.deadline_ms / W), n - 1)
                wins[db].misses += 1
            if r.start_ms is not None and r.exec_ms > 0:
                lo, hi = r.start_ms, r.finish_ms
                first, last = int(lo / W), min(int(hi / W), n - 1)
                for i in range(first, last + 1):
                    ov = min(hi, wins[i].end_ms) - max(lo, wins[i].start_ms)
                    if ov > 0:
                        wins[i].busy_ms += ov
        return wins

    def summary(self) -> dict:
        rel = len(self.records)
        ontime = sum(r.status == "ontime" for r in self.records)
        late = sum(r.status == "late" for r in self.records)
        dropped = sum(r.status == "dropped" for r in self.records)
        misses = late + dropped
        return {
            "released": rel,
            "ontime": ontime,
            "late": late,
            "dropped": dropped,
            "misses": misses,
            "dmr": misses / rel if rel else 0.0,
            "utilization": self.utilization(),
        }

    def per_task(self) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for r in self.records:
            d = out.setdefault(r.task_id, {"released": 0, "ontime": 0, "misses": 0})
            d["released"] += 1
            if r.status == "ontime":
                d["ontime"] += 1
            else:
                d["misses"] += 1
        for d in out.values():
            d["dmr"] = d["misses"] / d["released"] if d["released"] else 0.0
        return out


class EdfScheduler:
    def __init__(self, tasks: list[Task], pipeline: ImagePipeline, *,
                 drop_expired: bool = True):
        self.tasks = {t.task_id: t for t in tasks}
        self.pipeline = pipeline
        self.drop_expired = drop_expired

    def run(self, duration_ms: float) -> RunTrace:
        tasks = self.tasks
        next_release = {tid: t.offset_ms for tid, t in tasks.items()}
        job_counter = {tid: 0 for tid in tasks}
        ready: list[Job] = []           # min-heap by (deadline, release, task_id)
        trace = RunTrace(duration_ms=duration_ms)
        t = 0.0

        def release_due(now: float) -> None:
            # release every job whose release time has arrived; skip QoS-0 (dropped) tasks
            for tid, task in tasks.items():
                while next_release[tid] <= now and next_release[tid] < duration_ms:
                    rel = next_release[tid]
                    next_release[tid] += task.period_ms
                    if task.qos_level <= 0:
                        continue
                    job_counter[tid] += 1
                    heapq.heappush(ready, Job(
                        deadline_ms=rel + task.period_ms,
                        release_ms=rel,
                        task_id=tid,
                        job_id=job_counter[tid],
                        level=task.qos_level,
                    ))

        while t < duration_ms:
            release_due(t)
            if not ready:
                # worker idle: jump to the next release (skipping QoS-0 tasks)
                future = [next_release[tid] for tid, tk in tasks.items()
                          if tk.qos_level > 0 and next_release[tid] < duration_ms]
                if not future:
                    break
                t = min(future)
                continue

            job = heapq.heappop(ready)

            if self.drop_expired and job.deadline_ms <= t:
                trace.records.append(JobRecord(
                    task_id=job.task_id, job_id=job.job_id, level=job.level,
                    release_ms=job.release_ms, deadline_ms=job.deadline_ms,
                    start_ms=None, finish_ms=None, exec_ms=0.0, status="dropped"))
                continue

            frame = tasks[job.task_id].source.read()
            result = self.pipeline.process(frame, job.level)
            start, finish = t, t + result.exec_ms
            t = finish
            trace.busy_ms += result.exec_ms
            trace.records.append(JobRecord(
                task_id=job.task_id, job_id=job.job_id, level=job.level,
                release_ms=job.release_ms, deadline_ms=job.deadline_ms,
                start_ms=start, finish_ms=finish, exec_ms=result.exec_ms,
                status="late" if finish > job.deadline_ms else "ontime"))

        return trace

    def run_windowed(self, duration_ms: float, window_ms: float, controller) -> RunTrace:
        """Run the schedule window by window, letting the controller reassign each stream's
        QoS once per window. Jobs capture the QoS level in force at their release time, so a
        job released under decision d_{k-1} keeps that level even if its deadline falls in
        window k -- this is exactly the carry-in behaviour the ARGoS guarantee accounts for.

        The controller sees `controller.decide(k)` at the start of window k (it mutates the
        shared Task objects' qos_level), and `controller.observe(WindowObs)` at the end of
        window k once every deadline in that window has been resolved.
        """
        W = window_ms
        tasks = self.tasks
        n_windows = max(1, math.ceil(duration_ms / W))
        # max releasable jobs per window over the admitted set -> loss denominator N_job (Eq. 3)
        n_job = sum(math.ceil(W / t.period_ms) for t in tasks.values())

        next_release = {tid: t.offset_ms for tid, t in tasks.items()}
        job_counter = {tid: 0 for tid in tasks}
        ready: list[Job] = []
        trace = RunTrace(duration_ms=duration_ms)

        win_records: list[list[JobRecord]] = [[] for _ in range(n_windows)]
        win_busy = [0.0] * n_windows
        win_misses = [0] * n_windows
        win_released = [0] * n_windows

        def widx(x: float) -> int:
            return min(int(x / W), n_windows - 1)

        def record(r: JobRecord) -> None:
            trace.records.append(r)
            win_records[widx(r.deadline_ms)].append(r)
            win_released[widx(r.release_ms)] += 1
            if r.status != "ontime":
                win_misses[widx(r.deadline_ms)] += 1

        def release_due(now: float) -> None:
            for tid, task in tasks.items():
                while next_release[tid] <= now and next_release[tid] < duration_ms:
                    rel = next_release[tid]
                    next_release[tid] += task.period_ms
                    if task.qos_level <= 0:
                        continue
                    job_counter[tid] += 1
                    heapq.heappush(ready, Job(rel + task.period_ms, rel, tid,
                                              job_counter[tid], task.qos_level))

        def flush_overdue(bound: float) -> None:
            # every ready job with deadline <= bound is now overdue (its window has closed);
            # resolve it as a dropped miss. Jobs are atomic, so nothing is mid-execution.
            keep = []
            while ready:
                j = heapq.heappop(ready)
                if j.deadline_ms <= bound:
                    record(JobRecord(j.task_id, j.job_id, j.level, j.release_ms,
                                     j.deadline_ms, None, None, 0.0, "dropped"))
                else:
                    keep.append(j)
            for j in keep:
                heapq.heappush(ready, j)

        def close_window(k: int) -> None:
            controller.observe(WindowObs(
                index=k, utilization=win_busy[k] / W if W else 0.0,
                loss=win_misses[k] / n_job if n_job else 0.0,
                released=win_released[k], misses=win_misses[k], n_job=n_job,
                records=win_records[k]))

        controller.decide(0)
        t, cur = 0.0, 0
        while t < duration_ms:
            while cur < int(t / W) and cur < n_windows:
                flush_overdue((cur + 1) * W)
                close_window(cur)
                if cur + 1 < n_windows:
                    controller.decide(cur + 1)
                cur += 1
            release_due(t)
            if not ready:
                future = [next_release[tid] for tid, tk in tasks.items()
                          if tk.qos_level > 0 and next_release[tid] < duration_ms]
                nb = (cur + 1) * W
                cands = future + ([nb] if nb < duration_ms else [])
                if not cands:
                    break
                t = min(cands)
                continue

            job = heapq.heappop(ready)
            if self.drop_expired and job.deadline_ms <= t:
                record(JobRecord(job.task_id, job.job_id, job.level, job.release_ms,
                                 job.deadline_ms, None, None, 0.0, "dropped"))
                continue
            frame = tasks[job.task_id].source.read()
            result = self.pipeline.process(frame, job.level)
            start, finish = t, t + result.exec_ms
            t = finish
            trace.busy_ms += result.exec_ms
            i0, i1 = widx(start), widx(finish)
            for i in range(i0, i1 + 1):
                ov = min(finish, (i + 1) * W) - max(start, i * W)
                if ov > 0:
                    win_busy[i] += ov
            record(JobRecord(job.task_id, job.job_id, job.level, job.release_ms,
                             job.deadline_ms, start, finish, result.exec_ms,
                             "late" if finish > job.deadline_ms else "ontime"))

        while cur < n_windows:  # drain: resolve and report the final windows
            flush_overdue((cur + 1) * W)
            close_window(cur)
            cur += 1
        return trace
