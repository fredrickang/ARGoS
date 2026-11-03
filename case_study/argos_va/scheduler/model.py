"""Task and job model for the video-analytics server.

A task is a camera stream: it produces one frame-job every `period_ms`, with an implicit
deadline equal to the period (the frame must be handled before the next one is due). The
task's current QoS level fixes the resolution at which its frames are processed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..streams.source import FrameSource


@dataclass
class Task:
    task_id: int
    period_ms: float
    source: FrameSource
    qos_level: int = 1          # current QoS (resolution index into the pipeline ladder)
    offset_ms: float = 0.0      # release phase of the first job
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"stream{self.task_id}"


@dataclass(order=True)
class Job:
    """A single frame to process. Ordered by EDF key = (deadline, release, task_id)."""
    deadline_ms: float
    release_ms: float
    task_id: int
    job_id: int = field(compare=False)
    level: int = field(compare=False, default=1)


@dataclass
class JobRecord:
    task_id: int
    job_id: int
    level: int
    release_ms: float
    deadline_ms: float
    start_ms: float | None      # None if dropped without executing
    finish_ms: float | None
    exec_ms: float
    status: str                 # 'ontime' | 'late' | 'dropped'


@dataclass
class WindowObs:
    """What the controller sees at the end of a window: the runtime state ARGoS decisions
    are based on. `loss` is the deadline-miss loss ℓ_miss,k = misses / N_job (Eq. 3)."""
    index: int
    utilization: float
    loss: float
    released: int
    misses: int
    n_job: int
    records: list[JobRecord] = field(default_factory=list)
