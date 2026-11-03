"""Frame sources — where a task's images come from.

Each task has a source that yields frames at the task's native capture resolution. The
pipeline downscales to the chosen QoS resolution. A recorded-video / RTSP source can be
added later behind the same `read()` interface; the synthetic source keeps the scheduler
runnable with no external data or codecs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class FrameSource(ABC):
    """Yields (C, H, W) float frames on the target device."""

    @abstractmethod
    def read(self) -> torch.Tensor:
        ...


class SyntheticSource(FrameSource):
    """Random frames at a fixed native resolution. Real compute downstream; the pixels are
    noise (the prototype's timing behaviour does not depend on frame content)."""

    def __init__(self, native_hw: tuple[int, int] = (1080, 1920), device: str | None = None,
                 seed: int | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        # preallocate one frame and reuse it; frame *content* is irrelevant to scheduling
        self._frame = torch.rand(3, *native_hw, generator=gen).to(self.device)

    def read(self) -> torch.Tensor:
        return self._frame
