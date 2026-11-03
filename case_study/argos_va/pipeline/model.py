"""Shared image-processing pipeline.

One CNN, shared by every task. The QoS knob is the *input resolution*: the same model
runs on a frame downscaled to the resolution chosen for the task's current QoS level, so
higher QoS => larger input => more compute. Execution cost is measured, never assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision import models


# QoS ladder. Level 0 = drop the frame (no job released). Levels 1..L-1 are (H, W)
# input resolutions of increasing cost. Measured fcn_resnet50 cost on an RTX 3090:
#   180x320 ~5ms | 270x480 ~7ms | 360x640 ~11ms | 540x960 ~25ms | 720x1280 ~43ms
DEFAULT_LADDER: list[tuple[int, int] | None] = [
    None,          # 0: drop
    (180, 320),    # 1
    (270, 480),    # 2
    (360, 640),    # 3
    (540, 960),    # 4
    (720, 1280),   # 5
]


@dataclass
class InferResult:
    level: int
    hw: tuple[int, int] | None
    exec_ms: float
    label: object = None   # the model output (unused by the scheduler; kept for realism)


class ImagePipeline:
    """The shared perception model. `process(frame, level)` resizes the frame to the QoS
    resolution and runs the model, returning the real (GPU-synchronized) execution time."""

    def __init__(self, arch: str = "fcn_resnet50", device: str | None = None,
                 ladder: list[tuple[int, int] | None] | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ladder = ladder or DEFAULT_LADDER
        # random-init weights: we care about compute cost, not detection accuracy, and this
        # avoids any network download. Swap in pretrained weights for a functional server.
        if arch.startswith("fcn") or arch.startswith("deeplab"):
            self.model = getattr(models.segmentation, arch)(
                weights=None, weights_backbone=None)
        else:
            self.model = getattr(models, arch)(weights=None)
        self.model = self.model.to(self.device).eval()
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

    def num_levels(self) -> int:
        return len(self.ladder)

    def resolution(self, level: int) -> tuple[int, int] | None:
        return self.ladder[level]

    @torch.no_grad()
    def warmup(self, reps: int = 3) -> None:
        """Run every non-drop resolution a few times so cudnn autotune / lazy init does not
        pollute the first real measurement."""
        for level, hw in enumerate(self.ladder):
            if hw is None:
                continue
            frame = torch.rand(3, *hw, device=self.device)
            for _ in range(reps):
                self.process(frame, level)

    @torch.no_grad()
    def process(self, frame: torch.Tensor, level: int) -> InferResult:
        """frame: (C, H, W) float tensor on `self.device`, at its native resolution.
        Returns an InferResult whose exec_ms is the measured resize+forward time."""
        hw = self.ladder[level]
        if hw is None:  # drop
            return InferResult(level=level, hw=None, exec_ms=0.0, label=None)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = F.interpolate(frame.unsqueeze(0), size=hw, mode="bilinear", align_corners=False)
        y = self.model(x)
        out = y["out"] if isinstance(y, dict) else y
        label = out.detach()
        if self.device == "cuda":
            torch.cuda.synchronize()
        exec_ms = (time.perf_counter() - t0) * 1000.0
        return InferResult(level=level, hw=hw, exec_ms=exec_ms, label=label)
