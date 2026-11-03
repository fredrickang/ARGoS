"""Safety classifier: maps a candidate QoS decision to a deadline-miss probability, whose
complement is the safety score the risk controller thresholds.

ARGoS's guarantee does not depend on this being accurate (an inaccurate score only makes
SafetyGuard more conservative). We therefore start with a profile-free analytic scorer -- an
option the paper explicitly sanctions ("a safety score can be derived from the aggregate
utilization estimated using execution-time profiles"). Here the "profile" is the *runtime*
running-mean execution time C̄_i(q), so the scorer stays profile-free. A learned MLP
classifier can be dropped in behind the same `miss_prob` interface later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def miss_prob(self, decision: tuple[int, ...]) -> float:
        """P(a deadline miss occurs) if `decision` is applied, in [0, 1]."""


class UtilClassifier(Classifier):
    """Estimated-utilization scorer. Û(d) = Σ_i C̄_i(d_i) / T_i over non-dropped streams;
    miss probability ramps from 0 (at u_lo) to 1 (at u_hi). Unobserved (stream, level) costs
    fall back to a conservative per-stream estimate so untried configs are not assumed cheap.
    """

    def __init__(self, periods, tracker, *, default_ms: float = 25.0,
                 u_lo: float = 0.80, u_hi: float = 1.10):
        self.periods = periods          # index -> period_ms
        self.tracker = tracker          # ExecMeanTracker
        self.default_ms = default_ms
        self.u_lo = u_lo
        self.u_hi = u_hi

    def est_util(self, decision: tuple[int, ...]) -> float:
        u = 0.0
        for i, q in enumerate(decision):
            if q <= 0:
                continue
            c = self.tracker.mean(i, q)
            if c is None:  # untried config: assume the most costly seen for this stream
                c = self.tracker.max_seen(i) or self.default_ms
            u += c / self.periods[i]
        return u

    def miss_prob(self, decision: tuple[int, ...]) -> float:
        u = self.est_util(decision)
        if u <= self.u_lo:
            return 0.0
        if u >= self.u_hi:
            return 1.0
        return (u - self.u_lo) / (self.u_hi - self.u_lo)
