"""Safety risk estimator: maps a candidate QoS decision to a risk score (LOWER = safer),
which the adaptive threshold accepts iff ``r <= λ`` (see ``patrol.controllers.safety``).

This mirrors the **analytical** risk estimator of the main ARGoS core
(``patrol.controllers.rl._rl_analytical_util``): the risk is the aggregate EDF utilization

    r(d) = Σ_i C_i(d_i) / T_i   over non-dropped streams      (Eq. 4; analytical threshold 1).

ARGoS's guarantee does not depend on this being exact -- an inaccurate (over-)estimate only
makes SafetyGuard more conservative. We keep it profile-free: ``C_i(q)`` is the *runtime*
running-mean execution time ``C̄_i(q)``, not a pre-measured profile. A learned MLP classifier
(``r = miss_prob``) can be dropped in behind the same ``risk`` interface later (Case 3).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class RiskEstimator(ABC):
    @abstractmethod
    def risk(self, decision: tuple[int, ...]) -> float:
        """Risk score ``r`` of applying ``decision`` (LOWER = safer, lower-bounded by r_min=0)."""


class UtilRisk(RiskEstimator):
    """Analytical EDF-utilization risk, matching the main ARGoS analytical estimator:
    ``r(d) = Σ_i C̄_i(d_i) / T_i`` over non-dropped streams; accept iff ``r <= λ`` (λ = 1 is
    the EDF schedulability bound). Unobserved ``(stream, level)`` costs fall back to a
    conservative per-stream estimate so untried configs are not assumed cheap.
    """

    def __init__(self, periods, tracker, *, default_ms: float = 25.0):
        self.periods = periods          # index -> period_ms
        self.tracker = tracker          # ExecMeanTracker
        self.default_ms = default_ms

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

    def risk(self, decision: tuple[int, ...]) -> float:
        # Risk score = aggregate EDF utilization (lower = safer), identical in form to
        # patrol.controllers.rl._rl_analytical_util. No ramp/normalization: the raw utilization
        # is the score, thresholded against λ (λ = 1 == the EDF bound).
        return self.est_util(decision)
