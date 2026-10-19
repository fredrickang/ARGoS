"""Safety guards for the RL QoS controller.

The RL agent proposes a QoS decision each window; a *safety guard* validates that
decision before it is applied and, if it is rejected, drives the grouped-reduction
fallback (see ``RlControllerMixin._rl_safe_fallback``). Guards share one interface so
the controller flow is mode-agnostic:

    accept iff  guard.is_accepted(score)              # score = safety score f (higher = safer)
    per-window  guard.update_on_outcome(loss=..., score=..., x_state=...)

All guards operate on a **safety score** ``f`` (higher = safer, upper-bounded by ``fmax``),
matching the paper's SafetyGuard (§V-A). The controller supplies ``f`` from whichever scorer
is active — the learned classifier ``f = 1 - miss_prob`` (Case 3, ``fmax = 1``) or the
analytical EDF utilization margin ``f = 1 - Σ Cᵢ(qᵢ)/Tᵢ`` (Cases 1 & 2, ``fmax = 1``). The
guard itself is scorer-agnostic.

Modes (``controller.safety.mode``):

* ``rcrl``       — Adaptive Threshold Control (Algorithm 1). A single online threshold ``λ``
                   with a distribution-free long-run guarantee (Theorem 1):
                   accept iff safety score ``f ≥ λ``; update ``λ ← λ + γ (ℓ - α)`` per window.
                   Works with any scorer (learned or analytical).
* ``conformal``  — legacy split-conformal quantile guard (kept as a baseline).
* ``rt_analytic``— standalone analytic EDF util-≤-1 hard cap (``rt_shielding``); legacy
                   baseline, never combined with ``rcrl``. (For the paper's Cases 1 & 2 use
                   ``rcrl`` with ``score: analytical`` instead.)
* ``none``       — no shielding (accept everything, never fall back).
"""

import logging


logger = logging.getLogger(__name__)


class SafetyGuard:
    """Base guard: no shielding (``none`` mode)."""

    mode = "none"
    fallback_enabled = False
    fallback_start_window = 0
    fmax = 1.0

    def is_accepted(self, score):
        """Accept the decision given its safety score ``f`` (higher = safer)."""
        return True

    def needs_fallback(self, score):
        return not self.is_accepted(score)

    def update_on_outcome(self, *, loss, score=None, x_state=None):
        """Called once per window after the applied decision's outcome is observed."""
        return None

    def on_classifier_refined(self, predict_fn, extra_miss_samples=None):
        """Hook invoked after the classifier is fine-tuned (REFINE)."""
        return None

    def threshold_value(self):
        """Primary decision threshold at the current window (telemetry). None if N/A."""
        return None

    def theorem_bound(self, horizon):
        """Upper bound on the realized long-run miss rate over ``horizon`` windows."""
        return None

    def status(self):
        return {}


class AdaptiveRiskGuard(SafetyGuard):
    """Adaptive threshold control (Algorithm 1).

    Accept iff ``f ≥ λ`` for safety score ``f`` (higher = safer, bounded by ``fmax``);
    after each window update ``λ ← λ + γ (ℓ - α)`` where ``ℓ`` is the realized deadline-miss
    loss and ``α`` the target miss level. ``λ`` is deliberately **not** clipped above — the
    Theorem-1 proof only needs ``λ₁ ∈ [0, fmax]`` and the natural bound
    ``λ ≤ fmax + γ·max(α, 1-α)`` (Lemma 2); a hard clamp would break the guarantee.

    The scorer is external: ``f = 1 - miss_prob`` for the learned classifier (Case 3) or the
    analytical EDF utilization margin ``f = 1 - Σ Cᵢ/Tᵢ`` (Cases 1 & 2). For both,
    ``fmax = 1``. Case 1 (valid profiles, hard guarantee) uses ``α = 0`` and ``λ₁ = η = 0``;
    Lemma 1 then keeps ``λ ≡ 0``.
    """

    mode = "rcrl"

    def __init__(self, *, gamma=0.05, alpha=0.1, lambda_init=0.5, lambda_min=0.0,
                 fmax=1.0, fallback_enabled=True, fallback_start_window=0):
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.fmax = float(fmax)
        self.lambda_init = float(lambda_init)
        self.lam = float(lambda_init)
        # Optional lower clamp on λ. For any λ ≤ 0 the guard accepts everything
        # (safety scores ∈ [0,1]), so λ<0 is pure accumulated slack that only delays
        # re-engagement after a dangerous regime shift. Clamping at 0 removes that lag
        # while preserving Theorem 1 (the clamp corrections are non-negative and only
        # tighten the long-run bound). Set to None for the exact unclamped textbook rule.
        self.lambda_min = None if lambda_min is None else float(lambda_min)
        self.fallback_enabled = bool(fallback_enabled)
        self.fallback_start_window = int(fallback_start_window)

    @property
    def B(self):
        return max(self.alpha, 1.0 - self.alpha)

    def is_accepted(self, score):
        return float(score) >= self.lam

    def update_on_outcome(self, *, loss, score=None, x_state=None):
        self.lam += self.gamma * (float(loss) - self.alpha)
        if self.lambda_min is not None and self.lam < self.lambda_min:
            self.lam = self.lambda_min
        return self.lam

    def threshold_value(self):
        return self.lam

    def theorem_bound(self, horizon):
        # (1/T) Σ ℓ_t ≤ α + (fmax + γB) / (γT)   (Theorem 1, Eq. 14).
        if not horizon or horizon <= 0:
            return None
        return self.alpha + (self.fmax + self.gamma * self.B) / (self.gamma * horizon)

    def status(self):
        return {"lambda": round(self.lam, 4), "gamma": self.gamma,
                "alpha": self.alpha, "fmax": self.fmax}


class ConformalGuard(SafetyGuard):
    """Legacy split-conformal quantile guard (baseline).

    Accept iff ``miss_prob < threshold`` **and** ``(1 - miss_prob) > q_hat``, where
    ``q_hat`` is a rolling quantile of the safe scores observed on deadline-miss windows.
    Faithful port of the original ``_rl_*_safe_calibration`` logic.
    """

    mode = "conformal"

    def __init__(self, *, threshold_fn, delta=0.1, window=200, min_samples=20,
                 fallback_enabled=True, fallback_start_window=0):
        self._threshold_fn = threshold_fn
        self.delta = float(delta)
        self.window = int(window) if window else 0
        self.min_samples = int(min_samples)
        self.fallback_enabled = bool(fallback_enabled)
        self.fallback_start_window = int(fallback_start_window)
        self.scores = []
        self.calib_samples = []
        self.qhat = None

    def _threshold(self):
        return float(self._threshold_fn()) if self._threshold_fn else 0.5

    def is_accepted(self, score):
        # score = f = 1 - miss_prob, so miss_prob = 1 - score.
        if (1.0 - float(score)) >= self._threshold():
            return False
        if self.qhat is None:
            return True
        return float(score) > self.qhat

    def _recompute_qhat(self):
        if not self.scores:
            self.qhat = None
            return
        scores = sorted(self.scores)
        n = len(scores)
        k = int((n + 1) * (1.0 - self.delta) + 0.999999)
        k = max(1, min(k, n))
        self.qhat = scores[k - 1]

    def update_on_outcome(self, *, loss, score=None, x_state=None):
        if score is None:
            return None
        if loss == 1:
            if x_state is not None:
                self.calib_samples.append({"x_state": x_state, "y_miss": 1})
            self.scores.append(float(score))
            if self.window:
                self.scores[:] = self.scores[-self.window:]
            self._recompute_qhat()
        return self.qhat

    def on_classifier_refined(self, predict_fn, extra_miss_samples=None):
        # 1) Rebuild from existing calibration samples through the refined classifier.
        xs = [s["x_state"] for s in self.calib_samples if s.get("x_state") is not None]
        probs = predict_fn(xs) if xs else []
        rebuilt, rebuilt_scores = [], []
        for x_state, miss_prob in zip(xs, probs):
            rebuilt.append({"x_state": x_state, "y_miss": 1})
            rebuilt_scores.append(float(1.0 - miss_prob))

        # 2) If too few, extend from the full miss-window history.
        if len(rebuilt) < self.min_samples and extra_miss_samples:
            cand_x = [s["x_state"] for s in extra_miss_samples if s.get("x_state") is not None]
            cand_probs = predict_fn(cand_x) if cand_x else []
            for x_state, miss_prob in zip(cand_x, cand_probs):
                rebuilt.append({"x_state": x_state, "y_miss": 1})
                rebuilt_scores.append(float(1.0 - miss_prob))

        self.calib_samples = rebuilt
        self.scores = rebuilt_scores
        self._recompute_qhat()
        return self.qhat

    def threshold_value(self):
        return self.qhat

    def status(self):
        return {"q_hat": round(self.qhat, 4) if self.qhat is not None else None,
                "delta": self.delta, "n_scores": len(self.scores)}


class RtAnalyticGuard(SafetyGuard):
    """Analytic EDF util-≤-1 baseline (``rt_shielding``).

    The actual adjustment lives in ``RlControllerMixin.rt_shielding`` (it needs the
    scheduler's estimated exec tables). This guard only tags the mode so the controller
    routes to the analytic path instead of the learned classifier + fallback. It is a
    profile-based comparison baseline and is never combined with ``rcrl``.
    """

    mode = "rt_analytic"
    fallback_enabled = False


def build_safety_guard(ctrl, *, threshold_fn=None):
    """Build a safety guard from the ``controller`` config dict.

    Reads the nested ``controller.safety`` block when present. For configs that predate
    that block, falls back to the legacy flat keys so existing runs are unchanged.
    """
    safety = dict(ctrl.get("safety") or {})
    mode = safety.get("mode")

    if mode is None:
        # Back-compat shim: reconstruct the pre-``safety`` behaviour from flat keys.
        legacy_fallback = bool(ctrl.get("enable_profile_fallback", False))
        mode = "conformal" if legacy_fallback else "none"
        safety = {
            "conformal_delta": ctrl.get("profile_safe_delta", 0.1),
            "conformal_window": ctrl.get("profile_safe_window", 200),
            "conformal_min_samples": ctrl.get("profile_safe_min", 20),
            "fallback_enabled": legacy_fallback,
            "fallback_start_window": ctrl.get("profile_fallback_start_window", 500),
        }

    mode = str(mode).lower()

    if mode == "rcrl":
        return AdaptiveRiskGuard(
            gamma=safety.get("lambda_step_gamma", 0.05),
            alpha=safety.get("target_miss_alpha", 0.1),
            fmax=safety.get("fmax", 1.0),
            lambda_init=safety.get("lambda_init", 0.5),
            lambda_min=safety.get("lambda_min", 0.0),
            fallback_enabled=safety.get("fallback_enabled", True),
            fallback_start_window=safety.get("fallback_start_window", 0),
        )
    if mode == "conformal":
        return ConformalGuard(
            threshold_fn=threshold_fn,
            delta=safety.get("conformal_delta", 0.1),
            window=safety.get("conformal_window", 200),
            min_samples=safety.get("conformal_min_samples", 20),
            fallback_enabled=safety.get("fallback_enabled", True),
            fallback_start_window=safety.get("fallback_start_window", 0),
        )
    if mode == "rt_analytic":
        return RtAnalyticGuard()
    if mode == "none":
        return SafetyGuard()

    raise ValueError(f"Unknown controller.safety.mode: {mode!r}")
