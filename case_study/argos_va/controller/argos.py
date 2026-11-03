"""ARGoS controller for the video-analytics server.

Once per window it (1) builds the runtime state from observations only, (2) has the RL agent
propose a per-stream QoS decision, (3) validates it with SafetyGuard -- classifier score vs.
the adaptive risk threshold, with grouped-reduction fallback -- and applies the validated
decision, then (4) after the window, updates the threshold from the observed miss loss and
trains the agent.

The RL agent (`patrol.rl.dqn.DQNAgent`) and the adaptive risk controller
(`patrol.controllers.safety.AdaptiveRiskGuard`) are reused unchanged from the ARGoS core; this
module supplies the video-analytics state/reward/validation glue and the window loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Reuse the ARGoS core (RL agent + adaptive risk controller) from the *refactored* project:
# the importable package `patrol` lives inside the sibling PATROL_refactored/ tree. Prepend
# that path so `import patrol` resolves there, and assert the source to rule out any other
# `patrol` on sys.path (the legacy /root/PATROL tree is flat and has no `patrol/` package).
_PATROL_REFACTORED = Path(__file__).resolve().parents[3] / "PATROL_refactored"
if not _PATROL_REFACTORED.is_dir():
    raise RuntimeError(f"PATROL_refactored not found at {_PATROL_REFACTORED}")
if str(_PATROL_REFACTORED) not in sys.path:
    sys.path.insert(0, str(_PATROL_REFACTORED))
from patrol.rl.dqn import DQNAgent                       # noqa: E402
from patrol.controllers.safety import AdaptiveRiskGuard  # noqa: E402

import patrol as _patrol  # noqa: E402
assert Path(_patrol.__file__).resolve().parent.parent == _PATROL_REFACTORED, (
    f"`patrol` resolved to {_patrol.__file__}, expected it under {_PATROL_REFACTORED}")

from ..scheduler.model import Task, WindowObs
from .classifier import UtilClassifier
from .fallback import grouped_fallback


class ExecMeanTracker:
    """Running mean execution time C̄_i(q) per (stream index, QoS level), from observations."""

    def __init__(self):
        self._sum: dict[tuple[int, int], float] = {}
        self._cnt: dict[tuple[int, int], int] = {}

    def update(self, idx: int, level: int, exec_ms: float) -> None:
        k = (idx, level)
        self._sum[k] = self._sum.get(k, 0.0) + exec_ms
        self._cnt[k] = self._cnt.get(k, 0) + 1

    def mean(self, idx: int, level: int) -> float | None:
        k = (idx, level)
        return self._sum[k] / self._cnt[k] if self._cnt.get(k) else None

    def max_seen(self, idx: int) -> float | None:
        seen = [self._sum[k] / self._cnt[k] for k in self._cnt if k[0] == idx]
        return max(seen) if seen else None


class ArgosController:
    def __init__(self, tasks: list[Task], num_levels: int, *,
                 alpha: float = 0.1, gamma: float = 0.05, lambda_init: float = 0.5,
                 u_ref: float = 0.9, init_level: int = 1,
                 safety: bool = True, seed: int = 0,
                 save_path: str = "results/models/argos_dqn.pth"):
        self.tasks = sorted(tasks, key=lambda t: t.task_id)
        self.N = len(self.tasks)
        self.L = num_levels
        self.periods = [t.period_ms for t in self.tasks]
        self.T_max = max(self.periods)
        self.u_ref = u_ref
        self.safety_on = safety

        for t in self.tasks:               # start from a known level
            t.qos_level = init_level
        self.last_decision = tuple(init_level for _ in self.tasks)

        self.tracker = ExecMeanTracker()
        self.classifier = UtilClassifier(self.periods, self.tracker)
        self.guard = AdaptiveRiskGuard(gamma=gamma, alpha=alpha, lambda_init=lambda_init,
                                       fallback_enabled=safety)

        self.agent = DQNAgent(state_size=3 + 3 * self.N, action_dim=self.N,
                              action_bins=self.L, save_path=save_path)

        self.u_prev = 0.0
        self.m_prev = 0.0
        self._prev_state = None
        self._prev_action = None

        # telemetry
        self.loss_history: list[float] = []
        self.lambda_history: list[float] = []
        self.util_history: list[float] = []
        self.decisions: list[tuple[int, ...]] = []
        self.reward_history: list[float] = []
        self.fallback_windows: list[int] = []

    # ---- state ----
    def _state(self) -> np.ndarray:
        f = [float(self.N), self.u_prev, self.m_prev]
        for i, t in enumerate(self.tasks):
            q = self.last_decision[i]
            c = self.tracker.mean(i, q) or 0.0
            f += [q / max(self.L - 1, 1), t.period_ms / self.T_max, c / t.period_ms]
        return np.asarray(f, dtype=np.float32)

    # ---- validation ----
    def _accepted(self, decision: tuple[int, ...]) -> bool:
        if all(q == 0 for q in decision):    # d_safe: f = +inf, accepted for every threshold
            return True
        if not self.safety_on:
            return True
        return self.guard.is_accepted(self.classifier.miss_prob(decision))

    # ---- per-window hooks used by EdfScheduler.run_windowed ----
    def decide(self, k: int) -> tuple[int, ...]:
        state = self._state()
        action, _ = self.agent.get_action(state, self.agent.random_explore_policy, budget=None)
        decision = tuple(int(a) for a in np.asarray(action).reshape(-1)[:self.N])

        if not self._accepted(decision):
            fb = grouped_fallback(decision, self._accepted)
            self.fallback_windows.append(k)
            decision = fb

        for i, t in enumerate(self.tasks):
            t.qos_level = decision[i]
        self._prev_state = state
        self._prev_action = np.asarray(decision, dtype=np.int64)
        self.last_decision = decision
        self.decisions.append(decision)
        return decision

    def observe(self, obs: WindowObs) -> None:
        for r in obs.records:               # update runtime execution-time history
            if r.start_ms is not None and r.exec_ms > 0:
                self.tracker.update(r.task_id, r.level, r.exec_ms)

        if self.safety_on:                  # adaptive risk control: λ ← λ + γ(ℓ − α)
            self.guard.update_on_outcome(loss=obs.loss)

        self.u_prev, self.m_prev = obs.utilization, obs.loss

        if self._prev_state is not None:    # train the agent on the realized outcome
            value = float(np.mean(self._prev_action)) / max(self.L - 1, 1)
            reward, _ = self.agent.compute_reward_v2(
                obs.utilization, value, U_ref=self.u_ref, dmr=obs.loss)
            self.agent.store_transition(self._prev_state, self._prev_action, reward,
                                        self._state(), False)
            self.agent.train_model()
            self.reward_history.append(reward)

        self.loss_history.append(obs.loss)
        self.util_history.append(obs.utilization)
        self.lambda_history.append(self.guard.threshold_value() or 0.0)

    # ---- reporting ----
    def theorem_bound(self, horizon: int) -> float | None:
        return self.guard.theorem_bound(horizon)

    def running_avg_loss(self) -> list[float]:
        out, s = [], 0.0
        for i, l in enumerate(self.loss_history, 1):
            s += l
            out.append(s / i)
        return out
