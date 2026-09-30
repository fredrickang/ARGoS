# Implementation Plan — RCRL (Risk-Controlled RL Scheduling)

Migrating PATROL's safety shield from the current **conformal-quantile** design to the
**RCRL / Adaptive Risk Control (ARC)** framework described in *"Reinforcement Learning for
Real-Time Scheduling with Statistical Guarantees"* (RTSS'26 draft).

This document is the engineering plan. It maps every paper concept to a concrete code change,
flags the decisions to resolve, and sequences the work into phases.

---

## 1. What actually changes (and what doesn't)

The paper's **fallback** (Algorithm 2) and **RL agent** (Section VIII) are essentially the code
you already have. The redesign is confined to the **validation / risk-control layer**:

- **Remove:** the conformal calibration machinery — `rl_safe_qhat`, `rl_safe_scores`,
  `rl_safe_calib_samples`, `_rl_update_safe_calibration`, `_rl_rebuild_safe_calibration`, and the
  `profile_safe_*` config knobs.
- **Add:** a single online threshold **λ** with the update rule `λ_{t+1} = λ_t + γ(ℓ_t − α)`
  (Algorithm 1, line 13), a **safety-score** acceptance test `f(z,d) ≥ λ`, and the extended
  score function `f(z, d_safe) = 1` that guarantees a non-empty confidence set.
- **Keep (adapt only lightly):** the grouped-reduction fallback, the classifier MLP, online
  classifier REFINE on accuracy drop, and the branching DQN + reward.

Net effect: **less code**, plus a provable long-run guarantee (Theorem 1:
`limsup (1/T)Σℓ_t ≤ α`).

### Concept → code map

| Paper | Symbol | Current code | Action |
|-------|--------|--------------|--------|
| Safety score `f(z,d)` ∈ [0,1] | Eq. 4, 11 | MLP outputs **miss** prob | compute `f = 1 − miss_prob`; special-case `f(d_safe)=1` |
| Threshold `λ`, init 0.5 | Alg. 1 L1 | static `threshold=0.1224` + `q_hat` | single scalar `self.rl_lambda` |
| Accept iff `f ≥ λ` | Alg. 1 L6 | `_rl_is_conformal_safe` | `_rl_is_accepted(miss_prob)` = `(1−miss_prob) ≥ λ` |
| Update `λ ← λ+γ(ℓ−α)` | Alg. 1 L13 | `_rl_update_safe_calibration` | new `_rl_update_lambda(loss)` |
| Fallback (grouped reduction) | Alg. 2 | `_rl_build_grouped_fallback_candidates` + `_rl_safe_fallback` | keep; validate by λ; always admit `d_safe` |
| `d_safe` = all-zero QoS | §VI-E, Assum. 2 | implicit (`best_idx = last`) | make explicit + always-accepted |
| Classifier REFINE | Alg. 1 L15 | `_rl_maybe_finetune` | keep (now quality-only, not safety-critical) |
| RL update | Alg. 1 L14 | `store_transition`/`train_model` | keep; **fix the fallback-skip logic** (§5.1) |
| Step size γ=0.05, target α=0.1 | Assum. 1 | `profile_safe_delta` etc. | new `lambda_step_gamma`, `target_miss_alpha` |

---

## Implementation status (2026-08-10)

Phases 0–2 and the Phase 3 **logging** are implemented and validated:

- **`patrol/controllers/safety.py`** — `AdaptiveRiskGuard` (rcrl), `ConformalGuard` (baseline),
  `RtAnalyticGuard` (baseline), `SafetyGuard` (none) + `build_safety_guard` factory with a
  legacy back-compat shim.
- **`rl.py`** — guard wired into `_rl_prepare_pending` / `_rl_safe_fallback` / `_rl_qos_mgmt`;
  §5.1 skip removed; §5.3 `rt_analytic` kept off the RCRL path.
- **Telemetry** — per-window λ_t + running miss ratio logged; summary JSON `rcrl` block with
  `realized_miss_rate`, `theorem_bound = α+(1+γB)/(γT)`, `guarantee_satisfied`, λ trajectory.
- **`scripts/run_safety_ablation.py`** — sweeps `safety.mode × dynamic_mode` → comparison CSV.
- Validated: realized miss 0.0655 ≤ α=0.1 under bound 0.1105; λ_final matched Algorithm 1 exactly.

**λ lower-clamp (decided):** `AdaptiveRiskGuard` clamps λ at `lambda_min` (default **0.0**;
`null` = exact unclamped rule). For any λ ≤ 0 the guard already accepts everything, so the clamp
changes no steady-state decision and preserves Theorem 1 (clamp corrections `c_t ≥ 0` only tighten
the bound: `γΣ(ℓ−α) = (λ_{T+1}−λ_1) − Σc_t ≤ 1+γB`). Its only effect is faster re-engagement after a
dangerous regime shift (avoids λ sitting deep-negative). Paper note: present the clamped rule as the
algorithm with the one-line `Σc_t ≥ 0` addendum.

**Remaining:** longer-horizon + `dynamic_mode 1/2/3` evaluation; optional trajectory plotting;
Phase 4 reward/state alignment (tentative — deferred per author).

---

## 2. New module: extract the safety guard (`patrol/controllers/safety.py`)

Mirroring the paper restructure ("Safety guard를 section으로 빼고"), pull the safety logic out of
`RlControllerMixin` into its own strategy object. This also lets us keep the legacy guard behind a
flag for the paper's evaluation/ablation.

```python
# patrol/controllers/safety.py
class AdaptiveRiskGuard:
    """RCRL adaptive risk control (Algorithm 1)."""
    def __init__(self, *, gamma=0.05, alpha=0.1, lambda_init=0.5):
        self.gamma, self.alpha, self.lam = gamma, alpha, lambda_init

    def is_accepted(self, miss_prob):        # f(z,d) >= λ,  f = 1 - miss_prob
        return (1.0 - miss_prob) >= self.lam

    def update(self, loss):                  # λ_{t+1} = λ_t + γ(ℓ_t − α)   (unclipped)
        self.lam += self.gamma * (float(loss) - self.alpha)
        return self.lam

class ConformalGuard:                        # legacy — wraps existing q_hat logic
    ...
```

- `is_accepted` is the membership test for `C_λ(z)`.
- `update` is applied **once per window**, after the applied decision's outcome `ℓ_t` is observed.
- **Do not clip λ.** The Theorem-1 telescoping proof only needs `λ_1 ≥ 0` and the natural upper
  bound `λ_t ≤ 1+γB` (Lemma 1); a hard clamp would break the guarantee. λ may go negative during
  long safe stretches — that just means "accept everything", which is correct.

`RlControllerMixin` holds `self.safety_guard` (built from config) and delegates to it.

---

## 3. Rewritten control flow (`_rl_qos_mgmt`)

Re-expressing Algorithm 1 in the existing per-window structure
([rl.py:446](../patrol/controllers/rl.py#L446)):

```
Finalize previous window t-1  (outcome now observable):
    ℓ_{t-1} = 1[dmr_{t-1} > 0]                         # loss of the APPLIED decision
    guard.update(ℓ_{t-1})                              # Alg.1 L13  ← NEW, replaces calibration
    update classifier accuracy metrics                 # for REFINE
    maybe REFINE classifier                            # Alg.1 L15 (unchanged)

Decide window t:
    z_t = get_env()
    d_t = agent.get_action(z_t)                        # Alg.1 L4
    apply d_t to tasks
    miss_prob = classifier(z_t, d_t)
    if not guard.is_accepted(miss_prob):               # Alg.1 L6 (¬ accept)
        d_t = fallback(z_t)                            # Alg.1 L9 / Alg.2
        apply d_t to tasks
    record pending{win_id=t, action=d_t, x_state, miss_prob}

    reward = compute_reward(...)
    agent.store_transition(z_{t-1}, applied_action_{t-1}, r_{t-1}, z_t)   # Alg.1 L14
    agent.train_model()
```

Key point: the **λ update and the loss `ℓ` use the actually-applied (post-fallback) decision**, and
it fires **every window** (something is always applied). The current `pending["action"]` is already
overwritten to the fallback action in `_rl_prepare_pending`, so this is satisfied — but see §5.1.

---

## 4. Fallback under λ (`_rl_safe_fallback`, Algorithm 2)

Minimal edits to the existing method:

1. **Candidate generation** — unchanged (`_rl_build_grouped_fallback_candidates` already does the
   grouped uniform decrement, ≤ M candidates, ending in all-zero).
2. **Batched inference** — unchanged (`_rl_predict_miss_probs_batch`).
3. **Validation predicate** — replace `_rl_is_conformal_safe(cand_prob)` with
   `guard.is_accepted(cand_prob)` i.e. `(1 − cand_prob) ≥ λ`.
4. **`d_safe` always admitted** — the all-zero candidate implements `f=1`; force-accept it so the
   validated set `S` is never empty (Eq. 11 → acceptance prob = 1 → conditional risk = marginal
   risk, Eq. 10). Current `best_idx = len(candidates)-1` fallback already yields this; make it
   explicit and independent of λ.
5. **Selection** — unchanged: `argmax Σ q_i` over validated candidates.

---

## 5. Correctness items to resolve during implementation

### 5.1 Remove the fallback-skip in transition storage — **DECIDED: remove the skip**
Today ([rl.py:481-503](../patrol/controllers/rl.py#L481-L503)) `store_transition` is **skipped**
when a fallback fires, but the transition being stored is for the **previous** window
(`prev_state = z_{k-2}`, `prev_qos_list = a_{k-1}`, `reward = r_{k-1}`), while `fallback_applied`
reflects the **current** window k's pending flag — a one-window offset. The gate is decorrelated
from its own intent: it skips clean k-1 transitions when k happens to fall back, and stores
fallback k-1 transitions when k happens to be clean.

`prev_qos_list` is already the **applied** (post-fallback) action, which is exactly what
Algorithm 1 (L14) trains on. **Decision: delete the skip entirely** — always store the
applied-action transition every window (paper-consistent, and it removes code). Keep the existing
`dmr>0` double-store (miss up-weighting; correctly aligned to k-1). Collapse to:

```python
self.rl_agent.store_transition(
    state=prev_state, action=prev_qos_list, reward=reward,
    next_state=state, done=(self.sim.now() == self._next_env_change_time),
)
if self.window_log[self.w_id - 1]["dmr"] > 0:      # keep miss up-weighting
    self.rl_agent.store_transition(state=prev_state, action=prev_qos_list, reward=reward,
                                   next_state=state, done=(self.sim.now() == self._next_env_change_time))
self.rl_agent.train_model()
```

### 5.2 Classifier output semantics
Keep training the MLP on `y_miss` (no retraining needed); compute the safety score as
`f = 1 − sigmoid(logit)` in the guard. Document this inversion in one place so `λ` (a threshold on
*safety*) is never accidentally compared against a *miss* probability.

### 5.3 `rt_shielding` — **DECIDED: keep as a comparison baseline, never in the RCRL path**
The analytic util-≤-1 guard ([rl.py:536](../patrol/controllers/rl.py#L536)) relies on estimated
execution tables (`build_estimated_exec_tables`) — i.e., a **profile** — and an EDF util bound,
which is intentionally *not* what RCRL uses. It is retained as an **alternative safety mechanism
to compare against** in the evaluation (analytic profile-based guard vs. RCRL statistical guard).

**Decision:** do **not** delete it, but ensure it is **never invoked on the RCRL path**. It should
be reachable only as its own `controller.safety.mode` (e.g. `mode: rt_analytic`), selected
explicitly for baseline runs. Concretely: keep the method, do not wire it into `_rl_qos_mgmt`, and
make the `SafetyGuard` factory (§2) refuse to combine it with `mode: rcrl`. Grep before shipping to
confirm no accidental call from the RCRL flow.

> Related (not blocking): `rl_adaptation_on_env_change` → `hvdf_history`
> ([rl.py:585](../patrol/controllers/rl.py#L585)) also consumes `build_estimated_exec_tables`, so
> the RCRL controller's env-change **warm-start** currently seeds from an estimated profile. It's a
> heuristic reset, not a guarantee (does not affect Theorem 1), but it's a second profile touchpoint
> on the RL path. Leave as-is for now and footnote it; revisit if a reviewer presses the
> "fully profile-free" claim.

### 5.4 λ warm-up
Fallback should be active from `t=0` (λ₀=0.5 needs no calibration). Remove
`profile_fallback_start_window=500`. Optionally keep a tiny `rcrl_start_window: 0` knob for
experiments, default 0.

---

## 6. Config changes

Replace the `profile_safe_*` / fallback block with an explicit safety section
(`configs/run_sim.yaml`):

```yaml
controller:
  mode: rl
  ...
  safety:
    mode: rcrl                 # rcrl | conformal (legacy) | rt_analytic (baseline) | none
    classifier_model_path: ./patrol/models/checkpoints/model.pt
    target_miss_alpha: 0.1     # α   (Assumption 1)
    lambda_step_gamma: 0.05    # γ
    lambda_init: 0.5           # λ₀
    refine_on_accuracy_drop: true
    refine_acc_threshold: 0.8  # was profile_finetune_acc_thr
```

**Deprecate/remove:** `enable_profile_fallback`, `profile_fallback_start_window`,
`profile_safe_delta`, `profile_safe_window`, `profile_safe_min`. Keep the `profile_finetune_*` keys
(they drive REFINE) but nest them under `safety.refine_*` for clarity. Add a back-compat shim that
reads the old flat keys and warns.

---

## 7. Phased task list

**Phase 0 — scaffolding (no behavior change)**
- [ ] Add `patrol/controllers/safety.py` with `AdaptiveRiskGuard` + `ConformalGuard` (legacy wraps
      current logic verbatim).
- [ ] Wire `self.safety_guard` from `controller.safety.mode`; default `conformal` so existing runs
      are unchanged. Add config parsing + back-compat shim.

**Phase 1 — RCRL core**
- [ ] Implement `AdaptiveRiskGuard.is_accepted` / `update`.
- [ ] Add `_rl_update_lambda` call in the finalize step of `_rl_qos_mgmt`; log λ per window
      (reuse the `safe_calib` color channel).
- [ ] Point `_rl_safe_fallback` validation at `guard.is_accepted`; make `d_safe` always-accepted.
- [ ] Simplify `_rl_prepare_pending` to the single trigger `not guard.is_accepted(miss_prob)`.
- [ ] Delete `_rl_is_conformal_safe`, `_rl_update_safe_calibration`,
      `_rl_rebuild_safe_calibration`, and the `rl_safe_*` state (guarded so `ConformalGuard` keeps
      its own copy if we retain the legacy path).

**Phase 2 — bookkeeping + cleanup**
- [ ] §5.1: delete the fallback-skip; always store the applied-action transition (keep `dmr>0` double-store).
- [ ] §5.3: keep `rt_shielding` as a selectable `mode: rt_analytic` baseline; ensure it is **never**
      called from the RCRL path (factory refuses to mix it with `mode: rcrl`); grep to confirm.
- [ ] Update `_rl_maybe_finetune` hook-up under `safety.refine_*`.
- [ ] Remove `profile_fallback_start_window` gating.

**Phase 3 — instrumentation & evaluation**
- [ ] Log per-window: λ_t, ℓ_t, accept/fallback, running miss ratio `(1/t)Σℓ`.
- [ ] Add a summary-JSON field: realized long-run miss ratio vs α, and the Theorem-1 bound
      `α + (1+γB)/(γT)`.
- [ ] Ablation harness: `safety.mode ∈ {rcrl, conformal, none}` across `dynamic_mode ∈ {0,1,2,3}`.

**Phase 4 (optional / paper alignment, separate workstream)**
- [ ] Reward exactly to Eq. 18–21 (nullify `R_qos` on miss; band-based `R_util`; `P_miss=κ0+κ1·M`).
- [ ] State to Eq. 15–17 (add `N_k` and miss-ratio `M` to the system component) — **requires
      classifier/agent retrain** (state dim changes).
- [ ] **Openness / variable N_k** (tasks join/leave): variable-branch DQN via masking/padding and
      variable-length classifier input. Large change; own design doc.

---

## 8. Validation strategy

The whole point is Theorem 1, so validate it empirically:

1. **Guarantee check** — over a long horizon with `mode: rcrl`, confirm the realized deadline-miss
   ratio `(1/T)Σℓ_t → α` from above and stays under `α + (1+γB)/(γT)` for all T. Sweep
   `α ∈ {0.05, 0.1, 0.2}` and check the realized ratio tracks α.
2. **Distribution-shift robustness** — under `dynamic_mode 1/2/3`, the guarantee should hold with
   **no** recalibration step (this is ARC's advantage over the conformal guard). Compare settling +
   miss ratio against the legacy `conformal` guard and `none`.
3. **QoS quality** — since safety is guaranteed regardless of classifier accuracy, measure how much
   delivered value the RL agent + REFINE retains vs the conservative floor (all-`d_safe`).
4. **λ dynamics** — plot λ_t; verify it stays ≲ `1+γB` (Lemma 1) and reacts to miss bursts.

---

## 9. Decisions & open questions

**Resolved**
- **§5.1** — remove the skip; always store the applied-action transition (Alg. 1 L14). ✔
- **§5.3** — keep `rt_shielding` as a comparison baseline (`mode: rt_analytic`), never on the RCRL
  path. ✔

**Still open**
1. **Legacy path:** keep `ConformalGuard` as a selectable baseline for ablation? (Recommend keep —
   cheap, and gives an RCRL-vs-conformal comparison alongside RCRL-vs-analytic.)
2. **Phase 4 scope:** is variable-`N_k` openness in scope for *this* implementation, or a follow-up?
   It's the biggest lever on the "openness" motivation but also the biggest change (retrain +
   variable-branch net).
3. **α/γ defaults:** paper uses α=0.1, γ=0.05, λ₀=0.5 — adopt as defaults?
