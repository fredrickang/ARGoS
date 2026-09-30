# PATROL Safety Shielding

This document describes the **safety shielding** subsystem of PATROL — the layer that
protects the reinforcement-learning (RL) QoS controller from taking decisions that would
cause deadline misses. It is the distinctive research contribution of the project: a *learned,
self-calibrating, distribution-free* safety net wrapped around a Branching Dueling DQN.

Everything below refers to the reorganized `PATROL_refactored/` tree.

---

## 1. Where shielding fits

Once per **control window** (`window.size_ms`, default 5000 ms) the QoS controller reads the
previous window's statistics and re-assigns a QoS level to every task so that CPU utilization
tracks a reference target while deadline misses stay near zero. Execution times are stochastic
(a logistic-normal or Gaussian factor of WCET), so the controller can never *know* in advance
whether an action is safe.

```
 window boundary (EDF has just finished window w-1)
        │
        ▼
 RL agent proposes  action = [qos_0, qos_1, … qos_n]        patrol/rl/dqn.py
        │
        ▼
 ┌───────────────────────────── SAFETY SHIELD ─────────────────────────────┐
 │ (1) Learned miss predictor  → miss_prob                                  │
 │ (2) Conformal safety test   → is this action safe with 1-δ guarantee?    │
 │ (3) Safe fallback           → if unsafe, pick highest-value safe action  │
 │ (4) Online fine-tune        → if predictor drifts, retrain + recalibrate │
 │ (·) Hard RT shield          → analytic util ≤ 1.0 cap (last resort)      │
 └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 applied_action → task._task_info.cur_qos  (used by EDF for window w)
```

The shield lives in **[patrol/controllers/rl.py](../patrol/controllers/rl.py)** (`RlControllerMixin`),
and relies on the miss-prediction model interface in
**[patrol/controllers/profile.py](../patrol/controllers/profile.py)** (`ProfileControllerMixin`)
and the MLP defined in **[patrol/models/mlp.py](../patrol/models/mlp.py)**.

The whole thing is invoked from the window timer loop in
[patrol/sim/core.py](../patrol/sim/core.py) via `_qos_manager() → _rl_qos_mgmt()`.

---

## 2. The five mechanisms

### 2.1 Learned miss predictor (the "profile shield")

An **MLP binary classifier** predicts the probability that a candidate action will produce at
least one deadline miss in the coming window, *before* the action is applied.

- **Model** — `MLP` in [patrol/models/mlp.py:23](../patrol/models/mlp.py#L23): stacked
  `Linear → BatchNorm1d → ReLU → Dropout` blocks (default hidden `512,256,128,64`) ending in a
  single logit. Trained offline with `BCEWithLogitsLoss` (positive-class weighted to handle the
  rare-miss imbalance) on `(window_state → deadline_miss)` samples collected by the **profile**
  controller. Inference applies `sigmoid` to get `miss_prob ∈ [0,1]`.
- **State vector** — built by `_rl_build_profile_state`
  ([rl.py:61](../patrol/controllers/rl.py#L61), mirrors `_profile_build_state`
  [profile.py:164](../patrol/controllers/profile.py#L164)):

  ```
  x = [ w_sz,
        period_est_0, est_wcet_0(qos_0),
        period_est_1, est_wcet_1(qos_1),
        …
        period_est_n, est_wcet_n(qos_n) ]
  ```

  `period_est` is derived from how many jobs the task released last window (`w_sz / n_jobs`,
  walking backwards if the task was idle); `est_wcet` is the task's WCET table entry for the
  *candidate* QoS level. Crucially, the state encodes the **hypothetical action**, so the same
  model can score any candidate action, not just the one that ran.
- **Normalization & threshold** — the checkpoint stores feature `mean`/`std` (applied before
  inference) and a calibrated decision `threshold` (default `DEFAULT_MISS_THRESHOLD = 0.1224`,
  [profile.py:7](../patrol/controllers/profile.py#L7)). The threshold is chosen at train time by
  `find_best_threshold` (max validation accuracy, tie-broken toward 0.5).
- **Entry points** — `rl_shielding_predictor` ([rl.py:99](../patrol/controllers/rl.py#L99))
  returns `(pred, miss_prob, x_state)` for a single action;
  `_rl_predict_miss_probs_batch` ([rl.py:30](../patrol/controllers/rl.py#L30)) scores many
  candidate actions in one forward pass (used by the fallback search).

### 2.2 Conformal safety layer (distribution-free guarantee)

The raw classifier threshold is not enough — the model can be miscalibrated, and a single
threshold gives no statistical guarantee. PATROL adds a **split-conformal** style safety test on
top, so unsafe actions are rejected with a distribution-free **1 − δ** guarantee.

`_rl_is_conformal_safe(miss_prob)` ([rl.py:110](../patrol/controllers/rl.py#L110)) declares an
action safe **only if both** hold:

1. `miss_prob < threshold` — the classifier itself votes "safe"; **and**
2. `safe_prob = 1 − miss_prob  >  rl_safe_qhat` — the safe-probability clears the conformal
   quantile.

**Calibration of `rl_safe_qhat`** — `_rl_update_safe_calibration`
([rl.py:155](../patrol/controllers/rl.py#L155)) maintains a rolling buffer of the *safe scores*
`(1 − miss_prob)` observed **on windows that actually missed** (`y_miss == 1`). Given nonconformity
scores over these miss events, it computes the conformal quantile:

```
n     = number of stored miss scores (rolling window, profile_safe_window = 200)
k     = ⌈(n + 1)(1 − δ)⌉,   clamped to [1, n]           # δ = profile_safe_delta (0.1 / 0.01)
q_hat = sorted(safe_scores)[k − 1]
```

Intuitively `q_hat` is the safe-probability level that misses tend to sit *below*; requiring a
candidate's `safe_prob` to exceed `q_hat` rejects actions that "look like" historical misses,
with coverage controlled by δ.

### 2.3 Safe fallback (highest-value safe action)

When the proposed action is judged unsafe, `_rl_safe_fallback`
([rl.py:329](../patrol/controllers/rl.py#L329)) searches for a replacement.

- **Candidate set** — `_rl_build_grouped_fallback_candidates`
  ([rl.py:120](../patrol/controllers/rl.py#L120)) produces a *grouped descent*: start from the
  proposed action, then repeatedly decrement **every non-zero task's QoS by 1** until all reach 0.
  This yields a small, monotone ladder of progressively cheaper actions.
- **Scoring** — every candidate is scored in one batched forward pass, then filtered through the
  same `_rl_is_conformal_safe` test.
- **Selection** — among conformally-safe candidates, pick the one with the **highest total QoS**
  (`sum(cand)`), i.e. the *most valuable* action that is still safe. If none pass, fall back to the
  all-zero action (the safest possible). Returns `(chosen_action, chosen_prob, chosen_state)`.

**Gating & triggering** — `_rl_prepare_pending` ([rl.py:278](../patrol/controllers/rl.py#L278))
only runs fallback when:

- `enable_profile_fallback` is true (config), **and**
- `w_id ≥ profile_fallback_start_window` (default 500) — a warm-up so the calibration is populated
  before the shield starts overriding the policy.

Two trigger paths exist:

| Path | Condition | Meaning |
|------|-----------|---------|
| Classifier trip | `miss_prob ≥ threshold` | model predicts a miss |
| Conformal trip  | `miss_prob < threshold` **but** `safe_prob ≤ safe_qhat` | model says safe, conformal test disagrees |

When a fallback is applied, the tasks' `cur_qos` are overwritten to the chosen action and the
pending record is tagged `safe_fallback = True`.

**Interaction with RL learning** — in `_rl_qos_mgmt`
([rl.py:446](../patrol/controllers/rl.py#L446)), if a fallback was applied the transition is **not**
stored in the replay buffer (`store_transition` is skipped, [rl.py:481](../patrol/controllers/rl.py#L481)).
The agent is trained only on actions it actually got to execute, so the shield's overrides don't
poison the policy gradient. (On genuine misses without fallback, the transition is stored twice to
up-weight the negative signal.)

### 2.4 Online fine-tuning (drift adaptation)

The offline-trained predictor can drift as the environment changes.
`_rl_maybe_finetune` ([rl.py:239](../patrol/controllers/rl.py#L239)) monitors a running prediction
accuracy (`_rl_update_pred_metrics`, [rl.py:196](../patrol/controllers/rl.py#L196), which also
reports AUC) and retrains the MLP in place when it degrades:

- **Trigger** — `acc < profile_finetune_acc_thr` (0.8) **and** the fine-tune buffer is non-empty
  **and** `prev_win_id ≥ 100` **and** no cooldown active.
- **Retrain** — `_profile_finetune` ([profile.py:88](../patrol/controllers/profile.py#L88)) trains
  on the most recent windows (`profile_finetune_buffer = 500`, `epochs = 50`, `lr = 1e-4`,
  `batch = 128`) with `BCEWithLogitsLoss`, then **re-selects the decision threshold** on a held-out
  split (max accuracy, tie-broken toward 0.5) and writes it back into the live model.
- **Aftermath** — prediction-metric history is cleared, a cooldown is armed
  (`profile_finetune_cooldown = 50` windows) to prevent thrashing, and the conformal calibration is
  rebuilt.

**Calibration rebuild** — `_rl_rebuild_safe_calibration`
([rl.py:382](../patrol/controllers/rl.py#L382)) re-scores stored miss samples through the freshly
fine-tuned model to recompute `rl_safe_qhat`. If too few miss samples remain
(`profile_safe_min = 20`), it back-fills from the full sample history so the quantile stays
well-defined.

### 2.5 Hard real-time shield (analytic last resort)

`rt_shielding` ([rl.py:536](../patrol/controllers/rl.py#L536)) is a **model-free, analytic**
guardrail independent of the learned shield. It estimates each task's utilization from the
*measured* execution-time history (`build_estimated_exec_tables`,
[fcs.py:137](../patrol/controllers/fcs.py#L137)) and, while total estimated utilization exceeds
`1.0`, greedily reduces the QoS of the task with the **highest utilization gain** until the set is
schedulable under EDF. It is the schedulability-of-last-resort described in the README.

> **Status:** the method is implemented but **not currently wired into** `_rl_qos_mgmt`; it is
> available as a hard cap but dormant in the default flow. Search for `rt_shielding` before relying
> on it in an experiment.

---

## 3. Environment-change adaptation (supporting the shield)

When dynamic environment changes are injected (`controller.dynamic_mode > 0`), the core loop calls
`rl_adaptation_on_env_change` ([rl.py:578](../patrol/controllers/rl.py#L578), invoked from
[core.py:394](../patrol/sim/core.py#L394)). This is not a shield per se, but it keeps the shielded
controller stable across regime shifts:

- Reset RL exploration (`epsilon = 0.7`) and **clear the replay memory** — stale transitions no
  longer describe the new regime.
- **Warm-start** each task's QoS from an FCS-derived greedy allocation (`hvdf_history`,
  [fcs.py:176](../patrol/controllers/fcs.py#L176)) so the agent restarts from a sensible, feasible
  point rather than exploring blindly.
- Ramp the control window from `w_sz_min` back up to the base size (`dynamic_window`,
  [rl.py:592](../patrol/controllers/rl.py#L592)) — shorter windows react faster right after a change.

---

## 4. End-to-end control flow (`_rl_qos_mgmt`)

Per window, [rl.py:446](../patrol/controllers/rl.py#L446):

1. **Finalize the previous window.** Read its DMR → `miss = (dmr > 0)`. Save the sample, update the
   fine-tune buffer, update the conformal calibration (`_rl_update_safe_calibration`), update the
   prediction accuracy/AUC (`_rl_update_pred_metrics`), and maybe fine-tune (`_rl_maybe_finetune`).
2. **Build state & get action.** `get_env()` builds the RL state (util, ref-util, util-error, and
   per-task `[cur_qos, liveness, load]`); the DQN proposes an action.
3. **Apply & shield.** Write the proposed QoS to tasks, then `_rl_prepare_pending` runs the miss
   predictor + conformal test and applies a safe fallback if needed (overwriting `cur_qos`).
4. **Reward, store, train.** Compute the decoupled reward (util-proximity + value + safety); store
   the transition **only if no fallback fired**; then `train_model()`.

---

## 5. Configuration reference

Shield-related knobs under `controller:` (see
[configs/run_sim.yaml](../configs/run_sim.yaml), [configs/run_profile.yaml](../configs/run_profile.yaml)):

| Key | Default | Role |
|-----|---------|------|
| `profile_model_path` | `patrol/models/checkpoints/model.pt` | the learned shield MLP checkpoint |
| `enable_profile_fallback` | `true` (sim) / `false` (profile, smoke) | master switch for the conformal safe-fallback |
| `profile_fallback_start_window` | `500` | warm-up windows before fallback may override the policy |
| `profile_safe_delta` (δ) | `0.1` (`0.01` in run_profile) | conformal miscoverage level → tightness of `q_hat` |
| `profile_safe_window` | `200` | rolling window of miss scores used for calibration |
| `profile_safe_min` | `20` | minimum miss samples before back-filling calibration |
| `profile_finetune_acc_thr` | `0.8` | accuracy below which online fine-tuning triggers |
| `profile_finetune_buffer` | `500` | recent-window buffer size used for fine-tuning |
| `profile_finetune_batch` | `128` | fine-tune batch size |
| `profile_finetune_epochs` | `50` | fine-tune epochs |
| `profile_finetune_lr` | `1e-4` | fine-tune learning rate |
| `profile_finetune_cooldown` | `50` | windows to wait between fine-tunes |

The decision `threshold` is not a YAML knob — it is stored in the model checkpoint and updated
in place by fine-tuning.

---

## 6. How the shield is produced (pipeline)

The shield is trained, not hand-built. The full lifecycle:

1. **Profile** (`controller.mode: profile`) — `_profile_qos_mgmt`
   ([profile.py:202](../patrol/controllers/profile.py#L202)) randomizes QoS levels for target tasks
   and records `(x_state → y_miss)` samples under `dataset/profile/`.
2. **Split** — `python dataset/sample_and_split.py` → `dataset/train.pkl`, `dataset/val.pkl`.
3. **Train** — `python -m patrol.models.mlp --train … --val …` → a `.pt` checkpoint carrying
   `model_state`, `input_dim`, `hidden`, `mean`, `std`, and a calibrated `threshold`.
4. **Deploy** — point `profile_model_path` at the checkpoint and run `controller.mode: rl`; the
   conformal calibration and (optional) online fine-tuning then adapt it at runtime.

---

## 7. Summary — layered defense

| Layer | Type | Guarantee | Cost |
|-------|------|-----------|------|
| Miss predictor | learned (MLP) | probabilistic, calibrated threshold | one forward pass |
| Conformal test | statistical | distribution-free 1 − δ coverage | O(1) quantile lookup |
| Safe fallback | search | highest-value action passing the test | one batched forward pass |
| Online fine-tune | learned | recovers accuracy under drift | periodic retrain (cooldowned) |
| Hard RT shield | analytic | strict util ≤ 1.0 schedulability | O(tasks²), *currently dormant* |

Together these turn an otherwise-unconstrained RL policy into one that keeps deadline misses near
zero while still maximizing delivered value.
