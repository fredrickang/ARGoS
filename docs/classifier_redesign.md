# Miss-predictor classifier — feature & model redesign

The safety guarantee is only as useful as the classifier that scores candidate decisions.
The current classifier is the root cause of the fallback collapse and the controller's failure to
settle (see the fallback/REFINE diagnosis). This document redesigns it — **features first**, then
model and training.

## 1. What the current classifier gets (and why it fails)

`_rl_build_profile_state` / `_profile_build_state` build:

```
x = [ w_sz,  (T̂_1, C_1(q_1)),  (T̂_2, C_2(q_2)),  … ,  (T̂_N, C_N(q_N)) ]
```

- `C_i(q_i)` is read from **`tasks_wcet_tables[i][q_i]` — the static *nominal* WCET table**, not the
  runtime-observed execution time. This is a *profile* (contradicts the profile-free thesis) and is
  systematically wrong under the stochastic exec model (jobs run at ≈0.88×WCET here) and useless
  under heterogeneity/openness.
- **No previous-window state** — no `act_util`, `dmr`, or backlog. Misses are temporally correlated
  (backlog carries over); the model is blind to the current regime.
- **No total-utilization feature** — EDF feasibility is fundamentally `Σ C_i/T_i` vs 1, but the model
  gets raw per-task `T` and `C` separately and must *learn* to divide-and-sum. It never does well.
- **Fixed per-task concat** — tied to a specific `N` and task ordering; no generalization, no openness.

Observed consequences (from the fallback probe): the classifier outputs `miss_prob ≈ 1.0` across
almost the whole candidate ladder — **including `d_safe` (zero load), which physically cannot miss**
— i.e. a saturated cliff, not a function of load. The fallback therefore collapses to dropping all
jobs.

## 2. Design principle

Predict `P(miss in window k) = f(previous-window observed state, candidate decision's load estimated
from runtime history)`. Two feature groups, **all runtime-derived and N-invariant**:

### Group A — previous-window context (observed, from `window_log[k-1]`)
| feature | formula | source |
|---|---|---|
| `prev_util` | `act_util` last window | window_log |
| `prev_dmr` | `dmr` last window | window_log |
| `prev_util_err` | `ref_util − act_util` | window_log |
| `util_trend` | `act_util[k-1] − act_util[k-2]` | window_log (backlog/trend) |

### Group B — candidate decision load, from **runtime execution history**
Let `ĉ_i(q)` = mean and `σ̂_i(q)` = std of `tinfo.exec_history[i][q]` (observed per-job exec at QoS
`q`; 0 for `q=0`; monotone-interpolate empty entries as `build_estimated_exec_tables` already does).
Let `T̂_i` = period estimate = `prev_w_sz / n_jobs` (backfilled). Per-task util `u_i = ĉ_i(q_i)/T̂_i`.

| feature | formula | why |
|---|---|---|
| `est_util` | `Σ_i u_i` | **the** EDF feasibility signal; `d_safe → 0` |
| `est_util_tail` | `Σ_i (ĉ_i(q_i) + z·σ̂_i(q_i))/T̂_i`, z≈1.65 | stochastic overload / miss driven by the tail |
| `max_util` | `max_i u_i` | one heavy task can miss even at low total util |
| `std_util` | `std_i u_i` | load imbalance |
| `frac_active` | `#{i: q_i>0} / N` | how many tasks contend |
| `delta_util` | `est_util − prev_util` | how big a change this decision is vs now |
| `mean_qos` | `mean_i q_i / max_qos` | aggressiveness / value proxy |

≈ **11 features, independent of N** (vs the current `1 + 2N`). Compact, physically meaningful,
transferable across tasksets and task counts.

> **Hard sanity floor (inference):** if `est_util == 0` (all `q_i = 0`), return `miss_prob = 0`. No
> jobs are released, so a miss is impossible — never let the learned model claim otherwise.

## 3. Model

The relationship is close to physical (`miss ≈ σ(k·(util_tail − 1))`), so a **small** MLP suffices and
resists overfitting:

```
input(≈11) → Linear(64) → BN → ReLU → Dropout(0.1)
           → Linear(32) → BN → ReLU
           → Linear(1)  (logit)                # sigmoid at inference
```

Two properties worth adding:

- **Monotonicity in load.** `miss_prob` must be non-decreasing in `est_util` / `est_util_tail`. Enforce
  with a cheap finite-difference penalty during training (`relu(f(x) − f(x + Δ·e_util))` summed over a
  minibatch) — or a positive-weight monotone sub-network. This **guarantees the fallback ladder is
  monotone**, so the fallback finds the *least* reduction that is safe (graceful) instead of a cliff.
- **Calibration.** Temperature-scale the logits on a validation split so `miss_prob` is a true
  probability — important because the RCRL threshold `λ` compares against `1 − miss_prob`.

(Openness variant — optional **v3**: replace the aggregates with a Deep-Sets encoder — per-task MLP
`φ(u_i, q_i, …)` → mean/max pool → global MLP — for full per-task detail at variable `N`. Same
training/monotonicity story. Start with the aggregates; they already give N-invariance.)

## 4. Training & data

- **Recompute features from runtime history at BOTH train and inference** — profile mode already
  randomizes QoS per task, so the collected `(features, y_miss)` naturally span the `est_util` axis.
  Because features come from `exec_history` (which fills during collection), train- and
  inference-time features are consistent (this was *not* true before, where inference used the static
  table).
- Loss: `BCEWithLogitsLoss(pos_weight)` (imbalance) `+ λ_mono · monotonicity_penalty`.
- New `input_dim` ⇒ a fresh checkpoint. Pipeline unchanged: profile → split → train → deploy.
- REFINE keeps working and is now far more effective: with a monotone, load-based feature the model
  generalizes along the util axis, so finetuning on applied actions also corrects the *candidate*
  region (dissolving the train/query distribution-mismatch that muted REFINE before).

## 5. Why this fixes the observed failures

| failure | fix |
|---|---|
| `d_safe` predicted as miss → collapse | `est_util=0` feature + hard floor ⇒ `miss_prob≈0` |
| saturated cliff over the ladder | monotone-in-util model ⇒ smooth ladder ⇒ graceful fallback |
| static nominal WCET (profile, wrong) | runtime `exec_history` estimates ⇒ profile-free, calibrated |
| blind to regime/backlog | previous-window `util/dmr/trend` context |
| tied to fixed N, no openness | N-invariant aggregates (or Deep Sets) |
| REFINE blind to candidate region | load feature generalizes across the ladder from applied data |

## 6. Implementation steps

1. Add `_estimated_exec_stats()` → per-task `(ĉ_i(q), σ̂_i(q))` for all `q` from `exec_history`
   (extends `build_estimated_exec_tables` with std + interpolation).
2. Add `_build_miss_features(prev_win_id, qos_list)` computing the ≈11-feature vector above; wire it
   behind `classifier.feature_set: v2` (keep the old builder for A/B).
3. Shrink `MLP` (or add `hidden` default `64,32`); add the monotonicity penalty + temperature
   scaling in `patrol/models/mlp.py`; add the `est_util==0 → 0` inference floor in the predictor.
4. Collect a fresh profile dataset (mode `profile`) → train → point `classifier_model_path` at the
   new checkpoint.
5. Re-run the ablation + `debug_fallback.py`; expect a monotone ladder, graceful fallbacks, and — with
   a trustworthy classifier — the RL controller finally settling near `ref_util`.
