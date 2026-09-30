# RCRL controller — tuning summary (locked version)

Stable config snapshot: `configs/run_sim.stable.yaml` (copy of `configs/run_sim.yaml` at lock time).
This documents the final tuned defaults and the code changes that back them.

## Final `controller.rl` / `controller.safety` defaults

| Setting | Value | Why (evidence) |
|---------|-------|----------------|
| `safety.mode` | `rcrl` | Adaptive risk control; long-run miss ≤ α by Theorem-1. |
| `safety.target_miss_alpha` | 0.1 | Target deadline-miss rate. |
| `safety.fallback_start_window` | 200 | Delay shield so RL learns unshielded first. Sweep 0/100/200/300/500: 200 = best steady util, ~20× fewer interventions, guarantee holds. |
| `ref_util` | 0.7 | Start reference (experiments change from here). |
| `classifier.feature_set` | v2c | Candidate-only load features → separable fallback ladder. |
| `rl.epsilon` | 0.3 | Initial exploration. Settle is U-shaped in ε (3-seed); 0.3 ~2× faster settle, best QoS. |
| `rl.epsilon_reset` | 0.3 | Re-exploration on a sensed change. 0.7 caused post-change saturation + replace-thrash. |
| `rl.epsilon_decay_per_window` | true | Anneal ε every window → exploration fades after settling (jitter ~8× lower). |
| `rl.epsilon_decay` | 0.99 | 0.95 annealed too fast and killed post-change adaptation (util wandered, std 0.20). 0.99 settles before AND after changes. |
| `rl.explore` | budget | Budget-aware exploration from OBSERVED exec estimates (`build_estimated_exec_tables`, NOT the ground-truth wcet_table). Uniform explore over-allocated (~2.5× demand) → warm-up saturation. |
| `rl.explore_spread` | 0.25 | Gaussian spread of exploratory QoS around the budget centre. |
| `rl.reward` | v2 | `compute_reward_v2`: positive-capable, smooth Gaussian util-tracking + always-on QoS term. Fixes decoupled_sum (always ≤0 signal, narrow band that gated QoS off 90%). 3-seed: +15% QoS, ~1.7× faster settle. |
| `rl.reward_util_weight` | 3.0 | At 1:1 the QoS term kept util ABOVE a lowered ref (0.7→0.5 settled at 0.73). w_u=3 → ref_down tracks 0.48 AND QoS rises (tighter tracking uses the budget better). Improves every scenario. |
| `rl.hiddens_common/actions/value` | [256,256]/[128]/[128] | Bigger net: 3-seed halves steady jitter (0.031→0.014), keeps QoS, settles faster. |

## Validated behaviour (dynamic suite, `outputs/dynamic_final_wu3/`)

All 6 change scenarios (ref up/down, add, remove, replace, spaced) satisfy the Theorem-1
guarantee. Post-change tracking: ref_down→0.48 (target 0.5), ref_up→0.89 (0.9),
add/remove→0.70 (0.7), replace→0.61 (mild under-track). QoS 3.9–4.3 on ref scenarios.

## Code changes this session (all config-gated; defaults preserve behaviour)

- `patrol/rl/dqn.py`:
  - `compute_reward_v2(...)` — new positive-capable reward with `w_u`/`w_v` weights.
  - `get_action` — optional per-window ε anneal (`epsilon_decay_per_window`).
  - `DQNAgent.__init__` — `hiddens_common/actions/value` params (network size).
- `patrol/controllers/rl.py`:
  - `_rl_budget_explore_policy` — budget-aware exploration from observed exec estimates.
  - action-selection picks budget vs uniform explore; reward picks v2 vs decoupled_sum and
    passes `w_u`/`w_v`; `train_model` looped `updates_per_window` times.
- `patrol/sim/core.py`:
  - `controller.rl` knobs wired: `seed`, `explore`, `explore_spread`, `reward`,
    `reward_util_weight`, `reward_value_weight`, `updates_per_window`,
    `epsilon_decay_per_window`, `hiddens_*`, `load_path`/`save_path`/`save_model` (warm-start).
  - `_save_rl_policy` — persist policy at run end when `save_model: true`.

## Reusable experiment scripts

- `scripts/run_shield_delay_sweep.py` / `plot_shield_delay_sweep.py`
- `scripts/run_epsilon_sweep.py` / `plot_epsilon_sweep.py` (ε × seed, aggregates)
- `scripts/run_dynamic_experiments.py` (`--fig-dir`/`--fig-prefix` keep old figures)
- `scripts/pretrain_multi_taskset.py` (warm-start pretraining — see note below)

## Known limitations / not adopted

- **Warm-start** (`load_path`): wired and works for *static* cold-start removal, but a
  single-ref checkpoint fails to adapt to unseen refs (stuck), and the quick multi-taskset
  pretraining produced an over-allocating policy (shield fought it constantly). **Cold-start
  is the adopted default.** Fixing pretraining (converged episodes, forgetting mitigation,
  train without shield reliance) is future work.
- **replace** post-change tracking mildly under-shoots (~0.61 vs 0.7).
