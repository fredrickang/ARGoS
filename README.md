# ARGoS — Adaptive Risk-Guaranteed QoS Scheduling

A research simulator for **QoS-adaptive real-time scheduling** on a single processor,
built on top of [SimSo](https://github.com/MaximeCheramy/simso). This is the implementation
behind the ARGoS paper (formerly developed under the name *PATROL*; the importable Python
package is still called `patrol`).

Every periodic task can run at several **QoS levels** (higher QoS ⇒ larger WCET ⇒ more CPU).
Once per **control window**, a controller decides which level each task should use, so that CPU
utilization tracks a **reference target** and deadline misses stay near zero, while maximizing
the delivered **value** — under stochastic execution times and a changing environment.

The scheduler is plain **EDF**; the research contribution is the QoS **controller** on top of
it, and a learned **safety layer** that shields a reinforcement-learning controller from unsafe
decisions.

> Reorganized ("clean") layout grouped into an importable `patrol` package. The default config
> (`configs/run_sim.yaml`) runs the paper's method: RCRL adaptive risk control + guaranteed
> fallback, releasable-normalized miss loss (Eq. 3), and adaptive window sizing (§VII-C).
> `configs/run_sim.stable.yaml` pins the pre-paper baseline. See [Project layout](#project-layout).

---

## Quick start

```bash
pip install -r requirements.txt        # simso, torch, numpy, pyyaml, matplotlib, tqdm
# or:  pip install -e .

python run.py configs/smoke.yaml        # tiny FCS run — verifies the install end-to-end
python run.py configs/run_sim.yaml      # full RL run (default)
python run.py configs/run_profile.yaml  # profiling / data collection
```

All runs are driven entirely by a YAML config (see [Configuration](#configuration)).
Run from the project root so the `patrol` package is importable.

---

## Core idea

```
                per control window (w_sz ms)
  ┌──────────────────────────────────────────────────────────────┐
  │  EDF scheduler (SimSo)  ──►  window metrics: util, DMR, value  │
  │                                        │                       │
  │                                        ▼                       │
  │                              QoS controller decides            │
  │                          cur_qos[task] for next window         │
  │                        (FCS  /  RL+shield  /  Profile)         │
  └──────────────────────────────────────────────────────────────┘
```

- **Tasks** are periodic with a per-task `wcet_table[qos_level]`. QoS 0 means "skip / no value".
- **Value** delivered by a task grows with its QoS level (the greedy allocator uses `qos²`).
- Each window the controller reads the previous window's stats and re-assigns QoS levels to
  drive `actual_util → ref_util` with `DMR → ref_dmr`.
- Execution times are stochastic: actual runtime is a **logistic-normal** factor of WCET
  (`_exec_gen_logistic_norm`, parameters `mu`, `sigma`), so the controller never knows the true
  execution time in advance.

---

## The three controllers (`controller.mode`)

### 1. `fcs` — Feedback Control Scheduling (baseline) · `patrol/controllers/fcs.py`
A proportional controller on utilization error and deadline-miss error adjusts a scalar
**utilization budget**; a greedy value/util-gain allocator (`hvdf`) distributes that budget into
discrete per-task QoS levels. Deterministic, model-free, cheap.

### 2. `rl` — Branching Dueling DQN + learned safety shield · `patrol/controllers/rl.py`
A **branching dueling DQN** (`patrol/rl/dqn.py`) outputs a QoS level per task in one shot.
Reward blends **util-proximity, value, and safety**. On top of the policy sits the distinctive
**safety layer**:
- **Shield / miss predictor** — the profile MLP predicts a candidate action's deadline-miss
  probability *before* it is applied.
- **Conformal safety** — a running quantile (`rl_safe_qhat`) gives a distribution-free
  guarantee; unsafe actions trigger `_rl_safe_fallback` to the highest-value safer action.
- **Online fine-tuning** — when shield accuracy drifts, the MLP is re-trained on recent windows
  and the calibration is rebuilt.
- **Hard RT shield** — `rt_shielding` caps total estimated utilization ≤ 1.0 as a last resort.
- **Env-change adaptation** — reset ε/replay and warm-start QoS from an FCS-derived base.

### 3. `profile` — Data collection & miss-prediction training · `patrol/controllers/profile.py`
Randomizes QoS levels to gather `(window_state → deadline_miss)` samples for training the
miss-predictor MLP (`patrol/models/`). This produces the shield the RL controller uses.

---

## Dynamic environments (`controller.dynamic_mode`)

The engine injects environment changes at fixed intervals to test adaptivity:
`0` static · `1` reference-utilization changes · `2` background-load changes · `3` task-set changes.

---

## Project layout

```
run.py                     Entry point (loads YAML, builds SimSo config, runs the model)
configs/                   run_sim.yaml · run_profile.yaml · smoke.yaml
patrol/                    Importable package
├── sim/
│   ├── scheduler.py       EDF QoS scheduler — composed from the mixins below (class EdfQosScheduler)
│   ├── core.py            Engine: window timer loop, logging, dynamic-env injection, exec noise
│   ├── config.py          Task-set generation + SimSo Configuration building
│   ├── task_info.py       CustomTaskInfo (SimSo task + wcet_table / max_qos / cur_qos / t_type)
│   └── metrics.py         Exec-time noise, plotting, summary-metrics JSON (util/DMR/value/settling)
├── controllers/
│   ├── fcs.py             FCS controller + greedy QoS allocator (hvdf)
│   ├── rl.py              RL controller + safety shield + conformal fallback + online fine-tune
│   └── profile.py         Profiling mode + MLP miss-prediction interface
├── rl/
│   └── dqn.py             Branching dueling DQN agent and reward functions
└── models/
    ├── mlp.py             MLP miss-predictor (train)         [was mlp_training/train_mlp.py]
    ├── attention.py       Attention variant (train)
    ├── infer.py           Batch inference
    └── checkpoints/       Trained .pt checkpoints (model.pt, attn_model.pt)
tasksets/                  Task-set generators + generated .txt utilization files
scripts/                   Experiment sweeps (ref-util, similarity, per-taskset) + plotting
dataset/                   sample_and_split.py + collected data (gitignored):
                           profile/, rl/ (raw samples written at runtime) → train.pkl, val.pkl
outputs/                   Runtime outputs — window logs, plots, summary JSON,
                           RL agent checkpoint (dqn_branching_model.pth) — gitignored
```

The scheduler is assembled by **mixin composition** in `patrol/sim/scheduler.py`:

```python
class EdfQosScheduler(SchedulerCoreMixin, FcsControllerMixin, RlControllerMixin,
                      ProfileControllerMixin, UtilMixin, Scheduler):
    pass
```

SimSo resolves the class by the scheduler file's *stem* (`scheduler`), so `scheduler.py` also
exposes a `scheduler` alias for the class (`scheduler.classname` in the config is informational).

---

## Configuration

Key knobs (`configs/run_sim.yaml`):

```yaml
scheduler: { filename: patrol/sim/scheduler.py, classname: EdfQosScheduler }
taskset:   { path: ./tasksets/taskset_1.txt }
qos:       { level: 10, policy: linear }        # #QoS levels and spacing policy
window:    { size_ms: 5000 }                    # control window length
exec_time: { mu: 2, sigma: 0.4 }                # execution-time noise
controller:
  mode: rl            # fcs | rl | profile
  dynamic_mode: 0     # 0 static | 1 ref-util | 2 background load | 3 task-set
  ref_util: 0.9       # utilization target the controller tracks
  ref_dmr: 0.01       # target deadline-miss ratio
  profile_model_path: ./patrol/models/checkpoints/model.pt   # the shield MLP
  enable_profile_fallback: true                              # conformal safety net on/off
```

---

## End-to-end workflow

1. **Profile** — run `mode: profile` across task sets → raw samples under `dataset/profile/`.
2. **Split** — `python dataset/sample_and_split.py` → `dataset/train.pkl`, `dataset/val.pkl`
   (globs `dataset/profile/*.pkl` by default).
3. **Train the shield** — `python -m patrol.models.mlp --train dataset/train.pkl --val dataset/val.pkl`
   → a `.pt` checkpoint (with a calibrated decision `threshold`).
4. **Evaluate** — run `mode: rl` (with the shield) vs. `mode: fcs` baseline, across `dynamic_mode`s,
   using `scripts/*` sweeps.
5. **Analyze** — summary JSON + plots under `outputs/`.

## Metrics reported per run (`outputs/*_summary.json`)

Actual vs. reference utilization (avg/std) · DMR (avg/max, miss windows) · value & reward
trajectories · average QoS · **settling time** after each environment change · shield
prediction accuracy / AUC (RL mode).

---

## Notes on this layout

- Import paths changed from flat modules to the `patrol` package (e.g. `from DQN import DQNAgent`
  → `from patrol.rl.dqn import DQNAgent`); logic is unchanged.
- The scheduler class was renamed `edf_mono` → `EdfQosScheduler` (a backward-compatible
  `edf_mono` alias remains).
- `branchingPairswise.py` from the original tree was unused (imported nowhere) and is omitted.
- Large regenerable artifacts (sampled `.pkl` datasets, logs, experiment outputs) are not copied;
  they are produced by running the pipeline and are gitignored.
