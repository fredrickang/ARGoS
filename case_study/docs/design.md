# ARGoS-VA design

Detailed design for the edge video-analytics prototype of ARGoS. This is a **real** system:
real analytics models, real (or recorded) camera streams, real hardware, real measured
execution times. The ARGoS timing guarantee is exercised end-to-end.

## 1. System model realized

- **Streams (tasks).** Each admitted camera stream `τ_i` produces frames at a fixed rate
  `fps_i`, giving period `T_i = 1000/fps_i` ms and implicit deadline `D_i = T_i`. Streams may
  join/leave at runtime (open system).
- **Frames (jobs).** Each frame is a job released at its stream's period. A frame-job must
  finish (be processed or explicitly dropped) before the next frame of that stream is due,
  otherwise it is a deadline miss.
- **QoS configs.** Each stream supports `L` configurations indexed `0..L-1`:
  - `0` — drop the frame (no compute, no value). This is the per-task component of `d_safe`.
  - `1..L-1` — increasing model cost, e.g. `YOLOv8n@320 → n@640 → s@640 → m@640`, or a
    detect-every-Nth-frame ladder. Value grows with the level; cost grows monotonically but
    not necessarily proportionally.
- **`C_i(q)` unknown a priori.** The prototype never assumes a profile. It measures the wall
  time each config actually takes and maintains the running mean `C̄_i(q)` used in the RL state.

## 2. Compute model

The shared resource is a single inference worker (one GPU, or a fixed CPU pool exposed as one
server). Frame-jobs are queued and executed **earliest-deadline-first** by the dispatcher. Only
one job runs at a time on the worker (matching the single-CPU EDF model in the paper); this keeps
measured `C_i(q)` clean and the EDF semantics faithful. Batching across streams is out of scope
for v1 (it would break the one-job-at-a-time execution model and complicate loss attribution).

## 3. ARGoS controller (reused from `patrol`)

The controller runs once per window `W`. It reuses the ARGoS core in the `patrol` package rather
than reimplementing it:

- **RL agent** — branching dueling DQN (`patrol.rl.dqn`) with per-stream action branches over
  `L` QoS levels; shared taskset encoder for inter-stream awareness.
- **SafetyGuard** (`patrol.controllers.safety`):
  - *Classifier* `f(z_{k-1}, d_k) → [0,1]` safety score (MLP).
  - *Adaptive risk controller* — threshold update `λ_{k+1} = λ_k + γ(ℓ_miss,k − α)`.
  - *Guaranteed fallback* — grouped one-level-at-a-time QoS reduction, `d_safe` = all-zero.

The state fed to the agent is assembled from **runtime observations only**:
`z_k = [N_k, U_{k-1}, M_{k-1}] ∥ per-stream (q_{i,k-1}, T_i, C̄_{i,k-1}(q))`.

Open item: confirm how much of `patrol`'s controller is importable stand-alone vs. entangled
with the SimSo scheduler mixins. If entangled, factor the ARGoS core out of `patrol` into a
thin importable surface (agent + SafetyGuard) that both the simulator and this prototype call.

## 4. Runtime loop (per window k)

1. Assemble runtime state `z_{k-1}` from the previous window's measurements.
2. RL agent proposes `d_k`; SafetyGuard validates → `d̂_k` (fallback if rejected).
3. Apply `d̂_k`: streams process frames at their assigned configs for the whole window.
4. Measure per job: actual `C_i(q)`, deadline hit/miss (incl. carry-in jobs crossing the
   window boundary). Aggregate window utilization `U_k`, miss ratio `M_k = ℓ_miss,k` per Eq. (3),
   delivered value.
5. Update threshold `λ_{k+1}` from the observed `ℓ_miss,k`.
6. **Adaptive window sizing** — recompute `ε_carry,k` and grow `W_k` if needed to keep
   `ε_carry,k < α` (paper §VII-C: `W_k > T^H_k / α`).

## 5. Measurement & guarantee validation

The headline result the paper needs from this prototype: the **long-run deadline-miss ratio
stays below α** across a real, non-stationary workload (streams joining/leaving, content-driven
cost changes), *without any prior profile*. Log per window: `U_k`, `M_k`, running average miss
ratio, `λ_k`, applied `d̂_k`, delivered value, fallback invocations. Plot the running miss ratio
against `α` and the theorem envelope `α + (1+γB)/(γK)`.

## 6. Build plan (ordered)

1. **Config schema** (`config/`) — server config: streams, model ladder, `W`, `α`, `γ`, hardware.
2. **Pipeline** (`pipeline/`) — wrap 1 real detector at `L` configs behind a uniform
   `run(frame, level) → (result, exec_ms)` interface; level 0 = drop.
3. **Streams** (`streams/`) — recorded-file source first (reproducible), RTSP source later.
   Release frames on a real clock at each stream's fps.
4. **Scheduler** (`scheduler/`) — EDF dispatch over the single worker; job/window measurement.
5. **Controller** (`controller/`) — adapter that feeds runtime state into the `patrol` ARGoS
   core and returns validated per-stream configs.
6. **Runtime** (`runtime/`) — the window loop, metrics, adaptive window sizing.
7. **Scripts** — `run_server`, `profile_configs` (measure the cost ladder), experiment reproduction.

## 7. Open questions to resolve before coding

- Hardware target for the prototype (which GPU/CPU; is a GPU available in this environment?).
- Detector family + config ladder (YOLO sizes/resolutions vs. frame-skip vs. both).
- Recorded stream dataset (which videos, how many concurrent streams, what fps mix).
- Reusability of the `patrol` ARGoS core outside SimSo (see §3 open item).
- Whether v1 targets a live real-time server or a faithful real-hardware replay of recorded
  streams (still real compute, but deterministic arrival order for reproducible paper numbers).
