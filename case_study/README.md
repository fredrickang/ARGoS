# ARGoS-VA — Edge Video-Analytics Case Study

A **real prototype** that instantiates the [ARGoS](..) framework
(*Adaptive Risk-Guaranteed QoS Scheduling*) as an **edge video-analytics server**.

This is the concrete use case for the ARGoS paper. Where the `patrol` simulator studies
the mechanism on abstract synthetic task sets, this prototype runs it on a **real perception
pipeline over real (or recorded) camera streams on real hardware** — the setting the paper
describes as its representative instance.

> This directory lives under the ARGoS repo but is independent of the `patrol` simulator: it
> depends on `patrol` only as an installed package, reusing the ARGoS core (RL agent +
> SafetyGuard) and wrapping it around an actual video pipeline.

---

## Paper → system mapping

| Paper concept | This prototype |
|---|---|
| Task `τ_i` | One camera stream (an RTSP feed or a recorded video file) |
| Job | One frame of that stream, with implicit deadline `D_i = T_i = 1/fps` |
| QoS level `q_i ∈ {0..L-1}` | A processing configuration: `0` = drop the frame, higher = heavier model / higher resolution |
| Execution time `C_i(q_i)` | **Measured at runtime**, not profiled in advance — the real GPU/CPU cost of running config `q_i` on a frame |
| Shared remote CPU under EDF | The shared compute worker (single GPU or CPU pool) dispatching frame-jobs earliest-deadline-first |
| Scheduling window `W` | Fixed wall-clock interval over which QoS decisions hold and misses are counted |
| QoS decision `d_k` | Per-stream config choice for window `k`, produced by the RL agent |
| SafetyGuard | Classifier → safety score, adaptive risk controller (threshold `λ`), grouped-QoS fallback |
| Target risk `α` | User-specified bound on the long-run deadline-miss ratio |

Why this use case fits ARGoS's assumptions: the execution cost of each analytics config is
genuinely hard to profile ahead of time (it varies with hardware, frame content, and model),
streams join and leave as devices connect, and deadline misses (dropped/late frames) are
tolerable but must stay bounded.

---

## Architecture

```
   real / recorded camera streams
   ┌──────────┬──────────┬──────────┐
   │ stream 1 │ stream 2 │  ...      │   each = a periodic task, frames = jobs
   └────┬─────┴────┬─────┴────┬─────┘
        │ frames (released at 1/fps)     ← argos_va/streams
        ▼
   ┌───────────────────────────────────────────┐
   │ EDF dispatcher on shared compute (1 GPU)   │   ← argos_va/scheduler
   │ runs frame-job at its stream's QoS config  │      measures real C_i(q), deadline misses
   └───────────────────────────────────────────┘   ← argos_va/pipeline (the analytics models)
        │ per-window runtime state (N, U, M, per-task q/T/C̄)
        ▼
   ┌───────────────────────────────────────────┐
   │ ARGoS controller  (once per window W)      │   ← argos_va/controller  (reuses patrol core)
   │  RL agent → d_k → SafetyGuard → d̂_k        │
   │  classifier · adaptive λ · grouped fallback│
   └───────────────────────────────────────────┘
        │ applied QoS config per stream for window k+1
        └──────────────► back to the dispatcher
```

## Package layout

```
argos_va/
├── streams/     Stream ingestion — RTSP + recorded-file sources; frame release at native fps
├── pipeline/    Analytics models at multiple QoS configs (drop / low-res / full-res)
├── scheduler/   EDF dispatcher over the shared compute worker; real exec-time & miss measurement
├── controller/  ARGoS: RL agent + SafetyGuard, adapted from the patrol package
├── runtime/     Window loop, runtime-state assembly, metrics, adaptive window sizing
└── config/      Server configuration (streams, models, window, α, hardware)
configs/         Ready-to-run server configs
models/          Analytics model weights + measured per-config profiles
data/            Recorded camera streams for reproducible runs
scripts/         run the server · profile configs · reproduce experiments
results/         Per-window logs, plots, summary JSON
docs/            design.md — detailed design & open questions
```

## Status

Scaffold only. See [docs/design.md](docs/design.md) for the full design and the ordered
build plan. Nothing is implemented yet.
