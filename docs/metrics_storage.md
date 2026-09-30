# Metrics & logging storage — analysis and proposal

> **Status (2026-08-10): implemented.** `patrol/sim/recorder.py` (`RunRecorder`) writes a per-run
> directory `outputs/<log_dir>/runs/<run_id>/` with `meta.json` + streamed `windows.csv` +
> slim `summary.json`. Wired into `core.py` (`_init_recorder`, `_recorder_flush_ready` at the end of
> `_qos_manager`, finalize in `metrics._save_summary_metrics`). `run_id` includes safety mode + seed
> + microsecond timestamp → collision-free. The legacy `<base>_summary.json` (with the old arrays) is
> still written for back-compat; the ablation harness now prefers the per-run summary. Format: **CSV**
> (`load_windows()` helper provided). Follow-up: migrate `scripts/plot_*.py` off the summary arrays
> onto `windows.csv`, then drop the arrays from the legacy file.

## 1. How it's stored today

Data flows through three in-memory stores and one on-disk artifact.

**In-memory (grows O(#windows), never trimmed):**

| Store | Shape | Fields |
|-------|-------|--------|
| `self.window_log` | `{w_id: {...}}` | `w_range, act_util, ref_util, util_err, n_jobs, n_miss_jobs, dmr, avg_qos, value, reward, safe_fallback, rcrl_lambda` |
| `tinfo.win_log` (per task) | `{w_id: {...}}` | `exec_time, n_jobs, qos_level` |
| `tinfo.exec_history[qos]` | `list` | per-window exec means, per QoS level |

Plus RL buffers (`rl_samples`, `rl_pred_scores/labels`, `rl_finetune_buffer`, guard scores).

**Live logging:** `logger.info` colored string lines per window (`[Window][Summary NNNN] util=… dmr=…`,
`[RL][Guard] …`). Human-readable only; not machine-parseable.

**On-disk (once, at process exit via `atexit → _save_summary_metrics`):**
- One `*_summary.json` per run (~92 KB at 1999 windows), written to `window_log_dir`.
- Optional training-sample pickles under `dataset/profile/`, `dataset/rl/`.
- The filename encodes params: `scheduler_rl_dyn0_taskset_1_q10_linear_mu2_sig0.4_w5000_d10000000_summary.json`.

## 2. Problems

1. **Per-window raw data is thrown away.** `window_log` holds 12 fields × N windows, but only a
   handful survive to disk, and only as loose arrays. `act_util`, `reward`, `value`, `avg_qos`,
   `util_err`, `ref_util`, `safe_fallback` **per window are lost** after the run — you cannot
   re-plot util/reward/QoS trajectories post-hoc. `_save_window_log_plot` only works in-memory,
   mid-run.

2. **Parallel-array anti-pattern.** The summary already carries **6 positional arrays** that must be
   zipped by index: `window_job_counts` (N), `window_miss_job_counts` (N),
   `rcrl.miss_indicator_trajectory` (N), `rcrl.threshold_trajectory` (N), plus the *filtered*
   `miss_window_ids` / `miss_dmrs` (which index differently — desync risk). Every new per-window
   metric adds another array. No single source of truth.

3. **Aggregates and bulk trajectories conflated.** `summary.json` should be a small, stable-schema
   scalar report (means, stds, settle, guarantee) — but it now embeds 4×N-length arrays, so it
   grows linearly with duration (~92 KB now, ~1 MB for a 10× run) and is awkward to diff or collate
   across a sweep of hundreds of runs.

4. **Filename-as-database → collisions.** Runs are distinguished only by the encoded filename, which
   omits **seed, safety mode, α, γ, λ_min**. Two configs differing only in those write the **same
   file** and silently overwrite — this actually bit the clamped-vs-unclamped comparison earlier.

5. **No provenance.** The summary lacks the resolved config, seed, git SHA, timestamp, taskset hash,
   and full safety-guard params. A run is not self-describing or reproducible from its artifacts.

6. **Unbounded memory, no crash-resilience.** All stores grow with #windows and are only flushed at
   exit; a crash or `kill` loses the entire run, and very long runs are memory-bound.

## 3. Proposal — a per-run directory with three tiers

Replace "one fat JSON named by params" with a **self-describing per-run folder**:

```
outputs/<run_id>/
  meta.json        # provenance: resolved config, seed, git SHA, timestamp,
                   # taskset hash, safety params, code/schema version
  summary.json     # SMALL, stable-schema scalars/aggregates ONLY (no per-window arrays)
  windows.csv      # ONE row per window = the single source of truth for per-window data
  tasks.csv        # optional: one row per (window, task): exec_time, n_jobs, qos_level
```

- **`windows.csv`** — columns: `w_id, t_start, t_end, act_util, ref_util, util_err, n_jobs,
  n_miss_jobs, dmr, miss, avg_qos, value, reward, safe_fallback, rcrl_lambda, guard_mode`.
  This **replaces all 6 parallel arrays**. Plotting/analysis loads one table; a new metric is a new
  column (no desync). ~150 KB at 2000 rows, trivially loaded by `csv`/`numpy`/pandas.
- **`summary.json`** — scalars/aggregates only: `overall.*`, `settle`, `rcrl` (realized_miss_rate,
  theorem_bound, guarantee, α/γ/λ_min, fallback_rate, lambda_final/min/max), `rl_pred_acc/auc`.
  Drop the embedded N-length trajectories (they now live in `windows.csv`). Tiny (~2 KB), stable
  schema, cheap to diff and collate across a sweep.
- **`meta.json`** — makes each run reproducible and gives a unique `run_id` (e.g.
  `<mode>_<safety>_dyn<d>_seed<s>_<yyyymmdd-hhmmss>` or `+ short config hash`), which **fixes the
  filename collision**.

**Writer abstraction.** Introduce a small `RunRecorder` that owns `outputs/<run_id>/`, writes
`meta.json` at start, **appends a window row each window** (streaming → bounds memory + survives a
crash), and writes `summary.json` at exit. This decouples storage from the scheduler mixins and is
unit-testable. `window_log` can keep the last K windows in memory (for the controller's lookback)
and drop older rows after they're flushed.

**Format choice (no new hard dependency required):**

| Option | New dep | Streaming | Notes |
|--------|---------|-----------|-------|
| **CSV** (recommended) | none | yes (append) | universal, diffable, `numpy.genfromtxt`/pandas-optional |
| **JSONL** | none | yes (append) | one JSON obj/line, schema-flexible, great for crash-resilience |
| **Parquet** | pandas+pyarrow | no (batch) | smallest + typed + fastest partial reads, but adds deps |

Recommend **CSV** as the default `windows`/`tasks` format (zero new deps, streamable, everyone can
read it), with an optional Parquet writer auto-enabled if pandas is importable.

**Backward compatibility.** Keep emitting `*_summary.json` (slimmed) so existing consumers
(`scripts/plot_*.py`, `run_safety_ablation.py`) keep working; move the per-window arrays out behind
a deprecation window. The ablation harness already reads only the `rcrl`/`overall` scalar blocks, so
it is unaffected.

## 4. Suggested phasing

1. `RunRecorder` + per-run dir + `meta.json` (provenance, unique run_id → kills collisions).
2. Stream `windows.csv` each window from `window_log`; keep `summary.json` writing as-is.
3. Slim `summary.json` (drop N-length arrays once plots read `windows.csv`); update plotting scripts
   to load the table.
4. Optional: `tasks.csv`, Parquet-if-pandas, in-memory `window_log` trimming for long runs.
