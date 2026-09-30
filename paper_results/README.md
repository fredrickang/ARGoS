# paper_results/ — pinned results used in the paper

This directory is the **single, persistent home** for the runs, logs, figures, and tables that
the ARGoS paper reports. It is deliberately kept **separate from `outputs/`**, which is a
disposable, git-ignored scratch tree that experiment scripts overwrite freely.

```
Run experiments        ->  outputs/**            (fast, disposable, git-ignored)
Promote the keeper      ->  paper_results/…       (pinned, kept, with provenance)
```

Never edit an experiment's raw numbers by hand. If a result changes, re-run it and
re-`collect` it (see below) so the provenance stays truthful.

---

## Layout

```
paper_results/
├── README.md              this file — structure + experiment↔figure/table map
├── collect_result.py      promotes a chosen outputs/ run into experiments/<name>/
├── manifest.jsonl         append-only log: one line per collected result (name, run_id, metrics)
├── experiments/           one folder per reported result — the source of truth
│   └── <name>/
│       ├── meta.json         run metadata (taskset, seed, exec model, safety config)
│       ├── summary.json      aggregate metrics (RCRL guarantee, util, DMR, QoS, ε_carry…)
│       ├── windows.csv       per-window log (λ, dmr, util, avg_qos, fallback, shield_active)
│       ├── config.yaml       the exact config used (copied verbatim)
│       ├── provenance.json   where/when it came from, git sha, note, key metrics
│       └── <figure>.png      figure(s) rendered from this run (optional)
├── figures/               final figure files as they appear in the paper (fig1.png, …)
└── tables/                final table data (CSV/JSON) as it appears in the paper
```

`experiments/<name>/` is self-contained: config + logs + metrics + provenance for one result.
`figures/` and `tables/` hold the *paper-facing* artifacts (possibly assembled from several
experiments), so the manuscript references a stable path.

---

## How to add a result

1. **Run** the experiment into `outputs/` (any `scripts/run_*.py`, or `run.py <config>`), e.g.

   ```bash
   python scripts/run_safety_ablation.py --out-dir outputs/safety_ablation
   ```

2. **Promote** the run you want to report:

   ```bash
   python paper_results/collect_result.py fig4_guarantee \
       --out-dir outputs/safety_ablation/rcrl_dyn0 \
       --config configs/run_sim.yaml \
       --figure outputs/window_logs/demo_rcrl_visual.png \
       --note "RCRL, static task set, alpha=0.1, seed 12345"
   ```

   - `--out-dir` takes the **latest** `runs/*` under it; use `--run-dir outputs/.../runs/<id>`
     to pin a specific one.
   - `--no-windows` skips the (large) per-window CSV if you only need summary + figure.
   - Re-collecting the same `name` needs `--force`.

3. **Record it** in the map below (one row per figure/table).

---

## Experiment ↔ figure/table map  (fill in as results are finalized)

| Paper item | experiments/<name> | Script / config | α | dyn_mode | seeds | Status |
|---|---|---|---|---|---|---|
| _example_ Fig. (guarantee) | `_demo_rcrl` | `configs/run_sim.yaml` (demo) | 0.1 | 0 | 1 | example — delete |
| Fig. — long-run miss ≤ α | `fig_guarantee` | `run_safety_ablation.py` | 0.1 | 0 | ? | TODO |
| Fig. — λ / fallback trajectory | `fig_lambda_traj` | `run.py run_sim.yaml` | 0.1 | 0 | ? | TODO |
| Fig. — RCRL vs conformal vs none | `fig_safety_ablation` | `run_safety_ablation.py` | 0.1 | 0–3 | ? | TODO |
| Fig. — adaptive window / ε_carry | `fig_window_sizing` | `run_window_size_sweep.py` | 0.1 | 3 | ? | TODO |
| Fig. — inaccurate profiles (QoS) | `fig_inaccurate_profiles` | `run_inaccurate_profiles.py` | 0.1 | 0 | ? | TODO |
| Table — settling after env change | `tab_settling` | `run_dynamic_experiments.py` | 0.1 | 1–3 | ? | TODO |

> Update α / dyn_mode / seeds to the values actually used, and flip Status to ✅ once pinned.

---

## Notes

- **Determinism.** Runs seed `random.seed(22)` (task-set generation) and honor
  `controller.rl.seed` for replicates. Record the seed in `--note`; multi-seed sweeps keep each
  seed's run under its own `experiments/<name>_seed<k>/` or as rows in a collected CSV.
- **Provenance is authoritative.** `provenance.json` pins the source `outputs/` run dir and the
  collector git sha. If `outputs/` is later cleared, the pinned bundle here still stands alone.
- **Size.** `windows.csv` is ~80 KB per 600-window run — fine to keep. For very long runs use
  `--no-windows` and rely on `summary.json`.
