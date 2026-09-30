#!/usr/bin/env python
"""Promote a chosen simulation run into ``paper_results/`` as a pinned, paper-ready result.

Experiments are *run* into the disposable, gitignored ``outputs/`` tree (fast, overwritten
freely). When a run is the one you want to report, ``collect_result.py`` copies its canonical
artifacts (meta.json, summary.json, windows.csv) — plus the exact config and any figures — into
``paper_results/experiments/<name>/`` together with a ``provenance.json`` recording where it came
from, when, and the key metrics. It also appends a line to ``paper_results/manifest.jsonl``.

This keeps a clean, stable separation:

    outputs/**                 disposable working runs (gitignored)
    paper_results/experiments  the exact runs the paper reports (kept)

Usage
-----
    # promote the latest run under an experiment out-dir
    python paper_results/collect_result.py fig4_guarantee \
        --out-dir outputs/safety_ablation/rcrl_dyn0 \
        --config configs/run_sim.yaml \
        --figure outputs/window_logs/demo_rcrl_visual.png \
        --note "RCRL, static task set, alpha=0.1"

    # or promote one specific recorder run dir
    python paper_results/collect_result.py fig4_guarantee \
        --run-dir outputs/window_logs/runs/<run_id> ...

Re-collecting the same name refuses to clobber unless ``--force`` is given.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
from datetime import datetime


HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(HERE, "experiments")
MANIFEST = os.path.join(HERE, "manifest.jsonl")

# Key metrics surfaced into provenance.json for quick scanning (pulled from summary.json).
_SUMMARY_KEYS = [
    ("rcrl", "loss_mode"),
    ("rcrl", "target_alpha"),
    ("rcrl", "realized_miss_rate"),
    ("rcrl", "theorem_bound"),
    ("rcrl", "guarantee_satisfied"),
    ("rcrl", "fallback_rate"),
    ("rcrl", "eps_carry_max"),
    ("rcrl", "carry_in_precondition_ok"),
    ("rcrl", "w_sz_min"),
    ("rcrl", "w_sz_max"),
    ("rcrl", "adaptive_window"),
    ("overall", "avg_actual_util"),
    ("overall", "avg_dmr"),
    (None, "avg_qos_controllable"),
]


def _git_sha():
    for cand in (HERE, os.path.dirname(HERE)):
        try:
            sha = subprocess.check_output(
                ["git", "-C", cand, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if sha:
                return sha
        except Exception:
            continue
    return "unknown"


def _resolve_run_dir(args):
    if args.run_dir:
        if not os.path.isdir(args.run_dir):
            raise SystemExit(f"--run-dir not found: {args.run_dir}")
        return os.path.abspath(args.run_dir)
    if args.out_dir:
        runs = sorted(glob.glob(os.path.join(args.out_dir, "runs", "*")),
                      key=os.path.getmtime)
        runs = [r for r in runs if os.path.isdir(r)]
        if not runs:
            raise SystemExit(f"no runs/* under --out-dir: {args.out_dir}")
        return os.path.abspath(runs[-1])  # most recent
    raise SystemExit("provide either --run-dir or --out-dir")


def _extract_metrics(summary):
    out = {}
    for section, key in _SUMMARY_KEYS:
        node = summary.get(section, {}) if section else summary
        if isinstance(node, dict) and key in node:
            out[key] = node[key]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Pin a run into paper_results/.")
    p.add_argument("name", help="experiment / figure name, e.g. fig4_guarantee")
    p.add_argument("--run-dir", help="a specific recorder run dir (runs/<run_id>)")
    p.add_argument("--out-dir", help="experiment out-dir; latest runs/* is taken")
    p.add_argument("--config", help="path to the config YAML used (copied verbatim)")
    p.add_argument("--figure", action="append", default=[],
                   help="figure file(s) to copy alongside (repeatable)")
    p.add_argument("--note", default="", help="one-line description for provenance")
    p.add_argument("--no-windows", action="store_true",
                   help="skip copying windows.csv (keep the bundle small)")
    p.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    args = p.parse_args(argv)

    run_dir = _resolve_run_dir(args)
    dest = os.path.join(EXPERIMENTS_DIR, args.name)
    if os.path.exists(dest) and not args.force:
        raise SystemExit(f"{dest} already exists (use --force to overwrite)")
    os.makedirs(dest, exist_ok=True)

    copied = []
    for fname in ("meta.json", "summary.json"):
        src = os.path.join(run_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, fname))
            copied.append(fname)
    if not args.no_windows:
        src = os.path.join(run_dir, "windows.csv")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, "windows.csv"))
            copied.append("windows.csv")
    if args.config and os.path.exists(args.config):
        shutil.copy2(args.config, os.path.join(dest, "config.yaml"))
        copied.append("config.yaml")
    for fig in args.figure:
        if os.path.exists(fig):
            base = os.path.basename(fig)
            shutil.copy2(fig, os.path.join(dest, base))
            copied.append(base)

    summary = {}
    spath = os.path.join(run_dir, "summary.json")
    if os.path.exists(spath):
        try:
            summary = json.load(open(spath))
        except Exception:
            summary = {}
    meta = {}
    mpath = os.path.join(run_dir, "meta.json")
    if os.path.exists(mpath):
        try:
            meta = json.load(open(mpath))
        except Exception:
            meta = {}

    provenance = {
        "name": args.name,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "collector_git_sha": _git_sha(),
        "source_run_dir": run_dir,
        "run_id": meta.get("run_id"),
        "run_created_at": meta.get("created_at"),
        "note": args.note,
        "files": copied,
        "config": args.config,
        "key_metrics": _extract_metrics(summary),
    }
    with open(os.path.join(dest, "provenance.json"), "w") as fh:
        json.dump(provenance, fh, indent=2)

    with open(MANIFEST, "a") as fh:
        fh.write(json.dumps({
            "name": args.name,
            "collected_at": provenance["collected_at"],
            "run_id": provenance["run_id"],
            "note": args.note,
            "key_metrics": provenance["key_metrics"],
        }) + "\n")

    print(f"[collect] {args.name}: pinned {len(copied)} file(s) -> {os.path.relpath(dest, os.path.dirname(HERE))}")
    for f in copied:
        print(f"          - {f}")
    if provenance["key_metrics"]:
        print("          key metrics:")
        for k, v in provenance["key_metrics"].items():
            print(f"            {k}: {v}")


if __name__ == "__main__":
    main()
