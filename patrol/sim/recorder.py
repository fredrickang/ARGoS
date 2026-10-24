"""Per-run metrics recorder.

Each run writes a self-describing directory:

    outputs/<...>/runs/<run_id>/
        meta.json      provenance: config, seed, git sha, timestamp, safety params
        windows.csv    one row per window (the single source of truth for per-window data)
        summary.json   scalars/aggregates only

``windows.csv`` is streamed (one flushed row per window) so a killed run keeps its
partial data and memory need not hold the whole trace. It replaces the parallel
per-window arrays that used to be embedded in the summary JSON.
"""

import csv
import json
import subprocess
from pathlib import Path


class RunRecorder:
    SCHEMA_VERSION = 1

    # Fixed column schema for windows.csv. Missing keys are written blank, so the same
    # schema serves every controller mode (fcs/rl/profile).
    WINDOW_COLUMNS = [
        "w_id", "t_start", "t_end", "w_sz",
        "act_util", "ref_util", "util_err",
        "n_jobs", "n_miss_jobs", "n_job_releasable", "dmr", "miss", "eps_carry",
        "avg_qos", "value", "reward",
        "safe_fallback", "rcrl_lambda", "risk_score", "guard_mode", "shield_active",
    ]

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._win_path = self.run_dir / "windows.csv"
        self._win_file = open(self._win_path, "w", newline="", encoding="utf-8")
        self._win_writer = csv.DictWriter(
            self._win_file, fieldnames=self.WINDOW_COLUMNS, extrasaction="ignore"
        )
        self._win_writer.writeheader()
        self._win_file.flush()
        self._closed = False

    def write_meta(self, meta):
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

    @staticmethod
    def window_row(w_id, entry, guard_mode=""):
        wr = entry.get("w_range", (None, None))
        dmr = entry.get("dmr", 0.0) or 0.0
        sf = entry.get("safe_fallback")
        return {
            "w_id": w_id,
            "t_start": wr[0],
            "t_end": wr[1],
            "w_sz": entry.get("w_sz"),
            "act_util": entry.get("act_util"),
            "ref_util": entry.get("ref_util"),
            "util_err": entry.get("util_err"),
            "n_jobs": entry.get("n_jobs"),
            "n_miss_jobs": entry.get("n_miss_jobs"),
            "n_job_releasable": entry.get("n_job_releasable"),
            "dmr": dmr,
            "miss": 1 if dmr > 0.0 else 0,
            "eps_carry": entry.get("eps_carry"),
            "avg_qos": entry.get("avg_qos"),
            "value": entry.get("value"),
            "reward": entry.get("reward"),
            "safe_fallback": "" if sf is None else int(bool(sf)),
            "rcrl_lambda": entry.get("rcrl_lambda"),
            "risk_score": entry.get("risk_score"),
            "guard_mode": guard_mode or "",
            "shield_active": "" if entry.get("shield_active") is None else int(bool(entry.get("shield_active"))),
        }

    def append_window(self, w_id, entry, guard_mode=""):
        self._win_writer.writerow(self.window_row(w_id, entry, guard_mode))
        self._win_file.flush()

    def write_summary(self, summary):
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    def close(self):
        if self._closed:
            return
        try:
            self._win_file.close()
        except Exception:
            pass
        self._closed = True


def git_sha(cwd):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def make_run_id(base, *, safety_mode=None, seed=None, timestamp):
    """Build a collision-free run id. ``timestamp`` (a string) must be supplied by the
    caller — kept explicit so the recorder has no hidden clock dependency."""
    parts = [base]
    if safety_mode:
        parts.append(str(safety_mode))
    if seed is not None:
        parts.append(f"seed{seed}")
    parts.append(str(timestamp))
    return "__".join(parts)


def load_windows(run_dir):
    """Load ``windows.csv`` from a run directory into a list of typed dicts.

    Convenience reader for plotting/analysis; numeric columns are coerced to float
    (blank -> None). Falls back gracefully if the file is missing.
    """
    path = Path(run_dir) / "windows.csv"
    if not path.exists():
        return []
    int_cols = {"w_id", "n_jobs", "n_miss_jobs", "miss", "safe_fallback", "shield_active"}
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if v == "" or v is None:
                    row[k] = None
                elif k in int_cols:
                    row[k] = int(float(v))
                elif k == "guard_mode":
                    row[k] = v
                else:
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        row[k] = v
            rows.append(row)
    return rows
