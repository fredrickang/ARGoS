"""Visualize a scheduler run: an EDF Gantt chart (which frame ran when, and whether it met
its deadline) over a per-window utilization / deadline-miss-ratio timeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ..scheduler.edf import RunTrace
from ..scheduler.model import Task

C_ONTIME = "#2e7d32"
C_LATE = "#ef6c00"
C_DROP = "#c62828"
C_UTIL = "#1565c0"
C_DMR = "#c62828"
C_GRID = "#cfcfcf"


def plot_run(trace: RunTrace, tasks: list[Task], W: float, out_path: str | Path, *,
             ladder=None, alpha: float | None = None, title: str | None = None) -> Path:
    tasks = sorted(tasks, key=lambda t: t.task_id)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_g, ax_u) = plt.subplots(
        2, 1, figsize=(13, 2.4 + 0.5 * len(tasks)), sharex=True,
        gridspec_kw={"height_ratios": [max(2, len(tasks)), 2], "hspace": 0.12})

    # ---- Gantt: one row per task ----
    ylabels = []
    for row, task in enumerate(tasks):
        recs = [r for r in trace.records if r.task_id == task.task_id]
        ontime = [(r.start_ms, r.exec_ms) for r in recs
                  if r.status == "ontime" and r.start_ms is not None]
        late = [(r.start_ms, r.exec_ms) for r in recs
                if r.status == "late" and r.start_ms is not None]
        ax_g.broken_barh(ontime, (row - 0.38, 0.76), facecolors=C_ONTIME)
        ax_g.broken_barh(late, (row - 0.38, 0.76), facecolors=C_LATE)
        for r in recs:  # deadline miss markers
            if r.status == "dropped":
                ax_g.plot(r.deadline_ms, row, marker="x", ms=6, mew=1.6, color=C_DROP)
            elif r.status == "late":
                ax_g.plot(r.finish_ms, row, marker="|", ms=10, mew=1.6, color=C_DROP)
        res = None if ladder is None else ladder[task.qos_level]
        rlab = "drop" if res is None else f"{res[1]}x{res[0]}"
        ylabels.append(f"{task.name}\nq{task.qos_level}·{rlab}·{task.period_ms:.0f}ms")

    for k in range(1, int(trace.duration_ms / W) + 1):  # window boundaries
        ax_g.axvline(k * W, color=C_GRID, lw=0.8, zorder=0)
    ax_g.set_yticks(range(len(tasks)))
    ax_g.set_yticklabels(ylabels, fontsize=8)
    ax_g.set_ylim(-0.6, len(tasks) - 0.4)
    ax_g.invert_yaxis()
    ax_g.set_ylabel("stream (EDF)")
    ax_g.legend(handles=[
        mpatches.Patch(color=C_ONTIME, label="on-time"),
        mpatches.Patch(color=C_LATE, label="late frame"),
        plt.Line2D([], [], marker="x", ls="", color=C_DROP, label="dropped (expired)"),
    ], loc="upper right", ncol=3, fontsize=8, framealpha=0.9)
    ax_g.set_title(title or "EDF schedule pattern", fontsize=11, loc="left")

    # ---- Utilization + DMR per window ----
    wins = trace.windows(W)
    starts = [w.start_ms for w in wins]
    centers = [(w.start_ms + w.end_ms) / 2 for w in wins]
    ax_u.bar(starts, [w.utilization for w in wins], width=W, align="edge",
             color=C_UTIL, alpha=0.35, edgecolor=C_UTIL, linewidth=0.6, label="utilization")
    ax_u.axhline(1.0, color=C_UTIL, ls=":", lw=1, alpha=0.8)
    ax_u.plot(centers, [w.dmr for w in wins], color=C_DMR, marker="o", ms=3, lw=1.4,
              label="DMR")
    if alpha is not None:
        ax_u.axhline(alpha, color="#6a1b9a", ls="--", lw=1.2, label=f"α={alpha:g}")
    ax_u.set_ylim(0, max(1.15, max((w.utilization for w in wins), default=1) * 1.05))
    ax_u.set_ylabel(f"per-window (W={W:.0f}ms)")
    ax_u.set_xlabel("time (ms)")
    ax_u.set_xlim(0, trace.duration_ms)
    ax_u.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.9)
    ax_u.grid(axis="y", color=C_GRID, lw=0.6)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_guarantee(controller, out_path: str | Path, *, title: str | None = None) -> Path:
    """The ARGoS timing guarantee in action: per-window miss loss, its running average, the
    target α, the Theorem-1 bound envelope, and the adaptive threshold λ."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loss = controller.loss_history
    ravg = controller.running_avg_loss()
    lam = controller.lambda_history
    alpha = controller.guard.alpha
    ks = list(range(1, len(loss) + 1))
    bound = [controller.theorem_bound(k) for k in ks]

    fig, (ax, axl) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    ax.bar(ks, loss, width=1.0, color=C_DMR, alpha=0.25, label="per-window loss ℓ")
    ax.plot(ks, ravg, color=C_DMR, lw=2, label="running avg loss")
    ax.plot(ks, bound, color="#455a64", ls=":", lw=1.5, label="Theorem-1 bound")
    ax.axhline(alpha, color="#6a1b9a", ls="--", lw=1.5, label=f"target α={alpha:g}")
    ax.set_ylim(0, max(alpha * 3, max(loss, default=alpha) * 1.1, 1e-3))
    ax.set_ylabel("deadline-miss loss")
    ax.set_title(title or "ARGoS long-run deadline-miss guarantee", fontsize=11, loc="left")
    ax.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.9)
    ax.grid(axis="y", color=C_GRID, lw=0.6)

    axl.plot(ks, lam, color=C_UTIL, lw=1.5)
    axl.axhline(1.0, color=C_GRID, lw=1, ls=":")
    axl.set_ylabel("threshold λ")
    axl.set_xlabel("scheduling window k")
    axl.grid(axis="y", color=C_GRID, lw=0.6)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
