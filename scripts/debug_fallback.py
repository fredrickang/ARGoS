"""Inspect the RCRL safe-fallback decision.

Runs a short simulation, instruments ``_rl_safe_fallback`` to capture the full
grouped-reduction candidate ladder for every fallback event (QoS-sum, classifier
miss-probability, guard accept/reject, and which candidate is chosen), then reports:

  * sample ladders (so you can see WHY a given window collapsed or not),
  * how often the fallback lands on d_safe (drop everything) vs a graceful reduction,
  * classifier miss-prob / accept-rate as a function of reduction depth
    (this is the tell for an over-pessimistic / OOD classifier).

Usage:
    python scripts/debug_fallback.py [config.yaml] [--samples 8] [--min-window 120]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patrol.controllers.rl import RlControllerMixin
from patrol.sim.config import build_config, build_plot_name_from_config, load_sim_config
from simso.core import Model

EVENTS = []


def _install_probe():
    orig = RlControllerMixin._rl_safe_fallback

    def probed(self, action, miss_prob):
        prev = self.w_id - 1
        cands = self._rl_build_grouped_fallback_candidates(action)
        xs = [self._rl_build_profile_state(prev, c) for c in cands]
        probs = self._rl_predict_miss_probs_batch(xs)
        guard = self._rl_safety()
        lam = getattr(guard, "lam", None)
        ladder = [
            {"sum": int(sum(c)), "miss_prob": float(p),
             "accept": bool(sum(c) == 0 or guard.is_accepted(p))}
            for c, p in zip(cands, probs)
        ]
        chosen = orig(self, action, miss_prob)  # actual behavior
        chosen_sum = int(sum(chosen[0])) if isinstance(chosen, tuple) else None
        EVENTS.append({
            "win": self.w_id, "lambda": lam,
            "orig_sum": int(sum(action)), "orig_miss_prob": float(miss_prob),
            "ladder": ladder, "chosen_sum": chosen_sum,
        })
        return chosen

    RlControllerMixin._rl_safe_fallback = probed


def _run(cfg_path, duration_ms):
    sim_cfg = load_sim_config(cfg_path)
    if duration_ms:
        sim_cfg["simulation"]["duration_ms"] = duration_ms
    sim_cfg["controller"]["mode"] = "rl"
    plot_name = build_plot_name_from_config(sim_cfg)
    cfg = build_config(
        sim_cfg["scheduler"]["filename"], sim_cfg["scheduler"]["classname"],
        sim_cfg["taskset"]["path"], sim_cfg["qos"]["level"], sim_cfg["qos"]["policy"],
        mu=sim_cfg["exec_time"]["mu"], sigma=sim_cfg["exec_time"]["sigma"],
        exec_model=sim_cfg["exec_time"].get("model", "logistic"),
        w_sz=sim_cfg["window"]["size_ms"], ctrl=sim_cfg["controller"],
        fcs_ctrl_param=sim_cfg["fcs_ctrl_param"],
        dynamic_mode=sim_cfg["controller"].get("dynamic_mode", 0),
        dynamic_config=sim_cfg.get("dynamic_config", {}),
        window_log_dir="outputs/_debug_fallback",
        sim_duration=sim_cfg["simulation"]["duration_ms"], plot_name=plot_name,
    )
    Model(cfg).run_model()


def _report(samples, min_window):
    if not EVENTS:
        print("No fallback events captured.")
        return
    evs = [e for e in EVENTS if e["win"] >= min_window] or EVENTS
    print(f"\n=== fallback events: {len(EVENTS)} total, {len(evs)} at window>={min_window} ===")

    # (1) sample ladders spread across the run
    idxs = np.linspace(0, len(evs) - 1, min(samples, len(evs))).astype(int)
    for i in idxs:
        e = evs[i]
        print(f"\n[win {e['win']}]  lambda={e['lambda']:.3f}  "
              f"orig_sum={e['orig_sum']} (miss_prob={e['orig_miss_prob']:.3f})  "
              f"-> chosen_sum={e['chosen_sum']}")
        print("   sum | miss_prob | accept | chosen")
        for lv in e["ladder"]:
            mark = " <== CHOSEN" if lv["sum"] == e["chosen_sum"] else ""
            print(f"   {lv['sum']:3d} |   {lv['miss_prob']:.3f}   |  "
                  f"{'Y' if lv['accept'] else 'n'}    |{mark}")

    # (2) aggregate: where does the fallback land?
    chosen = np.array([e["chosen_sum"] for e in evs], dtype=float)
    orig = np.array([e["orig_sum"] for e in evs], dtype=float)
    retain = np.divide(chosen, orig, out=np.zeros_like(chosen), where=orig > 0)
    print(f"\n=== where fallback lands (window>={min_window}, n={len(evs)}) ===")
    print(f"  chose d_safe (sum==0)         : {np.mean(chosen == 0)*100:5.1f}%")
    print(f"  chose full drop of QoS budget : retained {retain.mean()*100:5.1f}% of orig sum on avg")
    for lo, hi in [(0, 0.01), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        frac = np.mean((retain >= lo) & (retain < hi))
        print(f"    retained [{lo:.2f},{hi:.2f}): {frac*100:5.1f}%  {'#'*int(40*frac)}")

    # (3) classifier behavior by reduction depth (0 = proposed action, deeper = lower QoS)
    print(f"\n=== classifier miss_prob & accept-rate by reduction depth ===")
    print("  depth (0=proposed) | avg QoS-sum | avg miss_prob | accept-rate | n")
    by_depth = {}
    for e in evs:
        for d, lv in enumerate(e["ladder"]):
            by_depth.setdefault(d, []).append(lv)
    for d in sorted(by_depth)[:12]:
        lvs = by_depth[d]
        s = np.mean([x["sum"] for x in lvs]); mp = np.mean([x["miss_prob"] for x in lvs])
        ar = np.mean([x["accept"] for x in lvs])
        print(f"     {d:3d}              |   {s:6.1f}    |    {mp:.3f}     |    {ar:.2f}    | {len(lvs)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", default="configs/run_sim.yaml")
    p.add_argument("--duration-ms", type=int, default=1000000)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--min-window", type=int, default=120)
    args = p.parse_args()

    _install_probe()
    _run(args.config, args.duration_ms)
    _report(args.samples, args.min_window)


if __name__ == "__main__":
    main()
