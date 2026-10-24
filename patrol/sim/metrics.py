import atexit
import json
import pickle
import logging
import math
import os
import random

logger = logging.getLogger(__name__)


class UtilMixin:
    
    def _exec_gen_logistic_norm(self, wcet, mu, sigma, rng):
        z = rng.gauss(mu, sigma)
        y = 1.0 / (1.0 + math.exp(-z))
        return wcet * y

    def _exec_gen_gaussian(self, wcet, cov, rng):
        # Actual execution time ~ Gaussian(mean = nominal wcet, std = cov * wcet), clipped >= 0.
        # Matches the paper's soft-RT model (mean given by the QoS level's nominal exec time).
        if wcet <= 0.0:
            return 0.0
        return max(0.0, rng.gauss(wcet, cov * wcet))

    def _investigate_peak(self, wcet, mu, sigma, iters=20):
        def sigmoid(t):
            return 1.0 / (1.0 + math.exp(-t))

        # solve g(t) = t - mu - sigma^2 * (2*sigmoid(t) - 1) = 0
        t = mu  # good initial guess
        s2 = sigma * sigma

        for _ in range(iters):
            y = sigmoid(t)
            g = t - mu - s2 * (2.0 * y - 1.0)
            gp = 1.0 - 2.0 * s2 * y * (1.0 - y)
            t -= g / gp

        y_peak = sigmoid(t)
        x_peak = wcet * y_peak
        return x_peak

    def _generate_similar_table(self, base_table, similarity, min_value=0, max_tries=100000, metric="mape"):
        if similarity < 0.0 or similarity > 1.0:
            raise ValueError("similarity must be in [0, 1]")

        metric = str(metric).lower()
        if metric not in ("mape", "cosine"):
            raise ValueError("metric must be 'mape' or 'cosine'")

        is_dict_rows = base_table and isinstance(base_table[0], dict)
        if is_dict_rows:
            key_order = list(base_table[0].keys())
            flat_base = [float(row[k]) for row in base_table for k in key_order]
        else:
            flat_base = [float(v) for row in base_table for v in row]
        n = len(flat_base)
        if n == 0:
            return []

        if similarity == 1.0:
            if base_table and isinstance(base_table[0], dict):
                key_order = list(base_table[0].keys())
                return [[row[k] for k in key_order] for row in base_table]
            return [list(row) for row in base_table]

        def _reshape(flat):
            out = []
            idx = 0
            for row in base_table:
                if is_dict_rows:
                    out.append(flat[idx:idx + len(key_order)])
                    idx += len(key_order)
                else:
                    out.append(flat[idx:idx + len(row)])
                    idx += len(row)
            return out

        def _mape(base, cand):
            total = 0.0
            count = 0
            for b, c in zip(base, cand):
                if b == 0.0:
                    continue
                total += abs((c - b) / b)
                count += 1
            return total / count if count else 0.0

        def _dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        def _norm(a):
            return math.sqrt(sum(x * x for x in a))

        if metric == "mape":
            target_mape = max(0.0, 1.0 - similarity)

            def _random_candidate():
                rel_errs = []
                for _ in range(n):
                    sign = -1.0 if random.random() < 0.5 else 1.0
                    rel_errs.append(sign * abs(random.gauss(0.0, 1.0)))

                mean_abs = sum(abs(r) for r in rel_errs) / n
                scale = target_mape / mean_abs if mean_abs > 0.0 else 0.0
                rel_errs = [r * scale for r in rel_errs]

                # Keep values positive by limiting negative error.
                rel_errs = [max(r, -0.95) for r in rel_errs]

                candidate = [max(min_value, b * (1.0 + r)) for b, r in zip(flat_base, rel_errs)]
                return candidate

            for _ in range(max_tries):
                candidate = _random_candidate()
                mape = _mape(flat_base, candidate)
                if abs(mape - target_mape) <= 1e-3:
                    return _reshape(candidate)

        if metric == "cosine":
            base_norm = _norm(flat_base)
            if base_norm == 0.0:
                rand_flat = [max(min_value, random.random()) for _ in range(n)]
                return _reshape(rand_flat)

            mean_base = sum(flat_base) / n
            for _ in range(max_tries):
                rand_flat = [max(min_value, random.random()) for _ in range(n)]
                mean_rand = sum(rand_flat) / n
                scale = mean_base / mean_rand if mean_rand > 0.0 else 1.0
                rand_flat = [max(min_value, v * scale) for v in rand_flat]

                bb = _dot(flat_base, flat_base)
                br = _dot(flat_base, rand_flat)
                rr = _dot(rand_flat, rand_flat)

                k0 = bb * bb * (1.0 - similarity * similarity)
                k1 = 2.0 * bb * br * (1.0 - similarity * similarity)
                k2 = br * br - (similarity * similarity) * bb * rr

                if abs(k2) < 1e-12:
                    if abs(k1) < 1e-12:
                        continue
                    alpha = -k0 / k1
                else:
                    disc = k1 * k1 - 4.0 * k2 * k0
                    if disc < 0.0:
                        continue
                    sqrt_disc = math.sqrt(disc)
                    a1 = (-k1 + sqrt_disc) / (2.0 * k2)
                    a2 = (-k1 - sqrt_disc) / (2.0 * k2)
                    alpha = max(a1, a2)

                if alpha <= 0.0:
                    continue

                candidate = [b + alpha * r for b, r in zip(flat_base, rand_flat)]
                if min(candidate) <= 0.0:
                    continue

                cand_norm = _norm(candidate)
                if cand_norm == 0.0:
                    continue
                cos_sim = _dot(flat_base, candidate) / (base_norm * cand_norm)

                if abs(cos_sim - similarity) <= 1e-3:
                    return _reshape(candidate)

        # Fallback: return a positive random table with the same mean scale.
        mean_base = sum(flat_base) / n
        rand_flat = [max(min_value, random.random()) for _ in range(n)]
        mean_rand = sum(rand_flat) / n
        scale = mean_base / mean_rand if mean_rand > 0.0 else 1.0
        rand_flat = [max(min_value, v * scale) for v in rand_flat]
        return _reshape(rand_flat)
    
    def _save_window_log_plot(self):
        # Save actual utilization and DMR over time as a PNG.
        if not self.window_log:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning("Plot skipped: %s", exc)
            return

        window_ids = sorted(self.window_log.keys())
        fallback_windows = [w for w in window_ids if self.window_log[w].get("safe_fallback")]
        act_utils = [self.window_log[w]["act_util"] for w in window_ids]
        ref_utils = [self.window_log[w].get("ref_util", self.ref_util) for w in window_ids]
        dmrs = [self.window_log[w]["dmr"] for w in window_ids]
        values = [self.window_log[w].get("value", 0.0) for w in window_ids]
        rewards = [self.window_log[w].get("reward", 0.0) for w in window_ids]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        ax1.plot(window_ids, act_utils, linewidth=1)
        ax1.plot(window_ids, ref_utils, color="tab:orange", linestyle="--", linewidth=1)
        ax1.set_ylabel("Actual Util")
        ax1.set_ylim(bottom=0.0)
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2.plot(window_ids, dmrs, color="tab:red", linewidth=1)
        ax2.set_ylabel("DMR")
        ax2.set_ylim(bottom=0.0)
        ax2.grid(True, linestyle="--", alpha=0.4)

        ax3.plot(window_ids, values, color="tab:green", linewidth=1)
        ax3.set_ylabel("Value")
        ax3.set_ylim(bottom=0.0)
        ax3.grid(True, linestyle="--", alpha=0.4)

        ax4.plot(window_ids, rewards, color="tab:purple", linewidth=1)
        ax4.set_ylabel("Reward")
        ax4.set_xlabel("Window ID")
        ax4.set_ylim(bottom=-5.0)
        ax4.grid(True, linestyle="--", alpha=0.4)

        env_change_windows = []
        for t in self.env_change_times:
            w_mark = None
            for w in window_ids:
                if self.window_log[w]["w_range"][1] >= t:
                    w_mark = w
                    break
            if w_mark is not None:
                env_change_windows.append(w_mark)

        for w in env_change_windows:
            for ax in (ax1, ax2, ax3, ax4):
                ax.axvline(w, color="0.3", linestyle="--", linewidth=0.8, alpha=0.5)

        for w in fallback_windows:
            ax1.axvline(w, color="tab:red", linestyle=":", linewidth=1.0, alpha=0.8)

        os.makedirs(os.path.dirname(self.window_log_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.window_log_path, dpi=150)
        plt.close(fig)
        self._save_summary_metrics()

    def _rcrl_summary(self, window_ids):
        """Safety-guard telemetry, including the RCRL Theorem-1 check.

        ``realized_miss_rate = (1/K) Σ ℓ_k`` is the quantity Theorem 1 bounds by
        ``α + (λ₁ − r_min + γB)/(γK)`` (risk framing, Eq. 12). The loss ``ℓ_k`` matches the
        shield's ``loss_mode``: the binary indicator ``1[dmr>0]``, the fractional per-window
        miss ratio ``dmr``, or the releasable-normalized ratio. All lie in [0,1].
        """
        guard = getattr(self, "_safety_guard", None)
        if guard is None or not window_ids:
            return None

        loss_mode = getattr(self, "_rl_loss_mode", "binary")

        def _loss(w):
            entry = self.window_log[w]
            if loss_mode == "releasable":
                denom = entry.get("n_job_releasable", 0)
                if denom > 0:
                    return min(1.0, float(entry.get("n_miss_jobs", 0)) / denom)
                return 1 if entry.get("dmr", 0.0) > 0.0 else 0
            d = entry.get("dmr", 0.0)
            return float(d) if loss_mode == "fraction" else (1 if d > 0.0 else 0)

        losses = [_loss(w) for w in window_ids]
        miss_ind = [1 if self.window_log[w].get("dmr", 0.0) > 0.0 else 0 for w in window_ids]
        lam_traj = [self.window_log[w].get("rcrl_lambda") for w in window_ids]
        fallbacks = [bool(self.window_log[w].get("safe_fallback")) for w in window_ids]
        total = len(losses)

        # The guarantee only covers SHIELDED windows. If a warmup ran unshielded
        # (fallback_start_window > 0), measure the realized miss rate over the shielded
        # subset; the overall (incl. warmup) is reported separately.
        shielded = [w for w in window_ids if self.window_log[w].get("shield_active")]
        guar_ids = shielded if (shielded and len(shielded) < total) else window_ids
        guar_losses = [_loss(w) for w in guar_ids]
        realized = sum(guar_losses) / len(guar_ids) if guar_ids else 0.0

        # Carry-in loss / window-sizing telemetry (paper Eq. 9, §VII-C). ε_carry,k must stay
        # below α for the Lemma-1 precondition; report its mean/max and the realized window sizes.
        eps_vals = [self.window_log[w].get("eps_carry") for w in window_ids]
        eps_vals = [e for e in eps_vals if e is not None and e != float("inf")]
        w_vals = [self.window_log[w].get("w_sz") for w in window_ids]
        w_vals = [x for x in w_vals if x is not None]

        block = {
            "mode": guard.mode,
            "loss_mode": loss_mode,
            "total_windows": total,
            "shielded_windows": len(guar_ids),
            "warmup_windows": total - len(guar_ids),
            "realized_miss_rate": realized,                 # over shielded windows
            "overall_miss_rate": (sum(losses) / total) if total else 0.0,
            "num_miss_windows": sum(miss_ind),
            "num_fallback_windows": sum(1 for f in fallbacks if f),
            "fallback_rate": (sum(1 for f in fallbacks if f) / total) if total else 0.0,
            "miss_indicator_trajectory": miss_ind,
            "threshold_trajectory": lam_traj,
        }

        alpha = getattr(guard, "alpha", None)
        if alpha is not None:
            block["target_alpha"] = alpha
        if eps_vals:
            block["eps_carry_avg"] = sum(eps_vals) / len(eps_vals)
            block["eps_carry_max"] = max(eps_vals)
            # Lemma-1 precondition: the guarantee needs α > ε_carry in every window.
            if alpha is not None:
                block["carry_in_precondition_ok"] = bool(max(eps_vals) < alpha)
        if w_vals:
            block["w_sz_min"] = min(w_vals)
            block["w_sz_max"] = max(w_vals)
            block["adaptive_window"] = bool(getattr(self, "_adaptive_window_enabled", False))
        if guard.mode == "rcrl":
            block["gamma"] = guard.gamma
            block["rmin"] = guard.rmin
            block["lambda_init"] = guard.lambda_init
            block["lambda_final"] = guard.lam
            lam_vals = [x for x in lam_traj if x is not None]
            if lam_vals:
                block["lambda_min"] = min(lam_vals)
                block["lambda_max"] = max(lam_vals)
            bound = guard.theorem_bound(len(guar_ids))  # horizon = shielded windows
            block["theorem_bound"] = bound
            block["guarantee_satisfied"] = (realized <= bound) if bound is not None else None
        return block

    def _save_summary_metrics(self):
        # Drain any release-lagged windows still pending so their DMR / λ update / recorded row
        # are finalized before the summary and the final flush (no-op unless RL + release mode).
        if hasattr(self, "_rcrl_drain_pending"):
            try:
                self._rcrl_drain_pending()
            except Exception as exc:
                logger.warning("RCRL drain at run end failed: %s", exc)
        window_ids = sorted(self.window_log.keys())
        if not window_ids:
            return

        def _mean(xs):
            return sum(xs) / len(xs) if xs else 0.0

        def _std(xs):
            if not xs:
                return 0.0
            m = _mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

        def _metrics_for(ids):
            act_utils = [self.window_log[w]["act_util"] for w in ids]
            dmrs = [self.window_log[w]["dmr"] for w in ids]
            values = [self.window_log[w].get("value", 0.0) for w in ids]
            rewards = [self.window_log[w].get("reward", 0.0) for w in ids]
            avg_qos = [self.window_log[w].get("avg_qos", 0.0) for w in ids]
            ref_utils = [self.window_log[w].get("ref_util", self.ref_util) for w in ids]
            return {
                "total_windows": len(ids),
                "avg_act_util": _mean(act_utils),
                "std_act_util": _std(act_utils),
                "avg_dmr": _mean(dmrs),
                "max_dmr": max(dmrs) if dmrs else 0.0,
                "avg_value": _mean(values),
                "avg_reward": _mean(rewards),
                "avg_qos": _mean(avg_qos),
                "start_ref_util": ref_utils[0] if ref_utils else 0.0,
                "end_ref_util": ref_utils[-1] if ref_utils else 0.0,
            }

        def _settle_time(ids, window_count=5, rel_tol=0.05, abs_tol=0.02):
            if not ids:
                return {
                    "settle_time_ms": None,
                    "settle_window_end": None,
                    "post_settle_util_avg": None,
                    "post_settle_util_std": None,
                    "post_settle_util_min": None,
                    "post_settle_util_max": None,
                }
            start_time = self.window_log[ids[0]]["w_range"][0]
            consecutive = 0
            for w in ids:
                act_util = self.window_log[w]["act_util"]
                ref_util = self.window_log[w].get("ref_util", self.ref_util)
                tol = rel_tol * ref_util if ref_util > 0.0 else abs_tol
                if abs(act_util - ref_util) <= tol:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= window_count:
                    end_time = self.window_log[w]["w_range"][1]
                    tail_ids = ids[ids.index(w):]
                    tail_utils = [self.window_log[t]["act_util"] for t in tail_ids]
                    util_avg = _mean(tail_utils) if tail_utils else None
                    util_std = _std(tail_utils) if tail_utils else None
                    util_min = min(tail_utils) if tail_utils else None
                    util_max = max(tail_utils) if tail_utils else None
                    return {
                        "settle_time_ms": end_time - start_time,
                        "settle_window_end": w,
                        "post_settle_util_avg": util_avg,
                        "post_settle_util_std": util_std,
                        "post_settle_util_min": util_min,
                        "post_settle_util_max": util_max,
                    }
            return {
                "settle_time_ms": None,
                "settle_window_end": None,
                "post_settle_util_avg": None,
                "post_settle_util_std": None,
                "post_settle_util_min": None,
                "post_settle_util_max": None,
            }

        summary = {
            "overall": _metrics_for(window_ids),
            "num_env_changes": len(self.env_change_times),
            "final_epsilon": float(getattr(self.rl_agent, "epsilon", 0.0)),
            "ctrl_mode": str(getattr(self, "ctrl_mode", "")),
            "settle": _settle_time(window_ids),
            "window_job_counts": [
                self.window_log[w].get("n_jobs", 0) for w in window_ids
            ],
            "window_miss_job_counts": [
                self.window_log[w].get("n_miss_jobs", 0) for w in window_ids
            ],
            "miss_window_ids": [
                w for w in window_ids if self.window_log[w].get("dmr", 0.0) > 0.0
            ],
            "miss_dmrs": [
                self.window_log[w]["dmr"]
                for w in window_ids
                if self.window_log[w].get("dmr", 0.0) > 0.0
            ],
        }

        # Average QoS level assigned to CONTROLLABLE tasks only (t_type == 1), read from
        # each task's per-window log. The `overall.avg_qos` above is a per-job average over
        # ALL tasks (uncontrollable/background sit at QoS 0), so it understates this.
        def _avg_qos_over(ids, tasks):
            vals = []
            for t in tasks:
                wl = getattr(t._task_info, "win_log", {})
                for w in ids:
                    e = wl.get(w)
                    if e is not None:
                        vals.append(e.get("qos_level", 0))
            return (sum(vals) / len(vals)) if vals else None

        ctrl_tasks = [t for t in self.tasks if getattr(t._task_info, "t_type", 1) == 1]
        summary["num_controllable_tasks"] = len(ctrl_tasks)
        summary["avg_qos_controllable"] = _avg_qos_over(window_ids, ctrl_tasks)
        settle_end = summary["settle"].get("settle_window_end")
        if settle_end is not None:
            post_ids = [w for w in window_ids if w >= settle_end]
            summary["avg_qos_controllable_post_settle"] = _avg_qos_over(post_ids, ctrl_tasks)
        else:
            summary["avg_qos_controllable_post_settle"] = None

        rcrl = self._rcrl_summary(window_ids)
        if rcrl is not None:
            summary["rcrl"] = rcrl

        change_log = getattr(self, "change_log", None)
        if change_log:
            summary["dynamic_changes"] = [{"window": w, "desc": d} for w, d in change_log]

        pred_scores = getattr(self, "rl_pred_scores", None)
        pred_labels = getattr(self, "rl_pred_labels", None)
        if pred_scores and pred_labels and len(pred_scores) == len(pred_labels):
            order = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i])
            y_true = [pred_labels[i] for i in order]
            n_pos = sum(1 for v in y_true if v == 1)
            n_neg = len(y_true) - n_pos
            auc = None
            if n_pos > 0 and n_neg > 0:
                rank_sum = 0.0
                for rank, v in enumerate(y_true, start=1):
                    if v == 1:
                        rank_sum += rank
                auc = (rank_sum - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)
            preds = [1 if s >= 0.5 else 0 for s in pred_scores]
            correct = sum(1 for p, y in zip(preds, pred_labels) if p == y)
            acc = correct / len(pred_labels) if pred_labels else 0.0
            summary["rl_pred_acc"] = acc
            summary["rl_pred_auc"] = auc

        dynamic_mode = int(getattr(self, "dynamic_mode", 0))
        if dynamic_mode and self.env_change_times:
            end_times = [self.window_log[w]["w_range"][1] for w in window_ids]
            change_times = sorted(self.env_change_times)
            boundary_indices = []
            for t in change_times:
                idx = None
                for i, end_t in enumerate(end_times):
                    if end_t >= t:
                        idx = i
                        break
                if idx is None:
                    idx = len(window_ids)
                boundary_indices.append(idx)
            boundary_indices = sorted(set(i for i in boundary_indices if i > 0))

            segments = []
            start_idx = 0
            for end_idx in boundary_indices + [len(window_ids)]:
                if end_idx <= start_idx:
                    continue
                ids = window_ids[start_idx:end_idx]
                start_time = self.window_log[ids[0]]["w_range"][0]
                end_time = self.window_log[ids[-1]]["w_range"][1]
                segments.append(
                    {
                        "window_start": ids[0],
                        "window_end": ids[-1],
                        "time_start": start_time,
                        "time_end": end_time,
                        "settle": _settle_time(ids),
                    }
                )
                start_idx = end_idx
            summary["settle_by_segment"] = segments
        
        os.makedirs(os.path.dirname(self.window_log_path) or ".", exist_ok=True)
    
        base = os.path.splitext(os.path.basename(self.window_log_path))[0]
        profile_samples = getattr(self, "profile_samples", None)
        if profile_samples:
            dataset_dir = os.path.join(os.getcwd(), "dataset", "profile")
            os.makedirs(dataset_dir, exist_ok=True)
            profile_seed = None
            ctrl_cfg = getattr(self, "data", {}).get("ctrl", {})
            if isinstance(ctrl_cfg, dict):
                profile_seed = ctrl_cfg.get("profile_seed")
            seed_tag = f"_seed{profile_seed}" if profile_seed is not None else ""
            profile_name = f"{base}_profile{seed_tag}.pkl"
            profile_path = os.path.join(dataset_dir, profile_name)
            with open(profile_path, "wb") as f:
                pickle.dump(profile_samples, f)
            summary["profile_samples_path"] = os.path.join("dataset", "profile", profile_name)
            summary["profile_samples_count"] = len(profile_samples)

        ctrl_cfg = getattr(self, "data", {}).get("ctrl", {})
        if isinstance(ctrl_cfg, dict) and ctrl_cfg.get("save_rl_samples", True):
            rl_samples = getattr(self, "rl_samples", None)
            if rl_samples:
                dataset_dir = os.path.join(os.getcwd(), "dataset", "rl")
                os.makedirs(dataset_dir, exist_ok=True)
                rl_name = f"{base}_rl_profile.pkl"
                rl_path = os.path.join(dataset_dir, rl_name)
                with open(rl_path, "wb") as f:
                    pickle.dump(rl_samples, f)
                summary["rl_samples_path"] = os.path.join("dataset", "rl", rl_name)
                summary["rl_samples_count"] = len(rl_samples)


        if getattr(self, "realized_similarity", None) is not None:
            summary["requested_similarity"] = getattr(self, "requested_similarity", None)
            summary["realized_similarity"] = self.realized_similarity

        # Legacy artifact: single fat JSON at the log-dir root (kept for back-compat with
        # existing plotting/collation scripts). The authoritative, self-describing output
        # is the per-run directory written by the recorder below.
        summary_name = f"{base}_summary.json"
        summary_path = os.path.join(self.window_log_dir, summary_name)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Per-run recorder: flush any trailing windows, then write a slim scalar-only
        # summary alongside meta.json + windows.csv. The per-window arrays are dropped
        # here because windows.csv is now their single source of truth.
        recorder = getattr(self, "recorder", None)
        if recorder is not None:
            try:
                self._recorder_flush_ready(window_ids[-1])
                slim = dict(summary)
                for k in ("window_job_counts", "window_miss_job_counts",
                          "miss_window_ids", "miss_dmrs"):
                    slim.pop(k, None)
                if isinstance(slim.get("rcrl"), dict):
                    slim["rcrl"] = {
                        k: v for k, v in slim["rcrl"].items()
                        if k not in ("miss_indicator_trajectory", "threshold_trajectory")
                    }
                slim["windows_csv"] = "windows.csv"
                recorder.write_summary(slim)
            except Exception as exc:
                logger.warning("Recorder summary write failed: %s", exc)
            finally:
                recorder.close()

    def gen_utils_qos(self, utils, qos_level, qos_policy):
        # Build per-task utilization tables; result is indexed as tasks[task_idx][qos].
        if qos_level <= 1:
            return [[u] for u in utils]

        def linear(u):
            # Evenly spaced utilizations up to the task's max util.
            step = u / (qos_level - 1)
            return [step * i for i in range(1, qos_level)]

        def geometric(u):
            # Geometric growth ending at u (ratio=2).
            ratio = 2.0
            u_min = u / (ratio ** (qos_level - 2))
            return [u_min * (ratio ** i) for i in range(qos_level - 1)]

        def random_levels(u):
            # Random, sorted levels in (0, u], with u as the max.
            samples = [random.uniform(0.0, u) for _ in range(qos_level - 2)]
            return sorted(samples + [u])

        if qos_policy == "linear":
            per_task = [linear(u) for u in utils]
        elif qos_policy == "geometric":
            per_task = [geometric(u) for u in utils]
        elif qos_policy == "random":
            per_task = [random_levels(u) for u in utils]
        else:
            raise ValueError(f"Unknown qos_policy: {qos_policy}")

        # Prepend QoS=0 utilization for each task.
        return [[0.0] + levels for levels in per_task]
