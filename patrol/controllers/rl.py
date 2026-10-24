import logging


logger = logging.getLogger(__name__)


class RlControllerMixin:
    _RL_LOG_COLORS = {
        "pred": "\x1b[33m",
        "finetune": "\x1b[35m",
        "safe_calib": "\x1b[36m",
        "safe_fallback": "\x1b[31m",
        "rebuild": "\x1b[34m",
    }

    def _rl_log(self, tag, msg, *args, color=None):
        prefix = f"[RL][{tag}]"
        if color:
            color_code = self._RL_LOG_COLORS.get(color, "")
            reset = "\x1b[0m" if color_code else ""
            logger.info(f"{color_code}{prefix} {msg}{reset}", *args)
            return
        logger.info(f"{prefix} {msg}", *args)

    def predict_miss_from_state(self, x_state, model_path=None):
        if not hasattr(self, "_profile_predict_miss"):
            raise RuntimeError("Profile prediction interface not available.")
        return self._profile_predict_miss(x_state, model_path=model_path)

    def _rl_predict_miss_probs_batch(self, x_states, model_path=None):
        if not x_states:
            return []
        if not hasattr(self, "_profile_load_model"):
            return [self.predict_miss_from_state(x, model_path=model_path) for x in x_states]

        if model_path is None:
            model_path = self.data.get("ctrl", {}).get("profile_model_path")
        if not model_path:
            return [self.predict_miss_from_state(x, model_path=model_path) for x in x_states]

        self._profile_load_model(model_path)
        model = self.profile_model["model"]
        device = self.profile_model.get("device")
        mean = self.profile_model.get("mean")
        std = self.profile_model.get("std")

        import numpy as np
        import torch

        x = np.array(x_states, dtype=np.float32)
        if mean is not None and std is not None:
            x = (x - mean) / std
        x_t = torch.from_numpy(x)
        if device is not None:
            x_t = x_t.to(device)
            model = model.to(device)
        temp = float(self.profile_model.get("temperature", 1.0)) or 1.0
        with torch.no_grad():
            probs = torch.sigmoid(model(x_t) / temp).detach().cpu().numpy().tolist()
        probs = [float(p) for p in probs]
        # Zero-load candidates (est_util == 0) cannot miss — hard floor (belt-and-suspenders).
        if hasattr(self, "_miss_floor"):
            probs = [0.0 if self._miss_floor(xs) else p for xs, p in zip(x_states, probs)]
        return probs

    def _rl_build_profile_state(self, prev_win_id, qos_list):
        if hasattr(self, "_profile_build_state"):
            return self._profile_build_state(prev_win_id, qos_list)

        state = [self.w_sz]
        if prev_win_id not in self.window_log:
            return state + [0.0] * (2 * len(self.tasks))

        prev_w_range = self.window_log[prev_win_id]["w_range"]
        prev_w_sz = prev_w_range[1] - prev_w_range[0]

        for idx, t in enumerate(self.tasks):
            tinfo = t._task_info
            tlog = tinfo.win_log.get(prev_win_id, {})
            n_jobs = tlog.get("n_jobs", 0)
            if n_jobs > 0:
                period_est = prev_w_sz / n_jobs
            else:
                period_est = 0.0
                for back_id in range(prev_win_id - 1, -1, -1):
                    back_log = tinfo.win_log.get(back_id, {})
                    back_jobs = back_log.get("n_jobs", 0)
                    if back_jobs > 0:
                        back_range = self.window_log[back_id]["w_range"]
                        back_w_sz = back_range[1] - back_range[0]
                        period_est = back_w_sz / back_jobs
                        break

            qos_level = qos_list[idx] if idx < len(qos_list) else tinfo.cur_qos
            wcet_table = getattr(self, "tasks_wcet_tables", None)
            if wcet_table is not None and idx < len(wcet_table) and qos_level < len(wcet_table[idx]):
                est = float(wcet_table[idx][qos_level])
            else:
                est = 0.0
            state.extend([period_est, est])

        return state

    def rl_shielding_predictor(self, action, *, model_path=None, threshold=None):
        prev_win_id = self.w_id - 1
        x_state = self._rl_build_profile_state(prev_win_id, action)
        prob = self.predict_miss_from_state(x_state, model_path=model_path)
        if threshold is None and hasattr(self, "_profile_get_threshold"):
            threshold = self._profile_get_threshold()
        if threshold is None:
            threshold = 0.5
        pred = 1 if prob >= threshold else 0
        return pred, prob, x_state

    # ---- risk estimator (paper §V-B): learned classifier OR analytical EDF utilization ------
    #      Risk score r: LOWER = safer, r_min = 0. Accept iff r <= lambda.

    def _rl_profile_table(self):
        """Execution-time table used as C_i(q) by the analytical risk estimator.

        'valid'  -> ground-truth WCET bounds (Case 1); 'invalid' -> estimated/possibly
        perturbed profile (Case 2). Falls back to whichever table is available.
        """
        if getattr(self, "_safety_profile", "invalid") == "valid":
            return getattr(self, "tasks_wcet_tables", None) or self.tasks_estimated_tables
        # invalid: optionally synthesize a controlled underestimate of the true WCET.
        underest = getattr(self, "_safety_profile_underest", 0.0) or 0.0
        if underest > 0.0:
            src = getattr(self, "tasks_wcet_tables", None) or self.tasks_estimated_tables
            return [[float(c) * (1.0 - underest) for c in row] for row in src]
        return getattr(self, "tasks_estimated_tables", None) or self.tasks_wcet_tables

    def _rl_analytical_util(self, action, table):
        """Total EDF utilization Σ C_i(q_i)/T_i of a candidate under the given profile table."""
        u = 0.0
        for t_id, t in enumerate(self.tasks):
            tinfo = t._task_info
            if not getattr(tinfo, "active", True):
                continue
            q = int(action[t_id]) if t_id < len(action) else tinfo.cur_qos
            row = table[t_id] if (table is not None and t_id < len(table)) else None
            if not row:
                continue
            c = float(row[q]) if q < len(row) else float(row[-1])
            if tinfo.period > 0:
                u += c / tinfo.period
        return u

    def _rl_candidate_risks(self, actions, x_states=None):
        """Risk scores r (LOWER = safer, r_min = 0) for a list of candidate decisions, plus the
        per-candidate miss probabilities (None for the analytical estimator).

        analytical: r = Σ C_i(q_i)/T_i (EDF total utilization, Eq. 4; analytical threshold 1).
        learned:    r = miss_prob (classifier), batched over candidates.
        """
        if getattr(self, "_safety_score_mode", "learned") == "analytical":
            table = self._rl_profile_table()
            risks = [self._rl_analytical_util(a, table) for a in actions]
            return risks, [None] * len(actions)
        if x_states is None:
            prev = self.w_id - 1
            x_states = [self._rl_build_profile_state(prev, a) for a in actions]
        probs = self._rl_predict_miss_probs_batch(x_states)
        risks = [float(p) for p in probs]
        return risks, probs

    def _rl_risk_and_prob(self, action):
        """Risk score (and miss prob, None for analytical) for a single candidate."""
        risks, probs = self._rl_candidate_risks([action])
        return risks[0], probs[0]

    def _rl_safety(self):
        """Lazily build and cache the safety guard from ``controller.safety`` config."""
        guard = getattr(self, "_safety_guard", None)
        if guard is None:
            from patrol.controllers.safety import build_safety_guard

            ctrl = self.data.get("ctrl", {})
            threshold_fn = (
                self._profile_get_threshold
                if hasattr(self, "_profile_get_threshold")
                else (lambda: 0.5)
            )
            guard = build_safety_guard(ctrl, threshold_fn=threshold_fn)
            self._safety_guard = guard
            self._rl_log("Safety", "guard mode=%s %s", guard.mode, guard.status(), color="safe_calib")
        return guard

    def _rl_build_grouped_fallback_candidates(self, action):
        cur = [max(0, int(a)) for a in action]
        candidates = [list(cur)]
        while any(v > 0 for v in cur):
            cur = [max(0, v - 1) for v in cur]
            candidates.append(list(cur))
        return candidates

    def _rl_save_sample(self, pending, miss):
        rl_samples = getattr(self, "rl_samples", None)
        if rl_samples is None:
            rl_samples = []
            self.rl_samples = rl_samples
        rl_samples.append(
            {
                "x_state": pending.get("x_state"),
                "action": pending.get("action"),
                "y_miss": miss,
            }
        )

    def _rl_update_finetune_buffer(self, pending, miss):
        finetune_buf = getattr(self, "rl_finetune_buffer", None)
        if finetune_buf is None:
            finetune_buf = []
            self.rl_finetune_buffer = finetune_buf
        if pending.get("x_state") is not None:
            finetune_buf.append(
                {
                    "x_state": pending.get("x_state"),
                    "y_miss": miss,
                }
        )
        return finetune_buf

    def _rl_update_pred_metrics(self, miss_prob, miss, prev_win_id):
        pred_scores = getattr(self, "rl_pred_scores", None)
        if pred_scores is None:
            pred_scores = []
            self.rl_pred_scores = pred_scores
        pred_labels = getattr(self, "rl_pred_labels", None)
        if pred_labels is None:
            pred_labels = []
            self.rl_pred_labels = pred_labels
        pred_scores.append(float(miss_prob))
        pred_labels.append(miss)

        total = len(pred_labels)
        threshold = self._profile_get_threshold() if hasattr(self, "_profile_get_threshold") else 0.5
        preds = [1 if s >= threshold else 0 for s in pred_scores]
        correct = sum(1 for p, y in zip(preds, pred_labels) if p == y)
        acc = correct / total if total else 0.0

        order = sorted(range(total), key=lambda i: pred_scores[i])
        y_true = [pred_labels[i] for i in order]
        n_pos = sum(1 for v in y_true if v == 1)
        n_neg = total - n_pos
        auc = None
        if n_pos > 0 and n_neg > 0:
            rank_sum = 0.0
            for rank, v in enumerate(y_true, start=1):
                if v == 1:
                    rank_sum += rank
            auc = (rank_sum - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)

        self._rl_log(
            "Pred",
            "[Window %s] acc=%.4f auc=%s last_pred=%s miss_prob=%.4f last_true=%s",
            prev_win_id,
            acc,
            f"{auc:.4f}" if auc is not None else "nan",
            1 if miss_prob >= threshold else 0,
            miss_prob,
            miss,
            color="pred",
        )
        return acc

    def _rl_maybe_finetune(self, acc, finetune_buf, prev_win_id):
        acc_thr = self.data.get("ctrl", {}).get("profile_finetune_acc_thr", 0.8)
        buf_size = self.data.get("ctrl", {}).get("profile_finetune_buffer", 500)
        cooldown = getattr(self, "rl_finetune_cooldown", 0)
        if cooldown > 0:
            self.rl_finetune_cooldown = cooldown - 1
            return
        if acc >= acc_thr or not finetune_buf or prev_win_id < 100:
            return

        batch_size = self.data.get("ctrl", {}).get("profile_finetune_batch", 128)
        epochs = self.data.get("ctrl", {}).get("profile_finetune_epochs", 50)
        lr = self.data.get("ctrl", {}).get("profile_finetune_lr", 1e-4)
        recent = finetune_buf[-buf_size:] if buf_size else list(finetune_buf)
        try:
            finetune_acc = self._profile_finetune(
                recent,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            self._rl_log(
                "Finetune",
                "done trigger_acc=%.4f buf=%s epochs=%s val_acc=%.4f",
                acc,
                len(recent),
                epochs,
                finetune_acc if finetune_acc is not None else -1.0,
                color="finetune",
            )
            self.rl_pred_scores = []
            self.rl_pred_labels = []
            self.rl_finetune_cooldown = self.data.get("ctrl", {}).get(
                "profile_finetune_cooldown", 50
            )
            # Let the safety guard recalibrate against the refined classifier (no-op for RCRL).
            extra_miss_samples = [
                s for s in (getattr(self, "rl_samples", None) or [])
                if s.get("y_miss") == 1 and s.get("x_state") is not None
            ]
            self._rl_safety().on_classifier_refined(
                self._rl_predict_miss_probs_batch, extra_miss_samples=extra_miss_samples
            )
        except Exception as exc:
            logger.warning("RL finetune failed: %s", exc)

    def _rl_prepare_pending(self, prev_win_id, action):
        applied_action = list(action)
        x_state = self._rl_build_profile_state(prev_win_id, applied_action)
        self.rl_action_pending = {
            "win_id": self.w_id,
            "action": list(applied_action),
            "x_state": x_state,
        }
        # Register in the per-window pending map so the outcome can be finalized with a lag
        # (release-mode DMR). Same dict object -> later key writes below update the stored entry.
        if not hasattr(self, "_rl_pending_by_win"):
            self._rl_pending_by_win = {}
        self._rl_pending_by_win[self.w_id] = self.rl_action_pending

        guard = self._rl_safety()
        # Snapshot the threshold that governs this window's validation (telemetry: RCRL λ_t).
        self.rl_action_pending["guard_threshold"] = guard.threshold_value()

        # Analytic EDF util-<=-1 baseline: bypass the learned classifier + fallback (Section 5.3).
        if guard.mode == "rt_analytic":
            adjusted = self.rt_shielding(list(applied_action))
            changed = list(adjusted) != list(applied_action)
            if changed:
                self.rl_action_pending["action"] = list(adjusted)
                self.rl_action_pending["x_state"] = self._rl_build_profile_state(prev_win_id, adjusted)
                self.rl_action_pending["fallback_from_action"] = list(action)
                for t_id, t in enumerate(self.tasks):
                    t._task_info.cur_qos = adjusted[t_id]
                self.rl_action_pending["safe_fallback"] = True
            return

        try:
            # Validate on the risk score r (learned: miss_prob; analytical: total utilization U).
            risk, miss_prob = self._rl_risk_and_prob(applied_action)
            self.rl_action_pending["risk"] = risk
            if miss_prob is not None:
                self.rl_action_pending["miss_prob"] = miss_prob
            fallback_enabled = (
                guard.fallback_enabled and self.w_id >= int(guard.fallback_start_window)
            )
            if fallback_enabled and guard.needs_fallback(risk):
                fb_action, fb_risk, fb_prob, fb_state = self._rl_safe_fallback(
                    applied_action, risk
                )
                if fb_action is not None:
                    applied_action = fb_action
                    self.rl_action_pending["action"] = list(fb_action)
                    self.rl_action_pending["x_state"] = fb_state
                    self.rl_action_pending["risk"] = fb_risk
                    if fb_prob is not None:
                        self.rl_action_pending["miss_prob"] = fb_prob
                    self.rl_action_pending["fallback_from_action"] = list(action)
                    for t_id, t in enumerate(self.tasks):
                        t._task_info.cur_qos = fb_action[t_id]
                self.rl_action_pending["safe_fallback"] = True
        except Exception as exc:
            logger.warning("RL prediction failed: %s", exc)

    def _rl_safe_fallback(self, action, orig_risk):
        """Grouped-reduction fallback (Algorithm 2).

        Among the grouped candidates that the guard accepts, pick the one with the
        largest total QoS. The all-zero decision ``d_safe`` (drop all jobs) is assigned a risk
        of ``-∞`` (Eq. 8) and is always admissible, so the validated set is never empty and a
        validated decision is always returned. Risk scores come from the active estimator
        (learned classifier r=miss_prob, or analytical EDF utilization r=U).
        Returns ``(action, risk, miss_prob, x_state)`` (miss_prob None under analytical).
        """
        prev_win_id = self.w_id - 1
        guard = self._rl_safety()
        candidates = self._rl_build_grouped_fallback_candidates(action)
        x_states = [self._rl_build_profile_state(prev_win_id, cand) for cand in candidates]
        risks, miss_probs = self._rl_candidate_risks(candidates, x_states=x_states)

        best_idx = None
        best_sum = -1
        for idx, (cand, cand_risk) in enumerate(zip(candidates, risks)):
            is_dsafe = (sum(cand) == 0)  # d_safe: risk -inf, always accepted (Eq. 8)
            if not (is_dsafe or guard.is_accepted(cand_risk)):
                continue
            cand_sum = sum(cand)
            if cand_sum > best_sum:
                best_sum = cand_sum
                best_idx = idx

        if best_idx is None:
            best_idx = len(candidates) - 1  # all-zero d_safe

        chosen_action = candidates[best_idx]
        chosen_risk = risks[best_idx]
        chosen_prob = miss_probs[best_idx]
        chosen_state = x_states[best_idx]
        self._rl_log(
            "SafeFallback",
            "orig_sum=%s orig_risk=%.4f | chosen_sum=%s chosen_risk=%.4f steps=%s %s",
            sum(action),
            orig_risk,
            sum(chosen_action),
            chosen_risk,
            len(candidates),
            guard.status(),
            color="safe_fallback",
        )
        return chosen_action, chosen_risk, chosen_prob, chosen_state

    def _apply_mode_change(self, prev_qos_list):
        """Apply the mode-change protocol to the validated decision now in ``cur_qos`` (d̂_k):
        a QoS decrease (or no change) takes effect immediately; every QoS increase is deferred
        until ALL carry-in jobs from the previous decision complete (paper §VI-A). A raise must
        not add demand while ANY previous-configuration job is still live — deferring only until
        the task's *own* carry-in finishes is unsafe, because another task's carry-in may still
        be running. Deferral only lowers the applied QoS transiently (never above the validated
        decision), so the acceptance SafetyGuard validated is preserved.
        """
        carryin = set(self._live_jobs())   # carry-in jobs live at this boundary
        pending = {}
        for t_id, t in enumerate(self.tasks):
            tinfo = t._task_info
            old_q = prev_qos_list[t_id] if t_id < len(prev_qos_list) else tinfo.cur_qos
            target = tinfo.cur_qos  # validated decision for this window
            if target > old_q and carryin:
                tinfo.cur_qos = old_q       # hold at old until all carry-ins drain
                pending[t_id] = target
            # decrease / unchanged / no carry-in present: keep target (already applied)
        self._mc_pending = pending
        self._mc_carryin = carryin if pending else set()

    def _rl_compute_reward(self):
        w = self.w_id - 1
        # Paper reward (Eq. 1): aggregate applied QoS, gated on a miss-free window.
        if getattr(self, "_rl_reward_mode", "v2") == "paper":
            qos_sum = 0
            for t in self.tasks:
                tinfo = t._task_info
                if not getattr(tinfo, "active", True):
                    continue
                qos_sum += tinfo.win_log.get(w, {}).get("qos_level", 0)
            reward, _ = self.rl_agent.compute_reward_qos_gated(
                qos_sum, dmr=self.window_log[w]["dmr"]
            )
            self.window_log[w]["reward"] = reward
            return reward

        last_win_val = self.window_log[self.w_id - 1]["value"] / (
            self.window_log[self.w_id - 1]["act_util"] * self.w_sz * (self.tasks[0]._task_info.max_qos) + 1e-6
        )
        reward_fn = (self.rl_agent.compute_reward_v2
                     if getattr(self, "_rl_reward_mode", "v2") == "v2"
                     else self.rl_agent.compute_reward_decoupled_sum)
        rkw = {}
        if getattr(self, "_rl_reward_mode", "v2") == "v2":
            rkw = {"w_u": getattr(self, "_rl_reward_wu", 1.0),
                   "w_v": getattr(self, "_rl_reward_wv", 1.0)}
        reward, _ = reward_fn(
            util=self.window_log[self.w_id - 1]["act_util"],
            value=last_win_val,
            U_ref=self.ref_util,
            dmr=self.window_log[self.w_id - 1]["dmr"],
            **rkw,
        )
        self.window_log[self.w_id - 1]["reward"] = reward
        return reward

    def _rl_budget_explore_policy(self, budget=None):
        """Budget-aware exploration using ONLY runtime-observed execution estimates --
        build_estimated_exec_tables(), the same source rt_shielding, the RL state, and the
        v2c classifier use. It never touches the ground-truth wcet_table (the true max-QoS
        demand is NOT known to the controller). Centre each ACTIVE task's QoS at the fraction
        of max that makes the ESTIMATED aggregate utilisation track ref_util, with a Gaussian
        spread. Cold start (no exec history yet -> estimates ~0): fall back to uniform.
        Inactive (dormant/removed) tasks stay at QoS 0."""
        import numpy as np
        b = float(budget if budget is not None else self.ref_util)
        max_qos = self.tasks[0]._task_info.max_qos
        est_tables = self.build_estimated_exec_tables()  # observed exec est per task/QoS
        umax = []
        for t_id, t in enumerate(self.tasks):
            ti = t._task_info
            if not getattr(ti, "active", True):
                umax.append(0.0)
                continue
            period = float(getattr(ti, "period", 0.0)) or 1.0
            est = est_tables[t_id]
            c_max = float(est[max_qos]) if max_qos < len(est) else 0.0
            umax.append(c_max / period)
        total = sum(umax)
        if total <= 1e-9:
            # No execution history observed yet: nothing to estimate the budget from.
            return self.rl_agent.random_explore_policy(b)
        frac = min(1.0, max(0.0, b / total))
        center = frac * max_qos
        spread = max(0.5, getattr(self, "_rl_explore_spread", 0.25) * max_qos)
        action = np.zeros(len(self.tasks), dtype=np.int64)
        for i in range(len(self.tasks)):
            if umax[i] <= 0.0:
                continue  # inactive -> QoS 0
            q = int(round(np.random.normal(center, spread)))
            action[i] = min(max_qos, max(0, q))
        return action

    def _rcrl_finalize_release_dmr(self, win_id):
        """For dmr_membership=release, rewrite the window's DMR from the release-window
        accumulators once (after the lag) all jobs released in it have terminated."""
        if getattr(self, "_dmr_membership", "terminate") != "release":
            return
        entry = self.window_log.get(win_id)
        if entry is None:
            return
        tot = self._rel_total.get(win_id, 0)
        miss_ct = self._rel_miss.get(win_id, 0)
        entry["n_jobs"] = tot
        entry["n_miss_jobs"] = miss_ct
        entry["dmr"] = (miss_ct / tot) if tot > 0 else 0

    def _rcrl_finalize_outcome(self, outcome_win, pending):
        """Apply one window's outcome: finalize its (possibly release-lagged) DMR, step the
        RCRL threshold λ on the realized loss, and update the miss-predictor. Used both online
        (lagged) and at run end to drain the trailing windows."""
        self._rcrl_finalize_release_dmr(outcome_win)  # no-op unless release mode
        entry = self.window_log[outcome_win]
        dmr = entry["dmr"]
        miss = 1 if dmr > 0 else 0
        # Loss the RCRL controller integrates (feeds the λ update and the realized-miss
        # accounting, so the quantity Theorem 1 bounds matches what the shield regulates):
        #   binary     -> 1{any miss}; fraction -> dmr (misses / actual jobs);
        #   releasable -> misses / max releasable jobs Σ⌈W/T_i⌉ (paper Eq. 3, ℓ_miss,k).
        # `miss` (binary) is kept separately for the miss-predictor label.
        loss_mode = getattr(self, "_rl_loss_mode", "binary")
        if loss_mode == "releasable":
            denom = entry.get("n_job_releasable", 0)
            n_miss = entry.get("n_miss_jobs", 0)
            ell = min(1.0, float(n_miss) / denom) if denom > 0 else float(miss)
        elif loss_mode == "fraction":
            ell = float(dmr)
        else:
            ell = float(miss)
        self.window_log[outcome_win]["safe_fallback"] = bool(pending.get("safe_fallback"))
        # Threshold that governed this window's validation (λ_t for RCRL).
        self.window_log[outcome_win]["rcrl_lambda"] = pending.get("guard_threshold")
        self.window_log[outcome_win]["risk_score"] = pending.get("risk")
        self._rl_save_sample(pending, miss)
        finetune_buf = self._rl_update_finetune_buffer(pending, miss)
        miss_prob = pending.get("miss_prob")
        risk = pending.get("risk")
        # Update the risk controller on the applied decision's outcome (Alg. 1 L13: threshold
        # step λ ← λ − γ(Mₖ − α); conformal calibration for the baseline). Only while the shield
        # is actually active — the controller must not integrate misses from windows it isn't
        # controlling (before fallback_start_window, or when the fallback is disabled), or λ
        # drifts uselessly.
        guard = self._rl_safety()
        shield_active = guard.fallback_enabled and outcome_win >= int(guard.fallback_start_window)
        self.window_log[outcome_win]["shield_active"] = bool(shield_active)
        if shield_active:
            guard.update_on_outcome(loss=ell, risk=risk, x_state=pending.get("x_state"))
            # Running realized loss over SHIELDED windows only — the quantity Theorem 1
            # bounds (binary miss rate, or mean fractional DMR under loss_mode=fraction).
            # Warmup (unshielded) windows are excluded: the guarantee only covers decisions
            # the shield actually controls.
            self._rcrl_loss_sum = getattr(self, "_rcrl_loss_sum", 0) + ell
            self._rcrl_win_count = getattr(self, "_rcrl_win_count", 0) + 1
            run_miss = self._rcrl_loss_sum / self._rcrl_win_count
            self._rl_log("Guard", "[Window %s] loss=%.3f run_miss=%.4f %s",
                         outcome_win, ell, run_miss, guard.status(), color="safe_calib")
        if miss_prob is not None:
            acc = self._rl_update_pred_metrics(miss_prob, miss, outcome_win)
            self._rl_maybe_finetune(acc, finetune_buf, outcome_win)

    def _rcrl_drain_pending(self):
        """Finalize any remaining (release-lagged) windows at run end, in window order, so the
        trailing `lag` windows get their DMR / λ update / recorded row before the summary."""
        pend = getattr(self, "_rl_pending_by_win", None)
        if not pend:
            return
        for win_id in sorted(pend):
            if win_id in self.window_log:
                self._rcrl_finalize_outcome(win_id, pend[win_id])
        self._rl_pending_by_win = {}

    def _rl_qos_mgmt(self):
        # 1) Finalize a prior window's OUTCOME (guard update / recorded DMR). Terminate-membership
        # finalizes the just-closed window (lag 0); release-membership lags by `_dmr_lag` windows
        # so late-terminating (missed) jobs are attributed to their release window first.
        if not hasattr(self, "_rl_pending_by_win"):
            self._rl_pending_by_win = {}
        outcome_win = self.w_id - 1 - int(self._resolve_dmr_lag())
        pending = self._rl_pending_by_win.get(outcome_win)
        if pending is not None and outcome_win in self.window_log:
            self._rcrl_finalize_outcome(outcome_win, pending)
            del self._rl_pending_by_win[outcome_win]
        prev_win_id = self.w_id - 1

        # 2) Build current state and request an action.
        state = self.get_env()
        self.rl_ctrl_states[self.w_id - 1] = state

        explore_policy = (self._rl_budget_explore_policy
                          if getattr(self, "_rl_explore_mode", "budget") == "budget"
                          else self.rl_agent.random_explore_policy)
        action, _ = self.rl_agent.get_action(state, explore_policy=explore_policy, budget=self.ref_util)

        # Dynamic add/remove: inactive (dormant/removed) tasks are masked to QoS 0 so the
        # fixed-size RL action space still models a changing active task set.
        for t_id, t in enumerate(self.tasks):
            if not getattr(t._task_info, "active", True):
                action[t_id] = 0

        # 3) Apply action to tasks (QoS update) and prepare prediction context.
        prev_state = self.rl_ctrl_states.get(self.w_id - 2) if self.w_id - 2 in self.rl_ctrl_states else [0.0] * len(state)

        prev_qos_list = []
        for t_id, t in enumerate(self.tasks):
            prev_qos_list.append(t._task_info.cur_qos)
            t._task_info.cur_qos = action[t_id]

        self._rl_prepare_pending(prev_win_id, action)

        # 3b) Mode-change protocol (Case 1): reconcile the validated decision d̂_k against the
        # previous one — apply decreases now, defer increases past any carry-in job (§VI-A).
        if getattr(self, "_mode_change", False):
            self._apply_mode_change(prev_qos_list)

        # 4) Compute reward, store transition, and train the RL agent.
        # Store the transition for the *applied* decision of the previous window every
        # window (Alg. 1 L14); prev_qos_list is already the applied (post-fallback)
        # action. The dmr>0 double-store up-weights miss windows.
        reward = self._rl_compute_reward()
        done = (self.sim.now() == self._next_env_change_time)
        self.rl_agent.store_transition(
            state=prev_state,
            action=prev_qos_list,
            reward=reward,
            next_state=state,
            done=done,
        )
        if self.window_log[self.w_id - 1]["dmr"] > 0:
            self.rl_agent.store_transition(
                state=prev_state,
                action=prev_qos_list,
                reward=reward,
                next_state=state,
                done=done,
            )
        if getattr(self, "_rl_train", True):
            for _ in range(getattr(self, "_rl_updates_per_window", 1)):
                self.rl_agent.train_model()

    def get_env(self):
        state = []

        # Global features.
        cur_id = self.w_id - 1

        cur_log = self.window_log[cur_id]
        prev_log = self.window_log.get(cur_id - 1)

        state_util = cur_log["act_util"]
        util_err = cur_log["util_err"]
        prev_util_err = prev_log["util_err"] if prev_log else 0.0

        state.extend([state_util, self.ref_util, util_err])

        # Per-task features.
        max_qos_levels = self.tasks[0]._task_info.max_qos

        for t in self.tasks:
            tinfo = t._task_info
            tlog = tinfo.win_log[cur_id]
            prev_tlog = tinfo.win_log.get(cur_id - 1)

            liveness = 1.0 if tinfo.cur_qos > 0 else 0.0

            load = tlog["exec_time"] / tlog["n_jobs"] if tlog["n_jobs"] > 0 else 0.0

            state.extend([tinfo.cur_qos, liveness, load])
        return state

    def rt_shielding(self, action):
        # if currently action is not safe, reduce qos level until safe
        estimated_table = self.build_estimated_exec_tables()
        total_util = 0.0
        for t_id, t in enumerate(self.tasks):
            tinfo = t._task_info
            est_exec = estimated_table[t_id]
            cur_qos = action[t_id]
            total_util += est_exec[cur_qos] / tinfo.period

        given_action = action.copy()
        activated = False
        while total_util > 1.0:
            activated = True
            # reduce qos level of the task with highest util gain
            max_util_gain = 0.0
            reduce_t_id = -1
            for t_id, t in enumerate(self.tasks):
                tinfo = t._task_info
                cur_qos = action[t_id]
                if cur_qos == 0:
                    continue
                est_exec = estimated_table[t_id]
                util_gain = est_exec[cur_qos] / tinfo.period - est_exec[cur_qos - 1] / tinfo.period
                if util_gain > max_util_gain:
                    max_util_gain = util_gain
                    reduce_t_id = t_id
            if reduce_t_id == -1:
                break
            action[reduce_t_id] -= 1
            total_util -= max_util_gain

        if activated:
            logger.warning(
                "[Window %04d] RL action %s adjusted to %s by RT shielding to ensure schedulability.",
                self.w_id - 1,
                given_action,
                action,
            )

        return action

    def rl_adaptation_on_env_change(self):
        """Adapt the RL controller when a sensed environment change fires. Strategy is
        configurable (dynamic_config.adaptation) so alternatives can be A/B'd:

          hvdf        : reset epsilon + clear memory + FCS-greedy QoS warm-start + window ramp
          reset_only  : reset epsilon + clear memory (NO warm-start) + window ramp
          keep_memory : KEEP epsilon + memory, FCS-greedy QoS warm-start + window ramp
          none        : do nothing (let the RL agent adapt purely online)
        """
        strategy = getattr(self, "adaptation_strategy", "hvdf")
        if strategy == "none":
            return

        if strategy in ("hvdf", "reset_only"):
            self.rl_agent.epsilon = getattr(self, "_rl_epsilon_reset", 0.7)
            self.rl_agent.memory.clear()

        if strategy in ("hvdf", "keep_memory"):
            base_settings = self.hvdf_history(self.ref_util)  # warm-start QoS (active tasks)
            for t_id, t in enumerate(self.tasks):
                t._task_info.cur_qos = base_settings[t_id] if getattr(t._task_info, "active", True) else 0

        if getattr(self, "_dynamic_window_enabled", False):
            self.dynamic_window()

    def dynamic_window(self):
        if self._env_changed:
            self.w_sz = self.w_sz_min
            self._w_sz_ramp_active = True
            self._w_sz_step_tick = 0
            self._w_sz_ramp_step_idx = 0
            self._env_changed = False

        if not self._w_sz_ramp_active:
            return

        self._w_sz_step_tick += 1
        if self._w_sz_step_tick < max(1, self.w_sz_step_interval):
            return

        self._w_sz_step_tick = 0
        self._w_sz_ramp_step_idx += 1
        next_w_sz = self.w_sz_min + self._w_sz_step_size * self._w_sz_ramp_step_idx

        if next_w_sz >= self.base_w_sz or self._w_sz_ramp_step_idx >= self._w_sz_ramp_steps:
            self.w_sz = self.base_w_sz
            self._w_sz_ramp_active = False
        else:
            self.w_sz = max(1, int(round(next_w_sz)))
