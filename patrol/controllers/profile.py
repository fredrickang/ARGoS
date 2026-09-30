import logging
import random


logger = logging.getLogger(__name__)

DEFAULT_MISS_THRESHOLD = 0.1224

# v2 miss-predictor features (N-invariant aggregates). Order is the single source of
# truth for the feature vector and must match training. est_util is tracked by index so
# the inference floor (zero load -> zero miss prob) can find it pre-normalization.
V2_FEATURE_NAMES = [
    "prev_util", "prev_dmr", "prev_util_err", "util_trend",
    "est_util", "est_util_tail", "max_util", "std_util",
    "frac_active", "delta_util", "mean_qos",
]
V2_EST_UTIL_IDX = V2_FEATURE_NAMES.index("est_util")
V2_TAIL_Z = 1.65  # ~p95 one-sided

# Feature-set variants. "v2c" (candidate-only) drops all previous-window context so the
# classifier's prediction depends ONLY on the candidate decision's load -> the fallback
# ladder is separable in est_util (reducing QoS lowers miss-prob) instead of being pinned
# by a constant backlog term shared across all candidates.
V2_CONTEXT_IDX = [0, 1, 2, 3, 9]  # prev_util, prev_dmr, prev_util_err, util_trend, delta_util
V2_CANDIDATE_IDX = [i for i in range(len(V2_FEATURE_NAMES)) if i not in V2_CONTEXT_IDX]
V2_FAMILY = ("v2", "v2c", "v2_nocontext")


class ProfileControllerMixin:
    _PROFILE_LOG_COLORS = {
        "pred": "\x1b[32m",
        "state": "\x1b[36m",
    }

    def _profile_log(self, tag, msg, *args, color=None):
        prefix = f"[Profile][{tag}]"
        if color:
            color_code = self._PROFILE_LOG_COLORS.get(color, "")
            reset = "\x1b[0m" if color_code else ""
            logger.info(f"{color_code}{prefix} {msg}{reset}", *args)
            return
        logger.info(f"{prefix} {msg}", *args)

    def _profile_load_model(self, model_path=None):
        if model_path is None:
            model_path = self.data.get("ctrl", {}).get("profile_model_path")
        if not model_path:
            return

        cached = getattr(self, "profile_model", None)
        if cached and cached.get("path") == model_path:
            return

        import torch
        from patrol.models.mlp import MLP

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(
            ckpt["input_dim"],
            ckpt["hidden"],
            ckpt.get("dropout", 0.0),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        self.profile_model = {
            "path": model_path,
            "model": model,
            "device": device,
            "mean": ckpt.get("mean"),
            "std": ckpt.get("std"),
            "threshold": float(ckpt.get("threshold", DEFAULT_MISS_THRESHOLD)),
            "temperature": float(ckpt.get("temperature", 1.0)),
            "feature_set": ckpt.get("feature_set", "v1"),
        }

    def _profile_get_threshold(self) -> float:
        model_info = getattr(self, "profile_model", None) or {}
        return float(model_info.get("threshold", DEFAULT_MISS_THRESHOLD))

    def _profile_predict_miss(self, x_state, model_path=None):
        if model_path is None:
            model_path = self.data.get("ctrl", {}).get("profile_model_path")
        if not model_path:
            raise ValueError("profile_model_path is not set")

        self._profile_load_model(model_path)

        model = self.profile_model["model"]
        device = self.profile_model.get("device")
        mean = self.profile_model.get("mean")
        std = self.profile_model.get("std")

        import numpy as np
        import torch

        if self._miss_floor(x_state):
            return 0.0

        x = np.array(x_state, dtype=np.float32)
        if mean is not None and std is not None:
            x = (x - mean) / std
        x_t = torch.from_numpy(x).unsqueeze(0)
        if device is not None:
            x_t = x_t.to(device)
            model = model.to(device)
        temp = float(self.profile_model.get("temperature", 1.0)) or 1.0
        with torch.no_grad():
            prob = torch.sigmoid(model(x_t) / temp).item()
        return prob

    def _profile_finetune(self, samples, *, epochs=2, batch_size=128, lr=1e-4, val_ratio=0.2):
        if not samples:
            return
        model_path = self.data.get("ctrl", {}).get("profile_model_path")
        if not model_path:
            logger.warning("Profile finetune skipped: profile_model_path is not set")
            return

        self._profile_load_model(model_path)
        model = self.profile_model["model"]
        device = self.profile_model.get("device")
        mean = self.profile_model.get("mean")
        std = self.profile_model.get("std")
        threshold = self._profile_get_threshold()

        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        x = np.array([s["x_state"] for s in samples], dtype=np.float32)
        y = np.array([s["y_miss"] for s in samples], dtype=np.float32)
        if mean is not None and std is not None:
            x = (x - mean) / std

        idx = np.arange(len(x))
        np.random.shuffle(idx)
        split = max(1, int(len(idx) * (1.0 - val_ratio)))
        train_idx = idx[:split]
        val_idx = idx[split:]
        if len(val_idx) == 0:
            val_idx = idx[:1]
            train_idx = idx[1:] if len(idx) > 1 else idx

        x_train = torch.from_numpy(x[train_idx]).to(device)
        y_train = torch.from_numpy(y[train_idx]).to(device)
        x_val = torch.from_numpy(x[val_idx]).to(device)
        y_val = torch.from_numpy(y[val_idx]).to(device)

        if len(x_train) < 2:
            logger.warning("Profile finetune skipped: need at least 2 samples, got %s", len(x_train))
            return
        batch_size = max(2, min(batch_size, len(x_train)))
        ds = TensorDataset(x_train, y_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(max(1, epochs)):
            for xb, yb in loader:
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            probs = torch.sigmoid(logits)
            candidates = torch.unique(probs).detach().cpu().numpy().tolist()
            candidates = [0.0] + candidates + [1.0]
            best_acc = -1.0
            best_thr = threshold
            best_dist = float("inf")
            for thr in candidates:
                preds = (probs >= float(thr)).float()
                acc = float((preds == y_val).sum().item()) / max(1, y_val.numel())
                dist = abs(float(thr) - 0.5)
                if acc > best_acc or (acc == best_acc and dist < best_dist):
                    best_acc = acc
                    best_thr = float(thr)
                    best_dist = dist
            self.profile_model["threshold"] = best_thr
            acc = best_acc
        return acc
    
    def _classifier_feature_set(self):
        ctrl = self.data.get("ctrl", {})
        cls = ctrl.get("classifier", {}) if isinstance(ctrl.get("classifier"), dict) else {}
        return str(cls.get("feature_set", ctrl.get("feature_set", "v1"))).lower()

    def _v2_indices(self):
        if self._classifier_feature_set() in ("v2c", "v2_nocontext"):
            return V2_CANDIDATE_IDX
        return list(range(len(V2_FEATURE_NAMES)))

    def _prev_window_size(self, prev_win_id):
        entry = self.window_log.get(prev_win_id)
        if entry and "w_range" in entry:
            lo, hi = entry["w_range"]
            return hi - lo
        return self.w_sz

    def _task_period_est(self, tinfo, prev_win_id, prev_w_sz):
        tlog = tinfo.win_log.get(prev_win_id, {})
        n_jobs = tlog.get("n_jobs", 0)
        if n_jobs > 0:
            return prev_w_sz / n_jobs
        for back_id in range(prev_win_id - 1, -1, -1):
            back_log = tinfo.win_log.get(back_id, {})
            back_jobs = back_log.get("n_jobs", 0)
            if back_jobs > 0:
                back_range = self.window_log[back_id]["w_range"]
                return (back_range[1] - back_range[0]) / back_jobs
        return 0.0

    @staticmethod
    def _interp_zeros(vals):
        # Fill zero entries (index >= 1) by linear interpolation between known values;
        # trailing zeros carry the last known value. Mirrors build_estimated_exec_tables.
        i = 1
        n = len(vals)
        while i < n:
            if vals[i] == 0.0:
                j = i + 1
                while j < n and vals[j] == 0.0:
                    j += 1
                if j < n:
                    step = (vals[j] - vals[i - 1]) / (j - i + 1)
                    for k in range(i, j):
                        vals[k] = vals[k - 1] + step
                    i = j
                else:
                    for k in range(i, n):
                        vals[k] = vals[i - 1]
                    break
            i += 1
        return vals

    def _estimated_exec_stats(self):
        """Per-task runtime execution estimate: (mean[q], std[q]) for q in 0..max_qos,
        drawn from the observed exec history (profile-free). Empty QoS levels are
        interpolated so every candidate QoS has an estimate.

        Prefers per-job DEMAND history (unaffected by CPU throttling) over delivered
        exec time — delivered time caps at capacity under saturation, which makes
        est_util indistinguishable across candidate decisions exactly when the fallback
        needs to tell them apart."""
        stats = []
        for t in self.tasks:
            tinfo = t._task_info
            maxq = tinfo.max_qos
            demand = getattr(tinfo, "demand_history", None)
            means = [0.0] * (maxq + 1)
            stds = [0.0] * (maxq + 1)
            for q in range(1, maxq + 1):
                hist = demand[q] if (demand and q < len(demand) and demand[q]) else (
                    tinfo.exec_history[q] if q < len(tinfo.exec_history) else [])
                if hist:
                    m = sum(hist) / len(hist)
                    means[q] = m
                    if len(hist) >= 2:
                        var = sum((h - m) ** 2 for h in hist) / (len(hist) - 1)
                        stds[q] = var ** 0.5
            self._interp_zeros(means)
            self._interp_zeros(stds)
            stats.append((means, stds))
        return stats

    def _build_miss_features_v2(self, prev_win_id, qos_list):
        n = len(self.tasks)
        if prev_win_id not in self.window_log:
            return [0.0] * len(self._v2_indices())

        prev = self.window_log[prev_win_id]
        prev_util = float(prev.get("act_util", 0.0))
        prev_dmr = float(prev.get("dmr", 0.0))
        prev_err = float(prev.get("util_err", self.ref_util - prev_util))
        prev2 = self.window_log.get(prev_win_id - 1)
        util_trend = prev_util - float(prev2.get("act_util", prev_util)) if prev2 else 0.0

        if qos_list is None:
            qos_list = [t._task_info.cur_qos for t in self.tasks]

        prev_w_sz = self._prev_window_size(prev_win_id)
        stats = self._estimated_exec_stats()

        utils, utils_tail, qos_norm = [], [], []
        n_active = 0
        # Aggregate over the ACTIVE task set only, so dormant/removed pool slots don't skew
        # the features (keeps them consistent with the all-active training distribution).
        for i, t in enumerate(self.tasks):
            tinfo = t._task_info
            if not getattr(tinfo, "active", True):
                continue
            q = int(qos_list[i]) if i < len(qos_list) else tinfo.cur_qos
            means, stds = stats[i]
            c = means[q] if 0 <= q < len(means) else 0.0
            s = stds[q] if 0 <= q < len(stds) else 0.0
            # Period is a known task parameter (paper assumption: T_i known, only C_i(.)
            # unknown). Using it avoids the saturation bias of estimating T from the number
            # of completed jobs, which shrinks under overload and hides the true util.
            period = float(getattr(tinfo, "period", 0.0)) or self._task_period_est(
                tinfo, prev_win_id, prev_w_sz)
            u = (c / period) if (period > 0 and q > 0) else 0.0
            ut = ((c + V2_TAIL_Z * s) / period) if (period > 0 and q > 0) else 0.0
            utils.append(u)
            utils_tail.append(ut)
            qos_norm.append(q / max(1, tinfo.max_qos))
            if q > 0:
                n_active += 1

        n_act = len(utils)  # number of active tasks
        est_util = sum(utils)
        est_util_tail = sum(utils_tail)
        max_util = max(utils) if utils else 0.0
        if n_act > 1:
            mu = est_util / n_act
            std_util = (sum((u - mu) ** 2 for u in utils) / n_act) ** 0.5
        else:
            std_util = 0.0
        frac_active = n_active / n_act if n_act else 0.0
        delta_util = est_util - prev_util
        mean_qos = sum(qos_norm) / n_act if n_act else 0.0

        full = [prev_util, prev_dmr, prev_err, util_trend,
                est_util, est_util_tail, max_util, std_util,
                frac_active, delta_util, mean_qos]
        return [full[i] for i in self._v2_indices()]

    def _miss_floor(self, x_state):
        """True when the candidate releases no load (est_util == 0) -> miss impossible."""
        if self._classifier_feature_set() not in V2_FAMILY:
            return False
        idx = self._v2_indices()
        try:
            pos = idx.index(V2_EST_UTIL_IDX)
            return len(x_state) == len(idx) and float(x_state[pos]) <= 1e-9
        except (TypeError, ValueError, IndexError):
            return False

    def _profile_build_state(self, prev_win_id, qos_list):
        if self._classifier_feature_set() in V2_FAMILY:
            return self._build_miss_features_v2(prev_win_id, qos_list)

        state = [self.w_sz]
        if prev_win_id not in self.window_log:
            return state + [0.0] * (2 * len(self.tasks))

        prev_w_range = self.window_log[prev_win_id]["w_range"]
        prev_w_sz = prev_w_range[1] - prev_w_range[0]

        if qos_list is None:
            qos_list = [t._task_info.cur_qos for t in self.tasks]

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

    def _profile_qos_mgmt(self):
        rng = getattr(self, "profile_rng", None)
        if rng is None:
            seed = self.data.get("ctrl", {}).get("profile_seed")
            rng = random.Random(seed)
            self.profile_rng = rng

        prev_win_id = self.w_id - 1
        pending = getattr(self, "profile_pending", None)
        if pending and pending.get("win_id") == prev_win_id and prev_win_id in self.window_log:
            samples = getattr(self, "profile_samples", None)
            if samples is None:
                samples = []
                self.profile_samples = samples
            dmr = self.window_log[prev_win_id]["dmr"]
            miss = 1 if dmr > 0 else 0
            miss_prob = None
            pred_label = None
            try:
                miss_prob = self._profile_predict_miss(pending["x_state"])
                pred_label = 1 if miss_prob >= self._profile_get_threshold() else 0
            except Exception as exc:
                logger.warning("Profile prediction failed: %s", exc)
            if miss_prob is not None:
                pred_scores = getattr(self, "profile_pred_scores", None)
                if pred_scores is None:
                    pred_scores = []
                    self.profile_pred_scores = pred_scores
                pred_labels = getattr(self, "profile_pred_labels", None)
                if pred_labels is None:
                    pred_labels = []
                    self.profile_pred_labels = pred_labels
                pred_scores.append(float(miss_prob))
                pred_labels.append(int(miss))
                total = len(pred_labels)
                thr = self._profile_get_threshold()
                preds = [1 if s >= thr else 0 for s in pred_scores]
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
                self._profile_log(
                    "Pred",
                    "count=%s acc=%.4f auc=%s",
                    total,
                    acc,
                    f"{auc:.4f}" if auc is not None else "nan",
                    color="pred",
                )
            samples.append(
                {
                    "x_state": pending["x_state"],
                    "y_miss": miss,
                }
            )
            # logger.info(
            #     "[Profile][Window %s] x_state=%s y_miss=%s y_pred=%s prob=%.4f",
            #     prev_win_id,
            #     pending["x_state"],
            #     miss,
            #     pred_label,
            #     miss_prob if miss_prob is not None else -1.0,
            # )
            self.profile_pending = None

        # Hold-and-settle profiling. Re-randomizing QoS every window keeps an overloaded
        # taskset perpetually saturated, so backlog from the previous config contaminates
        # every window's miss label (label ≈ constant 1, no learnable signal). Instead we
        # HOLD a random config for `settle + record` windows so the previous config's
        # backlog drains, then label only the last `record` (settled) windows — whose miss
        # outcome actually reflects the held decision.
        ctrl = self.data.get("ctrl", {})
        settle = int(ctrl.get("profile_settle_windows", 4))
        record = int(ctrl.get("profile_record_windows", 2))
        hold_len = max(1, settle + record)

        hold = getattr(self, "_profile_hold", None)
        if hold is None or hold["remaining"] <= 0:
            qos_list = []
            for t in self.tasks:
                tinfo = t._task_info
                if tinfo.t_type != 1:
                    qos_list.append(tinfo.cur_qos)
                    continue
                tinfo.cur_qos = rng.randint(0, tinfo.max_qos)
                qos_list.append(tinfo.cur_qos)
            hold = {"qos": qos_list, "remaining": hold_len}
            self._profile_hold = hold
        else:
            qos_list = hold["qos"]
            for t, q in zip(self.tasks, qos_list):
                t._task_info.cur_qos = q

        hold["remaining"] -= 1
        is_settled = hold["remaining"] < record  # last `record` windows of the hold

        if is_settled:
            x_state = self._profile_build_state(prev_win_id, qos_list)
            self.profile_pending = {"win_id": self.w_id, "x_state": x_state}
        else:
            self.profile_pending = None  # discard settling windows

        logger.debug("[Profile][State][Window %s] hold rem=%s settled=%s.",
                     prev_win_id, hold["remaining"], is_settled)
