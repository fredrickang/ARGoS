import logging
import math


logger = logging.getLogger(__name__)


class FcsControllerMixin:
    def _init_est_budget(self):
        # initial estimated budget calculation (controllable tasks only)
        init_budget = 0.0
        for t_id, t in enumerate(self.tasks):
            tinfo = t._task_info
            if tinfo.t_type != 1:  # uncontrollable tasks are not part of the FCS budget
                continue
            est_exec = self.tasks_estimated_tables[t_id]
            cur_qos = tinfo.cur_qos
            init_budget += est_exec[cur_qos] / tinfo.period
        return init_budget

    def _fcs_qos_mgmt(self):
        prev_win_id = self.w_id - 1

        # calc control input
        prev_log = self.window_log[prev_win_id]
        miss_err = self.ref_dmr - prev_log['dmr']
        util_err = self.ref_util - prev_log['act_util']
        control_input = min(
            self.fcs_ctrl_param['kp_m'] * miss_err,
            self.fcs_ctrl_param['kp_u'] * util_err
        )

        logger.info(
            "[Window %s] DMR: %.4f, Util: %.4f, QoS: %.4f, Control Input: %.4f",
            prev_win_id,
            prev_log["dmr"],
            prev_log["act_util"],
            prev_log["avg_qos"],
            control_input,
        )

        # adjust qos level of all tasks
        self.fcs_ctrl_param['cur_est_budget'] += control_input
        self.fcs_ctrl_param['cur_est_budget'] = max(0.0, self.fcs_ctrl_param['cur_est_budget'])

        logger.debug(
            "[Window %s] Adjusted Estimated Budget: %.4f",
            prev_win_id,
            self.fcs_ctrl_param["cur_est_budget"],
        )
        updated_qos_dict = self.hvdf(self.fcs_ctrl_param['cur_est_budget'])
        logger.debug("[Window %s] Updated QoS levels: %s", prev_win_id, updated_qos_dict)

        for t_id, t in enumerate(self.tasks):
            if t._task_info.t_type == 1:
                t._task_info.cur_qos = updated_qos_dict[t._task_info.identifier]
        
        estimated_util = self.estimated_total_util(updated_qos_dict)

        logger.debug(
            "[Window %s] Estimated Total Utilization after QoS adjustment: %.4f",
            prev_win_id,
            estimated_util,
        )

    def estimated_total_util(self, qos_dict):
        total_util = 0.0
        for t_id, t in enumerate(self.tasks):
            tinfo = t._task_info
            if tinfo.t_type != 1:  # target tasks only
                continue
            est_exec = self.tasks_estimated_tables[t_id]
            cur_qos = qos_dict[t._task_info.identifier]
            total_util += est_exec[cur_qos] / tinfo.period
        return total_util

    def hvdf(self, budget):
        # Greedy QoS allocation under a utilization budget.
        EPS = 1e-9

        # gather target tasks only
        target_tasks = {}
        target_tasks_estimated = {}
        for t_id, t in enumerate(self.tasks):
            tinfo = t._task_info
            if tinfo.t_type == 1:  # target task
                target_tasks[t._task_info.identifier] = t
                target_tasks_estimated[t._task_info.identifier] = self.tasks_estimated_tables[t_id]
                

        # Current QoS level per task (start from 0).
        qos_list = {t_id: 0 for t_id in target_tasks.keys()}

        # Keep upgrading while budget remains and an upgrade is possible.
        updated = True
        while budget > EPS and updated:
            updated = False

            # For each task, compute the ratio:
            # (value increase) / (util increase) for a +1 QoS upgrade.
            # Value is defined as qos_level * execution time.
            best_ratio = 0
            task_idx = -1
            util_incr = 0.0
            for t_id, t in target_tasks.items():
                tinfo = t._task_info
                cur_qos = qos_list[t_id]
                est_exec = target_tasks_estimated[t_id]

                # Skip if already at max QoS.
                if cur_qos >= tinfo.max_qos:
                    continue

                # Marginal value from QoS+1.
                val_diff = (cur_qos + 1) ** 2 - (cur_qos) ** 2

                # Marginal utilization from QoS+1.
                util_diff = est_exec[cur_qos + 1] / tinfo.period - est_exec[cur_qos] / tinfo.period

                # Ratio of value gain to util gain.
                ratio = val_diff / util_diff if util_diff > EPS else 0

                # Pick the best feasible upgrade within remaining budget.
                if ratio > best_ratio and util_diff <= budget:
                    best_ratio = ratio
                    task_idx = t_id
                    util_incr = util_diff

            # Apply the best upgrade found in this round.
            if task_idx >= 0:
                updated = True
                budget -= util_incr
                qos_list[task_idx] += 1

        return qos_list

    def build_estimated_exec_tables(self):
        estimated_tables = []

        for t in self.tasks:
            tinfo = t._task_info
            est_exec = [0.0]  # for qos_level 0

            for qos_level in range(1, tinfo.max_qos + 1):
                hist = tinfo.exec_history[qos_level]
                est = (sum(hist) / len(hist)) if hist else 0.0
                est_exec.append(est)

            # Interpolate zero entries.
            i = 1
            while i < len(est_exec):
                if est_exec[i] == 0.0:
                    j = i + 1
                    while j < len(est_exec) and est_exec[j] == 0.0:
                        j += 1

                    if j < len(est_exec):
                        step = (est_exec[j] - est_exec[i - 1]) / (j - i + 1)
                        for k in range(i, j):
                            est_exec[k] = est_exec[k - 1] + step
                        i = j
                    else:
                        for k in range(i, len(est_exec)):
                            est_exec[k] = est_exec[i - 1]
                        break
                i += 1

            # Round to 1 decimal place.
            est_exec = [round(v, 1) for v in est_exec]
            est_exec[0] = 0.0  # keep qos 0 exact

            estimated_tables.append(est_exec)

        return estimated_tables

    def hvdf_history(self, budget):
        # Greedy QoS allocation under a utilization budget.
        EPS = 1e-9

        estimated_tables = self.build_estimated_exec_tables()
        sim = self._table_similarity(self.tasks_estimated_tables, estimated_tables)
        logger.info(
            "Estimated table similarity: mape=%.4f, sim=%.4f, mae=%.4f, rmse=%.4f",
            sim["mape"],
            sim["similarity"],
            sim["mae"],
            sim["rmse"],
        )
        # Current QoS level per task (start from 0).
        qos_list = [0] * len(self.tasks)

        # Keep upgrading while budget remains and an upgrade is possible.
        updated = True
        while budget > EPS and updated:
            updated = False

            # For each task, compute the ratio:
            # (value increase) / (util increase) for a +1 QoS upgrade.
            # Value is defined as qos_level * execution time.
            best_ratio = 0
            task_idx = -1
            util_incr = 0.0
            for t_id, t in enumerate(self.tasks):
                tinfo = t._task_info
                cur_qos = qos_list[t_id]
                est_exec = estimated_tables[t_id]

                # Skip if already at max QoS.
                if cur_qos >= tinfo.max_qos:
                    continue

                # Marginal value from QoS+1.
                val_diff = (cur_qos + 1) ** 2 - (cur_qos) ** 2

                # Marginal utilization from QoS+1.
                util_diff = est_exec[cur_qos + 1] / tinfo.period - est_exec[cur_qos] / tinfo.period

                # Ratio of value gain to util gain.
                ratio = val_diff / util_diff if util_diff > EPS else 0

                # Pick the best feasible upgrade within remaining budget.
                if ratio > best_ratio and util_diff <= budget:
                    best_ratio = ratio
                    task_idx = t_id
                    util_incr = util_diff

            # Apply the best upgrade found in this round.
            if task_idx >= 0:
                updated = True
                budget -= util_incr
                qos_list[task_idx] += 1

        return qos_list

    def _table_similarity(self, table_a, table_b):
        # Compare two nested tables by flattening and computing simple metrics.
        vec_a = [float(v) for row in table_a for v in row]
        vec_b = [float(v) for row in table_b for v in row]
        n = min(len(vec_a), len(vec_b))
        if n == 0:
            return {"mape": 0.0, "similarity": 0.0, "mae": 0.0, "rmse": 0.0}

        vec_a = vec_a[:n]
        vec_b = vec_b[:n]

        abs_err = [abs(a - b) for a, b in zip(vec_a, vec_b)]
        sq_err = [(a - b) ** 2 for a, b in zip(vec_a, vec_b)]
        mae = sum(abs_err) / n
        rmse = math.sqrt(sum(sq_err) / n)

        pct_err = []
        for a, b in zip(vec_a, vec_b):
            if a == 0.0:
                continue
            pct_err.append(abs((b - a) / a))
        mape = sum(pct_err) / len(pct_err) if pct_err else 0.0
        similarity = max(0.0, 1.0 - mape)

        return {"mape": mape, "similarity": similarity, "mae": mae, "rmse": rmse}
