import atexit
import hashlib
import json
import logging
import math
import os
import random
from datetime import datetime

import simso.generator as sims
from simso.core import Timer
from patrol.rl.dqn import DQNAgent
from patrol.sim.recorder import RunRecorder, git_sha, make_run_id


logger = logging.getLogger(__name__)

_WINDOW_SUMMARY_COLOR = "\x1b[37m"
_WINDOW_SUMMARY_RESET = "\x1b[0m"


class SchedulerCoreMixin:
    def init(self):
        self.ready_list = []
        self.base_w_sz = int(self.data.get('w_sz', 100))
        self.w_sz = self.base_w_sz
        self.env_change_total = self.data.get('dynamic_config', {}).get('num_env_changes', 2) 
        self.env_change_interval_cycles = max(
            1, int(self.sim.duration // self.env_change_total)
        )
        self.env_change_interval_ms = int(
            self.env_change_interval_cycles / self.sim.cycles_per_ms
        )
        self._next_env_change_time = self.env_change_interval_cycles
        self._init_window_sz_config()
        
        self.tasks = [t for t in self.sim.task_list if t._task_info.t_type != 3]

        # Adaptive window sizing (paper §VII-C). Pick the smallest window length W_k whose
        # maximum carry-in loss ε_carry,k = N_k / Σ⌈W_k/T_i⌉ (Eq. 9) stays below the target
        # miss level α, preserving the Lemma-1 precondition (α > ε_carry) the guarantee needs.
        # Disabled by default (fixed W = base_w_sz); enable via controller.adaptive_window.
        _aw = (self.data.get('ctrl', {}) or {}).get('adaptive_window', {}) or {}
        self._adaptive_window_enabled = bool(_aw.get('enabled', False))
        self._aw_alpha = float((self.data.get('ctrl', {}).get('safety') or {})
                               .get('target_miss_alpha', 0.1))
        self._aw_margin = float(_aw.get('margin', 0.0))               # extra headroom below α
        self._aw_min = int(_aw.get('min_ms', max(1, self.base_w_sz // 20)))
        self._aw_max = int(_aw.get('max_ms', self.base_w_sz * 20))

        # Schedule-driven dynamic changes (dynamic_config.changes). When present these take
        # precedence over the legacy dynamic_mode schedules. Each event has an at_window and a
        # kind: 'ref_util' (value) or 'taskset' (op: add|remove|replace, count, util). Because
        # the schedule is known up front, the controller "senses" each change exactly when it fires.
        _dyn = self.data.get('dynamic_config', {}) or {}
        self.dynamic_changes = sorted(list(_dyn.get('changes', []) or []),
                                      key=lambda e: int(e.get('at_window', 0)))
        for _e in self.dynamic_changes:
            _e['_applied'] = False
        self.adaptation_strategy = str(_dyn.get('adaptation', 'reset_only')).lower()
        # Window-size ramp on env change (shrinks w_sz to react faster). Off by default now:
        # it is only invoked once per change (never re-ramps), so it just pins w_sz at w_sz_min
        # and inflates the window count. Set dynamic_config.dynamic_window: true to re-enable.
        self._dynamic_window_enabled = bool(_dyn.get('dynamic_window', False))
        self.change_log = []  # [(window, description)] for plotting/analysis

        # dynamic environment configuration
        self.dynamic_mode = int(self.data.get('dynamic_mode', 1))  # 1: change reference utilization 2: change available budget 3: change task set
        # for mode 1
        self.ref_util_schedule = [0.9,0.9,0.9,0.9]
        self._ref_util_idx = 0
        # for mode 2
        self.back_util_schedule = [0.0, 0.2, 0.4, 0.6]
        self._back_util_idx = 0
        # for mode 3
        self.new_util_schedule = [(self.data.get('dynamic_config', {}).get('num_of_target_tasks', 5), self.data.get('dynamic_config', {}).get('target_util', 0.0))] * self.env_change_total
        self._new_util_idx = 0

        self.window_log_dir = self.data.get('window_log_dir', 'window_logs')
        default_name = os.path.join(self.window_log_dir, f'window_log_dyn{self.dynamic_mode}.png')
        self.window_log_path = self.data.get('window_log_png', default_name)
        if os.path.dirname(self.window_log_path) == "":
            self.window_log_path = os.path.join(self.window_log_dir, self.window_log_path)

        # Gaussian execution time parameters
        self.mu = self.data.get('mu', 0.0)  # -2 ~ 2 // +4 : wcet
        self.sigma = self.data.get('sigma', 1.0)  # 0.1 ~ 1.0 // lower is deterministic

        # Execution-time noise model:
        #   'logistic' : wcet * sigmoid(N(mu, sigma))      (original PATROL model)
        #   'gaussian' : N(mean=wcet, std=exec_cov*wcet)   (paper's soft-RT model)
        self.exec_model = str(self.data.get('exec_model', 'logistic')).lower()
        self.exec_cov = float(self.data.get('exec_cov', self.sigma))

        # metadata
        self.w_id = 0  # global window id
        self.window_log = {}  # log dict for each window

        # logging state for current window
        self.cur_win_job_cnt = 0
        self.cur_win_exec_time = 0
        self.cur_win_qos = 0
        self.cur_win_dm_cnt = 0
        self.cur_win_term_cnt = 0
        self.cur_win_val = 0
        self.env_change_times = []
        
        # controller parameters
        self.ctrl_mode = self.data.get('ctrl', {}).get('mode', 'fcs')  # 'fcs' or 'RL'
        self.ref_util = self.data.get('ctrl', {}).get('ref_util', 0.9)
        self.ref_dmr = self.data.get('ctrl', {}).get('ref_dmr', 0.01)
        self.profile_seed = self.data.get('ctrl', {}).get('profile_seed')
        self.profile_rng = random.Random(self.profile_seed)
        self._init_task_rngs()

        # FCS controller parameters
        self.fcs_ctrl_param = dict(self.data.get('fcs_ctrl_param', {}))

        tasks_wcet_tables = [t._task_info.wcet_table for t in self.tasks]
        self.tasks_wcet_tables = [
            [wcet for _, wcet in sorted(task_wcet.items())]
            for task_wcet in tasks_wcet_tables
        ]

        # Estimated execution profiles = what the FCS controller *trusts* when allocating QoS.
        # Ground truth = self.tasks_wcet_tables (each QoS level's nominal exec time).
        # For the "inaccurate profiles" experiment, replace the estimate with a profile at a
        # target cosine similarity to ground truth; jobs still execute against the true table.
        sim_metric = self.fcs_ctrl_param.get('similarity_metric')
        sim_level = self.fcs_ctrl_param.get('similarity')
        self.requested_similarity = None
        self.realized_similarity = None
        if sim_metric and sim_level is not None and float(sim_level) < 1.0:
            sim_seed = self.fcs_ctrl_param.get('similarity_seed')
            if sim_seed is not None:
                random.seed(int(sim_seed))  # vary the perturbation per draw (task set/periods fixed earlier)
            self.tasks_estimated_tables = self._generate_similar_table(
                self.tasks_wcet_tables, similarity=float(sim_level), metric=str(sim_metric)
            )
            # Record the realized similarity (per the metric used) for sanity checking.
            self.requested_similarity = float(sim_level)
            if str(sim_metric).lower() == 'mape':
                self.realized_similarity = self._table_similarity(
                    self.tasks_wcet_tables, self.tasks_estimated_tables)['similarity']
            else:  # cosine
                flat_a = [v for row in self.tasks_wcet_tables for v in row]
                flat_b = [v for row in self.tasks_estimated_tables for v in row]
                dot = sum(x * y for x, y in zip(flat_a, flat_b))
                na = math.sqrt(sum(x * x for x in flat_a))
                nb = math.sqrt(sum(y * y for y in flat_b))
                self.realized_similarity = (dot / (na * nb)) if na > 0 and nb > 0 else None
        elif self.exec_model == 'gaussian':
            # Gaussian mean == nominal, so the accurate estimate is the nominal table itself.
            self.tasks_estimated_tables = [list(row) for row in self.tasks_wcet_tables]
        else:
            self.tasks_estimated_tables = [
                [self._investigate_peak(wcet, self.mu, self.sigma) / self.fcs_ctrl_param['Ga'] for wcet in task_wcet.values()]
                for task_wcet in tasks_wcet_tables
            ]

        self.fcs_ctrl_param['cur_est_budget'] = self._init_est_budget()

        # RL controller parameters
        # to be implemented

        # window states dict for RL controller
        # Optional per-run RNG seed (controller.rl.seed) to get replicates for sweeps.
        # dqn.py seeds everything to 22 at import; re-seed here BEFORE agent creation so
        # network init, exploration, replay sampling, and per-window exec draws all vary.
        _rlc = self.data.get('ctrl', {}).get('rl', {}) or {}
        _rl_seed = _rlc.get('seed')
        if _rl_seed is not None:
            from patrol.rl.dqn import set_seed as _set_seed
            _set_seed(int(_rl_seed))
        self.rl_agent = DQNAgent(
            hiddens_common=tuple(_rlc.get('hiddens_common', (64, 64))),
            hiddens_actions=tuple(_rlc.get('hiddens_actions', (32,))),
            hiddens_value=tuple(_rlc.get('hiddens_value', (32,))),
            state_size=3 + 3 * len(self.tasks),  # global  + per-task features
            action_dim=len(self.tasks),
            action_bins=self.tasks[0]._task_info.max_qos + 1,  # assume same granularity
            gamma=0.95,
            lr=1e-3,
            batch_size=32,
            replay_size=50_000,
            target_update_period=10,
            save_path=str(_rlc.get('save_path', "./outputs/dqn_branching_model.pth")),
            # Warm-start: load a pretrained policy so the agent starts competent instead of
            # tabula rasa (kills the cold-start warm-up learning phase). Loads into both the
            # online and target nets; None -> random init (cold start).
            load_path=(str(_rlc['load_path'])
                       if _rlc.get('load_path') and os.path.exists(str(_rlc['load_path']))
                       else None),
        )
        # Save the trained policy at run end (warm-start source for later runs).
        self._rl_save_model_on_exit = bool(_rlc.get('save_model', False))

        # Optional exploration overrides (controller.rl.*). epsilon=0 -> greedy, no exploration.
        _rl_cfg = self.data.get('ctrl', {}).get('rl', {}) or {}
        if 'epsilon' in _rl_cfg:
            self.rl_agent.epsilon = float(_rl_cfg['epsilon'])
        if 'epsilon_decay' in _rl_cfg:
            self.rl_agent.epsilon_decay = float(_rl_cfg['epsilon_decay'])
        if 'epsilon_min' in _rl_cfg:
            self.rl_agent.epsilon_min = float(_rl_cfg['epsilon_min'])
        if 'epsilon_decay_per_window' in _rl_cfg:
            self.rl_agent.epsilon_decay_per_window = bool(_rl_cfg['epsilon_decay_per_window'])
        self._rl_epsilon_reset = float(_rl_cfg.get('epsilon_reset', 0.7))
        self._rl_train = bool(_rl_cfg.get('train', True))  # False -> freeze policy (no learning)
        # Exploration allocation: 'budget' centres exploratory QoS on ref_util (moderate,
        # avoids the all-max saturation); 'uniform' = legacy randint [0, max_qos].
        self._rl_explore_mode = str(_rl_cfg.get('explore', 'budget')).lower()
        self._rl_explore_spread = float(_rl_cfg.get('explore_spread', 0.25))
        # Reward: 'v2' (positive-capable, balanced util/QoS) or 'decoupled_sum' (legacy).
        self._rl_reward_mode = str(_rl_cfg.get('reward', 'v2')).lower()
        self._rl_updates_per_window = int(_rl_cfg.get('updates_per_window', 1))
        self._rl_reward_wu = float(_rl_cfg.get('reward_util_weight', 1.0))
        self._rl_reward_wv = float(_rl_cfg.get('reward_value_weight', 1.0))

        # RCRL loss mode for the λ update / realized-miss accounting:
        #   'binary'     -> ℓ = 1{window had any deadline miss}  (original)
        #   'fraction'   -> ℓ = per-window miss ratio (missed jobs / jobs ACTUALLY released) ∈ [0,1]
        #   'releasable' -> ℓ = missed jobs / MAX releasable jobs Σ⌈W/T_i⌉ (paper Eq. 3). This is
        #                   the paper's ℓ_miss,k: normalizing by the max releasable (not the actual)
        #                   job count keeps the loss scale consistent across windows.
        # All three are bounded in [0,1], so Theorem-1 (B = max(α,1-α)) holds unchanged; only the
        # quantity the shield regulates changes. The miss PREDICTOR label stays binary.
        self._rl_loss_mode = str((self.data.get('ctrl', {}).get('safety') or {})
                                 .get('loss_mode', 'binary')).lower()

        # Window membership for the deadline-miss ratio (DMR = misses / jobs):
        #   'terminate' (default) -> a job is attributed to the window in which it TERMINATES.
        #        Counted once at termination (no carry-out double count); the denominator is the
        #        number of jobs that terminated in the window. Exact at window close.
        #   'release'   -> a job is attributed to the window in which it was RELEASED
        #        (activation_date). Because a missed job (abort_on_miss=False) terminates after
        #        its deadline -- possibly in a later window -- the release window's DMR is only
        #        complete a few windows later; it is finalized with a lag of `_dmr_lag` windows.
        # Either way, per-window EXECUTION time still accrues to the window it ran in, so the
        # utilization signal is unchanged.
        self._dmr_membership = str((self.data.get('ctrl', {}).get('safety') or {})
                                   .get('dmr_membership', 'terminate')).lower()

        # SafetyGuard scorer (paper §V-A), selecting how the safety score f(x) is produced:
        #   'learned'    -> MLP miss probability, f = 1 - p  (Case 3, unavailable profiles; default)
        #   'analytical' -> EDF utilization margin f = 1 - Σ C_i(q_i)/T_i  (Cases 1 & 2)
        # For 'analytical', safety.profile selects the execution-time table used as C_i(q):
        #   'valid'   -> ground-truth WCET (self.tasks_wcet_tables)        -> Case 1 (hard guarantee)
        #   'invalid' -> estimated/possibly-perturbed profile (self.tasks_estimated_tables) -> Case 2
        _safety_cfg = (self.data.get('ctrl', {}).get('safety') or {})
        self._safety_score_mode = str(_safety_cfg.get('score', 'learned')).lower()
        self._safety_profile = str(_safety_cfg.get('profile', 'invalid')).lower()
        # Controlled profile invalidity for Case 2: the analytical profile UNDERESTIMATES the true
        # WCET by this fraction, i.e. C_profile = (1 - u) * C_true (u=0.2 -> 20% underestimate).
        # Jobs still execute against the true WCET, so the shield over-admits and the adaptive
        # threshold must compensate. 0 -> use tasks_estimated_tables as-is.
        self._safety_profile_underest = float(_safety_cfg.get('profile_underest', 0.0))
        # Mode-change protocol (paper §VI-A), for the Case-1 hard guarantee across windows: a QoS
        # DECREASE takes effect immediately, while a QoS INCREASE is deferred until the carry-in
        # job (released under the previous decision) completes, so a raise never piles extra demand
        # on top of a still-running carry-in job. Off by default (Cases 2&3 use ε_carry/adaptive W).
        self._mode_change = bool(_safety_cfg.get('mode_change', False))
        # Mode-change deferral state: the set of carry-in jobs (live at the last boundary) that
        # must ALL complete before any deferred QoS increase takes effect, and the pending
        # per-task target levels waiting on them.
        self._mc_carryin = set()
        self._mc_pending = {}
        # Release-mode finalization lag (in windows). If not set explicitly it is derived from
        # the taskset as  max_i ceil( D_i / w ),  w = window size, D_i = task relative deadline
        # -- the largest number of windows a deadline spans, i.e. how long to wait before a
        # released job's outcome is known. See _resolve_dmr_lag (computed lazily once tasks exist).
        self._dmr_lag_explicit = (self.data.get('ctrl', {}).get('safety') or {}).get('dmr_lag', None)
        # release-mode per-window accumulators keyed by release-window id
        self._rel_total = {}
        self._rel_miss = {}

        self.rl_ctrl_states = {}

        # Size the first window before the controller timer starts (no-op unless enabled).
        if self._adaptive_window_enabled:
            self._apply_adaptive_window()

        # controller timer
        Timer(self.sim, self._timer_func, (), delay=self.w_sz, one_shot=True, prior=True, overhead=0).start()
        # save window log plot at process exit
        
        #atexit.register(self._save_window_log_plot)
        atexit.register(self._save_summary_metrics)
        if getattr(self, "_rl_save_model_on_exit", False):
            atexit.register(self._save_rl_policy)

        # Per-run metrics recorder (per-run dir + streamed windows.csv). Best-effort:
        # a recorder failure must never break the simulation.
        self.recorder = None
        self._recorder_flushed_upto = -1
        try:
            self._init_recorder()
        except Exception as exc:
            logger.warning("Recorder init failed: %s", exc)
            self.recorder = None

        if self.ctrl_mode != 'profile':
            try:
                self._profile_load_model()
            except Exception as exc:
                logger.warning("Profile model preload failed: %s", exc)

    def _run_base_name(self):
        return os.path.splitext(os.path.basename(self.window_log_path))[0]

    def _resolved_safety_mode(self):
        ctrl = self.data.get("ctrl", {})
        safety_cfg = dict(ctrl.get("safety") or {})
        mode = safety_cfg.get("mode")
        if mode is None:  # legacy back-compat shim (matches build_safety_guard)
            mode = "conformal" if ctrl.get("enable_profile_fallback", False) else "none"
        return mode, safety_cfg

    def _init_recorder(self):
        base = self._run_base_name()
        safety_mode, safety_cfg = self._resolved_safety_mode()
        tag = safety_mode if self.ctrl_mode == "rl" else self.ctrl_mode
        run_id = make_run_id(
            base,
            safety_mode=tag,
            seed=self.profile_seed,
            timestamp=datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        )
        run_dir = os.path.join(self.window_log_dir, "runs", run_id)
        self.recorder = RunRecorder(run_dir)
        self.run_dir = run_dir

        tw = getattr(self, "tasks_wcet_tables", [])
        tw_hash = hashlib.sha1(repr(tw).encode()).hexdigest()[:12] if tw else None
        self.recorder.write_meta({
            "schema_version": RunRecorder.SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_sha": git_sha(os.path.dirname(__file__)),
            "base_name": base,
            "ctrl_mode": self.ctrl_mode,
            "dynamic_mode": self.dynamic_mode,
            "ref_util": self.ref_util,
            "ref_dmr": self.ref_dmr,
            "w_sz": self.base_w_sz,
            "exec_model": self.exec_model,
            "mu": self.mu,
            "sigma": self.sigma,
            "num_tasks": len(self.tasks),
            "profile_seed": self.profile_seed,
            "safety_mode": safety_mode,
            "safety": safety_cfg,
            "sim_duration": self.data.get("sim_duration", getattr(self.sim, "duration", None)),
            "taskset_hash": tw_hash,
        })

    def _save_rl_policy(self):
        """Persist the trained RL policy at run end (warm-start source for later runs)."""
        agent = getattr(self, "rl_agent", None)
        if agent is None:
            return
        try:
            agent.save_model()
            logger.warning("[warm-start] saved RL policy -> %s", agent.save_path)
        except Exception as exc:
            logger.warning("Failed to save RL policy: %s", exc)

    def _resolve_dmr_lag(self):
        """Windows to defer release-mode DMR finalization: explicit override, else
        max_i ceil( D_i / w )  over admitted tasks -- the largest number of windows a task's
        relative deadline spans, i.e. how long to wait for a released job's outcome to be
        known (cached after first computation)."""
        if getattr(self, "_dmr_membership", "terminate") != "release":
            return 0
        if self._dmr_lag_explicit is not None:
            return int(self._dmr_lag_explicit)
        cached = getattr(self, "_dmr_lag_cached", None)
        if cached is not None:
            return cached
        w = float(self.base_w_sz)
        active = [t for t in self.tasks if getattr(t._task_info, "active", True)]

        def _deadline(t):
            ti = t._task_info
            return float(getattr(ti, "deadline", None) or getattr(ti, "period", 0) or 0)

        lag = max((int(math.ceil(_deadline(t) / w)) for t in active if _deadline(t) > 0),
                  default=1)
        self._dmr_lag_cached = lag
        logger.warning("[dmr] release-mode finalization lag = %d windows "
                       "(max_i ceil(D_i/w), w=%.0f)", lag, w)
        return lag

    # ---- carry-in loss / adaptive window sizing (paper §VII, Eq. 3 / 9) ------------------

    def _active_admitted(self):
        """Admitted task set T_k for the current window: active, non-background tasks."""
        return [t for t in self.tasks
                if getattr(t._task_info, "active", True) and t._task_info.t_type != 3]

    def _max_releasable_jobs(self, w_sz, tasks=None):
        """N_job = Σ_{τ_i∈T_k} ⌈W / T_i⌉ — the maximum number of jobs the admitted task set
        can release within a window of length ``w_sz`` (paper Eq. 3 denominator). Normalizing
        the miss loss by this (rather than by the jobs actually released) keeps the loss scale
        consistent across windows within a task-set configuration."""
        tasks = self._active_admitted() if tasks is None else tasks
        total = 0
        for t in tasks:
            period = float(getattr(t._task_info, "period", 0) or 0)
            if period > 0:
                total += int(math.ceil(w_sz / period))
        return total

    def _epsilon_carry(self, w_sz, tasks=None):
        """Maximum carry-in loss ε_carry,k = N_k / Σ⌈W/T_i⌉ (paper Prop. 2, Eq. 9). At most one
        carry-in job per admitted task (implicit deadlines), normalized by the max releasable
        jobs. Returns +inf when nothing can be released (the α > ε_carry precondition then fails)."""
        tasks = self._active_admitted() if tasks is None else tasks
        denom = self._max_releasable_jobs(w_sz, tasks)
        n = len(tasks)
        return (n / denom) if denom > 0 else float("inf")

    def _adaptive_window_length(self):
        """Smallest window length in [min_ms, max_ms] whose exact ε_carry < α·(1-margin).
        ε_carry is non-increasing in W (the releasable-job count only grows), so the predicate
        'ε_carry(W) < target' flips false→true exactly once; binary-search that boundary to get
        the smallest (most responsive) window that still satisfies the guarantee precondition."""
        tasks = self._active_admitted()
        target = self._aw_alpha * (1.0 - self._aw_margin)
        lo, hi = max(1, int(self._aw_min)), max(1, int(self._aw_max))
        if lo > hi:
            lo = hi
        # If even the largest allowed window cannot satisfy the precondition, clamp and warn.
        if self._epsilon_carry(hi, tasks) >= target:
            logger.warning("[adaptive-window] cannot reach ε_carry < %.4f within W ≤ %d "
                           "(N=%d); clamping W=%d — Lemma-1 precondition (α > ε_carry) may fail.",
                           target, hi, len(tasks), hi)
            return hi
        while lo < hi:
            mid = (lo + hi) // 2
            if self._epsilon_carry(mid, tasks) < target:
                hi = mid
            else:
                lo = mid + 1
        return max(1, lo)

    def _apply_adaptive_window(self):
        """Recompute and apply the adaptive window length W_k for the upcoming window. Only
        moves self.w_sz when the required length changes (i.e. after the admitted task set or
        its periods change), so a stable task set keeps a stable window."""
        if not getattr(self, "_adaptive_window_enabled", False):
            return
        # Adaptive sizing targets ε_carry < α, a Cases-2&3 (soft) mechanism. For the Case-1
        # hard guarantee (α = 0) no finite W satisfies ε_carry < 0; the carry-in transition is
        # instead handled by the mode-change protocol, so keep the fixed window here.
        if self._aw_alpha <= 0.0:
            return
        new_w = self._adaptive_window_length()
        if new_w != self.w_sz:
            eps = self._epsilon_carry(new_w)
            logger.warning("[adaptive-window] [Window %s] W %d -> %d (N=%d, ε_carry=%.4f, α=%.3f)",
                           self.w_id, self.w_sz, new_w, len(self._active_admitted()), eps,
                           self._aw_alpha)
            self.w_sz = new_w

    def _recorder_flush_ready(self, upto_wid):
        """Stream every finalized window (id <= upto_wid) not yet written."""
        rec = getattr(self, "recorder", None)
        if rec is None:
            return
        guard = getattr(self, "_safety_guard", None)
        gmode = getattr(guard, "mode", "") if guard is not None else ""
        for w in range(self._recorder_flushed_upto + 1, upto_wid + 1):
            entry = self.window_log.get(w)
            if entry is not None:
                try:
                    rec.append_window(w, entry, gmode)
                except Exception as exc:
                    logger.warning("Recorder append failed at window %s: %s", w, exc)
            self._recorder_flushed_upto = w

    def _init_window_sz_config(self):
        # Window size ramp configuration.
        self.w_sz_step_interval = int(self.data.get('w_sz_step_interval', 1))
        self.w_sz_min = max(1, self.base_w_sz // 10)
        interval_windows = max(
            1, int(round(self.env_change_interval_ms / max(1, self.base_w_sz)))
        )
        self.w_sz_ramp_windows = max(1, interval_windows // 4)
        self._w_sz_ramp_steps = max(
            1, math.ceil(self.w_sz_ramp_windows / max(1, self.w_sz_step_interval))
        )
        self._w_sz_step_size = (self.base_w_sz - self.w_sz_min) / self._w_sz_ramp_steps
        self._w_sz_ramp_active = False
        self._w_sz_step_tick = 0
        self._w_sz_ramp_step_idx = 0
        self._env_changed = False

    def _init_task_rngs(self):
        base_seed = 22
        for t in self.tasks:
            tinfo = t._task_info
            seed = base_seed + int(tinfo.identifier)
            tinfo.rng = random.Random(seed)

    def on_activate(self, job):
        # logger.debug("Job %s activated at time %s", job.name, self.sim.now())
        self.ready_list.append(job)
        job.cpu.resched()

    def on_terminated(self, job):
        # logger.debug("Job %s terminated at time %s", job.name, self.sim.now())
        self.ready_list.remove(job)

        self._update_job_log(job)

        # Mode-change protocol (paper §VI-A): a deferred QoS increase takes effect only once
        # EVERY carry-in job from the previous decision has completed — a raise must not add
        # demand while ANY previous-configuration job is still live. So drain the carry-in set
        # and apply all pending increases together when the last one finishes.
        if getattr(self, "_mode_change", False):
            carryin = self._mc_carryin
            if carryin and job in carryin:
                carryin.discard(job)
                if not carryin and self._mc_pending:
                    for t_id, target in self._mc_pending.items():
                        self.tasks[t_id]._task_info.cur_qos = target
                    self._mc_pending = {}

        job.cpu.resched()

    def _live_jobs(self):
        """All jobs currently live (ready or running). Called at a window boundary, these are
        exactly the carry-in jobs released under the previous decision."""
        jobs = list(self.ready_list)
        for cpu in self.processors:
            run = getattr(cpu, "running", None)
            if run is not None and run not in jobs:
                jobs.append(run)
        return jobs

    def schedule(self, cpu):
        # Choose the ready job with the earliest absolute deadline (EDF).
        if self.ready_list:
            job = min(self.ready_list, key=lambda j: j.absolute_deadline)
        else:
            job = None

        if job is not None:
            if cpu.running is not None:
                if cpu.running != job:
                    # Preemption: account for the time the previous job has run
                    # since it was scheduled or since the last window boundary.
                    # preemption occurs
                    cur_job = cpu.running
                    elapsed_time = self.sim.now() - max(cur_job.scheduled, self.window_log.get(self.w_id - 1, {}).get('w_range', (0, 0))[1])

                    self.cur_win_exec_time += elapsed_time
                    cpu.running.task._task_info.cur_win_exec_time += elapsed_time

                else:
                    # Same job continues, no re-schedule needed.
                    # continue execution
                    return (job, cpu)

            
            # if not occupied or preemption occurred, schedule the selected job
            job.scheduled = self.sim.now()
            
            # get updated qos level from task info
            tinfo = job.task._task_info
            cur_qos = tinfo.cur_qos
                        
            # update job's current qos level
            job.cur_qos = cur_qos

            if tinfo.t_type == 1:  # target tasks only
                if self.exec_model == 'gaussian':
                    aet = self._exec_gen_gaussian(
                        tinfo.wcet_table[cur_qos],
                        self.exec_cov,
                        tinfo.rng,
                    )
                else:
                    aet = self._exec_gen_logistic_norm(
                        tinfo.wcet_table[cur_qos],
                        self.mu,
                        self.sigma,
                        tinfo.rng,
                    )
                job.task._task_info.wcet = aet
                # Record true per-job demand (independent of scheduling throttling) so the
                # classifier's est_util reflects real load even when the CPU is saturated.
                if cur_qos < len(tinfo.demand_history):
                    tinfo.demand_history[cur_qos].append(aet)


        return (job, cpu)

    def _update_job_log(self, job):
        # logging
        
        tinfo = job.task._task_info
        
        self.cur_win_job_cnt += 1  # increase job count
        if job.cur_qos != 0:
            job.task._task_info.cur_win_n_jobs += 1

        # carry-in job execution time accumulation
        if job.scheduled < self.window_log.get(self.w_id - 1, {}).get('w_range', (0, 0))[1]:
            self.cur_win_exec_time += self.sim.now() - self.window_log[self.w_id - 1]['w_range'][1]
            if tinfo.t_type == 1: self.cur_win_val += job.cur_qos * (self.sim.now() - self.window_log[self.w_id - 1]['w_range'][1])

            job.task._task_info.cur_win_exec_time += self.sim.now() - self.window_log[self.w_id - 1]['w_range'][1]
        else:
            self.cur_win_exec_time += self.sim.now() - job.scheduled  # accumulate exec time
            if tinfo.t_type == 1: self.cur_win_val += job.cur_qos * (self.sim.now() - job.scheduled)  # accumulate value

            job.task._task_info.cur_win_exec_time += self.sim.now() - job.scheduled

        self.cur_win_qos += job.cur_qos  # accumulate qos

        missed = bool(job.exceeded_deadline)
        if missed:
            self.cur_win_dm_cnt += 1  # increase deadline miss count

        # DMR job membership (independent of exec/util/avg_qos accounting above).
        #  terminate: count this job in the CURRENT (termination) window -- once, here, so no
        #             carry-out double count. cur_win_term_cnt is the DMR denominator.
        #  release:   count this job in its RELEASE window's accumulator.
        self.cur_win_term_cnt += 1
        if self._dmr_membership == 'release':
            rw = int(job.activation_date // self.base_w_sz)
            self._rel_total[rw] = self._rel_total.get(rw, 0) + 1
            if missed:
                self._rel_miss[rw] = self._rel_miss.get(rw, 0) + 1

    def _clear_win_state(self):
        # clear current global window state
        self.cur_win_job_cnt = 0
        self.cur_win_exec_time = 0
        self.cur_win_qos = 0
        self.cur_win_dm_cnt = 0
        self.cur_win_term_cnt = 0  # jobs that terminated in this window (DMR denominator)
        self.cur_win_val = 0

        # clear per-task window state
        for t in self.tasks:
            tinfo = t._task_info
            tinfo.cur_win_exec_time = 0.0
            tinfo.cur_win_n_jobs = 0

    def _timer_logger(self):
        assert (self.w_id in self.window_log) is False, "duplicated window id"

        win_s = self.sim.now() - self.w_sz
        win_e = self.sim.now()

        for cpu in self.processors:
            job = cpu.running
            if not job:
                continue
            # carry-out job
            self.cur_win_exec_time += self.sim.now() - max(job.scheduled, win_s)
            job.task._task_info.cur_win_exec_time += self.sim.now() - max(job.scheduled, win_s)

            self.cur_win_qos += job.cur_qos
            self.cur_win_job_cnt += 1

            if job.cur_qos != 0:
                job.task._task_info.cur_win_n_jobs += 1

        # Deadline-miss ratio, by the configured job-window membership.
        #  terminate: misses / jobs that TERMINATED in this window (cur_win_term_cnt is the
        #             carry-out-free denominator; cur_win_dm_cnt already counts only terminations).
        #  release:   provisional here (jobs released in this window may still be running and
        #             miss later); finalized from self._rel_* with a lag in the controller.
        if self._dmr_membership == 'release':
            rw = self.w_id
            tot = self._rel_total.get(rw, 0)
            dmr = (self._rel_miss.get(rw, 0) / tot) if tot > 0 else 0
            n_jobs, n_miss = tot, self._rel_miss.get(rw, 0)
        else:
            n_jobs, n_miss = self.cur_win_term_cnt, self.cur_win_dm_cnt
            dmr = (self.cur_win_dm_cnt / self.cur_win_term_cnt) if self.cur_win_term_cnt > 0 else 0
        # add w_id window log
        # Max releasable jobs and carry-in loss for this window (paper Eq. 3 / Prop. 2). Uses
        # this window's own length (self.w_sz), which may vary under adaptive window sizing.
        _admitted = self._active_admitted()
        n_job_releasable = self._max_releasable_jobs(self.w_sz, _admitted)
        eps_carry = (len(_admitted) / n_job_releasable) if n_job_releasable > 0 else float("inf")

        self.window_log[self.w_id] = {
            "w_range": (win_s, win_e),  # cur_time - win_sz ~ cur_time
            "w_sz": self.w_sz,          # window length used for this window (may vary, §VII-C)
            "act_util": (self.cur_win_exec_time / self.w_sz),
            "n_jobs": n_jobs,
            "n_miss_jobs": n_miss,
            "n_job_releasable": n_job_releasable,  # Σ⌈W/T_i⌉ (Eq. 3 denominator)
            "eps_carry": eps_carry,                # ε_carry,k max carry-in loss (Prop. 2, Eq. 9)
            "dmr": dmr,
            "avg_qos": (self.cur_win_qos / self.cur_win_job_cnt) if self.cur_win_job_cnt > 0 else 0,
            "util_err": self.ref_util - (self.cur_win_exec_time / self.w_sz),
            "value": self.cur_win_val,
            "ref_util": self.ref_util,
        }

        for t in self.tasks:
            tinfo = t._task_info
            tinfo.win_log[self.w_id] = {
                "exec_time": tinfo.cur_win_exec_time,
                "n_jobs": tinfo.cur_win_n_jobs,
                "qos_level": tinfo.cur_qos,
            }
            if tinfo.cur_win_n_jobs > 0:
                tinfo.exec_history[tinfo.cur_qos].append(
                    tinfo.cur_win_exec_time / tinfo.cur_win_n_jobs
                )

        for t in self.tasks:
            tinfo = t._task_info
            tlog = tinfo.win_log[self.w_id]
            logger.debug(
                "[Window %s][Task %s] Exec Time: %.4f, #Jobs: %d, QoS Level: %d",
                self.w_id,
                t.name,
                tlog["exec_time"],
                tlog["n_jobs"],
                tlog["qos_level"],
            )

        # clear window state
        self._clear_win_state()

        self.w_id += 1  # update window id

    def _qos_manager(self):
        # last window util, dmr collect,
        # compute error with ref
        if self.ctrl_mode == 'fcs':
            self._fcs_qos_mgmt()
        if self.ctrl_mode == 'rl':
            self._rl_qos_mgmt()
        if self.ctrl_mode == 'profile':
            self._profile_qos_mgmt()

        last_win = self.w_id - 1
        # logger info print global
        logger.info(
            f"{_WINDOW_SUMMARY_COLOR}[Window][Summary %04d] util=%.4f dmr=%.4f avg_qos=%.4f value=%04f reward=%.4f{_WINDOW_SUMMARY_RESET}",
            last_win,
            self.window_log[last_win]["act_util"],
            self.window_log[last_win]["dmr"],
            self.window_log[last_win]["avg_qos"],
            self.window_log[last_win]["value"],
            self.window_log[last_win].get("reward", 0.0),
        )

        # Stream finalized window(s) to windows.csv (crash-resilient). Under release-mode DMR
        # the recorded DMR / guard threshold are only finalized after the lag, so hold the flush
        # back by that many windows (terminate mode: lag 0, flushes immediately as before).
        self._recorder_flush_ready(last_win - int(self._resolve_dmr_lag()))

    def _timer_func(self):
        # logging
        self._timer_logger()

        # QoS management
        self._qos_manager()

        # dynamic environment
        if self.dynamic_changes:
            self._apply_scheduled_changes()          # schedule-driven (preferred)
        elif self.dynamic_mode > 0:                  # legacy interval-driven schedules
            if self.env_change_interval_cycles > 0 and self.sim.now() >= self._next_env_change_time:
                self._update_env_settings()
                while self._next_env_change_time <= self.sim.now():
                    self._next_env_change_time += self.env_change_interval_cycles

        if self._env_changed and self.ctrl_mode == 'rl':
            self.rl_adaptation_on_env_change()
            self._env_changed = False

        # Adaptive window sizing (§VII-C): resize W_k for the upcoming window from the current
        # admitted task set. No-op unless enabled; only changes W when the task set/periods change.
        if getattr(self, "_adaptive_window_enabled", False):
            self._apply_adaptive_window()

        # restart timer
        Timer(self.sim, self._timer_func, (), delay=self.w_sz, one_shot=True, prior=True, overhead=0).start()

    def _update_env_settings(self):
        # dynamic mode 1: change reference utilization 2: change available budget 3: change task set

        if self.dynamic_mode == 0:
            return
        
        if self.dynamic_mode == 1:
            self.ref_util = self.ref_util_schedule[self._ref_util_idx % len(self.ref_util_schedule)]
            self._ref_util_idx += 1

        elif self.dynamic_mode == 2:
            background_util = self.back_util_schedule[self._back_util_idx % len(self.back_util_schedule)]
            self._back_util_idx += 1
            
            # check existing background tasks and update their utilization            
            for t in self.sim.task_list:
                tinfo = t._task_info
                if tinfo.t_type == 3:  # background task
                    tinfo.wcet = background_util * tinfo.period
                    tinfo.wcet_table = {0: tinfo.wcet}
            

        elif self.dynamic_mode == 3:
            # Determine how many tasks change and apply changes to wcet/period/deadline.
            
            # 일부 task들의 type를 2로 변경하여 신규 task로 간주
            # 신규 task들의 wcet/period/deadline를 변경
            ntask, util = self.new_util_schedule[self._new_util_idx % len(self.new_util_schedule)]
            self._new_util_idx += 1
            
            # exist paramter bug in simso generator function
            # u and n are swapped in function definition
            max_qos_utils = sims.task_generator.gen_uunifastdiscard(nsets = 1, u = int(ntask), n= util)[0]
            qos_utils = self.gen_utils_qos(max_qos_utils, self.tasks[0]._task_info.max_qos + 1, qos_policy='linear')
            
            index = 0
            for t_id, t in enumerate(self.tasks):
                if t._task_info.identifier >= ntask:
                    if self.ctrl_mode == 'fcs':
                        t._task_info.t_type = 2  # mark as newly added task
                    else:
                        t._task_info.t_type = 1  # under rl controller, treat as target task
                                        
                    t._task_info.wcet = round(max_qos_utils[index] * t._task_info.period, 3)
                    t._task_info.wcet_table = {q: round(u * t._task_info.period, 3) for q, u in enumerate(qos_utils[index])}
                    t._task_info.cur_qos = t._task_info.max_qos  # set to max qos initially
                    index += 1
            
            # for t_id in reversed(range(len(self.tasks))):
            #     t = self.tasks[t_id]
            #     if t_id < ntask:
            #         tinfo = t._task_info
            #         if self.ctrl_mode == 'fcs':
            #             tinfo.t_type = 2  # mark as newly added task
            #         else:
            #             tinfo.t_type = 1  # under rl controller, treat as target task
                                        
            #         tinfo.wcet = round(max_qos_utils[t_id] * tinfo.period, 3)
            #         tinfo.wcet_table = {q: round(u * tinfo.period, 3) for q, u in enumerate(qos_utils[t_id])}
            #         tinfo.cur_qos = tinfo.max_qos  # set to max qos initially
            #         print(tinfo.wcet_table)

        self.env_change_times.append(self.sim.now())
        self._env_changed = True

    # ---- schedule-driven dynamic changes (ref-util + task-set add/remove/replace) --------

    def active_tasks(self):
        return [t for t in self.tasks if t._task_info.active]

    def _apply_scheduled_changes(self):
        for ev in self.dynamic_changes:
            if not ev.get('_applied') and self.w_id >= int(ev.get('at_window', 0)):
                self._apply_change(ev)
                ev['_applied'] = True

    def _apply_change(self, ev):
        kind = str(ev.get('kind', 'ref_util')).lower()
        desc = kind
        if kind == 'ref_util':
            self.ref_util = float(ev.get('value', self.ref_util))
            desc = f"ref_util -> {self.ref_util}"
        elif kind == 'taskset':
            op = str(ev.get('op', 'add')).lower()
            count = int(ev.get('count', 1))
            util = float(ev.get('util', 0.15))
            if op == 'add':
                n = self._taskset_add(count, util); desc = f"add {n} task(s) util={util}"
            elif op == 'remove':
                n = self._taskset_remove(count); desc = f"remove {n} task(s)"
            elif op == 'replace':
                n = self._taskset_replace(count, util); desc = f"replace {n} task(s) util={util}"
        n_active = len(self.active_tasks())
        self.env_change_times.append(self.sim.now())
        self.change_log.append((self.w_id, f"{desc} | active={n_active}"))
        self._env_changed = True
        logger.warning("[Dynamic][Window %04d] %s | active_tasks=%s", self.w_id, desc, n_active)

    def _set_task_util(self, tinfo, max_util):
        """Regenerate a task's wcet_table for a target max-QoS utilization and reset its
        runtime exec history (so estimates reflect the new parameters, profile-free)."""
        qos_level = tinfo.max_qos + 1
        row = self.gen_utils_qos([float(max_util)], qos_level, 'linear')[0]
        tinfo.wcet_table = {q: round(u * tinfo.period, 3) for q, u in enumerate(row)}
        tinfo.wcet = tinfo.wcet_table[tinfo.max_qos]
        tinfo.exec_history = [[] for _ in range(qos_level)]
        tinfo.demand_history = [[] for _ in range(qos_level)]

    def _taskset_add(self, count, util):
        dormant = [t for t in self.tasks if not t._task_info.active]
        chosen = dormant[:count]
        for t in chosen:
            self._set_task_util(t._task_info, util)
            t._task_info.active = True
            t._task_info.cur_qos = 0  # controller raises it from here
        return len(chosen)

    def _taskset_remove(self, count):
        active = self.active_tasks()
        removable = active[-count:] if len(active) - count >= 1 else active[1:]  # keep >=1 active
        for t in removable:
            t._task_info.active = False
            t._task_info.cur_qos = 0
        return len(removable)

    def _taskset_replace(self, count, util):
        return min(self._taskset_remove(count), self._taskset_add(count, util))

