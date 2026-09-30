"""Task-set generation and SimSo configuration building.

Extracted from the original ``run_sim.py``. Behaviour is preserved; the only
structural change is that the simulation duration is passed explicitly to
:func:`build_config` instead of relying on a module-level ``SIM_DURATION``.
"""

from pathlib import Path
import logging
import os
import random
import re
import sys

import yaml

import simso.generator.task_generator as sims
from simso.configuration import Configuration

from patrol.sim.task_info import CustomTaskInfo


logger = logging.getLogger(__name__)


def gen_utils_qos_extend(utils, n, min_util, qos_level, qos_policy):
    # split the task utils into 2 groups: first n tasks and the rest
    group1 = utils[:n]
    group2 = utils[n:]

    # first group util is extended to at least min_util, with the same qos levels as before

    # generate lower bound of utils for the first group
    ## caution: gen_uunifastdiscard includes a bug (reversed order of parameters between u and n), so we need to put n and min_util in the reverse order
    uunifast_utils = sims.gen_uunifastdiscard(nsets=1, u=n, n=min_util)[0]

    # generate intermediate utils for the first group based on the original utils and the uunifast_utils
    # min_qos utils uunifast and max_qos utils original, with linear interpolation for intermediate qos levels
    qos_utils_group1 = []
    for i in range(n):
        task_utils = []
        for q in range(qos_level):
            if qos_level <= 1:
                task_utils.append(utils[i])
            else:
                # linear interpolation between uunifast_util and original util
                u_min = uunifast_utils[i]
                u_max = utils[i]
                step = (u_max - u_min) / (qos_level - 1)
                task_utils.append(u_min + step * q)
        qos_utils_group1.append(task_utils)

    # second group util follows basic_gen_utils_qos
    qos_utils_group2 = gen_utils_qos(group2, qos_level, qos_policy)

    # combine the two groups
    qos_utils = qos_utils_group1 + qos_utils_group2
    return qos_utils


def gen_utils_qos(utils, qos_level, qos_policy, min_frac=0.0):
    # Build per-task utilization tables; result is indexed as tasks[task_idx][qos].
    if qos_level <= 1:
        return [[u] for u in utils]

    if min_frac and min_frac > 0.0:
        # Minimum service level: every QoS level is nonzero, spanning [min_frac*u, u].
        # (The lowest QoS still consumes CPU -> the task can never be turned fully off.)
        def levels_min(u):
            lo = min_frac * u
            step = (u - lo) / (qos_level - 1)
            return [round(lo + step * i, 6) for i in range(qos_level)]
        return [levels_min(u) for u in utils]

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


def task_generator(path="", qos_level=3, qos_policy="linear",
                   num_uncontrollable=0, uncontrollable_util=0.0, qos_min_frac=0.0,
                   controllable_cmax=None, pool_extra=0, pool_default_util=0.15,
                   period_range=(500, 1000)):
    # Read per-task utilization targets from a CSV-like file.
    # Tasks from the file are the CONTROLLABLE (profiled) tasks that FCS adjusts.
    # `num_uncontrollable` extra tasks with no profile run at a fixed QoS and together
    # generate a total utilization of `uncontrollable_util` (the "uncontrollable workload").
    p = Path(path)
    if not p.exists():
        logger.error("Task file not found: %s", p)
        return None

    with p.open("r", encoding="utf-8") as f:
        content = f.read()
    utils = [float(x.strip()) for x in content.split(",") if x.strip()]
    ntasks = len(utils)

    # Split mode: when num_uncontrollable > 0, the last `num_uncontrollable` slots of the
    # (10-task) set become no-profile uncontrollable tasks whose total utilization is set to
    # `uncontrollable_util`; the first n_ctrl tasks stay controllable (FCS-adjusted, from file).
    num_uncontrollable = int(num_uncontrollable or 0)
    n_ctrl = max(0, ntasks - num_uncontrollable) if num_uncontrollable > 0 else ntasks

    # Optionally rescale the controllable tasks so their max-QoS total equals a fixed
    # C_max (the "specified capacity" case). Relative proportions from the file are kept.
    if controllable_cmax and controllable_cmax > 0 and n_ctrl > 0:
        s = sum(utils[:n_ctrl])
        if s > 0:
            f = float(controllable_cmax) / s
            utils = [u * f for u in utils[:n_ctrl]] + utils[n_ctrl:]

    # Build QoS utilization table per task.
    # qos_utils = gen_utils_qos_extend(utils, ntasks//2, min_util=0.2, qos_level=qos_level, qos_policy=qos_policy)
    qos_utils = gen_utils_qos(utils, qos_level, qos_policy, min_frac=qos_min_frac)

    # Pick a random period for each task. Range is configurable (taskset.period_range) so the
    # relative deadlines D_i -- and hence the release-mode lag k=max ceil(D_i/W) -- can be varied.
    period_range = tuple(period_range)
    periods = [random.randint(period_range[0], period_range[1]) for _ in range(ntasks)]

    # util = wcet / period  =>  wcet = util * period, per task and QoS level
    wcet_tables = [
        [round(u * periods[t_id], 3) for u in task_utils]
        for t_id, task_utils in enumerate(qos_utils)
    ]

    tinfo_list = []
    # Emit CustomTaskInfo for each controllable (profiled) task.
    for t_id in range(n_ctrl):
        tinfo_list.append(
            CustomTaskInfo(
                name=f"T{t_id+1}",
                identifier=t_id,
                task_type="Periodic",
                abort_on_miss=False,
                period=periods[t_id],
                activation_date=0,
                n_instr=0,
                mix=0.5,
                stack_file="",
                wcet=wcet_tables[t_id][0],  # max QoS wcet
                acet=0,
                et_stddev=0,
                deadline=periods[t_id],
                base_cpi=1.0,
                followed_by=None,
                list_activation_dates=None,
                preemption_cost=0,
                data=None,
                wcet_table={q: wcet_tables[t_id][q] for q in range(qos_level)},
                max_qos=qos_level - 1,
                # cur_qos=qos_level -1,
                cur_qos=0,
                t_type=1,  # controllable (profiled) target task
            )
        )

    # Dormant pool tasks (for dynamic "add"): controllable target tasks (t_type=1) created
    # up front but INACTIVE. They sit at QoS 0 (inert, zero load) until an "add" event
    # activates them. Pre-provisioning them keeps the RL action space fixed while still
    # letting the active task count grow/shrink at runtime.
    pool_extra = int(pool_extra or 0)
    if pool_extra > 0:
        pool_qos = gen_utils_qos([pool_default_util] * pool_extra, qos_level, qos_policy,
                                 min_frac=qos_min_frac)
        for k in range(pool_extra):
            period = random.randint(period_range[0], period_range[1])
            wtab = {q: round(u * period, 3) for q, u in enumerate(pool_qos[k])}
            tinfo_list.append(
                CustomTaskInfo(
                    name=f"P{k+1}", identifier=n_ctrl + k, task_type="Periodic",
                    abort_on_miss=False, period=period, activation_date=0, n_instr=0,
                    mix=0.5, stack_file="", wcet=wtab[qos_level - 1], acet=0, et_stddev=0,
                    deadline=period, base_cpi=1.0, followed_by=None,
                    list_activation_dates=None, preemption_cost=0, data=None,
                    wcet_table=wtab, max_qos=qos_level - 1, cur_qos=0,
                    t_type=1, active=False,  # dormant until "added"
                )
            )

    # Uncontrollable (no-profile) tasks: fixed single QoS level, total util = uncontrollable_util.
    # FCS never adjusts these (t_type != 1); they contribute a fixed workload the controller
    # must work around using only the controllable tasks above.
    if num_uncontrollable > 0:
        if uncontrollable_util and uncontrollable_util > 0:
            # gen_uunifastdiscard has swapped (u, n) params in simso -> pass count as u, util as n.
            unc_utils = sims.gen_uunifastdiscard(
                nsets=1, u=num_uncontrollable, n=float(uncontrollable_util)
            )[0]
        else:
            unc_utils = [0.0] * num_uncontrollable
        for j in range(num_uncontrollable):
            period = random.randint(500, 1000)
            wcet = round(unc_utils[j] * period, 3)
            tinfo_list.append(
                CustomTaskInfo(
                    name=f"U{j+1}",
                    identifier=n_ctrl + pool_extra + j,
                    task_type="Periodic",
                    abort_on_miss=False,
                    period=period,
                    activation_date=0,
                    n_instr=0,
                    mix=0.5,
                    stack_file="",
                    wcet=wcet,
                    acet=0,
                    et_stddev=0,
                    deadline=period,
                    base_cpi=1.0,
                    followed_by=None,
                    list_activation_dates=None,
                    preemption_cost=0,
                    data=None,
                    wcet_table={0: wcet},
                    max_qos=0,
                    cur_qos=0,
                    t_type=2,  # uncontrollable (no profile), fixed QoS
                )
            )

    # background tasks
    tinfo_list.append(
        CustomTaskInfo(
            name="Background",
            identifier=n_ctrl + pool_extra + num_uncontrollable + 1,
            task_type="Periodic",
            abort_on_miss=False,
            period=500,
            activation_date=0,
            n_instr=0,
            mix=0.5,
            stack_file="",
            wcet=0,
            acet=0,
            et_stddev=0,
            deadline=500,
            base_cpi=1.0,
            followed_by=None,
            list_activation_dates=None,
            preemption_cost=0,
            data=None,
            wcet_table={0: 0},
            max_qos=0,
            cur_qos=0,
            t_type=3,  # background task
        )
    )

    return tinfo_list


def build_config(
    scheduler_filename,
    scheduler_classname,
    taskset_path,
    qos_level,
    qos_policy,
    mu,
    sigma,
    w_sz,
    ctrl,
    fcs_ctrl_param,
    dynamic_mode,
    dynamic_config,
    window_log_dir,
    sim_duration,
    exec_model="logistic",
    exec_cov=None,
    num_uncontrollable=0,
    uncontrollable_util=0.0,
    qos_min_frac=0.0,
    controllable_cmax=None,
    plot_name=None,
    period_range=(500, 1000),
):
    # Create an empty SimSo configuration object.
    cfg = Configuration()

    # Register scheduler implementation (filename/classname).
    cfg.scheduler_info.filename = scheduler_filename
    cfg.scheduler_info.classname = scheduler_classname

    # Set simulation time scale and total duration in cycles.
    cfg.cycles_per_ms = 1
    cfg.duration = sim_duration * cfg.cycles_per_ms

    # Define CPU core(s) available to the simulator.
    cfg.add_processor(name="CPU_0", identifier=0, speed=1)

    # Load/generate task set and register it in the configuration.
    _dyn = dynamic_config or {}
    pool_extra = int((_dyn.get("dynamic") or {}).get("pool_extra", _dyn.get("pool_extra", 0)) or 0)
    tasks = task_generator(
        path=taskset_path, qos_level=qos_level, qos_policy=qos_policy,
        num_uncontrollable=num_uncontrollable, uncontrollable_util=uncontrollable_util,
        qos_min_frac=qos_min_frac, controllable_cmax=controllable_cmax,
        pool_extra=pool_extra, period_range=period_range,
    )
    for t in tasks:
        cfg.task_info_list.append(t)

    # Scheduler parameter sending
    cfg.scheduler_info.data['w_sz'] = w_sz  # window size in ms
    cfg.scheduler_info.data['mu'] = mu
    cfg.scheduler_info.data['sigma'] = sigma

    # Execution-time noise model ('logistic' | 'gaussian') and CoV for the gaussian model.
    cfg.scheduler_info.data['exec_model'] = exec_model
    cfg.scheduler_info.data['exec_cov'] = sigma if exec_cov is None else exec_cov

    # controller parameter
    cfg.scheduler_info.data['ctrl'] = ctrl
    cfg.scheduler_info.data['dynamic_mode'] = dynamic_mode

    # FCS controller parameters
    cfg.scheduler_info.data['fcs_ctrl_param'] = fcs_ctrl_param
    cfg.scheduler_info.data['window_log_dir'] = window_log_dir
    if plot_name is not None:
        cfg.scheduler_info.data['window_log_png'] = plot_name

    # Dynamic configuration
    cfg.scheduler_info.data['dynamic_config'] = dynamic_config

    # Validate configuration consistency before returning.
    cfg.check_all()
    return cfg


def load_sim_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _slugify(text):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_")


def build_plot_name_from_config(sim_cfg):
    taskset_name = Path(sim_cfg["taskset"]["path"]).stem
    mode = sim_cfg["controller"]["mode"]
    dynamic_mode = sim_cfg["controller"].get("dynamic_mode", 1)
    fcs_cfg = sim_cfg.get("fcs_ctrl_param", {})
    similarity = fcs_cfg.get("similarity")
    similarity_metric = fcs_cfg.get("similarity_metric")
    qos_level = sim_cfg["qos"]["level"]
    qos_policy = sim_cfg["qos"]["policy"]
    mu = sim_cfg["exec_time"]["mu"]
    sigma = sim_cfg["exec_time"]["sigma"]
    w_sz = sim_cfg["window"]["size_ms"]
    duration = sim_cfg["simulation"]["duration_ms"]
    scheduler = Path(sim_cfg["scheduler"]["filename"]).stem
    sim_tag = ""
    if similarity_metric is not None and similarity is not None:
        try:
            sim_value = float(similarity)
            sim_tag = f"{similarity_metric}_{sim_value:.1f}_"
        except (TypeError, ValueError):
            sim_tag = f"{similarity_metric}_{similarity}_"
    name = (
        f"{scheduler}_"
        f"{mode}_"
        f"dyn{dynamic_mode}_"
        f"{sim_tag}"
        f"{taskset_name}_"
        f"q{qos_level}_{qos_policy}_"
        f"mu{mu}_sig{sigma}_"
        f"w{w_sz}_d{duration}.png"
    )
    return _slugify(name)


def resolve_config_path(argv=None):
    """Config path from CLI arg, else ``SIM_CONFIG_PATH`` env, else default."""
    argv = sys.argv if argv is None else argv
    if len(argv) > 1 and argv[1]:
        return argv[1]
    return os.environ.get("SIM_CONFIG_PATH", "./configs/run_sim.yaml")
