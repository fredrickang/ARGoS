#!/usr/bin/env python
"""PATROL simulation entry point.

Usage:
    python run.py [config.yaml]

The config path may also be provided via the ``SIM_CONFIG_PATH`` environment
variable; it defaults to ``./configs/run_sim.yaml``.
"""

import logging
import random

from simso.core import Model

from patrol.sim.config import (
    build_config,
    build_plot_name_from_config,
    load_sim_config,
    resolve_config_path,
)


logging.basicConfig(
    level=logging.ERROR,  # DEBUG/INFO/WARNING/ERROR
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def main(argv=None):
    # Preserve the original global RNG seeding so task-set generation is deterministic.
    random.seed(22)

    sim_cfg = load_sim_config(resolve_config_path(argv))
    plot_name = build_plot_name_from_config(sim_cfg)
    window_log_dir = sim_cfg.get("output", {}).get("dir", "outputs/window_logs")
    sim_duration = sim_cfg["simulation"]["duration_ms"]

    cfg = build_config(
        sim_cfg["scheduler"]["filename"],
        sim_cfg["scheduler"]["classname"],
        sim_cfg["taskset"]["path"],
        sim_cfg["qos"]["level"],
        sim_cfg["qos"]["policy"],
        mu=sim_cfg["exec_time"]["mu"],
        sigma=sim_cfg["exec_time"]["sigma"],
        exec_model=sim_cfg["exec_time"].get("model", "logistic"),
        exec_cov=sim_cfg["exec_time"].get("cov"),
        num_uncontrollable=sim_cfg.get("uncontrollable", {}).get("num_tasks", 0),
        uncontrollable_util=sim_cfg.get("uncontrollable", {}).get("util", 0.0),
        qos_min_frac=sim_cfg["qos"].get("min_frac", 0.0),
        controllable_cmax=sim_cfg.get("controllable", {}).get("cmax"),
        w_sz=sim_cfg["window"]["size_ms"],
        period_range=tuple(sim_cfg["taskset"].get("period_range", (500, 1000))),
        ctrl=sim_cfg["controller"],
        fcs_ctrl_param=sim_cfg["fcs_ctrl_param"],
        dynamic_mode=sim_cfg["controller"].get("dynamic_mode", 1),
        dynamic_config=sim_cfg.get("dynamic_config", {}),
        window_log_dir=window_log_dir,
        sim_duration=sim_duration,
        plot_name=plot_name,
    )

    model = Model(cfg)
    model.run_model()


if __name__ == "__main__":
    main()
