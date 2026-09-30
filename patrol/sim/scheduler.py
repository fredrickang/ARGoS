import numpy as np
import random

from simso.core import Scheduler

from patrol.sim.core import SchedulerCoreMixin
from patrol.controllers.fcs import FcsControllerMixin
from patrol.controllers.rl import RlControllerMixin
from patrol.controllers.profile import ProfileControllerMixin
from patrol.sim.metrics import UtilMixin


random.seed(22)
np.random.seed(22)


class EdfQosScheduler(SchedulerCoreMixin, FcsControllerMixin, RlControllerMixin, ProfileControllerMixin, UtilMixin, Scheduler):
    pass


# SimSo's file-based loader resolves the scheduler class by the *module file stem*
# (see simso.core.Scheduler.get_cls), not by the config's `classname`. Since this file
# is `scheduler.py`, expose the class under that name so the loader finds it.
scheduler = EdfQosScheduler

# Backwards-compatible alias for the original class name.
edf_mono = EdfQosScheduler
