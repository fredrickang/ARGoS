try:
    from simso.core import TaskInfo
except ImportError:
    from simso.core.Task import TaskInfo

from typing import Any, Iterable, Optional

# WCET according to the QoS level 
class CustomTaskInfo(TaskInfo):
    def __init__(
        self,
        name: str,
        identifier: Any,
        task_type: Any,
        abort_on_miss: bool,
        period: Any,
        activation_date: Any,
        n_instr: Any,
        mix: Any,
        stack_file: Optional[str],
        wcet: Any,
        acet: Any,
        et_stddev: Any,
        deadline: Any,
        base_cpi: Any,
        followed_by: Any,
        list_activation_dates: Optional[Iterable[Any]],
        preemption_cost: Any,
        data: Any,
        *,
        
        # custom fields
        wcet_table: Any,
        max_qos: Any,
        cur_qos: Any,
        t_type: Any, # 1: target tasks 2: newerly added tasks 3: background tasks
        active: bool = True,  # False = dormant pool slot (masked to QoS 0 until "added")
    ) -> None:
        super().__init__(
            name,
            identifier,
            task_type,
            abort_on_miss,
            period,
            activation_date,
            n_instr,
            mix,
            (stack_file, "."),
            wcet,
            acet,
            et_stddev,
            deadline,
            base_cpi,
            followed_by,
            list_activation_dates,
            preemption_cost,
            data,
        )
        self.wcet_table = wcet_table  # dict of WCETs per QoS level
        self.max_qos = max_qos        # Maximum QoS level   
        self.cur_qos = cur_qos        # Current QoS level   
        self.cur_win_exec_time = 0.0
        self.cur_win_n_jobs = 0
        self.t_type = t_type            # Task type: 1, 2, or 3
        self.active = bool(active)      # dynamic add/remove: dormant slots start inactive

        self.win_log = {}  # For logging window-level data, window 별 qos history, window 동안 execute된 양, job count 
        
        self.exec_history = [[] for _ in range(max_qos + 1)]  # delivered CPU time / job per QoS (throttled under saturation)
        self.demand_history = [[] for _ in range(max_qos + 1)]  # sampled per-job execution DEMAND per QoS (not throttled)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={getattr(self, 'name', None)!r})"
