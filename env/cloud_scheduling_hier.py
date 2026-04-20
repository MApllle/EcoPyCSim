import numpy as np

from helper.create_jobs import initialize_user_requests_queue
from helper.create_server_farm import initialize_server_farms
from components.timeline import Timeline, TimelineEvent

from gymnasium import spaces
from pettingzoo import ParallelEnv


class CloudSchedulingEnvHier(ParallelEnv):
    """N+1 agent hierarchical cloud scheduling environment.

    Agents:
      - "global": selects which server farm (Discrete(num_server_farms))
      - "local_i": selects which server within farm i (Discrete(num_servers_per_farm))

    Sequential decision semantics per step:
      1. global selects farm_id
      2. only local_{farm_id}'s action takes effect; other locals are ignored
    """

    def __init__(
        self,
        num_jobs,
        num_server_farms,
        num_servers,
        alpha_energy=1.0,
        beta_task=0.5,
        gamma_lb=0.1,
        render_mode=None,
        use_heterogeneity: bool = True,
        hetero_weight: float = 0.3,
        server_proportions=None,
        **kwargs,
    ):
        assert num_servers % num_server_farms == 0, (
            "num_servers must be evenly divisible by num_server_farms"
        )
        self.num_jobs = num_jobs
        self.num_server_farms = num_server_farms
        self.num_servers = num_servers
        self.num_servers_per_farm = num_servers // num_server_farms

        self.alpha_energy = alpha_energy
        self.beta_task = beta_task
        self.gamma_lb = gamma_lb

        self.use_heterogeneity = use_heterogeneity
        self.hetero_weight = hetero_weight
        self.server_proportions = server_proportions
        self.render_mode = render_mode

        self.global_agent_id = "global"
        self.local_agent_ids = [f"local_{i}" for i in range(num_server_farms)]
        self.agents = [self.global_agent_id] + self.local_agent_ids
        self.possible_agents = self.agents[:]

        self._selected_farm_id = 0

        self.wall_time = 0
        self.timeline = Timeline()
        self.jobs = {}
        self.server_farms = {}

        self.handle_event = {
            TimelineEvent.Type.JOB_ARRIVAL: self._handle_job_arrival,
            TimelineEvent.Type.TASK_ARRIVAL: self._handle_task_arrival,
            TimelineEvent.Type.TASK_DEPARTURE: self._handle_task_departure,
        }

        self._last_hetero_reward = 0.0
        self._her_numerator = 0.0
        self._her_denominator = 0.0

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    def observation_space(self, agent):
        n = self.num_server_farms
        m = self.num_servers_per_farm
        if agent == self.global_agent_id:
            return spaces.Dict({
                "farm_avg_cpu_util":   spaces.Box(0, 1, shape=(n,), dtype=float),
                "farm_active_ratio":   spaces.Box(0, 1, shape=(n,), dtype=float),
                "farm_avg_efficiency": spaces.Box(0, 1, shape=(n,), dtype=float),
                "task_cpu":      spaces.Box(0, 1, shape=(1,), dtype=float),
                "task_ram":      spaces.Box(0, 1, shape=(1,), dtype=float),
                "task_deadline": spaces.Box(0, np.inf, shape=(1,), dtype=float),
                "wall_time":     spaces.Box(0, np.inf, shape=(1,), dtype=float),
            })
        else:
            return spaces.Dict({
                "cpus_utilization": spaces.Box(0, 1, shape=(m,), dtype=float),
                "efficiency_tiers": spaces.Box(0, 1, shape=(m,), dtype=float),
                "task_cpu":      spaces.Box(0, 1, shape=(1,), dtype=float),
                "task_ram":      spaces.Box(0, 1, shape=(1,), dtype=float),
                "task_deadline": spaces.Box(0, np.inf, shape=(1,), dtype=float),
                "wall_time":     spaces.Box(0, np.inf, shape=(1,), dtype=float),
                "is_selected":   spaces.Box(0, 1, shape=(1,), dtype=float),
            })

    def action_space(self, agent):
        if agent == self.global_agent_id:
            return spaces.Discrete(self.num_server_farms)
        else:
            return spaces.Discrete(self.num_servers_per_farm)

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.wall_time = 0
        self.time_limit = 0
        self.timeline.reset()

        self.jobs.clear()
        self.server_farms.clear()

        job_sequence = initialize_user_requests_queue(self.num_jobs, seed)
        for timeline, job in job_sequence:
            self.timeline.push(
                timeline,
                TimelineEvent(TimelineEvent.Type.JOB_ARRIVAL, data={"job": job}),
            )
            self.jobs[job.id] = job

        server_farms = initialize_server_farms(
            self.num_servers, self.num_server_farms, seed,
            server_proportions=self.server_proportions,
        )
        for sf in server_farms:
            self.server_farms[sf.id] = sf

        self._selected_farm_id = 0
        self.server_id = 0

        self.prev_total_energy = 0.0
        self.prev_rejected_tasks_count = 0
        self.prev_completed_jobs_count = 0
        self._last_hetero_reward = 0.0
        self._her_numerator = 0.0
        self._her_denominator = 0.0

        self.active_job_ids = []
        self.completed_job_ids = set()
        self.rejected_job_ids = set()
        self.rejected_tasks_count = 0
        self.task_rejected_status = False

        self.schedulable_tasks = False
        self.scheduled_tasks = set()

        self.scheduled_task_cpu = 0
        self.scheduled_task_ram = 0
        self.scheduled_task_deadline = 0

        self._load_initial_jobs()

        obs = {a: self._get_observation(a) for a in self.agents}
        infos = {a: self._get_info(a) for a in self.agents}
        return obs, infos

    def step(self, actions):
        if self.schedulable_tasks:
            farm_id = int(actions[self.global_agent_id])
            self._selected_farm_id = farm_id
            server_id = int(actions[f"local_{farm_id}"])

            routed = {"server_farm": farm_id, "server": server_id}
            self._take_action(routed)

            obs   = {a: self._get_observation(a) for a in self.agents}
            rew   = self._get_rewards()
            term  = {a: False for a in self.agents}
            trunc = {a: False for a in self.agents}
            info  = {a: self._get_info(a) for a in self.agents}

            self.scheduled_task_deadline = 0
            self.task_rejected_status = False
            return obs, rew, term, trunc, info

        self._resume_simulation()

        obs   = {a: self._get_observation(a) for a in self.agents}
        rew   = {a: 0 for a in self.agents}
        term  = {a: False for a in self.agents}
        if self.all_jobs_complete:
            term = {a: True for a in self.agents}
        trunc = {a: False for a in self.agents}
        info  = {a: self._get_info(a) for a in self.agents}
        return obs, rew, term, trunc, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observation(self, agent):
        task_cpu      = np.array([self.scheduled_task_cpu],      dtype=float)
        task_ram      = np.array([self.scheduled_task_ram],      dtype=float)
        task_deadline = np.array([self.scheduled_task_deadline], dtype=float)
        wall_time     = np.array([self.wall_time],               dtype=float)

        if agent == self.global_agent_id:
            farm_avg_cpu   = np.zeros(self.num_server_farms, dtype=float)
            farm_active    = np.zeros(self.num_server_farms, dtype=float)
            farm_avg_eff   = np.zeros(self.num_server_farms, dtype=float)
            for i, sf in enumerate(self.server_farms.values()):
                utils = [s.cpu_utilization_rate for s in sf.servers.values()]
                effs  = [s.efficiency_tier       for s in sf.servers.values()]
                farm_avg_cpu[i] = np.mean(utils) if utils else 0.0
                farm_active[i]  = sum(u > 0 for u in utils) / len(utils) if utils else 0.0
                farm_avg_eff[i] = np.mean(effs) if effs else 0.0
            return {
                "farm_avg_cpu_util":   farm_avg_cpu,
                "farm_active_ratio":   farm_active,
                "farm_avg_efficiency": farm_avg_eff,
                "task_cpu":      task_cpu,
                "task_ram":      task_ram,
                "task_deadline": task_deadline,
                "wall_time":     wall_time,
            }
        else:
            local_idx = int(agent.split("_")[1])
            sf = list(self.server_farms.values())[local_idx]
            cpus_util = np.array(sf.curr_cpus_util, dtype=float)
            eff_tiers = np.array(sf.efficiency_tiers, dtype=float)
            is_selected = np.array(
                [1.0 if local_idx == self._selected_farm_id else 0.0], dtype=float
            )
            return {
                "cpus_utilization": cpus_util,
                "efficiency_tiers": eff_tiers,
                "task_cpu":      task_cpu,
                "task_ram":      task_ram,
                "task_deadline": task_deadline,
                "wall_time":     wall_time,
                "is_selected":   is_selected,
            }

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self):
        # ── 1. 共享能耗节约项 ──
        curr_energy = sum(sf.get_price for sf in self.server_farms.values())
        prev_energy = self.prev_total_energy
        if not self.task_rejected_status and prev_energy > 0:
            denom = max(abs(prev_energy), 1.0)
            r_energy = float(np.clip((prev_energy - curr_energy) / denom, -1.0, 1.0))
        else:
            r_energy = 0.0
        self.prev_total_energy = curr_energy

        # ── 2. 共享任务结果项 ──
        rej_delta = max(0, self.rejected_tasks_count - self.prev_rejected_tasks_count)
        cmp_delta = max(0, self.num_completed_jobs   - self.prev_completed_jobs_count)
        if self.task_rejected_status:
            r_task = -min(1.2 * rej_delta, 8.0)
        else:
            r_task = 1.0 + 0.5 * cmp_delta
        self.prev_rejected_tasks_count = self.rejected_tasks_count
        self.prev_completed_jobs_count = self.num_completed_jobs

        # ── 3. 共享奖励 ──
        r_shared = self.alpha_energy * r_energy + self.beta_task * r_task

        # ── 4. 局部增量项（仅给被选中的 local agent）──
        farm_utils = [float(np.mean(sf.curr_cpus_util)) for sf in self.server_farms.values()]
        mean_util = float(np.mean(farm_utils))
        selected_util = farm_utils[self._selected_farm_id]
        r_lb = -abs(selected_util - mean_util)

        # ── 5. 分发到每个 agent ──
        rewards = {self.global_agent_id: r_shared}
        for i, lid in enumerate(self.local_agent_ids):
            if i == self._selected_farm_id:
                rewards[lid] = r_shared + self.gamma_lb * r_lb
            else:
                rewards[lid] = r_shared
        return rewards

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self, agent):
        if agent == self.global_agent_id:
            return {
                "active_job_ids":      self.active_job_ids,
                "completed_job_ids":   self.completed_job_ids,
                "rejected_job_ids":    self.rejected_job_ids,
                "rejected_tasks_count": self.rejected_tasks_count,
                "wall_time":           self.wall_time,
                "price":               round(
                    sum(sf.get_price for sf in self.server_farms.values()), 2
                ),
                "hetero_reward_total": self._last_hetero_reward,
                "jains_fairness":      self._compute_jains_fairness(),
                "active_server_ratio": self._compute_active_server_ratio(),
                "her":                 self._compute_her(),
                "selected_farm_id":    self._selected_farm_id,
            }
        else:
            return {"selected_farm_id": self._selected_farm_id}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def all_jobs_complete(self):
        return self.num_completed_jobs == len(self.jobs)

    @property
    def num_completed_jobs(self):
        return len(self.completed_job_ids)

    @property
    def num_active_jobs(self):
        return len(self.active_job_ids)

    @property
    def num_rejected_jobs(self):
        return len(self.rejected_job_ids)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_jains_fairness(self) -> float:
        utils = [
            s.cpu_utilization_rate
            for sf in self.server_farms.values()
            for s in sf.servers.values()
        ]
        n, total = len(utils), sum(utils)
        if n == 0 or total == 0:
            return 1.0
        return round((total ** 2) / (n * sum(u ** 2 for u in utils)), 4)

    def _compute_active_server_ratio(self) -> float:
        all_servers = [
            s for sf in self.server_farms.values() for s in sf.servers.values()
        ]
        if not all_servers:
            return 0.0
        active = sum(1 for s in all_servers if s.cpu_utilization_rate > 0)
        return round(active / len(all_servers), 4)

    def _compute_her(self) -> float:
        if self._her_denominator == 0:
            return 0.0
        return round(self._her_numerator / self._her_denominator, 4)

    # ------------------------------------------------------------------
    # Internal simulation helpers (mirrors CloudSchedulingEnv)
    # ------------------------------------------------------------------

    def _load_initial_jobs(self):
        arrived_jobs = []
        while not self.timeline.empty:
            wall_time, event = self.timeline.peek()
            try:
                job = event.data["job"]
                arrived_jobs.append((wall_time, job))
                self.timeline.pop()
            except KeyError:
                raise Exception("initial timeline must only contain jobs")

        for wall_time, job in arrived_jobs:
            self._handle_job_arrival(job, wall_time)
        self.wall_time = arrived_jobs[0][0]
        arrived_jobs.clear()
        self.schedulable_tasks = True

    def _take_action(self, actions):
        self._selected_farm_id = actions["server_farm"]
        self.server_id         = actions["server"]

        sf     = list(self.server_farms.values())[self._selected_farm_id]
        server = sf.servers[str(self.server_id)]

        self.wall_time, event = self.timeline.pop()
        try:
            task = event.data["task_arrival"]
        except KeyError:
            raise Exception("scheduling action timeline must only contain task arrival events")

        if task.job_id in self.rejected_job_ids:
            self._process_task_rejection(task)
            return

        self._handle_task_arrival(task)

        if server.is_available:
            scheduled = server.host_task_in_server(task)
            if scheduled:
                self.scheduled_task_cpu      = task.cpu
                self.scheduled_task_ram      = task.ram
                self.scheduled_task_deadline = self.wall_time + task.runtime
                self._process_task_scheduling(task)
                self._insert_task_departure_event(task)
                self.scheduled_tasks.add(task)
                self.schedulable_tasks = False
                self._her_numerator += task.cpu * server.efficiency_tier
                self._her_denominator += task.cpu
            else:
                self._process_task_rejection(task)
        else:
            self._process_task_rejection(task)

    def _resume_simulation(self):
        assert not self.timeline.empty, (
            "Timeline empty during resume — check rejection/completion logic"
        )
        while not self.timeline.empty:
            self.wall_time, event = self.timeline.peek()
            try:
                _ = event.data["task_arrival"]
                self.schedulable_tasks = True
            except KeyError:
                self.handle_event[event.type](**event.data)
                self.timeline.pop()

            if self.schedulable_tasks:
                break

    def _handle_job_arrival(self, job, wall_time=None):
        self.active_job_ids.append(job.id)
        ready_tasks = self._find_schedulable_tasks(job_ids=[job.id])
        for task in ready_tasks:
            self.timeline.push(
                wall_time,
                TimelineEvent(TimelineEvent.Type.TASK_ARRIVAL, data={"task_arrival": task}),
            )

    def _handle_task_arrival(self, task_arrival):
        task_arrival.arrival_time = self.wall_time

    def _handle_task_departure(self, task_departure):
        task = task_departure
        self.scheduled_task_cpu = task.cpu
        self.scheduled_task_ram = task.ram
        try:
            sf     = self.server_farms[task.server_farm_id]
            server = sf.servers[task.server_id]
            server.clear_completed_task_in_server(task)
            job = self.jobs[task.job_id]
        except Exception:
            if task.job_id in self.rejected_job_ids:
                return
            return

        task.departure_time = self.wall_time
        self._process_task_completion(task)

        if job.completed:
            self._process_job_completion(job)
        else:
            ready_tasks = self._find_schedulable_tasks()
            self._insert_ready_tasks_events(ready_tasks)

    def _find_schedulable_tasks(self, job_ids=None):
        if job_ids is None:
            job_ids = list(self.active_job_ids)
        return [
            task
            for job_id in job_ids
            for task in self.jobs[job_id].get_ready_tasks()
            if task not in self.scheduled_tasks and self._is_task_ready(task)
        ]

    def _is_task_ready(self, task):
        if task.status != 1 or task.status in (-1, 2, 3):
            return False
        job = self.jobs[task.job_id]
        for parent in job.get_parent_of_task(task.id):
            if parent.status not in (0, -1):
                return False
        return True

    def _insert_ready_tasks_events(self, ready_tasks):
        for task in ready_tasks:
            self.timeline.push(
                self.wall_time,
                TimelineEvent(TimelineEvent.Type.TASK_ARRIVAL, data={"task_arrival": task}),
            )
            self.scheduled_tasks.add(task)

    def _insert_task_departure_event(self, task):
        departure_time = round(self.wall_time + task.runtime, 2)
        self.timeline.push(
            departure_time,
            TimelineEvent(TimelineEvent.Type.TASK_DEPARTURE, data={"task_departure": task}),
        )

    def _process_task_scheduling(self, task):
        self.jobs[task.job_id].modify_task_status(task.id, 2)

    def _process_task_completion(self, task):
        self.jobs[task.job_id].modify_task_status(task.id, 0)

    def _process_job_completion(self, job):
        assert job.id in self.jobs
        self.active_job_ids.remove(job.id)
        self.completed_job_ids.add(job.id)
        job.time_completed = self.wall_time

    def _process_task_rejection(self, task):
        self.task_rejected_status = True
        self.schedulable_tasks    = False
        try:
            job = self.jobs[task.job_id]
            job.reject_task_and_cascade(task.id)
            self.rejected_tasks_count += job.number_of_rejected_tasks
            self.active_job_ids.remove(job.id)
            self.rejected_job_ids.add(task.job_id)
            self.jobs.pop(job.id)
        except Exception:
            return

    def render(self):
        pass
