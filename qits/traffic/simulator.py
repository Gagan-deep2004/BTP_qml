"""Mesoscopic queue-based urban traffic simulator (Δt = 1 s).

Each directed link has a "moving" FIFO (vehicles travelling at free-flow speed) and a
"queue" FIFO at its downstream stop line. Intersections discharge queued vehicles on
green approaches at the saturation flow rate, subject to storage space on the next
link (spillback). Vehicles route over minimal (distance-reducing) paths; among the
productive directions they choose according to `route_p[node, dir]`, which the
controller sets (uniform for baselines, VQC output p_ij for Q-ITS).
"""
import math
from collections import deque

import numpy as np
import pandas as pd

from .devices import ConsumerDevices
from .grid import Grid
from .signals import Signals


class Vehicle:
    __slots__ = ("vid", "origin", "dest", "t_depart", "has_device", "link",
                 "link_enter_t", "next_dir", "hops", "hops_min")

    def __init__(self, vid, origin, dest, t_depart, has_device, hops_min):
        self.vid, self.origin, self.dest, self.t_depart = vid, origin, dest, t_depart
        self.has_device = has_device
        self.link, self.link_enter_t, self.next_dir = -1, -1, -1
        self.hops, self.hops_min = 0, hops_min


class TrafficSimulator:
    def __init__(self, cfg, demand, controller, seed=42):
        net, sig = cfg["network"], cfg["signal"]
        self.cfg = cfg
        self.grid = Grid(net["rows"], net["cols"])
        self.dt = cfg["simulation"]["dt_s"]
        self.duration = cfg["simulation"]["duration_s"]
        self.ff_steps = math.ceil(net["link_length_m"] / net["free_flow_speed_mps"])
        self.storage = int(net["link_length_m"] * net["lanes"] / net["veh_spacing_m"])
        self.sat_per_step = net["sat_flow_vps"] * net["lanes"] * self.dt
        self.signals = Signals(self.grid.n_nodes, sig["cycle_s"], sig["lost_time_s"],
                               sig["split_min"], sig["split_max"], sig["initial_split"])
        self.rng = np.random.default_rng(seed + 1_000_003)
        self.device_penetration = cfg["devices"]["penetration"]
        self.devices = ConsumerDevices(cfg, self.grid, np.random.default_rng(seed + 2_000_003))

        # routing probabilities over N,E,S,W for each node (set by the controller)
        self.route_p = self.grid.valid_mask / self.grid.valid_mask.sum(1, keepdims=True)

        L = self.grid.n_links
        self.moving = [deque() for _ in range(L)]   # (ready_t, vehicle)
        self.queue = [deque() for _ in range(L)]    # vehicles waiting at stop line
        self.credit = np.zeros(L)
        self.entry = {}                             # (node, dir) -> deque of vehicles
        self.arrivals = np.zeros((self.grid.n_nodes, 4))  # cumulative stop-line arrivals per approach
        self.device_arrivals = np.zeros((self.grid.n_nodes, 4))  # same, device vehicles only

        self.demand = demand.sort_values("t_depart").reset_index(drop=True)
        self._dem_t = self.demand["t_depart"].to_numpy()
        self._dem_ptr = 0

        self.t = 0
        self.n_spawned = 0
        self.trips = []        # completed trips
        self.link_tt = []      # (t_exit, link, travel_time, has_device) — for δ_ij estimation
        self.step_log = []

        self.controller = controller
        controller.reset(self)

    # ---------------------------------------------------------------- helpers
    def occupancy(self, lid):
        return len(self.moving[lid]) + len(self.queue[lid])

    def has_space(self, lid):
        return self.occupancy(lid) < self.storage

    def link_occupancy(self):
        return np.array([self.occupancy(l) for l in range(self.grid.n_links)])

    def queue_by_approach(self):
        """(n_nodes, 4) queue lengths on each incoming approach (0 where no link)."""
        q = np.zeros((self.grid.n_nodes, 4))
        il = self.grid.in_link
        for v in range(self.grid.n_nodes):
            for a in range(4):
                if il[v, a] >= 0:
                    q[v, a] = len(self.queue[il[v, a]])
        return q

    def choose_dir(self, veh, node):
        dirs = self.grid.productive_dirs(node, veh.dest)
        if len(dirs) == 1:
            return dirs[0]
        w = self.route_p[node, dirs] + 1e-9
        return dirs[self.rng.choice(len(dirs), p=w / w.sum())]

    def _enter_link(self, veh, lid, t):
        veh.link, veh.link_enter_t = lid, t
        veh.hops += 1
        self.moving[lid].append((t + self.ff_steps, veh))

    def _leave_link(self, veh, t):
        self.link_tt.append((t, veh.link, t - veh.link_enter_t, veh.has_device))

    def _finish(self, veh, t):
        self._leave_link(veh, t)
        self.trips.append((veh.vid, veh.origin, veh.dest, veh.t_depart, t, veh.hops,
                           veh.hops_min, veh.has_device))

    # ------------------------------------------------------------------- step
    def step(self):
        t, g = self.t, self.grid
        self.controller.step(self, t)
        self.signals.step(t)

        # 1. spawn new vehicles into their origin entry buffer
        while self._dem_ptr < len(self._dem_t) and self._dem_t[self._dem_ptr] == t:
            row = self.demand.iloc[self._dem_ptr]
            o, d = int(row.origin), int(row.dest)
            veh = Vehicle(int(row.veh_id), o, d, t,
                          bool(self.rng.random() < self.device_penetration),
                          g.manhattan(o, d))
            self.entry.setdefault((o, self.choose_dir(veh, o)), deque()).append(veh)
            self._dem_ptr += 1
            self.n_spawned += 1

        # 2. insert at most one waiting vehicle per outgoing link per step
        for (o, d), buf in self.entry.items():
            if buf:
                lid = g.out_link[o, d]
                if self.has_space(lid):
                    self._enter_link(buf.popleft(), lid, t)

        # 3. vehicles reaching the end of their link join the stop-line queue (or finish)
        for lid in range(g.n_links):
            mv = self.moving[lid]
            u = g.link_to[lid]
            approach = (g.link_dir[lid] + 2) % 4
            while mv and mv[0][0] <= t:
                _, veh = mv.popleft()
                if u == veh.dest:
                    self._finish(veh, t)
                else:
                    veh.next_dir = self.choose_dir(veh, u)
                    self.queue[lid].append(veh)
                    self.arrivals[u, approach] += 1
                    if veh.has_device:
                        self.device_arrivals[u, approach] += 1

        # 4. discharge green approaches at saturation flow, respecting downstream storage
        green = self.signals.green_mask(t)
        for u in range(g.n_nodes):
            for a in range(4):
                lid = g.in_link[u, a]
                if lid < 0:
                    continue
                if not green[u, a]:
                    self.credit[lid] = 0.0   # red: next green starts with 1 s start-up loss
                    continue
                self.credit[lid] = min(self.credit[lid] + self.sat_per_step, 1.0)
                q = self.queue[lid]
                while self.credit[lid] >= 1.0 and q:
                    veh = q[0]
                    nl = g.out_link[u, veh.next_dir]
                    if not self.has_space(nl):
                        break  # spillback: blocked by full downstream link
                    q.popleft()
                    self._leave_link(veh, t)
                    self._enter_link(veh, nl, t)
                    self.credit[lid] -= 1.0

        # 5. consumer devices report; intersections rebuild x_i(t) and δ_ij(t)
        self.devices.observe(self, t)

        # 6. log
        occ = self.link_occupancy()
        waiting = sum(len(b) for b in self.entry.values())
        self.step_log.append((t, self.n_spawned - len(self.trips), waiting,
                              sum(len(q) for q in self.queue),
                              float(np.mean(occ / self.storage)), len(self.trips)))
        self.t += 1

    def run(self, duration=None):
        for _ in range(duration or self.duration):
            self.step()
        return self.results()

    # ---------------------------------------------------------------- results
    def results(self):
        trips = pd.DataFrame(self.trips, columns=["veh_id", "origin", "dest", "t_depart",
                                                  "t_arrive", "hops", "hops_min", "has_device"])
        trips["travel_time"] = trips.t_arrive - trips.t_depart
        trips["delay"] = trips.travel_time - trips.hops_min * self.ff_steps
        steps = pd.DataFrame(self.step_log, columns=["t", "in_network", "entry_waiting",
                                                     "total_queue", "congestion_index",
                                                     "completed"])
        n_unfinished = self.n_spawned - len(trips)
        summary = {
            "controller": self.controller.name,
            "spawned": self.n_spawned,
            "completed": len(trips),
            "completion_rate": len(trips) / max(self.n_spawned, 1),
            "avg_delay_s": float(trips.delay.mean()) if len(trips) else float("nan"),
            "avg_travel_time_s": float(trips.travel_time.mean()) if len(trips) else float("nan"),
            "throughput_vps": len(trips) / self.t,
            "congestion_index": float(steps.congestion_index.mean()),
            "avg_queue": float(steps.total_queue.mean()),
            "final_in_network": int(steps.in_network.iloc[-1]),
            "unfinished": n_unfinished,
        }
        return {"summary": summary, "trips": trips, "steps": steps}
