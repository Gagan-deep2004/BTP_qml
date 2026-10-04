"""SUMO in place of the base simulator: same interface, so the base controllers run unchanged.

Every second (same order as base_imp/qits/traffic/simulator.py):
  1. controller.step(sim, t): it reads the device features / queues and may set
     `signals.pending_split`, `signals.request_phase(...)` and `route_p` (exactly as in the base code);
  2. signals.step(t) (the base Signals class); its green mask is written to SUMO's traffic lights.
     The base model's 3 s all-red lost time after each green is shown in SUMO as 3 s amber;
  3. vehicles due at t are inserted. Like the base simulator, each vehicle follows a shortest path and
     at every junction picks among the directions that bring it closer to its destination with
     probability ∝ route_p[node, dir] (uniform for the baselines, the VQC's p_ij for Q-ITS). The choice
     for the junction at the end of a road is made when the vehicle enters that road;
  4. SUMO advances 1 s (car following, lane changes, junction conflicts, insertion queues);
  5. queues, stop-line arrivals and road travel times are updated and the consumer devices report.

Trip delay = SUMO time loss (time lost below the desired speed) + time waiting to enter the network,
the same definition as the SUMO dataset. `avg_delay_all_s` also counts vehicles not finished at the end.
"""
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import pandas as pd
import traci
import traci.constants as tc

from common import GridMap, binary, net_file
from qits.traffic.signals import Signals
from qits.traffic.simulator import Vehicle

from .devices import SumoDevices

SUB = [tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED]
# seconds traci keeps trying to reach a starting SUMO (each try ~1 s); many parallel runs on a
# busy machine can make a 25 x 25 SUMO take well over the default 60 s to start
CONNECT_RETRIES = 300


class SVehicle(Vehicle):
    __slots__ = ("counted",)


class SumoSimulator:
    def __init__(self, cfg, scfg, demand, controller, seed=42, port=None, gmap=None, duration=None):
        net, sig = cfg["network"], cfg["signal"]
        self.cfg, self.scfg, self.seed = cfg, scfg, seed
        self.gmap = gmap or GridMap(scfg)
        self.grid = g = self.gmap.grid
        self.dt = 1
        self.duration = duration or cfg["simulation"]["duration_s"]
        self.vf = net["free_flow_speed_mps"]
        self.ff_steps = math.ceil(self.gmap.length.mean() / self.vf)
        self.storage = int(self.gmap.length.mean() * net["lanes"] / net["veh_spacing_m"])
        self.q_speed = cfg["devices"]["queue_speed_mps"]
        self.signals = Signals(g.n_nodes, sig["cycle_s"], sig["lost_time_s"], sig["split_min"],
                               sig["split_max"], sig["initial_split"])
        self.lost = sig["lost_time_s"]
        self.rng = np.random.default_rng(seed + 1_000_003)
        self.device_penetration = cfg["devices"]["penetration"]
        self.devices = SumoDevices(cfg, self.gmap, np.random.default_rng(seed + 2_000_003))
        self.route_p = g.valid_mask / g.valid_mask.sum(1, keepdims=True)

        L = g.n_links
        self.moving = [deque() for _ in range(L)]
        self.queue = [deque() for _ in range(L)]
        self.pending = {}                                   # inserted into SUMO, waiting to enter
        self.active = {}                                    # driving
        self.arrivals = np.zeros((g.n_nodes, 4))
        self.device_arrivals = np.zeros((g.n_nodes, 4))
        self.approach = (g.link_dir + 2) % 4

        self.demand = demand.sort_values("t_depart").reset_index(drop=True)
        self._dem = self.demand[["veh_id", "t_depart", "origin", "dest"]].to_numpy()
        self._dem_ptr = 0
        self.t = 0
        self.n_spawned = 0
        self.trips = []
        self.link_tt = []
        self.step_log = []
        self.teleports = 0

        self._tmp = tempfile.TemporaryDirectory()
        self._tripinfo = os.path.join(self._tmp.name, "tripinfo.xml")
        cmd = [binary("sumo"), "-n", str(net_file(scfg)), "--begin", "0", "--step-length", "1",
               "--seed", str(seed), "--time-to-teleport", str(scfg["routing"]["time_to_teleport_s"]),
               "--no-step-log", "--no-warnings", "--duration-log.disable",
               "--tripinfo-output", self._tripinfo]
        self._label = f"sim{os.getpid()}_{seed}_{id(self)}"
        traci.start(cmd, port=port, numRetries=CONNECT_RETRIES, label=self._label)
        self.conn = traci.getConnection(self._label)
        if self.conn.simulation.getOption("net-file") and not os.path.samefile(
                self.conn.simulation.getOption("net-file"), net_file(scfg)):
            raise RuntimeError("connected to a SUMO running another network (TraCI port clash)")
        self._setup_lights()
        self.controller = controller
        controller.reset(self)

    # ------------------------------------------------------------------ lights
    def _setup_lights(self):
        """For every signalised junction: approach and green character ('G', or 'g' for a left turn
        that yields to oncoming traffic) of each controlled link index."""
        gm, g = self.gmap, self.grid
        self.tls = []
        for v in range(g.n_nodes):
            j = gm.junction_of_node[v]
            if j not in self.conn.trafficlight.getIDList():
                continue
            appr, gch = [], []
            for links in self.conn.trafficlight.getControlledLinks(j):
                in_lane, out_lane, _ = links[0]
                lin = gm.link_of_edge[in_lane.rsplit("_", 1)[0]]
                lout = gm.link_of_edge[out_lane.rsplit("_", 1)[0]]
                appr.append(self.approach[lin])
                gch.append("g" if g.link_dir[lout] == (g.link_dir[lin] + 3) % 4 else "G")
            self.tls.append((v, j, np.array(appr), np.array(gch)))
        self._tls_key = {}
        self._prev_green = np.zeros((g.n_nodes, 4), bool)
        self._amber = np.zeros((g.n_nodes, 4), int)

    def _apply_lights(self, t):
        green = self.signals.green_mask(t)
        self._amber[self._prev_green & ~green] = self.lost
        self._prev_green = green
        for v, j, appr, gch in self.tls:
            gv, av = green[v][appr], self._amber[v][appr] > 0
            key = (gv.tobytes(), av.tobytes())
            if self._tls_key.get(j) != key:
                self._tls_key[j] = key
                self.conn.trafficlight.setRedYellowGreenState(
                    j, "".join(np.where(gv, gch, np.where(av, "y", "r"))))
        np.maximum(self._amber - 1, 0, out=self._amber)

    # ---------------------------------------------------------------- helpers
    def queue_by_approach(self):
        q = np.zeros((self.grid.n_nodes, 4))
        lens = np.array([len(x) for x in self.queue])
        q[self.grid.link_to, self.approach] = lens
        return q

    def choose_dir(self, veh, node):
        dirs = self.grid.productive_dirs(node, veh.dest)
        if len(dirs) == 1:
            return dirs[0]
        w = self.route_p[node, dirs] + 1e-9
        return dirs[self.rng.choice(len(dirs), p=w / w.sum())]

    def _edge(self, node, d):
        return self.gmap.edge_of_link[int(self.grid.out_link[node, d])]

    def _plan_next(self, veh, lid):
        """Vehicle just entered road `lid`: choose its exit at the junction ahead."""
        u = int(self.grid.link_to[lid])
        if u == veh.dest:
            veh.next_dir = -1
            return [self.gmap.edge_of_link[lid]]
        veh.next_dir = self.choose_dir(veh, u)
        return [self.gmap.edge_of_link[lid], self._edge(u, veh.next_dir)]

    def _count_arrival(self, veh, lid):
        u, a = self.grid.link_to[lid], self.approach[lid]
        self.arrivals[u, a] += 1
        if veh.has_device:
            self.device_arrivals[u, a] += 1

    # ------------------------------------------------------------------- step
    def step(self):
        t, g, conn = self.t, self.grid, self.conn
        self.controller.step(self, t)
        self.signals.step(t)
        self._apply_lights(t)

        # 1. vehicles due now: first road chosen at the origin, then handed to SUMO
        while self._dem_ptr < len(self._dem) and self._dem[self._dem_ptr, 1] == t:
            vid, _, o, d = (int(x) for x in self._dem[self._dem_ptr])
            veh = SVehicle(vid, o, d, t, bool(self.rng.random() < self.device_penetration), g.manhattan(o, d))
            first = int(g.out_link[o, self.choose_dir(veh, o)])
            route = self._plan_next(veh, first)
            veh.link, veh.counted = first, False
            conn.route.add(f"r{vid}", route)
            conn.vehicle.add(f"v{vid}", f"r{vid}", depart="now", departLane="best", departSpeed="max")
            self.pending[f"v{vid}"] = veh
            self._dem_ptr += 1
            self.n_spawned += 1

        # 2. SUMO moves the traffic
        conn.simulationStep()
        self.teleports += conn.simulation.getStartingTeleportNumber()
        for sid in conn.simulation.getDepartedIDList():
            veh = self.pending.pop(sid)
            veh.link_enter_t, veh.hops = t, 1
            self.active[sid] = veh
            conn.vehicle.subscribe(sid, SUB)
        for sid in conn.simulation.getArrivedIDList():
            veh = self.active.pop(sid, None)
            if veh is not None:
                self.link_tt.append((t, veh.link, t - veh.link_enter_t, veh.has_device))
                self.trips.append((veh.vid, veh.origin, veh.dest, veh.t_depart, t + 1, veh.hops,
                                   veh.hops_min, veh.has_device))

        # 3. positions -> queues, stop-line arrivals, travel times, next turn
        for q in self.queue:
            q.clear()
        for m in self.moving:
            m.clear()
        link_of_edge, link_to, length = self.gmap.link_of_edge, g.link_to, self.gmap.length
        front = [[] for _ in range(g.n_links)]
        lids, speeds, devs = [], [], []
        for sid, r in conn.vehicle.getAllSubscriptionResults().items():
            veh = self.active.get(sid)
            if veh is None:
                continue
            lid = link_of_edge.get(r[tc.VAR_ROAD_ID], -1)
            if lid < 0:
                continue                                   # inside a junction
            if lid != veh.link:                            # entered a new road
                if not veh.counted:
                    self._count_arrival(veh, veh.link)
                self.link_tt.append((t, veh.link, t - veh.link_enter_t, veh.has_device))
                veh.link, veh.link_enter_t, veh.counted = lid, t, False
                veh.hops += 1
                conn.vehicle.setRoute(sid, self._plan_next(veh, lid))
            v, pos = r[tc.VAR_SPEED], r[tc.VAR_LANEPOSITION]
            if not veh.counted and v < self.q_speed and link_to[lid] != veh.dest:
                veh.counted = True
                self._count_arrival(veh, lid)
            if v < self.q_speed:
                front[lid].append((pos, veh))
            else:
                self.moving[lid].append((t + (length[lid] - pos) / self.vf, veh))
            lids.append(lid)
            speeds.append(v)
            devs.append(veh.has_device)
        for lid, lst in enumerate(front):
            if lst:
                lst.sort(key=lambda x: -x[0])              # stop line first
                self.queue[lid].extend(veh for _, veh in lst)

        # 4. consumer devices report; intersections rebuild x_i(t) and δ_ij(t)
        self.devices.observe(self, t, np.asarray(lids, int), np.asarray(speeds, float), np.asarray(devs, bool))

        occ = np.bincount(np.asarray(lids, int), minlength=g.n_links) / self.devices.link_storage
        self.step_log.append((t, self.n_spawned - len(self.trips), len(self.pending),
                              sum(len(q) for q in self.queue), float(occ.mean()), len(self.trips)))
        self.t += 1

    def run(self, duration=None):
        try:
            for _ in range(duration or self.duration):
                self.step()
            return self.results()
        finally:
            self.close()

    # ---------------------------------------------------------------- results
    @property
    def entry(self):
        """Vehicles waiting to enter the network (same role as the base simulator's entry buffers)."""
        return {0: list(self.pending.values())}

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _unfinished_delays(self):
        """Delay so far of every vehicle still in the network or waiting to enter (time loss + wait)."""
        out = [self.t - v.t_depart for v in self.pending.values()]
        for sid, veh in self.active.items():
            out.append(self.conn.vehicle.getTimeLoss(sid) + self.conn.vehicle.getDepartDelay(sid))
        return np.asarray(out, float)

    def results(self):
        unfinished = self._unfinished_delays()
        self.close()
        info = {int(e.get("id")[1:]): (float(e.get("timeLoss")), float(e.get("departDelay")))
                for e in ET.parse(self._tripinfo).getroot().iter("tripinfo")}
        self._tmp.cleanup()
        trips = pd.DataFrame(self.trips, columns=["veh_id", "origin", "dest", "t_depart", "t_arrive", "hops",
                                                  "hops_min", "has_device"])
        trips["travel_time"] = trips.t_arrive - trips.t_depart
        trips["time_loss"] = trips.veh_id.map(lambda v: info.get(v, (np.nan, 0))[0])
        trips["depart_delay"] = trips.veh_id.map(lambda v: info.get(v, (0, np.nan))[1])
        trips["delay"] = trips.time_loss + trips.depart_delay
        steps = pd.DataFrame(self.step_log, columns=["t", "in_network", "entry_waiting", "total_queue",
                                                     "congestion_index", "completed"])
        all_delays = np.concatenate([trips.delay.dropna().to_numpy(), unfinished])
        summary = {
            "controller": self.controller.name,
            "spawned": self.n_spawned,
            "completed": len(trips),
            "completion_rate": len(trips) / max(self.n_spawned, 1),
            "avg_delay_s": float(trips.delay.mean()) if len(trips) else float("nan"),
            "avg_delay_all_s": float(all_delays.mean()) if len(all_delays) else float("nan"),
            "avg_travel_time_s": float(trips.travel_time.mean()) if len(trips) else float("nan"),
            "throughput_vps": len(trips) / self.t,
            "congestion_index": float(steps.congestion_index.mean()),
            "avg_queue": float(steps.total_queue.mean()),
            "final_in_network": int(steps.in_network.iloc[-1]),
            "unfinished": len(unfinished),
            "teleports": self.teleports,
        }
        return {"summary": summary, "trips": trips, "steps": steps}
