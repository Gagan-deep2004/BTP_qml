"""Run SUMO on one trip file and record the consumer-device dataset (every second).

Consumer devices: each vehicle carries a phone / in-car unit with probability `penetration`. A device
reports its road, position (GPS noise N(0, 3 m)) and speed (noise N(0, 0.5 m/s)). The speeds are
SUMO's real vehicle speeds. Each intersection turns the reports on its 4 incoming roads into
16 features, with the same formulas as base_imp (qits/traffic/devices.py):

    density = reporting vehicles on the road / penetration / road storage
    speed   = mean reported speed / free-flow speed            (1 if nobody reports)
    queue   = reporting vehicles slower than 1 m/s / penetration / road storage
    inflow  = EWMA (span 30 s) of vehicles reaching the stop line / penetration / saturation flow

Road storage = SUMO road length x lanes / 7.5 m. A vehicle "reaches the stop line" of a road when
it first slows below 1 m/s there or leaves the road, whichever is first (not on its final road).
x_true: the same features from every vehicle's exact speed. delay (δ_ij, base resolution A7):
max(mean travel time reported by devices on that road in the last 60 s, free-flow time +
estimated queue / saturation flow).

Trip results (delay = SUMO timeLoss + insertion wait) come from SUMO's tripinfo output.
"""
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import traci
import traci.constants as tc

from common import GridMap, binary, net_file

SUB = [tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED]


def sumo_cmd(cfg, trips_xml, seed, tripinfo=None):
    r = cfg["routing"]
    cmd = [binary("sumo"), "-n", str(net_file(cfg)), "-r", str(trips_xml), "--junction-taz",
           "--begin", "0", "--end", str(cfg["simulation"]["duration_s"]), "--step-length", "1",
           "--seed", str(seed), "--time-to-teleport", str(r["time_to_teleport_s"]),
           "--no-step-log", "--no-warnings", "--duration-log.disable"]
    if r["rerouting_period_s"] > 0:
        cmd += ["--device.rerouting.probability", "1",
                "--device.rerouting.period", str(r["rerouting_period_s"])]
    if tripinfo:
        cmd += ["--tripinfo-output", str(tripinfo)]
    return cmd


def read_tripinfo(path):
    rows = []
    for ti in ET.parse(path).getroot().iter("tripinfo"):
        rows.append((int(ti.get("id")[1:]), float(ti.get("depart")) - float(ti.get("departDelay")),
                     float(ti.get("arrival")), float(ti.get("duration")), float(ti.get("timeLoss")),
                     float(ti.get("departDelay"))))
    trips = pd.DataFrame(rows, columns=["veh_id", "t_depart", "t_arrive", "duration", "time_loss",
                                        "depart_delay"])
    trips["travel_time"] = trips.t_arrive - trips.t_depart        # includes waiting to enter
    trips["delay"] = trips.time_loss + trips.depart_delay
    return trips


class Recorder:
    def __init__(self, cfg, gmap=None):
        self.cfg = cfg
        self.gmap = gmap or GridMap(cfg)
        g, net, dev = self.gmap.grid, cfg["network"], cfg["devices"]
        self.grid = g
        self.pen, self.vf = dev["penetration"], net["free_flow_speed_mps"]
        self.storage = self.gmap.length * net["lanes"] / net["veh_spacing_m"]     # (L,)
        self.ff_time = self.gmap.length / self.vf                                 # (L,)
        self.sat = net["sat_flow_vps"] * net["lanes"]
        self.alpha = 2.0 / (dev["inflow_span_s"] + 1.0)
        self.q_speed = dev["queue_speed_mps"]
        self.approach = (g.link_dir + 2) % 4                                       # (L,)

    def run(self, trips_xml, demand, seed, record=True, probes=False, port=None):
        """demand: DataFrame veh_id, t_depart, origin, dest. Returns dict with arrays and summaries.
        port: TraCI port. Give every run that executes at the same time its own port: with a shared
        "free port" lookup, parallel runs can pick the same port and talk to each other's SUMO."""
        cfg, g, gm = self.cfg, self.grid, self.gmap
        T, L, n = cfg["simulation"]["duration_s"], g.n_links, g.n_nodes
        dev = cfg["devices"]
        rng = np.random.default_rng(seed + 2_000_003)
        dest = dict(zip(demand.veh_id, demand.dest))
        link_of_edge = gm.link_of_edge
        link_to = g.link_to

        if record:
            X = np.zeros((T, n, 4, 4), np.float32)
            X_true = np.zeros((T, n, 4, 4), np.float32)
            q_est = np.zeros((T, L), np.float32)
            tt_sum = np.zeros((T, L))
            tt_cnt = np.zeros((T, L))
            inflow, inflow_true = np.zeros((n, 4)), np.zeros((n, 4))
        probe_rows = []
        steps = []
        state = {}               # vid -> [link, enter_t, counted_at_stop_line, has_device, dest]
        completed = teleports = waiting = 0
        every = 1 if record else 10            # the waiting list is long when the network is overloaded

        with tempfile.TemporaryDirectory() as tmp:
            tripinfo = os.path.join(tmp, "tripinfo.xml")
            label = f"rec{os.getpid()}_{seed}"
            traci.start(sumo_cmd(cfg, trips_xml, seed, tripinfo), port=port, label=label)
            conn = traci.getConnection(label)
            try:
                loaded = conn.simulation.getOption("route-files")
                if os.path.normcase(os.path.abspath(loaded)) != os.path.normcase(os.path.abspath(str(trips_xml))):
                    raise RuntimeError(f"connected to a SUMO running {loaded}, expected {trips_xml}")
                for t in range(T):
                    conn.simulationStep()
                    for vid in conn.simulation.getDepartedIDList():
                        if int(vid[1:]) not in dest:
                            raise RuntimeError(f"SUMO started {vid}, which is not in {trips_xml}")
                        state[vid] = [-1, t, False, bool(rng.random() < self.pen), dest[int(vid[1:])]]
                        if record:
                            conn.vehicle.subscribe(vid, SUB)
                    for vid in conn.simulation.getArrivedIDList():
                        s = state.pop(vid, None)
                        if record and s is not None and s[0] >= 0 and s[3]:
                            tt_sum[t, s[0]] += t - s[1]
                            tt_cnt[t, s[0]] += 1
                    completed += conn.simulation.getArrivedNumber()
                    teleports += conn.simulation.getStartingTeleportNumber()
                    running = conn.vehicle.getIDCount()
                    if t % every == 0:
                        waiting = len(conn.simulation.getPendingVehicles())
                    steps.append((t, running, waiting, completed, teleports))
                    if record:
                        self._observe(conn, t, state, rng, X, X_true, q_est, tt_sum, tt_cnt,
                                      inflow, inflow_true, probe_rows if probes else None)
            finally:
                conn.close()
            trips = read_tripinfo(tripinfo)

        steps = pd.DataFrame(steps, columns=["t", "running", "waiting_to_insert", "completed", "teleports"])
        steps["in_network"] = steps.running + steps.waiting_to_insert
        out = {"trips": trips, "steps": steps, "summary": self._summary(demand, trips, steps)}
        if record:
            out["x"] = X.reshape(T, n, 16)
            out["x_true"] = X_true.reshape(T, n, 16)
            out["delay"] = self._delays(q_est, tt_sum, tt_cnt)
            err = np.abs(out["x"] - out["x_true"]).reshape(T, n, 4, 4)
            valid = g.in_link >= 0
            out["feature_mae"] = err[:, valid].mean(axis=(0, 1))          # density, speed, queue, inflow
        if probes:
            cols = [np.concatenate(c) for c in zip(*probe_rows)] if probe_rows else [[]] * 5
            out["probes"] = pd.DataFrame(dict(zip(["t", "veh_id", "link", "pos_m", "speed_mps"], cols)))
            out["probes"][["pos_m", "speed_mps"]] = out["probes"][["pos_m", "speed_mps"]].round(2)
        return out

    # ------------------------------------------------------------------ per second
    def _observe(self, conn, t, state, rng, X, X_true, q_est, tt_sum, tt_cnt, inflow, inflow_true, probe_rows):
        g, L = self.grid, self.grid.n_links
        dev = self.cfg["devices"]
        link_of_edge, link_to = self.gmap.link_of_edge, g.link_to
        lids, speeds, has_dev, arr_lid, arr_dev, pos, vids = [], [], [], [], [], [], []
        for vid, r in conn.vehicle.getAllSubscriptionResults().items():
            s = state.get(vid)
            if s is None:
                continue
            road = r[tc.VAR_ROAD_ID]
            lid = link_of_edge.get(road, -1)            # -1: inside a junction
            if lid != s[0]:
                if s[0] >= 0:                           # left its previous road
                    if s[3]:
                        tt_sum[t, s[0]] += t - s[1]
                        tt_cnt[t, s[0]] += 1
                    if not s[2] and link_to[s[0]] != s[4]:
                        arr_lid.append(s[0])
                        arr_dev.append(s[3])
                if lid >= 0:
                    s[0], s[1], s[2] = lid, t, False
                else:
                    s[0] = -1
                if lid < 0:
                    continue
            elif lid < 0:
                continue
            v = r[tc.VAR_SPEED]
            if not s[2] and v < self.q_speed and link_to[lid] != s[4]:
                s[2] = True                             # joined the stop-line queue
                arr_lid.append(lid)
                arr_dev.append(s[3])
            lids.append(lid)
            speeds.append(v)
            has_dev.append(s[3])
            if probe_rows is not None:
                pos.append(r[tc.VAR_LANEPOSITION])
                vids.append(vid)

        lids, speeds, has_dev = np.asarray(lids, int), np.asarray(speeds), np.asarray(has_dev, bool)
        rep = np.maximum(0.0, speeds[has_dev] + rng.normal(0, dev["speed_noise_mps"], has_dev.sum()))
        dl = lids[has_dev]
        n_rep = np.bincount(dl, minlength=L)
        sum_rep = np.bincount(dl, weights=rep, minlength=L)
        n_slow = np.bincount(dl, weights=(rep < self.q_speed), minlength=L)
        n_true = np.bincount(lids, minlength=L)
        sum_true = np.bincount(lids, weights=speeds, minlength=L)
        n_true_slow = np.bincount(lids, weights=(speeds < self.q_speed), minlength=L)
        q_est[t] = n_slow / self.pen

        arr_lid, arr_dev = np.asarray(arr_lid, int), np.asarray(arr_dev, bool)
        a_all = np.bincount(arr_lid, minlength=L)
        a_dev = np.bincount(arr_lid[arr_dev], minlength=L) if arr_dev.any() else np.zeros(L)

        u, a = link_to, self.approach
        per_link = np.stack([n_rep / self.pen / self.storage,
                             np.where(n_rep > 0, sum_rep / np.maximum(n_rep, 1) / self.vf, 1.0),
                             n_slow / self.pen / self.storage], axis=1)
        per_true = np.stack([n_true / self.storage,
                             np.where(n_true > 0, sum_true / np.maximum(n_true, 1) / self.vf, 1.0),
                             n_true_slow / self.storage], axis=1)
        X[t, u, a, :3] = per_link
        X_true[t, u, a, :3] = per_true
        dev_arr, true_arr = np.zeros_like(inflow), np.zeros_like(inflow)
        dev_arr[u, a], true_arr[u, a] = a_dev / self.pen, a_all
        inflow += self.alpha * (dev_arr - inflow)
        inflow_true += self.alpha * (true_arr - inflow_true)
        X[t, :, :, 3] = inflow / self.sat
        X_true[t, :, :, 3] = inflow_true / self.sat
        np.clip(X[t], 0.0, 1.0, out=X[t])
        np.clip(X_true[t], 0.0, 1.0, out=X_true[t])

        if probe_rows is not None:
            pos = np.asarray(pos)[has_dev]
            noisy = np.clip(pos + rng.normal(0, dev["gps_noise_m"], len(pos)), 0, self.gmap.length[dl])
            ids = np.array([int(v[1:]) for v in np.asarray(vids)[has_dev]], np.int32)
            probe_rows.append((np.full(len(ids), t, np.int16), ids, dl.astype(np.int32),
                               noisy.astype(np.float32), rep.astype(np.float32)))

    def _delays(self, q_est, tt_sum, tt_cnt):
        """(T, n, 4) δ_ij: max(mean device travel time in the last window, ff + queue / sat)."""
        g, W = self.grid, self.cfg["devices"]["delay_window_s"]
        cs, cc = np.cumsum(tt_sum, 0), np.cumsum(tt_cnt, 0)
        lag = lambda c: np.vstack([np.zeros((W + 1, c.shape[1])), c[:-(W + 1)]])
        s, k = cs - lag(cs), cc - lag(cc)
        reported = np.where(k > 0, s / np.maximum(k, 1), 0.0)
        link_delay = np.maximum(reported, self.ff_time[None, :] + q_est / self.sat)     # (T, L)
        out = np.where(g.out_link >= 0, link_delay[:, np.where(g.out_link >= 0, g.out_link, 0)], 0.0)
        return out.astype(np.float32)

    def _summary(self, demand, trips, steps):
        T = self.cfg["simulation"]["duration_s"]
        return {"spawned": int((demand.t_depart < T).sum()), "completed": len(trips),
                "completion_rate": len(trips) / max(int((demand.t_depart < T).sum()), 1),
                "avg_delay_s": float(trips.delay.mean()) if len(trips) else float("nan"),
                "avg_travel_time_s": float(trips.travel_time.mean()) if len(trips) else float("nan"),
                "throughput_vps": len(trips) / T,
                "final_in_network": int(steps.in_network.iloc[-1]),
                "teleports": int(steps.teleports.iloc[-1])}
