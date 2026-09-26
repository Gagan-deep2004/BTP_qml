"""Consumer-device layer: smartphones / in-car units report noisy probe data, which each
intersection aggregates into its feature vector x_i(t) and its link-delay estimates δ_ij(t).

Only vehicles carrying a device (probability = penetration) are observed. Counts are
scaled up by 1/penetration. Features per incoming approach (N, E, S, W), each in [0, 1]:
    0 density   = estimated vehicles on link / storage
    1 speed     = mean reported speed / free-flow speed   (1.0 if no reports)
    2 queue     = estimated vehicles with speed < queue_speed / storage
    3 inflow    = EWMA of estimated stop-line arrivals (veh/s) / saturation flow
Flattened approach-major -> 16 features = 2^4 amplitudes for 4 qubits (Eq. 2).
Missing approaches (grid boundary) stay 0.
"""
from collections import deque

import numpy as np

N_FEATURES = 4
FEATURE_NAMES = ["density", "speed", "queue", "inflow"]


class ConsumerDevices:
    def __init__(self, cfg, grid, rng):
        d, net = cfg["devices"], cfg["network"]
        self.grid, self.rng = grid, rng
        self.pen = d["penetration"]
        self.speed_noise = d["speed_noise_mps"]
        self.gps_noise = d["gps_noise_m"]
        self.queue_speed = d["queue_speed_mps"]
        self.alpha = 2.0 / (d["inflow_span_s"] + 1.0)
        self.window = d["delay_window_s"]
        self.save_probes = d.get("save_probes", False)

        self.L = net["link_length_m"]
        self.vf = net["free_flow_speed_mps"]
        self.storage = int(net["link_length_m"] * net["lanes"] / net["veh_spacing_m"])
        self.sat = net["sat_flow_vps"] * net["lanes"]
        self.q_spacing = net["veh_spacing_m"] / net["lanes"]   # queue position spacing

        n, nl = grid.n_nodes, grid.n_links
        self.approach_of_link = (grid.link_dir + 2) % 4
        self.features = np.zeros((n, 4, N_FEATURES))       # device estimates
        self.true_features = np.zeros((n, 4, N_FEATURES))  # ground truth (for validation only)
        self.link_queue_est = np.zeros(nl)
        self.inflow = np.zeros((n, 4))
        self.true_inflow = np.zeros((n, 4))
        self._prev_dev_arr = np.zeros((n, 4))
        self._prev_arr = np.zeros((n, 4))
        self.recent_tt = [deque() for _ in range(nl)]      # (t_exit, travel_time) device reports
        self._tt_ptr = 0
        self.probes = []
        self.error_log = []

    # ------------------------------------------------------------------ API
    def x(self):
        """(n_nodes, 16) feature vectors for Eq. 2."""
        return self.features.reshape(self.grid.n_nodes, -1)

    def link_delay(self, ff_time):
        """(n_nodes, 4) estimated delay δ_ij (s) of leaving node i in each direction.
        δ = max(recent device-reported travel time, ff + estimated queue / sat flow).
        Invalid directions (no link) are 0; mask them with grid.valid_mask."""
        g = self.grid
        delay = np.zeros((g.n_nodes, 4))
        for v in range(g.n_nodes):
            for d in range(4):
                lid = g.out_link[v, d]
                if lid < 0:
                    continue
                queue_based = ff_time + self.link_queue_est[lid] / self.sat
                rt = self.recent_tt[lid]
                reported = np.mean([tt for _, tt in rt]) if rt else 0.0
                delay[v, d] = max(reported, queue_based)
        return delay

    # -------------------------------------------------------------- observe
    def observe(self, sim, t):
        g, rng = self.grid, self.rng
        f = np.zeros_like(self.features)
        tf = np.zeros_like(self.true_features)

        for lid in range(g.n_links):
            u, a = g.link_to[lid], self.approach_of_link[lid]
            mv, q = sim.moving[lid], sim.queue[lid]
            dev_mv = [veh for _, veh in mv if veh.has_device]
            dev_q = [veh for veh in q if veh.has_device]
            speeds = np.concatenate([
                np.maximum(0.0, self.vf + rng.normal(0, self.speed_noise, len(dev_mv))),
                np.abs(rng.normal(0, self.speed_noise, len(dev_q))),
            ])
            n_rep = len(speeds)
            n_slow = int((speeds < self.queue_speed).sum())
            self.link_queue_est[lid] = n_slow / self.pen

            f[u, a, 0] = n_rep / self.pen / self.storage
            f[u, a, 1] = speeds.mean() / self.vf if n_rep else 1.0
            f[u, a, 2] = n_slow / self.pen / self.storage

            n_true = len(mv) + len(q)
            tf[u, a, 0] = n_true / self.storage
            tf[u, a, 1] = len(mv) / n_true if n_true else 1.0  # moving at vf, queued at 0
            tf[u, a, 2] = len(q) / self.storage

            if self.save_probes:
                for ready, veh in mv:
                    if veh.has_device:
                        frac = 1.0 - (ready - t) / sim.ff_steps
                        self._probe(t, veh, lid, frac * self.L, self.vf)
                for k, veh in enumerate(q):
                    if veh.has_device:
                        self._probe(t, veh, lid, self.L - k * self.q_spacing, 0.0)

        # inflow: EWMA of stop-line arrivals per second (device-estimated and true)
        dev_arr = sim.device_arrivals - self._prev_dev_arr
        arr = sim.arrivals - self._prev_arr
        self._prev_dev_arr, self._prev_arr = sim.device_arrivals.copy(), sim.arrivals.copy()
        self.inflow += self.alpha * (dev_arr / self.pen - self.inflow)
        self.true_inflow += self.alpha * (arr - self.true_inflow)
        f[:, :, 3] = self.inflow / self.sat
        tf[:, :, 3] = self.true_inflow / self.sat

        valid = g.in_link >= 0
        f[~valid] = 0.0
        tf[~valid] = 0.0
        self.features = np.clip(f, 0.0, 1.0)
        self.true_features = np.clip(tf, 0.0, 1.0)

        # link travel-time reports from device vehicles, kept for `window` seconds
        new = sim.link_tt[self._tt_ptr:]
        self._tt_ptr = len(sim.link_tt)
        for t_exit, lid, tt, has_dev in new:
            if has_dev:
                self.recent_tt[lid].append((t_exit, tt))
        for rt in self.recent_tt:
            while rt and rt[0][0] < t - self.window:
                rt.popleft()

        err = np.abs(self.features - self.true_features)[valid]      # (n_valid, 4)
        self.error_log.append((t, *err.mean(axis=0)))

    def _probe(self, t, veh, lid, pos, speed):
        self.probes.append((t, veh.vid, lid,
                            float(np.clip(pos + self.rng.normal(0, self.gps_noise), 0, self.L)),
                            max(0.0, speed + self.rng.normal(0, self.speed_noise))))
