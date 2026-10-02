"""Consumer devices on live SUMO traffic: the same interface as base_imp's ConsumerDevices
(features, x(), link_delay(), link_queue_est, true_features), computed every second from SUMO's
real vehicle speeds with the formulas of sumo_data/recorder.py (= base_imp/qits/traffic/devices.py):

    density = reporting vehicles on the road / penetration / road storage
    speed   = mean reported speed / free-flow speed            (1 if nobody reports)
    queue   = reporting vehicles slower than 1 m/s / penetration / road storage
    inflow  = EWMA (span 30 s) of stop-line arrivals of device vehicles / penetration / saturation flow
    δ_ij    = max(mean device-reported travel time on the road in the last 60 s,
                  free-flow time + estimated queue / saturation flow)
"""
from collections import deque

import numpy as np


class SumoDevices:
    def __init__(self, cfg, gmap, rng):
        d, net = cfg["devices"], cfg["network"]
        g = gmap.grid
        self.grid, self.rng = g, rng
        self.pen = d["penetration"]
        self.speed_noise = d["speed_noise_mps"]
        self.queue_speed = d["queue_speed_mps"]
        self.alpha = 2.0 / (d["inflow_span_s"] + 1.0)
        self.window = d["delay_window_s"]
        self.vf = net["free_flow_speed_mps"]
        self.sat = net["sat_flow_vps"] * net["lanes"]
        self.link_storage = gmap.length * net["lanes"] / net["veh_spacing_m"]       # (L,)
        self.storage = float(self.link_storage.mean())
        self.approach = (g.link_dir + 2) % 4

        n, L = g.n_nodes, g.n_links
        self.features = np.zeros((n, 4, 4))
        self.true_features = np.zeros((n, 4, 4))
        self.link_queue_est = np.zeros(L)
        self.inflow = np.zeros((n, 4))
        self.true_inflow = np.zeros((n, 4))
        self._prev_dev_arr = np.zeros((n, 4))
        self._prev_arr = np.zeros((n, 4))
        self.recent_tt = [deque() for _ in range(L)]
        self._tt_ptr = 0
        self.probes = []
        self.error_log = []

    def x(self):
        """(n_nodes, 16) feature vectors for Eq. 2."""
        return self.features.reshape(self.grid.n_nodes, -1)

    def link_delay(self, ff_time):
        """(n_nodes, 4) estimated delay δ_ij (s) of leaving node i in each direction (0 = no road)."""
        g = self.grid
        out = np.where(g.out_link >= 0, g.out_link, 0)
        reported = np.array([np.mean([tt for _, tt in rt]) if rt else 0.0 for rt in self.recent_tt])
        link = np.maximum(reported, ff_time + self.link_queue_est / self.sat)
        return np.where(g.out_link >= 0, link[out], 0.0)

    def observe(self, sim, t, lids, speeds, has_dev):
        """lids, speeds, has_dev: every vehicle currently on a road (not inside a junction)."""
        g, L = self.grid, self.grid.n_links
        rep = np.maximum(0.0, speeds[has_dev] + self.rng.normal(0, self.speed_noise, int(has_dev.sum())))
        dl = lids[has_dev]
        n_rep = np.bincount(dl, minlength=L)
        n_slow = np.bincount(dl, weights=(rep < self.queue_speed), minlength=L)
        sum_rep = np.bincount(dl, weights=rep, minlength=L)
        n_true = np.bincount(lids, minlength=L)
        sum_true = np.bincount(lids, weights=speeds, minlength=L)
        n_true_slow = np.bincount(lids, weights=(speeds < self.queue_speed), minlength=L)
        self.link_queue_est = n_slow / self.pen

        f = np.zeros_like(self.features)
        tf = np.zeros_like(self.true_features)
        u, a, st = g.link_to, self.approach, self.link_storage
        f[u, a, 0] = n_rep / self.pen / st
        f[u, a, 1] = np.where(n_rep > 0, sum_rep / np.maximum(n_rep, 1) / self.vf, 1.0)
        f[u, a, 2] = n_slow / self.pen / st
        tf[u, a, 0] = n_true / st
        tf[u, a, 1] = np.where(n_true > 0, sum_true / np.maximum(n_true, 1) / self.vf, 1.0)
        tf[u, a, 2] = n_true_slow / st

        dev_arr = sim.device_arrivals - self._prev_dev_arr
        arr = sim.arrivals - self._prev_arr
        self._prev_dev_arr, self._prev_arr = sim.device_arrivals.copy(), sim.arrivals.copy()
        self.inflow += self.alpha * (dev_arr / self.pen - self.inflow)
        self.true_inflow += self.alpha * (arr - self.true_inflow)
        f[:, :, 3] = self.inflow / self.sat
        tf[:, :, 3] = self.true_inflow / self.sat
        self.features = np.clip(f, 0.0, 1.0)
        self.true_features = np.clip(tf, 0.0, 1.0)

        new = sim.link_tt[self._tt_ptr:]
        self._tt_ptr = len(sim.link_tt)
        for t_exit, lid, tt, dev in new:
            if dev:
                self.recent_tt[lid].append((t_exit, tt))
        for rt in self.recent_tt:
            while rt and rt[0][0] < t - self.window:
                rt.popleft()
        valid = g.in_link >= 0
        self.error_log.append((t, *np.abs(self.features - self.true_features)[valid].mean(axis=0)))
