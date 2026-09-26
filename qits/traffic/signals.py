"""Two-phase signals (NS / EW) in one of two modes.

split mode (default; fixed-time, rule-based, Q-ITS):
    Fixed cycle, zero offsets:  [NS green | all-red | EW green | all-red]
    NS green = S_i * (cycle - 2*lost). A new split written to `pending_split` takes effect
    at the start of the next cycle.
phase mode (DQN baseline, standard DQN traffic-signal formulation):
    The controller requests a phase per node with `request_phase`; a change inserts `lost`
    seconds of all-red before the new phase turns green.
"""
import numpy as np

from .grid import NS_APPROACHES


class Signals:
    def __init__(self, n_nodes, cycle_s=60, lost_time_s=3, split_min=0.2, split_max=0.8,
                 initial_split=0.5):
        self.n_nodes = n_nodes
        self.cycle = cycle_s
        self.lost = lost_time_s
        self.smin, self.smax = split_min, split_max
        self.split = np.full(n_nodes, initial_split, dtype=float)
        self.pending_split = self.split.copy()
        self.green_total = cycle_s - 2 * lost_time_s
        self.mode = "split"

    # ------------------------------------------------------------ phase mode
    def use_phase_mode(self):
        self.mode = "phase"
        self.phase = np.zeros(self.n_nodes, int)        # 0 = NS green, 1 = EW green
        self.next_phase = np.zeros(self.n_nodes, int)
        self.allred = np.zeros(self.n_nodes, int)       # remaining all-red seconds
        self.red_now = np.zeros(self.n_nodes, bool)
        self.time_in_phase = np.zeros(self.n_nodes, int)

    def request_phase(self, phases):
        switch = (np.asarray(phases) != self.phase) & (self.allred == 0)
        self.allred[switch] = self.lost
        self.next_phase[switch] = np.asarray(phases)[switch]

    # ------------------------------------------------------------------ step
    def is_cycle_start(self, t):
        return t % self.cycle == 0

    def step(self, t):
        if self.mode == "split":
            if self.is_cycle_start(t):
                self.split = np.clip(self.pending_split, self.smin, self.smax)
            return
        self.red_now = self.allred > 0
        self.allred[self.red_now] -= 1
        done = self.red_now & (self.allred == 0)
        self.phase[done] = self.next_phase[done]
        self.time_in_phase[done] = 0
        self.time_in_phase[~self.red_now] += 1

    def green_mask(self, t):
        """(n_nodes, 4) boolean: which approaches have green at time t."""
        if self.mode == "phase":
            ns_green = (self.phase == 0) & ~self.red_now
            ew_green = (self.phase == 1) & ~self.red_now
        else:
            tc = t % self.cycle
            g_ns = self.split * self.green_total
            ns_green = tc < g_ns
            ew_start = g_ns + self.lost
            ew_green = (tc >= ew_start) & (tc < ew_start + (self.green_total - g_ns))
        mask = np.zeros((self.n_nodes, 4), dtype=bool)
        for a in range(4):
            mask[:, a] = ns_green if a in NS_APPROACHES else ew_green
        return mask
