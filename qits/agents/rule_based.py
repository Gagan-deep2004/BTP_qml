"""Rule-based baseline (paper Sec. V): fixed cycle, phase split chosen from predefined
thresholds at each cycle start, no learning. Routing is uniform minimal.

The rule uses the NS share of stop-line arrivals during the previous cycle (plus the
current queues). Queues alone are biased at cycle start: the phase that just ended has
been served, so the other phase always looks busier.
"""
import numpy as np

from ..traffic.grid import E, N, S, W
from .base import Controller


class RuleBasedController(Controller):
    name = "rule_based"

    def __init__(self, high_ratio=0.65, low_ratio=0.35, split_high=0.65, split_low=0.35):
        self.high_ratio, self.low_ratio = high_ratio, low_ratio
        self.split_high, self.split_low = split_high, split_low

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg["rule_based"])

    def reset(self, sim):
        self._last_arrivals = sim.arrivals.copy()

    def step(self, sim, t):
        if not sim.signals.is_cycle_start(t) or t == 0:
            return
        demand = sim.arrivals - self._last_arrivals + sim.queue_by_approach()
        self._last_arrivals = sim.arrivals.copy()
        d_ns, d_ew = demand[:, N] + demand[:, S], demand[:, E] + demand[:, W]
        total = d_ns + d_ew
        ratio = np.divide(d_ns, total, out=np.full_like(total, 0.5), where=total > 0)
        split = np.full(len(ratio), 0.5)
        split[ratio > self.high_ratio] = self.split_high
        split[ratio < self.low_ratio] = self.split_low
        sim.signals.pending_split = split
