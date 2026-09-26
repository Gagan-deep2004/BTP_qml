"""End-to-end decision latency (Eq. 6):  T_total = T_ent + T_meas + T_class + T_proc.

All constants come from cfg["comms"]["latency"] and are assumptions (A15): the paper gives
no values. T_proc for a quantum node is modelled QPU time (circuits x shots x shot time)
plus measured classical CPU time.
"""


class LatencyModel:
    def __init__(self, cfg):
        self.lat = cfg["comms"]["latency"]
        self.link_m = cfg["network"]["link_length_m"]
        self.shots = cfg["quantum"]["shots"]

    def t_class(self, hops):
        return hops * (self.link_m / self.lat["fiber_speed_mps"] + self.lat["stack_delay_s"])

    def t_proc_qpu(self, n_circuits=1):
        return n_circuits * self.shots * self.lat["shot_time_s"]

    def qits(self, t_ent, hops, n_circuits=1, cpu_s=0.0, quantum_channel=True, crypto_s_per_hop=0.0):
        """Per-node decision latency; returns (total, components dict).
        n_circuits = 0 for the classical-policy ablation (T_proc = CPU time only);
        quantum_channel = False for the NoQKD ablation (no entanglement or measurement,
        classical crypto time added per hop)."""
        parts = {"T_ent": t_ent if quantum_channel else 0.0,
                 "T_meas": self.lat["t_meas_s"] if quantum_channel else 0.0,
                 "T_class": self.t_class(hops) + hops * crypto_s_per_hop,
                 "T_proc": self.t_proc_qpu(n_circuits) + cpu_s}
        return sum(parts.values()), parts
