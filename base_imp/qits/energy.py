"""Energy per 1800 s run (paper: E_comp(t) = P_cpu(t) Δt + P_qpu(t) Δt, plus communication).

Busy-time accounting, all constants in cfg["energy"] (A20):
    computation   = P_cpu x classical compute time + P_qpu x QPU busy time
                    QPU busy time = circuit evaluations x shots x shot time
    communication = transmitted bits x energy per bit
                    (consensus / state messages, plus QKD classical post-processing)
    QKD hardware  = P_qkd_link x number of quantum links x run duration

`row` is one line of results/qits/summary.csv or results/baselines/summary.csv.
"""
import math


def _num(row, key):
    v = row.get(key, 0.0)
    return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def qpu_busy_s(row, cfg):
    return _num(row, "circuit_evals") * cfg["quantum"]["shots"] * cfg["comms"]["latency"]["shot_time_s"]


def energy_breakdown(row, cfg, method, p_qpu_w=None, n_quantum_links=40):
    e = cfg["energy"]
    p_qpu = e["p_qpu_w"] if p_qpu_w is None else p_qpu_w
    duration = cfg["simulation"]["duration_s"]

    if method.startswith("qits"):
        quantum_policy = method != "qits_noqopt"
        quantum_channel = method != "qits_noqkd"
        cpu_s = 0.0 if quantum_policy else _num(row, "cpu_s_total")   # classical SPSA + policy
        comp = e["p_cpu_w"] * cpu_s + p_qpu * qpu_busy_s(row, cfg)
        bits = 8 * (_num(row, "consensus_bytes") + _num(row, "qkd_classical_bytes"))
        qkd_hw = e["p_qkd_link_w"] * n_quantum_links * duration if quantum_channel else 0.0
    elif method.startswith("dqn"):
        comp = e["p_cpu_w"] * _num(row, "cpu_s_total")                   # measured inference time
        bits = 8 * _num(row, "comm_bytes")
        qkd_hw = 0.0
    else:                                                                # fixed-time / rule-based
        comp, bits, qkd_hw = 0.0, 0.0, 0.0

    return {"computation_j": comp, "communication_j": bits * e["e_bit_j"], "qkd_hardware_j": qkd_hw,
            "total_j": comp + bits * e["e_bit_j"] + qkd_hw}
