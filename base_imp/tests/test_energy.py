import pytest

from qits.config import load_config
from qits.energy import energy_breakdown, qpu_busy_s


def test_energy_accounting():
    cfg = load_config()
    q_row = {"circuit_evals": 1000, "consensus_bytes": 1e5, "qkd_classical_bytes": 1e6, "cpu_s_total": 50.0}
    busy = 1000 * cfg["quantum"]["shots"] * cfg["comms"]["latency"]["shot_time_s"]
    assert qpu_busy_s(q_row, cfg) == pytest.approx(busy)

    e = energy_breakdown(q_row, cfg, "qits", p_qpu_w=100.0)
    assert e["computation_j"] == pytest.approx(100.0 * busy)            # simulated CPU time not counted
    assert e["communication_j"] == pytest.approx(8 * 1.1e6 * cfg["energy"]["e_bit_j"])
    assert e["qkd_hardware_j"] == pytest.approx(cfg["energy"]["p_qkd_link_w"] * 40 * 1800)

    noqopt = energy_breakdown({**q_row, "circuit_evals": 0}, cfg, "qits_noqopt")
    assert noqopt["computation_j"] == pytest.approx(cfg["energy"]["p_cpu_w"] * 50.0)
    assert energy_breakdown(q_row, cfg, "qits_noqkd")["qkd_hardware_j"] == 0.0
    assert energy_breakdown({}, cfg, "rule_based")["total_j"] == 0.0
    d = energy_breakdown({"cpu_s_total": 2.0, "comm_bytes": 1e6}, cfg, "dqn")
    assert d["total_j"] == pytest.approx(cfg["energy"]["p_cpu_w"] * 2.0 + 8e6 * cfg["energy"]["e_bit_j"])
