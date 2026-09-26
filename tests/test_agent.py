import numpy as np

from qits.agents.qits_agent import QITSController, signal_cost
from qits.config import load_config
from qits.traffic.demand import generate_demand
from qits.traffic.grid import Grid
from qits.traffic.simulator import TrafficSimulator


def _run(cfg, steps=120, **kw):
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    ctl = QITSController(cfg, seed=3, **kw)
    sim = TrafficSimulator(cfg, generate_demand(g, 3.0, steps, 3), ctl, 3)
    sim.run(steps)
    return sim, ctl


def test_signal_cost_optimum():
    S = np.linspace(0.2, 0.8, 601)
    best = S[np.argmin(signal_cost(S, 9.0, 4.0, 1.0, 60))]
    assert abs(best - 3 / (3 + 2)) < 1e-2        # S* = √9 / (√9 + √4)


def test_qits_controls_routing_and_signals():
    cfg = load_config()
    cfg["quantum"]["backend"] = "exact"
    sim, ctl = _run(cfg)
    assert len(ctl.epoch_log) == 11                          # decisions at t = 10, 20, ..., 110
    np.testing.assert_allclose(sim.route_p, ctl.p)
    np.testing.assert_allclose(sim.route_p.sum(1), 1.0)
    assert np.all(sim.route_p[sim.grid.valid_mask == 0] == 0)
    assert np.all((sim.signals.pending_split >= 0.2) & (sim.signals.pending_split <= 0.8))
    s = ctl.summary()
    assert 0.9 < s["pdr"] <= 1.0 and s["circuit_evals"] == 11 * (2 * 2 + 1) * 25


def test_classical_policy_interface():
    from qits.agents.classical_policy import ClassicalPolicy
    pol = ClassicalPolicy(seed=0)
    th = pol.init_params(3)
    assert th.shape == (3, 80)                        # 4 routing outputs + 1 split output, 16 features
    mask = np.array([[1.0, 1, 1, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
    p, E, S = pol.forward(np.random.default_rng(0).random((3, 16)), th, mask)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert np.all(p[mask == 0] == 0) and E.shape == (3, 4) and np.all(np.abs(E) <= 1)
    assert np.all((S >= 0.2) & (S <= 0.8))


def test_noqopt_uses_no_quantum_circuits():
    cfg = load_config()
    _, ctl = _run(cfg, policy_kind="classical")
    s = ctl.summary()
    assert s["circuit_evals"] == 0
    assert s["T_proc_s"] < 1e-3                       # CPU inference, no 8192-shot QPU time


def test_noqkd_uses_classical_channels():
    cfg = load_config()
    cfg["quantum"]["backend"] = "exact"
    _, ctl = _run(cfg, channel="classical")
    s = ctl.summary()
    assert s["reentanglements"] == 0 and s["qkd_classical_bytes"] == 0
    assert s["T_ent_s"] == 0 and s["T_meas_s"] == 0 and np.isnan(s["mean_fidelity"])
    assert s["consensus_bytes"] > 0 and s["pdr"] > 0.9


def test_no_consensus_variant_sends_no_messages():
    cfg = load_config()
    cfg["quantum"]["backend"] = "exact"
    _, ctl = _run(cfg, use_consensus=False)
    assert ctl.net.stats["msgs_sent"] == 0 and ctl.mu == 0.0
