import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qits.agents.dqn_agent import DQNController, DQNLearner, QNet, state_dim  # noqa: E402
from qits.config import load_config  # noqa: E402
from qits.traffic.demand import generate_demand  # noqa: E402
from qits.traffic.grid import Grid  # noqa: E402
from qits.traffic.simulator import TrafficSimulator  # noqa: E402


def test_qnet_architecture_matches_paper():
    q = QNet(19, (64, 128, 64))
    sizes = [m.out_features for m in q.net if isinstance(m, torch.nn.Linear)]
    assert sizes == [64, 128, 64, 2]


def test_dqn_learns_online_and_controls_phases():
    cfg = load_config()
    cfg["dqn"]["batch"] = 8
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    learner = DQNLearner(cfg, seed=0)
    ctl = DQNController(cfg, learner.q, learner=learner, epsilon=0.5, seed=0)
    sim = TrafficSimulator(cfg, generate_demand(g, 3.0, 300, 0), ctl, 0)
    sim.run(300)
    assert sim.signals.mode == "phase"
    assert learner.buffer.size == (300 // 5 - 2) * g.n_nodes      # one transition per node per decision
    assert learner.steps > 0 and np.isfinite(learner.losses).all()
    s = ctl.summary()
    assert s["decision_latency_s"] > 2 * cfg["dqn"]["latency"]["access_s"]
    assert s["comm_bytes"] > 0 and state_dim(cfg) == 19


def test_dqn_split_mode_uses_qits_action_space():
    cfg = load_config()
    cfg["dqn"]["batch"] = 4
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    learner = DQNLearner(cfg, seed=0, mode="split")
    ctl = DQNController(cfg, learner.q, learner=learner, epsilon=1.0, seed=0, mode="split")
    sim = TrafficSimulator(cfg, generate_demand(g, 3.0, 400, 0), ctl, 0)
    sim.run(400)
    assert ctl.name == "dqn_split" and sim.signals.mode == "split"
    assert learner.buffer.size == (400 // 60 - 1) * g.n_nodes   # decisions at t = 60..360
    assert set(np.round(sim.signals.split, 1)) <= {0.3, 0.4, 0.5, 0.6, 0.7}
