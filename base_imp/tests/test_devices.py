import numpy as np

from qits.agents.rule_based import RuleBasedController
from qits.config import load_config
from qits.traffic.demand import generate_demand
from qits.traffic.grid import Grid
from qits.traffic.simulator import TrafficSimulator


def _sim(cfg, steps=400, lam=3.0, seed=5):
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    demand = generate_demand(g, lam, steps, seed)
    sim = TrafficSimulator(cfg, demand, RuleBasedController.from_config(cfg), seed)
    sim.run(steps)
    return sim


def test_perfect_devices_match_ground_truth():
    cfg = load_config()
    cfg["devices"].update(penetration=1.0, speed_noise_mps=0.0)
    sim = _sim(cfg)
    np.testing.assert_allclose(sim.devices.features, sim.devices.true_features, atol=1e-12)


def test_feature_shape_range_and_boundary():
    cfg = load_config()
    sim = _sim(cfg)
    x = sim.devices.x()
    assert x.shape == (25, 16)
    assert x.min() >= 0.0 and x.max() <= 1.0
    # node 0 (top-left corner) has no N and no W approach -> those 8 features are zero
    f0 = sim.devices.features[0]
    assert np.all(f0[0] == 0) and np.all(f0[3] == 0)


def test_link_delay_estimates():
    cfg = load_config()
    sim = _sim(cfg)
    delay = sim.devices.link_delay(sim.ff_steps)
    valid = sim.grid.valid_mask.astype(bool)
    assert np.all(delay[valid] >= sim.ff_steps)
    assert np.all(delay[~valid] == 0)


def test_probe_logging():
    cfg = load_config()
    cfg["devices"]["save_probes"] = True
    sim = _sim(cfg, steps=100)
    probes = np.array(sim.devices.probes)
    assert len(probes) > 0
    assert probes[:, 3].min() >= 0 and probes[:, 3].max() <= cfg["network"]["link_length_m"]
