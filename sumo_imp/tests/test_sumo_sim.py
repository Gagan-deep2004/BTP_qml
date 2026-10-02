"""Tests for the SUMO connector (5 x 5 network). Run from the project root:
    python -m pytest sumo_imp/tests -q
"""
import numpy as np
import pandas as pd
import pytest

import sumo_imp  # noqa: F401  (paths)
from build_network import build
from common import GridMap, demand_csv, net_file
from generate_demand import make_trips, trips_to_frame
from qits.agents.base import Controller, FixedTimeController
from qits.agents.rule_based import RuleBasedController
from qits.traffic.grid import E, N, S, W
from sumo_imp.config import load_config
from sumo_imp.simulator import SumoSimulator

PORT = iter(range(34000, 34100))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    cfg, scfg = load_config(5)
    if not net_file(scfg).exists():
        build(scfg)
    gmap = GridMap(scfg)
    xml = make_trips(scfg, 2.0, 7, tmp_path_factory.mktemp("trips") / "t.xml")
    return cfg, scfg, gmap, trips_to_frame(gmap, xml)


def _sim(setup, ctl, duration=None, seed=7):
    cfg, scfg, gmap, demand = setup
    return SumoSimulator(cfg, scfg, demand, ctl, seed=seed, port=next(PORT), gmap=gmap, duration=duration)


def test_vehicles_are_conserved_and_take_shortest_paths(setup):
    sim = _sim(setup, FixedTimeController(), duration=600)
    for _ in range(600):
        sim.step()
    assert sim.n_spawned == len(sim.trips) + len(sim.active) + len(sim.pending)
    res = sim.results()
    trips = res["trips"]
    assert len(trips) > 500
    assert (trips.hops == trips.hops_min).all()               # only distance-reducing moves
    assert (trips.delay >= -1e-6).all() and trips.delay.notna().all()
    x = sim.devices.x()
    assert x.shape == (25, 16) and (x >= 0).all() and (x <= 1).all()


class _NSOnlyCheck(Controller):
    name = "check"

    def reset(self, sim):
        self.bad = 0

    def step(self, sim, t):
        if 5 < t < 25 or 35 < t < 55:                         # well inside NS green / EW green
            ns = t < 30
            for v, j, appr, _ in sim.tls[:: 3]:
                state = sim.conn.trafficlight.getRedYellowGreenState(j)
                for a, ch in zip(appr, state):
                    green = ch in "Gg"
                    self.bad += green != ((a in (N, S)) == ns)


def test_lights_follow_the_base_signal_plan(setup):
    ctl = _NSOnlyCheck()
    sim = _sim(setup, ctl, duration=60)
    sim.run()
    assert ctl.bad == 0


class _PreferEast(Controller):
    name = "east"

    def reset(self, sim):
        p = np.full((sim.grid.n_nodes, 4), 1e-6)
        p[:, E] = 1.0
        sim.route_p = p


def test_routing_follows_route_p(setup):
    sim = _sim(setup, _PreferEast(), duration=300)
    g, checked = sim.grid, 0
    for t in range(300):
        sim.step()
        if t % 20 == 19:
            for veh in sim.active.values():
                u = g.link_to[veh.link]
                if veh.next_dir >= 0 and E in g.productive_dirs(u, veh.dest):
                    assert veh.next_dir == E
                    checked += 1
    sim.close()
    assert checked > 50


@pytest.mark.parametrize("make", [lambda cfg: RuleBasedController.from_config(cfg), "dqn", "qits"])
def test_base_controllers_run_unchanged(setup, make):
    cfg = setup[0]
    if make == "dqn":
        import torch
        from qits.agents.dqn_agent import DQNController, DQNLearner
        torch.manual_seed(0)
        ctl = DQNController(cfg, DQNLearner(cfg).q, epsilon=0.0)
    elif make == "qits":
        from qits.agents.qits_agent import QITSController
        c = {**cfg, "quantum": {**cfg["quantum"], "backend": "exact"}}
        ctl = QITSController(c, seed=7)
    else:
        ctl = make(cfg)
    res = _sim(setup, ctl, duration=200).run()
    assert res["summary"]["completed"] > 50
    if make == "qits":
        assert len(ctl.epoch_log) == 19 and not np.allclose(ctl.p, ctl.p[0])
