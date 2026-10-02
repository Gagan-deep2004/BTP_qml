import numpy as np
import pandas as pd
import pytest

from qits.agents.base import FixedTimeController
from qits.agents.rule_based import RuleBasedController
from qits.config import load_config
from qits.traffic.demand import generate_demand
from qits.traffic.grid import E, N, S, W, Grid
from qits.traffic.signals import Signals
from qits.traffic.simulator import TrafficSimulator


def test_grid_links_and_directions():
    g = Grid(5, 5)
    assert g.n_nodes == 25
    assert g.n_links == 2 * (5 * 4 * 2)          # 80 directed links
    lid = g.out_link[0, S]                         # node 0 -> node 5 southbound
    assert g.link_to[lid] == 5 and g.in_link[5, N] == lid
    assert g.neighbour[0, N] == -1 and g.neighbour[0, W] == -1
    assert set(g.productive_dirs(0, 24)) == {S, E}
    assert g.productive_dirs(24, 20) == [W]


def test_demand_min_hops_and_rate():
    g = Grid(5, 5)
    df = generate_demand(g, lam=4.0, duration_s=1800, seed=42, min_hops=2)
    assert all(g.manhattan(o, d) >= 2 for o, d in zip(df.origin, df.dest))
    assert abs(len(df) / 1800 - 4.0) < 0.2


def test_signal_split_green_times():
    s = Signals(1, cycle_s=60, lost_time_s=3, initial_split=0.5)
    greens = np.array([s.green_mask(t)[0] for t in range(60)])
    assert greens[:, N].sum() == 27 and greens[:, E].sum() == 27
    assert not (greens[:, N] & greens[:, E]).any()  # never conflicting greens


def test_phase_mode_all_red_on_switch():
    s = Signals(2, lost_time_s=3)
    s.use_phase_mode()
    greens = []
    for t in range(12):
        if t == 2:
            s.request_phase(np.array([1, 0]))   # node 0 switches to EW, node 1 keeps NS
        s.step(t)
        greens.append(s.green_mask(t))
    g = np.array(greens)                        # (t, node, approach)
    assert g[:2, 0, N].all() and not g[:2, 0, E].any()
    assert not g[2:5, 0].any()                  # exactly 3 s all-red at t = 2, 3, 4
    assert g[5:, 0, E].all() and not g[5:, 0, N].any()
    assert g[:, 1, N].all()                     # node 1 unaffected
    assert not (g[:, :, N] & g[:, :, E]).any()


@pytest.mark.parametrize("lanes, expected_tt", [(1, 39), (2, 38)])
def test_single_vehicle_trip(lanes, expected_tt):
    cfg = load_config()
    cfg["network"]["lanes"] = lanes
    # 0 -> 2 is an eastbound trip through node 1 (arrives on its W approach)
    demand = pd.DataFrame({"veh_id": [0], "t_depart": [0], "origin": [0], "dest": [2]})
    sim = TrafficSimulator(cfg, demand, FixedTimeController(), seed=1)
    res = sim.run(200)
    trip = res["trips"].iloc[0]
    assert trip.hops == trip.hops_min == 2
    # reaches node 1 at t=8 during NS green; EW green starts at t=30. With 1 lane
    # (0.5 veh/s) there is a 1 s start-up loss -> leaves at 31; with 2 lanes at 30.
    assert trip.travel_time == expected_tt
    assert trip.delay == expected_tt - 2 * sim.ff_steps


def test_green_wave_has_no_delay():
    cfg = load_config()
    # 0 -> 10 southbound through node 5: arrives at t=8 during NS green -> no stop
    demand = pd.DataFrame({"veh_id": [0], "t_depart": [0], "origin": [0], "dest": [10]})
    sim = TrafficSimulator(cfg, demand, FixedTimeController(), seed=1)
    trip = sim.run(100)["trips"].iloc[0]
    assert trip.delay == 0


def test_vehicle_conservation():
    cfg = load_config()
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    demand = generate_demand(g, lam=3.0, duration_s=600, seed=7)
    sim = TrafficSimulator(cfg, demand, RuleBasedController.from_config(cfg), seed=7)
    sim.run(600)
    in_links = sum(len(m) + len(q) for m, q in zip(sim.moving, sim.queue))
    waiting = sum(len(b) for b in sim.entry.values())
    assert sim.n_spawned == len(sim.trips) + in_links + waiting
    assert all(sim.occupancy(l) <= sim.storage for l in range(g.n_links))
