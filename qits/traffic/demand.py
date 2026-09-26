"""Synthetic demand (the paper's dataset): Poisson arrivals, random OD pairs."""
import numpy as np
import pandas as pd


def generate_demand(grid, lam, duration_s=1800, seed=42, min_hops=2):
    """Per-second Poisson(lam) arrivals, each with a uniformly random origin/destination
    at least `min_hops` blocks apart. Returns a DataFrame sorted by departure time."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam, size=duration_s)
    t_depart = np.repeat(np.arange(duration_s), counts)
    n = len(t_depart)

    origin = rng.integers(0, grid.n_nodes, n)
    dest = rng.integers(0, grid.n_nodes, n)
    bad = np.array([grid.manhattan(o, d) < min_hops for o, d in zip(origin, dest)], dtype=bool)
    while bad.any():  # rejection-resample invalid pairs
        origin[bad] = rng.integers(0, grid.n_nodes, bad.sum())
        dest[bad] = rng.integers(0, grid.n_nodes, bad.sum())
        bad = np.array([grid.manhattan(o, d) < min_hops for o, d in zip(origin, dest)], dtype=bool)

    return pd.DataFrame({"veh_id": np.arange(n), "t_depart": t_depart,
                         "origin": origin, "dest": dest})
