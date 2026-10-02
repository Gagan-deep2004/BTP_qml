"""Shared helpers: configuration, file locations, SUMO binaries and the SUMO <-> grid mapping.

Intersections and links are numbered exactly as in base_imp (qits.traffic.grid.Grid), so the
SUMO dataset can be fed to the base and novelty controllers unchanged:
    node v = row * cols + col (row 0 = north edge),  directions 0 = N, 1 = E, 2 = S, 3 = W.
netgenerate places junction (col, row) at x = 100 col, y = 100 (rows - 1 - row).
"""
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "base_imp"))          # Grid numbering shared with the base code

from qits.traffic.grid import Grid  # noqa: E402

CONFIG = ROOT / "config.yaml"


def _sumo_home():
    if os.environ.get("SUMO_HOME"):
        return Path(os.environ["SUMO_HOME"])
    import sumo                                       # pip package eclipse-sumo
    os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    return Path(sumo.SUMO_HOME)


SUMO_HOME = _sumo_home()


def binary(name):
    exe = SUMO_HOME / "bin" / (name + (".exe" if os.name == "nt" else ""))
    return str(exe if exe.exists() else name)


def tool(name):
    return str(SUMO_HOME / "tools" / name)


# ---------------------------------------------------------------- configuration
def tag(cfg):
    return f"{cfg['network']['rows']}x{cfg['network']['cols']}"


def calibration_file(cfg):
    return ROOT / f"calibration_{tag(cfg)}.yaml"


def load_config(rows=None, cols=None):
    """config.yaml, optionally with another grid size, plus the calibration for that size."""
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if rows:
        cfg["network"]["rows"] = rows
        cfg["network"]["cols"] = cols or rows
    cal = calibration_file(cfg)
    if cal.exists():
        with open(cal, encoding="utf-8") as f:
            cfg["demand"].update((yaml.safe_load(f) or {}).get("demand", {}))
    return cfg


def save_calibration(cfg, lambda_max):
    with open(calibration_file(cfg), "w", encoding="utf-8") as f:
        f.write("# Written by calibrate_density.py. Do not edit by hand.\n")
        yaml.safe_dump({"demand": {"lambda_max": float(lambda_max)}}, f)


def density_lambdas(cfg):
    lam_max = cfg["demand"]["lambda_max"]
    if lam_max is None:
        raise RuntimeError(f"lambda_max not calibrated for {tag(cfg)}; run: python calibrate_density.py")
    return {k: round(f * lam_max, 3) for k, f in cfg["demand"]["density_levels"].items()}


# ---------------------------------------------------------------- file locations
def net_file(cfg):
    return ROOT / "network" / f"grid_{tag(cfg)}.net.xml"


def data_dir(cfg):
    return ROOT / "data" / tag(cfg)


def trips_file(cfg, density, seed):
    return data_dir(cfg) / "demand" / density / f"seed_{seed}.trips.xml"


def demand_csv(cfg, density, seed):
    return data_dir(cfg) / "demand" / density / f"seed_{seed}.csv"


def devices_file(cfg, density, seed):
    return data_dir(cfg) / "devices" / density / f"seed_{seed}.npz"


# ---------------------------------------------------------------- SUMO <-> grid
class GridMap:
    """Maps SUMO junction / edge ids to base-grid node / link ids, and back."""

    def __init__(self, cfg):
        import sumolib
        net = cfg["network"]
        self.grid = Grid(net["rows"], net["cols"])
        self.net = sumolib.net.readNet(str(net_file(cfg)))
        spacing = net["link_length_m"]
        g, rows = self.grid, net["rows"]

        self.node_of_junction = {}
        for j in self.net.getNodes():
            x, y = j.getCoord()
            col, row = int(round(x / spacing)), rows - 1 - int(round(y / spacing))
            self.node_of_junction[j.getID()] = row * net["cols"] + col
        self.junction_of_node = {v: k for k, v in self.node_of_junction.items()}

        self.link_of_edge, self.edge_of_link = {}, {}
        self.length = np.zeros(g.n_links)
        for e in self.net.getEdges():
            u = self.node_of_junction[e.getFromNode().getID()]
            v = self.node_of_junction[e.getToNode().getID()]
            d = int(np.flatnonzero(g.neighbour[u] == v)[0])
            lid = int(g.out_link[u, d])
            self.link_of_edge[e.getID()] = lid
            self.edge_of_link[lid] = e.getID()
            self.length[lid] = e.getLength()
        if len(self.link_of_edge) != g.n_links:
            raise ValueError(f"network has {len(self.link_of_edge)} edges, grid expects {g.n_links}")
