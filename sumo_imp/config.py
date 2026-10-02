"""One configuration for the SUMO runs: the base-paper configuration (base_imp/configs/base.yaml,
unchanged: quantum, comms, Q-ITS, DQN, signal and device settings) with the network size, density
calibration and demand taken from the SUMO dataset (sumo_data/config.yaml + calibration)."""
from pathlib import Path

from common import load_config as load_sumo_config      # sumo_data/common.py
from common import tag
from qits.config import load_config as load_base_config

SUMO_ROOT = Path(__file__).resolve().parent
SHARED = ["link_length_m", "lanes", "free_flow_speed_mps", "veh_spacing_m", "sat_flow_vps"]


def load_config(rows=None, cols=None):
    """Returns (cfg, scfg): cfg is what the base controllers read, scfg the SUMO dataset config."""
    scfg = load_sumo_config(rows, cols)
    cfg = load_base_config()
    for k in SHARED:
        if cfg["network"][k] != scfg["network"][k]:
            raise ValueError(f"network.{k} differs between base ({cfg['network'][k]}) and SUMO "
                             f"({scfg['network'][k]}) configs")
    cfg["network"]["rows"], cfg["network"]["cols"] = scfg["network"]["rows"], scfg["network"]["cols"]
    cfg["demand"]["lambda_max"] = scfg["demand"]["lambda_max"]
    cfg["demand"]["density_levels"] = scfg["demand"]["density_levels"]
    for k in ("signal", "devices"):
        for key in set(cfg[k]) & set(scfg[k]):
            if cfg[k][key] != scfg[k][key]:
                raise ValueError(f"{k}.{key} differs between base and SUMO configs")
    return cfg, scfg


def results_dir(scfg):
    return SUMO_ROOT / "results" / tag(scfg)
