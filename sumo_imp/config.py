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


DEMANDS = ("dataset", "closed_loop")


class Demand:
    """Which trips (and density calibration) an experiment uses.

    dataset      the sumo_data trips: λ levels calibrated with SUMO's own (navigation) routing
    closed_loop  trips at λ levels calibrated with the evaluation's own setup (fixed-time lights and
                 the base paper's routing, experiments/calibrate_closed_loop.py), so that low / medium /
                 high are 30 / 60 / 90 % of the saturation of the network as it is evaluated
    Models, θ and results of the two settings are kept apart by `suffix`.
    """

    def __init__(self, scfg, name="dataset"):
        if name not in DEMANDS:
            raise ValueError(f"demand must be one of {DEMANDS}")
        self.scfg, self.name = scfg, name
        self.suffix = "" if name == "dataset" else f"_{name}"

    @property
    def calibration_file(self):
        return results_dir(self.scfg) / "calibration_closed_loop.yaml"

    def lambdas(self):
        from common import density_lambdas
        if self.name == "dataset":
            return density_lambdas(self.scfg)
        import yaml
        if not self.calibration_file.exists():
            raise FileNotFoundError(f"{self.calibration_file} missing; run "
                                    "python -m sumo_imp.experiments.calibrate_closed_loop")
        lam_max = yaml.safe_load(self.calibration_file.read_text())["lambda_max"]
        return {k: round(f * lam_max, 3) for k, f in self.scfg["demand"]["density_levels"].items()}

    def csv(self, density, seed):
        from common import demand_csv
        if self.name == "dataset":
            return demand_csv(self.scfg, density, seed)
        return SUMO_ROOT / "data" / tag(self.scfg) / self.name / density / f"seed_{seed}.csv"
