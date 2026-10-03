"""The base-paper controllers, built from base_imp unchanged, with models trained on SUMO.

    fixed_time   fixed 60 s cycle, 50/50 split
    rule_based   threshold rule on the previous cycle's demand (paper Sec. V)
    dqn          DQN, phase choice every 5 s           (trained by experiments/train_dqn.py)
    dqn_split    DQN, split per cycle (Q-ITS's action space)
    qits         Q-ITS, Algorithm 1, from its SUMO pre-trained θ, learning online
    qits_noqopt / qits_noqkd / qits_nocons   paper Table IV ablations
    qits_routing_only / qits_signals_only    diagnostics (which action produces the gain)
"""
import numpy as np

from qits.agents.base import FixedTimeController
from qits.agents.rule_based import RuleBasedController

from .config import results_dir

QITS_VARIANTS = {
    "qits": {},
    "qits_noqopt": {"policy_kind": "classical"},
    "qits_noqkd": {"channel": "classical"},
    "qits_nocons": {"use_consensus": False},
    "qits_routing_only": {"control_signals": False},
    "qits_signals_only": {"control_routing": False},
}


def dqn_path(scfg, mode, suffix=""):
    return results_dir(scfg) / "dqn" / (("dqn" if mode == "phase" else "dqn_split") + suffix + ".pt")


def pretrain_variant(variant):
    """The routing/signal diagnostics run from the full Q-ITS θ."""
    return "qits" if variant in ("qits", "qits_routing_only", "qits_signals_only") else variant


def theta_path(scfg, backend, variant="qits", suffix=""):
    """Pre-trained θ per variant and demand setting."""
    return results_dir(scfg) / "qits" / f"theta_pretrained_{pretrain_variant(variant)}_{backend}{suffix}.npy"


def _dqn(cfg, scfg, seed, mode, suffix):
    import torch
    from qits.agents.dqn_agent import DQNController, DQNLearner
    torch.set_num_threads(1)
    path = dqn_path(scfg, mode, suffix)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; train it first: python -m sumo_imp.experiments.train_dqn")
    return DQNController(cfg, DQNLearner.load_qnet(cfg, path, mode), epsilon=0.0, seed=seed, mode=mode)


def _qits(cfg, scfg, seed, variant, suffix):
    from qits.agents.qits_agent import QITSController
    path = theta_path(scfg, cfg["quantum"]["backend"], variant, suffix)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run_eval.py pre-trains it")
    ctl = QITSController(cfg, seed=seed, theta0=np.load(path), **QITS_VARIANTS[variant])
    ctl.name = variant
    return ctl


def make_controller(name, cfg, scfg, seed, suffix=""):
    """suffix: demand setting of the trained models ("" = dataset, "_closed_loop")."""
    if name == "fixed_time":
        return FixedTimeController()
    if name == "rule_based":
        return RuleBasedController.from_config(cfg)
    if name in ("dqn", "dqn_split"):
        return _dqn(cfg, scfg, seed, "phase" if name == "dqn" else "split", suffix)
    if name in QITS_VARIANTS:
        return _qits(cfg, scfg, seed, name, suffix)
    raise KeyError(f"unknown controller {name!r}")
