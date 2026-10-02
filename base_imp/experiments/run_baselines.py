"""Run classical baselines (fixed-time, rule-based, trained DQN) on the generated dataset.

    python -m experiments.run_baselines --seeds 30
    python -m experiments.run_baselines --seeds 30 --controllers dqn
Output: results/baselines/summary.csv  (one row per controller x density x seed; merged
        with earlier rows, newest wins), results/baselines/runs/{controller}/..._delay.csv
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from experiments.generate_dataset import load_demand
from qits.agents.base import FixedTimeController
from qits.agents.rule_based import RuleBasedController
from qits.config import ROOT, density_lambdas, load_config
from qits.traffic.simulator import TrafficSimulator

OUT = ROOT / "results" / "baselines"


def _dqn(cfg, mode="phase"):
    import torch
    from qits.agents.dqn_agent import DQNController, DQNLearner
    torch.set_num_threads(1)
    name = "dqn" if mode == "phase" else "dqn_split"
    qnet = DQNLearner.load_qnet(cfg, ROOT / "results" / "dqn" / f"{name}.pt", mode)
    return DQNController(cfg, qnet, epsilon=0.0, mode=mode)


CONTROLLERS = {
    "fixed_time": lambda cfg: FixedTimeController(),
    "rule_based": RuleBasedController.from_config,
    "dqn": _dqn,                                        # standard: phase choice every 5 s
    "dqn_split": lambda cfg: _dqn(cfg, "split"),        # same action space as Q-ITS
}


def run_one(args):
    name, density, seed = args
    cfg = load_config()
    ctl = CONTROLLERS[name](cfg)
    if hasattr(ctl, "seed"):
        ctl.seed = seed
    res = TrafficSimulator(cfg, load_demand(density, seed), ctl, seed).run()
    run_dir = OUT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    trips = res["trips"].assign(minute=res["trips"].t_arrive // 60)
    trips.groupby("minute").delay.mean().reset_index().to_csv(
        run_dir / f"{density}_seed{seed}_delay.csv", index=False)
    if getattr(ctl, "decision_t", None):
        pd.DataFrame({"t": ctl.decision_t, "entropy": ctl.entropies}).to_csv(
            run_dir / f"{density}_seed{seed}_entropy.csv", index=False)
    extra = ctl.summary() if hasattr(ctl, "summary") else {}
    return {"density": density, "seed": seed, **res["summary"], **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (max n_runs)")
    ap.add_argument("--controllers", nargs="+", default=list(CONTROLLERS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"

    cfg = load_config()
    base = cfg["simulation"]["base_seed"]
    jobs = [(c, d, s) for c in args.controllers for d in density_lambdas(cfg)
            for s in range(base, base + args.seeds)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run_one, jobs))

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "summary.csv"
    if out.exists():
        df = pd.concat([pd.read_csv(out), df]).drop_duplicates(["controller", "density", "seed"],
                                                               keep="last")
    df.to_csv(out, index=False)

    order = list(density_lambdas(cfg))
    table = (df.groupby(["controller", "density"])
               [["avg_delay_s", "avg_travel_time_s", "congestion_index", "throughput_vps",
                 "completion_rate"]]
               .agg(["mean", "std"]).round(3))
    table = table.reindex(order, level="density")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(table)


if __name__ == "__main__":
    main()
