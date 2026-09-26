"""Run the Q-ITS controller (Algorithm 1) on the dataset.

1. Pre-training: `--pretrain-episodes` episodes on a demand seed outside the test set
   (cfg.qits.pretrain_seed), starting from random θ. Saves θ to results/qits/theta_pretrained.npy.
   (Mirrors the DQN baseline, which is trained before it is evaluated.)
2. Evaluation on test seeds / densities, starting from the pre-trained θ and continuing to
   learn online, exactly as in Algorithm 1.

    python -m experiments.run_qits --seeds 5
    python -m experiments.run_qits --seeds 30 --backend noisy
Output: results/qits/summary.csv, results/qits/runs/{density}_seed{seed}_{epochs|steps|delay}.csv
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from experiments.generate_dataset import load_demand
from qits.agents.qits_agent import QITSController
from qits.config import ROOT, density_lambdas, load_config
from qits.traffic.demand import generate_demand
from qits.traffic.grid import Grid
from qits.traffic.simulator import TrafficSimulator

OUT = ROOT / "results" / "qits"


def theta_path(backend, variant="qits"):
    """Pre-trained θ per variant: every ablation is pre-trained under its own setting.
    Full Q-ITS and the routing/signal diagnostics share the full Q-ITS file."""
    if variant in ("qits", "qits_routing_only", "qits_signals_only"):
        return OUT / f"theta_pretrained_{backend}.npy"
    return OUT / f"theta_pretrained_{variant}_{backend}.npy"


def pretrain(cfg, episodes, variant="qits"):
    q = cfg["qits"]
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    lam = density_lambdas(cfg)[q["pretrain_density"]]
    theta = None
    for ep in range(episodes):
        demand = generate_demand(g, lam, cfg["simulation"]["duration_s"], q["pretrain_seed"] + ep,
                                 cfg["demand"]["min_hops"])
        ctl = QITSController(cfg, seed=q["pretrain_seed"] + ep, theta0=theta, **VARIANTS[variant])
        res = TrafficSimulator(cfg, demand, ctl, q["pretrain_seed"] + ep).run()
        theta = ctl.theta
        print(f"[{variant}] pretrain episode {ep + 1}/{episodes}: delay={res['summary']['avg_delay_s']:.2f}s "
              f"final dθ={ctl.summary()['final_dtheta']:.3f}")
    return theta


def delay_by_minute(trips):
    trips = trips.assign(minute=trips.t_arrive // 60)
    return trips.groupby("minute").delay.mean().reset_index()


def run_one(job):
    density, seed, backend, use_pretrained, variant = job
    cfg = load_config()
    cfg["quantum"]["backend"] = backend
    theta0 = np.load(theta_path(backend, variant)) if use_pretrained else None
    ctl = QITSController(cfg, seed=seed, theta0=theta0, **VARIANTS[variant])
    res = TrafficSimulator(cfg, load_demand(density, seed), ctl, seed).run()
    run_dir = OUT / "runs" / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{density}_seed{seed}"
    pd.DataFrame(ctl.epoch_log).to_csv(run_dir / f"{stem}_epochs.csv", index=False)
    res["steps"].to_csv(run_dir / f"{stem}_steps.csv", index=False)
    delay_by_minute(res["trips"]).to_csv(run_dir / f"{stem}_delay.csv", index=False)
    return {"variant": variant, "density": density, "seed": seed, "backend": backend,
            **res["summary"], **ctl.summary()}


VARIANTS = {
    "qits": {},
    # ablations (paper Table IV)
    "qits_noqopt": {"policy_kind": "classical"},
    "qits_noqkd": {"channel": "classical"},
    "qits_nocons": {"use_consensus": False},
    # diagnostics (not in the paper): which traffic action produces the gain
    "qits_routing_only": {"control_signals": False},
    "qits_signals_only": {"control_routing": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--densities", nargs="+", default=None)
    ap.add_argument("--backend", default=None, help="exact | shots | noisy (default: config)")
    ap.add_argument("--pretrain-episodes", type=int, default=2)
    ap.add_argument("--no-pretrain", action="store_true", help="start every run from random θ")
    ap.add_argument("--variants", nargs="+", default=["qits"])
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel runs (each ~0.5 GB RAM; 16 exhausted 16 GB)")
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"          # inherited by worker processes

    cfg = load_config()
    backend = args.backend or cfg["quantum"]["backend"]
    cfg["quantum"]["backend"] = backend
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.no_pretrain:
        for v in args.variants:
            if not theta_path(backend, v).exists():
                np.save(theta_path(backend, v), pretrain(cfg, args.pretrain_episodes, v))

    base = cfg["simulation"]["base_seed"]
    densities = args.densities or list(density_lambdas(cfg))
    jobs = [(d, s, backend, not args.no_pretrain, v) for v in args.variants for d in densities
            for s in range(base, base + args.seeds)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run_one, jobs))

    df = pd.DataFrame(rows)
    out = OUT / "summary.csv"
    if out.exists():   # merge with earlier runs, newest result wins
        old = pd.read_csv(out)
        key = ["variant", "density", "seed", "backend"]
        df = pd.concat([old, df]).drop_duplicates(key, keep="last")
    df.to_csv(out, index=False)

    cols = ["avg_delay_s", "congestion_index", "decision_latency_s", "consensus_entropy", "pdr",
            "mean_fidelity"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(df[df.backend == backend].groupby(["variant", "density"])[cols].mean()
          .reindex(densities, level="density").round(4))


if __name__ == "__main__":
    main()
