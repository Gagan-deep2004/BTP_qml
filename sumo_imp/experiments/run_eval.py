"""Evaluate the base-paper controllers on the SUMO test trips (sumo_data/data/<R>x<C>/demand/).

Q-ITS: if its pre-trained θ is missing, it is first pre-trained for --pretrain-episodes episodes of
1800 s on SUMO trips outside the test seeds (cfg.qits.pretrain_seed = 1000+, medium density), exactly
as base_imp/experiments/run_qits.py does; each test run then starts from that θ and keeps learning
online (Algorithm 1). The DQN models must be trained first (train_dqn.py).

    python -m sumo_imp.experiments.run_eval --controllers fixed_time rule_based --workers 90
    python -m sumo_imp.experiments.run_eval --controllers qits --workers 90
    python -m sumo_imp.experiments.run_eval --rows 5 --seeds 42 43 --controllers fixed_time qits   # test
Output: sumo_imp/results/<R>x<C>/eval/summary.csv (one row per controller x density x seed; reruns
replace their rows) and runs/<controller>/<density>_seed<s>_{delay,epochs}.csv
"""
import argparse
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sumo_imp  # noqa: F401  (paths)
import numpy as np
import pandas as pd

from common import GridMap, demand_csv, density_lambdas  # noqa: E402
from generate_demand import make_trips, trips_to_frame  # noqa: E402
from sumo_imp.config import load_config, results_dir  # noqa: E402
from sumo_imp.controllers import QITS_VARIANTS, make_controller, theta_path  # noqa: E402
from sumo_imp.simulator import SumoSimulator  # noqa: E402


def pretrain_qits(cfg, scfg, variant, episodes, port):
    from qits.agents.qits_agent import QITSController
    q = cfg["qits"]
    gmap = GridMap(scfg)
    lam = density_lambdas(scfg)[q["pretrain_density"]]
    theta = None
    with tempfile.TemporaryDirectory() as tmp:
        for ep in range(episodes):
            seed = q["pretrain_seed"] + ep
            xml = make_trips(scfg, lam, seed, Path(tmp) / "trips.xml")
            ctl = QITSController(cfg, seed=seed, theta0=theta, **QITS_VARIANTS[variant])
            tic = time.perf_counter()
            res = SumoSimulator(cfg, scfg, trips_to_frame(gmap, xml), ctl, seed, port=port, gmap=gmap).run()
            theta = ctl.theta
            print(f"[{variant}] pretrain episode {ep + 1}/{episodes}: delay {res['summary']['avg_delay_s']:.1f} s, "
                  f"final dtheta {ctl.summary()['final_dtheta']:.3f} ({time.perf_counter() - tic:.0f} s)", flush=True)
    return theta


def run_one(job):
    name, density, seed, rows, backend, port = job
    cfg, scfg = load_config(rows)
    cfg["quantum"]["backend"] = backend
    ctl = make_controller(name, cfg, scfg, seed)
    demand = pd.read_csv(demand_csv(scfg, density, seed))
    tic = time.perf_counter()
    res = SumoSimulator(cfg, scfg, demand, ctl, seed, port=port).run()
    run_dir = results_dir(scfg) / "eval" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    trips = res["trips"].assign(minute=res["trips"].t_arrive // 60)
    trips.groupby("minute").delay.mean().reset_index().to_csv(run_dir / f"{density}_seed{seed}_delay.csv",
                                                               index=False)
    if getattr(ctl, "epoch_log", None):
        pd.DataFrame(ctl.epoch_log).to_csv(run_dir / f"{density}_seed{seed}_epochs.csv", index=False)
    extra = ctl.summary() if hasattr(ctl, "summary") else {}
    extra = {k: v for k, v in extra.items() if k not in res["summary"]}
    return {**res["summary"], "controller": name, "density": density, "seed": seed, "backend": backend,
            "wall_s": time.perf_counter() - tic, **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int)
    ap.add_argument("--controllers", nargs="+", default=["fixed_time", "rule_based", "dqn", "dqn_split", "qits"])
    ap.add_argument("--densities", nargs="+", default=["low", "medium", "high"])
    ap.add_argument("--seeds", type=int, nargs="+", help="default: the 30 dataset seeds")
    ap.add_argument("--backend", help="exact | shots | noisy (default: base config, noisy)")
    ap.add_argument("--pretrain-episodes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-port", type=int, default=31000, help="run i uses TraCI port base_port + i")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"                     # one core per run (inherited by the workers)

    cfg, scfg = load_config(args.rows)
    backend = args.backend or cfg["quantum"]["backend"]
    cfg["quantum"]["backend"] = backend
    out = results_dir(scfg) / "eval"
    out.mkdir(parents=True, exist_ok=True)

    for name in args.controllers:
        if name in QITS_VARIANTS:
            path = theta_path(scfg, backend, name)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                v = "qits" if name in ("qits_routing_only", "qits_signals_only") else name
                np.save(path, pretrain_qits(cfg, scfg, v, args.pretrain_episodes, args.base_port - 1))

    seeds = args.seeds or scfg["simulation"]["seeds"]
    rows = scfg["network"]["rows"]
    jobs = [(c, d, s, rows, backend) for c in args.controllers for d in args.densities for s in seeds]
    jobs = [(*j, args.base_port + i) for i, j in enumerate(jobs)]
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    results = []
    summary = out / "summary.csv"
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_one, jobs):
            results.append(r)
            print(f"{r['controller']:12s} {r['density']:6s} seed {r['seed']}: delay {r['avg_delay_s']:6.1f} s "
                  f"(all vehicles {r['avg_delay_all_s']:6.1f} s), completed {r['completion_rate']:.2f}, "
                  f"teleports {r['teleports']}, {r['wall_s']:.0f} s", flush=True)
            df = pd.DataFrame(results)
            if summary.exists():                  # newest result wins
                old = pd.read_csv(summary)
                key = ["controller", "density", "seed"]
                old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
                df = pd.concat([old, df], ignore_index=True)
            df.sort_values(["controller", "density", "seed"]).to_csv(summary, index=False)
    print(f"-> {summary}")


if __name__ == "__main__":
    main()
