"""Consumer-device dataset + validation (guide Section 3.1).

1. For every density x seed, run the reference (rule-based) controller and record what the
   intersections observe from consumer devices each second:
       data/devices/{density}/seed_{seed}.npz
           x       (T, n_nodes, 16)  device-estimated features  -> Eq. 2 input
           x_true  (T, n_nodes, 16)  ground-truth features (validation only)
           delay   (T, n_nodes, 4)   estimated δ_ij (s) per outgoing direction
   Note: features depend on the controller in the loop; each experiment re-records them
   live. This file is the reference dataset for inspection / normalisation checks.
2. Raw probe log (t, veh_id, link, pos_m, speed_mps) for seed 42 of each density:
       data/devices/{density}/probes_seed_42.parquet
3. Estimation error vs device penetration rate (medium density, seed 42):
       results/devices/penetration_sweep.csv / .png, results/devices/queue_estimate.png

    python -m experiments.generate_device_data [--seeds 30]
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.generate_dataset import load_demand
from qits.agents.rule_based import RuleBasedController
from qits.config import ROOT, density_lambdas, load_config
from qits.traffic.devices import FEATURE_NAMES
from qits.traffic.grid import W
from qits.traffic.simulator import TrafficSimulator

OUT_DATA = ROOT / "data" / "devices"
OUT_RES = ROOT / "results" / "devices"
MAE_COLS = ["mae_" + n for n in FEATURE_NAMES]


def record(cfg, density, seed):
    sim = TrafficSimulator(cfg, load_demand(density, seed), RuleBasedController.from_config(cfg), seed)
    xs, xts, ds = [], [], []
    for _ in range(sim.duration):
        sim.step()
        xs.append(sim.devices.x().astype(np.float32))
        xts.append(sim.devices.true_features.reshape(sim.grid.n_nodes, -1).astype(np.float32))
        ds.append(sim.devices.link_delay(sim.ff_steps).astype(np.float32))
    return sim, np.stack(xs), np.stack(xts), np.stack(ds)


def record_job(args):
    density, seed = args
    cfg = load_config()
    cfg["devices"]["save_probes"] = seed == cfg["simulation"]["base_seed"]
    sim, x, xt, d = record(cfg, density, seed)
    out = OUT_DATA / density
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"seed_{seed}.npz", x=x, x_true=xt, delay=d)
    if sim.devices.probes:
        pd.DataFrame(sim.devices.probes, columns=["t", "veh_id", "link", "pos_m", "speed_mps"]) \
          .to_parquet(out / f"probes_seed_{seed}.parquet", index=False)
    mae = np.abs(x - xt).reshape(len(x), -1, 4)[:, (sim.grid.in_link >= 0).ravel()].mean(axis=(0, 1))
    return {"density": density, "seed": seed, **dict(zip(MAE_COLS, mae))}


def penetration_job(pen):
    cfg = load_config()
    cfg["devices"]["penetration"] = pen
    density, seed = "medium", cfg["simulation"]["base_seed"]
    sim, x, xt, _ = record(cfg, density, seed)
    valid = (sim.grid.in_link >= 0).ravel()
    mae = np.abs(x - xt).reshape(len(x), -1, 4)[:, valid].mean(axis=(0, 1))
    return {"penetration": pen, **dict(zip(MAE_COLS, mae))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    base = cfg["simulation"]["base_seed"]
    jobs = [(d, s) for d in density_lambdas(cfg) for s in range(base, base + args.seeds)]
    OUT_RES.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(record_job, jobs))
        pens = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        sweep = pd.DataFrame(list(ex.map(penetration_job, pens)))

    err = pd.DataFrame(rows)
    err.to_csv(OUT_RES / "feature_error_by_run.csv", index=False)
    print("Mean absolute error of device features vs ground truth (normalised units):")
    print(err.groupby("density")[MAE_COLS].mean().round(4))

    sweep.to_csv(OUT_RES / "penetration_sweep.csv", index=False)
    print("\nError vs penetration (medium density):")
    print(sweep.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in FEATURE_NAMES:
        ax.plot(sweep.penetration, sweep["mae_" + name], "o-", label=name)
    ax.axvline(cfg["devices"]["penetration"], ls="--", c="gray", label="used")
    ax.set(xlabel="device penetration rate", ylabel="MAE (normalised feature)",
           title="Consumer-device estimation error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_RES / "penetration_sweep.png", dpi=150)

    # estimated vs true queue on the West approach of the centre node (medium, seed 42)
    dat = np.load(OUT_DATA / "medium" / f"seed_{base}.npz")
    centre = (cfg["network"]["rows"] // 2) * cfg["network"]["cols"] + cfg["network"]["cols"] // 2
    k = W * 4 + FEATURE_NAMES.index("queue")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(dat["x_true"][:600, centre, k], label="true queue", lw=1.5)
    ax.plot(dat["x"][:600, centre, k], label="device estimate", lw=1, alpha=0.8)
    ax.set(xlabel="time (s)", ylabel="queue / storage",
           title=f"Node {centre}, West approach: consumer-device queue estimate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_RES / "queue_estimate.png", dpi=150)
    print(f"\nsaved dataset to {OUT_DATA} and plots to {OUT_RES}")


if __name__ == "__main__":
    main()
