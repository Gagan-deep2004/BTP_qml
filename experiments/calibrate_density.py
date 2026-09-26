"""Calibrate the paper's 30 / 60 / 90 % saturation levels (guide Section 3.3).

Runs the rule-based controller for a sweep of arrival rates λ. The network is
"stable" at λ if the number of vehicles in the network stops growing (slope of
in_network over the second half of the run below a threshold). λ_max is the
largest λ that is stable for every calibration seed; densities are then
fraction * λ_max. Writes λ_max into configs/calibration.yaml.

    python -m experiments.calibrate_density
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qits.agents.rule_based import RuleBasedController
from qits.config import ROOT, density_lambdas, load_config, save_calibration
from qits.traffic.demand import generate_demand
from qits.traffic.grid import Grid
from qits.traffic.simulator import TrafficSimulator


def growth_slope(steps):
    half = steps.iloc[len(steps) // 2:]
    return float(np.polyfit(half.t, half.in_network, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", type=float, nargs="+", default=list(np.arange(2.0, 10.01, 0.5)))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--slope-tol", type=float, default=0.1,
                    help="max growth of vehicles-in-network (veh/s) considered stable")
    args = ap.parse_args()

    cfg = load_config(use_calibration=False)
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    out = ROOT / "results" / "calibration"
    out.mkdir(parents=True, exist_ok=True)

    rows, curves = [], {}
    for lam in args.lams:
        for seed in args.seeds:
            demand = generate_demand(g, lam, cfg["simulation"]["duration_s"], seed,
                                     cfg["demand"]["min_hops"])
            res = TrafficSimulator(cfg, demand, RuleBasedController.from_config(cfg), seed).run()
            s = res["summary"]
            slope = growth_slope(res["steps"])
            rows.append({"lambda": lam, "seed": seed, "slope_veh_per_s": slope,
                         "stable": slope < args.slope_tol,
                         **{k: s[k] for k in ("completion_rate", "avg_delay_s", "throughput_vps",
                                              "congestion_index", "final_in_network")}})
            if seed == args.seeds[0]:
                curves[lam] = res["steps"].in_network.to_numpy()
        r = pd.DataFrame(rows[-len(args.seeds):])
        print(f"λ={lam:5.2f}  stable={r.stable.sum()}/{len(r)}  slope={r.slope_veh_per_s.mean():+.3f}  "
              f"delay={r.avg_delay_s.mean():6.1f}s  throughput={r.throughput_vps.mean():.2f}  "
              f"CI={r.congestion_index.mean():.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration.csv", index=False)

    by_lam = df.groupby("lambda").agg(all_stable=("stable", "all"),
                                      avg_delay_s=("avg_delay_s", "mean")).reset_index()
    # λ_max = last λ before the first λ at which any seed becomes unstable
    unstable = by_lam[~by_lam.all_stable]
    first_bad = unstable["lambda"].min() if len(unstable) else np.inf
    ok = by_lam[by_lam["lambda"] < first_bad]
    lam_max = float(ok["lambda"].max()) if len(ok) else float(by_lam["lambda"].min())
    save_calibration({"demand": {"lambda_max": lam_max}})
    cfg["demand"]["lambda_max"] = lam_max
    print(f"\nλ_max = {lam_max}  ->  density levels (veh/s): {density_lambdas(cfg)}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lam, c in curves.items():
        ax[0].plot(c, label=f"λ={lam:g}")
    ax[0].set(xlabel="time (s)", ylabel="vehicles in network", title="Stability sweep (rule-based)")
    ax[0].legend(fontsize=6, ncol=3)
    ax[1].plot(by_lam["lambda"], by_lam.avg_delay_s, "o-")
    ax[1].axvline(lam_max, ls="--", c="gray", label="λ_max")
    ax[1].set(xlabel="λ (veh/s)", ylabel="avg delay of completed trips (s)", title="Delay vs λ")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=150)
    print(f"saved {out / 'calibration.csv'} and calibration.png")


if __name__ == "__main__":
    main()
