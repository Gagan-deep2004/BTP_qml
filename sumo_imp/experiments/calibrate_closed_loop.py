"""Density calibration for the closed-loop evaluation, then the test trips at those levels.

The sumo_data calibration measured saturation with SUMO's own (navigation) routing. The evaluation
routes vehicles like the base paper (random shortest path, ∝ route_p), which spreads traffic more
evenly, so its network saturates at a different λ. This script measures λ_max in the evaluation's
own setup (fixed-time lights + base routing, through SumoSimulator) with the same rule as
sumo_data/calibrate_density.py (vehicles-in-network growth over the second half, seed average,
stable if < max(0.1, 1.5 % λ) veh/s; doubling, then up to 3 finer passes), and then writes the 30
test trip files per density at 30 / 60 / 90 % of that λ_max (same randomTrips seeds 42-71).

    python -m sumo_imp.experiments.calibrate_closed_loop --workers 90
    python -m sumo_imp.experiments.calibrate_closed_loop --rows 5 --workers 8 --trip-seeds 42 43   # test
Output: sumo_imp/results/<R>x<C>/calibration_closed_loop.{yaml,csv,png}
        sumo_imp/data/<R>x<C>/closed_loop/<density>/seed_<s>.csv
"""
import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sumo_imp  # noqa: F401  (paths)
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from calibrate_density import growth_slope  # noqa: E402  (sumo_data)
from common import GridMap, tag  # noqa: E402
from generate_demand import make_trips, trips_to_frame  # noqa: E402
from qits.agents.base import FixedTimeController  # noqa: E402
from sumo_imp.config import Demand, load_config, results_dir  # noqa: E402
from sumo_imp.simulator import SumoSimulator  # noqa: E402


def run_one(job):
    rows, lam, seed, port = job
    cfg, scfg = load_config(rows)
    gmap = GridMap(scfg)
    with tempfile.TemporaryDirectory() as tmp:
        demand = trips_to_frame(gmap, make_trips(scfg, lam, seed, Path(tmp) / "trips.xml"))
    res = SumoSimulator(cfg, scfg, demand, FixedTimeController(), seed, port=port, gmap=gmap).run()
    return lam, seed, res["summary"], growth_slope(res["steps"]), res["steps"].in_network.to_numpy()


def make_one(job):
    rows, density, lam, seed = job
    _, scfg = load_config(rows)
    path = Demand(scfg, "closed_loop").csv(density, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        df = trips_to_frame(GridMap(scfg), make_trips(scfg, lam, seed, Path(tmp) / "trips.xml"))
    df.to_csv(path, index=False)
    return density, seed, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int)
    ap.add_argument("--lams", type=float, nargs="+", help="first λ values (default: sumo_data config)")
    ap.add_argument("--seeds", type=int, nargs="+", help="calibration seeds (default: sumo_data config)")
    ap.add_argument("--trip-seeds", type=int, nargs="+", help="test trip seeds (default: the 30 dataset seeds)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-port", type=int, default=35000)
    ap.add_argument("--skip-calibration", action="store_true", help="only (re)write the trips")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    _, scfg = load_config(args.rows)
    cal = scfg["calibration"]
    rows_n = scfg["network"]["rows"]
    out = results_dir(scfg)
    out.mkdir(parents=True, exist_ok=True)
    dem = Demand(scfg, "closed_loop")

    if not args.skip_calibration:
        lams, seeds = args.lams or cal["lams"], args.seeds or cal["seeds"]
        rows, curves = [], {}

        def sweep(lam_list):
            jobs = [(rows_n, lam, s, args.base_port + len(rows) + i)
                    for i, (lam, s) in enumerate((lam, s) for lam in lam_list for s in seeds)]
            with ProcessPoolExecutor(args.workers) as ex:
                for lam, seed, s, slope, curve in ex.map(run_one, jobs):
                    rows.append({"lambda": lam, "seed": seed, "slope_veh_per_s": slope, **s})
                    if seed == seeds[0]:
                        curves[lam] = curve
                    print(f"lam={lam:6.2f} seed {seed}: slope {slope:+.3f}  delay {s['avg_delay_s']:6.1f} s  "
                          f"all {s['avg_delay_all_s']:6.1f} s  completed {s['completion_rate']:.2f}  "
                          f"teleports {s['teleports']}", flush=True)
            by = pd.DataFrame(rows).groupby("lambda").agg(slope=("slope_veh_per_s", "mean"),
                                                          avg_delay_s=("avg_delay_s", "mean"),
                                                          avg_delay_all_s=("avg_delay_all_s", "mean"))
            by["stable"] = by.slope < np.maximum(cal["slope_tol_abs"], cal["slope_tol_frac"] * by.index)
            bad = by.index[~by.stable]
            first_bad = bad.min() if len(bad) else np.inf
            return by, first_bad, by.index[by.index < first_bad]

        by, first_bad, ok = sweep(lams)
        if not len(ok):
            raise RuntimeError("even the smallest lambda is unstable; check the calibration csv")
        while first_bad == np.inf and max(lams) < 500:
            lams = [max(lams) * 2, max(lams) * 4]
            print(f"\nall stable, trying higher: {lams}\n", flush=True)
            by, first_bad, ok = sweep(lams)
        for _ in range(3):
            if first_bad == np.inf or first_bad - ok.max() <= 0.05 * ok.max():
                break
            fine = [round(float(x), 3) for x in np.linspace(ok.max(), first_bad, 6)[1:-1]]
            print(f"\nrefining between {ok.max():g} and {first_bad:g}: {fine}\n", flush=True)
            by, first_bad, ok = sweep(fine)
        pd.DataFrame(rows).to_csv(out / "calibration_closed_loop.csv", index=False)
        print(by.round(3).to_string())
        lam_max = float(ok.max())
        dem.calibration_file.write_text(
            "# Written by sumo_imp/experiments/calibrate_closed_loop.py (fixed-time + base routing).\n"
            + yaml.safe_dump({"lambda_max": lam_max}), encoding="utf-8")

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for lam, c in sorted(curves.items()):
            ax[0].plot(c, label=f"lam={lam:g}")
        ax[0].set(xlabel="time (s)", ylabel="vehicles in network",
                  title=f"Closed-loop stability sweep ({tag(scfg)}, fixed-time, base routing)")
        ax[0].legend(fontsize=6, ncol=3)
        ax[1].plot(by.index, by.avg_delay_all_s, "o-")
        ax[1].axvline(lam_max, ls="--", c="k")
        ax[1].set(xlabel="lambda (veh/s)", ylabel="average delay, all vehicles (s)", title=f"lambda_max = {lam_max:g}")
        fig.tight_layout()
        fig.savefig(out / "calibration_closed_loop.png", dpi=120)

    lams = dem.lambdas()
    print(f"\nlambda_max (closed loop) -> density levels (veh/s): {lams}", flush=True)
    trip_seeds = args.trip_seeds or scfg["simulation"]["seeds"]
    jobs = [(rows_n, d, lam, s) for d, lam in lams.items() for s in trip_seeds]
    with ProcessPoolExecutor(args.workers) as ex:
        made = list(ex.map(make_one, jobs))
    for d in lams:
        n = [k for dd, _, k in made if dd == d]
        print(f"{d:6s}: {len(n)} trip files, {np.mean(n):.0f} vehicles on average", flush=True)
    print(f"-> {dem.csv('low', trip_seeds[0]).parent.parent}")


if __name__ == "__main__":
    main()
