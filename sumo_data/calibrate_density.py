"""Find λ_max, the largest arrival rate the network can serve, so that low / medium / high are
30 / 60 / 90 % of saturation (the paper's density levels). Rule (as in base_imp):

  run the fixed-time network for each λ and measure how fast vehicles-in-network (driving + waiting
  to enter) grows over the second half of the run; λ is stable if that growth, averaged over the
  seeds, is below max(slope_tol_abs, slope_tol_frac x λ); λ_max = the last λ before the first
  unstable λ. The absolute floor (0.1 veh/s, base_imp's rule) absorbs the random wiggle of
  vehicles-in-network, which on large grids is bigger than 1.5 % of a small λ.
  (base_imp required every seed to be stable; near saturation single-seed slopes are noisy, so the
  seed average is used here.)

    python calibrate_density.py                       # 25 x 25, λ list from config.yaml
    python calibrate_density.py --rows 5 --lams 2 3 4 5 6 7 8
Output: calibration_<R>x<C>.yaml, results/<R>x<C>/calibration.csv and .png
"""
import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ROOT, GridMap, density_lambdas, load_config, save_calibration, tag  # noqa: E402
from generate_demand import make_trips, trips_to_frame  # noqa: E402
from recorder import Recorder  # noqa: E402


def growth_slope(steps):
    half = steps.iloc[len(steps) // 2:]
    return float(np.polyfit(half.t, half.in_network, 1)[0])


def run_one(job):
    rows, cols, lam, seed, port = job
    cfg = load_config(rows, cols)
    gmap = GridMap(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        xml = make_trips(cfg, lam, seed, Path(tmp) / "trips.xml")
        res = Recorder(cfg, gmap).run(xml, trips_to_frame(gmap, xml), seed, record=False, port=port)
    return lam, seed, res["summary"], growth_slope(res["steps"]), res["steps"].in_network.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--lams", type=float, nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=40000, help="run i uses TraCI port base_port + i")
    ap.add_argument("--no-refine", action="store_true", help="skip the second, finer λ pass")
    args = ap.parse_args()
    cfg = load_config(args.rows, args.cols)
    cal = cfg["calibration"]
    lams, seeds = args.lams or cal["lams"], args.seeds or cal["seeds"]
    out = ROOT / "results" / tag(cfg)
    out.mkdir(parents=True, exist_ok=True)

    rows, curves = [], {}

    def sweep(lam_list):
        jobs = [(cfg["network"]["rows"], cfg["network"]["cols"], lam, s, args.base_port + len(rows) + i)
                for i, (lam, s) in enumerate((lam, s) for lam in lam_list for s in seeds)]   # own TraCI port
        with ProcessPoolExecutor(args.workers) as ex:
            for lam, seed, s, slope, curve in ex.map(run_one, jobs):
                rows.append({"lambda": lam, "seed": seed, "slope_veh_per_s": slope, **s})
                if seed == seeds[0]:
                    curves[lam] = curve
                print(f"lam={lam:6.2f} seed {seed}: slope {slope:+.3f}  delay {s['avg_delay_s']:6.1f} s  "
                      f"completed {s['completion_rate']:.2f}  teleports {s['teleports']}", flush=True)
        by_lam = pd.DataFrame(rows).groupby("lambda").agg(slope=("slope_veh_per_s", "mean"),
                                                          avg_delay_s=("avg_delay_s", "mean"))
        by_lam["stable"] = by_lam.slope < np.maximum(cal["slope_tol_abs"], cal["slope_tol_frac"] * by_lam.index)
        unstable = by_lam.index[~by_lam.stable]
        first_bad = unstable.min() if len(unstable) else np.inf
        return by_lam, first_bad, by_lam.index[by_lam.index < first_bad]

    by_lam, first_bad, ok = sweep(lams)
    if not len(ok):
        raise RuntimeError("even the smallest lambda is unstable; check results/<size>/calibration.csv "
                           "(teleports, completion) before trying smaller --lams")
    # all stable: double upwards until the network saturates (works for any grid size)
    while first_bad == np.inf and max(lams) < 500:
        lams = [max(lams) * 2, max(lams) * 4]
        print(f"\nall stable, trying higher: {lams}\n", flush=True)
        by_lam, first_bad, ok = sweep(lams)
    # refine between the last stable and the first unstable λ until the gap is < 5 % (at most 3 passes)
    for _ in range(0 if args.no_refine else 3):
        if not len(ok) or first_bad == np.inf or first_bad - ok.max() <= 0.05 * ok.max():
            break
        fine = [round(float(x), 3) for x in np.linspace(ok.max(), first_bad, 6)[1:-1]]
        print(f"\nrefining between {ok.max():g} and {first_bad:g}: {fine}\n", flush=True)
        by_lam, first_bad, ok = sweep(fine)
    pd.DataFrame(rows).to_csv(out / "calibration.csv", index=False)
    print(by_lam.round(3).to_string())
    if not len(ok):
        raise RuntimeError("even the smallest lambda is unstable; try smaller --lams")
    if first_bad == np.inf:
        print("WARNING: every lambda was stable; lambda_max may be higher, try larger --lams")
    lam_max = float(ok.max())
    save_calibration(cfg, lam_max)
    cfg["demand"]["lambda_max"] = lam_max
    print(f"\nlambda_max = {lam_max} veh/s  ->  density levels (veh/s): {density_lambdas(cfg)}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lam, c in curves.items():
        ax[0].plot(c, label=f"lam={lam:g}")
    ax[0].set(xlabel="time (s)", ylabel="vehicles in network", title=f"Stability sweep ({tag(cfg)}, fixed-time)")
    ax[0].legend(fontsize=6, ncol=3)
    ax[1].plot(by_lam.index, by_lam.avg_delay_s, "o-")
    ax[1].axvline(lam_max, ls="--", c="k")
    ax[1].set(xlabel="lambda (veh/s)", ylabel="average delay of completed trips (s)", title=f"lambda_max = {lam_max:g}")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=120)


if __name__ == "__main__":
    main()
