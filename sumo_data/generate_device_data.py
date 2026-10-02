"""Run SUMO on every trip file and save the consumer-device dataset.

    python generate_device_data.py                    # all densities x all seeds (needs generate_demand.py)
    python generate_device_data.py --rows 5 --seeds 42 --workers 2
Output per density / seed, in data/<R>x<C>/devices/<density>/:
    seed_<s>.npz      x (T, nodes, 16)       device features, the Q-ITS input (same layout as base_imp:
                                             approach N, E, S, W x density, speed, queue, inflow)
                      x_true (T, nodes, 16)  the same features from exact vehicle data (for validation)
                      delay (T, nodes, 4)    δ_ij estimate towards N, E, S, W
                      steps (T, 5)           t, running, waiting_to_insert, completed, teleports
    probes_seed_<s>.parquet (probe seeds only)   t, veh_id, link, pos_m, speed_mps
    seed_<s>_trips.csv.gz                    per-vehicle results from SUMO (delay, travel time)
results/<R>x<C>/device_runs.csv: one summary row per run (delays, completion, feature errors).
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import ROOT, GridMap, demand_csv, devices_file, load_config, tag, trips_file
from generate_demand import trips_to_frame
from recorder import Recorder

FEATURES = ["density", "speed", "queue", "inflow"]


def run_one(job):
    rows, cols, density, seed, port = job
    cfg = load_config(rows, cols)
    probes = seed in cfg["simulation"]["probe_seeds"]
    tic = time.perf_counter()
    gmap = GridMap(cfg)
    xml = trips_file(cfg, density, seed)
    demand = trips_to_frame(gmap, xml)                 # the vehicles SUMO will actually load
    csv = pd.read_csv(demand_csv(cfg, density, seed))
    if len(csv) != len(demand) or not csv.veh_id.equals(demand.veh_id):
        print(f"WARNING {density} seed {seed}: {xml.name} has {len(demand)} trips but the csv has {len(csv)}; "
              f"using the trip file", flush=True)
    res = Recorder(cfg, gmap).run(xml, demand, seed, record=True, probes=probes, port=port)
    path = devices_file(cfg, density, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=res["x"], x_true=res["x_true"], delay=res["delay"],
                        steps=res["steps"][["t", "running", "waiting_to_insert", "completed", "teleports"]]
                        .to_numpy(np.int32))
    res["trips"].to_csv(path.with_name(f"seed_{seed}_trips.csv.gz"), index=False)
    if probes:
        res["probes"].to_parquet(path.with_name(f"probes_seed_{seed}.parquet"), index=False)
    row = {"density": density, "seed": seed, **res["summary"],
           **{f"mae_{f}": float(e) for f, e in zip(FEATURES, res["feature_mae"])},
           "wall_s": time.perf_counter() - tic, "file_mb": path.stat().st_size / 1e6}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--densities", nargs="+", default=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=30000, help="run i uses TraCI port base_port + i")
    args = ap.parse_args()
    cfg = load_config(args.rows, args.cols)
    seeds = args.seeds or cfg["simulation"]["seeds"]
    jobs = [(cfg["network"]["rows"], cfg["network"]["cols"], d, s) for d in args.densities for s in seeds]
    jobs = [(*job, args.base_port + i) for i, job in enumerate(jobs)]      # one TraCI port per run
    out = ROOT / "results" / tag(cfg)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "device_runs.csv"
    done = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=["density", "seed"])

    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_one, jobs):
            rows.append(r)
            print(f"{r['density']:6s} seed {r['seed']}: delay {r['avg_delay_s']:6.1f} s, completed "
                  f"{r['completion_rate']:.2f}, teleports {r['teleports']}, feature MAE "
                  f"{r['mae_density']:.3f}/{r['mae_speed']:.3f}/{r['mae_queue']:.3f}/{r['mae_inflow']:.3f}, "
                  f"{r['wall_s']:.0f} s, {r['file_mb']:.1f} MB", flush=True)
    new = pd.DataFrame(rows)
    keep = done[~done.set_index(["density", "seed"]).index.isin(new.set_index(["density", "seed"]).index)]
    pd.concat([keep, new]).sort_values(["density", "seed"]).to_csv(log, index=False)


if __name__ == "__main__":
    main()
