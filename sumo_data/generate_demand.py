"""Vehicle trips with SUMO's randomTrips.py (the paper's demand: Poisson arrivals, random OD pairs).

Arrivals: binomial(n = 1000) per second with mean λ, i.e. Poisson(λ) for all practical purposes.
Origin / destination: random junctions (--junction-taz), at least `min_hops` blocks apart.
(randomTrips picks a junction through a random road leaving it, so a junction's chance is
proportional to its number of roads: corner junctions start ~half as many trips as inner ones.)

    python generate_demand.py                  # all densities x all seeds
    python generate_demand.py --rows 5 --seeds 42 43
Output per density / seed (data/<R>x<C>/demand/<density>/):
    seed_<s>.trips.xml   SUMO input
    seed_<s>.csv         veh_id, t_depart, origin, dest (base_imp node ids, same format as base data)
"""
import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET

import pandas as pd

from common import GridMap, demand_csv, density_lambdas, load_config, net_file, tool, trips_file

MIN_DISTANCE_M = 140   # > 100 m (adjacent junctions) and < 141.4 m (diagonal) -> Manhattan distance >= 2


def make_trips(cfg, lam, seed, out_xml):
    """Write a randomTrips trip file with mean arrival rate `lam` veh/s."""
    if cfg["demand"]["min_hops"] != 2:
        raise ValueError("MIN_DISTANCE_M encodes min_hops = 2")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, tool("randomTrips.py"), "-n", str(net_file(cfg)), "-o", str(out_xml),
           "-b", "0", "-e", str(cfg["simulation"]["duration_s"]), "-p", repr(1.0 / float(lam)),
           "--binomial", "1000", "-s", str(seed), "--junction-taz", "--min-distance", str(MIN_DISTANCE_M),
           "--no-validate", "--prefix", "v",
           "--trip-attributes", 'departLane="best" departSpeed="max"']
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_xml


def trips_to_frame(gmap, xml_path):
    rows = [(int(t.get("id")[1:]), int(float(t.get("depart"))),
             gmap.node_of_junction[t.get("fromJunction")], gmap.node_of_junction[t.get("toJunction")])
            for t in ET.parse(xml_path).getroot().iter("trip")]
    return pd.DataFrame(rows, columns=["veh_id", "t_depart", "origin", "dest"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--densities", nargs="+")
    args = ap.parse_args()
    cfg = load_config(args.rows, args.cols)
    gmap = GridMap(cfg)
    lams = density_lambdas(cfg)
    manifest = []
    for density in args.densities or list(lams):
        for seed in args.seeds or cfg["simulation"]["seeds"]:
            xml = make_trips(cfg, lams[density], seed, trips_file(cfg, density, seed))
            df = trips_to_frame(gmap, xml)
            hops = [gmap.grid.manhattan(o, d) for o, d in zip(df.origin, df.dest)]
            assert min(hops) >= cfg["demand"]["min_hops"]
            df.to_csv(demand_csv(cfg, density, seed), index=False)
            manifest.append({"density": density, "seed": seed, "lambda": lams[density],
                             "n_vehicles": len(df), "rate_vps": len(df) / cfg["simulation"]["duration_s"]})
            print(f"{density:6s} seed {seed}: {len(df)} vehicles ({manifest[-1]['rate_vps']:.2f} veh/s)",
                  flush=True)
    out = demand_csv(cfg, "x", 0).parent.parent / "manifest.csv"
    pd.DataFrame(manifest).to_csv(out, index=False)


if __name__ == "__main__":
    main()
