"""Generate the synthetic demand dataset (the paper's data): Poisson arrivals with random
OD pairs for every density level and seed.

    python -m experiments.generate_dataset
Output: data/demand/{density}/seed_{seed}.csv  +  data/demand/manifest.csv
"""
import pandas as pd

from qits.config import ROOT, density_lambdas, load_config
from qits.traffic.demand import generate_demand
from qits.traffic.grid import Grid


def demand_path(density, seed):
    return ROOT / "data" / "demand" / density / f"seed_{seed}.csv"


def load_demand(density, seed):
    return pd.read_csv(demand_path(density, seed))


def main():
    cfg = load_config()
    sim = cfg["simulation"]
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    seeds = range(sim["base_seed"], sim["base_seed"] + sim["n_runs"])

    manifest = []
    for density, lam in density_lambdas(cfg).items():
        for seed in seeds:
            df = generate_demand(g, lam, sim["duration_s"], seed, cfg["demand"]["min_hops"])
            path = demand_path(density, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            manifest.append({"density": density, "lambda": lam, "seed": seed,
                             "n_vehicles": len(df), "file": str(path.relative_to(ROOT))})
        print(f"{density:6s} λ={lam:.2f} veh/s: {len(seeds)} seeds written")

    out = ROOT / "data" / "demand" / "manifest.csv"
    pd.DataFrame(manifest).to_csv(out, index=False)
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
