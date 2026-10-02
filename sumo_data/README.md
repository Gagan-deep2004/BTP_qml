# SUMO dataset for Q-ITS

This folder generates the project's traffic data with **SUMO** (Eclipse SUMO, a microscopic traffic
simulator): the road network, the vehicle trips, and the consumer-device data (the 16 features per
intersection that Q-ITS uses). It does not change `base_imp/` or `novelty_imp/`; it only reuses the
base code's intersection numbering, so the data can be fed to those controllers unchanged.

## What is generated

| Item | How | Paper |
|---|---|---|
| Road network | `netgenerate`: 25 × 25 grid (625 intersections), 100 m between junctions, 2 lanes, 50 km/h | grid, 100 m roads |
| Traffic lights | fixed-time plan at every junction: 60 s cycle, 50/50 split, 3 s amber (base: `fixed_time`) | – |
| Vehicle trips | `randomTrips.py`: Poisson arrivals (binomial n = 1000 per second), random origin/destination junctions ≥ 2 blocks apart | Poisson arrivals, random OD |
| Density levels | λ = 30 / 60 / 90 % of the network's saturation rate λ_max, found by `calibrate_density.py` | 30 / 60 / 90 % saturation |
| Vehicle movement | SUMO car-following, lane changing, permissive left turns, rerouting every 60 s | – |
| Device data | 60 % of vehicles report road, position (± 3 m GPS noise) and speed (± 0.5 m/s), using SUMO's real speeds | consumer devices |
| 16 features | per incoming road (N, E, S, W): density, speed, queue, inflow; same formulas as `base_imp` | "density, speed, prediction horizon" |
| Duration / seeds | 1800 s, step 1 s, seeds 42–71 (30 runs) | 1800 s, 30 runs, seed 42 |

The paper's λ = 3–7 veh/s only gives 30–90 % saturation on a 25-intersection network. This dataset keeps
the paper's **saturation levels** and calibrates λ for the network size used.

## Running (on the server)

```bash
# once, from the project root (Python 3.12):
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # includes eclipse-sumo, traci, sumolib

cd sumo_data
WORKERS=16 bash run_all.sh                 # whole pipeline, 25 x 25
```

Use `tmux` or `nohup` so it keeps running after you log out:
`nohup bash run_all.sh > run_all.log 2>&1 &`

Steps, if you want to run them one by one (add `--rows 5` for a quick 5 × 5 test):

| Step | Command | Output |
|---|---|---|
| 1 | `python build_network.py` | `network/grid_25x25.net.xml` |
| 2 | `python calibrate_density.py --workers 16` | `calibration_25x25.yaml` (λ_max), `results/25x25/calibration.csv/.png` |
| 3 | `python generate_demand.py` | `data/25x25/demand/<density>/seed_<s>.trips.xml` and `.csv` |
| 4 | `python generate_device_data.py --workers 16` | `data/25x25/devices/<density>/seed_<s>.npz`, probes, trip results; `results/25x25/device_runs.csv` |

`--seeds` and `--densities` limit a step to some runs (e.g. `--seeds 42 43` for a quick check).

**Time and space (25 × 25, measured on a laptop):** one recorded 1800 s run takes about 5 min at a high
density, and its `.npz` is about 60 MB. The full pipeline is about 33 calibration runs + 90 dataset runs:
roughly 1–1.5 h with 16 workers or 2–3 h with 8. The dataset takes about 4–5 GB. Each worker needs
about 1 GB of RAM.

## Output files

`data/<R>x<C>/demand/<density>/seed_<s>.csv`: `veh_id, t_depart, origin, dest` (intersection ids
`row × cols + col`, row 0 = north), the same format as `base_imp/data/demand/`.

`data/<R>x<C>/devices/<density>/seed_<s>.npz` (load with `numpy.load`):

| Key | Shape | Meaning |
|---|---|---|
| `x` | (1800, nodes, 16) | device features, the Q-ITS input. Columns: approach N, E, S, W × (density, speed, queue, inflow), each in [0, 1] |
| `x_true` | (1800, nodes, 16) | the same features from exact vehicle data (to measure device error) |
| `delay` | (1800, nodes, 4) | estimated delay δ_ij (s) of the road towards N, E, S, W (0 = no road) |
| `steps` | (1800, 5) | t, vehicles driving, vehicles waiting to enter, completed trips, teleports |

`probes_seed_42.parquet`: raw device reports (t, veh_id, link, pos_m, speed_mps), seed 42 only.
`seed_<s>_trips.csv.gz`: every completed trip from SUMO (delay = time loss + wait to enter).
`results/<R>x<C>/device_runs.csv`: one row per run: average delay, completion, teleports, device-feature error.

## Differences from `base_imp` data (mention these in the report)

- SUMO roads are 83 m long, not 100 m: the junction itself takes ~17 m. Road storage uses the real length.
- Amber (3 s) instead of all-red lost time; left turns yield to oncoming traffic.
- SUMO saturates at a lower rate than the base simulator: on 5 × 5, λ_max = 3.4 veh/s in SUMO vs 6.5 in
  `base_imp`, mainly because left-turners waiting for gaps block a lane and junctions can jam.
- Vehicles pick the fastest route and re-plan every 60 s (navigation apps), instead of a random
  shortest path.
- A vehicle stuck for 300 s is teleported (SUMO default); the count is logged per run.
- Origins are drawn through a random road, so corner junctions start about half as many trips as inner ones.
