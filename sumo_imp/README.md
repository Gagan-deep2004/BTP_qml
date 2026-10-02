# Base paper on SUMO

Runs the base-paper controllers (`base_imp/`, unchanged) **closed-loop in SUMO**: SUMO moves the cars,
and every second the controllers read the consumer-device data and set the traffic lights and routes.
It uses the network, the density calibration and the 90 test trip files from `sumo_data/`.

| Controller | What it is |
|---|---|
| `fixed_time` | 60 s cycle, 50/50 split |
| `rule_based` | threshold rule on last cycle's demand (paper Sec. V) |
| `dqn` | DQN choosing the green phase every 5 s (trained on SUMO) |
| `dqn_split` | DQN choosing the split per cycle (same action space as Q-ITS) |
| `qits` | Q-ITS, Algorithm 1: VQC routing + signal split, QKD/OTP, consensus; pre-trained on SUMO, learning online |
| `qits_noqopt`, `qits_noqkd`, `qits_nocons` | paper Table IV ablations |

## How SUMO is connected (`simulator.py`)

`SumoSimulator` offers the same interface as the base simulator, so the controllers need no change:

- **Signals:** the base `Signals` class (split or phase mode) decides green/red; each second its state is
  written to SUMO's traffic lights. The base model's 3 s all-red lost time is shown as 3 s amber;
  left turns yield to oncoming traffic.
- **Routing:** as in the base code, each vehicle takes a shortest path, choosing at every junction
  among the directions that bring it closer to its destination with probability ∝ `route_p`
  (uniform for the baselines, the VQC's p_ij for Q-ITS).
- **Devices:** the 16 features and δ_ij come from SUMO's real speeds, with the formulas of the base
  code and of `sumo_data/` (60 % penetration, speed noise).
- **Delay:** SUMO time loss + time waiting to enter the network (as in `sumo_data/`).
  `avg_delay_all_s` also counts vehicles still driving at the end.

## Running on the server (25 × 25)

The 25 × 25 network and trips must exist (`sumo_data/`, `bash run_all.sh`). From the project root:

```bash
# 1. train the two DQNs and evaluate the other controllers, all at once (separate ports)
nohup python -m sumo_imp.experiments.train_dqn                          > dqn.log 2>&1 &
nohup python -m sumo_imp.experiments.train_dqn --mode split --port 33100 > dqn_split.log 2>&1 &
nohup python -m sumo_imp.experiments.run_eval --controllers fixed_time rule_based qits --workers 90 > eval_a.log 2>&1 &

# 2. when both DQN logs end with "saved ...":
nohup python -m sumo_imp.experiments.run_eval --controllers dqn dqn_split --workers 90 --base-port 32000 > eval_b.log 2>&1 &

# 3. tables
python -m sumo_imp.experiments.report
```

Q-ITS is pre-trained first (2 episodes of 1800 s, ~1 h on 25 × 25), then its 90 test runs start.
DQN training (500 episodes of 600 s) is the longest part; `--resume` continues after an interruption.
Quick check on 5 × 5: add `--rows 5` (and e.g. `--seeds 42 43`, `--episodes 4`).

Outputs, in `sumo_imp/results/<R>x<C>/`:

- `eval/summary.csv`: one row per run
- `eval/report.md`: delay tables, Q-ITS vs baselines
- `eval/runs/`: delay per minute; Q-ITS epoch logs
- `dqn/`: trained models and training curves
- `qits/`: pre-trained θ

Tests: `python -m pytest sumo_imp/tests -q` (5 × 5, ~30 s).
