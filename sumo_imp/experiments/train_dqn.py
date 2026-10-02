"""Train the DQN baseline on SUMO traffic (same procedure as base_imp/experiments/train_dqn.py).

Episodes of cfg.dqn.episode_s (600 s) on fresh SUMO trips (randomTrips, seeds 2000+, outside the test
seeds 42-71); density drawn at random from low/medium/high each episode; ε decays linearly from 1.0 to
0.05 over the first 60 % of the episodes; one network shared by all intersections.

    python -m sumo_imp.experiments.train_dqn                   # 25 x 25, 500 episodes, phase mode
    python -m sumo_imp.experiments.train_dqn --mode split
    python -m sumo_imp.experiments.train_dqn --rows 5 --episodes 20     # quick test
The model is saved every 25 episodes; --resume continues from the last checkpoint.
Output: sumo_imp/results/<R>x<C>/dqn/{dqn|dqn_split}.pt, _training_log.csv, _training_curve.png
"""
import argparse
import tempfile
import time
from pathlib import Path

import sumo_imp  # noqa: F401  (paths)
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import GridMap, density_lambdas  # noqa: E402
from generate_demand import make_trips, trips_to_frame  # noqa: E402
from qits.agents.dqn_agent import DQNController, DQNLearner  # noqa: E402
from sumo_imp.config import load_config  # noqa: E402
from sumo_imp.controllers import dqn_path  # noqa: E402
from sumo_imp.simulator import SumoSimulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--mode", default="phase", choices=["phase", "split"])
    ap.add_argument("--port", type=int, default=33000)
    ap.add_argument("--threads", type=int, default=4, help="torch threads")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg, scfg = load_config(args.rows)
    d = cfg["dqn"]
    episodes = args.episodes or d["episodes"]
    gmap = GridMap(scfg)
    lams = density_lambdas(scfg)
    path = dqn_path(scfg, args.mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_path = path.with_name(path.stem + "_training_log.csv")
    learner = DQNLearner(cfg, seed=d["train_seed_base"], mode=args.mode)
    log, start = [], 0
    if args.resume and path.exists() and log_path.exists():
        learner.q.load_state_dict(torch.load(path, weights_only=True))
        learner.target.load_state_dict(learner.q.state_dict())
        log = pd.read_csv(log_path).to_dict("records")
        start = len(log)
        print(f"resuming after episode {start} (replay buffer starts empty)")
    rng = np.random.default_rng(d["train_seed_base"])
    densities = [rng.choice(list(lams)) for _ in range(episodes)]     # same sequence when resuming

    tic = time.perf_counter()
    decay_eps = max(1, int(d["eps_decay_frac"] * episodes))
    with tempfile.TemporaryDirectory() as tmp:
        for ep in range(start, episodes):
            eps = max(d["eps_end"], d["eps_start"] - (d["eps_start"] - d["eps_end"]) * ep / decay_eps)
            density, seed = densities[ep], d["train_seed_base"] + ep
            xml = make_trips(scfg, lams[density], seed, Path(tmp) / "trips.xml")
            ctl = DQNController(cfg, learner.q, learner=learner, epsilon=eps, seed=seed, mode=args.mode)
            n_upd = len(learner.update_norms)
            res = SumoSimulator(cfg, scfg, trips_to_frame(gmap, xml), ctl, seed, port=args.port, gmap=gmap,
                                duration=d["episode_s"]).run()
            s = res["summary"]
            new_losses = learner.losses[n_upd:]
            log.append({"episode": ep, "density": density, "epsilon": eps, "avg_delay_s": s["avg_delay_s"],
                        "avg_delay_all_s": s["avg_delay_all_s"], "reward": float(np.mean(ctl.rewards)),
                        "loss": float(np.mean(new_losses)) if new_losses else np.nan,
                        "update_norm": float(np.mean(learner.update_norms[n_upd:])) if new_losses else np.nan,
                        "policy_entropy": ctl.summary()["policy_entropy"]})
            if (ep + 1) % 25 == 0 or ep + 1 == episodes:
                recent = pd.DataFrame(log[-25:])
                print(f"ep {ep + 1:4d}/{episodes}  eps={eps:.2f}  delay={recent.avg_delay_s.mean():6.1f}s  "
                      f"reward={recent.reward.mean():+.3f}  loss={recent.loss.mean():.4f}  "
                      f"({time.perf_counter() - tic:.0f}s)", flush=True)
                learner.save(path)
                pd.DataFrame(log).to_csv(log_path, index=False)

    df = pd.DataFrame(log)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for dens, sub in df.groupby("density"):
        ax[0].plot(sub.episode, sub.avg_delay_s.rolling(10, min_periods=1).mean(), label=dens)
    ax[0].set(xlabel="Episode", ylabel="Average delay (s), rolling 10", title=f"{path.stem} on SUMO: delay")
    ax[0].legend()
    ax[1].plot(df.episode, df.reward.rolling(10, min_periods=1).mean())
    ax[1].set(xlabel="Episode", ylabel="Mean reward", title="Reward")
    ax[2].plot(df.episode, df.loss.rolling(10, min_periods=1).mean())
    ax[2].set(xlabel="Episode", ylabel="TD loss", title="Loss")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path.with_name(path.stem + "_training_curve.png"), dpi=140)
    print(f"saved {path} ({time.perf_counter() - tic:.0f}s)")


if __name__ == "__main__":
    main()
