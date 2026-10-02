"""Train the DQN baseline (paper: trained until the average-delay metric converged).

Episodes of cfg.dqn.episode_s seconds on fresh demand (seeds 2000+, outside the test set),
density drawn at random from low/medium/high each episode; ε decays linearly from
eps_start to eps_end over the first eps_decay_frac of the episodes.

    python -m experiments.train_dqn [--episodes 500] [--mode phase|split]
Output: results/dqn/{dqn|dqn_split}.pt, ..._training_log.csv, ..._training_curve.png
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from qits.agents.dqn_agent import DQNController, DQNLearner
from qits.config import ROOT, density_lambdas, load_config
from qits.traffic.demand import generate_demand
from qits.traffic.grid import Grid
from qits.traffic.simulator import TrafficSimulator

OUT = ROOT / "results" / "dqn"


def model_name(mode):
    return "dqn" if mode == "phase" else "dqn_split"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--mode", default="phase", choices=["phase", "split"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    name = model_name(args.mode)

    cfg = load_config()
    d = cfg["dqn"]
    episodes = args.episodes or d["episodes"]
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    lams = density_lambdas(cfg)
    rng = np.random.default_rng(d["train_seed_base"])
    learner = DQNLearner(cfg, seed=d["train_seed_base"], mode=args.mode)
    OUT.mkdir(parents=True, exist_ok=True)

    log, tic = [], time.perf_counter()
    decay_eps = max(1, int(d["eps_decay_frac"] * episodes))
    for ep in range(episodes):
        eps = max(d["eps_end"], d["eps_start"] - (d["eps_start"] - d["eps_end"]) * ep / decay_eps)
        density = rng.choice(list(lams))
        seed = d["train_seed_base"] + ep
        demand = generate_demand(g, lams[density], d["episode_s"], seed, cfg["demand"]["min_hops"])
        ctl = DQNController(cfg, learner.q, learner=learner, epsilon=eps, seed=seed, mode=args.mode)
        n_upd = len(learner.update_norms)
        res = TrafficSimulator(cfg, demand, ctl, seed).run(d["episode_s"])
        s = res["summary"]
        log.append({"episode": ep, "density": density, "epsilon": eps,
                    "avg_delay_s": s["avg_delay_s"], "reward": float(np.mean(ctl.rewards)),
                    "loss": float(np.mean(learner.losses[n_upd:])) if len(learner.losses) > n_upd else np.nan,
                    "update_norm": float(np.mean(learner.update_norms[n_upd:])) if len(learner.update_norms) > n_upd else np.nan,
                    "policy_entropy": ctl.summary()["policy_entropy"]})
        if (ep + 1) % 25 == 0:
            recent = pd.DataFrame(log[-25:])
            print(f"ep {ep + 1:4d}/{episodes}  ε={eps:.2f}  delay={recent.avg_delay_s.mean():6.2f}s  "
                  f"reward={recent.reward.mean():+.3f}  loss={recent.loss.mean():.4f}  "
                  f"({time.perf_counter() - tic:.0f}s)", flush=True)
            learner.save(OUT / f"{name}.pt")

    learner.save(OUT / f"{name}.pt")
    df = pd.DataFrame(log)
    df.to_csv(OUT / f"{name}_training_log.csv", index=False)
    pd.DataFrame({"step": np.arange(len(learner.update_norms)), "update_norm": learner.update_norms}) \
      .iloc[::10].to_csv(OUT / f"{name}_update_norms.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for dens, sub in df.groupby("density"):
        ax[0].plot(sub.episode, sub.avg_delay_s.rolling(10, min_periods=1).mean(), label=dens)
    ax[0].set(xlabel="Episode", ylabel="Average delay (s), rolling 10", title=f"{name} training: delay")
    ax[0].legend()
    ax[1].plot(df.episode, df.reward.rolling(10, min_periods=1).mean())
    ax[1].set(xlabel="Episode", ylabel="Mean reward", title="Reward")
    ax2 = ax[1].twinx()
    ax2.plot(df.episode, df.epsilon, c="gray", ls="--")
    ax2.set_ylabel("ε")
    ax[2].plot(df.episode, df.loss.rolling(10, min_periods=1).mean())
    ax[2].set(xlabel="Episode", ylabel="TD loss", title="Loss")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_training_curve.png", dpi=140)
    print(f"saved model and logs to {OUT}  ({time.perf_counter() - tic:.0f}s)")


if __name__ == "__main__":
    main()
