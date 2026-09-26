"""Diagnostics of Q-ITS runs over time (mean over available seeds of one density).

    python -m experiments.plot_qits_run --density high
Output: results/qits/run_diagnostics_{density}.png
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from qits.config import ROOT

RUNS = ROOT / "results" / "qits" / "runs" / "qits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", default="high")
    args = ap.parse_args()

    ep = pd.concat(pd.read_csv(f) for f in RUNS.glob(f"{args.density}_seed*_epochs.csv"))
    dl = pd.concat(pd.read_csv(f) for f in RUNS.glob(f"{args.density}_seed*_delay.csv"))
    ep, dl = ep.groupby("t").mean(), dl.groupby("minute").mean()

    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    panels = [("dtheta_norm", "||Δθ|| per SPSA step"), ("entropy", "Consensus entropy ΔH (nats, Eq. 19)"),
              ("cost", "Local cost C_i (Eq. 9)"), ("split_mean", "Mean NS green split S_i"),
              ("latency_s", "Decision latency (s, Eq. 6)")]
    for a, (col, title) in zip(ax.flat, panels):
        a.plot(ep.index, ep[col])
        a.set(xlabel="Time (s)", title=title)
        a.grid(alpha=0.3)
    ax.flat[5].plot(dl.index, dl.delay, marker=".")
    ax.flat[5].set(xlabel="Arrival minute", title="Average vehicle delay (s)")
    ax.flat[5].grid(alpha=0.3)
    fig.suptitle(f"Q-ITS run diagnostics — {args.density} density")
    fig.tight_layout()
    out = ROOT / "results" / "qits" / f"run_diagnostics_{args.density}.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
