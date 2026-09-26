"""Quantum communication layer on its own (reproduces Fig. 1b + security behaviour).

Runs the 5x5 quantum network for 1800 s (1 s ticks, 10 s consensus epochs):
  A. no re-entanglement                -> fidelity decays 0.95 -> ~0.4   (paper Fig. 1b)
  B. adaptive re-entanglement (Eq. 17) -> sawtooth kept above ε_F = 0.85 (paper text: avg > 0.90)
  C. like B, plus an intercept-resend eavesdropper on one link during 600-1200 s
     (full and 30 % interception) -> QBER alarm, link dropped, detour via swapping

    python -m experiments.comms_demo
Output: results/comms/fig1b_fidelity.png, results/comms/eavesdropper.png, results/comms/stats.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qits.comms.entanglement import werner_qber
from qits.comms.network import QuantumNetwork
from qits.config import ROOT, load_config
from qits.traffic.grid import Grid

OUT = ROOT / "results" / "comms"
EVE_LINK = 20


def simulate(cfg, maintain=True, eve=0.0, seed=42):
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    rng = np.random.default_rng(seed)
    net = QuantumNetwork(cfg, g, rng)
    epoch = cfg["comms"]["epoch_s"]
    E = rng.normal(size=(g.n_nodes, 4))
    log = []
    for t in range(cfg["simulation"]["duration_s"]):
        net.tick(1.0)
        net.eve_fraction[EVE_LINK] = eve if 600 <= t < 1200 else 0.0
        if maintain:
            net.maintain()
        if t % epoch == 0:
            net.generate_keys(epoch)
            E, info = net.exchange(E)
            E += rng.normal(0, 0.3, E.shape)          # new local VQC outputs each epoch
            r = net.routes()[EVE_LINK]
            log.append({"t": t, "qber_eve_link": net.last_qber[EVE_LINK],
                        "eve_link_route_hops": 0 if r is None else len(r[0]),
                        "active_links": info["n_active"]})
        log_f = {"t": t, "F_link": net.F[0], "F_mean": net.F.mean(), "F_min": net.F.min()}
        log.append(log_f)
    df = pd.DataFrame(log).groupby("t").first().reset_index()
    return net, df


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    eps = cfg["comms"]["fidelity"]["eps_F"]

    runs = {"A: no re-entanglement": simulate(cfg, maintain=False),
            "B: adaptive re-entanglement": simulate(cfg, maintain=True),
            "C1: eavesdropper (100 %)": simulate(cfg, maintain=True, eve=1.0),
            "C2: eavesdropper (30 %)": simulate(cfg, maintain=True, eve=0.3)}

    rows = []
    for name, (net, df) in runs.items():
        s = net.stats
        rows.append({"scenario": name, "mean_F": df.F_mean.mean(), "min_F": df.F_min.min(),
                     "reentanglements": s["reentanglements"], "qkd_aborts": s["qkd_aborts"],
                     "PDR": net.pdr(), "rerouted_msgs": s["msgs_rerouted"],
                     "key_bits_generated": s["key_bits_generated"], "key_bits_used": s["key_bits_used"],
                     "consensus_kB": s["consensus_bytes"] / 1e3,
                     "qkd_classical_kB": s["qkd_classical_bytes"] / 1e3})
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(stats.round(4).to_string(index=False))

    # ---- Fig. 1b
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    a, b = runs["A: no re-entanglement"][1], runs["B: adaptive re-entanglement"][1]
    ax[0].plot(a.t, a.F_link, c="tab:blue", label="single link, no re-entanglement")
    ax[0].plot(b.t, b.F_link, c="tab:green", lw=0.8, label="single link, adaptive re-entanglement")
    ax[0].axhline(eps, ls="--", c="tab:red", label=f"threshold ε = {eps}")
    ax[0].set(xlabel="Time (s)", ylabel="Entanglement fidelity", title="Fidelity decay over time (Fig. 1b)")
    ax[0].legend(fontsize=8)
    ax[1].plot(b.t, b.F_mean, label="network mean F")
    ax[1].plot(b.t, b.F_min, lw=0.7, label="network min F")
    ax2 = ax[1].twinx()
    ax2.plot(b.t, werner_qber(b.F_mean) * 100, c="tab:gray", lw=0.7, label="QBER (mean)")
    ax2.set_ylabel("QBER (%)")
    ax[1].axhline(eps, ls="--", c="tab:red")
    ax[1].set(xlabel="Time (s)", ylabel="Fidelity", title="With adaptive re-entanglement (Eq. 17)")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, ncol=3)
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1b_fidelity.png", dpi=150)

    # ---- eavesdropper
    fig, ax = plt.subplots(figsize=(8, 3.5))
    qmax = cfg["comms"]["qkd"]["qber_max"] * 100
    for name in ("C1: eavesdropper (100 %)", "C2: eavesdropper (30 %)"):
        df = runs[name][1].dropna(subset=["qber_eve_link"])
        ax.plot(df.t, df.qber_eve_link * 100, marker=".", lw=0.8, label=name)
    ax.axhline(qmax, ls="--", c="tab:red", label="abort threshold 11 %")
    ax.axvspan(600, 1200, color="tab:red", alpha=0.08, label="attack window")
    ax.set(xlabel="Time (s)", ylabel="Estimated QBER on attacked link (%)",
           title="QKD eavesdropper detection (Sec. IV-B)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "eavesdropper.png", dpi=150)
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
