"""Single-intersection VQC + SPSA convergence (reproduces Fig. 1c).

Takes one real snapshot from the consumer-device dataset (centre node, medium density,
seed 42): features x_i(t) -> amplitude encoding (Eq. 2), link delays δ_ij(t) (normalised
by free-flow time). Optimises the local cost C_i(θ) (Eq. 9) with SPSA (Eq. 10-11) for
300 iterations, for each backend and perturbation size ξ, over several random θ(0).

    python -m experiments.single_node_convergence
Output: results/quantum/fig1c_convergence.png, results/quantum/single_node_convergence.csv
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qits.config import ROOT, load_config
from qits.quantum.cost import local_cost
from qits.quantum.encoding import amplitude_encode
from qits.quantum.spsa import spsa_step
from qits.quantum.vqc import VQCPolicy
from qits.traffic.grid import DIR_NAMES, Grid

OUT = ROOT / "results" / "quantum"


def pick_snapshot(cfg, density="medium"):
    seed = cfg["simulation"]["base_seed"]
    d = np.load(ROOT / "data" / "devices" / density / f"seed_{seed}.npz")
    rows, cols = cfg["network"]["rows"], cfg["network"]["cols"]
    node = (rows // 2) * cols + cols // 2
    spread = d["delay"][:, node].max(1) - d["delay"][:, node].min(1)
    t = int(np.argmax(spread[300:])) + 300        # after warm-up, clearest delay contrast
    return node, t, d["x"][t, node], d["delay"][t, node]


def run(policy, amp, delta, mask, theta0, iters, eta, xi, lam_var, rng, exact_policy):
    amp_b, delta_b, mask_b = amp[None], delta[None], mask[None]

    def J(thetas):
        n = len(thetas)
        p, _, _ = policy.forward(np.repeat(amp_b, n, 0), thetas, np.repeat(mask_b, n, 0))
        return local_cost(p, np.repeat(delta_b, n, 0), np.repeat(mask_b, n, 0), lam_var)

    def true_cost(theta):   # noiseless cost for monitoring only
        p, _, _ = exact_policy.forward(amp_b, theta, mask_b)
        return local_cost(p, delta_b, mask_b, lam_var)[0], p[0]

    theta, log = theta0[None].copy(), []
    for it in range(iters):
        new, g, _, _ = spsa_step(J, theta, eta, xi, rng)
        c, p = true_cost(new)
        log.append({"iter": it, "dtheta_norm": float(np.linalg.norm(new - theta)),
                    "cost": float(c), **{f"p_{DIR_NAMES[j]}": float(p[j]) for j in range(4)}})
        theta = new
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--inits", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config()
    q = cfg["quantum"]
    grid = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    node, t, x, delay = pick_snapshot(cfg)
    ff = np.ceil(cfg["network"]["link_length_m"] / cfg["network"]["free_flow_speed_mps"])
    delta = delay / ff
    mask = grid.valid_mask[node]
    amp = amplitude_encode(x)
    print(f"node {node}, t={t}s, δ/ff (N,E,S,W) = {np.round(delta, 2)}")

    exact = VQCPolicy.from_config(cfg, seed=0, backend="exact")
    settings = [("exact", q["perturbation"]), ("noisy", q["perturbation"]),
                ("exact", 0.1), ("noisy", 0.1)]
    rows = []
    for backend, xi in settings:
        for init in range(args.inits):
            pol = VQCPolicy.from_config(cfg, seed=100 + init, backend=backend)
            theta0 = pol.init_params(1, q["init_scale"])[0]
            log = run(pol, amp, delta, mask, theta0, args.iters, q["learning_rate"], xi,
                      q["lam_var"], np.random.default_rng(init), exact)
            rows += [{"backend": backend, "xi": xi, "init": init, **r} for r in log]
        last = pd.DataFrame(rows).query("backend == @backend and xi == @xi and iter >= @args.iters - 20")
        print(f"{backend:5s} ξ={xi:<5} final cost={last.cost.mean():.3f}  "
              f"final ||Δθ||={last.dtheta_norm.mean():.4f}  p_best={last[[f'p_{d}' for d in DIR_NAMES]].mean().round(2).to_dict()}")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "single_node_convergence.csv", index=False)

    best_cost = local_cost(np.eye(4)[[np.argmin(np.where(mask > 0, delta, np.inf))]],
                           delta[None], mask[None], q["lam_var"])[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for (backend, xi), g in df.groupby(["backend", "xi"], sort=False):
        m = g.groupby("iter").mean(numeric_only=True)
        norm = m.dtheta_norm.rolling(10, min_periods=1).mean()
        label = f"{backend}, ξ={xi}"
        ax[0].plot(norm / norm.iloc[:10].mean(), label=label)
        ax[1].plot(m.cost, label=label)
    ax[0].set(xlabel="Iteration", ylabel="Parameter oscillation ||Δθ|| (normalised)",
              title="Convergence of circuit parameters (Fig. 1c)")
    ax[1].axhline(best_cost, ls="--", c="gray", label="optimum (all flow to min-delay link)")
    ax[1].set(xlabel="Iteration", ylabel="Local cost C_i(θ) (Eq. 9, noiseless)",
              title=f"Node {node}: cost, mean of {args.inits} inits")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1c_convergence.png", dpi=150)
    print(f"saved {OUT / 'fig1c_convergence.png'}")


if __name__ == "__main__":
    main()
