"""SPSA optimiser (Eq. 10-11, 23-24), batched over nodes.

    ĝ_i = [J(θ_i + ξΔ_i) - J(θ_i - ξΔ_i)] / (2ξ) · Δ_i ,   Δ_i ∈ {±1}^d
    θ_i <- θ_i - η ĝ_i
The paper calls this "QN-SPSA", but Eq. 10 is plain SPSA, which is what is implemented (A12).
"""
import numpy as np


def spsa_step(J, thetas, eta, xi, rng):
    """J: (B, P) -> (B,) objective, evaluated once on the stacked (2B, P) batch.
    Returns (new thetas, gradient estimates, J(θ+), J(θ-))."""
    B = len(thetas)
    delta = rng.choice([-1.0, 1.0], size=thetas.shape)
    values = J(np.concatenate([thetas + xi * delta, thetas - xi * delta]))
    j_plus, j_minus = values[:B], values[B:]
    g = ((j_plus - j_minus) / (2 * xi))[:, None] * delta
    return thetas - eta * g, g, j_plus, j_minus
