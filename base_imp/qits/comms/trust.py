"""Trust score (Eq. 14), quantum-consensus Laplacian (Eq. 15) and consensus update (Eq. 16)."""
import numpy as np


def trust_score(F, length_km, alpha=1.0, beta=0.5):
    """Eq. 14:  T_ij(t) = α F_ij(t) - β L_ij."""
    return alpha * np.asarray(F) - beta * np.asarray(length_km)


def laplacian(n_nodes, pairs, weights):
    """Eq. 15: [L_Q]_ij = -T_ij for linked i != j, [L_Q]_ii = Σ_k T_ik, 0 otherwise."""
    L = np.zeros((n_nodes, n_nodes))
    for (i, j), w in zip(pairs, weights):
        L[i, j] -= w
        L[j, i] -= w
        L[i, i] += w
        L[j, j] += w
    return L


def consensus_update(E, L, eta_c):
    """Eq. 16:  E_i <- E_i - η Σ_j T_ij (E_i - E_j)   ==   E <- E - η L_Q E."""
    return E - eta_c * (L @ E)
