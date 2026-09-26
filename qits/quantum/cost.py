"""Local routing cost (Eq. 9 / 22), batched over nodes.

    C_i(θ_i) = Σ_j p_ij δ_ij + λ Var_j(p_ij)       (j over valid neighbours N(i))

δ_ij is normalised by the free-flow link time so λ has a scale-free meaning.
"""
import numpy as np


def local_cost(p, delta, valid_mask, lam_var):
    """p, delta, valid_mask: (B, 4) -> (B,)."""
    n_valid = valid_mask.sum(1)
    expected_delay = (p * delta * valid_mask).sum(1)
    mean = (p * valid_mask).sum(1) / n_valid
    var = (((p - mean[:, None]) ** 2) * valid_mask).sum(1) / n_valid
    return expected_delay + lam_var * var
