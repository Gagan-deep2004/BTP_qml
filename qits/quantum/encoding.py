"""Amplitude encoding of traffic features into a quantum state (Eq. 2).

    |ψ_i(t)> = Σ_k sqrt( T_i^k(t) / ||T_i(t)||_1 ) |k>

Features are non-negative, so the square roots of their L1-normalised values form a
unit-L2-norm amplitude vector. A small ε keeps the all-zero vector well defined (the
paper notes the numerical instability of near-zero vectors).
"""
import numpy as np


def amplitude_encode(x, eps=1e-6):
    """x: (..., 2**n) non-negative features -> (..., 2**n) real amplitudes with unit L2 norm."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, None) + eps
    return np.sqrt(x / x.sum(axis=-1, keepdims=True))
