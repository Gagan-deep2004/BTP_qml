"""Evaluation metrics shared by all controllers (guide Section 10)."""
import numpy as np
from scipy import stats


def routing_entropy(p, valid_mask):
    """Eq. 19: ΔH = -Σ_k p_k log p_k of each node's normalised routing weights (nats),
    averaged over nodes."""
    p = np.where(valid_mask > 0, p, 0.0)
    p = p / p.sum(1, keepdims=True)
    h = -np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-300, None)), 0.0), axis=1)
    return float(h.mean())


def entropic_stability_gain(h_baseline, h_qits):
    """Eq. 20."""
    return (h_baseline - h_qits) / h_baseline * 100.0


def ci95(values):
    """Eq. 25: mean ± 1.96 σ / sqrt(N)."""
    v = np.asarray(values, float)
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(v.mean()), float(half)


def paired_ttest(a, b):
    """Paired t-test over seeds (same demand files for both methods)."""
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def improvement(baseline, qits):
    """Table V: (X_baseline - X_QITS) / X_baseline x 100."""
    return (baseline - qits) / baseline * 100.0
