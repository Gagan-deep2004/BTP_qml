"""Entanglement-based QKD (BB84/E91-style) and one-time-pad encryption (Eq. 13).

Security logic kept as in the paper (Sec. IV-B):
  * sifted-key QBER is estimated on a disclosed sample; above 11 % the batch is aborted
    and the link is dropped until it is re-entangled;
  * an intercept-resend eavesdropper on a fraction f of the pairs adds ~f/4 to the QBER;
  * the probability that an eavesdropper escapes detection on n checked bits is (3/4)^n.
"""
import numpy as np


def h2(p):
    p = float(p)
    return 0.0 if p <= 0 or p >= 1 else -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def p_undetected(n_bits):
    return 0.75 ** n_bits


def bb84(n_raw, qber_channel, rng, qber_max=0.11, sample_frac=0.1, eve_fraction=0.0,
         ec_efficiency=1.2):
    """One key-generation batch over `n_raw` measured pairs.

    Returns dict(key, qber_est, aborted, n_sifted, classical_bytes). `key` is the final
    secret key held identically by both ends (error correction / privacy amplification are
    modelled through the final length 1 - 2 h(Q), not bit by bit)."""
    qber = qber_channel + 0.25 * eve_fraction * (1 - 2 * qber_channel)
    alice = rng.integers(0, 2, n_raw, dtype=np.uint8)
    sift = rng.integers(0, 2, n_raw) == rng.integers(0, 2, n_raw)      # bases agree
    ka = alice[sift]
    kb = ka ^ (rng.random(ka.size) < qber).astype(np.uint8)
    test = rng.random(ka.size) < sample_frac
    n_test = int(test.sum())
    est = float((ka[test] != kb[test]).mean()) if n_test else 0.0

    # classical post-processing traffic: basis announcements (both ways), disclosed sample,
    # error-correction syndrome
    n_keep = int((~test).sum())
    classical_bytes = (2 * n_raw + 2 * n_test + ec_efficiency * h2(est) * n_keep) / 8

    if est > qber_max:
        return {"key": np.zeros(0, np.uint8), "qber_est": est, "aborted": True,
                "n_sifted": int(ka.size), "classical_bytes": classical_bytes}
    n_final = int(n_keep * max(0.0, 1 - 2 * h2(est)))
    return {"key": ka[~test][:n_final], "qber_est": est, "aborted": False,
            "n_sifted": int(ka.size), "classical_bytes": classical_bytes}


def to_bits(values):
    """float32 array -> bit array (uint8)."""
    return np.unpackbits(np.asarray(values, dtype=np.float32).view(np.uint8))


def from_bits(bits):
    return np.packbits(bits).view(np.float32)


def otp(bits, key):
    """Eq. 13:  M = R ⊕ K  (and R = M ⊕ K)."""
    return bits ^ key[:bits.size]
