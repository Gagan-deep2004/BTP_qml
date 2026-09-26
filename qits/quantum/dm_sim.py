"""Batched NumPy density-matrix simulator of the Eq. 7 ansatz with the paper's noise model.

Each gate is followed by (same order as qits.quantum.noise.make_noise_model / Aer):
    1. depolarising error   ρ -> (1 - p) ρ + p Tr_q(ρ) ⊗ I / 2^k     (k = 1 or 2 qubits)
    2. thermal relaxation on every gate qubit (T1, T2, gate time; T2 <= 2 T1, ground state 0):
         ρ00 += γ ρ11,  ρ11 *= (1 - γ),  ρ01, ρ10 *= exp(-t / T2),   γ = 1 - exp(-t / T1)
Verified against Qiskit Aer (method="density_matrix") in tests/test_quantum.py; ~500x faster
for batches because it avoids per-circuit construction.
"""
import numpy as np

from .vqc import ansatz_layer_angles


class NoisyDMSimulator:
    def __init__(self, n_qubits, depth, t1_s=85e-6, t2_s=90e-6, p1=2.5e-4, p2=1.3e-3,
                 gate_time_1q_s=35e-9, gate_time_2q_s=300e-9):
        self.n, self.depth = n_qubits, depth
        self.p1, self.p2 = p1, p2
        self.relax_1q = (1 - np.exp(-gate_time_1q_s / t1_s), np.exp(-gate_time_1q_s / t2_s))
        self.relax_2q = (1 - np.exp(-gate_time_2q_s / t1_s), np.exp(-gate_time_2q_s / t2_s))

    # tensor layout: (B, row_{n-1..0}, col_{n-1..0}); qubit q -> row axis n-q, col axis 2n-q
    def _row(self, q):
        return self.n - q

    def _col(self, q):
        return 2 * self.n - q

    def _unitary_1q(self, rho, U, q):
        r, c = self._row(q), self._col(q)
        rho = np.moveaxis(np.einsum("bij,b...j->b...i", U, np.moveaxis(rho, r, -1)), -1, r)
        rho = np.moveaxis(np.einsum("bij,b...j->b...i", U.conj(), np.moveaxis(rho, c, -1)), -1, c)
        return rho

    def _cx(self, rho, ctrl, tgt):
        for off in (0, self.n):           # rows, then columns
            c_ax, t_ax = self._row(ctrl) + off, self._row(tgt) + off
            rho = rho.copy()
            idx = [slice(None)] * rho.ndim
            idx[c_ax] = 1
            sub = rho[tuple(idx)]
            rho[tuple(idx)] = np.flip(sub, axis=t_ax - 1 if t_ax > c_ax else t_ax)
        return rho

    def _depolarize(self, rho, p, qubits):
        if p == 0:
            return rho
        k = len(qubits)
        axes = [self._row(q) for q in qubits] + [self._col(q) for q in qubits]
        moved = np.moveaxis(rho, axes, list(range(-2 * k, 0)))
        shp = moved.shape
        m = moved.reshape(shp[:-2 * k] + (2 ** k, 2 ** k))
        red = np.trace(m, axis1=-2, axis2=-1)
        mixed = red[..., None, None] * np.eye(2 ** k) / 2 ** k
        m = (1 - p) * m + p * mixed
        return np.moveaxis(m.reshape(shp), list(range(-2 * k, 0)), axes)

    def _relax(self, rho, q, gamma, coh):
        r, c = self._row(q), self._col(q)
        m = np.moveaxis(rho, (r, c), (-2, -1)).copy()
        m[..., 0, 0] += gamma * m[..., 1, 1]
        m[..., 1, 1] *= 1 - gamma
        m[..., 0, 1] *= coh
        m[..., 1, 0] *= coh
        return np.moveaxis(m, (-2, -1), (r, c))

    def run(self, amps, thetas):
        """amps (B, 2^n) pure input states, thetas (B, 2 n d) -> final density matrices (B, 2^n, 2^n)."""
        n, B = self.n, len(amps)
        psi = amps.astype(complex)
        rho = np.einsum("bi,bj->bij", psi, psi.conj()).reshape((B,) + (2,) * (2 * n))
        for l in range(self.depth):
            for q, U in enumerate(ansatz_layer_angles(thetas, l, n)):
                rho = self._unitary_1q(rho, U[0], q)           # RY
                rho = self._depolarize(rho, self.p1, [q])
                rho = self._relax(rho, q, *self.relax_1q)
                rho = self._unitary_1q(rho, U[1], q)           # RZ
                rho = self._depolarize(rho, self.p1, [q])
                rho = self._relax(rho, q, *self.relax_1q)
            for q in range(n):
                t = (q + 1) % n
                rho = self._cx(rho, q, t)
                rho = self._depolarize(rho, self.p2, [q, t])
                rho = self._relax(rho, q, *self.relax_2q)
                rho = self._relax(rho, t, *self.relax_2q)
        return rho.reshape(B, 2 ** n, 2 ** n)

    def probabilities(self, amps, thetas):
        rho = self.run(amps, thetas)
        return np.clip(np.einsum("bii->bi", rho).real, 0, None)
