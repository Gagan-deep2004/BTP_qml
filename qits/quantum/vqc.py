"""Variational quantum circuit per intersection (Eq. 7) and its outputs (Eq. 8).

Ansatz (d layers):   U(θ) = Π_l [ ⊗_k RZ(θ_{l,k,1}) RY(θ_{l,k,0}) ] [ Π_ring CX ]
Parameter layout:    θ[2*(l*n_q + k)] -> RY on qubit k in layer l, θ[... + 1] -> RZ   (A16)

Outputs read from the measured distribution over 2^n_q basis states (Qiskit little-endian:
bit q of index k is qubit q):
    p_ij   routing probabilities toward N,E,S,W = marginal of qubits (0, 1)      (Eq. 8)
    E_i    expectation values <Z_q> for every qubit (exchanged in consensus)   (Eq. 16)
    S_i    NS green split from <Z_{n_q-1}>                                        (A5)

Backends
    exact  : NumPy statevector, exact probabilities (fast; noiseless)
    shots  : exact + multinomial sampling of `shots` (shot noise only)
    noisy  : NumPy density matrix with the paper's NISQ noise model + `shots` sampling
             (the paper's setting; qits/quantum/dm_sim.py)
    aer    : same as `noisy` but run through Qiskit Aer (reference; slow)
NumPy backends are verified against Qiskit Statevector / Aer in tests/test_quantum.py.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .noise import make_noise_model

BACKENDS = ("exact", "shots", "noisy", "aer")


# --------------------------------------------------------------------------- Qiskit
def build_ansatz(n_qubits=4, depth=8):
    th = ParameterVector("θ", 2 * n_qubits * depth)
    qc = QuantumCircuit(n_qubits)
    for l in range(depth):
        for q in range(n_qubits):
            k = 2 * (l * n_qubits + q)
            qc.ry(th[k], q)
            qc.rz(th[k + 1], q)
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)
    return qc, th


def full_circuit(amp, ansatz, params, theta, measure=True, prep="initialize"):
    """State preparation |ψ> (Eq. 2) followed by the bound ansatz U(θ).
    prep="density_matrix" loads |ψ><ψ| with Aer's SetDensityMatrix (the Aer density-matrix
    method does not support `initialize`); state loading is then ideal and noise acts on U(θ)."""
    n = ansatz.num_qubits
    qc = QuantumCircuit(n)
    if prep == "density_matrix":
        from qiskit.quantum_info import DensityMatrix
        from qiskit_aer.library import SetDensityMatrix
        qc.append(SetDensityMatrix(DensityMatrix(np.asarray(amp, dtype=complex))), range(n))
    else:
        qc.initialize(np.asarray(amp, dtype=complex), range(n))
    qc.compose(ansatz.assign_parameters({params: theta}), inplace=True)
    if measure:
        qc.measure_all()
    return qc


# ---------------------------------------------------------------- NumPy statevector
def ansatz_layer_angles(thetas, layer, n_qubits):
    """Batched gate matrices of one layer: list over qubits of (RY (B,2,2), RZ (B,2,2))."""
    B = len(thetas)
    mats = []
    for q in range(n_qubits):
        k = 2 * (layer * n_qubits + q)
        a, b = thetas[:, k] / 2, thetas[:, k + 1] / 2
        c, s = np.cos(a), np.sin(a)
        ry = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)
        rz = np.zeros((B, 2, 2), complex)
        rz[:, 0, 0], rz[:, 1, 1] = np.exp(-1j * b), np.exp(1j * b)
        mats.append((ry, rz))
    return mats


def _apply_1q(psi, mats, axis):
    """psi: (B, 2, ..., 2); mats: (B, 2, 2) applied on `axis`."""
    psi = np.moveaxis(psi, axis, -1)
    psi = np.einsum("bij,b...j->b...i", mats, psi)
    return np.moveaxis(psi, -1, axis)


def _apply_cx(psi, c_axis, t_axis):
    psi = psi.copy()
    idx = [slice(None)] * psi.ndim
    idx[c_axis] = 1
    sub = psi[tuple(idx)]
    t_sub = t_axis - 1 if t_axis > c_axis else t_axis
    psi[tuple(idx)] = np.flip(sub, axis=t_sub)
    return psi


def statevector_batch(amps, thetas, n_qubits, depth):
    """Exact final statevectors for a batch. amps: (B, 2^n), thetas: (B, 2*n*d) -> (B, 2^n)."""
    B = amps.shape[0]
    psi = amps.astype(complex).reshape((B,) + (2,) * n_qubits)
    axis = lambda q: n_qubits - q          # qubit q <-> tensor axis (batch is axis 0)
    for l in range(depth):
        for q, (ry, rz) in enumerate(ansatz_layer_angles(thetas, l, n_qubits)):
            psi = _apply_1q(psi, rz @ ry, axis(q))          # RY first, then RZ
        for q in range(n_qubits):
            psi = _apply_cx(psi, axis(q), axis((q + 1) % n_qubits))
    return psi.reshape(B, -1)


# --------------------------------------------------------------------------- policy
class VQCPolicy:
    def __init__(self, n_qubits=4, depth=8, backend="exact", shots=8192, noise=None, seed=None):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.n_qubits, self.depth = n_qubits, depth
        self.dim = 2 ** n_qubits
        self.n_params = 2 * n_qubits * depth
        self.backend, self.shots = backend, shots
        self.rng = np.random.default_rng(seed)
        self.n_circuit_evals = 0
        # bit table: bits[k, q] = value of qubit q in basis state k
        self.bits = (np.arange(self.dim)[:, None] >> np.arange(n_qubits)[None, :]) & 1
        if backend == "noisy":
            from .dm_sim import NoisyDMSimulator
            self.dm = NoisyDMSimulator(n_qubits, depth, **(noise or {}))
        if backend == "aer":
            from qiskit_aer import AerSimulator
            self.ansatz, self.params = build_ansatz(n_qubits, depth)
            self.sim = AerSimulator(method="density_matrix",
                                    noise_model=make_noise_model(**(noise or {})),
                                    seed_simulator=seed)

    @classmethod
    def from_config(cls, cfg, seed=None, backend=None):
        q = cfg["quantum"]
        return cls(q["n_qubits"], q["depth"], backend or q["backend"], q["shots"], q["noise"], seed)

    def init_params(self, n, scale=np.pi):
        return self.rng.uniform(-scale, scale, size=(n, self.n_params))

    def distribution(self, amps, thetas):
        """Measured distribution over the 2^n basis states: (B, 2^n)."""
        amps, thetas = np.atleast_2d(amps), np.atleast_2d(thetas)
        self.n_circuit_evals += len(amps)
        if self.backend == "aer":
            circs = [full_circuit(a, self.ansatz, self.params, t, prep="density_matrix")
                     for a, t in zip(amps, thetas)]
            result = self.sim.run(circs, shots=self.shots).result()
            probs = np.zeros((len(circs), self.dim))
            for i in range(len(circs)):
                for key, c in result.get_counts(i).items():
                    probs[i, int(key.replace(" ", ""), 2)] += c
            return probs / self.shots

        if self.backend == "noisy":
            probs = self.dm.probabilities(amps, thetas)
        else:
            probs = np.abs(statevector_batch(amps, thetas, self.n_qubits, self.depth)) ** 2
        probs /= probs.sum(1, keepdims=True)
        if self.backend in ("shots", "noisy"):
            probs = self.rng.multinomial(self.shots, probs) / self.shots
        return probs

    def outputs(self, probs, valid_mask, split_min=0.2, split_max=0.8):
        """probs (B, 2^n) -> routing p (B, 4), expectations E (B, n_q), signal split S (B,)."""
        route = np.zeros((len(probs), 4))
        np.add.at(route.T, np.arange(self.dim) % 4, probs.T)          # marginal of qubits 0,1
        route *= valid_mask
        tot = route.sum(1, keepdims=True)
        uniform = valid_mask / valid_mask.sum(1, keepdims=True)
        route = np.where(tot > 1e-12, route / np.maximum(tot, 1e-12), uniform)
        E = probs @ (1 - 2 * self.bits)                                 # <Z_q>
        S = split_min + (split_max - split_min) * (1 + E[:, -1]) / 2
        return route, E, S

    def forward(self, amps, thetas, valid_mask, split_min=0.2, split_max=0.8):
        return self.outputs(self.distribution(amps, thetas), valid_mask, split_min, split_max)
