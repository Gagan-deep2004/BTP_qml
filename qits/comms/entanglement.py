"""Entangled links: Bell pairs, fidelity (Eq. 1/12), swapping (Eq. 4), information gain (Eq. 17).

Each link shares a Werner state   ρ(F) = F |Φ+><Φ+| + (1 - F)/3 (I - |Φ+><Φ+|),
whose fidelity with |Φ+> = (|00> + |11>)/√2 is exactly F.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

PHI_PLUS = Statevector(np.array([1, 0, 0, 1]) / np.sqrt(2))


def werner_state(F):
    P = DensityMatrix(PHI_PLUS).data
    return DensityMatrix(F * P + (1 - F) / 3 * (np.eye(4) - P))


def fidelity(rho):
    """Eq. 1 / 12:  F_ij = Tr(ρ_ij |Φ+><Φ+|)."""
    return float(state_fidelity(rho, PHI_PLUS))


def swap_fidelity(f1, f2):
    """Fidelity after entanglement swapping of two Werner pairs (closed form of Eq. 4)."""
    return f1 * f2 + (1 - f1) * (1 - f2) / 3


def swap_via_circuit(f1, f2):
    """Eq. 4 simulated explicitly: pairs (0,1) and (2,3); the middle node holds qubits 1 and 2,
    performs a Bell measurement (deferred: CX, H, then controlled X/Z corrections on qubit 3).
    Returns the fidelity of the resulting pair (0,3)."""
    rho = werner_state(f2).tensor(werner_state(f1))        # qubits 3,2 | 1,0
    qc = QuantumCircuit(4)
    qc.cx(1, 2)
    qc.h(1)
    qc.cx(2, 3)          # X correction if m2 = 1
    qc.cz(1, 3)          # Z correction if m1 = 1
    rho_03 = partial_trace(rho.evolve(qc), [1, 2])
    return fidelity(rho_03)


def werner_qmi(F):
    """Quantum mutual information I = S(ρ_A) + S(ρ_B) - S(ρ_AB) of a Werner pair (bits).
    Both marginals are maximally mixed (S = 1)."""
    ev = np.array([F] + [(1 - F) / 3] * 3)
    ev = ev[ev > 1e-15]
    return 2.0 + float(np.sum(ev * np.log2(ev)))


def werner_qber(F):
    """Z-basis bit error rate when measuring a Werner pair: 2 (1 - F) / 3."""
    return 2.0 * (1.0 - np.asarray(F)) / 3.0
