"""NISQ noise model from the paper: T1 = 85 µs, T2 = 90 µs, single-/two-qubit gate
errors 2.5e-4 / 1.3e-3, applied as depolarising error composed with thermal relaxation."""
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error


def make_noise_model(t1_s=85e-6, t2_s=90e-6, p1=2.5e-4, p2=1.3e-3,
                     gate_time_1q_s=35e-9, gate_time_2q_s=300e-9):
    nm = NoiseModel()
    e1 = depolarizing_error(p1, 1).compose(thermal_relaxation_error(t1_s, t2_s, gate_time_1q_s))
    tr2 = thermal_relaxation_error(t1_s, t2_s, gate_time_2q_s)
    e2 = depolarizing_error(p2, 2).compose(tr2.expand(tr2))
    nm.add_all_qubit_quantum_error(e1, ["ry", "rz"])
    nm.add_all_qubit_quantum_error(e2, ["cx"])
    return nm
