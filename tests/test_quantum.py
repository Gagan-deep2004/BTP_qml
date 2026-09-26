import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qits.config import load_config
from qits.quantum.cost import local_cost
from qits.quantum.dm_sim import NoisyDMSimulator
from qits.quantum.noise import make_noise_model
from qits.quantum.encoding import amplitude_encode
from qits.quantum.spsa import spsa_step
from qits.quantum.vqc import VQCPolicy, build_ansatz, full_circuit, statevector_batch

rng = np.random.default_rng(0)


def test_amplitude_encoding_eq2():
    x = rng.random((5, 16))
    amp = amplitude_encode(x, eps=0.0)
    np.testing.assert_allclose(np.linalg.norm(amp, axis=1), 1.0)
    np.testing.assert_allclose(amp ** 2, x / x.sum(1, keepdims=True))   # |amp|^2 = T^k / ||T||_1
    assert np.all(np.isfinite(amplitude_encode(np.zeros(16))))           # zero vector handled


def test_ansatz_parameter_count():
    qc, th = build_ansatz(4, 8)
    assert len(th) == 64 and qc.num_parameters == 64
    assert qc.count_ops()["cx"] == 32


def test_numpy_statevector_matches_qiskit():
    ansatz, params = build_ansatz(4, 8)
    amps = amplitude_encode(rng.random((3, 16)))
    thetas = rng.uniform(-np.pi, np.pi, (3, 64))
    ours = statevector_batch(amps, thetas, 4, 8)
    for a, t, psi in zip(amps, thetas, ours):
        ref = Statevector(full_circuit(a, ansatz, params, t, measure=False)).data
        np.testing.assert_allclose(psi, ref, atol=1e-10)


def test_outputs_routing_expectations_split():
    pol = VQCPolicy(backend="exact")
    probs = np.zeros((1, 16))
    probs[0, 0b1001] = 1.0                 # qubits 0 and 3 are |1>, qubits 1, 2 are |0>
    mask = np.ones((1, 4))
    p, E, S = pol.outputs(probs, mask)
    np.testing.assert_allclose(p[0], [0, 1, 0, 0])      # (q0, q1) = (1, 0) -> j = 1 = East
    np.testing.assert_allclose(E[0], [-1, 1, 1, -1])
    assert S[0] == pytest.approx(0.2)                   # <Z_3> = -1 -> split_min
    # masking: East invalid -> fall back to uniform over valid directions
    p, _, _ = pol.outputs(probs, np.array([[1.0, 0, 1, 1]]))
    np.testing.assert_allclose(p[0], [1 / 3, 0, 1 / 3, 1 / 3])


def test_numpy_density_matrix_matches_aer():
    """NumPy noisy simulator == Aer density-matrix simulation with the same noise model."""
    from qiskit_aer import AerSimulator
    cfg = load_config()
    noise = cfg["quantum"]["noise"]
    ansatz, params = build_ansatz(4, 8)
    amps = amplitude_encode(rng.random((2, 16)))
    thetas = rng.uniform(-np.pi, np.pi, (2, 64))
    ours = NoisyDMSimulator(4, 8, **noise).run(amps, thetas)
    sim = AerSimulator(method="density_matrix", noise_model=make_noise_model(**noise))
    for a, t, rho in zip(amps, thetas, ours):
        qc = full_circuit(a, ansatz, params, t, measure=False, prep="density_matrix")
        qc.save_density_matrix()
        ref = np.asarray(sim.run(qc).result().data(0)["density_matrix"])
        np.testing.assert_allclose(rho, ref, atol=1e-9)
    # noise really acts: purity < 1 and distribution differs from the noiseless one
    assert np.all(np.einsum("bij,bji->b", ours, ours).real < 0.99)


def test_noisy_backend_close_to_exact():
    cfg = load_config()
    amps = amplitude_encode(rng.random((2, 16)))
    thetas = rng.uniform(-np.pi, np.pi, (2, 64))
    exact = VQCPolicy.from_config(cfg, seed=1, backend="exact").distribution(amps, thetas)
    for backend in ("noisy", "aer"):
        noisy = VQCPolicy.from_config(cfg, seed=1, backend=backend).distribution(amps, thetas)
        tvd = 0.5 * np.abs(exact - noisy).sum(1)
        assert np.all(tvd < 0.15)     # NISQ noise + 8192 shots: small but non-zero deviation
        np.testing.assert_allclose(noisy.sum(1), 1.0)


def test_local_cost_eq9():
    p = np.array([[0.7, 0.1, 0.1, 0.1]])
    delta = np.array([[1.0, 2.0, 3.0, 4.0]])
    c = local_cost(p, delta, np.ones((1, 4)), lam_var=0.5)
    assert c[0] == pytest.approx(0.7 + 0.2 + 0.3 + 0.4 + 0.5 * np.var(p[0]))


def test_spsa_minimises_quadratic():
    J = lambda th: (th ** 2).sum(1)
    th = np.ones((2, 10))
    r = np.random.default_rng(1)
    for _ in range(300):
        th, *_ = spsa_step(J, th, eta=0.05, xi=0.01, rng=r)
    assert np.all(J(th) < 1e-3)
