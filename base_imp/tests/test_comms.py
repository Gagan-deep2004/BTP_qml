import numpy as np
import pytest

from qits.comms.entanglement import (fidelity, swap_fidelity, swap_via_circuit, werner_qber,
                                     werner_qmi, werner_state)
from qits.comms.latency import LatencyModel
from qits.comms.network import QuantumNetwork
from qits.comms.qkd import bb84, from_bits, otp, p_undetected, to_bits
from qits.comms.trust import consensus_update, laplacian, trust_score
from qits.config import load_config
from qits.traffic.grid import Grid


def _net(seed=0):
    cfg = load_config()
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    return cfg, g, QuantumNetwork(cfg, g, np.random.default_rng(seed))


# ---------------------------------------------------------------- entanglement
@pytest.mark.parametrize("F", [0.25, 0.6, 0.85, 0.95, 1.0])
def test_werner_fidelity_eq1(F):
    assert fidelity(werner_state(F)) == pytest.approx(F)


@pytest.mark.parametrize("f1,f2", [(0.95, 0.95), (0.9, 0.8), (1.0, 0.7)])
def test_swapping_formula_matches_circuit_eq4(f1, f2):
    assert swap_via_circuit(f1, f2) == pytest.approx(swap_fidelity(f1, f2), abs=1e-10)


def test_quantum_mutual_information_eq17():
    assert werner_qmi(1.0) == pytest.approx(2.0)
    assert werner_qmi(0.25) == pytest.approx(0.0, abs=1e-12)
    assert werner_qmi(0.95) > werner_qmi(0.85)


def test_fidelity_threshold_matches_qber_threshold():
    # paper: fidelity threshold 0.85 and QBER threshold 11 % are consistent
    assert werner_qber(0.85) == pytest.approx(0.10)
    assert werner_qber(0.835) == pytest.approx(0.11)


# ------------------------------------------------------------------------- QKD
def test_bb84_accepts_good_link_rejects_bad_and_eavesdropper():
    rng = np.random.default_rng(1)
    good = bb84(20000, werner_qber(0.95), rng)
    assert not good["aborted"] and good["key"].size > 3000
    assert bb84(20000, werner_qber(0.80), rng)["aborted"]                    # QBER 13 % -> abort
    assert bb84(20000, werner_qber(0.95), rng, eve_fraction=1.0)["aborted"]  # intercept-resend
    assert p_undetected(128) < 1e-15


def test_otp_roundtrip_eq13():
    rng = np.random.default_rng(2)
    E = np.array([0.1, -0.5, 0.9, 0.0], np.float32)
    key = rng.integers(0, 2, 200, dtype=np.uint8)
    cipher = otp(to_bits(E), key)
    assert not np.array_equal(cipher, to_bits(E))
    np.testing.assert_array_equal(from_bits(otp(cipher, key)), E)


# ------------------------------------------------------------ trust/consensus
def test_trust_and_laplacian_eq14_15():
    assert trust_score(0.95, 0.1, 1.0, 0.5) == pytest.approx(0.90)
    L = laplacian(3, [(0, 1), (1, 2)], [0.9, 0.8])
    np.testing.assert_allclose(L.sum(1), 0, atol=1e-12)
    np.testing.assert_allclose(L, L.T)
    assert np.linalg.eigvalsh(L).min() > -1e-12


def test_consensus_converges_to_average_eq16():
    L = laplacian(4, [(0, 1), (1, 2), (2, 3)], [0.9] * 3)
    E = np.array([[1.0], [0.0], [-1.0], [4.0]])
    for _ in range(300):
        E = consensus_update(E, L, 0.2)
    np.testing.assert_allclose(E, 1.0, atol=1e-6)


# --------------------------------------------------------------------- network
def test_fidelity_decay_reproduces_fig1b():
    cfg, g, net = _net()
    for _ in range(1800):
        net.tick(1.0)
    assert 0.35 < net.F.mean() < 0.45          # paper Fig. 1b ends near 0.4


def test_maintenance_keeps_links_usable():
    cfg, g, net = _net()
    Fs = []
    for _ in range(1800):
        net.tick(1.0)
        net.maintain()
        Fs.append(net.F.copy())
    Fs = np.array(Fs)
    assert Fs.min() >= cfg["comms"]["fidelity"]["eps_F"]
    assert Fs.mean() > 0.90                     # paper: average fidelity stays above 0.90
    assert net.stats["reentanglements"] > 0


def test_exchange_delivers_and_detects_eavesdropper():
    cfg, g, net = _net()
    net.generate_keys(10.0)
    E = np.random.default_rng(3).normal(size=(g.n_nodes, 4))
    E_new, info = net.exchange(E)
    assert info["n_active"] > 0.9 * net.n_links
    np.testing.assert_allclose(E_new.mean(0), E.mean(0), atol=1e-12)   # consensus preserves mean
    # an eavesdropper on link 0 is detected by QBER and the link is dropped
    net.eve_fraction[0] = 1.0
    net.generate_keys(10.0)
    assert net.compromised[0] and net.stats["qkd_aborts"] >= 1
    routes = net.routes()
    assert routes[0] is None or len(routes[0][0]) == 3        # direct channel no longer used


def test_latency_components_eq6():
    cfg = load_config()
    total, parts = LatencyModel(cfg).qits(t_ent=0.002, hops=1)
    assert set(parts) == {"T_ent", "T_meas", "T_class", "T_proc"}
    assert total == pytest.approx(sum(parts.values()))
