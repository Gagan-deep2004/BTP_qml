"""Q-ITS controller: Algorithm 1 run by every intersection, batched over all nodes.

Every second:   quantum links age; adaptive re-entanglement (Eq. 17)            [l. 12-13]
Every epoch (10 s), for every node i:
    x_i(t) from consumer devices -> |ψ_i(t)>                           (Eq. 2)   [l. 4]
    δ_ij(t) from consumer devices (normalised by free-flow time)
    SPSA on J_i(θ_i) = C_i(θ_i) + κ D_sig(S_i) + μ Σ_j ||E_i(θ_i) - E_j||²      [l. 5-7]
                      (Eq. 9)     (A5)            (Eq. 21, neighbours' consensus states)
    forward pass -> p_ij (Eq. 8), E_i, S_i
    QKD keys, OTP-encrypted exchange of E_i, consensus step (Eq. 13-16)          [l. 8-11]
    p_ij -> vehicle routing at node i; S_i -> green split from the next cycle
    decision latency (Eq. 6) from T_ent, hops, modelled QPU time
"""
import time

import numpy as np

from ..comms.latency import LatencyModel
from ..comms.network import ClassicalNetwork, QuantumNetwork
from ..metrics import routing_entropy
from ..quantum.cost import local_cost
from ..quantum.encoding import amplitude_encode
from ..quantum.spsa import spsa_step
from ..quantum.vqc import VQCPolicy
from ..traffic.grid import E, N, S, W
from .base import Controller
from .classical_policy import ClassicalPolicy


def signal_cost(split, d_ns, d_ew, sat, cycle):
    """Clearance-time proxy of the critical NS / EW demand under green split S (A5):
    (d_ns / S + d_ew / (1 - S)) / (sat * cycle). Minimised at S* = √d_ns / (√d_ns + √d_ew)."""
    return (d_ns / split + d_ew / (1.0 - split)) / (sat * cycle)


class QITSController(Controller):
    name = "qits"

    def __init__(self, cfg, seed=42, theta0=None, use_consensus=True, policy=None,
                 control_routing=True, control_signals=True, policy_kind="quantum",
                 channel="quantum"):
        """Ablations (Table IV): policy_kind="classical" -> NoQOpt, channel="classical" -> NoQKD,
        use_consensus=False -> NoCons."""
        self.cfg, self.seed = cfg, seed
        self.theta0 = theta0
        self.use_consensus = use_consensus
        self.policy_kind, self.channel = policy_kind, channel
        # diagnostics: switch off one of the two traffic actions (not part of the paper)
        self.control_routing, self.control_signals = control_routing, control_signals
        self._policy = policy
        q, c = cfg["quantum"], cfg["comms"]
        self.eta, self.xi, self.lam = q["learning_rate"], q["perturbation"], q["lam_var"]
        self.iters = cfg["qits"]["spsa_iters_per_epoch"]
        self.kappa = cfg["qits"]["kappa_signal"] if control_signals else 0.0
        self.mu = c["consensus"]["mu"] if use_consensus else 0.0
        self.epoch = c["epoch_s"]
        sig = cfg["signal"]
        self.smin, self.smax, self.cycle = sig["split_min"], sig["split_max"], sig["cycle_s"]

    # ------------------------------------------------------------------ setup
    def reset(self, sim):
        g, n = sim.grid, sim.grid.n_nodes
        self.n = n
        self.rng = np.random.default_rng(self.seed + 3_000_017)
        if self._policy is not None:
            self.policy = self._policy
        elif self.policy_kind == "classical":
            self.policy = ClassicalPolicy(seed=self.seed + 4_000_037)
        else:
            self.policy = VQCPolicy.from_config(self.cfg, seed=self.seed + 4_000_037)
        init_scale = (self.cfg["qits"]["classical_init_scale"] if self.policy_kind == "classical"
                      else self.cfg["quantum"]["init_scale"])
        self.theta = (self.theta0.copy() if self.theta0 is not None
                      else self.policy.init_params(n, init_scale))
        net_cls = ClassicalNetwork if self.channel == "classical" else QuantumNetwork
        self.net = net_cls(self.cfg, g, np.random.default_rng(self.seed + 5_000_011))
        self.latency = LatencyModel(self.cfg)
        self.mask = g.valid_mask
        self.sat = self.cfg["network"]["sat_flow_vps"] * self.cfg["network"]["lanes"]
        self.storage = sim.storage
        self.E_cons = np.zeros((n, self.policy.n_qubits))
        self.A = np.zeros((n, n))                     # neighbours with a live channel last epoch
        self.t_ent_acc = np.zeros(n)
        self.p = self.mask / self.mask.sum(1, keepdims=True)
        self.epoch_log = []

    # ------------------------------------------------------------------- step
    def step(self, sim, t):
        self.net.tick(1.0)
        t_ent = self.net.maintain()
        if t_ent.any():
            for col in (0, 1):
                np.maximum.at(self.t_ent_acc, self.net.pairs[:, col], t_ent)
        if t > 0 and t % self.epoch == 0:
            self._decide(sim, t)

    def _consensus_penalty(self, E, reps):
        """μ Σ_{j∈N_Q(i)} ||E_i - E_j||² with E_j = neighbours' consensus states (Eq. 21)."""
        if self.mu == 0.0 or not self.A.any():
            return 0.0
        deg = np.tile(self.A.sum(1), reps)
        AE = np.tile(self.A @ self.E_cons, (reps, 1))
        AE2 = np.tile(self.A @ (self.E_cons ** 2).sum(1), reps)
        return self.mu * (deg * (E ** 2).sum(1) - 2 * (E * AE).sum(1) + AE2)

    def _decide(self, sim, t):
        tic = time.perf_counter()
        n = self.n
        amps = amplitude_encode(sim.devices.x())
        delta = sim.devices.link_delay(sim.ff_steps) / sim.ff_steps
        f = sim.devices.features
        demand = f[:, :, 2] * self.storage + f[:, :, 3] * self.sat * self.cycle
        d_ns = np.maximum(demand[:, N], demand[:, S])
        d_ew = np.maximum(demand[:, E], demand[:, W])

        def J(thetas):
            reps = len(thetas) // n
            p, Ev, Sv = self.policy.forward(np.tile(amps, (reps, 1)), thetas,
                                            np.tile(self.mask, (reps, 1)), self.smin, self.smax)
            cost = local_cost(p, np.tile(delta, (reps, 1)), np.tile(self.mask, (reps, 1)), self.lam)
            cost = cost + self.kappa * signal_cost(Sv, np.tile(d_ns, reps), np.tile(d_ew, reps),
                                                   self.sat, self.cycle)
            return cost + self._consensus_penalty(Ev, reps)

        dtheta = np.zeros(n)
        for _ in range(self.iters):
            new, _, _, _ = spsa_step(J, self.theta, self.eta, self.xi, self.rng)
            dtheta = np.linalg.norm(new - self.theta, axis=1)
            self.theta = new

        t_fwd = time.perf_counter()
        p, Ev, Sv = self.policy.forward(amps, self.theta, self.mask, self.smin, self.smax)
        t_fwd = (time.perf_counter() - t_fwd) / n     # per-node CPU inference (classical policy)
        cost = local_cost(p, delta, self.mask, self.lam)

        hops = np.zeros(n, int)
        self.net.generate_keys(self.epoch)
        if self.use_consensus:
            self.E_cons, info = self.net.exchange(Ev)
            self.A = ((-info["laplacian"]) > 0).astype(float)
            np.fill_diagonal(self.A, 0.0)
            hops = info["hops"]

        # Eq. 6 per node; components are linear, so the mean over nodes is the mean of parts
        quantum_policy = getattr(self.policy, "is_quantum", True)
        lat, parts = self.latency.qits(
            self.t_ent_acc.mean(), hops.mean(),
            n_circuits=1 if quantum_policy else 0, cpu_s=0.0 if quantum_policy else t_fwd,
            quantum_channel=self.net.is_quantum, crypto_s_per_hop=self.net.crypto_s_per_hop)
        self.t_ent_acc[:] = 0.0

        self.p = p
        if self.control_routing:
            sim.route_p = p
        if self.control_signals:
            sim.signals.pending_split = Sv

        self.epoch_log.append({
            "t": t, "cost": float(cost.mean()), "dtheta_norm": float(dtheta.mean()),
            "entropy": routing_entropy(p, self.mask), "split_mean": float(Sv.mean()),
            "F_mean": float(self.net.F.mean()), "active_links": int(self.A.sum() // 2),
            "latency_s": float(lat), **{k: float(v) for k, v in parts.items()},
            "cpu_s": time.perf_counter() - tic,
        })

    # ---------------------------------------------------------------- summary
    def summary(self):
        s, log = self.net.stats, self.epoch_log
        mean = lambda k: float(np.mean([r[k] for r in log])) if log else float("nan")
        return {
            "decision_latency_s": mean("latency_s"),
            "T_ent_s": mean("T_ent"), "T_meas_s": mean("T_meas"),
            "T_class_s": mean("T_class"), "T_proc_s": mean("T_proc"),
            "consensus_entropy": mean("entropy"),
            "final_dtheta": float(np.mean([r["dtheta_norm"] for r in log[-5:]])) if log else float("nan"),
            "mean_fidelity": mean("F_mean") if self.net.is_quantum else float("nan"),
            "pdr": self.net.pdr(),
            "reentanglements": s["reentanglements"],
            "qkd_aborts": s["qkd_aborts"],
            "consensus_bytes": s["consensus_bytes"],
            "qkd_classical_bytes": s["qkd_classical_bytes"],
            "circuit_evals": self.policy.n_circuit_evals,
            "cpu_s_total": float(sum(r["cpu_s"] for r in log)),
        }
