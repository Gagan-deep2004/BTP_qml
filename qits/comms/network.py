"""Quantum communication network between intersections (Sec. III, IV, IV-B, IV-C).

One entangled link Q_ij per pair of adjacent intersections. Per link the network tracks
fidelity F_ij (Werner model with exponential decay), a QKD key pool, and a compromised flag.

    tick(dt)          fidelity decays with link age (+ noise)                    Fig. 1b
    maintain()        re-entangle if F < eps_F, link compromised, or the info gain
                      I(ρ_fresh) - I(ρ_now) > θ_ent                              Eq. 17, Alg. 1 l.12-13
    generate_keys(dt) BB84 on the entangled pairs; abort + drop link if QBER > 11 %  Eq. 13, Sec. IV-B
    routes()          usable channel per pair: direct link if trust >= eps_T, else a 3-hop
                      detour around a grid square via entanglement swapping      Eq. 4, 14, Sec. IV-C
    exchange(E)       OTP-encrypted, key-authenticated exchange of E_i over the routes,
                      then one consensus step on the resulting Laplacian         Eq. 13, 15, 16
"""
import numpy as np

from ..traffic.grid import E as EAST, S as SOUTH
from .entanglement import swap_fidelity, werner_qber, werner_qmi
from .qkd import bb84, from_bits, otp, to_bits
from .trust import consensus_update, laplacian, trust_score


class QuantumNetwork:
    is_quantum = True
    crypto_s_per_hop = 0.0

    def __init__(self, cfg, grid, rng):
        c = cfg["comms"]
        self.cfg, self.grid, self.rng = c, grid, rng
        self.fid, self.re, self.qkd = c["fidelity"], c["reentangle"], c["qkd"]
        self.tr, self.lat = c["trust"], c["latency"]
        self.eta_c = c["consensus"]["eta_c"]

        # undirected links between adjacent intersections
        pairs = [(v, grid.neighbour[v, d]) for v in range(grid.n_nodes) for d in (EAST, SOUTH)
                 if grid.neighbour[v, d] >= 0]
        self.pairs = np.array(pairs)
        self.n_links = len(pairs)
        self.link_of = {}
        for e, (u, v) in enumerate(pairs):
            self.link_of[(u, v)] = self.link_of[(v, u)] = e
        self.length_km = cfg["network"]["link_length_m"] / 1000.0

        self.F = np.full(self.n_links, self.fid["f0"])
        self.age = np.zeros(self.n_links)
        self.compromised = np.zeros(self.n_links, bool)
        self.eve_fraction = np.zeros(self.n_links)          # intercept-resend attacker per link
        self.key_pool = [np.zeros(0, np.uint8) for _ in range(self.n_links)]
        self.i_fresh = werner_qmi(self.fid["f0"])

        self.stats = {"reentanglements": 0, "qkd_aborts": 0, "msgs_sent": 0, "msgs_delivered": 0,
                      "msgs_rerouted": 0, "consensus_bytes": 0.0, "qkd_classical_bytes": 0.0,
                      "key_bits_generated": 0, "key_bits_used": 0, "no_key_drops": 0}
        self.last_qber = np.zeros(self.n_links)

    # ------------------------------------------------------------------ state
    def tick(self, dt=1.0):
        self.age += dt
        f0, tau = self.fid["f0"], self.fid["tau_s"]
        self.F = 0.25 + (f0 - 0.25) * np.exp(-self.age / tau)
        self.F = np.clip(self.F + self.rng.normal(0, self.fid["noise_std"], self.n_links), 0.25, 1.0)

    def trust(self, F=None, hops=1):
        F = self.F if F is None else F
        return trust_score(F, hops * self.length_km, self.tr["alpha"], self.tr["beta_per_km"])

    def maintain(self):
        """Adaptive entanglement scheduling. Returns per-link entanglement time T_ent (s)."""
        gain = self.i_fresh - np.array([werner_qmi(f) for f in self.F])       # Eq. 17
        need = (self.F < self.fid["eps_F"]) | self.compromised | (gain > self.re["theta_ent_bits"])
        t_ent = np.zeros(self.n_links)
        for e in np.flatnonzero(need):
            attempts = self.rng.geometric(self.re["p_success"])
            t_ent[e] = attempts * self.re["attempt_time_s"]
            self.age[e] = 0.0
            self.F[e] = self.fid["f0"]
            self.compromised[e] = False
            self.stats["reentanglements"] += 1
        return t_ent

    def generate_keys(self, dt=1.0):
        n_raw = int(self.qkd["raw_rate_bps"] * dt)
        for e in range(self.n_links):
            if self.compromised[e]:
                continue
            r = bb84(n_raw, float(werner_qber(self.F[e])), self.rng, self.qkd["qber_max"],
                     self.qkd["sample_frac"], self.eve_fraction[e], self.qkd["ec_efficiency"])
            self.last_qber[e] = r["qber_est"]
            self.stats["qkd_classical_bytes"] += r["classical_bytes"]
            if r["aborted"]:
                self.compromised[e] = True            # suspected eavesdropper: drop link
                self.key_pool[e] = np.zeros(0, np.uint8)
                self.stats["qkd_aborts"] += 1
            else:
                pool = np.concatenate([self.key_pool[e], r["key"]])
                self.key_pool[e] = pool[-self.qkd["key_pool_max_bits"]:]
                self.stats["key_bits_generated"] += r["key"].size

    def usable(self):
        return (~self.compromised) & (self.F >= self.fid["eps_F"]) & \
               (self.trust() >= self.tr["eps_T"])

    # ----------------------------------------------------------------- routes
    def routes(self):
        """For each link (i,j): (path of link ids, effective trust) or None if no channel."""
        ok = self.usable()
        out = []
        for e, (i, j) in enumerate(self.pairs):
            if ok[e]:
                out.append(([e], float(self.trust()[e])))
                continue
            best = None
            d = next(k for k in range(4) if self.grid.neighbour[i, k] == j)
            for p in ((d + 1) % 4, (d + 3) % 4):                 # detour around a grid square
                a, b = self.grid.neighbour[i, p], self.grid.neighbour[j, p]
                if a < 0 or b < 0:
                    continue
                path = [self.link_of[(i, a)], self.link_of[(a, b)], self.link_of[(b, j)]]
                if not ok[path].all():
                    continue
                f = swap_fidelity(swap_fidelity(self.F[path[0]], self.F[path[1]]), self.F[path[2]])
                t = float(self.trust(f, hops=3))
                if t >= self.tr["eps_T"] and (best is None or t > best[1]):
                    best = (path, t)
            out.append(best)
        return out

    def _send(self, bits, path):
        """Encrypt + authenticate hop by hop with each link's QKD key; returns received bits or None."""
        need = bits.size + self.qkd["mac_bits"]
        if any(self.key_pool[e].size < need for e in path):
            self.stats["no_key_drops"] += 1
            return None
        for e in path:
            key, self.key_pool[e] = self.key_pool[e][:need], self.key_pool[e][need:]
            self.stats["key_bits_used"] += need
            cipher = otp(bits, key)                          # Eq. 13 on the wire
            self.stats["consensus_bytes"] += self.lat["header_bytes"] + need / 8
            if self.rng.random() < self.lat["classical_loss_per_hop"]:
                return None
            bits = otp(cipher, key)                          # receiver decrypts with same key
        return bits

    # --------------------------------------------------------------- consensus
    def exchange(self, E):
        """Share E_i with quantum neighbours and run one consensus step (Eq. 15-16).
        Returns (E_new, info) where info has the delivered-link Laplacian and per-node hop counts."""
        routes = self.routes()
        pairs, weights = [], []
        max_hops = np.zeros(self.grid.n_nodes, int)
        for e, (i, j) in enumerate(self.pairs):
            r = routes[e]
            self.stats["msgs_sent"] += 2
            if r is None:
                continue
            path, t_ij = r
            got_j = self._send(to_bits(E[i]), path)          # i -> j
            got_i = self._send(to_bits(E[j]), path[::-1])    # j -> i
            delivered = (got_j is not None) + (got_i is not None)
            self.stats["msgs_delivered"] += delivered
            if len(path) > 1:
                self.stats["msgs_rerouted"] += delivered
            if got_j is None or got_i is None:
                continue                                     # use only symmetric exchanges
            assert np.array_equal(from_bits(got_j), E[i].astype(np.float32))
            pairs.append((i, j))
            weights.append(t_ij)
            max_hops[i] = max(max_hops[i], len(path))
            max_hops[j] = max(max_hops[j], len(path))
        L = laplacian(self.grid.n_nodes, pairs, weights)
        return consensus_update(E, L, self.eta_c), {"laplacian": L, "hops": max_hops,
                                                     "n_active": len(pairs)}

    def pdr(self):
        s = self.stats
        return s["msgs_delivered"] / s["msgs_sent"] if s["msgs_sent"] else 1.0


class ClassicalNetwork(QuantumNetwork):
    """Q-ITS-NoQKD ablation: the same neighbour exchange over classical authenticated channels
    (AES-GCM-style: +28 B nonce/tag per message, +crypto time per hop). No entanglement,
    no fidelity decay, no key pool; links are always usable, trust uses F = 1."""

    is_quantum = False

    def __init__(self, cfg, grid, rng):
        super().__init__(cfg, grid, rng)
        cc = cfg["comms"]["classical_crypto"]
        self.overhead_bytes, self.crypto_s_per_hop = cc["overhead_bytes"], cc["per_hop_s"]
        self.F = np.ones(self.n_links)

    def tick(self, dt=1.0):
        pass

    def maintain(self):
        return np.zeros(self.n_links)

    def generate_keys(self, dt=1.0):
        pass

    def usable(self):
        return np.ones(self.n_links, bool)

    def _send(self, bits, path):
        for _ in path:
            self.stats["consensus_bytes"] += self.lat["header_bytes"] + bits.size / 8 + self.overhead_bytes
            if self.rng.random() < self.lat["classical_loss_per_hop"]:
                return None
        return bits
