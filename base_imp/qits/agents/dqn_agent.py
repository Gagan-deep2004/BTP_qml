"""DQN signal-control baseline (paper Sec. V, ref. [18]).

* Q-network: FC [64, 128, 64] + ReLU, Adam lr 1e-3, γ = 0.95, ε-greedy, replay buffer,
  target network. One network is shared by all intersections and trained centrally.
* State of intersection i (19): its 16 consumer-device features (same input as Q-ITS),
  one-hot current phase, time in phase / 60.
* Action, mode "phase" (standard DQN traffic-signal formulation): every 5 s choose which phase
  gets green (NS / EW); a change costs 3 s all-red.
  Action, mode "split" (same action space as Q-ITS and rule-based): at each cycle start choose
  the NS green split from {0.3, 0.4, 0.5, 0.6, 0.7}.
* Reward: -(stop-line queues at i, averaged over the decision interval) / link storage.
* Routing: uniform over shortest paths (the DQN only controls signals, as in the paper).
* Decision latency: centralised — state uplink to a server at the centre intersection,
  batched inference, action downlink (constants in cfg.dqn.latency, A15).
"""
import time

import numpy as np
import torch
from torch import nn

from .base import Controller

SPLIT_ACTIONS = np.array([0.3, 0.4, 0.5, 0.6, 0.7])


def n_actions(mode):
    return 2 if mode == "phase" else len(SPLIT_ACTIONS)


def state_dim(cfg):
    return 16 + 2 + 1


class QNet(nn.Module):
    def __init__(self, n_in, hidden=(64, 128, 64), n_out=2):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, dim, rng):
        self.s = np.zeros((capacity, dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, dim), np.float32)
        self.cap, self.ptr, self.size, self.rng = capacity, 0, 0, rng

    def add_batch(self, s, a, r, s2):
        for i in range(len(s)):
            self.s[self.ptr], self.a[self.ptr], self.r[self.ptr], self.s2[self.ptr] = s[i], a[i], r[i], s2[i]
            self.ptr = (self.ptr + 1) % self.cap
            self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = self.rng.integers(0, self.size, n)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]))


class DQNLearner:
    """Holds the online/target networks, optimiser and replay buffer across episodes."""

    def __init__(self, cfg, seed=0, mode="phase"):
        d = cfg["dqn"]
        torch.manual_seed(seed)
        self.cfg = d
        self.q = QNet(state_dim(cfg), d["hidden"], n_actions(mode))
        self.target = QNet(state_dim(cfg), d["hidden"], n_actions(mode))
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=d["lr"])
        self.buffer = ReplayBuffer(d["buffer"], state_dim(cfg), np.random.default_rng(seed))
        self.steps = 0
        self.update_norms, self.losses = [], []

    def train_step(self):
        if self.buffer.size < self.cfg["batch"] * 10:
            return
        s, a, r, s2 = self.buffer.sample(self.cfg["batch"])
        with torch.no_grad():
            y = r + self.cfg["gamma"] * self.target(s2).max(1).values
        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        before = [p.detach().clone() for p in self.q.parameters()]
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.update_norms.append(float(torch.sqrt(sum(((p.detach() - b) ** 2).sum()
                                                      for p, b in zip(self.q.parameters(), before)))))
        self.losses.append(float(loss.detach()))
        self.steps += 1
        if self.steps % self.cfg["target_update_steps"] == 0:
            self.target.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    @staticmethod
    def load_qnet(cfg, path, mode="phase"):
        q = QNet(state_dim(cfg), cfg["dqn"]["hidden"], n_actions(mode))
        q.load_state_dict(torch.load(path, weights_only=True))
        q.eval()
        return q


class DQNController(Controller):
    def __init__(self, cfg, qnet, learner=None, epsilon=0.0, seed=42, mode="phase"):
        self.cfg, self.qnet, self.learner = cfg, qnet, learner
        self.epsilon, self.seed, self.mode = epsilon, seed, mode
        self.name = "dqn" if mode == "phase" else "dqn_split"
        self.interval = cfg["dqn"]["decision_s"] if mode == "phase" else cfg["signal"]["cycle_s"]

    def reset(self, sim):
        if self.mode == "phase":
            sim.signals.use_phase_mode()
        self.q_acc, self.q_steps = 0.0, 0
        self.rng = np.random.default_rng(self.seed + 6_000_011)
        g = sim.grid
        centre = (g.rows // 2) * g.cols + g.cols // 2
        self.hops = np.array([g.manhattan(v, centre) for v in range(g.n_nodes)])
        self.prev_s = self.prev_a = None
        self.rewards, self.entropies, self.latencies, self.decision_t = [], [], [], []
        self.bytes = 0.0
        self.infer_s = 0.0

    def _state(self, sim):
        sig = sim.signals
        if self.mode == "phase":
            phase = np.eye(2)[sig.phase]
            tip = np.minimum(sig.time_in_phase, 60)[:, None] / 60.0
        else:
            phase = np.stack([sig.split, 1 - sig.split], 1)
            tip = np.zeros((sig.n_nodes, 1))
        return np.hstack([sim.devices.x(), phase, tip]).astype(np.float32)

    def step(self, sim, t):
        self.q_acc = self.q_acc + sim.queue_by_approach().sum(1)
        self.q_steps += 1
        if t == 0 or t % self.interval != 0:
            return
        s = self._state(sim)
        tic = time.perf_counter()
        with torch.no_grad():
            qv = self.qnet(torch.from_numpy(s)).numpy()
        infer = time.perf_counter() - tic
        self.infer_s += infer
        a = qv.argmax(1)
        explore = self.rng.random(len(a)) < self.epsilon
        a[explore] = self.rng.integers(0, qv.shape[1], explore.sum())
        if self.mode == "phase":
            sim.signals.request_phase(a)
        else:
            sim.signals.pending_split = SPLIT_ACTIONS[a]

        r = -self.q_acc / self.q_steps / sim.storage     # mean queue over the interval
        self.q_acc, self.q_steps = 0.0, 0
        self.rewards.append(float(r.mean()))
        z = qv - qv.max(1, keepdims=True)
        pi = np.exp(z) / np.exp(z).sum(1, keepdims=True)                    # softmax(Q), τ = 1
        self.entropies.append(float(-(pi * np.log(pi + 1e-12)).sum(1).mean()))
        self.decision_t.append(t)
        self._account(s.shape[1], len(a), infer)

        if self.learner is not None:
            if self.prev_s is not None:
                self.learner.buffer.add_batch(self.prev_s, self.prev_a, r, s)
            self.learner.train_step()
        self.prev_s, self.prev_a = s, a

    def _account(self, dim, n, infer):
        lat = self.cfg["dqn"]["latency"]
        hdr = self.cfg["comms"]["latency"]["header_bytes"]
        per_node = (2 * lat["access_s"] + 2 * self.hops * lat["backhaul_per_hop_s"]
                    + n * lat["server_per_node_s"] + infer)
        self.latencies.append(float(per_node.mean()))
        up, down = hdr + 4 * dim, hdr + 1
        self.bytes += float(((up + down) * (1 + self.hops)).sum())    # access link + backhaul hops

    def summary(self):
        return {"decision_latency_s": float(np.mean(self.latencies)) if self.latencies else float("nan"),
                "policy_entropy": float(np.mean(self.entropies)) if self.entropies else float("nan"),
                "comm_bytes": self.bytes, "cpu_s_total": self.infer_s}
