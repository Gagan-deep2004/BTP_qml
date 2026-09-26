"""Classical stand-in for the VQC, for the Q-ITS-NoQOpt ablation (paper Table IV:
"replaces the variational quantum optimizer with a classical SPSA optimizer").

Same interface, same SPSA training and same kinds of outputs as VQCPolicy. As in the VQC,
the signal split has its own output, separate from the routing outputs (the VQC reads
routing from qubits 0,1 and the split from qubit 3):
    z   = W x            W: 5 x 16 = 80 parameters, x = amplitude-encoded features
    p   = softmax(z[0:4]) over valid neighbours                  (routing, like Eq. 8)
    E   = tanh(z[0:4])   4 values in [-1, 1], like <Z_q>         (consensus state)
    S   = split_min + (split_max - split_min) (1 + tanh(z[4])) / 2
"""
import numpy as np


class ClassicalPolicy:
    is_quantum = False

    def __init__(self, n_features=16, n_routes=4, seed=None):
        self.n_features, self.n_qubits = n_features, n_routes     # n_qubits: size of E
        self.n_out = n_routes + 1
        self.n_params = n_features * self.n_out
        self.rng = np.random.default_rng(seed)
        self.n_circuit_evals = 0

    def init_params(self, n, scale=np.pi):
        return self.rng.uniform(-scale, scale, size=(n, self.n_params))

    def forward(self, amps, thetas, valid_mask, split_min=0.2, split_max=0.8):
        W = thetas.reshape(len(thetas), self.n_out, self.n_features)
        z = np.einsum("bof,bf->bo", W, amps)
        zr = z[:, :self.n_qubits]
        logits = np.where(valid_mask > 0, zr, -np.inf)
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        E = np.tanh(zr)
        S = split_min + (split_max - split_min) * (1 + np.tanh(z[:, -1])) / 2
        return p, E, S
