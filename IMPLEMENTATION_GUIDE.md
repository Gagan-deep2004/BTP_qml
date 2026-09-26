# Q-ITS Base Paper — Step-by-Step Implementation Guide

**Base paper:** T. Ahmad et al., "Quantum-Enhanced ITS Leveraging Consumer Devices for Real-Time Urban
Traffic Optimization in Smart Cities," *IEEE Trans. Consumer Electronics*, vol. 72, no. 1, Feb. 2026.
DOI: 10.1109/TCE.2026.3650949

**Code availability:** none. No GitHub/Zenodo/Code Ocean link exists and the paper has no data-availability
statement. Everything below is a from-scratch reimplementation.

**Dataset:** the paper uses **no public dataset**. All traffic is *synthetic* (Poisson arrivals on a grid,
random origin–destination pairs), and "consumer device data" is simulated probe data from vehicles.
Section 3 shows how to generate and save this dataset reproducibly, plus an optional real-data extension.

---

## 0. Overview — what you are building

```
            ┌──────────────── Simulation loop (Δt = 1 s, T = 1800 s) ────────────────┐
 demand ──► │ Traffic simulator (grid, links, queues, 2-phase signals, vehicles)      │
 (Poisson)  │        │ probe reports (speed, position) from consumer devices          │
            │        ▼                                                                │
            │ Node feature builder  x_i(t)  (16 features / intersection)             │
            │        ▼                                                                │
            │ Q-ITS agent per node:  amplitude encode (Eq.2) → VQC U(θ_i) (Eq.7)      │
            │        → routing probs p_ij (Eq.8), signal split S_i, expectation E_i   │
            │        → cost C_i (Eq.9) + consensus term (Eq.21) → SPSA update (Eq.10-11)│
            │        ▼                                                                │
            │ Quantum comm layer: fidelity F_ij (Eq.1/12), QKD + OTP (Eq.13),         │
            │   trust T_ij (Eq.14), Laplacian consensus (Eq.15-16), swapping (Eq.4),  │
            │   adaptive re-entanglement (Eq.17), latency model (Eq.6)                │
            └────────────────────────────────────────────────────────────────────────┘
 Baselines: rule-based fixed-time controller, DQN controller.   Ablations: NoQOpt, NoQKD, NoCons.
 Outputs: Fig 1(a–d), Fig 2(a–d), Table IV (ablation), Table V (overall gains).
```

The **security layer (QKD, OTP, fidelity, trust)** is implemented exactly as in the paper and then left
unchanged for your later improvement work, which will focus on the traffic optimization side.

---

## 1. Environment setup

```powershell
# Python 3.11 recommended (qiskit-aer has Windows wheels)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install qiskit qiskit-aer numpy scipy pandas networkx torch matplotlib seaborn tqdm pyyaml pyarrow
pip freeze > requirements.txt
```

Quick check:

```python
import qiskit, qiskit_aer
print(qiskit.__version__, qiskit_aer.__version__)
```

---

## 2. Project structure

```
BTP_qml/
├── configs/
│   └── base.yaml               # every number from Table III + all assumptions (Section 4)
├── qits/
│   ├── traffic/
│   │   ├── grid.py             # grid graph, directed links, neighbour directions N/E/S/W
│   │   ├── demand.py           # Poisson arrivals + OD pairs  (DATASET GENERATOR)
│   │   ├── devices.py          # consumer-device probe reports (penetration, GPS noise)
│   │   ├── signals.py          # 2-phase signal with green split S_i
│   │   └── simulator.py        # mesoscopic queue simulator
│   ├── quantum/
│   │   ├── encoding.py         # Eq. 2 amplitude encoding
│   │   ├── vqc.py              # Eq. 7 ansatz, Eq. 8 probabilities, E_i
│   │   ├── noise.py            # T1/T2 + gate-error noise model
│   │   └── spsa.py             # Eq. 10-11 / 23-24
│   ├── comms/                  # SECURITY / COMMUNICATION LAYER (keep as in paper)
│   │   ├── entanglement.py     # Eq. 1/12 fidelity, decay, Eq. 4 swapping, Eq. 17 info gain
│   │   ├── qkd.py              # BB84 key generation, QBER check, Eq. 13 OTP
│   │   ├── trust.py            # Eq. 14 trust score, Eq. 15 Laplacian
│   │   ├── consensus.py        # Eq. 16 consensus update
│   │   └── latency.py          # Eq. 6 latency model, message/overhead accounting
│   ├── agents/
│   │   ├── qits_agent.py       # Algorithm 1
│   │   ├── rule_based.py       # fixed-time baseline
│   │   └── dqn_agent.py        # DQN baseline [64,128,64]
│   └── metrics.py              # delay, congestion index, entropy (Eq.19-20), PDR, energy, CI (Eq.25)
├── experiments/
│   ├── generate_dataset.py
│   ├── calibrate_density.py
│   ├── train_dqn.py
│   ├── run_main.py             # 3 methods × 3 densities × 30 seeds
│   ├── run_ablation.py         # Table IV
│   └── make_figures.py         # Fig 1, Fig 2, Table V
├── data/                       # generated dataset (Section 3)
└── results/                    # CSV logs, figures
```

---

## 3. Dataset

### 3.1 Primary dataset: synthetic, following the paper (this is what the paper did)

| Item | Paper value | What you generate |
|---|---|---|
| Road network | Grid, 100 m segments, bidirectional | `5×5` grid (main), `50×50` (scaling test) — see note |
| Arrivals | Poisson, λ ∈ [3, 7] veh/s | per-second Poisson counts for 3 density levels |
| Density levels | 30 %, 60 %, 90 % saturation | λ calibrated per grid (Section 3.3) |
| OD pairs | random source–destination per vehicle | uniform random nodes, `origin ≠ dest` |
| Duration | 1800 s, Δt = 1 s | 1800 steps |
| Seeds | 30 runs, seed = 42 | seeds 42 … 71 |

> **Why 5×5 as the main grid?** Table III says "Network size = 25" while the text says 50×50 (2,500
> intersections). λ = 3–7 veh/s is ~30–90 % saturation for a **25-node** grid but would leave a 2,500-node
> grid almost empty. Use 5×5 for the main results and report 50×50 as a scalability run with λ scaled by
> node count. Note this in your report as an inconsistency you resolved.

**Demand file** — `data/demand_{density}_{seed}.csv`

| veh_id | t_depart | origin | dest |
|---|---|---|---|

```python
# qits/traffic/demand.py
import numpy as np, pandas as pd

def generate_demand(n_nodes, lam, T=1800, seed=42, min_hops=2, grid_cols=5):
    rng = np.random.default_rng(seed)
    rows, vid = [], 0
    for t in range(T):
        for _ in range(rng.poisson(lam)):              # Poisson arrivals per second
            while True:
                o, d = rng.integers(0, n_nodes, 2)
                (ro, co), (rd, cd) = divmod(o, grid_cols), divmod(d, grid_cols)
                if abs(ro - rd) + abs(co - cd) >= min_hops:
                    break
            rows.append((vid, t, int(o), int(d))); vid += 1
    return pd.DataFrame(rows, columns=["veh_id", "t_depart", "origin", "dest"])
```

**Consumer-device probe file** (written by the simulator while running) — `data/probe_{density}_{seed}.parquet`

| t | veh_id | link_id | pos_m | speed_mps | has_device |
|---|---|---|---|---|---|

Device model (`qits/traffic/devices.py`):
- Each vehicle carries a device with probability `penetration = 0.6` (configurable; the paper doesn't give a value).
- Reports every 1 s: link id, position with GPS noise `N(0, 3 m)`, speed with noise `N(0, 0.5 m/s)`.
- A vehicle is "queued" if its reported speed is < 1 m/s.

**Node feature dataset** — `data/features_{density}_{seed}.npy`, shape `(T, N_nodes, 16)`

For each intersection *i* and each of its 4 incoming approaches (N, E, S, W), 4 features = **16 features → 4 qubits**:

| # | Feature | From consumer devices |
|---|---|---|
| 1 | density (veh/m) | reported vehicles on link ÷ penetration ÷ link length |
| 2 | mean speed (m/s) | mean of reported speeds |
| 3 | queue length (veh) | reported with speed < 1 m/s ÷ penetration |
| 4 | predicted inflow (veh/s) | EWMA (span 30 s, α = 2/31) of device-reported stop-line arrivals ÷ penetration — the paper's "prediction horizon feature" |

Missing approaches (boundary nodes) are zero-filled. Each feature is normalised to [0, 1] by a fixed physical
scale, so no statistics leak between runs: density ÷ link storage, speed ÷ free-flow speed, queue ÷ link storage,
inflow ÷ saturation flow. Add a small ε before encoding (Eq. 2 is undefined for all-zero vectors).

Link delay δ_ij (resolution A7): `max(mean device-reported travel time on link i→j in the last 60 s,
free-flow time + estimated queue ÷ saturation flow)`.

**Implemented:** `experiments/generate_device_data.py` records `data/devices/{density}/seed_{seed}.npz`
(`x`, `x_true`, `delay` per second) using the rule-based controller, plus a probe parquet for seed 42.
It also reports estimation error against ground truth and against penetration rate (`results/devices/`).
At 60 % penetration the mean absolute error is 0.02–0.09 in normalised units.

### 3.2 Optional real-world extension (useful later for your improvement chapter)

Real camera-derived traffic on grid networks from the Traffic Signal Control open datasets
(https://traffic-signal-control.github.io/):
- **Hangzhou 4×4** (16 intersections) and **Jinan 3×4** (12 intersections): real camera data, `roadnet.json` + `flow.json`.
- **Manhattan 16×3 / 28×7**: larger real networks.

The flow files give per-vehicle `startTime` and `route`, so they can replace `demand.csv` directly. Use these
after the base reproduction works; the base paper itself used only synthetic data.

### 3.3 Calibrating the density levels

"Saturation" is not defined in the paper. Define it empirically:

1. Run the **rule-based** controller for λ = 1, 2, …, 12 veh/s (5×5 grid, 1 seed).
2. λ_max = the largest λ where the number of vehicles in the network stays bounded (does not grow
   linearly until the end of the run).
3. Low / Medium / High = 0.3·λ_max, 0.6·λ_max, 0.9·λ_max.

Expect λ_max ≈ 7–8 veh/s for 5×5, which puts the levels close to the paper's [3, 7] range. Save the values to `configs/base.yaml`.

---

## 4. Ambiguities in the paper and the resolution to use

The paper is underspecified in many places. Put this table in your report so your examiners can see which
choices are yours.

| # | Paper says | Problem | Resolution in this implementation |
|---|---|---|---|
| A1 | 50×50 grid (text) vs 25 (Table III) | inconsistent | 5×5 main, 50×50 scaling run |
| A2 | 30 runs (text) vs 50 trials (Table III) | inconsistent | 30 seeds (42–71) |
| A3 | DQN 500 episodes vs 5000 episodes | inconsistent | 500 episodes × 600 s (report it) |
| A4 | Eq. 3 cost vs Eq. 9 cost | two different costs | Eq. 9/22 (the one used in Algorithm 1) |
| A5 | Eq. 5: `S_i(t+1) = S_i(t) − η∇θ C_i` | signal state minus a θ-gradient; signal not modelled | S_i = green split of NS phase, read from ⟨Z₃⟩ of the VQC (Section 6.3); optimised through θ |
| A6 | Routing by sampling p_ij | ignores vehicle destinations | vehicle chooses among neighbours that reduce distance to its destination, weighted by p_ij |
| A7 | δ_ij(t) "delay on edge" | not defined | mean travel time on link (i→j) reported by devices in the last 60 s + queue ÷ sat-flow |
| A8 | λ used for Poisson rate *and* variance weight | symbol clash | `lam_arrival` vs `lam_var` = 1.0 (tune 0.1–5) |
| A9 | θ used for circuit params *and* Eq. 17 threshold | symbol clash | `theta_ent` = 0.1 bits |
| A10 | ε used for fidelity *and* trust threshold | symbol clash | `eps_F` = 0.85, `eps_T` = 0.8 |
| A11 | μ = "consensus step size" (Table III) vs weight in Eq. 21 | two roles | μ = 0.01 as Eq. 21 weight; consensus step `eta_c` = 0.2 |
| A12 | "QN-SPSA" | Eq. 10 is plain SPSA | implement Eq. 10 exactly (plain SPSA) |
| A13 | Fidelity: avg > 0.90 (text), Fig 1b drops to 0.4 | contradictory | Fig 1b = decay without re-entanglement; with the 0.85 threshold you get a sawtooth that stays ≥ 0.85 |
| A14 | Eq. 17 H(·) "quantum entropy" of E_i | E_i are classical numbers | use quantum mutual information of the shared pair ρ_ij (Section 7.5) |
| A15 | Energy, latency, PDR | no models given | explicit parameterised models (Sections 7.6, 9); all constants go in the config |
| A16 | Eq. 7 uses the same θ for RY and RZ | probably a typo | separate parameters: 2·n_q·d = 64 parameters |
| A17 | SPSA perturbation ξ = 0.01 with 8192 shots on noisy NISQ | shot noise (~0.011) swamps a 0.02 finite difference; the noisy optimisation does not converge (tested) | ξ = 0.1 for noisy runs; `experiments/single_node_convergence.py` shows ξ = 0.01 stalls at cost 1.85 while ξ = 0.1 reaches 1.42 (optimum 1.38) |
| A19 | DQN action space not specified | a DQN that picks phases every 5 s has a far stronger actuator than Q-ITS's per-cycle split, which confounds the comparison | two DQN baselines: `dqn` (phase, standard) and `dqn_split` (same action space as Q-ITS) |
| A20 | Energy model: no power values given | the result is set entirely by assumed QPU power | busy-time model (`qits/energy.py`, `cfg.energy`) with P_qpu = 25 kW, plus a sensitivity sweep and break-even power |
| A21 | NoQOpt: "classical SPSA optimizer" undefined | the classical stand-in must be a fair model | linear policy (5×16 weights, separate split output), same SPSA and cost, weights initialised in ±0.1. Initialising it in the VQC's ±π saturated its outputs and made it look 8 % *worse* than rule-based |
| A18 | Noisy simulation of `initialize` state loading | Aer's density-matrix method does not support `initialize` | state loading treated as ideal (SetDensityMatrix); NISQ noise on every ansatz gate |

---

## 5. Step 1 — Traffic simulator (`qits/traffic/`)

A mesoscopic queue-based simulator is enough and runs quickly.

**Network (`grid.py`)**
- Nodes: `R×C` grid, id = `r*C + c`. Directions N = (−1,0), E = (0,+1), S = (+1,0), W = (0,−1).
- Directed links for each adjacent pair (both directions); length L = 100 m, 2 lanes per direction.
  With 1 lane, λ_max was only 3 veh/s, below the paper's λ = [3, 7]; 2 lanes gives λ_max = 6.5.
- Free-flow speed 13.89 m/s (50 km/h), so free-flow time ≈ 7.2 s (8 simulation steps).
- Storage capacity = 2 × 100 / 7.5 ≈ 26 veh; saturation flow = 0.5 veh/s per lane.

**Signal (`signals.py`)**
- Two phases: NS (serves incoming links from N and S) and EW.
- Cycle C = 60 s, green split `S_i ∈ [0.2, 0.8]` for NS, with 3 s lost time per phase.

**Simulation step (`simulator.py`)**, every Δt = 1 s:
```
1. spawn:     vehicles with t_depart == t enter a queue at their origin node (the "entry" approach)
2. travel:    vehicles on a link whose (entry_time + free_flow_time) <= t join the link's downstream queue
3. discharge: for each node, for each green approach, move up to sat_flow*Δt vehicles from the
              queue head: if node == dest -> exit (record travel time); else
              next = routing_policy(vehicle, node); move if next link has storage space (spillback)
4. signals:   advance the phase timers; at the start of each cycle apply the new split S_i
5. devices:   write probe reports; update node features x_i(t)
6. metrics:   log per-step queues, #vehicles in network, link occupancies
```

**Routing policy interface** (used by every controller):
```python
def productive_neighbours(node, dest, grid):   # neighbours that reduce Manhattan distance (1 or 2)
    ...
def route(vehicle, node, p_node, rng):         # p_node: length-4 array over N,E,S,W
    cand = productive_neighbours(node, vehicle.dest, grid)
    w = np.array([p_node[d] for d in cand]) + 1e-9
    return cand[rng.choice(len(cand), p=w / w.sum())]
```
Baselines pass a uniform `p_node`, which gives random minimal routing.

**Vehicle delay** = (arrival time − depart time) − (minimum hops × free-flow time).

✅ **Checkpoint:** with the rule-based controller at low λ, average delay should be small (a few seconds per
signal). At λ > λ_max the number of vehicles in the network grows without bound.

---

## 6. Step 2 — Quantum policy (`qits/quantum/`)

### 6.1 Amplitude encoding — Eq. 2

```python
# qits/quantum/encoding.py
import numpy as np

def amplitude_encode(x, eps=1e-6):
    """Eq. 2: |ψ> = Σ_k sqrt(T^k / ||T||_1) |k>. len(x) must be 2**n_q (16 -> 4 qubits)."""
    x = np.clip(np.asarray(x, float), 0.0, None) + eps
    return np.sqrt(x / x.sum())            # unit L2 norm automatically
```

### 6.2 Variational circuit — Eq. 7

```python
# qits/quantum/vqc.py
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def build_ansatz(n_q=4, depth=8):
    th = ParameterVector("θ", 2 * n_q * depth)
    qc = QuantumCircuit(n_q)
    k = 0
    for _ in range(depth):
        for q in range(n_q):
            qc.ry(th[k], q); qc.rz(th[k + 1], q); k += 2
        for q in range(n_q):                 # ring connectivity E_CX
            qc.cx(q, (q + 1) % n_q)
    return qc, th

def full_circuit(amp, ansatz, n_q=4):
    qc = QuantumCircuit(n_q)
    qc.initialize(amp, range(n_q))           # state loading |ψ_i(t)>
    qc.compose(ansatz, inplace=True)
    return qc
```

### 6.3 Reading the outputs — Eq. 8, E_i and S_i

Qubit roles (n_q = 4):
- **Qubits 0, 1:** measured outcome `j ∈ {0,1,2,3}` maps to neighbour {N, E, S, W}, so p_ij = |⟨j|U|ψ⟩|² (Eq. 8).
- **Qubit 3:** signal split `S_i = 0.2 + 0.6 · (1 + ⟨Z₃⟩)/2` (resolution A5).
- **E_i** = vector of ⟨Z_q⟩ for q = 0..3. This is the "variational expectation value" exchanged in consensus.

**Exact (fast, for development/training):**
```python
from qiskit.quantum_info import Statevector, Pauli

def evaluate_exact(amp, ansatz, params, valid_mask):
    sv = Statevector(amp).evolve(ansatz.assign_parameters(params))
    p = sv.probabilities([0, 1])                         # index j = b0 + 2*b1
    p = p * valid_mask; p = p / p.sum()                  # mask boundary neighbours
    E = np.array([sv.expectation_value(Pauli("I"*(3-q) + "Z" + "I"*q)).real for q in range(4)])
    S = 0.2 + 0.6 * (1 + E[3]) / 2
    return p, E, S
```

**Noisy with shots (as in the paper), batched across all nodes:**
```python
# qits/quantum/noise.py
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

def make_noise_model(T1=85e-6, T2=90e-6, p1=2.5e-4, p2=1.3e-3, t1q=35e-9, t2q=300e-9):
    nm = NoiseModel()
    e1 = depolarizing_error(p1, 1).compose(thermal_relaxation_error(T1, T2, t1q))
    tr = thermal_relaxation_error(T1, T2, t2q)
    e2 = depolarizing_error(p2, 2).compose(tr.expand(tr))
    nm.add_all_qubit_quantum_error(e1, ["ry", "rz", "u", "sx", "x"])
    nm.add_all_qubit_quantum_error(e2, ["cx"])
    return nm

sim = AerSimulator(noise_model=make_noise_model())
# circuits = [full_circuit(...).measure_all() for every node and every ±perturbation]
# tc = transpile(circuits, sim, basis_gates=["ry","rz","cx","u","sx","x"])
# result = sim.run(tc, shots=8192).result()      # ONE batched call per decision step
```
From the counts (the bitstring is little-endian, so bit q is `key[-1-q]`) compute the marginal over qubits 0 and 1 → p_ij, and
⟨Z_q⟩ = P(bit_q = 0) − P(bit_q = 1).

### 6.4 Local cost — Eq. 9 / 22

```python
def local_cost(p, delta, lam_var=1.0):
    """C_i = Σ_j p_ij δ_ij + λ Var(p_ij). δ normalised by free-flow time."""
    return float(p @ delta + lam_var * np.var(p))
```
Signal term (needed so that S_i is actually optimised; this is resolution A5, so label it as your interpretation):
`C_i += kappa * (q_NS / S_i + q_EW / (1 - S_i)) / (sat_flow * cycle)`, with `kappa = 1`.

### 6.5 SPSA — Eq. 10–11 / 23–24

```python
# qits/quantum/spsa.py
def spsa_step(J, theta, eta=0.05, xi=0.01, rng=None):
    delta = rng.choice([-1.0, 1.0], size=theta.shape)          # Δ_i ∈ {±1}^d
    g = (J(theta + xi * delta) - J(theta - xi * delta)) / (2 * xi) * delta   # Eq. 10
    return theta - eta * g, g                                  # Eq. 11
```
Here `J(θ_i) = C_i(θ_i) + μ · Σ_{j∈N_Q(i)} ||E_i(θ_i) − E_j^(t)||²`, the local part of Eq. 21, with μ = 0.01.

> ⚠️ With ξ = 0.01 and 8192 shots, shot noise (~0.011) is similar in size to the finite difference, so
> gradients are mostly noise. **Tested:** the noisy backend with ξ = 0.01 stalls, and with ξ = 0.1 it converges like
> the exact backend (resolution A17).

**Implemented backends** (`qits/quantum/vqc.py`):

| Backend | What it simulates | Speed |
|---|---|---|
| `exact` | NumPy statevector, checked against Qiskit `Statevector` | 0.04 ms/circuit |
| `shots` | exact + multinomial sampling of 8192 shots | 0.04 ms/circuit |
| `noisy` | NumPy density matrix with the paper's T1/T2 + gate-error model, then 8192 shots; matches Aer to 1e-9 | 2 ms/circuit |
| `aer` | the same noisy simulation run through Qiskit Aer (reference) | 27 ms/circuit |

✅ **Checkpoint (done):** `python -m experiments.single_node_convergence` uses a real snapshot (centre node,
δ/ff = [1.83, 2.69, 1.36, 4.78]). Within 300 iterations about 95 % of routing probability moves to the lowest-delay
link (S), and ||Δθ|| falls to about 0.2 of its initial value (`results/quantum/fig1c_convergence.png`).

---

## 7. Step 3 — Quantum communication / security layer (`qits/comms/`)

Implement it as in the paper and then leave it unchanged.

### 7.1 Fidelity — Eq. 1 / 12

Model each link's shared pair as a Werner state
`ρ = F|Φ+⟩⟨Φ+| + (1−F)/3 (I − |Φ+⟩⟨Φ+|)`.

```python
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
PHI = Statevector(np.array([1, 0, 0, 1]) / np.sqrt(2))

def werner(F):
    P = DensityMatrix(PHI).data
    return DensityMatrix(F * P + (1 - F) / 3 * (np.eye(4) - P))

state_fidelity(werner(0.9), PHI)   # Eq. 1 -> 0.9
```

### 7.2 Fidelity decay and re-entanglement (Fig. 1b)

`F(t) = 0.25 + (F0 − 0.25)·exp(−age/τ) + N(0, 0.005)`, with F0 = 0.95 and τ ≈ 1200 s.
- **Without re-entanglement** this reproduces Fig. 1(b), which falls from 0.95 to about 0.4 over 1800 s.
- **With threshold** `eps_F = 0.85`, when F < 0.85 you either re-entangle (age = 0, add T_ent latency) or reroute via
  the neighbour with the higher trust score. This produces the sawtooth that keeps the average ≥ 0.90 (paper text).

### 7.3 Entanglement swapping — Eq. 4

For Werner states: `F_ik = F_ij·F_jk + (1 − F_ij)(1 − F_jk)/3`.
Optional: check it with a 4-qubit Qiskit circuit (Bell measurement on the middle node + X/Z corrections).

### 7.4 QKD + one-time pad — Eq. 13 (security, keep as in the paper)

```python
# qits/comms/qkd.py
def h2(p): return 0.0 if p <= 0 or p >= 1 else -p*np.log2(p) - (1-p)*np.log2(1-p)

def bb84(n_raw, F, rng, qber_max=0.11, sample_frac=0.1):
    qber = 2 * (1 - F) / 3                        # Werner state: Z-basis error rate
    a_bits  = rng.integers(0, 2, n_raw)
    sift    = rng.integers(0, 2, n_raw) == rng.integers(0, 2, n_raw)   # bases match
    ka = a_bits[sift]
    kb = ka ^ (rng.random(ka.size) < qber)
    test = rng.random(ka.size) < sample_frac
    est = float((ka[test] != kb[test]).mean()) if test.any() else 0.0
    if est > qber_max:
        return None, est                          # abort, drop/re-entangle link
    n_final = int((~test).sum() * max(0.0, 1 - 2 * h2(est)))   # EC + privacy amplification (abstracted)
    return ka[~test][:n_final], est

def otp(msg_bits, key_bits):                      # Eq. 13: M = R XOR K
    return msg_bits ^ key_bits[:msg_bits.size]
```
Note that F = 0.85 gives QBER = 10 %, just under the paper's 11 % limit, so the two thresholds agree.
Security bound: P_undetected ≤ (3/4)^n. Log it for n = 128.

Each node keeps a **key pool per link**. Every consensus message consumes key bits equal to its size, and
when the pool is empty the node waits for more QKD, which adds latency.

### 7.5 Trust, Laplacian, consensus, adaptive entanglement — Eq. 14–17

```python
T_ij = alpha * F_ij - beta * L_ij_km          # Eq. 14, alpha=1, beta=0.5 ; drop link if T_ij < eps_T
# Eq. 15: L_Q[i,j] = -T_ij (i≠j, linked), L_Q[i,i] = Σ_k T_ik, else 0   (build with numpy)
# Eq. 16: E_i <- E_i - eta_c * Σ_j T_ij (E_i - E_j)                     (eta_c = 0.2, stable if eta_c*deg*max(T) < 1)
```
Eq. 17 (resolution A14): take the quantum mutual information of the shared pair
`I(ρ) = S(ρ_A) + S(ρ_B) − S(ρ_AB)`. For a Werner state S(ρ_A) = S(ρ_B) = 1:
```python
def werner_qmi(F):
    ev = np.array([F] + [(1 - F) / 3] * 3); ev = ev[ev > 0]
    return 2 + float(np.sum(ev * np.log2(ev)))
# re-establish link if werner_qmi(F0) - werner_qmi(F_ij) > theta_ent
```

### 7.6 Latency — Eq. 6 and communication overhead

`T_total = T_ent + T_meas + T_class + T_proc`. Put every constant in the config:

| Term | Q-ITS | Centralised DQN baseline |
|---|---|---|
| T_ent | 0 if link fresh, else Geometric(p=0.5) × 1 ms | — |
| T_meas | 10 µs | — |
| T_class | neighbour hop: 100 m / (2·10⁸ m/s) + 1 ms stack | uplink to central server: 10 ms + hops × 2 ms, same downlink |
| T_proc | modelled QPU time: shots × 5 µs, plus CPU time | inference time + queueing at the central server |

**Communication overhead:** count bytes. Q-ITS sends 4 floats (E_i) plus a MAC to each quantum neighbour. The DQN
sends the full 16-feature state to the centre and receives an action back. Rule-based sends nothing (report as N/A).

**Packet delivery rate (PDR):** a message counts as delivered unless (a) the link has T_ij < eps_T with no alternative
route, (b) QKD aborted, or (c) classical loss with `p_loss = 0.01 × hops`.

### 7.7 Implemented (`qits/comms/`, `experiments/comms_demo.py`)

- `QuantumNetwork` holds 40 links (one per adjacent intersection pair on the 5×5 grid) and tracks fidelity,
  age, key pool and a compromised flag for each.
- Fidelity is checked every second. Keys are generated and consensus runs every 10 s epoch.
- Re-entanglement is triggered by any of: F < 0.85, a QBER abort, or an information gain above
  θ_ent = 0.35 bits (Eq. 17), which fires at about F = 0.88.
- Rerouting (Sec. IV-C) uses a 3-hop detour around a grid square with double entanglement swapping.
  The detour almost never passes the trust threshold, because three swaps plus 0.3 km of length cost about
  0.2 in trust. A dropped link is therefore simply left out of the Laplacian for that epoch, and the nodes keep
  their cached consensus state.
- Swapping is checked against an explicit Bell-measurement circuit (`swap_via_circuit`). The fidelity
  threshold of 0.85 corresponds to a QBER of 10 %, consistent with the paper's 11 % abort limit.

**Results over 1800 s:**

| Scenario | Mean F | Min F | Re-entanglements | PDR |
|---|---|---|---|---|
| No re-entanglement | 0.61 (decays to 0.40) | 0.38 | 0 | 10 % |
| Adaptive re-entanglement | 0.918 | 0.88 | 607 | 98.9 % |

- The adaptive run's PDR of 98.9 % is close to the paper's 98.2 %.
- **Intercept-resend eavesdropper:** full interception raises QBER to about 26 % and is detected in 60 of 60
  epochs. 30 % interception raises QBER to about 11 % and is detected in only 23 of 60 epochs. The threshold test
  misses low-rate attacks; privacy amplification with 1 − 2h(Q) still accounts for the leaked information.
- **Overhead finding:** QKD classical post-processing (basis sifting, QBER sample, error-correction syndrome) is
  about 41 MB per run, against 0.63 MB of consensus messages. Report this when comparing communication overhead.

---

## 8. Step 4 — Q-ITS agent: Algorithm 1 (`qits/agents/qits_agent.py`)

Decision epoch every 10 s, with 2 SPSA iterations per epoch (about 360 iterations per run, matching Fig. 1c's 300).

```
for each epoch t:
    for each node i (batched):
        x_i  <- features from consumer devices            # Alg.1 line 4
        amp  <- amplitude_encode(x_i)                      # Eq. 2
        p_i, E_i, S_i <- VQC(amp, θ_i)                     # Eq. 8; line 5
        δ_i  <- device-reported link delays                # A7
        repeat 2:
            θ_i, g <- spsa_step(J_i, θ_i)                  # lines 6-7, Eq. 10-11
    for each quantum link (i,j) with F_ij > eps_F:         # line 8
        exchange E_i, E_j using OTP with QKD keys          # line 9, Eq. 13
    E <- consensus update                                  # line 11, Eq. 16
    for each link: if info gain > theta_ent: re-entangle   # lines 12-13, Eq. 17
    push p_i to simulator routing; push S_i to signal (applied at next cycle start)
    log: θ-norm change, F_ij, entropy ΔH (Eq. 19), latency (Eq. 6), bytes, key usage
```

Fallback from Section IV-C: if every quantum link of node i is down, keep the cached `E_i^last` and continue with the last p_i and S_i.

### 8.1 Implemented (`qits/agents/qits_agent.py`, `experiments/run_qits.py`)

- All 25 nodes are batched, so each epoch costs one circuit batch per SPSA evaluation:
  2 iterations × 2 × 25 plus 25 for the forward pass, which is 125 circuits.
- The SPSA objective is `C_i + κ·D_sig(S_i) + μ Σ_j ||E_i(θ_i) − E_j||²`, where E_j are the neighbours' consensus
  states from the previous epoch.
- Signal demand for D_sig is (device queue + device inflow × cycle) on the critical approach of each phase.
- **Pre-training:** 2 episodes on demand seed 1000 (medium density, outside the test seeds), then the test runs
  start from that θ and keep learning online. This mirrors the DQN being trained before evaluation.
- About 55 s per run on the noisy backend; 6 parallel workers (16 workers ran out of RAM).

**First results** (noisy backend, seeds 42–46, paired against the same demand files):

| Density | Rule-based | Fixed-time | Q-ITS | vs rule-based |
|---|---|---|---|---|
| Low | 26.03 s | 26.33 s | 24.35 s | −6.5 % (p = 4e-4) |
| Medium | 27.59 s | 27.96 s | 24.78 s | −10.2 % (p = 9e-5) |
| High | 31.15 s | 30.97 s | 27.16 s | −12.8 % (p = 1e-4) |

- **Diagnostic** (not in the paper): routing only gives 24.9 / 25.9 / 28.2 s, and signals only gives
  26.2 / 27.7 / 31.3 s. The learned routing produces the gain. With uniform demand, split control adds little on its
  own but helps in combination.
- **Decision latency is about 100 ms**, dominated by T_proc = 8192 shots × 12 µs. T_ent adds under 3 ms, and
  T_class about 1 ms per hop.

---

## 9. Step 5 — Baselines (`qits/agents/`)

**Rule-based (`rule_based.py`):** fixed cycle of 60 s. At each cycle start the NS share r of demand
(arrivals in the previous cycle + current queues) sets the split: 0.65 if r > 0.65, 0.35 if r < 0.35, else 0.5.
Routing is uniform minimal. Queues alone are biased at cycle start, because the phase that just ended has been
served; an early version that used only queues performed worse than plain fixed-time.

**DQN (`dqn_agent.py`)**, matching the paper's hyper-parameters:
- MLP [64, 128, 64] with ReLU, Adam lr = 1e-3, γ = 0.95, ε-greedy from 1.0 to 0.05.
- Replay buffer 100k, batch 64, Huber loss, target network update every 500 gradient steps.
- Parameter-shared across all intersections and trained centrally. The paper says "centralized", but one
  joint action space over 25 nodes is intractable, so use parameter sharing.
- State per node (19 dims): the **same 16 consumer-device features Q-ITS uses**, plus a 2-dim phase (or split)
  encoding and time in phase. This gives both methods the same information.
- Reward: −(mean stop-line queue at the node over the decision interval) ÷ link storage.
- Training: 500 episodes × 600 s (resolution A3). Each episode draws a random density and a fresh demand
  seed (2000+, outside the test seeds). Evaluation is greedy (ε = 0) on the 30 test seeds.
  Routing is uniform minimal, since the DQN only controls signals.
- Its policy entropy (for Fig. 2b) is the entropy of softmax(Q/τ) with τ = 1.
- Two variants (resolution A19):

| Variant | Action | Why |
|---|---|---|
| `dqn` | every 5 s choose the green phase (NS/EW); a switch costs 3 s all-red | the standard DQN traffic-signal formulation |
| `dqn_split` | at each 60 s cycle start choose the split from {0.3 … 0.7} | the same action space as Q-ITS and rule-based, so it isolates the effect of the learning method |

- **Results (30 seeds, mean delay ± 95 % CI):**

| Controller | Low | Medium | High | Latency | Comm (consensus / state msgs) |
|---|---|---|---|---|---|
| Fixed-time | 26.17 ± 0.15 | 27.87 ± 0.11 | 30.98 ± 0.14 | – | – |
| Rule-based | 25.74 ± 0.17 | 27.43 ± 0.13 | 31.09 ± 0.19 | – | – |
| DQN-split | 26.18 ± 0.19 | 28.81 ± 0.15 | 32.47 ± 0.18 | 35 ms | 288 kB |
| **DQN (phase)** | **12.89 ± 0.12** | **15.22 ± 0.09** | **19.59 ± 0.14** | 35 ms | 3.6 MB |
| Q-ITS (noisy) | 24.05 ± 0.13 | 24.95 ± 0.11 | 26.97 ± 0.14 | 100 ms | 0.63 MB (+ 40.5 MB QKD) |

- Q-ITS vs rule-based: −6.6 / −9.0 / −13.2 % delay. Vs DQN-split: −8.1 / −13.4 / −16.9 %.
  All paired t-tests give p < 1e-18.
- **The standard phase DQN beats Q-ITS by 38–87 %.** Acyclic 5 s phase control is a far stronger
  actuator than one split per 60 s cycle. The paper's DQN (34 / 48 / 67 s) must have been much weaker than a
  standard DQN in a comparable setting. With the same action space (DQN-split), Q-ITS wins, and its advantage
  comes from the learned routing (Section 8.1).
- Q-ITS decision latency (100 ms, dominated by 8192 shots) is about 3× the centralised DQN's (35 ms). This is the
  opposite of the paper's claimed 42 % latency reduction.
- This points straight at the improvement phase: Q-ITS's signal actuation (one split per cycle) is its weakest part.
- **Latency model (centralised):** 2 × 10 ms RSU access, plus 2 ms per backhaul hop to a server at the centre
  intersection, plus 0.2 ms per intersection of server handling, plus measured inference time.
  **Comm overhead:** state upload (20 B header + 19 × 4 B) and action download (21 B), counted over every hop.

**Energy (Fig. 2d, Table V):** `E = P_cpu·t_cpu + P_qpu·t_qpu`. Measure t_cpu as wall-clock time, and model t_qpu as
shots × circuit time. P_cpu = 65 W and P_qpu are assumptions, and the result depends entirely on P_qpu, so state that clearly.

---

## 10. Step 6 — Metrics (`qits/metrics.py`)

| Metric | Definition |
|---|---|
| Average vehicle delay | mean over completed trips of (travel time − free-flow time) |
| Congestion index | mean over links and time of occupancy ÷ storage capacity |
| Decision latency | mean T_total per decision (Eq. 6) |
| Communication overhead | total bytes sent per decision epoch |
| PDR | delivered ÷ sent messages |
| Consensus entropy ΔH | Eq. 19 on the normalised routing weights p_ij after consensus, averaged over nodes |
| Entropic stability gain | Eq. 20: (ΔH_base − ΔH_QITS) / ΔH_base × 100 |
| Convergence | iterations until ||Δθ|| < 0.05 (compare Q-ITS with DQN weight-update norm) |
| Fidelity | mean F_ij over time |
| Energy | Section 9 |
| Statistics | mean ± std over 30 seeds, CI₉₅ = M̄ ± 1.96·σ/√30 (Eq. 25), paired t-test `scipy.stats.ttest_rel` |

---

## 11. Step 7 — Experiments to reproduce each figure and table

| Output | Experiment |
|---|---|
| **Fig 1(a)** delay vs density | 3 methods × {30, 60, 90 %} × 30 seeds, mean delay |
| **Fig 1(b)** fidelity vs time | one link, 1800 s, **without** re-entanglement (plus the version with it) |
| **Fig 1(c)** ‖Δθ‖ vs iteration | Q-ITS, one node, first 300 SPSA iterations |
| **Fig 1(d)** consensus entropy vs time | Q-ITS, mean ΔH over nodes |
| **Fig 2(a)** delay vs time | 3 methods, high density, moving average |
| **Fig 2(b)** policy entropy vs time | 3 methods |
| **Fig 2(c)** update norm | Q-ITS vs DQN (rule-based stays at 0) |
| **Fig 2(d)** energy | bar chart, 3 methods |
| **Table IV** ablation | Full, NoQOpt, NoQKD, NoCons × 30 seeds |
| **Table V** overall | baseline avg, Q-ITS avg, improvement % for delay, energy, entropy |

**Ablation definitions:**
- **NoQOpt:** replace the VQC with a classical softmax-linear policy (64 parameters, same SPSA). Keep QKD and consensus.
- **NoQKD:** classical authenticated channel (AES-GCM overhead of +28 B/msg, +0.5 ms). No fidelity-based drops. Keep VQC and consensus.
- **NoCons:** μ = 0 and no Eq. 16 update. Keep VQC and QKD.

**Paper's reported numbers, for reference:**
- Fig 1(a) delays (s), Q-ITS / DQN / Rule:
  - 30 %: 28 / 34 / 39
  - 60 %: 41 / 48 / 56
  - 90 %: 57 / 67 / 73
- Table IV delay reduction: Full 22.3 %, NoQOpt 15.8 %, NoQKD 18.7 %, NoCons 19.2 %.

You are **not** expected to match these exactly. The paper is internally inconsistent: Table V says 42.1 % delay
reduction, Table IV 22.3 %, and the text 28 %. Report honestly what your implementation gives. The gaps are useful
material for your later improvement chapter.

---

### 11.1 Implemented (week 7): `python -m experiments.make_figures`

This builds every figure and table from saved results into `results/figures/`:
`fig1.png`, `fig2.png`, `fig_energy_sensitivity.png`, `table_iv_ablation`, `table_v_overall`,
`significance_delay`, `headline_vs_paper` and `energy_breakdown` (each table as .csv and .md).

**Table IV (30 seeds × 3 densities; delay reduction vs rule-based, latency vs centralised DQN):**

| Variant | Paper delay red. | Ours delay red. | Delay low / med / high (s) | Latency |
|---|---|---|---|---|
| Q-ITS (Full) | 22.3 ± 1.4 % | 9.6 ± 0.9 % | 24.05 / 24.95 / 26.97 | 100 ms |
| NoQOpt (classical policy) | 15.8 ± 1.6 % | **22.5 ± 0.8 %** | 20.21 / 21.53 / 23.46 | 1.5 ms |
| NoQKD | 18.7 ± 1.2 % | 9.7 ± 0.9 % | 24.06 / 24.92 / 26.91 | 100 ms |
| NoCons | 19.2 ± 1.5 % | 9.7 ± 0.9 % | 23.87 / 25.05 / 27.00 | 99 ms |

- **The quantum optimiser does not help; a classical policy of the same kind does better.**
  - With identical features, cost, SPSA training and pre-training, a 5×16 linear softmax policy reduces delay
    2.3× more than the VQC, at 1/65 of the latency.
  - Diagnostic (5 seeds, high density): the classical policy's result depends strongly on its initialisation.
    With weights in ±π it saturates (37.3 s); with ±0.1 it gets 23.5 s. Regularising the ±π version to the VQC's
    routing entropy (λ = 10) does not rescue it (40.8 s), so saturation, not herding, was the problem.
- **QKD and consensus have no measurable effect on traffic** (differences are within the CI).
  - With μ = 0.01 (paper), the consensus term is about 1 % of the cost.
  - QKD only secures messages; it never changes a routing or signal decision.
  - The paper's graded ablation (15.8–19.2 %) cannot arise from the mechanism it describes.
- **Energy:** QPU busy time is 2200 s per run (8192 shots × 12 µs × 22 375 circuits, summed over 25 node QPUs).
  - At 25 kW that is 15.5 kWh per run, against about 8 J for the DQN.
  - Q-ITS computation would match the DQN only below about 3.5 mW of QPU power
    (`fig_energy_sensitivity.png`).
- **Entropic stability gain (Eq. 20):** 15.9 % vs rule-based uniform routing; the paper reports 61.3 %.
- **Fig. 2b:** the DQN's softmax(Q) entropy at τ = 1 is near-uniform because the gaps between its Q-values are
  small. It acts greedily, so its "policy entropy" is not comparable with Q-ITS's routing entropy. The figure plots
  entropies relative to a uniform policy and states this caveat.

## 12. Compute budget and speed tips

- 5×5 grid × 180 epochs × 2 SPSA iterations × 3 circuit evaluations ≈ 27k circuits per run. This is fast with the
  exact statevector (about a minute).
- Noisy runs at 8192 shots: batch every node's circuits into **one** `sim.run([...])` call per epoch.
  Run the noisy version for all 30 seeds if time allows; otherwise run it for 5 seeds and state that.
- 50×50 scaling run: exact statevector only, 1–3 seeds.
- Run seeds in parallel with `multiprocessing.Pool` (one process per core).
- Save all logs as CSV/parquet under `results/{method}/{density}/{seed}/` so plotting never re-runs the simulations.

---

## 13. Suggested timeline (about 8 weeks)

| Week | Work | Deliverable |
|---|---|---|
| 1 | Environment, grid, demand generator, simulator + rule-based controller | Delay vs λ curve, λ_max calibrated |
| 2 | Consumer-device probes, feature builder, dataset generation | `data/` folder for 3 densities × 30 seeds |
| 3 | Encoding, VQC, SPSA on a single node (exact) | Fig 1(c) |
| 4 | Comms layer: fidelity, BB84/OTP, trust, consensus, swapping, latency | Fig 1(b), unit tests |
| 5 | Full Q-ITS agent (Algorithm 1) in the loop; noisy backend | Q-ITS runs |
| 6 | DQN baseline training + evaluation | DQN runs |
| 7 | All experiments, ablations, statistics | Fig 1, Fig 2, Tables IV, V |
| 8 | Write-up: reproduction results + list of paper gaps, which leads into the improvement phase | Report chapter |

---

## 14. Unit tests worth writing (pytest)

- `amplitude_encode` output has L2 norm 1, and 16 inputs give a 16-length vector.
- `build_ansatz(4, 8)` has 64 parameters.
- `state_fidelity(werner(F), PHI) == F`.
- `werner_qmi(1.0) == 2` and `werner_qmi(0.25) == 0`.
- `bb84` aborts for F = 0.80 (QBER 13.3 %) and succeeds for F = 0.95.
- `otp(otp(m, k), k) == m`.
- Consensus (Eq. 16) on a connected graph converges to the average of the E_i.
- The simulator conserves vehicles: spawned = in network + exited.
