"""Build every figure and table of the paper's evaluation from saved results (no re-running).

    python -m experiments.make_figures
Inputs : results/baselines/, results/qits/, results/dqn/, results/quantum/ (+ a 5 s comms re-simulation)
Outputs: results/figures/
    fig1.png      (a) delay vs density  (b) fidelity over time  (c) VQC parameter convergence
                  (d) consensus entropy across nodes
    fig2.png      (a) delay vs time  (b) policy entropy vs time  (c) parameter-update norm
                  (d) energy breakdown
    fig_energy_sensitivity.png   Q-ITS vs DQN energy as a function of assumed QPU power
    table_iv_ablation.{csv,md}   Table IV
    table_v_overall.{csv,md}     Table V
    headline_vs_paper.{csv,md}   every numeric claim of the abstract vs this reproduction
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comms_demo import simulate as simulate_comms
from qits.config import ROOT, load_config
from qits.energy import energy_breakdown, qpu_busy_s
from qits.metrics import ci95, entropic_stability_gain, improvement, paired_ttest, routing_entropy
from qits.traffic.grid import Grid

OUT = ROOT / "results" / "figures"
DENS = ["low", "medium", "high"]
METHODS = {  # key: (label, colour)
    "fixed_time": ("Fixed-time", "tab:gray"),
    "rule_based": ("Rule-based ITS", "tab:green"),
    "dqn_split": ("DQN-split (same actions as Q-ITS)", "tab:purple"),
    "dqn": ("DQN (phase control)", "tab:orange"),
    "qits": ("Q-ITS (proposed)", "tab:blue"),
}
ABLATIONS = {"qits": "Q-ITS (Full)", "qits_noqopt": "Q-ITS -NoQOpt", "qits_noqkd": "Q-ITS -NoQKD",
             "qits_nocons": "Q-ITS -NoCons"}


# ------------------------------------------------------------------ loading
def load_all():
    b = pd.read_csv(ROOT / "results" / "baselines" / "summary.csv")
    q = pd.read_csv(ROOT / "results" / "qits" / "summary.csv")
    q = q[q.backend == "noisy"].copy()
    q["controller"] = q["variant"]
    q["comm_bytes"] = q["consensus_bytes"]
    return pd.concat([b, q], ignore_index=True)


def runs_dir(method):
    if method.startswith("qits"):
        return ROOT / "results" / "qits" / "runs" / method
    return ROOT / "results" / "baselines" / "runs" / method


def mean_series(method, density, kind, key, value):
    files = sorted(runs_dir(method).glob(f"{density}_seed*_{kind}.csv"))
    if not files:
        return None
    return pd.concat(pd.read_csv(f) for f in files).groupby(key)[value].mean()


def fmt_change(baseline, value):
    """Improvement in % (Table V formula); very large increases are shown as a ratio."""
    if not baseline:
        return "n/a"
    imp = improvement(baseline, value)
    return f"{imp:.1f} %" if imp > -1000 else f"{value / baseline:.3g}x higher"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def save_table(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(md_table(df), encoding="utf-8")


# ------------------------------------------------------------------ figures
def fig1(df, cfg):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    x = [cfg["demand"]["density_levels"][d] * 100 for d in DENS]

    # (a) delay vs density
    a = ax[0, 0]
    for m, (label, col) in METHODS.items():
        stats = [ci95(df[(df.controller == m) & (df.density == d)].avg_delay_s) for d in DENS]
        a.errorbar(x, [s[0] for s in stats], yerr=[s[1] for s in stats], marker="o", capsize=3,
                   label=label, color=col)
    a.set(xlabel="Traffic density (% saturation)", ylabel="Avg vehicle delay (s)",
          title="(a) Delay vs traffic density (30 seeds, 95 % CI)")
    a.legend(fontsize=8)

    # (b) fidelity over time (5 s re-simulation of the comms layer)
    a = ax[0, 1]
    _, no_re = simulate_comms(cfg, maintain=False)
    _, re = simulate_comms(cfg, maintain=True)
    a.plot(no_re.t, no_re.F_link, label="no re-entanglement")
    a.plot(re.t, re.F_link, lw=0.8, label="adaptive re-entanglement (Eq. 17)")
    a.axhline(cfg["comms"]["fidelity"]["eps_F"], ls="--", c="tab:red", label="ε = 0.85")
    a.set(xlabel="Time (s)", ylabel="Entanglement fidelity", title="(b) Fidelity decay over time")
    a.legend(fontsize=8)

    # (c) VQC parameter convergence (single node, noisy backend, ξ = 0.1)
    a = ax[1, 0]
    conv = pd.read_csv(ROOT / "results" / "quantum" / "single_node_convergence.csv")
    for (backend, xi), g in conv.groupby(["backend", "xi"]):
        m = g.groupby("iter").dtheta_norm.mean().rolling(10, min_periods=1).mean()
        a.plot(m.index, m / m.iloc[:10].mean(), label=f"{backend}, ξ={xi}")
    a.set(xlabel="Iteration", ylabel="||Δθ|| (normalised)", title="(c) Convergence of circuit parameters")
    a.legend(fontsize=8)

    # (d) consensus entropy across nodes over time
    a = ax[1, 1]
    for d, col in zip(DENS, ("tab:blue", "tab:orange", "tab:green")):
        s = mean_series("qits", d, "epochs", "t", "entropy")
        a.plot(s.index, s.values, color=col, alpha=0.3, lw=0.8)
        a.plot(s.index, s.rolling(12, min_periods=1).mean(), color=col, lw=2, label=f"{d} (2-min mean)")
    a.set(xlabel="Simulation time (s)", ylabel="Consensus entropy ΔH (nats)",
          title="(d) Consensus entropy across nodes (Q-ITS)")
    a.legend(fontsize=8)
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1.png", dpi=150)


def uniform_routing_entropy(cfg):
    g = Grid(cfg["network"]["rows"], cfg["network"]["cols"])
    return routing_entropy(g.valid_mask / g.valid_mask.sum(1, keepdims=True), g.valid_mask)


def fig2(df, cfg, energy):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) delay vs time, high density
    a = ax[0, 0]
    for m, (label, col) in METHODS.items():
        s = mean_series(m, "high", "delay", "minute", "delay")
        if s is not None:
            a.plot(s.index, s.values, marker=".", label=label, color=col)
    a.set(xlabel="Arrival minute", ylabel="Avg delay (s)", title="(a) Delay vs time (high density)")
    a.legend(fontsize=8)

    # (b) decision-policy entropy over time, high density, relative to a uniform policy
    #     (methods choose among different numbers of options: 2-4 directions, 2 phases, 5 splits)
    a = ax[0, 1]
    s = mean_series("qits", "high", "epochs", "t", "entropy")
    a.plot(s.index, s.values / uniform_routing_entropy(cfg), label="Q-ITS: routing distribution p_ij",
           color=METHODS["qits"][1])
    for m, k in (("dqn", 2), ("dqn_split", 5)):
        s = mean_series(m, "high", "entropy", "t", "entropy")
        if s is not None:
            a.plot(s.index, s.values / np.log(k), label=f"{METHODS[m][0].split(' (')[0]}: softmax(Q), τ=1",
                   color=METHODS[m][1])
    a.axhline(1.0, color=METHODS["rule_based"][1], ls="--", label="Rule-based: uniform routing")
    a.set(xlabel="Time (s)", ylabel="ΔH / ΔH_uniform",
          title="(b) Policy entropy relative to uniform (high density)")
    a.text(0.02, 0.03, "DQN acts greedily; softmax(Q) is near-uniform because Q-gaps << τ",
           transform=a.transAxes, fontsize=7, color="gray")
    a.legend(fontsize=8)

    # (c) parameter update norm vs training progress
    a = ax[1, 0]
    conv = pd.read_csv(ROOT / "results" / "quantum" / "single_node_convergence.csv")
    q = conv[(conv.backend == "noisy") & (conv.xi == cfg["quantum"]["perturbation"])]
    q = q.groupby("iter").dtheta_norm.mean().rolling(10, min_periods=1).mean()
    a.plot(np.linspace(0, 100, len(q)), q / q.iloc[:10].mean(), label="Q-ITS (SPSA, 300 iterations)",
           color=METHODS["qits"][1])
    for m in ("dqn", "dqn_split"):
        name = "dqn" if m == "dqn" else "dqn_split"
        f = ROOT / "results" / "dqn" / f"{name}_update_norms.csv"
        if f.exists():
            u = pd.read_csv(f).update_norm.rolling(200, min_periods=1).mean()
            a.plot(np.linspace(0, 100, len(u)), u / u.iloc[:50].mean(), label=f"{METHODS[m][0]} (training)",
                   color=METHODS[m][1])
    a.axhline(0, color=METHODS["rule_based"][1], label="Rule-based (no parameters)")
    a.set(xlabel="Training progress (%)", ylabel="||Δθ|| (normalised)", title="(c) Parameter update norm")
    a.legend(fontsize=8)

    # (d) energy breakdown: grouped bars, log scale, totals annotated (assumptions in cfg.energy)
    a = ax[1, 1]
    methods = ["rule_based", "dqn_split", "dqn", "qits"]
    x, w = np.arange(len(methods)), 0.26
    parts = (("computation_j", "computation", "tab:blue"), ("communication_j", "communication", "tab:orange"),
             ("qkd_hardware_j", "QKD hardware", "tab:red"))
    for k, (col_name, label, col) in enumerate(parts):
        vals = np.array([energy.loc[m, col_name] for m in methods])
        a.bar(x + (k - 1) * w, np.where(vals > 0, vals, np.nan), w, label=label, color=col)
    for i, m in enumerate(methods):
        tot = energy.loc[m, "total_j"]
        a.text(i, 1.5 * max(tot, 0.05), "0 J" if tot == 0 else f"{tot:.3g} J", ha="center", fontsize=8)
    a.set_yscale("log")
    a.set_ylim(1e-2, 1e9)
    a.set_xticks(x, [METHODS[m][0].split(" (")[0] for m in methods], fontsize=8)
    a.set(ylabel="Energy per 30-min run (J, log)",
          title=f"(d) Energy breakdown (assumed P_qpu = {cfg['energy']['p_qpu_w'] / 1e3:g} kW)")
    a.legend(fontsize=8, loc="upper left")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2.png", dpi=150)


def energy_table(df, cfg):
    rows = {}
    for m in ["fixed_time", "rule_based", "dqn_split", "dqn", *ABLATIONS]:
        sub = df[df.controller == m]
        if len(sub) == 0:
            continue
        parts = pd.DataFrame([energy_breakdown(r, cfg, m) for r in sub.to_dict("records")]).mean()
        rows[m] = parts
    return pd.DataFrame(rows).T


def fig_energy_sensitivity(df, cfg):
    q = df[df.controller == "qits"].to_dict("records")
    d = df[df.controller == "dqn"].to_dict("records")
    sweep = np.logspace(-3, np.log10(3e4), 60)
    e_q = [np.mean([energy_breakdown(r, cfg, "qits", p)["total_j"] for r in q]) for p in sweep]
    e_q_comp = [np.mean([energy_breakdown(r, cfg, "qits", p)["computation_j"] for r in q]) for p in sweep]
    e_d = np.mean([energy_breakdown(r, cfg, "dqn")["total_j"] for r in d])
    t_qpu = np.mean([qpu_busy_s(r, cfg) for r in q])
    fig, a = plt.subplots(figsize=(7, 4.5))
    a.loglog(sweep, np.array(e_q) / 3.6e6, label="Q-ITS total (incl. QKD hardware)")
    a.loglog(sweep, np.array(e_q_comp) / 3.6e6, ls="--", label="Q-ITS computation only")
    a.axhline(e_d / 3.6e6, color="tab:orange", label="DQN (phase) total")
    a.set(xlabel="Assumed QPU power while running (W)", ylabel="Energy per 30-min run (kWh)",
          title=f"Energy sensitivity (QPU busy {t_qpu:.0f} s per run, summed over 25 node QPUs)")
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_energy_sensitivity.png", dpi=150)
    return e_d / t_qpu, t_qpu      # break-even QPU power for computation alone


# ------------------------------------------------------------------- tables
def per_seed(df, method, col):
    return df[df.controller == method].pivot_table(index="seed", columns="density", values=col)


def table_iv(df):
    """Delay reduction vs rule-based and decision-latency reduction vs the centralised DQN,
    per seed averaged over the three densities; mean ± std over the 30 seeds (paper format)."""
    rule, dqn_lat = per_seed(df, "rule_based", "avg_delay_s"), per_seed(df, "dqn", "decision_latency_s")
    rows = []
    for m, label in ABLATIONS.items():
        d = per_seed(df, m, "avg_delay_s")
        if d.empty:
            continue
        red = ((rule.loc[d.index] - d) / rule.loc[d.index] * 100).mean(axis=1)
        lat = per_seed(df, m, "decision_latency_s")
        lred = ((dqn_lat.loc[lat.index] - lat) / dqn_lat.loc[lat.index] * 100).mean(axis=1)
        by_d = {f"delay {k} (s)": round(d[k].mean(), 2) for k in DENS}
        rows.append({"Variant": label, "Delay reduction vs rule-based (%)": f"{red.mean():.1f} ± {red.std():.1f}",
                     "Decision latency reduction vs DQN (%)": f"{lred.mean():.1f} ± {lred.std():.1f}",
                     **by_d, "latency (ms)": round(lat.values.mean() * 1000, 1), "runs": d.size})
    return pd.DataFrame(rows)


def table_v(df, energy):
    q = df[df.controller == "qits"]
    rows = []
    for base in ("rule_based", "dqn"):
        b = df[df.controller == base]
        items = [("Average delay (s)", b.avg_delay_s.mean(), q.avg_delay_s.mean()),
                 ("Energy per run (kWh)", energy.loc[base, "total_j"] / 3.6e6, energy.loc["qits", "total_j"] / 3.6e6)]
        if base == "rule_based":   # Eq. 19-20: routing entropy vs the rule-based uniform routing
            items.append(("Consensus entropy of routing (nats)", uniform_routing_entropy(load_config()),
                          q.consensus_entropy.mean()))
        for name, bv, qv in items:
            rows.append({"Baseline": METHODS[base][0], "Metric": name, "Baseline avg": f"{bv:.4g}",
                         "Q-ITS avg": f"{qv:.4g}",
                         "Improvement": fmt_change(bv, qv)})
    return pd.DataFrame(rows)


def headline(df, energy, cfg, breakeven_w):
    q, r, d, ds = (df[df.controller == m] for m in ("qits", "rule_based", "dqn", "dqn_split"))
    h_rule = uniform_routing_entropy(cfg)
    comm_q = q.consensus_bytes.mean()
    comm_q_all = comm_q + q.qkd_classical_bytes.mean()

    pct = fmt_change

    rows = [
        ("Communication overhead reduction", "37.4 %",
         f"{pct(d.comm_bytes.mean(), comm_q)} vs DQN (consensus msgs only); "
         f"{pct(d.comm_bytes.mean(), comm_q_all)} incl. QKD post-processing"),
        ("Decision latency reduction", "42.1 %",
         f"{pct(d.decision_latency_s.mean(), q.decision_latency_s.mean())} vs centralised DQN "
         f"({q.decision_latency_s.mean() * 1e3:.0f} ms vs {d.decision_latency_s.mean() * 1e3:.0f} ms)"),
        ("Congestion index reduction", "23.9 %",
         f"{pct(r.congestion_index.mean(), q.congestion_index.mean())} vs rule-based; "
         f"{pct(d.congestion_index.mean(), q.congestion_index.mean())} vs DQN"),
        ("Packet delivery ratio", "98.2 %", f"{q.pdr.mean() * 100:.1f} %"),
        ("Entropic stability gain (Eq. 20)", "61.3 %",
         f"{entropic_stability_gain(h_rule, q.consensus_entropy.mean()):.1f} % vs rule-based uniform routing"),
        ("Average delay reduction", "42.1 % (Table V)",
         f"{pct(r.avg_delay_s.mean(), q.avg_delay_s.mean())} vs rule-based; "
         f"{pct(ds.avg_delay_s.mean(), q.avg_delay_s.mean())} vs DQN-split; "
         f"{pct(d.avg_delay_s.mean(), q.avg_delay_s.mean())} vs DQN"),
        ("Energy reduction", "16.7 % / 36.9 %",
         f"{pct(energy.loc['dqn', 'total_j'], energy.loc['qits', 'total_j'])} vs DQN at "
         f"P_qpu = {cfg['energy']['p_qpu_w'] / 1e3:g} kW; Q-ITS computation matches DQN only if "
         f"P_qpu < {breakeven_w * 1e3:.2g} mW"),
        ("Mean entanglement fidelity", "> 0.90", f"{q.mean_fidelity.mean():.3f}"),
        ("VQC convergence", "within 90 iterations in 88 % of runs",
         "noisy ξ=0.01 (paper) does not converge; ξ=0.1 converges in ~150-300 iterations (Fig. 1c)"),
    ]
    return pd.DataFrame(rows, columns=["Claim", "Paper", "This reproduction"])


def significance(df):
    q = df[df.controller == "qits"].set_index(["density", "seed"]).avg_delay_s
    rows = []
    for m in ("fixed_time", "rule_based", "dqn_split", "dqn"):
        b = df[df.controller == m].set_index(["density", "seed"]).avg_delay_s
        for d in DENS:
            idx = q.loc[d].index.intersection(b.loc[d].index)
            t, p = paired_ttest(b.loc[d].loc[idx].values, q.loc[d].loc[idx].values)
            rows.append({"baseline": m, "density": d, "baseline_delay": round(b.loc[d].mean(), 2),
                         "qits_delay": round(q.loc[d].mean(), 2),
                         "improvement_%": round(improvement(b.loc[d].mean(), q.loc[d].mean()), 1),
                         "t": round(t, 1), "p": f"{p:.1e}"})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    energy = energy_table(df, cfg)
    energy.assign(total_kwh=energy.total_j / 3.6e6).to_csv(OUT / "energy_breakdown.csv")

    fig1(df, cfg)
    fig2(df, cfg, energy)
    breakeven_w, t_qpu = fig_energy_sensitivity(df, cfg)

    t4, t5 = table_iv(df), table_v(df, energy)
    save_table(t4, "table_iv_ablation")
    save_table(t5, "table_v_overall")
    save_table(significance(df), "significance_delay")
    hl = headline(df, energy, cfg, breakeven_w)
    save_table(hl, "headline_vs_paper")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 120)
    print("TABLE IV\n", t4.to_string(index=False), "\n\nTABLE V\n", t5.to_string(index=False),
          "\n\nHEADLINE\n", hl.to_string(index=False))
    print(f"\nQPU busy time per run (sum over 25 node QPUs): {t_qpu:.0f} s; "
          f"break-even QPU power: {breakeven_w * 1e3:.3g} mW")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
