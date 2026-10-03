"""Result tables for the SUMO evaluation.

    python -m sumo_imp.experiments.report [--rows 5] [--demand closed_loop]
Output: sumo_imp/results/<R>x<C>/eval[_closed_loop]/report.md
  1. delay per controller and density: mean ± 95 % CI over seeds (paper Eq. 25), for completed trips
     and for all vehicles (unfinished ones counted with the delay so far), plus completion and teleports;
  2. Q-ITS vs each baseline: improvement = (X_baseline - X_QITS) / X_baseline x 100 (paper Table V),
     paired t-test over the same seeds.
"""
import argparse

import sumo_imp  # noqa: F401  (paths)
import pandas as pd

from qits.metrics import ci95, improvement, paired_ttest  # noqa: E402
from sumo_imp.config import DEMANDS, Demand, load_config, results_dir  # noqa: E402

DENSITIES = ["low", "medium", "high"]


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int)
    ap.add_argument("--reference", default="qits", help="controller compared against the others")
    ap.add_argument("--demand", default="dataset", choices=DEMANDS)
    args = ap.parse_args()
    _, scfg = load_config(args.rows)
    dem = Demand(scfg, args.demand)
    out = results_dir(scfg) / f"eval{dem.suffix}"
    df = pd.read_csv(out / "summary.csv")
    order = [c for c in ["fixed_time", "rule_based", "dqn_split", "dqn", "qits"] if c in set(df.controller)]
    order += sorted(set(df.controller) - set(order))

    rows = []
    for c in order:
        for d in DENSITIES:
            x = df[(df.controller == c) & (df.density == d)]
            if not len(x):
                continue
            m, h = ci95(x.avg_delay_s)
            ma, ha = ci95(x.avg_delay_all_s)
            rows.append({"controller": c, "density": d, "runs": len(x), "delay (s)": f"{m:.1f} ± {h:.1f}",
                         "delay, all vehicles (s)": f"{ma:.1f} ± {ha:.1f}",
                         "completed": f"{x.completion_rate.mean():.3f}",
                         "teleports / run": f"{x.teleports.mean():.1f}"})
    text = (f"# SUMO evaluation ({scfg['network']['rows']} x {scfg['network']['cols']}, demand: {args.demand}, "
            f"lambda per density {dem.lambdas()} veh/s)\n\n"
            "Delay = SUMO time loss + waiting to enter, mean ± 95 % CI over seeds.\n\n" + md(pd.DataFrame(rows)))

    ref = args.reference
    if ref in set(df.controller):
        comp = []
        for c in order:
            if c == ref:
                continue
            for d in DENSITIES:
                a = df[(df.controller == c) & (df.density == d)].set_index("seed")
                b = df[(df.controller == ref) & (df.density == d)].set_index("seed")
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                for metric in ("avg_delay_s", "avg_delay_all_s"):
                    t, p = paired_ttest(a.loc[common, metric], b.loc[common, metric])
                    comp.append({"baseline": c, "density": d, "metric": metric, "seeds": len(common),
                                 "baseline mean": round(a.loc[common, metric].mean(), 1),
                                 f"{ref} mean": round(b.loc[common, metric].mean(), 1),
                                 f"{ref} improvement %": round(improvement(a.loc[common, metric].mean(),
                                                                          b.loc[common, metric].mean()), 1),
                                 "p (paired t)": f"{p:.1e}"})
        if comp:
            text += (f"\n## {ref} vs baselines\n\nImprovement = (baseline - {ref}) / baseline x 100 "
                     "(positive: lower delay with " + ref + ").\n\n" + md(pd.DataFrame(comp)))
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
