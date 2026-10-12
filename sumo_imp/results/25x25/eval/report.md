# SUMO evaluation (25 x 25, demand: dataset, lambda per density {'low': 2.784, 'medium': 5.568, 'high': 8.352} veh/s)

Delay = SUMO time loss + waiting to enter, mean ± 95 % CI over seeds.

| controller | density | runs | delay (s) | delay, all vehicles (s) | completed | teleports / run |
|---|---|---|---|---|---|---|
| fixed_time | low | 30 | 139.5 ± 0.5 | 133.7 ± 0.4 | 0.854 | 0.0 |
| fixed_time | medium | 30 | 142.4 ± 0.4 | 136.4 ± 0.3 | 0.851 | 0.0 |
| fixed_time | high | 30 | 148.1 ± 0.4 | 141.8 ± 0.3 | 0.846 | 0.0 |
| rule_based | low | 30 | 144.7 ± 0.5 | 138.5 ± 0.4 | 0.851 | 0.0 |
| rule_based | medium | 30 | 147.5 ± 0.4 | 141.2 ± 0.3 | 0.848 | 0.0 |
| rule_based | high | 30 | 153.4 ± 0.3 | 146.7 ± 0.3 | 0.843 | 0.0 |
| dqn_split | low | 30 | 138.9 ± 0.5 | 133.2 ± 0.5 | 0.854 | 0.0 |
| dqn_split | medium | 30 | 141.9 ± 0.3 | 135.8 ± 0.3 | 0.852 | 0.0 |
| dqn_split | high | 30 | 147.5 ± 0.3 | 141.2 ± 0.3 | 0.846 | 0.0 |
| dqn | low | 30 | 55.2 ± 0.3 | 53.9 ± 0.2 | 0.903 | 16.5 |
| dqn | medium | 30 | 65.3 ± 0.2 | 63.5 ± 0.2 | 0.896 | 18.6 |
| dqn | high | 30 | 77.1 ± 0.2 | 74.8 ± 0.2 | 0.888 | 14.8 |
| qits | low | 30 | 141.0 ± 0.5 | 135.2 ± 0.5 | 0.853 | 0.0 |
| qits | medium | 30 | 144.4 ± 0.4 | 138.3 ± 0.3 | 0.850 | 0.0 |
| qits | high | 30 | 150.6 ± 0.4 | 144.0 ± 0.3 | 0.845 | 0.0 |
| qits_routing_only | low | 10 | 139.1 ± 0.8 | 133.4 ± 0.7 | 0.852 | 0.0 |
| qits_routing_only | medium | 10 | 142.1 ± 0.7 | 136.0 ± 0.6 | 0.851 | 0.0 |
| qits_routing_only | high | 10 | 147.9 ± 0.5 | 141.6 ± 0.5 | 0.846 | 0.0 |
| qits_signals_only | low | 10 | 141.7 ± 0.9 | 135.7 ± 0.7 | 0.852 | 0.0 |
| qits_signals_only | medium | 10 | 145.1 ± 0.6 | 138.8 ± 0.5 | 0.849 | 0.0 |
| qits_signals_only | high | 10 | 150.9 ± 0.5 | 144.3 ± 0.5 | 0.845 | 0.0 |

## qits vs baselines

Improvement = (baseline - qits) / baseline x 100 (positive: lower delay with qits).

| baseline | density | metric | seeds | baseline mean | qits mean | qits improvement % | p (paired t) |
|---|---|---|---|---|---|---|---|
| fixed_time | low | avg_delay_s | 30 | 139.5 | 141.0 | -1.0 | 3.7e-11 |
| fixed_time | low | avg_delay_all_s | 30 | 133.7 | 135.2 | -1.1 | 1.4e-11 |
| fixed_time | medium | avg_delay_s | 30 | 142.4 | 144.4 | -1.4 | 1.0e-16 |
| fixed_time | medium | avg_delay_all_s | 30 | 136.4 | 138.3 | -1.3 | 2.0e-17 |
| fixed_time | high | avg_delay_s | 30 | 148.1 | 150.6 | -1.7 | 2.4e-18 |
| fixed_time | high | avg_delay_all_s | 30 | 141.8 | 144.0 | -1.6 | 1.7e-18 |
| rule_based | low | avg_delay_s | 30 | 144.7 | 141.0 | 2.6 | 1.5e-20 |
| rule_based | low | avg_delay_all_s | 30 | 138.5 | 135.2 | 2.4 | 1.1e-20 |
| rule_based | medium | avg_delay_s | 30 | 147.5 | 144.4 | 2.1 | 1.5e-21 |
| rule_based | medium | avg_delay_all_s | 30 | 141.2 | 138.3 | 2.1 | 5.1e-22 |
| rule_based | high | avg_delay_s | 30 | 153.4 | 150.6 | 1.9 | 1.9e-19 |
| rule_based | high | avg_delay_all_s | 30 | 146.7 | 144.0 | 1.8 | 8.0e-20 |
| dqn_split | low | avg_delay_s | 30 | 138.9 | 141.0 | -1.5 | 7.3e-16 |
| dqn_split | low | avg_delay_all_s | 30 | 133.2 | 135.2 | -1.5 | 6.9e-18 |
| dqn_split | medium | avg_delay_s | 30 | 141.9 | 144.4 | -1.8 | 3.8e-20 |
| dqn_split | medium | avg_delay_all_s | 30 | 135.8 | 138.3 | -1.8 | 4.6e-20 |
| dqn_split | high | avg_delay_s | 30 | 147.5 | 150.6 | -2.0 | 8.0e-21 |
| dqn_split | high | avg_delay_all_s | 30 | 141.2 | 144.0 | -2.0 | 2.6e-21 |
| dqn | low | avg_delay_s | 30 | 55.2 | 141.0 | -155.4 | 2.0e-57 |
| dqn | low | avg_delay_all_s | 30 | 53.9 | 135.2 | -150.8 | 1.4e-58 |
| dqn | medium | avg_delay_s | 30 | 65.3 | 144.4 | -121.1 | 4.3e-61 |
| dqn | medium | avg_delay_all_s | 30 | 63.5 | 138.3 | -117.7 | 8.1e-62 |
| dqn | high | avg_delay_s | 30 | 77.1 | 150.6 | -95.2 | 1.8e-58 |
| dqn | high | avg_delay_all_s | 30 | 74.8 | 144.0 | -92.6 | 7.3e-60 |
| qits_routing_only | low | avg_delay_s | 10 | 139.1 | 140.8 | -1.2 | 1.6e-05 |
| qits_routing_only | low | avg_delay_all_s | 10 | 133.4 | 135.2 | -1.3 | 7.1e-06 |
| qits_routing_only | medium | avg_delay_s | 10 | 142.1 | 144.4 | -1.6 | 5.7e-07 |
| qits_routing_only | medium | avg_delay_all_s | 10 | 136.0 | 138.2 | -1.6 | 4.6e-08 |
| qits_routing_only | high | avg_delay_s | 10 | 147.9 | 150.1 | -1.5 | 4.9e-07 |
| qits_routing_only | high | avg_delay_all_s | 10 | 141.6 | 143.6 | -1.4 | 1.8e-06 |
| qits_signals_only | low | avg_delay_s | 10 | 141.7 | 140.8 | 0.6 | 1.4e-02 |
| qits_signals_only | low | avg_delay_all_s | 10 | 135.7 | 135.2 | 0.4 | 5.3e-02 |
| qits_signals_only | medium | avg_delay_s | 10 | 145.1 | 144.4 | 0.5 | 4.9e-03 |
| qits_signals_only | medium | avg_delay_all_s | 10 | 138.8 | 138.2 | 0.4 | 4.2e-03 |
| qits_signals_only | high | avg_delay_s | 10 | 150.9 | 150.1 | 0.5 | 6.0e-03 |
| qits_signals_only | high | avg_delay_all_s | 10 | 144.3 | 143.6 | 0.5 | 1.1e-02 |
