# SUMO evaluation (25 x 25, demand: closed_loop, lambda per density {'low': 3.936, 'medium': 7.872, 'high': 11.808} veh/s)

Delay = SUMO time loss + waiting to enter, mean ± 95 % CI over seeds.

| controller | density | runs | delay (s) | delay, all vehicles (s) | completed | teleports / run |
|---|---|---|---|---|---|---|
| fixed_time | low | 30 | 140.6 ± 0.5 | 134.7 ± 0.4 | 0.852 | 0.0 |
| fixed_time | medium | 30 | 146.9 ± 0.3 | 140.6 ± 0.2 | 0.847 | 0.0 |
| fixed_time | high | 30 | 164.1 ± 0.5 | 157.8 ± 0.5 | 0.832 | 0.0 |
| rule_based | low | 30 | 145.2 ± 0.5 | 139.0 ± 0.4 | 0.849 | 0.0 |
| rule_based | medium | 30 | 152.2 ± 0.3 | 145.5 ± 0.3 | 0.844 | 0.0 |
| rule_based | high | 30 | 168.5 ± 0.5 | 161.7 ± 0.4 | 0.831 | 0.0 |
| dqn_split | low | 30 | 144.7 ± 0.4 | 138.4 ± 0.4 | 0.849 | 0.0 |
| dqn_split | medium | 30 | 148.6 ± 0.3 | 142.2 ± 0.3 | 0.846 | 0.0 |
| dqn_split | high | 30 | 164.5 ± 0.5 | 158.0 ± 0.5 | 0.833 | 0.0 |
| dqn | low | 30 | 72.0 ± 0.4 | 70.4 ± 0.4 | 0.892 | 24.8 |
| dqn | medium | 30 | 82.4 ± 0.2 | 79.9 ± 0.2 | 0.885 | 23.9 |
| dqn | high | 30 | 101.5 ± 0.4 | 98.4 ± 0.4 | 0.873 | 19.7 |
| qits | low | 30 | 142.1 ± 0.5 | 136.1 ± 0.4 | 0.851 | 0.0 |
| qits | medium | 30 | 149.3 ± 0.3 | 142.8 ± 0.3 | 0.846 | 0.0 |
| qits | high | 30 | 166.5 ± 0.4 | 159.9 ± 0.4 | 0.832 | 0.0 |
| qits_routing_only | low | 30 | 140.1 ± 0.4 | 134.3 ± 0.4 | 0.852 | 0.0 |
| qits_routing_only | medium | 30 | 146.8 ± 0.3 | 140.5 ± 0.2 | 0.847 | 0.0 |
| qits_routing_only | high | 30 | 164.5 ± 0.5 | 158.1 ± 0.5 | 0.833 | 0.0 |
| qits_signals_only | low | 30 | 142.6 ± 0.5 | 136.5 ± 0.4 | 0.850 | 0.0 |
| qits_signals_only | medium | 30 | 149.6 ± 0.3 | 143.1 ± 0.3 | 0.845 | 0.0 |
| qits_signals_only | high | 30 | 167.4 ± 0.4 | 160.7 ± 0.4 | 0.831 | 0.0 |

## qits vs baselines

Improvement = (baseline - qits) / baseline x 100 (positive: lower delay with qits).

| baseline | density | metric | seeds | baseline mean | qits mean | qits improvement % | p (paired t) |
|---|---|---|---|---|---|---|---|
| fixed_time | low | avg_delay_s | 30 | 140.6 | 142.1 | -1.1 | 3.3e-12 |
| fixed_time | low | avg_delay_all_s | 30 | 134.7 | 136.1 | -1.0 | 1.5e-13 |
| fixed_time | medium | avg_delay_s | 30 | 146.9 | 149.3 | -1.6 | 4.0e-22 |
| fixed_time | medium | avg_delay_all_s | 30 | 140.6 | 142.8 | -1.6 | 6.8e-25 |
| fixed_time | high | avg_delay_s | 30 | 164.1 | 166.5 | -1.4 | 4.9e-12 |
| fixed_time | high | avg_delay_all_s | 30 | 157.8 | 159.9 | -1.3 | 2.1e-10 |
| rule_based | low | avg_delay_s | 30 | 145.2 | 142.1 | 2.1 | 6.4e-21 |
| rule_based | low | avg_delay_all_s | 30 | 139.0 | 136.1 | 2.0 | 1.7e-21 |
| rule_based | medium | avg_delay_s | 30 | 152.2 | 149.3 | 1.9 | 3.8e-21 |
| rule_based | medium | avg_delay_all_s | 30 | 145.5 | 142.8 | 1.9 | 6.6e-23 |
| rule_based | high | avg_delay_s | 30 | 168.5 | 166.5 | 1.2 | 1.1e-11 |
| rule_based | high | avg_delay_all_s | 30 | 161.7 | 159.9 | 1.1 | 1.5e-11 |
| dqn_split | low | avg_delay_s | 30 | 144.7 | 142.1 | 1.8 | 9.1e-16 |
| dqn_split | low | avg_delay_all_s | 30 | 138.4 | 136.1 | 1.7 | 1.9e-16 |
| dqn_split | medium | avg_delay_s | 30 | 148.6 | 149.3 | -0.4 | 3.1e-06 |
| dqn_split | medium | avg_delay_all_s | 30 | 142.2 | 142.8 | -0.4 | 5.5e-08 |
| dqn_split | high | avg_delay_s | 30 | 164.5 | 166.5 | -1.2 | 1.7e-10 |
| dqn_split | high | avg_delay_all_s | 30 | 158.0 | 159.9 | -1.2 | 2.5e-10 |
| dqn | low | avg_delay_s | 30 | 72.0 | 142.1 | -97.3 | 3.3e-56 |
| dqn | low | avg_delay_all_s | 30 | 70.4 | 136.1 | -93.4 | 1.5e-56 |
| dqn | medium | avg_delay_s | 30 | 82.4 | 149.3 | -81.2 | 6.7e-60 |
| dqn | medium | avg_delay_all_s | 30 | 79.9 | 142.8 | -78.7 | 8.5e-62 |
| dqn | high | avg_delay_s | 30 | 101.5 | 166.5 | -64.0 | 1.8e-55 |
| dqn | high | avg_delay_all_s | 30 | 98.4 | 159.9 | -62.5 | 6.5e-56 |
| qits_routing_only | low | avg_delay_s | 30 | 140.1 | 142.1 | -1.5 | 1.2e-15 |
| qits_routing_only | low | avg_delay_all_s | 30 | 134.3 | 136.1 | -1.4 | 5.3e-16 |
| qits_routing_only | medium | avg_delay_s | 30 | 146.8 | 149.3 | -1.7 | 1.3e-22 |
| qits_routing_only | medium | avg_delay_all_s | 30 | 140.5 | 142.8 | -1.6 | 7.8e-23 |
| qits_routing_only | high | avg_delay_s | 30 | 164.5 | 166.5 | -1.2 | 2.6e-11 |
| qits_routing_only | high | avg_delay_all_s | 30 | 158.1 | 159.9 | -1.1 | 1.2e-10 |
| qits_signals_only | low | avg_delay_s | 30 | 142.6 | 142.1 | 0.3 | 8.0e-03 |
| qits_signals_only | low | avg_delay_all_s | 30 | 136.5 | 136.1 | 0.3 | 4.6e-03 |
| qits_signals_only | medium | avg_delay_s | 30 | 149.6 | 149.3 | 0.3 | 6.7e-04 |
| qits_signals_only | medium | avg_delay_all_s | 30 | 143.1 | 142.8 | 0.2 | 1.2e-03 |
| qits_signals_only | high | avg_delay_s | 30 | 167.4 | 166.5 | 0.5 | 6.6e-05 |
| qits_signals_only | high | avg_delay_all_s | 30 | 160.7 | 159.9 | 0.5 | 2.3e-05 |
