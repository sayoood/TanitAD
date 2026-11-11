# Tc20s repair on rank-1 twins vs rank-0 (2026-11-11)
Evidence class: MEASURED, own, CPU, `p5c_labels.py repair` (wraps UNCHANGED P5-S `iterate20`, cubic, ITERS 3; blob 294644df). Raw JSON: `repair_rank1.json`, `repair_rank0.json`.
Rule: the 300 rows with the lowest sha256("log|token") among the rows of that rank whose log has a local nuPlan DB (482 of 1,075 logs; see HANDOFF). Same rule for both ranks, so the two samples share 188 of 300 tokens.
Metric: sim8 DAC violation of the 8-pose plan executed by NAVSIM's LQR + bicycle (`navsim_dac.simulated_states`, `violations`). Interval: 10,000 log-cluster bootstrap draws, seed 20261111 (this answers only "another draw of logs", not training or inference variance).

| sample | n (logs) | raw traj sim8 fail | Tc20s sim8 fail | raw - Tc20s | 95% CI | raw-only / tc-only / both |
|---|---|---|---|---|---|---|
| rank-1 | 300 (184) | 5.33 % (16) | 0.67 % (2) | +4.67 pp | [2.33, 7.21] | 14 / 0 / 2 |
| rank-0 control | 300 (190) | 3.67 % (11) | 0.67 % (2) | +3.00 pp | [1.06, 5.28] | 9 / 0 / 2 |

Reading: the repair has the same direction and no regression on rank-1 (0 rows made worse, in both ranks). Rank-1 twins fail more often when raw and are repaired at least as much. Absolute rates are lower than P5-S's held-out set (-7.07 pp), a different population (all-bank rows with a local DB, not the 11 held-out logs), so the magnitudes are not comparable with P5-S.
Limits: n=300 per rank (16 and 11 raw failures, small counts); sample restricted to logs with a local DB; two-sided difference between ranks not tested.
Controls: identity of the labeller vs banked P5-S labels 100/100 bit-identical, max abs diff 0.0 (`CONTROL_P5S_BITIDENT.json`); 0 errors in both runs.
