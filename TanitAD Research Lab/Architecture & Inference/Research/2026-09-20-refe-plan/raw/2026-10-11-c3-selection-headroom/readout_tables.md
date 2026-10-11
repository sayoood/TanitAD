**Verdict by the committed rule: SCORER-BOUND.**

Population (MEASURED, `D:\Projects\TanitAD\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\raw\2026-10-11-c3-selection-headroom\tokens.json`): 529 zeroed + 529 non-zeroed tokens, 123 logs (89 carry a zeroed token); 64 proposals each.  Estimator: paired log-cluster bootstrap over 123 logs, 10000 resamples, percentile 95 %, seed 20260927; both strata on the SAME log draw.

| read | value (95 % CI) | artifact |
|---|---|---|
| (a) rescuable share of zeroed tokens | **86.0 % [80.2 %, 91.7 %]** (455/529); bar 50 % | `table.npz` -> `readout.json` a_rescuable |
| (b) actual PDMS x100, full-navtest mix | **88.28 [87.35, 89.11]** (zeroed 0.00 [0.00, 0.00]; non-zeroed 92.30 [91.33, 93.17]) | `readout.json` b_pdms_x100 |
| (b) random PDMS x100, full-navtest mix | **73.42 [71.42, 75.42]** (zeroed 29.72 [23.44, 37.20]; non-zeroed 75.40 [73.32, 77.50]) | `readout.json` b_pdms_x100 |
| (b) oracle PDMS x100, full-navtest mix | **98.16 [97.79, 98.54]** (zeroed 83.12 [77.60, 88.59]; non-zeroed 98.84 [98.52, 99.14]) | `readout.json` b_pdms_x100 |
| (b) paired actual minus random, mix | 14.86 [12.96, 16.63] (zeroed -29.72 [-37.20, -23.44]; non-zeroed 16.89 [14.90, 18.74]) | `readout.json` b_paired_differences_x100 |
| (b) paired oracle minus actual, mix | 9.88 [9.09, 10.78] (zeroed 83.12 [77.60, 88.59]; non-zeroed 6.55 [5.80, 7.39]) | `readout.json` b_paired_differences_x100 |
| (b) paired oracle minus random, mix | 24.74 [22.80, 26.67] (zeroed 53.39 [48.39, 58.87]; non-zeroed 23.44 [21.43, 25.38]) | `readout.json` b_paired_differences_x100 |
| (c) EP headroom, best safe minus pick, non-zeroed tokens with a safe proposal (529/529) | **0.1234 [0.1105, 0.1378]** EP (12.34 [11.05, 13.78] x100) | `readout.json` c_ep_headroom |

Fidelity gate: PASS (`gate_fidelity.json`).  Random < actual in the mix: True (random 73.415, actual 88.277).  Sanity: the mix-weighted actual 88.277 vs the banked full-navtest 88.247: banked value inside the mix CI = True.