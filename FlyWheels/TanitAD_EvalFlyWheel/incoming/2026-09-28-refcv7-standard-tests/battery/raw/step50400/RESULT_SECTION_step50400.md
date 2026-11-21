## RESULT — refcv7 step50400 (2026-10-04T14:49:14+0200)

* **G0 as registered: FAIL** — eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0)
* **G0-A2 (seeds 0..7): PASS**
* **G0-A6 (THE GATE, SPEC A6; 24 inference seeds): FAIL**; mutation terms moved: M1 22 (detected), M2 62 (detected), M4 2 (detected); wrapper clause PASS; reasons: ['eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0)', 'eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0)', 'eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0)', 'eval_agent_det_ap2_stroller_0_20 [DETECTION] in-run 0.33333 vs seeds mean 0.25 (sd 0)', 'eval_agent_det_ap2_stroller_all [DETECTION] in-run 0.16667 vs seeds mean 0.125 (sd 0)']
* **G0-A5 (24 seeds; reported beside the A6 gate): PASS**
* **G0-A6 (REGISTERED; measured numerics floor): FAIL** — eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0); floor-rescued terms: []; threshold-target terms: eval_tacv6_goal_conf_bce in-run 0.14591 interval [0.14224843595836026, 0.18356920461913012] OK
* **Tier:** T1 self-action OPEN LOOP (status UNRULED for an action-free model); never closed loop, never driving performance. **Estimator:** FULL-SET point estimate; paired episode-cluster bootstrap, n_boot 2000, seed 0.
* **Inference-seed floor:** None m ADE (training-seed floor NOT measured).

| bar | verdict | seed 0 | seed 1 |
|---|---|---|---|

| BAR-R7-N1 (NavSim) | not this battery | — | — |

**Four families, paired `os − ha0_ext` (seed 0):** 
* strategic: NOT APPLICABLE (n 0): strategic layer OFF
