## RESULT — refcv7 step30000 (2026-10-04T06:58:23+0200)

* **G0 as registered: FAIL** — eval_agent_det_ap2_bus_40_60 [DETECTION] in-run 0.1 vs seeds mean 0.125 (sd 0); eval_box3d_det_ap2_bus_20_40 [DETECTION] in-run 0.25 vs seeds mean 0.33333 (sd 0); eval_box3d_det_ap2_stroller_0_20 [DETECTION] in-run 0.16667 vs seeds mean 0.14286 (sd 0)
* **G0-A2 (seeds 0..7): FAIL** — eval_tacv6_goal_conf_bce [SMOOTH] in-run 0.15189 vs seeds mean 0.1542 (sd 0)
* **G0-A5 (THE GATE, SPEC A5; 24 inference seeds): FAIL**; mutation terms moved: M1 7 (detected), M2 59 (detected), M4 2 (detected); wrapper clause PASS; reasons: ['eval_tacv6_goal_conf_bce [SMOOTH] in-run 0.15189 vs seeds mean 0.1542 (sd 0)']
* **Tier:** T1 self-action OPEN LOOP (status UNRULED for an action-free model); never closed loop, never driving performance. **Estimator:** FULL-SET point estimate; paired episode-cluster bootstrap, n_boot 2000, seed 0.
* **Inference-seed floor:** None m ADE (training-seed floor NOT measured).

| bar | verdict | seed 0 | seed 1 |
|---|---|---|---|

| BAR-R7-N1 (NavSim) | not this battery | — | — |

**Four families, paired `os − ha0_ext` (seed 0):** 
* strategic: NOT APPLICABLE (n 0): strategic layer OFF
