# VERDICT STOPPERC stage 1, arm A: SMOKE (code check, never a result): PERC-NULL  (SMOKE: a code check with stand-in arms, never a result)

Tier: **T1-kin (self-action open loop; m = 10, causal speed bucket, dead-reckoned nav, logged scene; deterministic planner), eval-141; a planner diagnostic, never driving performance**. Estimator: taniteval.ci.paired_episode_cluster_bootstrap (cells, rates) and paired_stat_cluster_bootstrap (AUROC), B=2000, seed 0, episodes resampled with both arms on the same draw. It answers 'another draw of EPISODES' only; 'another training run' is the floor max(|NULL_rep - NULL_s0|, |NULL_rep2 - NULL_s0|). ci.py loaded from F:/TanitAD-scratch/stopbal_read/tree/taniteval/taniteval/ci.py.
Evidence class: MEASURED (ours; dumps, zero training). Code of record 3e264bbf. NULL reuse state: **SMOKE (stand-ins)**. STRATEGIC: **PAUSED** (PI item 25), not a pass, not omitted.
Controls: 0 problem(s).
Control 5 (the chain's G-SUITE / the Master Mind's pre-launch suite step) is NOT checked here.

## Outcome: **PERC-NULL**
* holding predicates: ['PERC-NULL']
* committed next step: the lever is eliminated with its row. Next: L-DAG if S0-L passed, otherwise the 416x1024 source (Q5)
  * PERC-NULL: the lever is eliminated with its row. Next: L-DAG if S0-L passed, otherwise the 416x1024 source (Q5)

## The four gates, per seed (delta = A minus the same-seed NULL; floor = max of the two replicate gaps)

| gate | seed | NULL -> A | delta [CI] | floor | separated | verdict |
|---|---|---|---|---|---|---|
| P (tactical ADE, 117 stop-point windows, m) | s0 |  | +0.0000 [+0.0000, +0.0000] | +0.7027 | False | - |
| NR (tactical ADE, all windows, m) | s0 |  | +0.0000 [+0.0000, +0.0000] | +0.1826 | False | - |
| G-AUROC goal (STOPCUE) | s0 | 0.8294 -> 0.8294 | +0.0000 [+0.0000, +0.0000] | +0.0302 | False | - |
| lon-STOP AUROC (not a cost) | s0 | 0.8383 -> 0.8383 | +0.0000 [+0.0000, +0.0000] | +0.0017 | False | - |
| G-FP (fpr at 0.5, non-stop windows) | s0 | 0.0054 -> 0.0054 | +0.0000 [+0.0000, +0.0000] | +0.0103 | False | - |
| hit rate beside | s0 | 0.0598 -> 0.0598 | +0.0000 [+0.0000, +0.0000] | +0.0684 | False | - |
| P, fraction of the 1.5 m ceiling recovered (beside) | s0 | | -0.000 (of STOPORC's 1.5392 m: -0.000) | | | |
| P (tactical ADE, 117 stop-point windows, m) | s1 |  | +0.0000 [+0.0000, +0.0000] | +0.7027 | False | - |
| NR (tactical ADE, all windows, m) | s1 |  | +0.0000 [+0.0000, +0.0000] | +0.1826 | False | - |
| G-AUROC goal (STOPCUE) | s1 | 0.8578 -> 0.8578 | +0.0000 [+0.0000, +0.0000] | +0.0302 | False | - |
| lon-STOP AUROC (not a cost) | s1 | 0.7979 -> 0.7979 | +0.0000 [+0.0000, +0.0000] | +0.0017 | False | - |
| G-FP (fpr at 0.5, non-stop windows) | s1 | 0.0085 -> 0.0085 | +0.0000 [+0.0000, +0.0000] | +0.0103 | False | - |
| hit rate beside | s1 | 0.1197 -> 0.1197 | +0.0000 [+0.0000, +0.0000] | +0.0684 | False | - |
| P, fraction of the 1.5 m ceiling recovered (beside) | s1 | | -0.000 (of STOPORC's 1.4768 m: -0.000) | | | |

Per-seed flags: [{"P": false, "NR_cost": false, "NR_ok": true, "G_AUROC_goal_gain": false, "G_AUROC_lon_cost": false, "G_AUROC": false, "G_FP_cost": false, "G_FP_ok": true}, {"P": false, "NR_cost": false, "NR_ok": true, "G_AUROC_goal_gain": false, "G_AUROC_lon_cost": false, "G_AUROC": false, "G_FP_cost": false, "G_FP_ok": true}]

## Four families (48 cells per seed: 2 window sets x (2 bands x 11 metrics + 2 TAC accuracies)); named costs and gains are separated, on their side and above the floor

* s0: costs none; gains none
* s1: costs none; gains none
* LONGITUDINAL, LATERAL and TACTICAL are read per cell in `RESULT_STOPPERC_S1.json` (`A.seeds.sN.cells`, `G_FP`, `hit_beside`). STRATEGIC: **PAUSED** (PI item 25).

## Floors (card section 8: max of the two replicate gaps)

* P cell floor +0.7027 m, NR cell floor +0.1826 m; hit +0.0684, fpr +0.0103, AUROC goal +0.0302, lon +0.0017
* P floor from the two replicates: |NULL_rep - NULL_s0| +0.7027, |NULL_rep2 - NULL_s0| +0.7027
