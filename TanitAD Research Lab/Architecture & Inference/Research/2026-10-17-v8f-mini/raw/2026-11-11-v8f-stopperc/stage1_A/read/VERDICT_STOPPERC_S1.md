# VERDICT STOPPERC stage 1, arm A: PERC-NULL

Tier: **T1-kin (self-action open loop; m = 10, causal speed bucket, dead-reckoned nav, logged scene; deterministic planner), eval-141; a planner diagnostic, never driving performance**. Estimator: taniteval.ci.paired_episode_cluster_bootstrap (cells, rates) and paired_stat_cluster_bootstrap (AUROC), B=2000, seed 0, episodes resampled with both arms on the same draw. It answers 'another draw of EPISODES' only; 'another training run' is the floor max(|NULL_rep - NULL_s0|, |NULL_rep2 - NULL_s0|). ci.py loaded from F:/TanitAD-scratch/stopbal_read/tree/taniteval/taniteval/ci.py.
Evidence class: MEASURED (ours; dumps, zero training). Code of record 3e264bbf. NULL reuse state: **DIFFERS**. STRATEGIC: **PAUSED** (PI item 25), not a pass, not omitted.
Controls: 0 problem(s).
Control 5 (the chain's G-SUITE / the Master Mind's pre-launch suite step) is NOT checked here.

## Outcome: **PERC-NULL**
* holding predicates: ['PERC-NULL']
* committed next step: the lever is eliminated with its row. Next: L-DAG if S0-L passed, otherwise the 416x1024 source (Q5)
  * PERC-NULL: the lever is eliminated with its row. Next: L-DAG if S0-L passed, otherwise the 416x1024 source (Q5)

## The four gates, per seed (delta = A minus the same-seed NULL; floor = max of the two replicate gaps)

| gate | seed | NULL -> A | delta [CI] | floor | separated | verdict |
|---|---|---|---|---|---|---|
| P (tactical ADE, 117 stop-point windows, m) | s0 |  | +0.1530 [-0.0777, +0.3854] | +0.7436 | False | - |
| NR (tactical ADE, all windows, m) | s0 |  | -0.0009 [-0.0423, +0.0423] | +0.1590 | False | - |
| G-AUROC goal (STOPCUE) | s0 | 0.8306 -> 0.8360 | +0.0054 [-0.0017, +0.0125] | +0.0398 | False | - |
| lon-STOP AUROC (not a cost) | s0 | 0.8429 -> 0.8398 | -0.0030 [-0.0101, +0.0050] | +0.0245 | False | - |
| G-FP (fpr at 0.5, non-stop windows) | s0 | 0.0054 -> 0.0067 | +0.0013 [-0.0013, +0.0044] | +0.0116 | False | - |
| hit rate beside | s0 | 0.1026 -> 0.1111 | +0.0085 [-0.0463, +0.0631] | +0.0427 | False | - |
| P, fraction of the 1.5 m ceiling recovered (beside) | s0 | | -0.102 (of STOPORC's 1.5392 m: -0.099) | | | |
| P (tactical ADE, 117 stop-point windows, m) | s1 |  | -0.0277 [-0.2194, +0.1630] | +0.7436 | False | - |
| NR (tactical ADE, all windows, m) | s1 |  | +0.0029 [-0.0383, +0.0435] | +0.1590 | False | - |
| G-AUROC goal (STOPCUE) | s1 | 0.8566 -> 0.8599 | +0.0034 [-0.0031, +0.0110] | +0.0398 | False | - |
| lon-STOP AUROC (not a cost) | s1 | 0.8179 -> 0.8170 | -0.0009 [-0.0102, +0.0110] | +0.0245 | False | - |
| G-FP (fpr at 0.5, non-stop windows) | s1 | 0.0063 -> 0.0081 | +0.0018 [-0.0009, +0.0053] | +0.0116 | False | - |
| hit rate beside | s1 | 0.1368 -> 0.1197 | -0.0171 [-0.0504, +0.0118] | +0.0427 | False | - |
| P, fraction of the 1.5 m ceiling recovered (beside) | s1 | | +0.018 (of STOPORC's 1.4768 m: +0.019) | | | |

Per-seed flags: [{"P": false, "NR_cost": false, "NR_ok": true, "G_AUROC_goal_gain": false, "G_AUROC_lon_cost": false, "G_AUROC": false, "G_FP_cost": false, "G_FP_ok": true}, {"P": false, "NR_cost": false, "NR_ok": true, "G_AUROC_goal_gain": false, "G_AUROC_lon_cost": false, "G_AUROC": false, "G_FP_cost": false, "G_FP_ok": true}]

## Four families (48 cells per seed: 2 window sets x (2 bands x 11 metrics + 2 TAC accuracies)); named costs and gains are separated, on their side and above the floor

* s0: costs none; gains ['sp|TAC-LON', 'all|operative_0_2s:dk_hw_abs_err_m', 'all|operative_0_2s:dk_tg_abs_err_s']
* s1: costs ['sp|operative_0_2s:heading_mae_deg', 'sp|TAC-LAT']; gains ['sp|tactical_2_6s:cross_mae_m']
* LONGITUDINAL, LATERAL and TACTICAL are read per cell in `RESULT_STOPPERC_S1.json` (`A.seeds.sN.cells`, `G_FP`, `hit_beside`). STRATEGIC: **PAUSED** (PI item 25).

## Floors (card section 8: max of the two replicate gaps)

* P cell floor +0.7436 m, NR cell floor +0.1590 m; hit +0.0427, fpr +0.0116, AUROC goal +0.0398, lon +0.0245
* P floor from the two replicates: |NULL_rep - NULL_s0| +0.7436, |NULL_rep2 - NULL_s0| +0.0995

## Master Mind context (added 2026-11-13 ~22:30, after the read; nothing above is changed)

**The outcome stands as written: PERC-NULL.** At both seeds none of the four gates separates (controls: 0 problems). By the card the lever is eliminated with its row. One W (0.0533) was used for both seeds, as the card fixes it.

**What the read can and cannot see (power).** The P floor is max(|NULL_rep − NULL_s0| = 0.7436 m, |NULL_rep2 − NULL_s0| = 0.0995 m) = **0.7436 m**. STOPBAL's P floor was 0.7027 m (`stopbal/read/RESULT_STOPBAL.json`). On the 117 stop-point windows, one training run against another moves tactical ADE by about 0.7 m, half the 1.5 m STOPORC ceiling. With one seed per arm, P can certify only an effect larger than about half the ceiling. That is a property of this rig (H-ESTIM-SEED-1 class), not of arm A. The AUROC gate is far less noisy and says the same thing: STOPCUE dAUC +0.0054 [−0.0017, +0.0125] (s0) and +0.0034 [−0.0031, +0.0110] (s1). Neither is separated even on the episode interval, and both are an order of magnitude below the AUROC floor 0.0398. Arm A did not move stop perception at either seed.

**Why this points at the encoder (HYPOTHESIS, consistent with a second instrument).** Arm A's head is read by nothing at inference: its gradient shapes the shared ADAPTER on top of FROZEN camera tokens (card section 5). The proof programme's FULLRES read (H-V8F-PROOF-14, T0 frozen-token probes, 2026-11-13) found no stop cue above ego motion in the frozen tokens even at full resolution (CUE-ABSENT-FULLRES; C640 minus E −0.0619 [−0.1308, +0.0024]). A probe negative is not a learnability negative (TRAPS). But if the cue is weak in the frozen tokens, an adapter-level auxiliary has little to amplify. The lever that changes the tokens is the encoder, which is exactly the card's next step.

**Next lever (card, PERC-NULL row): S0-L, then L-DAG if S0-L passes.** S0-L is zero training: STOPCUE on the LoRA-tuned encoder L vs B-prime s0, Amendment 5 (GAIN iff the paired dAUC point > 0.037 AND CI lower > 0).
* **Blocked on:** L_s0 still training on the LoRA pod (step 12,000 of 19,164 at 05:47 UTC on 11-12, ~34 s/step; ends about 2026-11-15, ESTIMATED).
* **Unblocked and started now:** the S0-L work items that do not need L (`s0l/README_S0L.md`): a T1-kin dump driver for BPRIME_s0 / L_s0 (`run_dumps_stopbal.py` is pinned to the 7 BAL tags) and B-prime s0's T1-kin dumps.
* **Still needing L:** the L-merged eval-141 cache and `check_l_ckpt.py` on the real L blob.
* **After S0-L:** L-DAG needs its own PI go (~15.3 box-2 h re-encode, `PREREG_V8F_LORA.md:155`). Q5 (416x1024 re-encode) stays deferred: the card ties it to R, which stage 0's FLAT deferred.

**Run notes (none moves a rule, an arm or a window; details in `MM_CONFIRM_S1.md` N1-N4, `FIX_S1_STOPDEC_CMD.md`, `../box2/tests/DRYRUN_FIXES.md` sections 3 and 4):**
* Two checker defects were fixed before the arms they blocked were re-run.
* NULL_rep's dev-box bank steps ran by the chain's own tail after a C:-low stop.
* The STOPDEC launcher's cmd parenthesis bug was fixed in a copy (the pinned original is unchanged).
* An unplanned same-seed replicate of A_s0 is banked on HF and is not read.
