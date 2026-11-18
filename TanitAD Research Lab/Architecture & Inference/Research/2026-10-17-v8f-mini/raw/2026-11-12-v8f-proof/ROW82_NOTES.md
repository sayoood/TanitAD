# ROW82_NOTES: sources of every number in GAC_ROW_82.md

Paths: SQ = F:/TanitAD-scratch/v8proof/spatial/q2 ; CR = C:/Users/Admin/v8proof_c/ctxroll ; S1 = F:/TanitAD-scratch/v8proof/spatial/s1.
Card hash: `sha256sum PREREG_V8F_PROOF_ADD19.md` = 1255463b6f9e2f27e238257500c748c1ecd93f403adba5c43acbdda668800b32. Form mirrored from GAC_ROWS_39.md line 1 (H-V8F-PROOF-50).

## I, level at +2.0 s (SQ/out_t)
* Rolled-out / copy / ceiling R2, seed 0: VERDICT_RB_PF_s0.md "Lead gap" (0.375 [0.173, 0.516], 0.458, 0.503); seed 1: VERDICT_RB_PF_s1.md (0.409 [0.180, 0.571], 0.521, 0.596). Keys in RESULT_RB_PF_s{0,1}.json: `rb`, `i_minus_ii`, `i_minus_floor`, `iii_minus_i`.
* Rollout minus copy: VERDICT_RB_PF_s0.md -0.082 [-0.206, +0.018]; VERDICT_RB_PF_s1.md -0.112 [-0.220, -0.027].
* Rollout minus ego floor: same files, +0.197 [-0.023, +0.495] / +0.231 [+0.005, +0.529].
* Rollout minus rollout of P: PAIR_RB_PF_s0_minus_P_s0.md (-0.125 [-0.247, -0.019]); PAIR_RB_PF_s1_minus_P_s1.md (-0.097 [-0.236, +0.011]).
* Outcome line of each verdict file reads `IMAGINE-FLAT` per seed (the old per-seed label); the ADD19 name TARGET-FLAT is by section 4 of the card applied to the intervals above.

## V, change (SQ/out_t)
* CHANGE_RB_PF_s0_P_s0.json/.md: PF_s0 rollout -0.009 [-0.123, +0.090], ceiling +0.189 [+0.074, +0.324], pair PF minus P -0.015 [-0.138, +0.094]. CHANGE_RB_PF_s1_P_s1.json/.md: PF_s1 -0.167 [-0.342, -0.045], ceiling +0.190 [+0.043, +0.329], pair -0.222 [-0.406, -0.107].
* Controls k1 to k3 PASS in both files (PF and P). The "beside" ego block alone of section 4 V was not copied (not looked up in these files; names nothing).

## S, step
* PF_s0 0.914 [0.896, 0.933], P_s0 0.788 [0.770, 0.807], P_s1 0.796 [0.779, 0.814]: SQ/final/STEP_CTX_FINAL.md. PF minus P at seed 0 +0.126 [+0.119, +0.134]: SQ/final/VERDICT_MOTION_TARGET_FINAL.md section 2.3.
* PF_s1 0.905: CR/RESULT_CTXROLL_PF_s1.json key `one_step_mse.ratio_context_1_over_copy` = 0.9047 (point value only). **MISSING:** bootstrap interval and the PF_s1 minus P_s1 pair (STEP_CTX_FINAL has no PF_s1 row; ctx_step.py not run for it). Named STEP-SAME on seed 0 alone because STEP-IMPROVES needs both seeds. Seed 1 point value is above P_s1 (0.905 vs 0.796), consistent.

## R, position
* Seed 0 pair +0.003 [-0.015, +0.021], spread 0.0450: SQ/final/RESULT_MOTION_TARGET_FINAL.json `position_read.pairs.PF_s0_minus_P_s0`, `position_read.spread_six`.
* Seed 1 point values: S1/RESULT_S1_PF_s1.json and RESULT_S1_P_s1.json key `main.r_a.estimates.gap|s` (PF 0.5358 [0.4943, 0.5738], P 0.5373 [0.4939, 0.5746]). **MISSING:** the paired PF_s1 minus P_s1 difference (no PAIR_S1_PF_s1 file; collate_final.py pairs did not include it). Not computed here (rule 4). R is therefore not named at both seeds. Controls t1 to t4: VERDICT_S1_PF_s1.md PASS; RESULT_S1 controls_pass True for PF_s0, PF_s1, P_s0, P_s1.

## F, families and headway (SQ/pool4/read_pf2)
* q3_pf2/RESULT_Q3.json (`pairs.PF_minus_P`), VERDICT_Q3.md: name NO-COST GAIN-TACTICAL, all cells and per-seed values copied from its table; controls ALL_PASS q1 to q5; estimator and counts from its header.
* hw_pf2/RESULT_HEADWAY.json, VERDICT_HEADWAY.md: cell U -0.0160 [-0.0455, +0.0101] (per seed -0.0249 / -0.0071), time-gap error 2.0 s -0.1917 [-0.4233, -0.0436]; h1 to h6 ALL_PASS; "NOT NAMED (no question of the card for this pair)".
* Only PF minus P is in these files; no PF minus HIER pair for two seeds was read (row 50 reported HIER for one seed). Not claimed.
* The 0.0492 spread for joint error rate is the "spread max" column of VERDICT_Q3.md.

## Controls and run history
* t1 to t6, k1 to k3: out_t verdict/change files. x2 to x4 PASS, x1 not available: CR/RESULT_CTXROLL_PF_s{0,1}.json `controls`. e1 PASS and build checks and "gate tokens not re-checked": SQ/final/VERDICT_MOTION_TARGET_FINAL.md lines 204 to 208 (those were verified for PF_s0 and the shared build; the build is not re-verified for PF_s1 here).
* Run history: SQ/pool4/ops/wait_target_d.out (ZZ READP STOP PF_s1 collect, AssertionError import-path, 2026-11-16), redo_lead_read_pf_s1.sh header (junction made 2026-11-18, call unchanged), redo_lead_read_pf_s1.out (ZZ CTXROLL ARM DONE PF_s1 21:38:57Z, ZZ REDO-LEAD rc 0). Lead read files mtime 2026-11-18 22:36 to 22:37 (local time, UTC+1). The Q3 and headway files date from the earlier read of 2026-11-16 and did not depend on the failed step.

## Missing or doubtful
1. K (token read): pixel job not run. Row says "not read".
2. S at seed 1: interval and pair not read; R at seed 1: pair not read. The row names S by seed 0 alone and does not name R.
3. x1 not evaluated; per-arm pod gate token not re-checked for PF_s1 (no pod access); build checks for PF_s1 inherited from the shared build record.
4. The 48 episodes on the change read is quoted from row 50 (CHANGE files for PF show 441 windows; CHANGE_CTX_PF_s1 shows 471 windows, 49 episodes for the R1 variant, a different window set).
5. "Next step" wording follows card section 6 bullet 3; its sentence "together with stage 0c" was not repeated because stage 0c was not read here.
6. History-seeded variant (R3h minus R1) of PF_s1 exists (SQ/ctx/PAIR_RB_PF_s1_R3h_minus_R1.md, +0.121 [+0.059, +0.208]) and was left out: it names nothing and the row 50 form carries it only for seed 0 as a reference.
