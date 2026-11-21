# HANDOFF: MOTION and TARGET final collation (2026-11-15)

State: done. Nothing staged, committed or pushed. No pod, box 2 or GPU used. No per-arm read re-run.

## Deliverables (all in F:/TanitAD-scratch/v8proof/spatial/q2/final/)
* VERDICT_MOTION_TARGET_FINAL.md (controls first per card; PR and PE both-seed names; RES, RR, PF one seed; attribution; four-family line; six-seed spread section; limits; consistency with rows 35 to 40)
* RESULT_MOTION_TARGET_FINAL.json (every number with its source file; names per seed; R names under the six, three and two-seed spreads)
* GAC_ROW_DRAFT.md (one line, row H-V8F-PROOF-50)
* out_q3_twoseed/ (pairs PE:P, PE:HIER, PR:P, PR:G, PR:HIER, P:HIER; seeds 0,1) and out_q3_oneseed/ (PF:P, PF:HIER, PF:PR, RES:G, RES:HIER, RR:G, RR:HIER; seed 0), each with VERDICT_Q3.md and RESULT_Q3.json; reader q3_read.py unchanged, controls q1 to q5 PASS
* STEP_CTX_FINAL.md/.json (ctx_step.py on DIAG_CTX files of all arms, same tool as the banked STEP_CTX; G, P rows equal the banked values)
* collate_final.py, make_verdict.py (rebuild the JSON and the verdict from the result files)

## What was read from where
Lead, change, ADD17 variants: spatial/q2/out_m, out_t, out_p, out, ctx. Step: C:/Users/Admin/v8proof_c/ctxroll/DIAG_CTX_<ARM>.npz. Position and spread: spatial/s1/RESULT_S1_<ARM>.json and OOF_S1_<ARM>.npz (paired difference recomputed with s1_read.paired_from; P_s0 minus G_s0 reproduces the banked +0.0104). Family: T1 dumps in hiergap/pull and C:/Users/Admin/v8proof_c/pull. Build checks: spatial/motion/checks, spatial/target/checks.

## Missing or inconsistent (named, not re-run)
* x1 control of the rollout read (reproduce the banked pixel rollout) is not evaluated for PR, PE, PF, RES, RR: the pixel job was not run (RESULT_CTXROLL_<ARM>.json says available false and the tool prints PASS). It would come from the pixel read (latvis_gw family / gw_read_arm.sh). Evaluated for P_s0 only.
* K, the token read of PF: missing, same pixel job.
* ADD19 V reference, the ego block alone on the same eval windows: not produced by rb_change.py; only the train-probe value (+0.291) exists.
* Per-arm pod gate PASS tokens: pod files, not checked (no pod access).
* Not trained: ROLL_s0, ROLL_s1, RR_s1, RES_s1, PF_s1, PEF_s0, PEF_s1. So no attribution sentence for RR, no both-seed name for RES, RR, PF.
* The S read of ADD19 (paired one-step difference) did not exist in any file; computed here with the ctx_step.py resampling.
* PF_s0 oddity reported as read: R3g latent skill -13.29 and lead read R3g minus copy -0.871 (context grown from its own predictions).

## Scratch
Created: F:/TanitAD-scratch/v8proof/spatial/q2/final/tmp/ (empty). Nothing else temporary; no deletions made. The folder out_q3_* and the .py files are deliverables.

## Not done / for the lead
Register row H-V8F-PROOF-50 is a draft for the lead to land. Row 33 sentence about the two flat levers needs no change.
