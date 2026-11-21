# HANDOFF -- LONHEAD_C phase 2 (refcv9 LONHEAD phase-2 agent, 2026-10-26)

**FINAL (16:05 CET): phase 2 COMPLETE -> `result/LONHEAD_VERDICT.{json,md}` = FAIL_F (head better, F1 and F2 FAIL at s0 and s1; G PASS; A2 separated, A1 not; controls ALL_PASS). Manifest `MANIFEST_LONHEAD_PHASE2.sha256`. Next lever (prereg s5): class-conditioned progress head on the fresh split -- blocked on an MM-sealed prereg. Everything below is the mid-run state, kept for the record.**

## State
* Registered hashes asserted (prereg 6c81d4b0, AMENDMENT1 715b9df1, LH_FROZEN 6fd3ea1e, DER_FROZEN 08d1085d, LH_FIT_INFO a7e53f5e, lh_fit.py fb330f7f).
* **K-HEAD = GO** (`result/K_HEAD.json`): LH renormalised decel AUC **0.7305 [0.6749, 0.7855]** vs reference 0.685 (native reproduced 0.6851 on the
  same 2,495 union windows, 139 episodes; T0; episode-resampling bootstrap B 2000 seed 0). LH - native +0.0454 [-0.0125, +0.1054] (descriptive).
* Labels built (`labels/labels_{LHS,DER}.npz`, sha in `labels/labels.sha256`, info `labels/LABELS_INFO.json`): LH / DER argmax on all 4,754 roll
  windows; eval z_tac is bit-identical across inference seeds 0/1 (asserted) so one label file serves both LHS seeds (as SELFSHARP's s1 did).
* Composer validated (`result/COMPOSER_VALIDATION.json`): SF.compose + RS.retime_plan on the SELFSHARP traces reproduces the LANDED
  FACTORED_DEPLOY8 bit for bit (plan8 / pos8 / S6 pos) at s0 and s1; BASE-os == OSONLY_DEPLOY8; mutation (ssos) detected.

## Running (detached; survive this session)
1. `code/chain_lh.sh` (launched 14:07): GPU rolls one at a time under the shared `gpu_lock.py` lock: LHS s0 -> LHS s1 -> DER s0 (~35 min each),
   traces to `F:/refcv9_lonhead/rolls/trace_{LHS_s0,LHS_s1,DER_s0}.npz`. Markers `ZZLH*ZZ` in `logs/chain_lh.log` (read only the part after the
   "restarted for the live run" line; the earlier block is the dry run). Done = `ZZLHALLROLLSZZ`.
2. `code/post_lh.sh` (launched 14:14): waits for ALLROLLS, then `code/sub_fx_lhs/run_lhs_fact8.sh` (the SHIP-R8 Amendment-5 forward + TAC + verdict on
   the LHS traces; outputs `fwd_lhs/`; ~10 min CPU), then `code/lh_bars.py` -> `result/LONHEAD_VERDICT.{json,md}`. Markers `ZZPOST*ZZ` in
   `logs/post_lh.log`; done = `ZZPOSTENDZZ ... stage=done` with a `ZZPOSTBARSZZ` line.

## How to check
`tail -3 logs/chain_lh.log; cat logs/post_lh.log; grep -c windows logs/roll_LHS_s0.log` (do NOT print roll logs: the upstream roll code writes
12-char clip-id prefixes into them -- never land them).

## Exact next step
When `ZZPOSTBARSZZ` is present: read `result/LONHEAD_VERDICT.md`, check `controls_ALL_PASS`, write `MANIFEST_LONHEAD_PHASE2.sha256` over the
deliverables (code/*.py, code/*.sh, code/sub_fx_lhs/*, labels/*, result/*, fwd_lhs/out/ship_r8/*, fwd_lhs/out/forward/*, fwd_lhs/out/arms/**/DEPLOY8_POS_*,
logs/chain_lh.log, logs/post_lh.log, logs/k_head.log, logs/lh_bars*.log), hand back. If a stage failed: the failing marker names the log.
