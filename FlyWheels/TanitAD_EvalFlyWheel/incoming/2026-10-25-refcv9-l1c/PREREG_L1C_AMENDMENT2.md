# PREREG_L1C — AMENDMENT 2 (operational; sealed BEFORE any fresh clip is rolled and before any fresh z_tac exists)

*Architecture & Data agent, 2026-10-25. Parents: `PREREG_L1C.md` sha256 74b8ba88... (unedited) and `PREREG_L1C_AMENDMENT1.md` (sealed 2026-10-25T13:29:38+0100; its header
text "~14:30 Berlin" is a narrative typo, the seal time is the file's). State at sealing: the 300 fresh clips are FETCHED and their episode caches are being built; no label, clock row,
v9 row, window list, roll, z_tac, head fit or score of a fresh clip exists. Nothing here moves a bar, an arm, a control value or an outcome. It writes down how the parent's
words are implemented, so that no choice is made after a number is seen. The code that implements it is hashed in `CODE_SHA256_AT_AMENDMENT2.txt`.*

## A2.1 The split rule applied literally
`F:/l1c/select/final_ids_fresh.json` = the first 300 candidates in sha256(clip_id) order (all 330 fetched candidates fetched with 0 failures, mp4 sha256 re-verified, 12 frames decoded
non-zero). If a clip fails ANY later step of its chain (episode build, label emission, clock-sidecar row, v9 row, dataset index) it is logged by sha12 with the reason, removed, and the
next candidate in order (301, 302, ...; already fetched) takes its place and goes through the same chain. The split is the first 300 whose whole chain completes. n < 300 is reported as n.

## A2.2 Platform control (A1.5, D-platform) — windows and comparison
The 24 first-by-sha256 eval139 clips (`F:/l1c/select/platform_ids.json`) are rebuilt on this box and rolled on the **banked eval139 windows** of those clips
(`windows_banked24.json`, 824 windows, from the 48,852 per-frame battery's `windows_s0.json`; the fresh split's own grid is `t % 5 == 0` and is NOT used here), inference seed 0.
Their shipped counterpart is the L1 eval bank `ztac_eval_s0.npz` (z_tac is bit-identical across inference seeds, L1 K6), joined on (clip sha12, ws).
Two of the 24 clips have no clock-sidecar row (refused by the builder's K1/K3 rules); the SHIPPED sidecar refuses the same two, so they are rolled as shipped.
**D-platform** = AUC of a PCA-32 logistic separating shipped from rebuilt z_tac on identical windows with episode folds shared by both classes, plus the relative z_tac difference
(against the same ratio between two random shipped windows) and the per-window mean slot distance between the two plans. Reading rule (A1.5): D-platform AUC > 0.70 => D-fresh is
reported "platform-confounded". No bar.

## A2.3 Controls as implemented (any failure => outcome VOID, reported as VOID with the failing control named; no O1/O2/O3 label is given)
* K1: H0 minus ha0_ext_pose (S2 compliant, ADE) equals the banked RETIME_TRAINFIT cell AND the prereg literals -0.0343 / -0.0330 within 1e-4.
* K2: CF2 - CF2s on S2 recomputed with `l1_followup.cross_fit` equals the prereg literals -0.0044 / -0.0045 AND the L1 `FOLLOWUP.json` values within 1e-4.
* K3: the constant head (mean of the fresh targets) has max eval R2 <= 1e-12 on every instant at both seeds.
* K4: the fresh bank's own plan equals the roll's own os plan (`nav_true.traj` of the dump) with max abs <= 1e-5 m (the value is recorded; the prereg's "0.0 m" is the expected reading);
  the bank is final, has the expected window count, is finite and has no all-zero row; the eval banks reproduce their dumps as in L1.
* K5: the fresh bank joins every fresh panel row; the bank's v0 equals the panel v0 (< 1e-3); the panel decisions and validation checks PASS.
* V-m: H2cm (H2c with eval z_tac deranged across episodes) is NOT separated below H2cs at either seed.
* Arm identities: the gain-1 re-timing equals os exactly; the zero-progress prior equals the human-anchored pose prior.
* Braking-AUC controls C1-C5 of `l1c_roc.py` (GT AUC 1, constant 0.5, banked rates and AUCs reproduce).

## A2.4 Bars as implemented (the parent's words, made mechanical in `l1c_verdict.py`)
* P1: paired H2c - H2cs ADE cell on S2 compliant windows has hi < 0 at BOTH inference seeds and |delta| > |inference-seed floor| (OS s0 vs s1, S2 ADE).
* P2: paired H2c - H0 ADE cell has hi < 0 at BOTH seeds.
* G: over every metric of the four-family table for H2c - H0 (S2): a row is "worse at both seeds" when the cell is separated in the worse direction at both seeds (lower-is-better metrics:
  lo > 0; the higher-is-better rows `TAC_*` and `LON_speed_within_0p5`: hi < 0); G fails if any row is worse at both seeds. Braking AUC: `l1c_roc.py` (the L2 `l2_roc.py` definition, unchanged)
  on the S6 compliant windows; G fails if H2c's AUC minus H0's AUC has a paired CI < 0 at both seeds ("AUC-DOWN"). H0 for this purpose is the frozen per-frame RETIME applied to the os plan
  (arm `H0`); the AUC against the banked stage is reported beside and does not enter G. Distance keeping and strategic state: n = 0 with the reason, as in L1.
* Outcome: VOID if a control failed; else O1 = P1 and P2 and G; O2 = P1 and not (P2 and G); O3 = not P1, read with D-fresh (> 0.70 flagged) as the parent states.
* Estimator on every cell: `pair_read_r8.paired_cell` -> paired episode-cluster bootstrap, B 2000, seed 0, models fixed (answers "another draw of EPISODES"; blind to training variance and
  inference variance; ONE training seed, so a separated cell is necessary and not sufficient for a lever claim, H-ESTIM-SEED-1).

## A2.5 Fit
Rows: the fresh S2 ceiling-compliant windows (mask: `compliant_reread_r8`, model-free: gt_max_speed_6s vs the fed ceiling). Heads: `l1_fit_score.cv_head` / `head_fit` / `PS.cv_lambda` / `PS.ridge_fit`
with PREREG_L1's grids (k in {4,8,16,32,64,128,256}, lambda in {1e-2 .. 1e4}), 5-fold EPISODE CV, fold seed 0, fitted on the fresh rows only. S6 heads are fitted on the fresh S6 compliant rows
(secondary surface, feeds the braking-AUC bar). The eval rows are scored, never tuned on. Branch PRIMARY: all three `v_lim` features are in fit and scoring.

## A2.6 Sensitivities committed now (BESIDE: no bar, never relabel the verdict; each is a refit on a subset of the fresh rows scored on eval139 S2 compliant at both seeds)
* S-r0: fresh clips whose own egomotion passes the r0 hard gates (A1.1).  * S-rig A / S-rig B: the fresh clips of each camera rig.
* Learning curve: the first 75 / 150 / 225 of the 300 clips in sha256 order (it prices "more clips" in the O2/O3 next-lever choice).
* Speed distribution of the fresh compliant rows against eval139's (quantiles of v0, share above 14 m/s).
Subsets with fewer than 300 rows or 15 episodes are skipped and reported as skipped.

## A2.7 D-fresh
Domain AUC (PCA-32 logistic, 5-fold episode CV, `l1_followup.domain_auc` unchanged) of fresh compliant S2 z_tac vs eval139 compliant S2 z_tac (seed 0). Flag threshold 0.70 (parent).

## A2.8 Next lever (Rule Zero), committed before the data
Whatever the outcome, the report names the next lever and, unless it needs compute/corpus/PI authority not held, starts it. Candidates, in order of the arithmetic the result gives:
O1 -> the SHIP-R8 battery request on the new stage; O2 -> the nonlinear head (the L1 `H2n` recipe) fitted on the same fresh split, then the learning-curve-implied number of further fresh clips;
O3 with D-fresh flagged -> a distribution-matched fresh split (rig/speed-matched draw from the same universe); O3 with D-fresh not flagged -> close L1 for refcv8 and carry the progress head
to the refcv9 design as a calibrated-split object rather than a stage upgrade. VOID -> the failing control is the next lever.
