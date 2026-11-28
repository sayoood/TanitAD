# PREREG_COMBO -- do the K-draw ensemble (ENS-K) and the arc-length re-timing (RETIME) STACK on refcv8's E1 error?

Written 2026-10-19T07:34+0200 (dev box), BEFORE any combined number exists.
Author: EvalFlyWheel analyst (combo stream). Status: PRE-REGISTRATION. Evidence class of everything below: DESIGN (no new measurement).

## 0. State assertions at write time (checked, not remembered)

* `raw/combo_ens_retime/` was EMPTY at 2026-10-19T07:33:34+0200 (`ls -la` listed only `.` and `..`); at the moment of writing this file it holds this file and nothing else.
* `code/combo_ens_retime_r8.py`, `code/test_combo_ens_retime_r8.py` and `code/run_combo_ens_retime_r8.sh` DO NOT EXIST yet. No COMBO cell, no stacking delta, no ENS->RETIME plan has been computed or looked at.
* The result script refuses to write if this file's sha256 differs from the value pinned in the script, and refuses to overwrite an existing result file of the same tree/gains in its output directory.

## 1. What is and is not being claimed (read first)

* This is a **lever-stacking experiment**, not a bar of record. **BAR-R8-E1 of record stays the single-draw `os`** (scored vs `ha0_ext_pose` since D-R8-E1-REF, PI 2026-10-19; `ha0_ext` beside).
  A combined arm is a NEW arm: it costs up to xK planner draws per cycle (ENS) plus zero extra GPU for RETIME (an O(S) post-processing of the plan). A combo PASS does not convert a single-draw FAIL into a PASS of the bar.
* **Tier of every number: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).** Estimator: paired episode-cluster bootstrap, models FIXED, B 2000, seed 0, 95 % percentile
  (`pair_read_r8.paired_cell` -> `taniteval.ci`). It answers "would another draw of EPISODES say this?" ONLY. The two DISJOINT draw sets A/B answer "another INFERENCE run"; nothing answers "another TRAINING run" (single training seed).
* **The two levers are NOT independent measurements of one thing; they are fitted/selected on overlapping data.** RETIME's frozen gains are fitted on EVAL episodes (below), so the 30k read with those gains is **EXPLORATORY and in-sample for the gains**. It is never a bar pass.
* **TRANSFER (named once, binds every cell):** the frozen RETIME gains were fitted on SINGLE-DRAW `os` plans (`FROZEN_GAINS_RETIME_*.json`, "plan: p_k = C_os(s'_k) ..."). Applying them to an AVERAGED plan is a transfer to a plan whose polyline is
  shorter/straighter than the mean of the draws' polylines (the mean of curves has less arc length than the mean of arc lengths) -- the gains were not fitted for that plan class. No gain is refitted on ENS plans in this package (see 4.5 for the one beside-arm that avoids the transfer).

## 2. Prior knowledge disclosed before registering (so the registration is not mistaken for blind)

SEEN (MEASURED, ours): `raw/kdraw/KDRAW_E1.json` / `.md` (EMA-30k compliant S2, 4,159 windows / 138 episodes): vs `ha0_ext_pose`: single draws -0.0136 / -0.0127 (seeds 0/1) ... ENS-2 -0.0189 [-0.0335, -0.0035], ENS-4A -0.0225 [-0.0373, -0.0071], ENS-4B -0.0210 [-0.0363, -0.0056], ENS-8 -0.0232;
vs `ha0_ext`: ENS-4A -0.0174, ENS-4B -0.0158; floor_ENS-4 0.00155 m; NR all PASS; blend flag False.
`raw/a0k0_retime/RESULTS_BRIEF.md`, `FORWARD_RETIME_step30000ema_unc.json`, `FROZEN_GAINS_RETIME_ema30k.json`: cross-fitted RETIME_XF vs `ha0_ext_pose` -0.0321 / -0.0316 (S2 compliant, seeds 0/1); the FROZEN (in-sample) RETIME on single draws -0.0320 [-0.0427, -0.0211] / -0.0315 [-0.0420, -0.0207];
RETIME - os (ADE) -0.0184 / -0.0188; S6 ACC-LAT HOLDS for single draws; frozen S2 gains g = [0.9968, 0.5834, 0.4260, 0.3825] (all-windows fit, seeds 0/1 pooled).
**NOT seen:** any RETIME applied to an averaged plan; ENS-2 on the set {4,5}; every STACKING cell; every combo NR / lateral row.
Arithmetic that frames the question (ESTIMATE, not a prediction I am committed to): relative to the single-draw `os` the levers buy about -0.0185 (RETIME) and -0.0089 (ENS-4); additive would put the combo near -0.041 vs `ha0_ext_pose`,
the better single lever sits near -0.032. Sub-additivity is expected if both mostly remove the same arc-length noise; the transfer caveat above can also produce interference. The data decide.

## 3. Arms (S2 grid, 4,754 windows, scored on the ceiling-compliant mask; plans [N, 4, 2], ego frame, 0.5/1.0/1.5/2.0 s)

Notation: `os_s` = the EMA-checkpoint's `os` plan of inference seed s (seeds 0,1 from the battery dumps, 2..7 from the K-draw rolls, all in `KDRAW_E1.json`'s seeds table); `HP` = `ha0_ext_pose` (rebuilt by the retime
script's own `surface_inputs`, the same builder FORWARD_RETIME used); `g` = the frozen RETIME gain vector for the S2 grid; `retime(P, HP, g)` = `a0k0_headroom_retime_r8.retime_apply`.

| arm | definition (per window) |
|---|---|
| **ENS-K** (alone) | `ensemble_mean` = float64 mean of the K draws' `os`, cast to float32 (`kdraw_e1_r8.ensemble_mean`) |
| **RET-alone-K** (alone) | per-window MEAN over the K draws of the ADE (and of every component) of `retime(os_s, HP, g)`; i.e. the expected SINGLE-draw RETIME in that draw set |
| **COMBO-K** (PRIMARY) | `retime(ENS-K plan, HP, g)` -- ENS first, then RETIME with the FROZEN gains, applied unchanged, ungated (the `RETIME` table) |
| beside: COMBO-G-K | the same with the `GATED_RETIME` table (v0 bin of each window) |
| beside: RT-THEN-ENS-K | float64 mean of `retime(os_s, HP, g)` over the K draws (gains applied to each draw, where they were fitted; then averaged: NO transfer). Reported to size the transfer effect: paired `COMBO-K - RT-THEN-ENS-K` |

Draw sets (S2). **K=4 is the registered K** (the K of KDRAW's bar): **A = {0,1,2,3}, B = {4,5,6,7}** (disjoint). **K=2** (descriptive, nested): **A2 = {0,1}, B2 = {4,5}** (A2 is the KDRAW ENS-2 control set).
**K=8** (all draws) is the dose-response endpoint and cannot decide a label. **K=1** = the eight single draws (RET-alone, gain 1 = `os`).
**Gain sources (never refitted here; this is the whole point of "frozen"):**
* (i) `raw/a0k0_retime/FROZEN_GAINS_RETIME_ema30k.json` (sha256 4ecde275554b1aaa...): fitted on ALL 139 EVAL episodes, seeds 0/1 pooled -> **EXPLORATORY ONLY**, label suffix `EVAL-FITTED GAINS`.
* (ii) `raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json` WHEN IT EXISTS (written by the queued train-clip fit chain together with `FROZEN_GAINS_RETIME_train.sha256`; the script refuses a file whose sha256 differs, whose `fit_source` is not `train`, or whose `fit_guard.n_overlap_with_eval139` is not 0). This is **the clean form**.

## 4. Bars (committed in advance; mechanics = `kdraw_e1_r8.py` / `pair_read_r8.py`; every cell is a paired episode-cluster bootstrap on the compliant windows)

Let `floor(X) = |mean ADE(X on set A) - mean ADE(X on set B)|` (compliant windows) -- the arm's own inference-replicate floor (the K-draw analogue of the battery's seed floor). Separated better = `delta < 0 AND hi < 0`; separated worse = `lo > 0`.

**4.1 E1 (primary, PI ruling D-R8-E1-REF):** `COMBO-4 - ha0_ext_pose`, ADE 0-2 s, sets A and B. **E1-PASS** iff on BOTH sets separated better AND `|delta| > 2 x floor(COMBO-4)` (`kdraw_e1_r8.arm_verdict`; PARTIAL/FAIL as there).
`COMBO-4 - ha0_ext` (the old echo) is reported beside with the same verdict mechanics. Not a gate of the label.

**4.2 STACKING (the question).** Per set (A, B), K=4, paired cells in ADE 0-2 s:
`ST_ens = COMBO-4 - ENS-4` (does RETIME add on top of the ensemble?), `ST_ret = COMBO-4 - RET-alone-4` (does the ensemble add on top of RETIME?). Also reported: `ST_best = COMBO-4 - (the better of ENS-4 alone / RET-alone-4 on that set, by mean ADE on that set)`.
A stacking cell **STACKS** iff on BOTH sets it is separated better AND `|delta| > 2 x max(floor(COMBO-4), floor(its comparator))`.

**4.3 ACC-LAT (the turn condition of PREREG_RETIME, applied to the combination).** The five LATERAL rows `LAT_heading_mae_deg, LAT_curvature_mae_1pm_valid, LAT_yaw_rate_mae_radps_valid, LAT_cross_mae_m, LAT_cross_abs_final_m`; error metrics.
ACC-LAT **FAILS** iff any row of `COMBO - (the unmodified ensemble plan)` is separated worse (lo > 0) at ANY of:
* (a) **S2, K=4, set A or set B** (`COMBO-4 - ENS-4`);
* (b) **S6, K=2, draws {0,1}** (`COMBO-2(S6) - ENS-2(S6)`, built from the banked `s6_s0`/`s6_s1` panels with the frozen S6 gain vector). **S6 draws exist ONLY for inference seeds 0 and 1** (the battery's S6 step; the K-draw rolls are S2-only, `KDRAW` Addendum A), so S6 ACC-LAT is **testable at K=2 only, on ONE set** (no replicate). **S6 at K=4 is UNTESTED and is named as such in every report.**
Beside (not part of the condition): the same rows vs the single-draw `os` of each draw; all-windows set. **Vacuity guard:** an ACC-LAT HOLD counts as a finding only if the combination actually moved the ADE it was applied to (`ST_ens` separated better on both sets for (a); `COMBO-2(S6) - ENS-2(S6)` ADE separated better for (b)); otherwise it is reported as "holds vacuously".

**4.4 NR vs refcv7-50,400 (all four families, `pair_read_r8.METRICS`; reported per set and per K).** Exactly `pair_read_r8.bar_NR_metric` / `rollup_NR` on the S2 headline rows (ADE, LON speed + |along| at 2 s, LAT heading + curvature + yaw rate) against BOTH banked refcv7 inference seeds
(`b_refcv7_50400_s0/_s1`, a SINGLE refcv7 draw: favourable to the arm); **secondary comparator = the refcv7 2-draw MEAN plan** (the fairer one): per headline row, FAIL iff lo > 0. TACTICAL rows are computed and printed (no NR headline row is defined for TACTICAL on S2 in `METRICS`);
STRATEGIC: NOT APPLICABLE, n = 0 (layer off; `--no-strategic` in the run's argv). S6 NR is not covered (S2 only; the S6 K=2 block carries its own rows).

**4.5 Diagnostics (descriptive, never gates):** arc-length shrinkage of the averaged plan (`L_end(ENS-K) / mean_s L_end(os_s)`, windows with all L>0) and fraction of windows whose RETIME progress extrapolates / is non-monotone (`retime_diag`); the paired transfer cell `COMBO-4 - RT-THEN-ENS-4`;
the mode-disagreement subset (spread of the K 2-s endpoints > 1.0 m, KDRAW section 5) read for COMBO-4 vs ENS-4 curvature; dose response K = 1, 2, 4, 8.

## 5. Committed outcomes (decided on K = 4, sets A and B, in this order of precedence)

1. **HARMS** iff ANY of: (h1) `ST_ens` OR `ST_ret` is **separated worse** (lo > 0) on at least one set -- reading "worse than either" as *worse than at least one deployable single lever*; (h2) ACC-LAT FAILS (4.3 a or b);
   (h3) COMBO-4 fails an NR family (single-refcv7-draw mechanics, 4.4) on a set where the corresponding ENS-4 alone PASSES that family (the retime step introduced the regression).
2. **STACKS** iff not HARMS AND both stacking cells STACK (4.2, both sets) AND E1-PASS (4.1) AND ACC-LAT HOLDS non-vacuously AND NR PASS in every family on both sets.
3. **NO-STACK** otherwise (the combo is not separated from the better single lever, or is separated better than one but not the other, or fails the 2x-floor rule). The report must say which single lever it is not separated from.

Mandatory qualifiers on the label line: `EXPLORATORY` + `EVAL-FITTED GAINS` (gain source i), or `CLEAN GAINS (train-fitted)` (gain source ii); the tree; "single training seed"; "S6 K=4 UNTESTED"; the secondary-NR (refcv7 2-draw mean) caveat if it FAILS where the primary passes.
K=2 gets the same mechanics and is printed as a **descriptive** label that decides nothing. A GATED / RT-THEN-ENS label, if printed, is **descriptive** and decides nothing.

| outcome | what it means, committed in advance |
|---|---|
| **STACKS** | the levers remove separable error terms. The E1 candidate becomes ENS-K draws + frozen RETIME gains (cost xK GPU for ENS; RETIME free). At 30k + eval-fitted gains this is "may exist"; only the registered read (section 7) counts. Next: run section 7; PI decides the K cost. |
| **NO-STACK** | they remove (mostly) the same error. Deploy the cheaper lever: **RETIME alone (zero inference cost)** is the E1 lever unless ENS-K alone separates better than RETIME alone, in which case say so; the xK cost buys nothing the gain vector does not. Next lever (named, NOT run here): a train-fitted RETIME refit on ENS plans (needs K-draw rolls on the 4,572 train clips: GPU, PI-level). |
| **HARMS** | the levers interfere. Read `COMBO-4 - RT-THEN-ENS-4`: if RT-THEN-ENS is clean, the cause is the TRANSFER (gains fitted for single-draw polylines applied to an averaged one) and the repair is per-draw RETIME then average; if not, the cause is elsewhere and is NOT explained by this experiment. Do not hand-craft a fourth rescaling to rescue the family (PREREG_RETIME standing commitment). |

## 6. Controls that must read their known value (any failure => the analysis prints VOID, never a verdict)

* `C-MASK` compliant-mask sha256 = `4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129`, 4,159 / 4,754 windows, 138 episodes (30k EMA tree; a function of the logged windows + fed ceiling, so it must hold for every tree, else VOID until explained).
* `C-ID` / `C-DIST`: every seed's `ws, eid, clip_index, g, ha0_ext, v0` bit-identical to seed 0; every pair of seeds' `os` differs by a mean L2 > 1e-4 m (the refcv6-loader RNG-reset defect would make an 8-draw ensemble one draw).
* `C-ALIGN`: the panel-built `HP`/`G`/`ws`/`clip` rows equal the dump rows (positive assertion on every array); seeds 0/1 dump `os` == panel `os` bit for bit.
* `C-R0926` (model-free, mask-free, all 4,754 windows): `ha0_ext_pose - ha0_ext` = +0.0055 [+0.0036, +0.0076] (literal, MEASURED 2026-09-26); the pose-echo compliant mean ADE = 0.2858 m (literal, KDRAW_E1).
* `C-G1`: gain vector 1 reproduces the ENS plan **bit for bit** for every K-set (array level), and the cells equal the ENS cells.
* `C-G0`: gain vector 0 puts every point on the ENS polyline at the PRIOR's chord-sum arc length -- compared with an independently authored loop-based walker (max error <= 1e-6 m on the float64 positions).
* `C-K1`: K=1 with gain 1 reproduces the banked single-draw `os` cells of `KDRAW_E1.json` -- literals: vs `ha0_ext` seed 0 -0.0085 [-0.0229, +0.0065], seed 1 -0.0076 [-0.0216, +0.0073]; vs `ha0_ext_pose` seed 0 -0.0136 [-0.0286, +0.0018], seed 1 -0.0127 [-0.0272, +0.0024] (to 4 dp).
* `C-K1R` (30k EMA tree with gain source i only; otherwise printed N/A): K=1 with the FROZEN gains reproduces the independent `FORWARD_RETIME_step30000ema_unc.json` cells for `RETIME_FROZEN30k` -- literals vs `ha0_ext_pose` seed 0 -0.0320 [-0.0427, -0.0211], seed 1 -0.0315 [-0.0420, -0.0207]; vs `ha0_ext` seed 0 -0.0269 [-0.0373, -0.0161], seed 1 -0.0264 [-0.0367, -0.0158].
* `C-ENS` (30k EMA tree only): ENS-2 {0,1}, ENS-4A, ENS-4B alone reproduce the `KDRAW_E1.json` pose cells -0.0189 [-0.0335, -0.0035], -0.0225 [-0.0373, -0.0071], -0.0210 [-0.0363, -0.0056] (literals); other trees: equal to `raw/kdraw/<tree>/KDRAW_E1.json` if present, else UNVERIFIED (printed, verdict carries the word).
* `C-ONPATH`: every COMBO point lies on the ENS polyline (max distance <= 1e-6 m, float64) -- `retime_diag`. `C-TRI`: ADE(ENS-K) <= mean of the draws' ADE on every window (analytic).
* Test arm (`test_combo_ens_retime_r8.py`): a mutation that skips the re-time (COMBO := ENS) must turn the stacking verdict from STACKS to NO-STACK on a planted fixture and `C-K1R` RED; a mutation that shuffles the window pairing must change a cell; a mutation that double-applies the gain must break `C-G1`.

## 7. The registered read (FINAL checkpoint, EMA weights, step 48,852, train-fitted gains) -- what that run needs, so the final chain calls the script UNCHANGED

```
sh <PKG>/code/run_combo_ens_retime_r8.sh step48852ema_unc train        # CPU only, OMP_NUM_THREADS=6, memory-gated (FreeVirtualMemory >= 6 GB)
   == python code/combo_ens_retime_r8.py run --tree step48852ema_unc --gains train --out-dir <PKG>/raw/combo_ens_retime/step48852ema_unc_train
```
Preconditions (the script fails loud, naming the missing one; it never substitutes):
1. **K-draw rolls at 48,852:** `KD_STEP=48852 sh code/kdraw_queue_r8.sh` -> `E:/refcv8_eval_kit/kdraw/step48852ema_unc/dump_s{2..7}/` each with `KDRAW_VERIFY.json` `ok: true` (needs `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt`; ~2,730.6 s of GPU per os-only seed, ~4.6 h, MEASURED at 30k). UNVERIFIED: the exact checkpoint file names under the final battery.
2. **The final battery's banked trees:** `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_unc/{dump_s0,dump_s1,panel_s0,panel_s1,s6_s0,s6_s1,...}` (seeds 0/1 draws, the refcv7 baselines `b_refcv7_50400_s*`, the decisions for the prior, the S6 panels for the K=2 ACC-LAT block).
3. **The train-fitted gains:** `raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json` + `.sha256` (queued: `chain_retime_trainfit_r8.sh`, `ZZTFPOSTENDZZ rc=0`, `RESULTS_TRAINFIT.md` exists). The gains were fitted on the 30k-EMA plans: **applying them to 48,852 is a second transfer (checkpoint)** on top of the first (single-draw -> averaged) and the third (train -> eval); the final report names all three.
4. **Not available and NOT requested here:** S6-grid K-draws at seeds 2..7 (a GPU S6 roll). Without them S6 ACC-LAT stays K=2-only. If the PI wants S6 at K=4 that is a separate GPU job, named as the blocker for that sub-claim only.
The 30k clean-gains read (`--tree step30000ema_unc --gains train`) uses the identical protocol once item 3 exists; it is a milestone read, not the registered one. The 30k outcome does NOT pre-decide the final label.

## 8. Standing commitments

No threshold, set, statistic, arm or gain source will be changed after the first COMBO cell is read. A deviation is logged as a deviation. The eval-fitted 30k label is reported as EXPLORATORY with the sentence "bar of record = single-draw `os`; gains eval-fitted (in-sample); transfer unmodelled".
Any result file is written by the script with this file's sha256 recorded inside it; `PREREG_COMBO.sha256` is written after this file and pinned in the code.
