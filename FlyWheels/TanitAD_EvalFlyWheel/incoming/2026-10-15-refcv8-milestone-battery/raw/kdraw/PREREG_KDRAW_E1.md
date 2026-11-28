# PREREG_KDRAW_E1 -- does a K-draw inference ensemble of refcv8 pass BAR-R8-E1 on the ceiling-compliant S2 set?

Written 2026-10-18T23:59+0200 (dev box), BEFORE any new roll (inference seeds 2-7) exists and before any K=4 / K=8 / MEDOID cell was computed.
Author: EvalFlyWheel engineer (kdraw stream). Status: PRE-REGISTRATION. Evidence class of everything below: DESIGN (no new measurement).

## 0. What is and is not being claimed (read first)

* **The bar of record is the SINGLE-DRAW `os`.** BAR-R8-E1 (SPEC_REFCV8_DRAFT section 5, scored under A-R8-CLAMP-2) is `os - ha0_ext`, S2 ADE 0-2 s,
  < 0 and SEPARATED at BOTH inference seeds, on the ceiling-compliant windows. Nothing in this document changes it.
* A **K-draw ensemble is a NEW ARM, not the same arm**: it runs the planner K times per window and combines the K plans with model outputs only
  (no labels, no ego channel beyond what each draw already consumed). It is a **deployment-cost option for the PI**: inference cost scales with K
  (K planner draws per cycle). Whether the K draws can share the trunk forward (only the sampler stochastic) is UNMEASURED here, so the cost is stated as "up to xK".
  A K-draw PASS does **not** convert a single-draw FAIL into a PASS of BAR-R8-E1; it documents what the extra compute buys.
* The EMA-30k read is **EXPLORATORY** (the checkpoint is a milestone, not the registered one). The **registered read is the FINAL checkpoint
  (step 48,852, EMA weights, expected ~2026-10-20 19:00 Berlin)**, with this identical protocol (section 7).
* Tier of every number: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**. Estimator: **paired episode-cluster
  bootstrap, B 2000, seed 0, 95 % percentile, models FIXED** (`taniteval.ci.paired_episode_cluster_bootstrap` via `pair_read_r8.paired_cell`).
  It answers "would another draw of EPISODES say this?" ONLY. The "another inference run" question is answered by the second, DISJOINT seed set (below).
  "Another TRAINING run" is NOT answered (one training seed): the verdict is always "single training seed".

## 1. Prior knowledge disclosed before registering (so the pre-registration is not mistaken for blind)

MEASURED (primary: `raw/e1_decomp/E1_DECOMP.json`, levers.ENS2 and seeds, compliant set, 4,159 windows / 138 episodes, mask sha 4ebc6b86...):
single-draw E1 = -0.0085 [-0.0229, +0.0065] (seed 0), -0.0076 [-0.0216, +0.0073] (seed 1); ENS2 (mean of seeds 0,1) = -0.0138 [-0.0279, +0.0012]
(not separated); ENS2 minus seed-0 draw = -0.0052 [-0.0068, -0.0038]. Echo mean ADE 0.2807 m; refcv8 single draw 0.2722 m; ENS2 0.2669 m.
Seed-0 vs seed-1 plans differ by 0.0896 m mean L2 per window (`plan_l2_seed0_vs_seed1_mean_m`).
**These K=1 and K=2 cells have been SEEN. No K=4, K=8, MEDOID, or seed-2..7 number has.** If the mean of K plans has variance-reduction gain
proportional to (1 - 1/K) the K=4 cell is expected near the CI edge -- i.e. this is a genuinely open question, and the CI width (~+-0.0145 m, set by
episodes, not by K) is what decides it. That arithmetic is an ESTIMATE, not a prediction I am committed to.

## 2. Arms (all on the SAME 4,754-window S2 grid, scored on the compliant mask)

Plans are the S2 `os` plans (4 waypoints at 0.5/1.0/1.5/2.0 s, ego frame) of the EMA-30k checkpoint (`E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt`,
md5 7cffcd7e7ebf3c8ed22de316c71bbd0d per the banked `battery_summary.json`), one draw per inference seed. Draws = inference seeds 0..7
(seeds 0,1 are the banked dumps; seeds 2..7 are NEW rolls, `os`-only mode exactly as the battery rolls every seed after the first).

| arm | definition (per window) | seed sets |
|---|---|---|
| **ENS-K** | element-wise MEAN of the K draws' plans (float64 mean, cast to float32) | K=4: A={0,1,2,3}, B={4,5,6,7} (DISJOINT); K=8: all |
| **MEDOID-K** | the SINGLE draw (a real, valid single plan; cannot blend two modes) whose mean L2 distance over the 4 S2 waypoints to the K-mean plan is smallest; ties -> lowest seed | same sets |
| reference: SINGLE | each seed's own `os` (K=1), seeds 0..7 | -- |
| control: ENS-2 | mean of seeds {0,1} | must reproduce -0.0138 [-0.0279, +0.0012] |

K=8 has no disjoint replicate (it uses every draw); it is reported as the dose-response endpoint and **cannot decide a verdict by itself**.

## 3. The bar, per arm (committed in advance)

For each of ENS-4 and MEDOID-4, the cell is E1 = (arm ADE 0-2 s) - (`ha0_ext` ADE 0-2 s) on the compliant windows (n expected 4,159 windows,
138 episodes), on set A and on set B. Define `floor_arm = |mean ADE(arm, A) - mean ADE(arm, B)|` (compliant windows) -- the arm's own inference-replicate
floor, the K-draw analogue of the battery's `|mean(os,s0) - mean(os,s1)|`.

* **PASS**: on BOTH sets: delta < 0 AND CI upper (hi) < 0 AND |delta| > 2 x floor_arm.
* **PARTIAL**: not PASS, and delta < 0 on BOTH sets. Sub-label (mandatory): the number of sets with hi < 0 (0/2 or 1/2), and whether the only miss is the
  "within 2 x floor" (NOT PROVEN) rule.
* **FAIL**: delta >= 0 on at least one set.

Overall outcome of this experiment: **PASS** iff at least one of ENS-4 / MEDOID-4 is PASS; **PARTIAL** iff none is PASS and at least one is PARTIAL;
**FAIL** otherwise. If ENS-4 PASSes and MEDOID-4 does not, the report must say so and must evaluate the blending flag in section 5 -- an ENS-only pass is
not described as "the planner is better", it is described as "the averaged plan is better".

No threshold, window set, seed set, statistic or arm will be changed after the first K=4 cell is read. A deviation is logged as a deviation, not folded in.

## 4. Non-regression (NR) vs refcv7-50,400 -- reported, not a gate of this experiment

For each K=4 / K=8 arm and each K=4 set, the four families as `pair_read_r8` computes them on the S2 compliant windows, arm minus refcv7-50,400 `os`
(banked per-inference-seed arms `b_refcv7_50400_s0` and `_s1`, from `panel_s0`): ADE (ade_m); LONGITUDINAL (speed MAE, |along-track| at 2 s, +secondary);
LATERAL (heading MAE, curvature MAE masked, yaw-rate MAE masked, +secondary); TACTICAL (trajectory-derived lat/lon decision correct). NR rule as in
`pair_read_r8.bar_NR_metric`: OK iff the CI lower bound <= 0 (not separated worse) against BOTH refcv7 seeds; FAIL (separated worse) otherwise.
STRATEGIC: NOT APPLICABLE, n = 0 (`--no-strategic` in the run's argv; `PAIRED_READ_step30000ema_unc.json` -> `strategic`).
**Asymmetry stated once**: refcv7-50,400 is itself stochastic and exists only as seeds 0 and 1, so each K-arm is compared with a SINGLE refcv7 draw
(favourable to the K-arm) and, as a fairer secondary, with the refcv7 2-draw mean. S6 / E3 are NOT covered here (S2 only).

## 5. Mode-disagreement diagnostic (so averaging cannot hide mode blending)

* spread(window) = the largest pairwise L2 distance between the K draws' **2-s endpoints** (waypoint index 3). Reported: the fraction of windows
  (compliant set; all windows beside it) with spread > 1.0 m, per K-set and for K=8.
* On those windows (spread > 1.0 m, defined from the K-set's own draws) and on all compliant windows: LATERAL curvature MAE (masked) and heading MAE of
  ENS, MEDOID, and the mean over the K single draws; paired ENS - MEDOID on the disagreement subset with the same estimator.
* **BLEND FLAG (mechanical)**: raised for ENS-K iff on the disagreement subset ENS curvature MAE minus MEDOID curvature MAE has CI lower bound > 0
  (ENS separated worse than a real single plan on the lateral shape). A raised flag downgrades an ENS-only pass to "PASS WITH BLEND FLAG".

## 6. Controls that must read their known value (any failure => the analysis prints VOID, not a verdict)

1. `C-K1`: K=1 reproduces the banked single-draw cells to 4 dp: seed 0 -0.0085 [-0.0229, +0.0065], seed 1 -0.0076 [-0.0216, +0.0073] (literals).
2. `C-K2`: ENS-2 reproduces -0.0138 [-0.0279, +0.0012] to 4 dp.
3. `C-MASK`: compliant mask sha256 = 4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129; 4,159 / 4,754 windows; 138 episodes.
4. `C-ID`: every new seed's `ws`, `eid`, `clip_index`, `g` and `ha0_ext` are bit-identical to seed 0's (same windows, model-free arms untouched) and its
   `gt_max_speed_6s` / `v_max_ms` extras reproduce the mask.
5. `C-DIST`: no two seeds' `os` plans are identical (a mean pairwise plan L2 > 1e-4 m for every pair) -- the refcv6-loader RNG-reset defect
   ("every inference seed became seed 0", measured 2026-09-24) would make an 8-draw ensemble 1 draw.
6. `C-TRI`: ADE(mean plan) <= mean of the draws' ADEs on every window (analytic).
7. `C-CAP`: for every rolled seed, the speed-ceiling cap changes ZERO plans on the compliant windows (capped == uncapped there), so the panel's `os`
   is the capped primary. If this control cannot be run it is printed UNVERIFIED and the verdict carries that word.
8. `C-DET` (roller): a prefix re-roll of seed 1 (os-only) reproduces the banked `dump_s1` rows (bit-identical, or the measured difference is reported);
   same for seed 0 (full mode) vs `dump_s0` and seed 0 (os-only) vs `dump_s0`. If the roller is not deterministic per seed, that is stated in the verdict.
9. Test arm (code): `test_kdraw_e1_r8.py` -- shuffled window pairing must change the E1 cell; K=1 must reproduce the banked cell; a deliberate-regression
   MEDOID that returns the mean must fail the analytic symmetric-mode check.

## 7. The registered read (FINAL checkpoint, step 48,852)

Identical arms, sets, bar, controls and thresholds on the EMA weights of the final checkpoint: seeds 0,1 from the final battery's dumps, seeds 2..7 rolled by
`kdraw_queue_r8.sh` with `KD_STEP=48852` (os-only, same windows file as the battery), analysed by `kdraw_e1_r8.py --tree step48852ema_unc`. The compliant
mask must again hash to 4ebc6b86... (it is a function of the logged windows and the fed ceiling, not of the weights) -- if it does not, the final read is
VOID until explained. The 30k outcome does NOT pre-decide the final read; a 30k PASS is "this lever may exist", a 30k FAIL is "no evidence at 30k", neither is the registered answer.

## 8. Committed outcome table (what each result means for the PI)

| result at the final read | statement committed in advance |
|---|---|
| single-draw `os` PASSES BAR-R8-E1 | the bar is met by the arm of record; the K-draw rows are an optional extra, not needed |
| single-draw FAILS, ENS-4 or MEDOID-4 PASS (no blend flag) | the bar is met only by a K-draw arm: report as "BAR-R8-E1 not met by the arm of record; met by a K-draw arm at inference cost up to xK" -- a PI decision (deploy K draws vs not) |
| single-draw FAILS, ENS-4 PASS with BLEND FLAG, MEDOID-4 not PASS | the averaged plan, not the planner, is better; not claimed as a planner improvement |
| PARTIAL | direction consistent, not separated on both sets: more seeds do not fix an interval set by episodes -- the lever is exhausted at this K |
| FAIL | the K-draw lever is eliminated for E1 at this checkpoint; the remaining E1 levers are the ones ranked in `raw/e1_decomp` (force_s2 / retime / milestones) |

## 9. Addendum A (execution facts fixed 2026-10-19 00:16, still before any new roll result; changes no threshold, set, statistic or arm)

* **Seed 0 is a FULL-mode roll, seeds 1..7 are os-only rolls.** The banked battery rolled seed 0 with every SPEC 3.2 arm and every later seed `--os-only` (SPEC A3). Each roll
  seeds the global RNG ONCE (`refcv3_arm.run_dump`: `torch.manual_seed(infer_seed)`), so the arm set changes how much of the stream the `os` draws share. All eight
  `os` plans are still independent draws from the same stochastic planner (DDIM sampler, `manifest['inference_seed']['stochastic_at_inference'] True`); control `C-DIST`
  asserts they are pairwise different, and `C-DET` measures what an arm-set change does to a seed's `os` (case C4 vs C1). Seeds 2..7 are rolled os-only, as seed 1 was.
* **S2 only.** A roll is the 4,754-window S2 grid; the S6 panel is a later battery step built from the same roll. E3 (S6) is out of scope for this experiment.
* **Same pipeline by construction.** New seeds run the battery's own child (`roll_seed_r8.py`, unedited) with the battery's argv/environment and the banked `windows_s0.json` byte for byte;
  a roll counts only if `KDRAW_VERIFY.json` (written from the dump CONTENT) says ok: 4,754 windows / 139 episodes, the model-free arms (`ws eid clip_index g ha0_ext v0`) bit-identical to
  banked seed 0, the mask inputs in the extras bit-identical, `os` finite, checkpoint md5 equal to the banked 7cffcd7e....
* **Roller wall time (MEASURED, banked `roll_s1.json`): 2,730.6 s per os-only seed on an otherwise idle GPU** (seed 0 full: 8,537.3 s). Six seeds = about 4.6 h of GPU plus the control.
* **Scheduling** (not science): GPU lock never broken; two free-commit checks >= 10 GB 60 s apart; the float64 probe and force_s2 go first (queue yields, caps 4 h / 6 h).
