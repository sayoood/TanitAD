# PREREG -- does refcv7-r101-s0 @ 50,400 USE its past camera frames? (EvalFlyWheel, 2026-10-12)

Written BEFORE any scored number of this experiment exists. Its sha256 is recorded in RESULT.md.
Tier: **T1 open loop** (self-action open loop on logged eval139 frames; status UNRULED for an action-free model;
never closed loop, never "driving performance"). Eval only, zero training. Checkpoint: `D:/refcv7_eval_kit/ckpt/ckpt_50400.pt`
(md5 b418d0fc4a92a6848c246a6a7c50207b per MODEL_REGISTRY / r7base RESULT). Tip: `48435a3` (agent/arch-inf-20260803).

## 1. Hypothesis
H-HIST-1: the trained planner depends on the real temporal content of its 10-frame camera window (8 rows x K=3, 10 Hz).
H0 (what the refcv8 input review's section 2.5 must be able to retain): replacing that content with what NAVSIM can
supply does not move the four metric families beyond the rig's own inference-seed floor.

## 2. Arms (same checkpoint, same windows, same nav/v0/ego-history/v_max/labels -- ONLY the camera pixels change)
Raw slots s=0..9 are the 10 distinct 10 Hz frames t0-0.9..t0 (row r, block k = slot r+k; slot 9 = t0).
* **A  REAL**  inference seed 0, frames as trained.      **A-rep**: REAL with inference seed 1 (the noise floor).
* **B  HOLD** (= the review's STACK-HOLD): every row's K=3 stack <- 3 copies of THAT ROW's newest frame
  (row r = slot r+2 x3). The 8 rows still differ (PhiTac/trunk still see 10 Hz row-to-row motion); only the
  intra-stack 0.1/0.2 s motion that the 1x1 K-fusion reads is removed. Isolates the fusion.
  (Interpretation recorded: the brief's "every past frame in every row replaced by the current frame" is read per ROW;
  read globally it would be identical to arm C and the arm would be redundant.)
* **C  NAVSIM-ST**: the bridge's primary construction (`slot_sources`, ST): EVERY raw slot <- the t0 frame (8 identical rows).
* **D  NAVSIM-NT**: the bridge's NT construction: each slot <- the NAVSIM 2 Hz frame (t0-1.5,-1.0,-0.5,0) nearest in
  time (ties -> later) with DT 0.1: slots 0,1 <- frame t0-1.0 (taken from the cache, ONE row earlier than slot 0's
  window; clamped to the episode's first frame if it does not exist, count reported), slots 2..6 <- frame t0-0.5
  (= slot 4), slots 7..9 <- t0. (Bridge: `tanitad_navsim_bridge.py::slot_sources`, `refcv6_bridge.py::slot_sources6`.)
The sampler draw is the same for A/B/C/D (same inference seed 0, same RNG consumption order), so A-vs-B/C/D contrasts
isolate the INPUT; A-rep isolates the SAMPLER.
Bridge ego history (linear interpolation from 2 Hz states) is deliberately NOT applied: ego history stays REAL in
every arm so the contrast is about CAMERA history only.

## 3. Grid
S6 surface (battery SPEC s3.1: S2 windows that have a valid 60-step future; horizons 1..6 s, dt 1.0), subsampled to
**every 3rd S6-eligible window per episode in ws order (G3)** -- selected from the banked dump's `ws` and
`gt_future_valid_ext` only, no outcome data, all 139 episodes. S2 (0.5..2.0 s) reads are the SAME windows' first 4 slots.
If a full-S6 pass is later run it is reported separately and does not replace G3. Not the K=8 EVAL-DIAG grid.

## 4. Metrics (all four families, per family, never pooled) and estimator
LONGITUDINAL: speed MAE, along-track MAE, target-speed acc(+-0.5 m/s), accel MAE, **LEAD-VIOL (share_of_at_risk, human keeps >=1.0 s;
lead windows = B1 k12/dt0.5 block, S6 grid) with the plan/human progress ratio beside it**, min headway / time gap / TTC (with n_closing).
LATERAL: heading MAE, curvature MAE, yaw-rate MAE (A3 valid-masked cell), cross-track MAE.  TACTICAL: traj-derived lateral
and longitudinal decision accuracy + kappa.  STRATEGIC: NOT APPLICABLE (strategic layer OFF, PI R5), n=0.  Plus ADE/FDE.
Estimator: FULL-SET mean; interval = episode-cluster bootstrap (`taniteval/ci.py`, B=2000, seed 0, 95 %), **paired**
for every arm-vs-A delta (never quadrature). Question each interval answers: *would another draw of EPISODES say this?*
The A-rep delta answers *would another INFERENCE run say this?*. **Neither answers training-seed variance** (single
checkpoint): the verdict is about THIS checkpoint, not about the recipe.

## 5. Primary metrics and the bar (fixed now)
P1 ADE_m (S6), P2 along-track MAE (S6), P3 cross-track MAE (S6), P4 TACTICAL longitudinal decision acc (S6),
P5 LEAD-VIOL share_of_at_risk (S6). Everything else is reported, none promoted afterwards.
delta_m = Delta(arm - A). Seed floor e_m = |metric(A-rep) - metric(A)|. Equivalence margin d_m = 2 % of A's value
(absolute 0.01 for accuracies and for LEAD-VIOL).
* arm is **EQUIVALENT** on m  iff the paired CI of delta_m lies inside [-d_m, +d_m].
* arm is **DEGRADED** on m   iff the paired CI excludes 0 on the harmful side AND |delta_m| > max(d_m, 3 e_m).
  (harmful = larger error/violation, lower accuracy.)  **IMPROVED** is the mirror image.   Else INDETERMINATE.
* Arm verdict: EQUIVALENT if all five primaries are EQUIVALENT; DEGRADED if P1 is DEGRADED or >=2 primaries are; else MIXED.
Control that must read a known value (checked BEFORE the arms are read): the A re-roll's `os` plan reproduces the banked
`dump_s0` plan on the same windows (|diff| = 0, or reported); A-rep vs A differ by sampler only; arm E (BLIND, below) must go RED.
Deliberate-regression arm **E  BLIND**: one constant grey image in all slots; must be DEGRADED on P1 (else the instrument is
blind and no verdict stands).  Cost-only: E is run on the first 12 episodes of G3.

## 6. Both outcomes, committed in advance
* **O1** C (ST) and D (NT) both EQUIVALENT: the past frames are not load-bearing for this checkpoint on PhysicalAI T1 ->
  refcv9 may adopt the NAVSIM-native contract (K=1, rows at 0.5 s, W=4; 10 -> 4 trunk frames; ~56 % FLOPs, ESTIMATED). (The
  training-seed caveat stays: one checkpoint.)
* **O2** C or D DEGRADED (and E RED): the history is used; NAVSIM numbers carry a history-mismatch penalty; refcv9 trains with
  2 Hz-hold augmentation (rows built the NT way, p~0.3) -- and B tells whether the penalty lives in the K-fusion
  (B DEGRADED) or in the 8-row/PhiTac/trunk content (B EQUIVALENT, C/D DEGRADED).
* **O3** MIXED / INDETERMINATE: report exactly which primaries moved; the next lever is the full-S6 pass, not an interpretation.
An arm that misses is reported as written; no margin or metric is changed after data.
