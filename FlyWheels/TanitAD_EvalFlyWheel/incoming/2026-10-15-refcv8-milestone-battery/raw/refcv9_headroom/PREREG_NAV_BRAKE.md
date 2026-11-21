# PREREG H-R9-NAV-BRAKE-1 — does the clip-level nav command drive refcv8's cruise false-brake?

Registered 2026-10-19 ~23:50 Berlin, BEFORE any count below was computed. Sealed by `PREREG_NAV_BRAKE.sha256`.

## Why now (provenance of the hypothesis — post-hoc, stated)

The 40k judgement video of episode 114 (`E:/refcv8_eval_kit/videos/refcv8_ema40k_ep114_cruise_false_brake_HEADS.mp4`, t = 12.0 s) shows a straight night road, nav input TURN_L, the tactical lateral head at TURN_L 56 % vs GT LANE_KEEP, the lon head BRAKE_TO 79 %, and the plan decelerating −5.9 m/s while the human cruises. The video agent also MEASURED that the banked `decisions.nav_cmd` is constant per episode in 139/139 eval episodes (`…/2026-10-19-refcv8-videos/raw/NAV_CONSTANCY_step40000ema.json`) — a clip-level route command, so a TURN command is present on windows where the turn is far away or already behind. The hypothesis was formed from ONE example; this test is the first look at the population.

## Data (banked, CPU only, no model forward)

* refcv8 EMA-40k tree `C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc`, S6 panel `s6_s{0,1}` + `decisions/`, inference seeds 0 and 1. Secondary: EMA-30k tree `step30000ema_unc` (beside, not a clause).
* refcv7-50,400 on the same windows: the panel's embedded `b_refcv7_50400_s{seed}` (nav_true only).
* Population: S6 ceiling-compliant windows (`DEPLOY8_POS_<tree>_s<seed>.npz` `S6_compliant`) with GT class `cruise` under `cruise_brake_retime_r8.plan_class_thr(G, v0, dt 1.0, thr 1.5)` — the SAME definition as H-R9-BRAKE-LOCUS-1 / H-R9-L2-HEADROOM-1.
* Plan per nav condition: `decisions/plan_full_<cond>` (8 slots 0.5,1,1.5,2,3,4,5,6 s), sub-sampled to the S6 slots 1..6 s (indices 1,3,4,5,6,7). CONTROL C1: `plan_full_nav_true` at those slots must equal the panel's `os` to ≤ 1e-4 m on every window, or the run is VOID.
* Strata: TURN = `nav_cmd` ∈ {1 TURN_L, 2 TURN_R}; FOLLOW = `nav_cmd` 0. STRAIGHT6 = |GT lateral offset at 6 s| < 2.0 m (ego frame at t0).
* Estimator: paired episode-cluster bootstrap (`pair_read_r8.paired_cell`), B 2000, on the same windows. Answers "another draw of EPISODES" only; single training seed.

## Primary quantity

`dFD(cond) = false-decel(nav_true) − false-decel(cond)` on GT-cruise windows, per stratum, where false-decel = the plan's class is `decel`. A positive dFD(nav_zero) means removing the nav input RELEASES false brakes.

## Committed outcomes (both directions written before data)

* **NAV-DRIVES-BRAKE** iff (i) dFD(nav_zero) on TURN ∧ GT-cruise windows is separated > 0 at BOTH seeds, AND (ii) its point estimate exceeds dFD(nav_zero) on FOLLOW ∧ GT-cruise windows at both seeds.
* **NAV-NOT-THE-LEVER** iff dFD(nav_zero) on TURN ∧ GT-cruise is NOT separated > 0 at EITHER seed.
* **MIXED** otherwise (e.g. one seed, or separated but not larger than FOLLOW).

## Descriptive, outside the label (printed, never a clause)

* false-decel rate of refcv8 and refcv7 per stratum {TURN, FOLLOW} × {STRAIGHT6, curved}, with the paired r8 − r7 difference;
* dFD for nav_flipped and nav_shuffled;
* the share of refcv8's nav_true false-brake windows that are TURN, against TURN's share of GT-cruise windows;
* the same on the 30k tree.

## Caveats written in advance

* `nav_zero` zeroes the nav INPUT the battery injects; which nav channels that covers (clip-level v7 token vs per-frame v9 args) is being traced separately (2026-10-19 nav-trace agent). If nav_zero turns out to be off-distribution for refcv8, a positive dFD can be an off-distribution artefact; the flipped and shuffled arms are the in-distribution checks.
* An intervention on the nav input of a fixed checkpoint, T1 open loop; it says what the trained model does with the input, not what a retrained model would do.
