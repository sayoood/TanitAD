# READING_FRESH2 -- how the FRESH2 agent reads the sealed PREREG_FRESH2.md (sha256 956f87cd...) where its words admit more than one implementation

Written 2026-10-27 by the FRESH2 evaluation agent BEFORE any FRESH2 z_tac, LON6 label, a0 / psi_dot0 feature or score existed (the episode build was
still running), and recorded with its sha256 in `READING_FRESH2.sha256`; `code/fresh2_stage1.py --run` refuses to run if this file changed. The sealed
prereg is not edited. Where a reading is a choice, the alternative is computed BESIDE and reported, never used to pick a verdict.

## R1. The stage-1 population and GT (s3 "Population and labels")
The sealed sentence says *"exactly PREREG_LHA0_D2 s3's K-HEAD definitions (S6-valid, nav-compliant, GT in {decel = BRAKE_TO/CREEP/HOLD, cruise =
FOLLOW/CRUISE})"*. PREREG_LHA0_D2 s3 and the code that implements it (`d2_khead.load_population` -> `l2_common.load_train_dump`) define GT
**kinematically** from the human future (`dv = v(5-6 s) - v0`: decel if dv <= -1.5 m/s, cruise if |dv| < 1.5) and use BRAKE_TO+CREEP+HOLD vs
FOLLOW+CRUISE as the **score's** renormalisation sets, not as the GT. "Compliant" there is the speed-ceiling compliance (`gt_max_speed_6s <= v_max_ms`).
* **PRIMARY (reads the bars):** the code-exact K-HEAD rule -- `d2_khead.load_population` imported unedited, its dump / bank / feature paths re-pointed
  at FRESH2, restricted to CONFIRM episodes. Reason: s3 says "exactly", the brief says "exactly as d2_khead.py / d2_common.py", and the number this
  prereg sets out to confirm (FUSE +0.0450 [+0.0185, +0.0716] over LH on eval139) was computed on that population.
* **BESIDE (non-gating):** the parenthetical read literally -- GT decel = LON6 label in {BRAKE_TO, CREEP, HOLD}, cruise = {FOLLOW, CRUISE}, from the
  FRESH2 v9 release built WITH the lead pass (`lh_klabel.trainer_lon_targets`), on S6-valid & compliant CONFIRM windows of clips the lead pass covers.
  Same arms, estimator and bars, reported as descriptive. If the two readings disagree on a bar, both are reported; the Master Mind decides.

## R2. `LH_G` = "LH's recipe (`lh_fit.py`: features, lambda grid by out-of-fold CE, 5 folds by episode, seed 0)"
`lh_fit.cv` (imported unedited) scores the WHOLE sealed LONHEAD grid -- logistic lambda {1e-3 .. 100} AND MLP weight decay {1e-4, 1e-3, 1e-2} -- and
freezes the config with the lowest out-of-fold CE; LH itself came out of that grid (logistic lambda 100). LH_G runs that same function unchanged on the
new rows; whichever config wins is frozen and named in the result. Rows: the L1C FIT rows (`lh_fit.load_fresh(..., lead_clips_only=True)`, 8,178 rows /
290 episodes) + the GROW rows by the same row rule (every GROW window whose LON6 label is one of the six classes, on lead-pass clips). GROW episode ids are
offset by +100000 before `folds_of` (the two pulls' integer clip indices overlap). The LH_G file's sha256 is written before any CONFIRM row is scored.

## R3. FUSE / FUSE_perm
FUSE = z(D2 logit) + z(LH logit) with LH's logit = log P(decel set) - log P(follow+cruise) (`d2_common.lh_logit`), each z-scored over the evaluated
windows without labels -- d2_khead's `fuse_D2_LH` verbatim. FUSE_perm = the same with a0 and psi_dot0 permuted within each CONFIRM episode over ALL of
that episode's bank windows (d2_khead's scope, restricted to CONFIRM episodes), episodes in sorted order, `np.random.default_rng(0)`,
`d2_common.permute_within_episode`; v0 stays with its window; the D2 component is re-z-scored over its own permuted scores.

## R4. Clips without obstacle.offline
A clip whose `obstacle.offline` member does not exist in the dataset (the first pull had 8 / 300) is still USABLE (s1.3 lists the episode build and the
clock sidecar as the chain steps). It stays in the PRIMARY population (kinematic GT does not read the lead pass) and is excluded from the GROW fit rows
(lh_fit's lead-clips-only rule) and from the BESIDE LON6 reading (its FOLLOW / CRUISE labels cannot be formed).

## R5. Refusals
A clip is refused when the v2 episode build writes no file, `s2_geom_emit_v7.emit_one` raises, or the clock sidecar refuses it (the L1C chain steps).
Logged by sha12 + step + reason, excluded, not replaced; the downstream builders are re-run on the usable set.
