# PREREG_L1 — AMENDMENT 2 (braking AUC of every L1 arm on S6; requested by the Master Mind ~12:00; sealed BEFORE any arm is scored)

*2026-10-25 Berlin. Base `PREREG_L1.md` sha256 41ba4b2f…; Amendment 1 sha256 5193dc29…. At sealing: eval seed-0 bank done, eval seed-1
and train banks rolling; NO head fit, NO arm scored on 48,852 data. Nothing below changes an arm, a bar, a split, a grid, or the S2
verdict mechanics of PREREG_L1 §7-8. It ADDS one secondary read.*

## Why

MEASURED by L2 (726f996, `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-25-refcv9-l2/roc/`, prereg 5a84da9a…): refcv8's cruise false-braking
is an operating point, not lost discrimination — braking AUC refcv8 + stage 0.7445 / 0.7361, refcv7 0.7266 / 0.7182 (seeds 0 / 1), Δ +0.018
[−0.008, +0.046]. Both discriminate braking at only ~0.73: that is the speed-choice defect D1 the L1 progress head targets. L1's arms DO
produce S6 plans (the pre-registered secondary S6 surface, heads fit on the train S6 rows), so the AUC is read on S6, not S2.

## Definition (identical to L2, reused not re-derived)

`code/l1_roc.py` imports `dv_of` / `cls_of` / `auc` from `code/l2roc_ref/l2_roc.py`, **byte-identical** to 726f996's `l2_roc.py`
(sha256 5f7cc04733ce18f32d2b4bcb5d84901cc0bd3d7d64a5b5c7142af34c0c540261), and the population / v0 / GT / refcv8 + stage
(`DEPLOY8_POS_step48852ema_pf_s{seed}.npz` `S6_pos`) / refcv7 from L2's loader `l2_common.load_eval_pf` unchanged:
score dv = |P(6 s) − P(5 s)| / dt − v0; positives = GT-decel (dv_GT ≤ −1.5 m/s), negatives = GT-cruise (|dv_GT| < 1.5 m/s), ceiling-compliant
S6 windows (L2: 639 / 1,856, 136 episodes); AUC = P(score_pos < score_neg) + ½ P(tie). The L1 arms' S6 plans are banked by
`l1_fit_score.py` (`result/arms_S6_s{seed}.npz`) and aligned by (clip, ws).

**Estimator**: paired episode-cluster bootstrap over the population's episodes, B 2000, numpy seed 0 — L2's loop, with every arm evaluated on
the SAME draw sequence; it answers *another draw of EPISODES* only. Reported at both inference seeds. Paired deltas: every L1 arm vs
**refcv8 + stage** (`S6_pos`) and vs **refcv7**.

## Controls (must read their values; a failure voids THIS read only and is reported)

C1 GT AUC = 1.0 · C2 constant score AUC = 0.5 · C3 false-decel / missed-decel of stage / os / r7 reproduce `ZERO_GPU.json` (≤ 1e-4) ·
C4 the banked L2 AUCs (stage / os / r7) and the stage − r7 bootstrap CI reproduce `L2_ROC.json` exactly · C5 the L1 OS arm on S6 equals the
L2 loader's os (max |diff| 0) · population: the L1 S6 rows equal L2's compliant rows.

## Committed readings (per arm, primary arms H2 and H2s; every arm reported)

* **AUC-UP** vs X: paired Δ CI > 0 at both seeds — the arm discriminates braking better than X.
* **AUC-DOWN** vs X: paired Δ CI < 0 at both seeds — the arm discriminates braking worse than X.
* **AUC-SAME** otherwise.

Committed interpretations, X = refcv8 + stage:
* **H2 AUC-UP** ⇒ the latent-driven progress head improves braking discrimination on S6, i.e. it acts on D1 at the 6 s horizon, not only
  the S2 re-timing; recorded as S6 support for whatever the S2 outcome is (it cannot upgrade an S2 O2 to O1).
* **H2 AUC-SAME** ⇒ the head does not move the braking operating curve on S6; D1's discrimination part is untouched by L1 (the
  operating-point part is L2's).
* **H2 AUC-DOWN** ⇒ a guard failure on the secondary S6 surface: reported beside the S2 verdict, blocks any L1 ship until diagnosed, does
  not change the S2 outcome label.
* H2 − H2s is read the same way (does vision add braking discrimination over the scalar floor).
