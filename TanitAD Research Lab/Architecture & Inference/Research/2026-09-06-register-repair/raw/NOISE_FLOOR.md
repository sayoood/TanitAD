<!-- REPAIRED 2026-09-06 by the register-repair agent. Provenance below; do not hand-edit. -->
# NOISE_FLOOR -- REPAIRED 2026-09-06 (the 2026-09-05 original CRASHED mid-write)

!! **What was here before, and why it was dangerous.** The 2026-09-05 generation of this file
is a **TRUNCATED artifact that reads like a complete one**: 1,128 bytes containing a title,
two correct paragraphs of methodology, one section heading -- and then a Python traceback.
`noise_floor.py` died on a **cp1252 `UnicodeEncodeError`** printing `⇒` to a cp1252
stdout, *after* the header had been flushed. It carries **no numbers at all**, yet it was
cited in `GOALS_AND_CLAIMS.md` (H-ESTIM-SEED-1) as an evidence path for a false-positive
rate. The crashed bytes are preserved verbatim at
`.../2026-09-06-register-repair/raw/NOISE_FLOOR.CRASHED.orig.md` (md5 in that package's
`RESULT.md`), and the failure is logged in `RETRACTION_LOG.md` 2026-09-06.

**Repair, and the guard that comes with it.** Regenerated 2026-09-06 by
`.../2026-09-06-register-repair/raw/noise_floor.py` -- the same script with (a) every
non-ASCII character removed from every `print()`, and (b) a **completion marker emitted as
the last line**. `.../2026-09-06-register-repair/raw/assert_complete.py` asserts that marker
before any number here may be quoted, and ships with a deliberate-regression arm: it must
report `INCOMPLETE` on the preserved crashed copy and `COMPLETE` on this file.

**Evidence class MEASURED (ours).** Source: the panel's own banked training logs,
`.../2026-09-05-withheld-bank-panel/raw/<ARM>/metrics.jsonl`. **Zero GPU.**
!! **Tier: NONE of T0/T1/T2 applies** -- these are TRAINING-LOG deltas, not an eval.
!! **Scope: the v7-tiny rig** (~19 M params, 48 non-parity training episodes, 2,000 steps).
It bounds what a *tiny-rig* separated result is worth and nothing else.
!! **This is the TRAIN-LOG half of the floor only.** The EVAL-level floor -- the one that
prices what the panel actually scores -- is `A0b_replicate` vs `A0_fixed` in
`raw/panel_report.json`: **6 separated cells of 42 = 14.3 %**, re-derived 2026-09-06.
The figures "3 of 18" and "~17 %" do NOT reproduce at any scoping and were never in this file.

---

# run-to-run noise floor, from the identical-config warm-up stretch

An arm with `--withheld-bank-warmup N` rolls the FIXED bank for steps <= N, so
over that stretch it IS A0's configuration. Divergence there is nondeterminism.

## A1_pred vs A0_fixed -- 9 identical-config rows (steps <= 450)
non-zero differences: 66/72  => training is NOT deterministic

| metric | mean |delta| | max |delta| | first-row |delta| | last-row |delta| |
|---|---|---|---|---|
| loss | 0.49495 | 1.72752 | 0.01408 | 0.42345 |
| traj | 0.05003 | 0.14880 | 0.01731 | 0.01172 |
| withheld_speed_mae | 0.19403 | 0.70135 | 0.00007 | 0.20518 |
| kept_speed_mae | 0.32692 | 1.21987 | 0.00090 | 0.06351 |
| goal2s_err_m | 0.46449 | 2.04176 | 0.00095 | 0.49224 |
| sel_v3 | 0.12544 | 0.38850 | 0.00439 | 0.06548 |
| anchor_acc | 0.02778 | 0.08334 | 0.00000 | 0.08333 |
| goal_tac | 0.75803 | 2.71051 | 0.00208 | 0.61566 |

## A2_random vs A0_fixed -- 9 identical-config rows (steps <= 450)
non-zero differences: 69/72  => training is NOT deterministic

| metric | mean |delta| | max |delta| | first-row |delta| | last-row |delta| |
|---|---|---|---|---|
| loss | 0.53715 | 2.36306 | 0.00162 | 0.42657 |
| traj | 0.06192 | 0.15229 | 0.02051 | 0.07078 |
| withheld_speed_mae | 0.34646 | 1.08828 | 0.00166 | 0.48625 |
| kept_speed_mae | 0.63948 | 1.96384 | 0.00027 | 1.05396 |
| goal2s_err_m | 0.29467 | 1.13906 | 0.00041 | 0.14757 |
| sel_v3 | 0.24965 | 0.56142 | 0.01125 | 0.16284 |
| anchor_acc | 0.05555 | 0.08334 | 0.00000 | 0.08333 |
| goal_tac | 0.94807 | 3.01220 | 0.00124 | 0.03307 |

## A0b_replicate exists -- the EVAL-level floor is in the panel report
   (A0b vs A0 paired delta = same config, same seed, run twice)

<!-- ARTIFACT-COMPLETE: NOISE_FLOOR v2 -->
