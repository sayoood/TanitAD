# PREREG — D-SAFE-CAL: is the safety barrier's THRESHOLD the reason RL found nothing?

`PRE-REGISTRATION, 2026-08-29, TanitAD_TrainingFlyWheel. Committed BEFORE any
training step and BEFORE any recalibrated number exists. Successor to P-RC21,
which exited at C (nothing improved) across three reward compositions and four
anchor strengths.`

---

## 0. Why this is a NEW experiment and not a knob on an old one

P-RC21 Addendum 4 measured that **`d_safe = 5.0 m` marks 45.1 % of competent human
driving as unsafe** on this corpus (n=71 windows with a followable track; 54.9 %
of human futures clear it with obstacles static, 56.3 % against real tracks). The
trust region pulls the policy toward the demonstration distribution and the
proximity barrier pushes it away, with the threshold set **where the human lives**.
The two objectives are opposed by construction.

⇒ Changing `d_safe` changes **what "safe" means**, not how hard we optimise for it.
That is a specification change, so it gets its own pre-registration with both
outcomes committed, per the operating standard. ⛔ It is explicitly **NOT** a
re-run of a P-RC21 arm with a different constant.

⚠️ **What P-RC21's nulls do and do not license.** Every prior null was measured
against the mis-calibrated target, so they are **uninformative** about the
recalibrated one. Uninformative is not promising: the prior on this experiment is
not raised by the campaign's failures, only unconstrained by them.

## 1. The change, stated exactly

| | current | proposed |
|---|---|---|
| `proximity_safe_m` | **5.0** (round number, no derivation) | **2.0** — the ~p15 of the human's own clearance distribution |
| windows where the barrier fires against ground truth | **45.1 %** | **15.5 %** (MEASURED, tracks column) |
| derivation | none | percentile of the demonstration distribution on this corpus |

Reference percentiles of human clearance against real tracks (`raw/p_rc21_rerun/target_calibration.json`):
**p5 0.787 m · p25 3.245 m · p50 5.916 m · p75 17.426 m · mean 10.981 m**.

⚠️ **2.0 m is a CHOICE, and it is committed here so it cannot be tuned later.** It
sits between p5 and p25, is above the physical contact radius (`ego_r + obs_r` =
2.0 m — i.e. the barrier begins exactly where the collision term ends, with no
dead zone and no overlap), and leaves the barrier firing on a minority of windows.
⛔ **No sweep over `d_safe`.** A threshold selected by which value produces the
best result is the 2026-08-22 ridge-probe failure in a new costume.

## 2. Arms (all T0, all NON-PARITY pilot corpus, all decoder-only)

| arm | `w_anchor` | `proximity_safe_m` | purpose |
|---|---|---|---|
| **A** | 1.0 | **2.0** | the test |
| **B** | 10.0 | **2.0** | does the conclusion survive a tight trust region |
| **C (control)** | 1.0 | 5.0 | ⭐ **the paired re-run of the banked arm**, same seed, same windows — so any difference is attributable to the threshold and nothing else |
| **D (floor)** | 1.0 | 2.0, weight 0 | proximity present but inert ⇒ isolates the threshold from the mere presence of the term |

2,000 steps each, seed 0, `decoder_steps=2` (⛔ **not** larger — Addendum 3 measured
that budgets above the trained operating point diverge: sel-ADE 0.654 → 1.529 m at
steps=8, SEPARATED).

## 3. Readout — committed now

Same instrument as P-RC21: eval-mode deterministic readout, 120 fixed windows,
paired episode-cluster bootstrap 4000 reps, DEFAULT spec for R1 so arms stay
comparable across the campaign.

⭐ **AND, per TRAIN-C6, each exit below names the FIELD that will carry its
answer.** `R1_fan_reward.mean`, `R2_fan_collision_rate.mean`, `R3_sel_ade_m.mean`,
`R4_component_means["proximity"]` — all four exist in the readout's return today;
**plus one field that does NOT yet exist and must be added before the first arm
runs:** `R5_gt_clearance_violation_frac` — the fraction of scored windows on which
the *selected* candidate is inside `proximity_safe_m`. Without it, exit 3 is
unmeasurable, which is exactly the defect TRAIN-C6 records.

## 4. ⛔ BOTH OUTCOMES, COMMITTED BEFORE ANY NUMBER EXISTS

| # | if | then |
|---|---|---|
| **1** | **ΔR2 (fan collision) falls with paired separation in A, and C reproduces the banked null** | ⭐ **The threshold WAS the blocker.** The mis-calibrated barrier is the single cause of P-RC21's exit. ⇒ recalibrate the spec, re-open the RL line, and re-run the anchor sweep once against the corrected target. |
| **2** | **A ≈ C: neither separates, and R4 proximity again moves < 0.01** | ⛔ **The threshold was NOT the blocker and the RL line is CLOSED on this base model.** The barrier is not what stopped it; the honest conclusion is that decoder-only RL on a 128-anchor fan has no headroom on this task, and P4 stops spending on it. |
| **3** | **R2 falls but `R5` rises** (fewer fan collisions, selected path now violating clearance more often) | ⚠️ **Reward hacking, caught by the field added in §3.** The policy improved the scored fan while degrading what is actually driven ⇒ reject, and the finding is about the fan/selection split, not about safety. |
| **4** | **R3 (sel-ADE) degrades with separation in A while R2 is flat** | The recalibrated barrier still fights the trust region, only later. ⇒ same close as outcome 2, with the added note that the opposition is not threshold-specific. |
| **5** | **D (weight 0) moves anything at all** | ⛔ **Instrument failure**, not a result — an inert term cannot change the objective. Stop, diagnose the harness, and re-run nothing until it reads flat. |

## 5. What is NOT claimed, whatever the outcome

⛔ Nothing about refcv3 (different architecture). ⛔ Nothing about driving — this is
**T0**; a capability claim needs T1 via `taniteval/tools/t1_eval.py`. ⛔ Nothing
about DDv2's method on its own stack. ⛔ No parity comparison: the pilot corpus is
NON-PARITY and no number here enters cross-arm tables.

## 6. Cost and gating

~2 h on the dev-box 4060 for four arms at 2,000 steps (MEASURED rate from P-RC21:
~0.2 h/10k steps ⇒ ~0.04 h/2k, plus readout). ⛔ **Thor is not used.** Gated on the
`R5` field landing with its test first — an exit that cannot be measured is the
defect this prereg was written to avoid repeating.
