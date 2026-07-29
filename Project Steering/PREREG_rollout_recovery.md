# PRE-REGISTRATION — rollout-recovery training (the fix E-CR indicated)

**Written 2026-07-29 07:3x UTC, BEFORE any code was changed and before the run exists.**
Both outcomes committed. PI authorisation: *"use just one gpu"* (2026-07-29) — proceed on a single
A40 rather than wait for pod1's 8× A6000.

## Why this experiment, in one paragraph

**E-CR (C61 resolved) MEASURED that the imagination failure is COMPOUNDING, not task difficulty.**
On 761 windows / 40 episode clusters with the paired episode-cluster bootstrap (B=2000, interval on
the difference), CR rose **3.50 → 80.77** over 0.4–2.0 s, all four horizons separated, `p = 1.0`.
⭐ The load-bearing fact: **the teacher-forced arm is FLAT** (0.00719 / 0.00814 / 0.00801 / 0.00723)
— one step from truth is as accurate at 2.0 s as at 0.4 s. ⇒ **the model does not struggle with the
horizon; it struggles with its own output.** That is exactly the defect rollout-recovery targets.

**Mechanism (HorizonDrive, PUBLISHED):** train on **prediction-corrupted histories** — during
training, replace some of the predictor's context with the model's *own* predictions, so it learns to
recover from its own error instead of only ever seeing clean ground truth.

## The arms

| arm | training | purpose |
|---|---|---|
| **RR** (this run) | predictor context corrupted with own predictions at rate `p` | the treatment |
| **v1** `flagship4b-speedjerk-30k` | unchanged, already trained | the incumbent control |

⚠️ **This is a NEW TRAINING RUN, not an eval.** ~4 days on one A40 at ~13 s/step for 30 k steps.

## PRIMARY endpoint — and it is NOT ADE

⭐ **PRIMARY: `CR_k` at k = 4/8/16/20**, the same instrument E-CR used
(`e_k = 1 − cos(z_hat_k, z_true_k)`, latent error, **no decoder in the path**), on the **same
40-episode val surface**, with the **PAIRED episode-cluster bootstrap** (`taniteval/ci.py`, B=2000),
**interval on the DIFFERENCE**. ⛔ No CI on the ratio is computed and none may be quoted.

**Success = CR_k falls, with the paired CI excluding zero, at k=16 and k=20** (where compounding
dominates: v1 sits at 64.21 and 80.77 there).

⚠️ **SECONDARY, reported but NOT the verdict: `wm_fidelity_ade_2s` vs v1's 0.4271** (40 eps) /
**0.4108** (600 eps). ⛔ **CR and ADE may NOT be traded off against each other** — C63 measured that
the bridge between latent error and metres runs through a `step_readout` that is non-linear and
domain-sensitive (a two-true-latent decode scored **6.76 m** at 2 s vs the rollout's 0.53 m).
**Do not convert one into the other in either direction.**

## Outcomes, committed in advance

| outcome | reading | consequence |
|---|---|---|
| **CR falls, CI excludes 0 at k=16 AND k=20** | rollout-recovery works | ⭐ the mechanism is validated; it becomes a standing part of the recipe, and the Koopman/spectral lever becomes the natural *second* attenuator |
| **CR flat, CI covers 0** | the corruption schedule was too weak, OR compounding is not trainable away at this scale | ⚠️ **Report as a NEGATIVE, do not re-tune and re-report.** A schedule sweep after seeing the result is the forking-paths defect. If a sweep is wanted, it is a NEW pre-registration. |
| **CR falls but ADE regresses beyond v1's CI** | the model traded trajectory accuracy for latent stability | ⛔ **NOT a win.** Report both; the PI decides whether the trade is acceptable. Do NOT headline the CR gain alone. |
| **CR rises** | the corruption is harming the predictor | stop; the schedule is wrong, not the idea |

## Registered in advance — the confounds

1. ⚠️ **RR trains from a DIFFERENT starting point than v1 was compared at.** State plainly whether
   RR is trained **from scratch** or **fine-tuned from v1's checkpoint**, and hold that fixed. A
   fine-tune and a from-scratch arm are not the same experiment.
2. ⚠️ **One A40, not eight A6000s.** Batch/accum must be recorded; if effective batch differs from
   v1's, **that is a confound and must travel with the result.**
3. ⚠️ **Parity is sacred.** RR trains on `physicalai-train-e438721ae894` (2,376 eps, skip-hash
   `f09e44db`). ⛔ Anything that re-selects episodes must be REFUSED — and note **C64** happened
   because a re-selection did not inherit the val exclusion.
4. ⚠️ **The corruption rate `p` and schedule are FROZEN at launch** and recorded here before the run.
   Changing them mid-run makes the arm uninterpretable.

## Cost and sequencing

**~4 days, one A40 (pod3).** ⛔ **eval stays FREE** for the C64 option-A comparison when v2corpus
finishes (~10 h) — that is why this goes on pod3 and not on both.

## Evidence class

| item | class |
|---|---|
| E-CR's CR table motivating this | **MEASURED (ours)**, paired episode-cluster bootstrap |
| HorizonDrive's rollout-recovery mechanism | **PUBLISHED** (arXiv 2605.11596) — ⚠️ its ~20 s horizon is in a **pixel-reconstructive VAE latent** scored with video metrics; **the mechanism transfers, the number does not** |
| that rollout-recovery will reduce our CR | **HYPOTHESIS** — this experiment is the test |
