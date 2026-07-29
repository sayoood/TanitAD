# PRE-REGISTRATION — rollout-recovery training (the fix E-CR indicated)

> ## ⛔⛔ REVISED 2026-07-29 07:5x UTC — THE ORIGINAL EXPERIMENT WAS WRONG. READ THIS FIRST.
>
> **I registered "implement rollout-recovery". IT IS ALREADY IMPLEMENTED, AND v1 ALREADY USED IT.**
>
> **MEASURED, `train_worldmodel.py:58` — `_rollout_loss` IS the HorizonDrive mechanism, verbatim:**
> *"Recursive K-step rollout: feed the 1-step prediction back into the window and predict again."*
> ```python
> z_hat = model.predictor(win_s, win_a)[1]
> loss  = loss + (z_hat - fut_states[:, idx_of[j-1]]).pow(2).mean()
> win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)   # ← prediction fed back
> ```
> It is gated on `K = cfg.train.rollout_k`, `if K > 1`.
>
> **MEASURED, `MODEL_REGISTRY.md:172,176` (the authoritative source for training args):**
> v1 `flagship4b-speedjerk-30k` trained with **`--rollout-k 4`**.
>
> ⇒ ⭐ **ROLLOUT-RECOVERY AT K=4 DID NOT PREVENT THE COMPOUNDING.** E-CR measured CR
> **3.50 / 15.16 / 64.21 / 80.77** at k=4/8/16/20 on that very arm.
>
> **Two facts fall straight out, and they reframe the question:**
> 1. **K=4 covers 0.4 s of a 2.0 s evaluated horizon** — the model practises recovery over 4 steps
>    and is then measured over 20, **5× beyond anything it ever trained on**.
> 2. **CR is already 3.50 AT k=4 ITSELF** — so K=4 recovery does not even fix K=4.
>
> ⇒ **The question is NOT "does rollout-recovery work" but "does rollout_k MATCHED TO THE EVALUATED
> HORIZON reduce CR?"** The revised design is §R below. ⛔ **I nearly spent ~4 GPU-days
> re-implementing a mechanism that ships in the repo** — the same class as C63 (building on an
> assumption never checked), caught this time *before* the GPU cost rather than after.
>
> ⛔ **AND THE FREE ANSWER IS NOT AVAILABLE.** `flagship4b-v2-30k` does run `rollout_k=12`, but it is
> **ABANDONED at step 7,800** (v1 is 29,999) and `--v2` enables **ten levers at once**
> (`ego_to_planners`, `fa_dropout 0.30`, `goal_decode`, `nav_dropout 0.5`, anchored tactical decoder…).
> Comparing its CR to v1's would confound `rollout_k` with **nine other changes and with training
> length**. It is not a control.
>
> ## §R — THE REVISED EXPERIMENT (supersedes the arms table below)
>
> **RR-FT: fine-tune v1's checkpoint with `rollout_k` RAISED TO MATCH THE EVALUATED HORIZON.**
>
> | | |
> |---|---|
> | **start** | v1 `flagship4b-speedjerk-30k` step 29,999 — **fine-tune, NOT from scratch** (registered per confound 1) |
> | **change** | `--rollout-k 20` (matching E-CR's k=20). **EXACTLY ONE FLAG DIFFERS from v1.** |
> | **length** | ~3–5 k steps, **not 30 k** — this tests the mechanism, it does not produce a deployable arm |
> | **hardware** | **ONE A40 (pod3)**, per the PI's *"use just one gpu"*. eval stays free for C64-A |
> | **cost** | ⚠️ **ESTIMATED, not measured** — the rollout loss is **20 sequential predictor calls per step** vs v1's 4, so step time will rise **materially**. Measure `step_s` over the first 200 steps and **report the real figure before committing to the full 5 k.** |
>
> **PRIMARY unchanged: CR_k at k=4/8/16/20**, paired episode-cluster bootstrap, interval on the
> DIFFERENCE, same 40-episode surface. **Success = CR falls with CI excluding 0 at k=16 and k=20.**
>
> **New outcome registered, because it is now the most likely one:**
> | outcome | reading |
> |---|---|
> | **CR falls at k≤20 but the run is much slower** | the fix works but costs compute — report the **step-time ratio** alongside, so the trade is visible |
> | **CR unchanged at K=20** | ⭐ **compounding is NOT trainable away by horizon-matched recovery** — a strong, publishable negative that would redirect effort to the architectural levers (spectral/Koopman) rather than the training schedule |
> | **CR falls only at k≤4** | recovery generalises no further than it trains — argues for K = the deployed horizon, always |
>
> ⚠️ **A fine-tune is NOT the same experiment as a from-scratch arm** and its result may not be
> quoted as if it were. If RR-FT is positive, the from-scratch confirmation is a separate,
> separately-registered run.

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

---

## §R2 — DESIGN FIX 2026-07-29 07:5x UTC: the control changes, not the treatment

**MEASURED — v1's exact trainer is NOT recoverable from the fleet.** HEAD's
`train_flagship4b.py` has **no `--jerk-weight` and no `--aux-accel`** (its own `--help`), while
`MODEL_REGISTRY.md:172` lists v1's flags as `--speed-input --jerk-weight 0.02 --aux-accel`,
`rollout_k=4`. A fleet-wide search found only `pod2:/workspace/speed_input/stack/scripts/
train_flagship4b.py` (Jul 14, 421 lines) — it carries `--jerk-weight` and `--rollout-k` but
**still no `--aux-accel`**. ⇒ **No trainer on the fleet reproduces v1's launch.** The registry
already warned this at `:177` (*"trained with a pod-side trainer that was never committed"*).

⛔ **Therefore "fine-tune v1 with exactly one flag different" IS NOT ACHIEVABLE.** A fine-tune from
HEAD would differ in **three** ways — `rollout_k` (intended) plus losing `jerk_weight=0.02` and
`aux_accel` (unintended) — and any CR change could be attributed to the missing loss terms.

### ⭐ THE FIX: make the CONTROL a matched fine-tune, not v1 itself

| arm | start | trainer | `rollout_k` |
|---|---|---|---|
| **RR-CTL** (control) | v1 ckpt step 29,999 | HEAD `train_flagship4b.py` | **4** (v1's value) |
| **RR-20** (treatment) | v1 ckpt step 29,999 | HEAD `train_flagship4b.py` | **20** (matched to the evaluated horizon) |

⇒ **The jerk/aux difference is COMMON TO BOTH ARMS AND CANCELS.** The only difference between
RR-CTL and RR-20 is `rollout_k`, which is precisely what the primary endpoint tests.

⛔ **CONSEQUENCE THAT MUST TRAVEL:** the comparison is **RR-20 vs RR-CTL**, ⛔ **NOT RR-20 vs v1**.
v1's published CR (3.50 / 15.16 / 64.21 / 80.77) is a **different training configuration** and may
**not** be used as the control for this experiment. It remains the reference for *why* the
experiment exists, not the arm it is measured against.

⚠️ **Both arms lose `jerk_weight` and `aux_accel` relative to v1.** Neither is a deployable
successor to v1; this is a **mechanism test**, and its checkpoints must not be promoted.

**PRIMARY unchanged:** `CR_k` at k=4/8/16/20, paired episode-cluster bootstrap, interval on the
DIFFERENCE, same 40-episode surface. **Success = RR-20's CR is lower than RR-CTL's with the CI
excluding zero at k=16 and k=20.**

**Length:** set from a MEASURED `step_s` over the first 200 steps of RR-20, not assumed. Both arms
run the SAME number of steps — ⛔ an unequal-length comparison is not a comparison.
