# ⭐ E-CR v2 — C61 RESOLVED: the imagination decay is COMPOUNDING, not task difficulty

**MEASURED 2026-07-29, v1 `flagship4b-speedjerk-30k` step 29,999, `val40cache`, **761 windows** /
40 episodes** — a subset of the canonical surface, see the window-count caveat below. Artifacts: `ecr_v2.py`, `ecr_v2_ci.py`,
`ecr_v2.json`, `ecr_v2_ci.json`, `ecr_v2_arrays.npz` (pod3 `/workspace/`).

**Pre-registration:** `Project Steering/PREREG_deep_research_2026-07-29.md`.
**Estimator:** **PAIRED episode-cluster bootstrap** (`taniteval/ci.py`, B = 2000). The interval is on
the **DIFFERENCE** `e_rollout − e_teacher_forced`. ⛔ **No CI on the RATIO is computed and none may
be quoted.**

## The result — PRIMARY, 761 windows / 40 episodes

`e_k = 1 − cos(z_hat_k, z_true_k)` — latent error, **no decoder in the path** (that decoder was the
C63 confound). Artifacts `ecr_v2_full.json` / `ecr_v2_full_ci.json`.

| k | horizon | `e_rollout` | `e_teacher_forced` | **CR** | Δ (roll − tf) | CI95 | separated |
|---|---|---|---|---|---|---|---|
| 4 | 0.4 s | 0.025194 | **0.007189** | **3.50** | +0.0180 | [0.0143, 0.0222] | ✅ |
| 8 | 0.8 s | 0.123405 | **0.008140** | **15.16** | +0.1153 | [0.0696, 0.1654] | ✅ |
| 16 | 1.6 s | 0.514255 | **0.008009** | **64.21** | +0.5062 | [0.4029, 0.6096] | ✅ |
| 20 | 2.0 s | 0.583723 | **0.007227** | **80.77** | +0.5765 | [0.4752, 0.6735] | ✅ |

`p_delta_gt0 = 1.0` at every horizon.

### ⭐ It REPLICATES — the first run's window defect did not drive the result

The first run (488 windows, before the keep-mask fix below) gave CR **3.58 / 13.64 / 64.45 / 77.81**;
this one, on a **56 % larger and differently-skewed** window set, gives **3.50 / 15.16 / 64.21 /
80.77**. Same direction, same order of magnitude, all four separated in both. **The verdict does not
depend on which windows survived**, which is the check that matters after a non-random exclusion is
found. Artifacts for the first run are kept as `ecr_v2.json` / `ecr_v2_ci.json`.

## ⭐ The load-bearing observation: the teacher-forced arm is FLAT

**`e_teacher_forced` = 0.00724 / 0.00779 / 0.00770 / 0.00733 across k = 4 → 20.** A prediction made
one step from **truth** is just as accurate at 2.0 s into the episode as at 0.4 s.

⇒ **THE TASK DOES NOT GET HARDER WITH HORIZON.** Every bit of the growth in the rollout arm comes
from feeding predictions forward.

## Verdict against the pre-registered outcomes

| registered outcome | fired? |
|---|---|
| **CR ≈ 1, CI covering** ⇒ H-TASK, "accelerating decay" FALSIFIED, no architecture change | ❌ |
| **CR rising super-linearly, CI excluding** ⇒ **H-COMPOUND** | ✅ **THIS ONE** |
| rising but CI covers ⇒ UNDERPOWERED | ❌ (CI excludes at all four k) |

⭐ **H-COMPOUND CONFIRMED. H-TASK FALSIFIED.** CR rises **3.58 → 77.81** over 0.4 → 2.0 s.

## What this unblocks — and what it does NOT

**UNBLOCKED** (the prereg's H-COMPOUND branch):
1. ⭐ **Rollout-recovery training** — HorizonDrive's mechanism, training on prediction-corrupted
   histories. This is now the *indicated* fix and the highest-value training change the programme has.
2. **E-ROLL** (recursive k=1 past 2 s) — its registered expectation "we expect it to diverge" is now
   *explained* rather than merely predicted.
3. **The Koopman/spectral lever** becomes a legitimate SECOND experiment — its whole premise is
   attenuating multiplicative error growth, and we have now shown that growth is real. ⚠️ Still LOW
   confidence, still wrong domain (proprioceptive, no ViT), and still risks erasing the speed/scale
   magnitude our speed fix bought (R² 0.965) — run it **after** rollout-recovery, reporting CR and
   speed R² jointly.

⛔ **NOT established, and must not be inferred:**
- **This does not restore the retracted "decay accelerates" wording.** C61 retracted a claim the
  *then-available measurement could not support*. That retraction was correct on its own terms; this
  is a **new measurement on a different surface** that happens to support the same direction.
  The lesson of C61 stands unchanged.
- **This says nothing in metres.** `1 − cos` is latent error. The link to ADE runs through
  `step_readout`, which C63 measured to be non-linear and domain-sensitive
  (a two-true-latent decode scored **6.76 m** at 2 s vs the rollout's 0.53 m). **Do not convert
  latent error into trajectory error, in either direction.**
- **CR's magnitude is not a physical quantity.** 77.81 is a ratio of cosine distances whose
  denominator is near a noise floor (0.0073). Quote the **direction and separation**, not the number
  as if it were an effect size in the world.

## ⚠️ WINDOW COUNT — 488, not the canonical 881, and the loss is NOT random

The canonical 40-episode surface carries **881** windows. This run scored **488 (55.4 %)**.
I first wrote "881" into this document from the canonical figure rather than from the artifact;
`ecr_v2_ci.json` says 488. **Read the artifact, not the number you expect.**

**Cause — a defect in my driver, not in the data.** Teacher forcing needs the true future latents for
all K=20 steps, which the driver gathers from the windows starting 1…20 frames later. When any window
in a batch lacks a full future — i.e. sits within 20 frames of its episode end — the driver executes
`continue` and **discards the ENTIRE BATCH of 8**, not just the offending window.

⇒ **The dropped windows are systematically END-OF-EPISODE**, plus up to 7 innocent neighbours each
time. That is a non-random exclusion and must travel with the number.

**Does it change the verdict? No — and this was then TESTED rather than asserted.**

✅ **FIXED 2026-07-29 03:0x** — per-window keep-mask instead of discarding the batch. The re-run
selects **761 / 881 windows (86.4 %)**, recovering **273**. The residual 120 genuinely lack a 20-step
future (they sit within 20 frames of an episode end) — that is a **legitimate exclusion inherent to
the measurement**, not a defect, and it cannot be removed without shortening K.

⭐ **The re-run REPLICATES the verdict** (see the primary table above): CR 3.50 / 15.16 / 64.21 /
80.77 against the first run's 3.58 / 13.64 / 64.45 / 77.81, all separated in both. So the
end-of-episode skew was **not** driving the result.

⛔ Even at 761 windows, **do not call this "the canonical 881-window surface."** It is 86.4 % of it,
and the missing 6 % is structurally end-of-episode.

## Why v2 and not v1

E-CR v1 built CR on **decoded displacement** and returned CR < 1 — outside both registered outcomes.
C63 diagnosed it: `step_readout` is tuned to the predictor's own output distribution, so any arm fed
true latents leaves its domain. v2 removes the decoder entirely; both arms are the predictor's own
outputs compared against the encoder's, so nothing crosses a distribution boundary.

⚠️ **The same precondition question applies to v2 and was checked:** `world.encode` and
`world.encode_window[:, -1]` are the same space (cosine 0.999995), so `z_true` is a legitimate target
for `z_hat`. And `cos(z_hat, z_true_next) = 0.98377` — the predictor *is* aiming at the encoder's
next output, even though it lands closer to "stay put" (`cos(z_hat, z_last_ctx) = 0.99872`).

## Evidence class

| claim | class |
|---|---|
| the CR table + intervals | **MEASURED (ours)** — **761** windows (86.4 % of 881; replicated on a 488-window subset) / 40 eps, paired episode-cluster bootstrap B=2000 |
| teacher-forced arm is flat in k | **MEASURED** |
| H-COMPOUND | **MEASURED**, per the pre-registered decision rule |
| rollout-recovery training will help | **HYPOTHESIS** — indicated by the mechanism, not yet tested here |
