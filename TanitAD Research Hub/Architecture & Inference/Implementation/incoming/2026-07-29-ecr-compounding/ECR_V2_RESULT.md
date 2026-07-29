# ⭐ E-CR v2 — C61 RESOLVED: the imagination decay is COMPOUNDING, not task difficulty

**MEASURED 2026-07-29, v1 `flagship4b-speedjerk-30k` step 29,999, `val40cache`, 881 windows /
40 episodes** — the canonical parity surface. Artifacts: `ecr_v2.py`, `ecr_v2_ci.py`,
`ecr_v2.json`, `ecr_v2_ci.json`, `ecr_v2_arrays.npz` (pod3 `/workspace/`).

**Pre-registration:** `Project Steering/PREREG_deep_research_2026-07-29.md`.
**Estimator:** **PAIRED episode-cluster bootstrap** (`taniteval/ci.py`, B = 2000). The interval is on
the **DIFFERENCE** `e_rollout − e_teacher_forced`. ⛔ **No CI on the RATIO is computed and none may
be quoted.**

## The result

`e_k = 1 − cos(z_hat_k, z_true_k)` — latent error, **no decoder in the path** (that decoder was the
C63 confound).

| k | horizon | `e_rollout` | `e_teacher_forced` | **CR** | Δ (roll − tf) | CI95 | separated |
|---|---|---|---|---|---|---|---|
| 4 | 0.4 s | 0.025924 | **0.007235** | **3.58** | +0.0187 | [0.0142, 0.0242] | ✅ |
| 8 | 0.8 s | 0.106293 | **0.007790** | **13.64** | +0.0985 | [0.0602, 0.1443] | ✅ |
| 16 | 1.6 s | 0.496283 | **0.007700** | **64.45** | +0.4886 | [0.3848, 0.5945] | ✅ |
| 20 | 2.0 s | 0.570082 | **0.007327** | **77.81** | +0.5628 | [0.4613, 0.6665] | ✅ |

`p_delta_gt0 = 1.0` at every horizon.

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
| the CR table + intervals | **MEASURED (ours)** — 881 windows / 40 eps, paired episode-cluster bootstrap B=2000 |
| teacher-forced arm is flat in k | **MEASURED** |
| H-COMPOUND | **MEASURED**, per the pre-registered decision rule |
| rollout-recovery training will help | **HYPOTHESIS** — indicated by the mechanism, not yet tested here |
