<title>SPEC E-AI-LDAD-0 — is the frozen trunk's latent displacement action-organised?</title>

# `E-AI-LDAD-0` — pre-registration

`Research Lab · 2026-09-18 (LAB-RUN-015) · Architecture & Inference · transfer test of Delta-JEPA (arXiv 2606.31232, full text read today)`
⛔ **Pinned before the code was run** (`raw/prereg_pin.json`).

## 1 · Question

Delta-JEPA's LDAD decodes the executed action from the latent **displacement** `Δz = z_{t+1} − z_t`
and reports that this beats decoding from concatenated endpoints (+0.67 to +12.60 pp planning
success, 3 seeds, Table 2). Our encoder is **frozen** (AI13-2), so an encoder-side LDAD loss cannot
act on it. The admissible question for us is therefore: **does the frozen refcv5-v2 trunk's
displacement ALREADY carry the ego action beyond what raw pixels carry?**

## 2 · Data and design

Same bank, speed join and **same seeded fit/score clip split** as VGEO-3/4/5
(`bevhead-20260913`, 134 clips, seed `20260913`, 94 fit / 40 scored). Pairs at a gap of **K = 8
frames (0.8 s)** on the STEP-4 grid. Targets (ego → label derivation, admissible): **LON** =
`(v[t+K] − v[t]) / 0.8` m/s²; **LAT** = mean yaw rate on `[t, t+K)` rad/s. Inputs: **vision only**.
Ridge; λ by clip-grouped 5-fold CV **on the fit split only**. R² = 1 − ΣSSE/ΣSST with SST about
the **fit-split** mean. `n` and `d` printed per arm.

Arms: `latent_dz` (704-d) · `latent_concat` (1408) · `latent_start_only` (704) · `pixel_dp` (1440) · `pixel_concat` (2880).

## 3 · Controls — each must read a KNOWN value

| id | bar | failure means |
|---|---|---|
| C-const (fit mean) | R² **exactly 0.0000** | the scorer is not what it says |
| C-oracle (target + 1 % noise) | R² **≥ 0.99** | the pipeline cannot detect a perfect signal ⇒ VOID-POWER |
| C-shuffle (`Δz` to a random end frame of the same clip) | CI upper **< R²(latent_dz)** and point **≤ 0.05** | structure survives a time shuffle ⇒ scene leakage, not dynamics ⇒ VOID-LEAK |
| pixel floor | reported | a latent that does not beat pixels has added nothing |

## 4 · Committed outcomes (per target, LON and LAT read separately)

Gate: controls pass. Then, primary = `latent_dz − pixel_dp` (paired clip bootstrap):

| branch | condition | consequence |
|---|---|---|
| **DZ-TRUNK** | CI lower > 0 **and** `latent_dz − latent_start_only` CI lower > 0 | the frozen displacement is action-organised beyond pixels and beyond the scene ⇒ an LDAD lever for us is **predictor-side** (supervise `ẑ_{t+1} − z_t`), and it is cheap |
| **DZ-SCENE** | `latent_dz − latent_start_only` CI contains 0 or < 0 | the action is read off the scene, not the displacement ⇒ LDAD's mechanism is not what the trunk offers |
| **DZ-TIE / DZ-PIXELS** | primary CI contains 0 / upper < 0 | the frozen trunk adds no displacement geometry for the action ⇒ LDAD transfers only with an **unfrozen or adapted** encoder, which feeds AI13-2 as a cost of freezing |
| ⚠️ WORSE-BRANCH | anything off-table | stated explicitly |

⚠️ **Scope.** A linear probe that FAILS proves only "not linearly decodable". `latent_concat`
contains `latent_dz` as a linear function, so the concat-vs-delta comparison Delta-JEPA runs
**cannot** be made with a linear probe and is not attempted. One trunk, one checkpoint, clip
bootstrap only (`H-ESTIM-SEED-1`). ⛔ Tier: none; nothing here is a driving claim.
