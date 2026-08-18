# Encoder localisation — where does REF-A lose the information?

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **Zero GPU on Thor. Zero pod.**
**Every claim is stamped.** `MEASURED` (ours + artifact) · `PUBLISHED` · `INHERITED` (not re-verified)
· `ESTIMATED` · `HYPOTHESIS`.

---

## 0. ⛔ THE PARADOX AS BRIEFED DOES NOT SURVIVE ITS OWN SCOPE CHECK

The brief poses: *"a pretrained encoder that reads the scene 91× better produces a model that drives
5× worse"*, and calls the ablation clean on the registry's word. **Before measuring the middle of
REF-A's pipeline I checked that the two endpoints describe the same system. They do not.**

**`MEASURED` (reading our own source and our own banked run-meta; `file:line` and artifact given for
every row, so none of this needs me to be believed).**

### 0.1 The two DINOv2s are the same WEIGHTS on DIFFERENT IMAGES

| axis | C104's DINOv2 (reads `lead_gap` **0.44997**) | REF-A's DINOv2 (the trained arm) | source |
|---|---|---|---|
| source frame | **256 × 640**, `projection: cylindrical`, **`hfov_deg: 120.0`** | **256 × 256 square**, `ftheta_crop`, `f_eff_px = F_REF = 266` | `…/2026-08-18-pooling-ladder-ER10/raw/dino_meta.json` (`frame`) · `stack/tanitad/data/physicalai.py:140-143` (`CORPUS_META`) + `stack/tanitad/data/calib.py:38` (`F_REF = 266.0`) |
| ⭐ **horizontal field** | **120.0°** | **51.39°** = `2·atan(128/266)` | DERIVED from the two rows above |
| encoder input | **224 × 560**, aspect **0.4000**, isotropic | **224 × 224**, aspect **1.0** | `dino_meta.json` (`external_input_hw`) · `stack/scripts/dino_precompute.py:44-47` (`interpolate(size=(size,size))`, `size=224`) |
| token grid | **16 × 40 = 640** | **16 × 16 = 256** | same two |
| temporal content per token | **3 sub-frames CONCATENATED** ⇒ `d_model_tokens` **2304** | **latest frame only** ⇒ **768** | `dino_meta.json` (`sub_frames_concatenated: 3`) · `dino_precompute.py:43` (`frames[:, -3:]`, i.e. the latest RGB triplet = ONE image) |
| window set | **130 clips, 2 809 frames, `parity: False`**, lead-ENRICHED probe set | **40 val episodes, 881 windows**, the parity corpus | `dino_meta.json` (`n_episodes`, `parity.train_parity`) · `taniteval/results/driving_refa-dinov2.json` (`n_windows`, `n_episodes`) |

⛔ **REF-A's frozen DINOv2 saw 51.4° of horizontal field. C104's saw 120.0° — 2.33× more.** It also
saw one frame where C104's saw three, on a square grid where C104's had a 16×40 wide grid, over a
different and deliberately lead-enriched clip set.

⇒ **`0.44997` is not a fact about the features REF-A consumed.** It is a fact about a different
tensor produced by the same weights. The inference *"the same encoder gives opposite results"* is a
**scope error of the `df` / Thor-`free` / cgroup-`usage_in_bytes` family: a real number read against
the wrong system.**

### 0.2 ⛔ The brief's tier stamp is inverted — BOTH endpoints are tier-0

The brief states *"the 5.1× is a **T1** driving number … do not compare across tiers."*
**`MEASURED`:**

```
taniteval/results/driving_refa-dinov2.json   block = taniteval.driving/tier0
taniteval/results/driving_flagship-30k.json  block = taniteval.driving/tier0
both:  claim_strength = "open-loop / weak (arXiv:2605.00066)"
```

⇒ The 2.1675 vs 0.4271 comparison is **T0 open-loop**, self-declared *weak*, **not T1**. Per
`EVAL_DOCTRINE` T0 is *"a WM diagnostic — NEVER driving performance"*, so **"drives 5× worse" is not
an admissible reading of it** at all. *(The good news: both sides are the SAME tier, so the ratio is
internally valid as a T0 diagnostic — it is the words "driving performance" that are inadmissible,
not the arithmetic.)*

### 0.3 ⛔ The registry's "differ in exactly two things" is already known false, in-repo

`MODEL_REGISTRY.md:2899` (D-A4) claims flagship and REF-A *"differ in exactly two things (encoder,
SIGReg target)"*. **This was already refuted and the refutation is sitting unmerged into the
registry:** `TanitAD Research Hub/Architecture & Inference/ARCHITECTURE_WIRING_COMPARISON.md:670` —
REF-A **has no `ImaginationField`** (flagship: **22,055,683 params**); `RefAModel.__init__` never
constructs one. ⇒ **A third axis, and a whole mechanism.**

⚠️ **And a fourth, which I add here:** the deployed arm ran **`--adapter temporal`**
(`MODEL_REGISTRY.md:1925`), i.e. `TemporalGridAdapter` in
`stack/experiments/reset-speed4b/refa_plus.py:23` — **not** the `DinoAdapter` / `DinoGridAdapter` at
`stack/tanitad/refs/refa.py:108,130` that the brief names. The brief's rung-2 target is the wrong
class.

### 0.4 ⛔ The banked dump cannot answer the briefed question — it holds NO latents

`taniteval/results/windows_refa-dinov2.pt` (96 104 B) contains exactly:
`pred [881,4,2] · gt [881,4,2] · cv [881,4,2] · eid (881) · speed [881] · head_deg [881] · wp_steps (4)`.
⇒ **Trajectories only.** No adapter output, no predictor latent. **Rungs 2 and 3 of the briefed
ladder are not computable from banked local data** — they need the checkpoint
(`Sayood/tanitad-refa-dinov2-4b`) forwarded over REF-A-geometry features. ⏳ **PI decision** (§4).

### 0.5 What "CONTAMINATED" meant — checked before use, as instructed

`eff_refa-dinov2.CONTAMINATED-20260720-215641.json` is one of **eleven** files renamed in a single
sweep at 2026-07-20 21:56–22:03 (`eff_flagship-nospeed`, `refa-dynin`, `refa-ijepa`, `refb`,
`refb-10k`, `refc-xl*`, …). Every one is **~6–8 KB against its clean sibling's ~18–24 KB**, i.e. a
truncated/partial earlier run superseded by a complete one. **I did not read either file as data**
— nothing in this document depends on `eff_*`.

---

## 1. What this leaves as the real question

The 91× is not a measurement of REF-A's input. So the honest decomposition is **not** "encoder vs
adapter vs predictor" — it is:

> **Is the gap the ENCODER, or the FIELD OF VIEW AND TEMPORAL CONTENT WE HANDED IT?**

⇒ **The cheapest discriminating experiment is to hold the encoder, the windows, the labels, the
pool, the ridge and the seeds FIXED, and vary ONLY the input geometry** — running DINOv2 at REF-A's
51.4° single-frame condition against C104's 120° three-frame condition on the **same 2 809 banked
windows**. That is a 2×2 factorial, it needs no checkpoint, no download and no pod, and **both
outcomes redirect the programme**:

| outcome | reading |
|---|---|
| REF-A-geometry DINOv2 **collapses** toward our encoder's 0.005 | the 91× is an artifact of field of view + temporal stacking. **"Swap the encoder" is the wrong fix**; the input pipeline is. |
| REF-A-geometry DINOv2 **holds** near 0.45 | the input is exonerated, the 91× survives into REF-A's actual features, and the loss is genuinely downstream — adapter or predictor. **Then the checkpoint download is justified.** |

**Pre-registered in §2 with both outcomes committed in advance, before any arm was run.**

---

## 2. PRE-REGISTRATION — E-GEOM, the 2×2 field-of-view × temporal-content ablation

**Written and staged BEFORE any arm was built or run.** Everything except the two declared factors
is held byte-identical: same 2 809 banked rows (same clips, same frame indices, same order), same
`lead130_agents.jsonl` join, same split, same `AvgPool2d`→16-cell readout, same fixed Gaussian
random projection to 2 048, same ridge, same seeds, same estimator.

### 2.1 The four arms

| arm | horizontal field | sub-frames per token | DINOv2 input | token grid | pool kernel → cells |
|---|---|---|---|---|---|
| **`wide3f`** | **120.0°** (full frame) | **3** (concat, d 2304) | 224 × 560 | 16 × 40 | `(4,10)` → 16 |
| **`wide1f`** | 120.0° | **1** (latest, d 768) | 224 × 560 | 16 × 40 | `(4,10)` → 16 |
| **`refa3f`** | **51.39°** (centre crop) | 3 (concat, d 2304) | 224 × 224 | 16 × 16 | `(4,4)` → 16 |
| **`refa1f`** | **51.39°** | **1** (latest, d 768) | 224 × 224 | 16 × 16 | `(4,4)` → 16 |

`wide3f` **is C104's exact condition** and doubles as the **replication gate**: if it does not
reproduce `lead_gap` ≈ **0.44997**, nothing else in this document is read.
`refa1f` **is REF-A's condition** (51.4° field, single frame, 16×16 grid, 4×4 pool — the same
`SpatialGridReadout(grid=4)` geometry `TemporalGridAdapter` uses).

The crop is computed, never hard-coded: the cylindrical frame is linear in angle
(640 px / 120° = 5.3333 px/deg), so 51.39° ⇒ **274 px**, centred.

### 2.2 Declared residuals — what this emulation does NOT reproduce

Stated so they cannot be discovered later as an unlogged confound:
1. **Projection**: cylindrical here, `ftheta_crop` in REF-A. The radial warp differs.
2. **Vertical field**: the cylindrical frame gives **45.4°** (`2·atan(128/305.577)`); REF-A had
   **51.4°**. The crop keeps all 256 rows, so `refa*` resizes 256×274 → 224×224, a **7 % anisotropic
   stretch**. Both residuals are **far smaller than the 2.33× horizontal factor under test**, and
   both act on `refa*` only.
3. **Corpus**: these 130 clips are the lead-enriched probe set (`parity: False`), not REF-A's val40.
   ⇒ The ablation is **within-window paired**; it is not a re-measurement of REF-A's own windows.

### 2.3 Targets, controls, estimator — the traps the brief names, each answered

- **Targets**: `lead_gap`, `ego_v0`, `lead_closing` (C104's three quoted rungs).
- ⛔ **`intercept_col=-1`** on every fit — the C92 repair. The harness **refuses to start** if
  `ridge_fit` lacks the parameter.
- ⛔ **Trivial-proxy control, per arm**: every rung is reported **raw AND with `v0` partialled out**
  (`corr_partial_v0`, `r2_ceiling_partial_v0`). C104's `lead_gap` **0.45 collapses to partial r²
  0.120** — the partialled column is the one that carries the claim.
- ⛔ **Positive control**: **`PC-LOCAL` / `PC-DIST`**, our own tokens through the deployed pool
  (C109: **never `PC-2OBJ`**, which is inert at the deployed ratio by construction).
- ⛔ **Seeds**: projection seeds **0, 1, 2** on every arm, **spread reported**, per C103.
- ⛔ **Estimator**: **episode-cluster bootstrap** (`taniteval/ci.py`), paired form for arm-vs-arm
  deltas. **Never `overlapping_holdout_se`.** A per-seed spread is reported AS a spread and is
  **not** a confidence interval (C109).
- ⛔ **Degenerate-input check** (C119) + the **K1 sd-ratio guard** (C97) run as banked by the ER10
  harness.
- **Tier**: every number below is **T0-DIAGNOSTIC**. It is a readout, not driving.

### 2.4 The verdicts, committed in advance

Read on **`lead_gap`, partial-`v0`**, on the **paired** `refa1f` − `wide3f` delta:

| | criterion | conclusion it licenses |
|---|---|---|
| **G-INPUT** | `refa1f` loses **≥ 50 %** of `wide3f`'s partial r², CI excluding 0 | **The input pipeline, not the encoder, is the dominant term.** C104's 91× does not describe REF-A. The next experiment is the FIELD, and the checkpoint download is NOT yet justified. |
| **G-ENCODER** | `refa1f` retains **≥ 80 %**, CI containing 0 for the delta | **The input is exonerated**; the 91× survives into REF-A's real features ⇒ the loss is downstream. **This is what justifies asking the PI for the checkpoint.** |
| **G-MIXED** | anything between | report the decomposition per factor and take **neither** conclusion; the 2×2 still attributes FOV vs temporal separately. |

⚠️ **A collapse on `refa1f` does NOT by itself exonerate the adapter or the predictor** — it only
removes the evidence that currently indicts the encoder. Rungs 2–3 stay open either way.
