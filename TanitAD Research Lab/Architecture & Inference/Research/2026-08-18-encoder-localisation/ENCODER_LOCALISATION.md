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

### 2.5 Geometry worked out BEFORE the numbers existed — what should and should NOT move

Written while `wide3f`'s cache was still being written to disk, **before any r² was produced**. It is
recorded because an expectation formed after seeing results is worthless, and because it makes the
`refa*` arms falsifiable rather than merely descriptive.

**`lead_gap` should survive the crop.** `gt_lead_gap` selects the nearest agent with
`|cy| ≤ 1.75 m`, `0 < cx ≤ LEAD_MAX_M` (`…/2026-08-17-slot-probe-parity/code/sp2_probe.py:99-113`).
A lead at `cx > 3.6 m` subtends `atan(1.75/3.6) = 25.9°` — **inside** REF-A's ±25.7° half-field. So
the crop removes almost no in-corridor leads.
**Resolution on the lead is also preserved**: `wide` maps 640 px → 560 px (×0.875), so the central
274 px becomes ~240 px ≈ **17 patches**; `refa` maps 274 px → 224 px (×0.818) = **16 patches**.
Vertically both are 256 → 224. ⇒ **The lead object is the same size in patches in both arms.**

⇒ ⭐ **A collapse of `lead_gap` on `refa*` therefore could NOT be explained by "the lead left the
frame" or "the lead got smaller" — the geometry above forbids both.** That is what makes this arm
worth running rather than assuming.

**`ego_v0` is the one that should suffer.** What the crop *does* remove is the **peripheral ±34° on
each side**, which is where optical flow is largest and where egomotion speed is most legible.
⚠️ **And that is exactly REF-A's diagnosed failure mode**: the four-ablation bottleneck diagnosis
attributed **71–83 % of REF-A's error to SPEED/SCALE magnitude**, not rotation or imagination
(`INHERITED`, prior-session diagnosis — flagged as not re-verified by me).

⇒ **The sharp, pre-registered secondary reading:** if the arms separate on **`ego_v0`** while
`lead_gap` holds, then REF-A's speed blindness has a **measured cause in the input pipeline** —
a 51.4° field discards the peripheral flow that carries speed — and it is **not** evidence about
the encoder at all. If instead everything holds, the input is exonerated and the indictment moves
downstream, to the adapter and the predictor.

---

## 3. RESULT — the input pipeline is EXONERATED, and the paradox SURVIVES the scope correction

**`MEASURED` 2026-08-18, dev-box RTX 4060, zero Thor, zero pod.** Artifacts: `raw/encloc_<arm>.json`,
`raw/encloc_summary.json`, build/ladder logs per arm. **Tier: T0-DIAGNOSTIC throughout.**

### 3.1 ⭐ The replication gate passes to five decimal places

`wide3f` is C104's exact condition, rebuilt here from the raw frames through my own cache builder and
my own patched ladder:

| target | C104 (`er10_dino.json`) | this run (`encloc_wide3f.json`) |
|---|---|---|
| `ego_v0` | 0.71733 ± 0.00781 | **0.71733 ± 0.00781** |
| `lead_gap` | 0.44997 ± 0.02465 | **0.44997 ± 0.02465** |
| `lead_closing` | 0.01713 ± 0.00142 | **0.01713 ± 0.00142** |

⇒ **Identical.** The instrument reproduces the published condition, so every arm below is measured on
a gate-passing harness rather than on one that merely agrees with itself (C94).

### 3.2 The 2×2 — and the direction is the OPPOSITE of the hypothesis

r² ceiling, mean ± sd over projection seeds 0/1/2, `intercept_col=-1`, episode-cluster bootstrap:

| arm | field | frames | grid | `ego_v0` | `lead_gap` | `lead_closing` |
|---|---|---|---|---|---|---|
| **`wide3f`** (C104) | 120.0° | 3 | 16×40 | **0.71733** ± 0.0078 | **0.44997** ± 0.0247 | 0.01713 ± 0.0014 |
| `wide1f` | 120.0° | 1 | 16×40 | 0.69067 ± 0.0098 | 0.45223 ± 0.0199 | 0.01280 ± 0.0023 |
| `refa3f` | 51.39° | 3 | 16×16 | 0.61050 ± 0.0664 | 0.48570 ± 0.0163 | 0.05263 ± 0.0076 |
| ⭐ **`refa1f`** (**REF-A's condition**) | **51.39°** | **1** | **16×16** | **0.67147** ± 0.0157 | ⭐ **0.52850** ± 0.0388 | **0.04837** ± 0.0034 |

**Partial-`v0` (the trivial-proxy control, the column that carries the claim):**
`wide3f` r = **+0.346** · `wide1f` **+0.389** · `refa3f` **+0.409** · ⭐ `refa1f` **+0.523**.

⇒ ⛔ **NOTHING COLLAPSES. `refa1f` — the geometry REF-A actually consumed — reads `lead_gap` at
1.17× C104's headline arm, and 1.51× on the partialled correlation.** `lead_closing` is **2.8×**
higher. Only `ego_v0` gives ground, and only by **6.4 %** (0.6715 vs 0.7173).

**Both pre-registered directions were committed in §2.4 before the run. The one that fired is
`G-ENCODER`: `refa1f` retains ≥ 80 % on every rung — it retains 94 %, 117 % and 282 %.**

### 3.3 ⛔ My own §2.5 prediction was WRONG, and I am recording that rather than quietly dropping it

§2.5 predicted `ego_v0` would be the rung that suffered, because the crop removes the peripheral
optical flow that carries speed, and because REF-A's diagnosed failure was speed/scale. **Measured:
`ego_v0` falls only 6.4 %, and `lead_gap` — which I predicted would merely survive — IMPROVES 17 %.**

⇒ The peripheral-flow argument was another **mechanism narrated from a diagram** — the same
root-cause class C104 logged. It is recorded here because a pre-registration whose failed half is
deleted is not a pre-registration.
⭐ **And it kills a candidate explanation for REF-A's speed blindness**: a 51.4° field does *not*
cost DINOv2 its speed readout, so REF-A's speed failure cannot be blamed on its field of view.

### 3.4 The positive control fires hardest on exactly the arm that needed it

⛔ `PC-LOCAL` (**not** `PC-2OBJ` — C109 measured that one inert at the deployed ratio by
construction), planted into a 2×2 token block wholly inside one pooled cell, **with the block
derived per grid** rather than hard-coded to 16×40 — the hard-coded columns 18–19 do not exist on a
16×16 grid:

| arm | un-planted `lead_closing` | **PC-LOCAL planted** |
|---|---|---|
| `wide3f` | 0.01713 | **0.38900** |
| `wide1f` | 0.01280 | **0.35095** |
| ⭐ `refa3f` (square grid) | 0.05263 | ⭐ **0.91140** ± 0.0026 |

⇒ **The `s16` square-grid arm has MORE power to see a pooling-destroyed signal than the wide arm,
not less.** A null on `refa*` could not have been an instrument failure — and there is no null on
`refa*` to explain.

### 3.5 What is NOT claimed

- `lead_closing` **fails its own K1 guard on every un-planted arm** (`K1_PASSES: false`, `R² < 0`,
  `pred_sd/gt_sd ≈ 0.09`). It is a **degenerate near-constant readout in all four arms**, including
  C104's. ⇒ The 2.8× is a ratio of two numbers that are both inadmissible as capability claims; it is
  reported for completeness and **must not be quoted as "REF-A's features read closing rate"**.
  ⚠️ **This also applies to C104's own `0.01713 vs 0.00000`** — that rung failed K1 there too.
- These are **the lead-enriched 130-clip probe windows, `parity: False`** — not REF-A's val40. The
  ablation is within-window and paired; it is not a re-measurement of REF-A's own eval.
- **T0-DIAGNOSTIC. A readout is not driving.**

### 3.6 ⭐ The control arm separates FIELD from GRID — and the crop is a NET BENEFIT

`refa1f` changes two things at once against `wide1f`: the field (120° → 51.4°) **and** the grid
(16×40 → 16×16, 640 → 256 tokens, plus the 7 % stretch). `squash1f` keeps the **full 120°** on the
**same 16×16 square grid**, which splits them. All three arms are single-frame, so the temporal axis
is held fixed:

| step | what changes | `ego_v0` | `lead_gap` | `lead_closing` |
|---|---|---|---|---|
| `wide1f` → `squash1f` | **GRID only** (16×40 → 16×16) | 0.6907 → 0.6267 (**−9.3 %**) | 0.4522 → 0.4551 (+0.6 %) | 0.0128 → 0.0135 (+5 %) |
| `squash1f` → `refa1f` | **FIELD only** (120° → 51.4°) | 0.6267 → 0.6715 (**+7.1 %**) | 0.4551 → 0.5285 (**+16.1 %**) | 0.0135 → 0.0484 (**+258 %**) |

⇒ ⭐ **The narrow field HELPS on every rung, `ego_v0` included.** With the token budget fixed at 256,
spending it on the 51.4° corridor where the in-corridor lead lives beats spreading it over 120°. The
grid reduction is the only term with a cost, and it costs `ego_v0` alone.

⇒ **Every one of the three input-pipeline differences between C104's DINOv2 and REF-A's is now
measured, and NONE of them destroys the signal.** Temporal stacking: inert (§3.2, `wide3f` vs
`wide1f`). Grid: −9 % on one rung. Field: **positive on all three**.

---

## 4. WHAT THIS CHANGES

### 4.1 ⛔ "The encoder is the constraint" does not survive contact with REF-A

C104's finding — that through our deployed pool `dinov2-base` reads `lead_gap` **0.44997** where our
v6 encoder reads **0.00496** — **is not challenged here.** I reproduced it to five decimals (§3.1).
Our trained encoder is weak on that rung, and that stands.

⛔ **What does not survive is the INFERENCE drawn from it**, that a better encoder is therefore the
lever. **REF-A is the experiment that already ran that lever**, and this document establishes the
step everyone was missing: **REF-A's frozen features really did carry the signal** — at **1.17×**
C104's own headline number and **1.51×** on the partialled correlation — and REF-A is still the
**worst arm in the programme** (T0 ADE@2s **2.1675 [1.9081, 2.4212]** vs flagship-30k **0.4271**).

⇒ ⭐ **A representation reading `lead_gap` at r² 0.53 produced a driver 5.1× worse than one reading
it at 0.005.** On this evidence **linear readability of the frozen representation does not predict
driving quality at all**, and "swap in a stronger encoder" is **not supported by our own strongest
test of it**.

### 4.2 ⭐ This converges with the parallel reconciliation stream, from the opposite side

`…/Benchmarks & Eval/Implementation/incoming/2026-08-18-refa-reconciliation/REFA_RECONCILIATION.md`
(same branch, same day, independent) ran the **driving-side** counterpart, E-RECON-1, and hit its
pre-registered **null**: REF-A's paired deficit vs the flagship is **+1.7150 m on windows with a
lead** and **+1.7295 m without one**, contrast **−0.0146 [−0.5988, +0.5551]**, not separated.

| stream | question | answer |
|---|---|---|
| **theirs** (driving side) | does having a lead in frame change REF-A's deficit? | **No** — "readable ≠ usable" |
| **mine** (input side) | did REF-A's encoder even deliver a readable lead? | ⭐ **Yes — better than C104's arm** |

⇒ Together these close the loop that neither closes alone: **the information is present at REF-A's
input, and it does not reach REF-A's driving.** Their §3 lists **four** flagship/REF-A differences
against the registry's claimed two (encoder, SIGReg target, the **absent 22.06 M `ImaginationField`**,
grounding depth **13.43 M vs 4.48 M**) and states that **none had been isolated**. ⭐ **One now is:**
the **fifth** difference — the input geometry — which I add to their list and **measure inert**.

### 4.3 Where the loss can still be, and how tightly this bounds it

⭐ **The adapter's ARCHITECTURE is bounded as adequate, without the checkpoint.** REF-A's deployed
`TemporalGridAdapter` (`stack/experiments/reset-speed4b/refa_plus.py:23`) is
`SpatialGridReadout(256, 1536, grid=4, d_readout=128)` — `AvgPool2d((4,4))` to 16 cells, then
`Linear(1536→128)` per cell, out 2048 (`stack/tanitad/models/readout.py:118-125`). **The ladder's
`s16` arm applies that same pool and compresses the same 16 cells to the same 2048 features** — and
measures **0.4857 / 0.5285**.
⚠️ **Stated as a BOUND, not an equivalence:** the ladder's projection is **dense** across cells while
the adapter's is **block-diagonal with weights shared across cells**, which is strictly less
expressive. ⇒ §3.2 is an **upper bound** on what REF-A's adapter architecture can preserve, and the
bound is high. **It does not exonerate the adapter's LEARNED weights.**

Remaining candidates, none measured:

| candidate | why it is still live | what it needs |
|---|---|---|
| ⭐ **the adapter's TRAINED weights** | `refa.py:19-31` names **"collapse-to-easy-targets"** as the adapter's *known* failure mode, and `refa_train4b.py:360` logs `adapter_std` as the collapse monitor — **the run's own collapse detector exists and nobody has read it** | the checkpoint (§5), or the run's `metrics.json` |
| the predictor latent | untested at any rung | the checkpoint |
| the 22.06 M absent `ImaginationField` + shallower grounding | their §3 #3/#4, unisolated | a training arm, not a probe |
| ⭐ **"readable ≠ usable" as the real answer** | their E-RECON-1 null is direct evidence for it | already measured; §5 registers the sharpening |

### 3.7 The paired deltas, and the pre-registered verdict

**Paired episode-cluster bootstrap across caches** (2 000 draws, 70 episode clusters, per projection
seed; row identity asserted element-wise, not assumed — the summariser **refuses** to pair if the
eval targets or episode ids differ). ⛔ Never `overlapping_holdout_se`. Artifact:
`raw/encloc_summary.json`.

**`refa1f` − `wide3f`, Δr², per seed:**

| target | raw Δ | partial-`v0` Δ | any seed separated? | retention |
|---|---|---|---|---|
| `ego_v0` | −0.0397, −0.0580, −0.0399 | — | **no** | **0.936** |
| `lead_gap` | +0.1446, +0.0033, +0.0878 | +0.1537, +0.0663, +0.1243 | **no** | **1.175** |
| `lead_closing` | +0.0326, +0.0331, +0.0281 | +0.0325, +0.0328, +0.0279 | no | 2.824 (degenerate — §3.5) |

⚠️ **Read honestly:** on `lead_gap` the point estimate is positive on **all three seeds** raw and
partialled, but the interval does **not** exclude zero. ⇒ **The admissible claim is "`refa1f` is not
worse", not "`refa1f` is better."** That is exactly what the question needed.

⭐ **The ONLY CI-separated negative anywhere in the design is `squash1f`'s `ego_v0`** (Δ −0.108,
−0.093, −0.071, separated on all 3 seeds) — i.e. the **grid** reduction at full field. `refa1f`,
which carries the same small grid, is **not** separated, because the crop compensates for it (§3.6).

### ⇒ VERDICT: `G-ENCODER`, as pre-registered in §2.4

> *"`refa1f` retains ≥ 80 %, CI containing 0 for the delta ⇒ the input is exonerated; the 91× survives
> into REF-A's real features ⇒ the loss is downstream. This is what justifies asking the PI for the
> checkpoint."*

**Retention 0.936 / 1.175 / 2.824 — all ≥ 80 %. No separated delta on any rung. Both conditions met.**
`G-INPUT` (≥ 50 % loss) is refused: not one rung lost anything.

---

## 5. WHAT I RECOMMEND, AND THE ONE THING THAT NEEDS THE PI

### 5.1 ⏳ PI DECISION — the checkpoint download, with the size measured rather than estimated

Rungs **2 (adapter output)** and **3 (predictor latent)** of the briefed ladder are **not computable
from anything on this box** (§0.4: the banked dump holds trajectories only, and there is no REF-A
checkpoint in any local cache — two probes, HF hub cache and the whole scratchpad).

**`MEASURED` via `HfApi.model_info(files_metadata=True)`, no bytes fetched:**

| repo | file | bytes |
|---|---|---|
| `Sayood/tanitad-refa-dinov2-4b` | `ckpt.pt` | **1 905 662 297 (1.906 GB)** |
| | `config.json` | 983 |
| | **total** | **1 905 666 517 (1.906 GB)** |

⛔ **I did not download it.** A brief is not consent. **This is a one-line yes/no for the PI.**

⭐ **But note what §4.3 already bought without it:** the adapter's *architecture* is bounded as
adequate, so the download now buys a **much sharper** question than the brief posed — not *"does the
adapter destroy it?"* but **"did TRAINING collapse the adapter?"**, which the run's own
`adapter_std` monitor (`refa_train4b.py:360`) was built to answer.
⚠️ **And `metrics.json` from the training run may answer it with NO download at all** — if it was
banked, `adapter_std` over 30 k steps settles the collapse question directly. **I could not locate it
locally; it was on `tanitad-pod3`.** ⇒ Worth one look before spending the 1.9 GB.

### 5.2 ⭐ E-ADAPT-0 — implemented and staged, runnable with zero download

§4.3's bound is loose in one specific way: the ladder projects **densely** across the 16 cells, while
`SpatialGridReadout` uses **one `Linear(d_model→128)` shared by every cell** — strictly less
expressive. I have implemented `--block-diag-proj 128` in `code/encloc_ladder.py`, which replaces the
dense projection with exactly that tied per-cell shape, making the operator REF-A's own up to the
weights being random rather than trained.

```
encloc_ladder.py --cache <tok_refa1f.pt> --arms s16 --block-diag-proj 128 \
    --targets ego_v0 lead_gap lead_closing --proj-seeds 0 1 2
```

**Both outcomes committed now:** if the tied per-cell map **preserves** `lead_gap`, the adapter's
architecture is exonerated outright and only its **trained weights** remain ⇒ the download is aimed
at one question. If it **collapses**, the adapter architecture is indicted **without any checkpoint
at all**, and the fix is a wider `d_readout` or an untied per-cell map — a cheap training change.

### 5.2b ⭐ E-ADAPT-0 RAN — the adapter's ARCHITECTURE is exonerated OUTRIGHT

I did not leave §5.2 registered; I ran it. **`MEASURED`, `raw/encloc_blockdiag_refa1f.json`,
`raw/log_blockdiag_refa1f.txt`** — `refa1f` geometry, `s16` pool, **one tied random
`Linear(768→128)` applied to every one of the 16 cells** (REF-A's `SpatialGridReadout` shape), 3
projection seeds, `intercept_col=-1`:

| projection | `ego_v0` | `lead_gap` | partial-`v0` r | `lead_closing` |
|---|---|---|---|---|
| dense (§3.2 `refa1f`) | 0.67147 | 0.52850 | +0.523 | 0.04837 |
| ⭐ **tied per-cell (adapter shape)** | **0.64860** ± 0.0193 | **0.48880** ± 0.0225 | **+0.484** | 0.04777 ± 0.0086 |
| C104's headline (`wide3f`) | 0.71733 | 0.44997 | +0.346 | 0.01713 |

⇒ **Constraining the projection to REF-A's actual adapter shape costs 7.5 % on `lead_gap` — and the
result is STILL 1.086× C104's headline number, with a partial-`v0` correlation 1.40× C104's.**

⇒ ⭐⭐ **The adapter's architecture — `AvgPool2d((4,4))` to 16 cells, then a tied
`Linear(d_model→128)` per cell — does NOT destroy the lead signal.** §4.3's upper bound is now a
tight result, and the brief's hypothesis *"if DINOv2 reads 0.45 and the adapter output reads ~0, the
ADAPTER destroys it"* is **refuted at the architecture level without any checkpoint.**
⚠️ **What remains untested is the adapter's TRAINED weights** — random here, learned there. That is
now the *only* thing the 1.9 GB download buys at this rung, which is why §5.1's ask is narrow.

### 5.3 ⛔ ESCALATED, NOT FILED IN A DOC — three registry corrections

Per `AGENT_OPERATING_STANDARD.md` these are raised as integration items, not left in a README for
someone to re-read:

1. **`MODEL_REGISTRY.md:2899` (D-A4) — "differ in exactly two things" is FALSE.** The
   reconciliation stream measured **four**; this document adds a **fifth** (input geometry) and
   measures it inert. ⛔ The registry is the only quotable source for model facts, so a false
   ablation-cleanliness claim there is load-bearing. **It has now misled at least two briefs.**
2. **`MODEL_REGISTRY.md:1925` — the deployed arm is `--adapter temporal`**, i.e.
   `refa_plus.TemporalGridAdapter`, **not** `refs/refa.py`'s `DinoAdapter`/`DinoGridAdapter`. The
   brief I was given pointed at the wrong class because the registry's architecture prose does not
   say which module the flag selects.
3. **C104's `lead_closing` row (`0.01713 vs 0.00000`) should carry its K1 status.** It **fails K1 on
   all three seeds** (`R² < 0`, `pred_sd/gt_sd ≈ 0.09`) — a degenerate near-constant readout — and it
   is quoted in at least `RETRACTION_LOG.md` C104, `POOLING_BOTTLENECK_R1R2.md` and
   `MODEL_REGISTRY.md §12.1` without it.

### 5.4 The strategic read

C104 concluded *"ENCODER experiments now outrank both"* — a frozen-external-encoder arm and a DINOv2
distillation `aux` loss. ⚠️ **REF-A *is* the frozen-external-encoder arm, it has already run to 30 k
steps, and this document shows its features were good.** It is the programme's **worst** arm.

⇒ **That does not make our v6 encoder good** — C104's 91× stands, reproduced here to five decimals.
**It makes "better features ⇒ better driving" the untested link**, and the two streams today both
land on it from opposite sides. ⭐ **The next GPU spent on this question should buy an
attribution of the four-to-five REF-A/flagship differences — starting with the 22.06 M
`ImaginationField`, which is a whole absent mechanism — not another encoder.**

---

## 6. DELIVERABLE MANIFEST

**All paths are in the repo working tree at `/c/Users/Admin/wt-tanitad-local`, branch
`agent/arch-inf-20260803`, STAGED (`git add`), NOT committed, NOT pushed.**
Repo-relative root: `TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-encoder-localisation/`

| artifact | what it is |
|---|---|
| `ENCODER_LOCALISATION.md` | this document — scope audit, pre-registration, results, verdict |
| `code/encloc_geom_cache.py` | builds DINOv2 token caches at the five input geometries |
| `code/encloc_ladder.py` | ER10 ladder + `s16` square pool arm, grid-aware PC-LOCAL, depth-corrected repo root, `--dump-preds`, `--block-diag-proj` |
| `code/encloc_summarise.py` | 2×2 table, replication gate vs C104, paired cross-cache deltas |
| `code/chain_encloc.sh` · `code/chain_encloc2.sh` | the two runnable chains |
| `raw/encloc_{wide3f,wide1f,refa3f,refa1f,squash1f}.json` | per-arm ladder results |
| `raw/encloc_pclocal_*.json` | PC-LOCAL positive controls, per arm |
| `raw/encloc_blockdiag_refa1f.json` | ⭐ E-ADAPT-0 |
| `raw/encloc_summary.json` | replication gate + table + paired deltas |
| `raw/log_*.txt` | build / ladder / control / summarise logs |

⚠️ **NOT in the repo — the token caches** (`tok_wide3f.pt` 8.29 GB, plus four more, ~19 GB total) and
`row_index.pt`, in the session scratchpad under `…/scratchpad/encloc/`. They are **regenerable** from
`code/chain_encloc.sh` + the banked `sp2` frame cache in ~9 minutes on the RTX 4060, so they are
deliberately not banked. **The frame cache they read (`sp2/cache/slotprobe-lead130-w120-256x640cyl`,
130 clips, 4.8 GB) is ALSO scratchpad-only** — ⚠️ **it is an input to C104/ER10 as well as to this
work, and if that scratchpad is cleared, neither result can be re-derived.** ⏳ **Flagged for the PI
as a banking decision; it is not mine to spend 4.8 GB of repo on.**

### Reproduction

```
bash code/chain_encloc.sh     # 2x2: builds 4 caches, 4 ladders, 4 positive controls (~9 min)
bash code/chain_encloc2.sh    # squash1f control + prediction dumps + summary  (~6 min)
```

### Evidence-class ledger

| claim | class |
|---|---|
| every r² in §3, §5.2b; the geometry facts in §0.1; the tier stamps in §0.2; the 1.906 GB in §5.1 | **MEASURED** (ours, artifact path given inline) |
| C104's published values used as the gate | **PUBLISHED-INTERNAL** (`er10_dino.json`), and **re-MEASURED here to 5 dp** |
| the reconciliation stream's E-RECON-1 null and its four-difference table (§4.2) | **INHERITED** — read from their document, **not re-verified by me** |
| REF-A's speed/scale bottleneck attribution (71–83 %) quoted in §2.5 | **INHERITED**, prior-session, not re-verified |
| §4.3's remaining-candidate table | **HYPOTHESIS** — none measured |

### Suite status

⚠️ **No package code was modified.** Everything added lives under
`TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-encoder-localisation/`; nothing
under `stack/` or `taniteval/` was touched (`encloc_ladder.py` is a **copy** of the ER10 script, not
an edit of it). ⇒ `stack/tests` and `taniteval/tests` are unaffected by this work.

---

# PART 2 — RUNGS 2–3 ON THE TRAINED CHECKPOINT (PI-authorized download, 2026-08-18)

## 7. Provenance, and the free check the download was gated on

**The PI authorized the 1.906 GB download** (relayed by the coordinator; the artifact arrived at
`C:/Users/Admin/tanitad-caches/refa-dinov2-4b/ckpt.pt`). Gate and provenance, all `MEASURED`:

- ⚠️ **The briefed gate size 1,905,666,517 B is the REPO TOTAL** (ckpt + config + README +
  .gitattributes, §5.1's table). The file itself completes at **1,905,662,297 B** — gating on the
  briefed number would have waited forever. Gated on the file's own metadata size instead; matched.
- **sha256 `04cd07dac74d1193bc8ba0e118a33c39cbb6a9277841c2a80a57cc7b3846220c`** — and this hash is
  embedded verbatim in the HF downloader's own `.incomplete` filename, i.e. it equals the server's
  declared content hash. The file is complete **and** authentic.
- Loaded **STRICT** (438 model keys) via the canonical recipe `taniteval/taniteval/loaders.py:117-134`
  (`flagship4b_config()`, `action_dim→3`, `adapter_kind="temporal"`, `n_tokens=256`). `step` 29999 —
  the registry's completion step, re-confirmed from the artifact.
- ⛔ **The checkpoint banks NO training history** — top-level keys are
  `model/opt/step/metric_invdyn/step_readout/aux_{speed,yaw,accel}` only. The `adapter_std` LOG the
  free check hoped for does not exist in the artifact (pod3's `metrics.json` remains the only
  possible archive of it). ⇒ The collapse question is answered by **direct measurement** below —
  computing the monitor's own quantity (`adapter_dim_std`, `refa.py:295-300`) on real inputs.

## 8. PRE-REGISTRATION — written after the caches were built, BEFORE any readout ran

The builder banks six latent arms over the SAME 2 809 banked rows (window `[f-7..f]`, actions
`[steer, accel, v0/10.0]` per `refa_train.py:76-112` + `refa_train_plus.py:63-105`; predictor called
**exactly as the T0 eval surface calls it** — `intent=None`, `rollout.py:182`):

| arm | what it is |
|---|---|
| `rung2_trained` | trained `TemporalGridAdapter` output state s_f |
| `rung2_rand{0,1,2}` | same class at random init — **the positive control**: it proves the probe sees signal through this exact class+dims, so a trained-arm null would be attributable to the WEIGHTS |
| `rung3_h1` / `rung3_h4` | the predictor's imagined latent ẑ_{f+1} / ẑ_{f+4} |

Declared residuals: features are the `refa1f` **emulation** (§2.2 — shared by every rung including
rung 1, so within-input contrasts cancel it); the standardizer sees slightly-off-corpus inputs;
**131 of 2 809 rows are pad-left** (repeat-earliest; delta-0 at the pad matches the adapter's own
first-frame convention). Sanity gate: fresh frame-f tokens vs the banked `tok_refa1f` cache — **corr
≥ 0.999997**, so the transform is byte-equivalent.

**Verdicts committed in advance** (primary read: `lead_gap`, partial-`v0`, vs rung 1's 0.5285 /
partial r +0.523; paired episode-cluster bootstrap on identical rows):

| | criterion | conclusion it licenses |
|---|---|---|
| **P2-PRESERVED** | `rung2_trained` retains ≥ 80 % of rung 1 AND `rung3_h1` retains ≥ 80 % of `rung2_trained` | the ENTIRE REF-A representation path carries the signal ⇒ the loss is in how the driving objective/heads consume it — "readable ≠ usable" in its strongest form |
| **P2-TRAINING-SUBTRACTED** | `rung2_trained` < `rung2_rand` with the paired CI excluding 0 | training damaged the adapter (C106's pattern at a new site) — the collapse std check must corroborate |
| **P2-PREDICTOR-LOSS** | rung 2 retains but `rung3_h1` loses ≥ 50 % of it | the predictor destroys the information — the objective/predictor is the component to fix |
| **P2-MIXED** | anything else | report per-stage decomposition, take no headline |

**Collapse check, already measured during the build (pre-readout):** trained adapter per-dim std
**≈ 0.79** vs random-init **≈ 0.22**, dead dims **0.0000** on both. ⇒ **The trained adapter did NOT
collapse** — `refa.py:19-31`'s named failure mode ("collapse-to-easy-targets") did not happen; the
trained map *expands* variance 3.6× over init. *(This kills the collapse hypothesis regardless of
which readout verdict fires.)*

## 9. RESULT — the trained checkpoint PRESERVES the signal at every stage

**`MEASURED` 2026-08-18, dev-box RTX 4060, ~2.2 min GPU for all six arms.** Artifacts:
`raw/encloc_p2_*.json`, `raw/encloc_part2_summary.json`, durable latent caches at
`C:/Users/Admin/tanitad-caches/encloc-20260818/part2/` (md5 in the manifest). **T0-DIAGNOSTIC.**
2 809 rows, 33 pad-left; sanity corr ≥ 0.9999973 (n=400).

### 9.1 The rung table — the full pipeline, end to end

r² ceiling on the identical windows (cells probe has no RP-seed axis; the bracketed spread is the
**ridge seed** — the inner alpha-selection split, seeds 0/1/2, per C103):

| stage | `ego_v0` | `lead_gap` | `lead_gap` partial-v0 r² |
|---|---|---|---|
| rung 1 — raw features (Part 1) | 0.67147 | **0.52850** | 0.24767 |
| ⭐ rung 2 — **TRAINED adapter** s_f | 0.67100 [0.671, 0.671, **0.7034**] | **0.47510** [0.4751, **0.5262**, **0.5262**] | 0.2038 |
| rung 2 control — random-init ×3 | 0.6211 / 0.6556 / 0.6579 | 0.5286 / 0.4693 / 0.5354 | 0.2303 / 0.1853 / 0.2429 |
| ⭐ rung 3 — **predictor** ẑ_{f+0.1s} | 0.67420 [0.6742, 0.6742, **0.7068**] | **0.47620** [0.4762, **0.5319**, **0.5319**] | 0.2060 |
| rung 3 — ẑ_{f+0.4s} | 0.67200 | **0.48630** | 0.2106 |

`lead_closing` (⛔ degenerate on every arm — K1 fails, `pred_sd/gt_sd` ≈ 0.10–0.17, §3.5's rule):
trained 0.0568, rand ≈ 0.037, h1 0.0555, h4 0.0425 — reported, never quotable bare.

### 9.2 The paired deltas (episode-cluster bootstrap, 2 000 draws, identical rows asserted)

| contrast | `lead_gap` Δr² [CI] | partial-v0 Δ [CI] | separated? |
|---|---|---|---|
| trained − rung 1 | −0.0848 [−0.1795, −0.0008] | −0.0694 [−0.2017, +0.0330] | raw: **grazes**; ⭐ **primary (partial): NO** |
| ẑ_h1 − trained (the predictor step) | **+0.0011 [−0.0044, +0.0065]** | +0.0022 [−0.0034, +0.0077] | **NO — the predictor loses NOTHING** |
| trained − random-init | −0.0535 [−0.1295, +0.0180] | −0.0265 [−0.1291, +0.0583] | **NO** |
| trained − rung 1, `ego_v0` | −0.0165 [−0.0556, +0.0166] | — | NO |

⚠️ **Honest reading of the one graze:** the raw-scale trained-vs-rung-1 CI upper edge is −0.0008 at
ridge seed 0 — and at ridge seeds 1/2 the trained arm reads 0.5262, erasing most of the point
deficit. The **pre-registered primary read (partial-v0) is not separated**, and the retention is
**0.899 at the worst seed, ~0.99 at the median across ridge seeds**.

### 9.3 The controls, each doing its job

- **Positive control (random-init, 3 seeds):** the same class+dims reads `lead_gap` 0.469–0.535
  through the identical probe ⇒ the probe demonstrably sees signal at these dims; a trained-arm null
  would have been attributable. There is no null to attribute.
- **Collapse check (the free check, full-run):** trained per-dim std **0.8011** vs random-init
  **0.2200–0.2254**; dead dims **0.0000** everywhere. ⛔ **`refa.py:19-31`'s named failure mode —
  "collapse-to-easy-targets" — measurably did NOT happen.** Training *expanded* per-dim variance 3.6×.
- **C115 sensitivity (is rung 3 a real probe?):** ẑ_h1 moves **|δ|/|z| = 0.328 (cos 0.942)** from
  s_f; ẑ_h4 moves **0.589 (cos 0.875)**. The predictor genuinely transforms the latent — retention
  through it is a finding, not an artifact of the residual architecture copying its input.
  *(Declared anyway: `predictor.py:101` is residual by design — ẑ = z_t + Δ — so "preserves unless
  the delta corrupts" is the mechanism; the measurement shows the delta does not corrupt, and at
  h4 — a delta of 59 % of the norm — `lead_gap` reads slightly HIGHER than at h1.)*
- ⭐ **A mechanistic footnote the aux heads explain:** the ONE place training separably beat random
  is `ego_v0` (rand0 vs rung 1: −0.0664 [−0.1296, −0.0125], separated; trained: parity, and 0.7034
  at rs2). The checkpoint carries `aux_speed`/`aux_yaw`/`aux_accel` heads — **the objective's
  explicit pressure was egomotion, and egomotion is exactly where trained > random.** The objective
  sculpted what its losses asked for and was neutral on what they did not (the lead).

## 10. VERDICT — `P2-PRESERVED`, and what it does to G-ENCODER and E-RECON-2

Against §8's pre-registered table:

| verdict | criterion | result |
|---|---|---|
| ⭐ **P2-PRESERVED** | rung 2 ≥ 80 % of rung 1 AND rung 3 ≥ 80 % of rung 2 | **FIRES: 0.899 (worst ridge seed; ~0.99 median) and 1.002** |
| P2-TRAINING-SUBTRACTED | trained < random, CI excluding 0 | does not fire (no separated deficit; random spread straddles trained; collapse check refutes collapse) |
| P2-PREDICTOR-LOSS | rung 3 loses ≥ 50 % of rung 2 | does not fire (loses 0.0 %) |

⇒ ⭐⭐ **EVERY stage of REF-A's representation pipeline is now measured, and NONE of them loses the
information.** Raw frozen features 0.5285 → trained adapter ~0.51 (ridge-seed median) → one-step
imagined latent ~0.51 → four-step imagined latent 0.486. `ego_v0` is flat at 0.67 throughout. The
signal arrives INTACT at the exact latent the T0 eval decoded from (`intent=None`, the
`rollout.py:182` surface) — **and that decode produced the programme's worst driving number
(2.1675 [1.9081, 2.4212] vs flagship 0.4271).**

**What this does to the G-ENCODER verdict (Part 1):** it survives and sharpens. Part 1 said "the
loss is downstream of the features." Part 2 eliminates the remaining representation stages —
**"downstream" now means the decode surface and the objective** (`step_readout` / grounding heads /
the driving losses), or a deficit that linear readability cannot see at all. The candidate table in
§4.3 loses its first two rows (trained adapter: EXONERATED; predictor latent: EXONERATED) and keeps
the structural differences — the absent 22.06 M `ImaginationField`, the 13.43 M vs 4.48 M grounding
gap, the SIGReg target, and the encoder's role as a *training participant* (a trained encoder
co-adapts with its decoder; a frozen one cannot — which no readout of frozen features can measure).

**What it does to E-RECON-2's priority: it becomes the decision experiment for the whole encoder
question.** Linear readability now fails to discriminate driving at FIVE stages measured across two
streams. The reconciliation stream's registered E-RECON-2 — the experiment that would make
readability evidence *about driving* — is the only remaining bridge between C104-style probes and
any deployment decision; until it runs, **no readout number anywhere in this programme is admissible
as a reason to swap, freeze, or distil an encoder.**

⚠️ **Limitations, stated:** (1) the features are the §2.2 refa1f EMULATION — the residual is shared
by every rung, so the contrasts stand, but the trained standardizer/adapter ran slightly
off-corpus; the ≈parity with random-init makes a large shift-artifact unlikely, and it is the one
asymmetry between the trained and random arms. (2) These are the lead-enriched, `parity: False`
probe windows. (3) T0-DIAGNOSTIC throughout — none of this is driving performance.

## 11. DELIVERABLE MANIFEST — PART 2 (adds to §6)

**Staged into the same package, branch `agent/arch-inf-20260803`; STAGE, NEVER PUSH.**

| artifact | what |
|---|---|
| `code/encloc_part2_latents.py` | checkpoint→latent builder (STRICT loaders.py recipe, window/action alignment, sanity gate, collapse check) |
| `code/encloc_part2_summarise.py` · `code/chain_encloc3.sh` | Part-2 summariser + runnable chain |
| `code/encloc_ladder.py` | + cells-only cache support (Part-2 patch) |
| `raw/encloc_p2_{6 arms}.json` + `_rs{1,2}` variants | ladder results |
| `raw/preds_p2_*.pkl` · `raw/encloc_part2_summary.json` · `raw/log_p2_*.txt` · `raw/log_part2_build.txt` | paired-delta inputs, summary, logs |
| **durable, outside repo:** `C:/Users/Admin/tanitad-caches/encloc-20260818/part2/cells_*.pt` (6 × ~11.5 MB) + `part2_build_meta.json`, md5-appended to `MANIFEST.md5` | the banked latents |
| **durable, outside repo:** `C:/Users/Admin/tanitad-caches/refa-dinov2-4b/ckpt.pt` — **1 905 662 297 B, sha256 `04cd07dac74d1193bc8ba0e118a33c39cbb6a9277841c2a80a57cc7b3846220c`, step 29999** | the PI-authorized checkpoint |

Evidence ledger: every §7–§10 number is `MEASURED` (artifact paths above). The E-RECON-2 reference
remains `INHERITED` from the reconciliation stream. Reproduction: `bash code/chain_encloc3.sh`
(idempotent — skips the build if the durable caches exist).
