# 2026-09-15 — measurements made while answering the PI's refcv6 questions

**Architecture & Inference · Master Mind · CPU only (dev box), no GPU, no training.**

These are the numbers behind `Project Steering/REFCV6_CLARIFICATION.md`.
- Every script is in `raw/` beside its JSON output.
- No clip id is written anywhere; outputs are aggregates only.

---

## M1 · DiffusionDrive truncation noise and step semantics (PUBLISHED-CODE, computed)

`raw/dd_noise_constants.{py,json}`. The configuration is read from the released code and the arithmetic done with `diffusers` 0.40.0 defaults (β 1e-4 → 0.02, `scaled_linear`).

| quantity | value |
|---|---|
| DD V1 waypoint noise at t = 8 (inference) | σ **0.899 m (x) / 0.727 m (y)**, i.i.d. per waypoint |
| DD V1 waypoint noise at t = 49 (training maximum) | 2.682 m / 2.168 m |
| DD V2 waypoint noise at t = 8 (normalisation x/50, y/20) | 1.579 m / 0.632 m |
| DD step 10 → 9: fraction of the gap to the noisy input kept | **0.9475** |
| ours, step 10 → 0 | **0.2828** |
| ours: control noise at t = 8 (`control_norm` 4.0 / 3.0) | 0.1264 m/s² along · 0.0948 m/s² lateral |
| ours: position noise (8 i.i.d. slots, linearised, v0 ≥ 4 m/s; ESTIMATED) | 0.5 s 0.016 / 0.012 m · 2 s **0.145 / 0.109 m** · 6 s **0.860 / 0.645 m** |

Two corrections follow from this:
- `refc_sampler.py:19-21` states "~2.3 m / ~1.7 m at 6 s". That holds only for noise constant across slots.
- The "timestep injected per layer" in `refc_sampler.py:38` is wrong: the code adds it once (`refc.py:2013-2014`).

## M2 · H3: do refcv5-v2's stride-32 tokens carry their own position? (MEASURED)

`raw/h3_token_position_probe.{py,json}`.

**Why it matters:** the anchor decoder adds no positional encoding to its 160 key/value tokens (`refc.py:2292`).

**Design** (fixed before the run):
- Tokens: cached frozen tokens of the 139 B1 eval clips, every 40th frame.
- Split: clip-disjoint by sha12 (test `% 4 == 0`, val `% 8 == 1` carved from fit).
- Probe: ridge onto one-hot targets, features z-scored on fit, λ picked on val.

| target | test accuracy | chance | controls (majority · permuted-label probe · true probe vs permuted labels) |
|---|---|---|---|
| column (20-way) | **0.8621** (±1 column 0.9528) | 0.05 | 0.0500 · 0.0492 · 0.0500 |
| row (8-way) | **0.9748** | 0.125 | 0.1250 · 0.1246 · 0.1262 |

n: 31,360 test tokens / 196 frames / 39 clips; fit 63,360 / 79 clips; d = 704.

⚠️ λ was picked at the smallest grid value, so these accuracies are a lower bound for a linear read.

⇒ **Position is linearly available from content.** Whether the decoder uses it is not tested here.

## M3 · H5: participation ratio of refcv5-v2's tokens (MEASURED)

`raw/h5_token_participation.{py,json}`. Same cached tokens, 697 frames, 111,520 tokens.

| surface | participation ratio (σ²) | top-1 energy | top-10 energy |
|---|---|---|---|
| token level | **4.059** | 0.4644 | 0.8444 |
| frame-pooled | **3.499** | 0.4538 | 0.9765 |
| control: isotropic, token shape | 699.59 (≈ d) | — | — |
| control: rank-1 | 1.0000 | — | — |

Programme reference: G-RANK frozen DINOv3 on our frames reads 8.56. ⚠️ That is a different model and dimension.

## M4 · H5b: the same measure on a random-init twin, same 200 frames (MEASURED)

`raw/h5b_random_init_twin.{py,json}`.
- Frames: 40 eval clips × 5 stacked rows, decoded by the trainer's own `_decode_stacked`, /255.
- Architecture: identical encoder (`in_channels 9`, width 88, blocks 3/6/16/6).

| encoder | tokens PR (top-1) | pooled PR (top-1) | max \|token\| |
|---|---|---|---|
| **trained** refcv5-v2 (strict load, eval) | **4.246** (0.451) | 2.876 (0.547) | 37.5 |
| random init, BN on batch statistics (**scale-matched**) | **16.134** (0.244) | 1.173 (0.923) | 42.5 |
| random init, BN at init statistics (degenerate: activations vanish) | 1.914 (0.715) | 1.040 (0.980) | 0.28 |
| control: isotropic / rank-1 | 688.77 / 1.00 | — | — |

⇒ **Training concentrated the token field ≈ 4× relative to a scale-matched random init.** The frame-pooled vector starts near rank 1 and rises to 2.9.

⚠️ A participation ratio cannot separate a collapse (e.g. LAW's un-EMA'd self-target) from task specialisation. The discriminating experiment is a tiny-rig arm with `LAW_WEIGHT` 0 or an EMA target (≈ 2 h, not run).

---

## What these change in the record

| claim | before | after |
|---|---|---|
| "REF-C's decoder is position-blind" | implied by "no positional encoding" | ⛔ withdrawn as a gap: position is linearly recoverable (M2) |
| "control-space noise ≈ DD's spread" (`refc_sampler.py` docstring) | 2.3 / 1.7 m at 6 s | 0.86 / 0.65 m at 6 s; ≈ 55–60× smaller than DD at 0.5 s (M1) |
| "LAW flattens the trunk" | hypothesis | supported, not proven (M4) |

---

## M5 · H4: does a REF-C trunk's BEV content grow with training exposure? (MEASURED 2026-09-16)

Pre-registered in `raw/h4_prereg.md` **before any H4 number was read**; run overnight on the dev box, 2.5 GPU-h.

**The one variable: the checkpoint.** refcv4b `@9,500` vs `@40,284` of the **same run** — the two `config.json`
files are **md5-identical** (`58ad809aaf241729c1695bec2ab30d8e`), both checkpoints carry 551 model keys, 396
`core.encoder.*` keys and the same 117-anchor bank, and each step is read from the checkpoint's own `step` field.
Everything else is held: the extractor (`raw/h4_extract_tokens.py`, a copy of P4a whose only additions are
`--ckpt`, `--v2ep`, `--no-pix` and a clip list recovered from the banked index), the head, 6,000 steps, the
content-blind split, fp32 inference. **Both extractions are row-identical to the banked P4 index** (asserted in
the extractor), which is what makes this panel paired with the banked controls.

| arm | AP (rule B) | CI95 | IoU 0–30 m |
|---|---|---|---|
| `const` | **0.2761858** | — | 0.2224 |
| `prior` | 0.3773 | [0.3297, 0.4267] | 0.2941 |
| `shuffled` (zero information, banked) | 0.3801 | [0.3322, 0.4286] | 0.2931 |
| `pixel` floor (banked, checkpoint-independent) | 0.3849 | [0.3317, 0.4307] | 0.2883 |
| **refcv4b @9,500** | **0.3972** | [0.3465, 0.4469] | 0.3036 |
| **refcv4b @40,284** | **0.4080** | [0.3570, 0.4566] | 0.3120 |
| refcv4b @40,284, probe seed 1 | 0.4035 | [0.3527, 0.4607] | 0.3220 |

n = 34 test clips / 6,713 rows / 7,928,485 scored cells; paired clip-cluster bootstrap, 2,000 resamples, seed 0.

**Gates (the instrument, before any reading):**
- `const` reads the test prevalence **exactly** (0.2761858034668666 both sides) ✅
- `shuffled` 0.3801 ≤ `prior` 0.3773 + 0.02 ✅

**The contrast:**

| pair | ΔAP | CI95 | separated? |
|---|---|---|---|
| **@40,284 − @9,500** | **+0.0108** | **[−0.0078, +0.0309]** | **NO** |
| probe-seed floor F = \|seed 0 − seed 1\| | 0.0045 | — | bar = 3F = **0.0135** |
| @40,284 − shuffled | +0.0279 | [+0.0008, +0.0570] | yes |
| @40,284 − prior | +0.0307 | [+0.0027, +0.0606] | yes |
| @40,284 − **pixel** | +0.0231 | [−0.0126, +0.0647] | **no** |
| @9,500 − shuffled / prior / pixel | +0.0171 / +0.0199 / +0.0123 | all cover 0 | no |

⇒ **H4 is REFUTED as written**: the interval covers 0 **and** the gap is below 3× the floor. **4.2× more training
exposure did not separably increase the trunk's BEV content.** Neither refcv4b checkpoint separates from the raw
pixel floor, which is the same verdict the frozen refcv5-v2 trunk got (AP 0.4140 vs pixels 0.3849).

**Scope, declared in the pre-registration:** two checkpoints of **one** run are n = 2 on one trajectory; exposure is
confounded with optimisation progress; a probe bounds what a *probe* recovers, not what the trunk contains.

**What it changes:** the "the trunk is data-limited" reading loses its support **within this run's range**, and the
weight moves to the two remaining hypotheses — the missing **pretrained prior** (H1) and **planning-only
supervision** (H2). Both are in `REFCV6_CLARIFICATION.md` §6.4.

⚠️ **Naming hazard in `raw/h4_panel.json`:** the panel script hard-codes the names `main` / `main_s1` for its
verdict, so in that file **`main` = refcv4b @40,284** and **`main_s1` = the same checkpoint at probe seed 1** —
*not* the banked refcv5-v2 arm. The run directories (`tok_dir`) name the token set each arm actually read.
