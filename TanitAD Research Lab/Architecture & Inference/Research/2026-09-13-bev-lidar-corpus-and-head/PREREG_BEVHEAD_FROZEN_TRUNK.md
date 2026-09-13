# PRE-REGISTRATION — `E-BEVHEAD-FROZEN-1`: does a transformer BEV head read LiDAR occupancy out of the FROZEN refcv5-v2 trunk?

**Date** 2026-09-13 (Europe/Berlin) · **Author** Architecture & Inference FlyWheel ·
**Branch** `agent/arch-inf-20260803`
**Status** written **BEFORE any head was trained and before any P4 number exists.** The only
numbers known when this was written are label statistics from P2 (marginals) and the P0/P2b
instrument measurements, all quoted below with their artifacts.

---

## 0. Why this rung, and why it is the cheap one

The PI asked for a BEV **transformer** trained on the LiDAR BEV as ground truth
(2026-09-11, verbatim in `RESULT.md`). The joint-training ladder `E-BEVLIDAR-CAP-1` is
multi-day and cannot run on the dev-box RTX 4060. This rung asks the prerequisite
question in ~1 GPU-hour: **does the trunk the planner already has carry BEV occupancy that a
transformer head can decode, with the trunk FROZEN?** Freezing removes capacity competition
by construction, so this rung cannot say anything about `E-BEVLIDAR-CAP-1`, and it does not
try to.

⚠️ Prior evidence taken seriously, not assumed: WP-A found real trunk tokens did not transfer
for AGENT LOCALISATION (test AP ≈ marginal), and the 8×20 token grid is a lossy address
space (quantisation floor, still load-bearing). Both are reasons this rung may fail; both are
why the raw-pixel floor and the shuffled arm are in the panel.

## 1. Inputs — vision only, and where each fact is read

| item | value | source |
|---|---|---|
| trunk | refcv5-v2 step 40,284, `core.encoder.*` (396 keys, STRICT load), **90,458,632** params, frozen, eval-mode BN | `C:\Users\Admin\hf-refcv5v2\ckpt_40284.pt`; `code/p4a_extract_tokens.py` |
| input | `*.v2ep.pt` PNG 256×640 cylindrical, D-015 3-frame stack, `/255` by a device-tensor divisor, fp32 | trainer path, cited in `p4a_extract_tokens.py` |
| tokens | final stride-32 map **[704, 8, 20]** of stacked row *j* | `refc.py:3161` (last window frame's `fmap`) |
| target | P2 `polar48` **[48 range × 40 az]**, 1.25 m × 3°, 60 m, HFOV 120°, raw frame *i = j + 2* | `code/p2_build_corpus.py`, `code/bev_gt_loader.py` |
| supervised/scored cells | `label_valid[i]` ∧ `cam_vis ≥ 128` ∧ **rule B observed** (`~shadow ∨ occ ∨ n_pts>0`) | `bev_gt_loader.BevClip.scored_mask` |

⛔ LiDAR is a LABEL. No LiDAR-derived quantity is an input of any arm.

## 2. Split — clip-disjoint, content-blind, fixed here

All P2 artifacts that pass every content check (quarantined clips listed in RESULT, never
silently dropped). Sort clips by `sha256(clip_id)[:12]` as a hex string; position *k*:

* **test** ⇔ `k % 4 == 0` — scored ONCE, at the end, with the checkpoint chosen on val
* **val** ⇔ `k % 8 == 1` — carved from fit; early stopping and the IoU threshold τ
* **train** ⇔ otherwise

## 3. Arms — one panel, identical scored cells

| arm | what | the question it answers |
|---|---|---|
| **`main`** | transformer head on frozen trunk tokens, seed 0 | the lever |
| **`main_s1`** | `main`, seed 1 (init + data order) | ⛔ the TRAINING-variance floor (`H-ESTIM-SEED-1`) |
| **`shuffled`** | `main` with (target, mask) pairs PERMUTED across train frames (fixed derangement) | zero image↔target information, same head, same gradient path |
| **`pixel`** | the same head on raw pixels: 9×64×160 (4×4 area mean of the same stack), unfolded 8×8 → 8×20 tokens of 576-d | a learned representation that does not beat raw input has added nothing |
| **`const`** | the train-set occupancy marginal on every cell | must read the NO-INFORMATION value exactly |
| **`prior`** | the train-set per-cell marginal | the positional floor |

**Head** (the sized variant B, `B_xattn_polar48x40_d192_L3`, made trainable): 1,920 learned
queries (N(0, 0.02)) + fixed 2-D sinusoidal PE of (range, azimuth); KV = Linear(d_in→192) +
fixed 2-D sinusoidal PE of (token row, token col); 3 × pre-norm `TransformerDecoderLayer`
(d 192, 6 heads, FF 768, dropout 0.1); LayerNorm + Linear(192→1). ⚠️ The 09-11 sizing class
had ZERO-initialised queries and NO positional encodings — trained as written, every query
is identical and every image token is unaddressed, so it could not localise anything. The
parameter count changes by the final LayerNorm (+384) and n_out 2→1 (−193); exact count is
printed by the trainer.

**Training** (all trained arms identical except the named lever): BCE-with-logits over
scored cells; AdamW lr 3e-4, wd 0.01, 500 warmup steps, cosine to 0; batch 32; **6,000
steps**; bf16 autocast; val AP every 500 steps; the best-val checkpoint is the one scored.

## 4. Metrics — test split, per range band, n and d printed

* **AP** — pooled over all scored test cells (`sklearn.metrics.average_precision_score`).
* **IoU** of the occupied class at τ chosen on val (per arm), applied to test.
* **Bands** 0–15, 15–30, 30–45, 45–60 m: AP, IoU, prevalence, n cells.
* **Secondary mask** rule C (`visible_first_hit`): AP/IoU on the camera-faithful subset.
* **Estimator** for every difference: **paired clip-cluster bootstrap** over test clips,
  2,000 resamples, seed 0; AP recomputed per resample from per-clip score histograms
  (2,001 probability bins); IoU from per-clip TP/FP/FN counts. ⛔ `overlapping_holdout_se`
  is never used. The replicate floor `F = |AP(main) − AP(main_s1)|` is measured IN THIS PANEL.

**Which variance the interval answers (CLAUDE.md):** the bootstrap answers *another draw of
test CLIPS*; `main_s1` answers *another TRAINING RUN*. The head is deterministic at inference
(no sampling), so there is no inference-variance term.

## 5. Committed criteria — fixed here, before any number

**Instrument gates (G) — if any fails, no arm is read (twin F3):**

* **G1** `const` test AP equals the test prevalence to within 1e-6.
* **G2** `shuffled` test AP ≤ `prior` test AP + 0.02 (a head with no image information can at
  best learn the positional prior).

**`E-BEVHEAD-FROZEN-1` PASSES iff ALL of:**

* **B1 information** — `main − shuffled` AP ≥ **+0.05**, bootstrap 95 % CI excludes 0, and
  ≥ **3 × F**.
* **B2 beyond position** — `main − prior` AP ≥ **+0.05**, CI excludes 0, and ≥ **3 × F**.
* **B3 beyond pixels** — `main − pixel` AP > 0, CI excludes 0, and ≥ **3 × F**.
* **B4 usefulness** — `main` test AP (all ranges) ≥ **0.60** AND `main` IoU in the **0–30 m**
  bands (pooled) ≥ **0.45**.

**FAILS** otherwise, and the failure names its next lever — committed now:

| twin | fires when | next lever (cheapest first, all 4060-feasible) |
|---|---|---|
| `F1` | B1 or B2 fails | the s32 tokens do not expose occupancy to this head ⇒ **L1: frozen stride-16 tokens** (stage-3 map, 352 × 16 × 40 = 640 tokens), same head |
| `F2` | B1, B2 pass; B3 fails | the trunk adds nothing over pixels ⇒ **L1**, then **L2: unfreeze the last ResNet stage** on cached stage-3 features |
| `F3` | G1 or G2 fails | instrument broken ⇒ fix before reading any arm |
| `F4` | B1–B3 pass; B4 fails | information present, not yet useful ⇒ **L1**, then **L2**, then an azimuth-aligned attention prior (the cylindrical column ↔ azimuth map is exact) |

⛔ Per RULE ZERO a FAIL is reported as written AND the named lever is run in the same
session if it fits the 4060; a lever that does not fit is named with what blocks it.
⛔ No bar above may be changed after a P4 number exists. A lever run after a failure gets its
own row, scored on the SAME test clips with the SAME bars, and is labelled post-hoc where it
is.

## 6. Scope — so nobody quotes this wider than it is

* **139-clip B1 EVAL slice only**, ~87 train clips. NON-PARITY corpus. A level here is not a
  level anywhere else.
* **Frozen trunk.** Says nothing about capacity competition (`E-BEVLIDAR-CAP-1`) and nothing
  about planner metrics. No T0/T1 tier applies: this is a representation probe, not a driving
  evaluation, so the four planner metric families do not apply to it and are not claimed.
* ⚠️ The refcv5-v2 trunk was trained on B1 train clips; the 141 eval clips are its held-out
  set (disjointness asserted in RESULT before quoting).
