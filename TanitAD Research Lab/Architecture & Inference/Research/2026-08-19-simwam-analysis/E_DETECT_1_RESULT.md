# E-DETECT-1 — a perception head reads vehicles off DINOv3 and does NOT read them off v6

`MEASURED (ours; dev-box RTX 4060, 2026-08-21)` · **T0-DIAGNOSTIC** ·
pre-registered in `PREREG_E_DETECT_1.md` **before any number** · scored on the
**same 5,617 keys, same episode-disjoint folds, same episode-cluster bootstrap**
as E-TRUNK-2 · **Thor untouched throughout.**

⭐ The PI's design: *"why are we not designing and training a
perception/prediction head, it extracts the bounding boxes of vehicles based on
the frozen trunk … we can easily supervise it based on the gt object data."*

---

## 1. ⭐⭐ The headline

**The instrument works, and it separates the two encoders cleanly.**

Nine previous nulls all shared one weakness: they read `z_t` with a scalar ridge,
so "the trunk carries nothing" could never be distinguished from "the readout
cannot express it". This experiment removes that ambiguity by including an arm
whose content is known — and that arm scores **AP 0.3673 / AUC 0.9098**.

Against that validated instrument:

* **DINOv3's token field clears the no-feature floor decisively** — 0.1884 vs
  0.1242, non-overlapping intervals.
* ⛔ **v6's token field is STATISTICALLY INDISTINGUISHABLE FROM RAW PIXELS** —
  0.0923 [0.0815, 0.1034] vs 0.0912 [0.0814, 0.1014] — at the same granularity,
  through the same head, on the same folds.

## 2. The ladder — every arm, one shared head

Target: **BEV occupancy**, 15 × 8 cells of 4 m × 4 m over 0–60 m forward and
±16 m lateral. **Base rate 5.6676 %**, 6.801 occupied cells/frame, 0.14 % empty
frames. Head: 120 cross-attention BEV queries, d 192, 2 layers, 30 epochs —
identical across arms bar the input projection width.

| arm | n_tok | d | AP | 95 % CI | AUC | loc err | vs `prior` |
|---|---|---|---|---|---|---|---|
| ⛔ **`oracle`** — perfect perception | 640 | 64 | **0.3673** | [0.3479, 0.3867] | 0.9098 | 3.76 m | instrument ceiling |
| ⭐ **`oracle_pooled`** — perfect, through v6's 4×4 | 16 | 64 | **0.2414** | [0.2208, 0.2620] | 0.8333 | 4.94 m | **what the readout PERMITS** |
| **`dino_tokens`** | 640 | 1024 | **0.1884** | [0.1702, 0.2075] | 0.7400 | 6.38 m | **ABOVE** |
| `dino_pooled` | 16 | 1024 | 0.1416 | [0.1275, 0.1550] | 0.6861 | 6.88 m | overlaps |
| ⛔ **`prior`** — closed form, NO features | – | – | 0.1242 | [0.1123, 0.1365] | 0.6965 | 5.80 m | — |
| `v6_tokens` | 640 | 768 | 0.0923 | [0.0815, 0.1034] | 0.6195 | 7.59 m | BELOW |
| ⛔ **`pixel`** — raw 16×16×3 patches | 640 | 768 | 0.0912 | [0.0814, 0.1014] | 0.6262 | 7.63 m | BELOW |
| `v6_cells` — **the deployed latent** | 16 | 128 | 0.0888 | [0.0788, 0.0990] | 0.6246 | 7.82 m | BELOW |
| ⛔ **`pixel_pooled`** — raw patches, v6's 4×4 pool | 16 | 768 | 0.0853 | [0.0767, 0.0956] | 0.6183 | 7.86 m | BELOW |
| `v6_tokens_pooled` — pool, no projection | 16 | 768 | 0.0877 | [0.0788, 0.0979] | 0.6173 | 7.81 m | BELOW |

⚠️ `oracle` is not near 1.0 **by construction**: range is withheld (depth must
come from image row), 2,063 vehicle instances project outside the 120° FOV, and
the grid quantises to 4 m. 0.3673 is this head's realistic ceiling on this task,
not a shortfall.

## 3. Three findings

### 3.1 ⛔ v6's encoder adds nothing a detector can use over its own input

`v6_tokens` and `pixel` are the same arm to within noise. `pixel` is literally
the 16×16×3 patches the encoder consumes, so this is not "v6 is worse than a
strong baseline" — it is **v6 is worse than nothing, i.e. the encoder's output
is no more decodable than its input.** Meanwhile DINOv3, on the identical
instrument, doubles the score.

### 3.2 ⭐ The 4×4 readout costs a quarter to a third of the signal — CONFIRMED

| pushed through v6's own 40× pool | before | after | cost |
|---|---|---|---|
| perfect perception (`oracle`) | 0.3673 | 0.2414 | **−34.3 %** |
| DINOv3 (`dino_tokens`) | 0.1884 | 0.1416 | **−24.8 %** |

This is the azimuth-resolution arithmetic made real: the readout gives **4
azimuth bins over a 120° FOV = 30°/bin**, against a target needing 14.5°/bin at
10 m and 3.9°/bin at 58 m. The cost is genuine and reproducible across two very
different inputs.

⚠️ **AND THE PROJECTION IS FREE.** `v6_tokens_pooled` (16 × 768, pool only)
0.0877 vs `v6_cells` (16 × 128, pool + learned 768→128) 0.0888. The lossy step
is the **pool**, not the projection.

### 3.3 ⛔ CORRECTION — the readout geometry does NOT explain v6's null

Earlier in this session I proposed the pooling ceiling as a candidate
explanation for `v6_cells`. **`oracle_pooled` refutes that.** The deployed
readout permits **0.2414**; `v6_cells` reaches **0.0888**, i.e. **37 % of what
its own readout allows**, and below the prior. The pooling is a real and
separately-worth-fixing cost, **but it is not why v6's latent fails.** The
content is not there before the pool either — `v6_tokens` at full 640-token
resolution is already at pixel level.

*(Root-cause class: a plausible mechanism quoted as an explanation before the
control that could refute it had been run. The control was already
pre-registered; the error was in the narration, not the design.)*

### 3.4 ⛔ AND THE SUCCESSOR EXPLANATION FAILED THE SAME WAY — caught by `pixel_pooled`

Having retired the geometry story, the next candidate was found in our own
source: v6's only spatial objective is `o3_masked_cell_loss` →
`MaskedCellPredictor`, which predicts *"MASKED readout-grid cells"*, `[B, C, d_r]`
with **C = 16** — the pooled cells, **never the 640 patch tokens**. That is a FACT
about the code, and it predicts something sharp: **v6 should beat raw pixels AT
CELL GRANULARITY**, where its objective actually applies pressure.

| granularity | v6 | raw pixels | delta |
|---|---|---|---|
| **cell (16)** — O3 operates here | `v6_cells` 0.0888 [0.0788, 0.0990] | `pixel_pooled` 0.0853 [0.0767, 0.0956] | **+0.0035**, CIs OVERLAP |
| **token (640)** — no pressure at all | `v6_tokens` 0.0923 [0.0815, 0.1034] | `pixel` 0.0912 [0.0814, 0.1014] | +0.0011 |

⛔ **v6 is indistinguishable from raw pixels at BOTH granularities, INCLUDING ITS
OWN.** So the defect is **not** that the pressure is applied at the wrong
resolution — the objective produces nothing decodable *even where it applies*.

⇒ This is what `PREREG_E_DENSE_1.md` was revised on **before any arm trained**:
the question is the NATURE of the pressure, not its resolution. Two candidate
explanations have now died to two cheap controls; the surviving one is being
tested rather than asserted.

## 4. The overfitting question, settled rather than left open

Five arms initially landed **below** a closed-form prior, which is only possible
if the head is adding non-generalising variation. Two measurements settle it.

**Variance decomposition** (`e_detect_variance.py`): every arm has across-frame
variance fraction **0.69–0.83** with mean-map correlation **+0.91…+0.98** against
the true prior. So the heads learn the prior's shape correctly and then lose
ground on frame-specific detail — overfitting, not collapse.

**Capacity ladder** (`e_detect_capacity.py`, 6 rungs, 646,529 → 11,609 params):

| rung | `v6_cells` AP | `dino_pooled` AP |
|---|---|---|
| d192 l2 e30 (incumbent) | 0.0888 | 0.1416 |
| d96 l2 e30 | 0.0904 | 0.1241 |
| d48 l2 e30 | 0.0945 | 0.1246 |
| d48 l1 e30 | 0.0954 | 0.1294 |
| d24 l1 e30 | 0.0971 | 0.1359 |
| d48 l1 **e10** | **0.1033** | **0.1504** |

⇒ Shrinking the head helps both arms, and **`v6_cells` never clears the prior at
any capacity** while `dino_pooled` clears it at most. **The ranking is robust to
head size, so it is not a head artefact.** Epochs matter more than width — the
10-epoch rung wins for both.

⚠️ **THIS SWEEP IS A SHAPE, NOT A RESULT.** Adopting its winning rung and
quoting that AP would be selection on the eval fold — the winner's-curse shape
`SEL-1` refuses. Any capacity change must be re-run identically on every arm
before a single number is quoted.

## 5. What this does and does not license

✅ **Licensed:** *at step 20,000/30,000, on 130 clips, v6's encoder carries no
more decodable vehicle-localisation content than the raw patches it consumes,
while frozen DINOv3 on the identical instrument carries measurably more; and
v6's 4×4 readout independently costs 25–34 % of whatever signal reaches it.*

### 5.1 ⛔⛔ THE BOUND THAT MATTERS MOST: decodability is NOT known to predict driving, and the one paired comparison we have runs the OTHER WAY

Everything above is **T0**. Before any of it is allowed to motivate an
architecture change, the programme's own registry has to be read:

| arm | ADE@2s (full-set) | source |
|---|---|---|
| `refa-dinov2` — REF-A DINOv2 4B, **frozen DINO encoder** | **2.1675** [1.9081, 2.4212] | `MODEL_REGISTRY.md` row 12 |
| flagship v1 | — | — |
| **paired delta** | **flagship > REF-A by +2.6200 m** [2.0945, 3.2570], winning **95.9 %** of 881 windows | `MODEL_REGISTRY.md` |

⇒ **The trunk that decodes the environment FAR better than v6 is also the trunk
that drove 2.62 m WORSE.** That is the same fact `C129` was retracted over, and
it is the reason this document stops at T0.

⚠️ **It is a CONFOUNDED comparison, and that cuts both ways.** REF-A differs from
the flagship in far more than its encoder — it is a frozen-encoder *plus
supervised-head* recipe, and its own failure was separately diagnosed as
speed/scale magnitude (71–83 %), not perception. So this does **not** show that
decodability *hurts* driving. What it shows is that **the link from decodability
to driving is UNTESTED in this programme, and the only evidence pointing at it
points the wrong way.**

⇒ **What E-DENSE-1 can therefore buy, stated honestly:** evidence about whether
we CAN make a latent decodable. **Whether a decodable latent drives better is a
SEPARATE, T1 question that needs its own experiment**, and no result in this
document — however clean — licenses skipping it.

⛔ **NOT licensed:**
* any **T1 / driving** claim — this is T0, and §5.1 is why that matters here
  rather than being boilerplate.
* any claim about **v6 at 336 M on 2,376 episodes**. These features are step
  **20,000 of 30,000** on **130 clips**.
* any claim that v6's *predictor* fails — **only the encoder was probed here.**
  Reading `ẑ_{t+k}` is Phase 2 and has never been measured.
* ⚠️ any claim that DINOv3 is "good enough" — 0.1884 against an oracle's 0.3673
  is roughly **half** of what this head achieves on clean evidence.

## 6. Falsifiers that actually ran

1. **`prior`** — closed form, no features. Five arms fell below it, which is what
   exposed the overfitting and forced §4.
2. **`pixel`** — raw patches. Paired delta vs `prior`: **−0.0331
   [−0.0406, −0.0263]**, i.e. raw pixels are significantly *worse* than the
   prior, in isolation the clean overfitting signature.
3. **`oracle`** — without it, §1 would have been unreadable; every arm at or
   below the floor is equally consistent with "no content" and "broken head".
4. **`oracle_pooled`** — refuted my own geometry-ceiling explanation (§3.3).
5. **`v6_tokens_pooled`** — separated pool from projection.
6. **Metric self-test** (`e_detect_box_selftest.py`) — perfect predictions score
   AP exactly 1.0000, random score 0.0086 with 95.2° yaw error (circular
   chance), scrambled scores keep recall 1.0 while AP collapses.

## 7. ⭐ THE BOX HEAD — the PI's actual design, and the "state" half of the question

`e_detect_box.py`, K = 40 slots (MEASURED to cover 100.00 % of in-grid instances;
⚠️ the feasibility figure "K=16 covers 100 %" was a 6,000-frame sample — on the
real probe rows K=16 covers 96.13 %), Hungarian matching, same folds, same
episode-cluster bootstrap. The metric is validated in
`e_detect_box_selftest.py`: perfect predictions score **AP exactly 1.0000**,
random positions **0.0086 with 95.2° yaw error** (the circular chance level), and
scrambled scores keep recall 1.000 while AP collapses — so AP prices ranking
independently of boxes.

| arm | AP@1m | AP@2m | AP@4m | recall | ctr | ext | **yaw** |
|---|---|---|---|---|---|---|---|
| **`dino_tokens`** | 0.0196 | **0.1259** | 0.3329 | 0.478 | 1.254 m | 0.309 m | 28.4° |
| `oracle` | 0.0099 | 0.0946 | 0.3286 | 0.534 | 1.259 m | **0.265 m** | **16.1°** |
| `dino_pooled` | 0.0060 | 0.0701 | 0.2501 | 0.426 | 1.296 m | 0.330 m | 33.0° |
| `oracle_pooled` | 0.0061 | 0.0674 | 0.2738 | 0.453 | 1.290 m | 0.267 m | **17.5°** |
| ⛔ **`const`** — trained constant set, THE FLOOR | 0.0079 | 0.0583 | 0.2476 | 0.406 | 1.226 m | 0.320 m | 31.1° |
| `pixel` | 0.0038 | **0.0401** | 0.1905 | 0.370 | 1.281 m | 0.324 m | 34.8° |
| `v6_tokens` | 0.0033 | **0.0401** | 0.1917 | 0.370 | 1.296 m | 0.299 m | 35.6° |
| `v6_cells` | 0.0025 | 0.0323 | 0.1589 | 0.362 | 1.304 m | 0.334 m | 34.3° |

Only `dino_tokens` and `oracle` clear the floor. `v6_tokens`, `v6_cells` and
`pixel` are all **below** it.

⭐ **`v6_tokens` AND `pixel` AGREE TO FOUR DECIMAL PLACES ON AP@2m (0.0401) AND
THREE ON RECALL (0.370).** The grid said indistinguishable; the box head, a
completely different readout with a different matcher and a different metric,
says the same thing to four digits.

### 7.1 ⛔ The STATE half of the PI's design: heading is not there

The PI asked for a head that "predicts their states". Extent and heading are
exactly what the grid could not express, and they split the arms cleanly:

* **Only the two ORACLE arms recover heading** — 16.1° and 17.5°, against a
  90° circular-chance level and the `const` floor's 31.1°.
* **`dino_tokens` carries a little** — 28.4°, modestly better than the floor.
* ⛔ **v6 carries none** — 35.6° (tokens) and 34.3° (cells), i.e. **WORSE than a
  trained constant**. Same for extent: 0.299/0.334 m against the floor's 0.320 m.

⇒ **No trunk in the programme supports a vehicle-STATE forecast today.** DINOv3
supports position and a trace of heading; v6 supports neither. That is a harder
result than the occupancy table alone, and it is the one that bears on the
"predict their states from the predicted latent" half of the design — because a
state forecast needs per-object identity, which only the slot head can carry.

## 8. Still running

* **paired deltas across all arms**, and the **prior-anchored** grid re-run.
* **E-DETECT-1T** (`e_detect_traj.py`) — is the encoder improving? v6 @11,250 vs
  @20,000 on matched rows.
* ⭐ **E-DENSE-1** (`PREREG_E_DENSE_1.md`) — the fix experiment this result
  motivated.

## 9. Manifest

| artifact | where |
|---|---|
| pre-registration | `…/simwam-analysis/PREREG_E_DETECT_1.md` |
| this result | `…/simwam-analysis/E_DETECT_1_RESULT.md` |
| target + patch banks | `…/simwam-analysis/code/e_detect_prep.py` |
| grid head + arms | `…/simwam-analysis/code/e_detect.py` |
| oracle control | `…/simwam-analysis/code/e_detect_oracle.py` |
| box head | `…/simwam-analysis/code/e_detect_box.py` |
| box metric self-test | `…/simwam-analysis/code/e_detect_box_selftest.py` |
| box result | `…/simwam-analysis/raw/e_detect_box.json` |
| variance decomposition | `…/simwam-analysis/code/e_detect_variance.py` |
| capacity ladder | `…/simwam-analysis/code/e_detect_capacity.py` |
| paired deltas | `…/simwam-analysis/code/e_detect_paired.py` |
| raw | `…/simwam-analysis/raw/e_detect.json`, `e_detect_capacity.json`, `e_detect_variance.json`, `e_detect_occ_stats.json` |
