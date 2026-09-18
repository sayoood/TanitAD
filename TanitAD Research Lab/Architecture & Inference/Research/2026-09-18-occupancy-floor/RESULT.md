# The occupancy floor — what a no-information predictor scores against SAM3 map GT

**Date:** 2026-09-18 · **Evidence class: MEASURED (ours)** · **Zero GPU** ·
⛔ **No tier stamp and no metric family: this is a property of the CORPUS, not of any
model.** Nothing here is a capability claim about anything.

## Why this before any model read

`PREREG_S1_AGENT_SEAM_AND_COLLISION_GATE.md` §8 orders it: *"`S1-GATE-ORACLE` and
`S1-RANDOM` need **no** trained perception and run first: they bound the problem and make
everything after them readable."*

⭐ An occupancy IoU quoted **without** this floor is the *oracle gap needs a random
control* error in a new costume — **a number that looks like skill and may be prevalence.**
⭐ And it is the part that does not rot: the floor is a property of the corpus, so it is
measured once and reused by every later arm.

## Method

Every SAM3 map on the dev box, every 20th frame. Scored on **SEEN cells only** — an unseen
cell carries no evidence, the same rule `dac_from_drivable` applies, which makes every
number here an **upper** bound on what an unseen-blind predictor faces.

| | |
|---|---|
| clips read | **135** (the 4 without a map are absent here too) |
| frames sampled | **1,475** (stride 20) |
| cells | **11,328,000** |
| cells SEEN | **10,068,274** = **88.88 %** of the grid |

## The floor

| predictor | IoU vs GT | |
|---|---|---|
| **all-drivable** | **0.3412** | = `P(drivable \| seen)` — **the no-information IoU** |
| none-drivable | **0.0000** | degenerate control |
| **GT against itself** | **1.0000** | ⛔ **INSTRUMENT CHECK** — if this is not exactly 1.0 the comparison is wired wrong and nothing above is readable |

⇒ **Any occupancy head must beat IoU 0.3412 to have learned anything at all.**

## By range band — and the flatness is the finding

| band [m] | seen cells | `P(drivable \| seen)` |
|---|---|---|
| 0–15 | 2,587,607 | **0.3518** |
| 15–30 | 2,773,144 | 0.3507 |
| 30–45 | 2,524,507 | 0.3339 |
| 45–60 | 2,183,016 | **0.3250** |

⭐ **Prevalence barely moves with range — 2.7 points from the nearest band to the
farthest.** Two consequences:

* **A head cannot game the pooled metric by specialising near or far.** A per-band read
  would have been mandatory if the bands differed sharply; measured, they do not, so the
  pooled 0.3412 is honest on its own — **and that is a result, not an assumption.**
* **The far field is not mostly unseen.** 45–60 m still carries 2.18 M seen cells, 84 % of
  the nearest band's. The SAM3 maps are **dense**, so a collision gate's range is not
  limited by map coverage.

## What this does NOT establish

* ⛔ **Nothing about any model.** No checkpoint was run. The predicted occupancy's quality
  is **not** measured here and remains `S1-GATE-PRED`'s blocking dependency.
* ⚠️ **IoU on the drivable channel only.** The map carries 9 channels; a collision gate
  needs *drivable*, so that is what is bounded. Other channels are not characterised.
* ⚠️ **Sampled every 20th frame**, not every frame. With 1,475 frames over 135 clips the
  sampling error on a prevalence near 0.34 is far below the 2.7-point range spread, but
  the number is a sample and is stated as one.

## Artifacts

`raw/occ_floor.json` · `code/occ_floor.py`. Clip identifiers appear only as **sha12**.

## ⭐ The head is now MEASURED against that floor — and it starts at the DEGENERATE one

The floor above needed a companion: **where does the head actually sit?** `map_iou_drivable`
is now computed **in the same block as the map loss, from the same tensors**, so divergence
between them is structurally impossible — and that adjacency is what makes the loss the
instrument check.

### ⛔ The instrument check, run first

Re-ran the eval pass on the **same 400 windows** with the new metric in place:

| | banked (`cov2_evalB`) | this run (`cov3_evalB`) |
|---|---|---|
| `eval_map` | **2.35437** | **2.35437** |
| `eval_windows` | 400 | 400 |

⇒ the loss reproduces **exactly**, so the IoU beside it is admissible. Had it not, the IoU
would have been a number from a different forward pass wearing this one's name.

### The reading

| | |
|---|---|
| `eval_map_iou_drivable` | **0.0** |
| `eval_map_pred_drivable_frac` | **0.0** — the head calls **no** cell majority-drivable |
| `eval_map_gt_drivable_frac` | 0.35473 — consistent with the corpus floor 0.3412 |

⭐ **The untrained head starts at the DEGENERATE floor (0.0), not the prevalence floor
(0.3412).** The head is trained with a **soft CE against fraction vectors**, so
`softmax[drivable]` *is* a predicted cell-coverage fraction — the same quantity the GT
thresholds — and at init nothing reaches 0.5. **Any training must move it off exactly
zero**, and the distance to travel is to 0.3412 merely to match a constant.

### ⚠️ The weakness this exposed, and the companion it earned

A head that learned the shape but stayed **under-confident** — 0.45 on genuinely drivable
cells — would score IoU **0.0**, indistinguishable from one that learned nothing. So
`map_pred_drivable_prob_mean` is logged beside it: threshold-free, and it moves
continuously toward the GT's ~0.35.

**MEASURED after 1 step: 0.0717 (train) / 0.0728 (eval).**

⚠️ **Not 1/9 = 0.1111.** A randomly initialised head does not produce equal logits, so
uniform is the *expectation of a draw* and not the value of any particular one. Quoting
0.111 as "the init value" would have been theory standing in for a measurement — the
number above is the measurement.

⇒ **Quote the two together.** The IoU answers *"usable by a gate yet?"*; the mean answers
*"is it learning at all?"*. Neither answers the other's question.

### ⛔ Still not established

This is the **untrained** position. `S1-GATE-PRED` needs a **trained** map head, and the
ten arms that would produce one are pod work (`PREREG_REFCV6.md`: *"NO TRAINING RUN
LAUNCHED"*). What has changed is that the instrument, the floor and the starting point all
now exist, so the first trained number will be readable the moment it arrives.

<!-- OCC-FLOOR-UNTRAINED-HEAD-MEASURED-2026-09-18 -->
