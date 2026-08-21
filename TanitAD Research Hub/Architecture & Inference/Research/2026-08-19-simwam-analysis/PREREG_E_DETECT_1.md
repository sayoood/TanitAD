# PRE-REGISTRATION — E-DETECT-1: can a detection/forecast head read vehicles off a frozen trunk?

`PRE-REGISTERED 2026-08-21, BEFORE THE HEAD PRODUCED ANY NUMBER.`
**T0-DIAGNOSTIC.** Dev-box only — **Thor untouched.** PI's design.

---

## 1. Why this is a better instrument than every probe so far

Nine confounds are excluded and the decodability nulls stand. But **every probe
to date reads `z_t` with a scalar readout**. This design differs on two axes that
matter:

1. ⭐ **It tests the PREDICTOR.** Task B reads **`ẑ_{t+k}`** — the *predicted*
   latent — and asks whether the predicted future contains the *future* object
   states. **That is the world-model claim, and nothing in this programme has
   ever measured it.** All nine exclusions concern the encoder.
2. ⭐ **It is a STRUCTURED decoder matched to the data.** `lead_gap` collapses a
   set of objects to one scalar. A set head with spatial inductive bias is not
   the same instrument — and E-PROBE-POWER showed a *generic* expressive decoder
   (MLP) merely overfits, so inductive bias is the variable worth changing.

⚠️ **This is NOT the `aux` arm that failed.** That supervised a scalar **during
training** and asked whether it transferred (it did not). This **freezes the
trunk and decodes** — a measurement, not an intervention.

## 2. Data — MEASURED on the exact 5,617 probe rows

⚠️ **CORRECTED before any result was read.** The feasibility pass quoted *"6.32
vehicles/frame, max 20"*; that was a **6,000-frame sample**, not the 5,617 probe
rows this experiment scores. On the actual rows the numbers are below, and
`occupied cells ≤ vehicles` now holds on **every** row (it did not across the two
samples, which is what exposed the mismatch).

| | |
|---|---|
| grid | x_fwd **0–60 m**, y **±16 m** — **the v6 raster's own extent** |
| cells | **15 × 8 = 120**, each **4 m × 4 m** (≈ one car footprint) |
| vehicles/frame in grid | **mean 7.030, max 35** |
| **occupied cells/frame** | **mean 6.801, max 33** |
| **base rate** | **5.6676 %** of cells |
| frames with zero in-grid vehicles | **0.14 %** (8 / 5,617) |
| cells never occupied | **0 of 120** |
| source | `obstacle.offline` join, ego-frame cuboids, 10 classes |
| frames | 640 × 256 **cylindrical**, `f_ref` 305.58, PNG |

Vehicle classes: `automobile, heavy_truck, bus, trailer, other_vehicle`
(199,543 / 5,865 / 2,229 / 2,468 / 523 instances). `person` (77,829) and `rider`
(7,145) are **excluded** from the target — this measures *vehicles*, per the PI's
wording.

⭐ **The target is a BEV OCCUPANCY FIELD, not K slots.** Set prediction with
Hungarian matching was the first design; a dense metric grid is strictly better
here because it needs no matcher (one less failure mode), handles the 0–35 count
range natively, and gives **every arm the identical 120-d output space** — so the
640-token and 16-cell arms are compared on exactly the same task.

## 3. Arms — all trunks FROZEN, one shared head

| arm | features | what it isolates |
|---|---|---|
| `v6_tokens` | 640 × 768 | the v6 encoder field |
| `v6_cells` | 16 × 128 = 2048 | ⭐ **the DEPLOYED latent — what pooling costs on a SPATIAL task** |
| `dino_tokens` | 640 × 1024 | encoder reference at matched granularity |
| `dino_pooled` | 16 × 1024 | reference **through v6's own 40× pool** |
| ⛔ **`pixel`** | 640 × 768, **raw 16×16×3 patches** | **the floor. Non-negotiable.** |

⭐ **`pixel` replaces the planned `random_tokens`, and is a STRICTLY STRONGER
floor.** A random ViT is a random projection *of these very patches*; the patches
themselves are strictly more informative and cost no forward pass. So `pixel` is
an **upper bound on what any untrained encoder of this architecture could hand
the head** — and it tests, for free, the one hypothesis E-V6SHAPE left standing
(that low-level photometric structure suffices).

⛔⛔ **AND A SECOND, CLOSED-FORM FLOOR THAT DECIDES THE EXPERIMENT: `prior`** —
the per-fold train-mean occupancy map, no features at all. **MEASURED at
AP 0.1401 / AUC 0.708 against a 0.0567 base rate**: simply knowing where cars
*usually* are already scores **2.5× the base rate**. A head with learned queries
and token position embeddings can reach that without reading a single feature
value. **Any arm not clearly above `prior` has demonstrated nothing** — the
winner's-curse shape SEL-1 refuses, in a detector costume.

## 4. Tasks

* **A — DETECT.** From frozen features at *t*, predict the set of in-grid
  vehicles at *t*: `(present, cx, cy)` × K=16, Hungarian-matched.
* ⭐ **B — FORECAST.** From **`ẑ_{t+k}`**, predict the set at *t+k*. For v6 this
  rolls `predictor_op` from the banked step-20000 checkpoint. ⚠️ **DINOv3 has no
  predictor**, so a matched predictor must be trained on its features for the
  comparison to be fair — Task B is therefore **PHASE 2** and is not run until A
  reports.

## 5. ⛔ Committed decision rule — written before any number

Primary: **matched-position error (m)** and **F1 @ 2 m**, episode-disjoint folds,
episode-cluster bootstrap of the pooled statistic.

| observation | conclusion, committed now |
|---|---|
| `dino_tokens` clearly beats `prior`+`pixel`, **and `v6_tokens` does not** | ⭐ **the v6 encoder does not carry localisable vehicles** — the nulls generalise from scalars to structure, and the readout is not the culprit |
| **both** token arms beat `prior`+`pixel`, and **`v6_cells` collapses** | ⭐ **the POOLING is the defect for spatial content** — which the scalar probes could not have shown, and which makes the readout a live fix |
| `v6_tokens` ≈ `dino_tokens` and **both beat `prior`+`pixel`** | ⛔ **every scalar null was an INSTRUMENT artefact.** The trunk carries objects; `lead_gap`-style readouts cannot see them. Would require re-opening E-TRUNK-2's conclusions. |
| **nothing beats `prior`** | the head is drawing a spatial prior; **Task A is uninformative** and Phase 2 must not run |
| `v6_cells` ≈ `v6_tokens` | pooling is cheap for this target too — consistent with the DINOv3 pool result (−16 % only) |

⚠️ **Mixed outcomes are reported per arm with intervals.** No pooling across
arms, no "overall" score.

## 6. Falsifiers built in

1. **`prior` and `pixel`** — §3. If either matches a trained arm, that arm has
   shown nothing.
2. **Identical head, identical schedule, identical folds** across arms; only the
   input features change.
3. **A shuffled-feature control** (`<arm>_shuf`) on the best arm: permute
   features across frames within the eval fold, targets left in place. It must
   collapse to `prior`'s level, or the head is reading frame-independent
   structure.
4. **Report the empty-frame rate** (0.14 %) and the base rate (5.67 %) beside
   every AP, so no number is read without its floor.
5. **Identical head, params reported per arm**; only the input projection differs
   in width.

## 7. What this CANNOT settle

* ⛔ **Not driving.** T0. Detection quality is not planning quality.
* ⛔ **Not LeWorldModel** — the E-V6SHAPE gate is still open (Push-T never run).
* ⚠️ **Not "v6 at 336 M"** — the banked features are step **20,000 of 30,000**,
  and the probe corpus is **130 clips**, not 2,376 episodes.
* ⚠️ **A negative on Task A does not license a negative on Task B.** A trunk
  could carry objects poorly at *t* and still propagate what it has; the reverse
  is also possible. They are separate claims.

## 8. Manifest

| artifact | where |
|---|---|
| this pre-registration | `…/simwam-analysis/PREREG_E_DETECT_1.md` |
| head + training | `…/simwam-analysis/code/e_detect.py` |
| target + patch banks | `…/simwam-analysis/code/e_detect_prep.py` |
| result (to be written) | `…/simwam-analysis/E_DETECT_1_RESULT.md` + `raw/e_detect.json` |
