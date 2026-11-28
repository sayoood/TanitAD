# THE INPUT GEOMETRY DECISION — horizontal FOV, input resolution, and frame shape

**Date:** 2026-07-27 (local, Europe/Berlin). **Stream:** fov-crop-audit.
**Question as briefed:** the encoder sees **51.4°** of a **120.5°** front camera. Is that still the
right trade?
**Question as RETARGETED mid-run by the PI:** *"I think we need at least 100 degree… by the way it's
high time we reviewed the 256 px resolution we chose."* ⇒ **what input geometry and resolution
should the program adopt, and what does it cost.**
**Hosts:** Parts 1 and 3 and the shape benchmark on the **dev box** (RTX 4060, 8.6 GiB). The Part 2
sweep was **finished on pod2** (A40, 46 GiB) after the pseudo-sim arm panel released it.
⛔ **pod1 was never touched — it is training.** pod3 and the eval pod were never touched.
Before launching on pod2 I verified it was idle (`nvidia-smi`: 0 MiB used, 0 % util, no compute
processes) and checked disk with a **real 500 MB `dd` write** (415 MB/s), never `df`.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED`
(another agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.

🔒 **PhysicalAI-AV is gated-confidential.** No clip UUID and no raw content appears in this folder.
Clips carry an integer index only; the UUID map stays on the dev box outside the repo.

**Estimator, everywhere:** episode/clip-cluster bootstrap (`taniteval/taniteval/ci.py`,
`episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap`, **B = 2000**).
⛔ **`overlapping_holdout_se` is not used anywhere in this document.**
Rank metrics use the **repaired** `taniteval/taniteval/rank_metrics.py` (`ties="collapse"`), never
the stable-argsort comparator that scored 1.726× chance.

---

# 0. VERDICT IN ONE BOX

> ## **The 51.4-degree crop IS costing us — but only at intersections, only above ~75 degrees, and the right fix is a WIDE frame, not a wide square. Meanwhile the premise that this breaks a comma2k19 training mix is FALSE: the deployed flagship trains on PhysicalAI alone.**

### The five things this run settles

**1. ⭐ THE LOSS IS REAL AND IT IS CONCENTRATED WHERE THE PI SAID.** At an intersection, an agent
sits in the **cropped-away** band (already in the front camera's own pixels, thrown away before the
model looks) **1.443x [1.226, 1.651]** more often than at matched baseline — and for the
*decision-relevant* cross traffic that defines an intersection, **6.192x [1.738, 13.155]**
(**8.187x [3.107, 15.975]** on the powered 450-clip secondary). **Today's crop misses 93.6 % of
decision-relevant cross-traffic samples.**
**This is evidence and not an artefact because the same instrument returns NULLS where nulls belong:**
the band the model already sees is flat (`ALL/IN_CROP` **1.049 [0.917, 1.167]**), and the
undiscriminating "anything off-front" statistic is flat too (**1.063 [0.961, 1.148]**,
independently reproducing the sibling stream's 1.009).

**2. ⭐ A WIDER CROP BUYS ALMOST NOTHING BELOW ~75 DEG, AND A LOT ABOVE IT.** Of the
decision-relevant content missed today, widening to **70 deg** recovers **3.4 %**; widening to
**100 deg** recovers **31.3 %** — **9.2x more**. Decision-relevant content captured goes
**6.4 % -> 35.7 %** at 100 deg. **A cautious 60-70 deg compromise pays the full resolution price for
essentially none of the benefit. The PI's ">= 100 deg" is the right side of the knee.**

**3. ⭐ AND THE HEADLINE THE COST TABLE FORCES: A SQUARE 100-DEGREE FRAME IS THE WRONG SHAPE.**
MEASURED on the 4060, spill-filtered: `640x640 @100 deg` costs **9.37x** inference / **10.39x**
training. **`256x640 @100 deg` costs 2.78x / 3.06x for the SAME field and the SAME on-axis angular
resolution (4.686 vs today's 4.643 px/deg) — 3.4x cheaper.** And the sensor is only 1080 rows tall,
so a *square* 100-deg crop is **32 % replicate-padded invented pixels**, while the letterbox is
**0.06 %**. Taking 100 deg at 256x256 instead is free in compute but costs **2.48x** angular
resolution (4.643 -> 1.874 px/deg) *and* the same 32 % padding. **Recommendation: `256 x 640`.**

**4. 🔴 THE CROSS-CORPUS PRICE IS MUCH SMALLER THAN THE BRIEF ASSUMES, BECAUSE A LOAD-BEARING
PREMISE IS FALSE.** The deployed v1 (`flagship4b-speedjerk-30k`) trains on **PhysicalAI-AV alone,
100 %** — MEASURED three ways, including the run's own committed config JSON and the fact that
`--data cached` discards every cache dir after the first (`train_flagship4b.py:186-188`).
The "comma2k19 + PhysicalAI 0.40/0.60 mix" is a **stale docstring belonging to a different model**
(`p0-sB01-realmix`, base250cam). **There is no comma training mixture to break.** comma2k19's own
sensor ceiling is **65.2 deg** — it can never supply 100 deg — but it is an **OOD eval** corpus, not
a training corpus.

**5. ⛔ AND THE RESULT THAT CUTS AGAINST THE WIDENING CASE, REPORTED AT EQUAL PROMINENCE.**
For **lane changes** — powered at 44 clusters — peripheral content before the manoeuvre is
**significantly LOWER** than baseline (**0.759 [0.528, 0.997]**, separated *below* 1). Drivers change
lane when the surroundings are emptier. **Widening is an INTERSECTION lever on this evidence, not a
general one.** The pre-registered "no differential loss" outcome was reachable, and for one of the
PI's three situations something stronger than it actually occurred.

### ⛔ What this run does NOT answer

**The model-side sweep returned `REFUSED`.** 53 of 500 clips were extracted before the wall-clock
ran out (~31.5 s/clip on a contended GPU), giving **20** held-out intersection clusters against a
pre-registered bar of **40**, so **no verdict was emitted and no per-arm AP is quoted** (section
6.4). **`REFUSED` is the power guard firing, not a null about FOV.** Everything in points 1-5 above
is independent of it: Part 1 fits nothing and the cost table is a timing measurement. **The harness
is complete, staged and resumable — escalation 6 asks for ~1-2 GPU-hours on an idle pod to finish
it.**

### And the thing that must be fixed before anyone re-caches

🔴 **The episode-cache key is PROVABLY BLIND to crop geometry.** MEASURED: `F_REF = 266` and
`F_REF = 133` produce the **identical** key `eafe5e4eb363`. A re-crop that keeps `size=256` and
forgets to hand-edit a string tag writes a directory named `physicalai-train-e438721ae894`
containing **different pixels**, and passes every guard — the parity content check hashes only
filenames. **A non-square frame cannot be expressed in the key's `params` at all.** Episode-selection
parity does survive a re-crop (it is a re-cache, not a re-selection) — **but only if this is fixed
first.**

### Scoring against the bars set BEFORE the numbers existed

| pre-registered question | bar | result |
|---|---|---|
| Is the cropped-away band differentially loaded in the PI's situations? | CI must exclude 1.0 | **intersection YES** (1.443 / 6.192) · **lane change YES, in the OPPOSITE direction** (0.759) |
| Can the Part 1 rule return a failing value? | `NO DIFFERENTIAL LOSS` must be reachable | **YES — and something stronger fired on lane change** |
| Can the power guard refuse? | < 40 clusters -> no verdict | **YES — fired on lane change (36) and roundabout (6) held-out** |
| Does the instrument return nulls where nulls belong? | `IN_CROP` and `ALL/OFF_FRONT` must be flat | **YES — 1.049 [0.917, 1.167] and 1.063 [0.961, 1.148]** |
| C-FID (fidelity) | rebuilt baseline == the repo's own crop | **crop box exact on 100 % of clips; 0.0 % of pixels off by >1** |
| Shim fidelity | reshaped trunk == deployed trunk at 256x256 | **bit-identical, `max_abs_diff = 0.0`** |
| Spill filter | any capacity claim must survive it | **3 rows REFUSED and re-measured**; all quoted training numbers are from the flat regime |

*Estimator everywhere: paired episode/clip-cluster bootstrap, B = 2000 (`taniteval/ci.py`).*
*⛔ `overlapping_holdout_se` appears nowhere. Rank metrics use the repaired `rank_metrics.py`.*

---

# 1. PRE-REGISTRATION — written before any number in §3–§6 existed

*(Staged first, then measured. The outcome rules below are executed in code by
`scripts/fov_verdict.py`; the verdict in §0 is that function's return value, not a sentence written
after looking at the tables.)*

## 1.1 What is already settled and is NOT re-derived here

| fact | value | class |
|---|---|---|
| canonical effective focal | `F_REF = 266` at 256 px | MEASURED, `stack/tanitad/data/calib.py:38` |
| retained half-angle | `atan(128/266) = 25.697°` → **51.39°** full | MEASURED, `calib.py:89-96` |
| `camera_front_wide_120fov` native | **120.5°**, spans −60.3°…+60.3° | INHERITED, H2 §A.1/A.2 |
| discarded before the model looks | **57 %** of the front camera's horizontal field | INHERITED, H2 §A.2 |
| pooled split of agent-frames outside the crop | **12.65 %** cropped-away · **11.69 %** genuinely off-front | INHERITED, H2 §A.2 |
| rig split | exactly the three 120° f-theta cameras, the **same 29.1 %** of clips; cy bimodal ≈534 / ≈751; geometric-centre crop is **~215 px** wrong for rig B | INHERITED, H2 §A.3 |
| situation labels, held-out clusters | **153** lane-change · **264** intersection · 26 roundabout | INHERITED, `2026-07-26-situation-classifier` §5 |
| camera-only arm above chance | **2.17×** (lane change) · **2.60×** (intersection) | INHERITED, same |

Every projection in this document uses **per-clip `(cx, cy)` and per-clip 6-DoF extrinsics**
(`crux.clip_rig`), so the rig-B ~215 px error cannot enter. That is the confound the brief names,
and it is controlled by construction, not by assumption.

## 1.2 Part 1 — the per-situation loss (no training required)

**Statistic.** Every `obstacle.offline` 3-D agent sample is projected into the front-wide camera with
the clip's own intrinsics and assigned to exactly one **band**:

| band | definition |
|---|---|
| `IN_CROP` | inside the 256×256 canonical crop actually fed to the encoder (square of half-side `r(25.697°)` about `(cx, cy)`) |
| `CROPPED_AWAY` | inside the native front-wide frame **but outside that crop** — already in the front camera's own pixels |
| `OFF_FRONT` | outside the native front-wide frame — only a cross/rear camera can see it |

Reported **per situation**, on the **held-out** side of the situation classifier's own chunk split,
against the **matched not-in-situation baseline on the same clips** (a band that always contains
something proves nothing — the sibling stream's `any_off_front` 1.009 [0.970, 1.045] is the worked
example). Populations, nested and all three reported:

- **P-ALL** — every agent sample.
- **P-NEAR** — range ≤ 40 m (the 3 s anticipation horizon's reach).
- **P-CROSS** — agents that satisfy the intersection discriminator itself (moving ≥ 2 m/s,
  perpendicular, constant-velocity path crossing the ego's realised path within 40 m). **This is the
  decision-relevant population**; the other two are context.

**Secondary — the recovery curve.** For a grid of candidate half-angles the fraction of currently
missed decision-relevant agent-frames that a *wider crop alone* would recover. This is the geometric
ceiling on Part 2 and it needs no classifier.

**FAILING VALUE, stated before measurement.** Part 1 returns `NO DIFFERENTIAL LOSS` if, for **both**
powered situations, the in-situation vs matched-baseline **lift of `CROPPED_AWAY` presence has a CI
containing 1.0**. That outcome is reachable: the sibling stream's `any_off_front` lift for
intersection is 1.009 [0.970, 1.045] — an already-measured null of exactly this shape on a
neighbouring statistic. If Part 1 returns `NO DIFFERENTIAL LOSS`, the case for a wider crop rests on
the pooled 12.65 % alone and is **not** situation-driven.

## 1.3 Part 2 — does a wider crop actually help? (the FOV sweep)

**Probe.** The situation classifier's image-only arm, re-fit per FOV arm. Frozen deployed-v1
encoder + readout (`v1_trunk.pt`, STRICT-loaded), PCA **rank 16** (the measured dose-response
optimum; raw-2048 concat is 17–25 % worse), closed-form ridge. Identical clips, identical frames,
identical labels, identical split in every arm — **only the crop geometry changes.**

**Arms.**

| arm | half-angle | what it isolates |
|---|---|---|
| `A_51` | 25.697° (**v1's actual input**) | baseline |
| `B_70` | 35° | FOV + its own resolution/padding cost |
| `C_90` | 45° | " |
| `D_120` | 60.25° (native) | " |
| `R_blur@X` | 25.697°, resampled to arm X's angular resolution and back | **the resolution confound, alone** |
| `P_pad@X` | 25.697°, with arm X's replicate-padded fraction imposed | **the padding confound, alone** |
| `M_match@X` | 25.697° with **both** X's blur and X's padding | the matched-degradation control |

⚠️ **Two confounds are controlled, not assumed.** At fixed 256 px a wider FOV *must* lose angular
resolution, and — because the sensor is 1080 tall and the crop is square about `(cx, cy)` — a wide
crop also *must* replicate-pad rows the sensor never captured. Both are costs of the wide arm, so
the honest contrast is **wide vs. matched-degradation**, not wide vs. pristine baseline.

**Primary contrast.** `AP(X) − AP(M_match@X)`, paired clip-cluster bootstrap, B = 2000, per
situation. **Secondary.** `AP(X) − AP(A_51)` — whether wider pays *after* paying its own costs.

**Outcome rules, fixed now:**

- **CONFIRM** — some wide arm beats `M_match@X` with a separated CI **and** its point estimate also
  beats `A_51`, on at least one powered situation. ⇒ the 51.4° trade is costing us; recommend a
  concrete new `F_REF` (or per-corpus geometry) with its cross-corpus price stated.
- **REFUTE** — no wide arm's CI vs `M_match@X` excludes 0 upward on either powered situation, and
  every `AP(X) − AP(A_51)` point estimate is ≤ 0. ⇒ **51.4° stands and the periphery genuinely
  needs a second camera.** This strengthens the H2 sensor-activation case.
- **PARTIAL** — wide beats `M_match@X` (the information *is* out there) but does **not** beat
  `A_51` (its resolution/padding cost eats the gain). ⇒ the recommendation is **not** a wider
  `F_REF` at 256 px; it is a **wider canvas at constant `f_eff`** (more pixels, same angular
  resolution) or a **second front tap**, and the price is compute, not geometry. I would then
  quantify that compute price and hand the choice to the PI rather than change `F_REF`.

**⚠️ The rule can return a FAILING value, and it can refuse.** Two guards, both able to fire:

- **`REFUSED` (vacuity guard).** If `A_51` does not clear chance with a separated CI on **either**
  powered situation in *this* reduced universe, or a situation has < 40 held-out positive clusters,
  **no verdict is emitted for it**. A sweep whose baseline cannot see the target cannot rank crops,
  and the honest output is a refusal, not a REFUTE. *(This is the program's most recent fix
  pattern — refuse to emit a vacuous diagnostic.)*
- **`INSTRUMENT-BLIND`.** A known-worse arm must be *detected* as worse. `R_blur@D` (the 51.4° crop
  degraded to the native-FOV arm's angular resolution — a ~3.6× resolution loss with **zero**
  information added) must score below `A_51` with a separated CI. If a degradation that large is
  not separated, the sweep has no power and returns `INSTRUMENT-BLIND` instead of REFUTE.

**Bi-directional validation** (both required, both able to fail):

- **C-FID (fidelity).** `A_51` rebuilt here from the raw mp4 must reproduce the cached episode's
  crop it is supposed to equal. Measured as pixel agreement against the local episode cache.
- **C-NEG (deliberately failing input).** Column-shuffled features must land at chance
  (`AP/base ≈ 1`, CI containing the base rate). If C-NEG separates, the harness is leaking and no
  arm is quotable.

## 1.4 Part 3 — the cross-corpus price

`F_REF = 266` exists so comma2k19 and PhysicalAI share one action→pixel geometry. A wider
PhysicalAI crop breaks that. To be quantified, not asserted: which corpora/arms become
incomparable, what a **per-corpus** or **rig-conditioned** geometry would cost, and whether the
non-fisheye path (`pinhole_rectify`, `f_eff == F_REF` by construction) changes the answer.

## 1.5 Declared limitations, before the numbers

1. ⛔ **The dev box does NOT hold the parity episode cache.** Its local cache is keyed
   `14231cd29c74`, not `e438721ae894`. **Part 2 therefore rebuilds every arm from the raw mp4s and
   is a self-contained paired experiment.** Its absolute APs are **not** comparable to the situation
   classifier's pod3 numbers and are never quoted as such. What *is* valid is the within-sweep
   contrast, because every arm sees the same clips, frames, labels and split.
2. ⛔ **Part 2's universe is the 500 locally-decodable R0 clips**, not 2,376. Power is stated per
   situation and the vacuity guard above is what protects the conclusion.
3. ⚠️ `obstacle.offline` is `scene:obstacles:autolabels:v2` — machine labels. Systematic misses of
   small/distant agents attenuate **every** band statistic in Part 1, in the direction of
   understating the periphery's content.
4. ⚠️ Windows/WDDM spills to host RAM instead of OOMing, so any capacity claim here carries a spill
   filter (`torch.cuda.max_memory_allocated` **and** wall-clock per batch; a >2× slowdown at
   constant work is reported as a spill, not a capacity).

---

## 1.6 ⚠️ SCOPE AMENDMENT — the retarget, logged before the numbers it affects

The PI's direction arrived **after** §1.1–§1.5 were staged and after Part 1 had already run. What
changed and what did not:

| | |
|---|---|
| **UNCHANGED** | every rule in §1.2–§1.3, the estimator, the outcome names, both guards, and the requirement that a wider FOV measuring WORSE is reported as cleanly as a win |
| **ADDED** | a **resolution axis** and an **aspect-ratio axis** to the sweep, and the compute price of each shape as a first-class MEASURED deliverable |
| **ADDED rule** | the recommendation must now name an **input shape**, not only an `F_REF` |

⚠️ **A hypothesis from the PI is still a hypothesis.** He has stated that a wider field *"will
improve performance… because they are considering more parts of the world."* This document tests
that. **A wide arm that measures worse is an admissible, publishable outcome and is reported with
the same prominence as a win.**

**Extra failing values the added axes make reachable, stated before measurement:**

- **Resolution.** If `A_51_384` (1.5x the pixels, *identical* field) does not beat `A_51_256`,
  then "more pixels help" is NOT supported at today's field, and the resolution ask has no
  evidence behind it on this probe. ⚠️ Note the direction of the handicap (§1.7): this arm is
  evaluated under a train/test shape shift that can only hurt it, so a null here is **weak**
  evidence and must be reported as weak, not as a refutation.
- **Aspect.** If `F100_640x256` (the letterbox) is **worse** than `F100_640sq` with a separated
  CI, the square's extra vertical field is carrying real signal and the 2.5x token saving is not
  free. That is a fully reachable outcome and it would kill the headline recommendation.

## 1.7 ⚠️ The one-directional handicap on every non-256 arm — declared before the numbers

The probe is the **frozen deployed-v1** encoder + readout. Its weights were trained at
**256x256 / 51.4°**. Evaluating them at 384², 640² or 640x256 requires resampling the learned
positional embedding (the standard ViT resolution-transfer recipe, `scripts/shape_shim.py`), which
imposes a train/test shape shift **on the new shapes only**.

**The bias runs one way: against the higher-resolution and non-square arms.** Therefore

- a higher-resolution or letterbox arm that **WINS** under this handicap is **strong** evidence;
- one that merely **ties or loses** is **weak** evidence and may be reading the handicap rather
  than the geometry. Any shape actually adopted would be **retrained**, not shimmed.

This is stated here so it cannot be produced afterwards to explain away an inconvenient result.

---

# 2. PART 1 — the per-situation loss, MEASURED

**Class: MEASURED (ours).** Tier: **CONFIRMED** for intersection; **no verdict** for lane change and
roundabout (below the pre-registered 40-cluster bar).
Artifacts: `artifacts/bands_summary.json`, `artifacts/band_stats_heldout.json`,
`artifacts/band_stats_all.json`. Code: `scripts/fov_bands.py`, `scripts/fov_band_stats.py`.

**Universe.** 480 parity episodes whose chunk has `obstacle.offline` locally **and** whose clip has
per-clip calibration; **450 admitted** by the C-ALIGN floor. **2,650,584 agent samples.**
Held-out side: 341 clips. Every projection uses the clip's own `(cx, cy)` + 6-DoF extrinsics, so the
rig-B ~215 px error cannot enter.

**C-ALIGN reproduces the sibling stream exactly, independently.** Median position residual
**0.0018 m**, p90 **0.02845 m**, **450/480** admitted — the same three numbers
`2026-07-26-situation-classifier` §4 reports from a separate implementation. That is an independent
reproduction of the obstacle-clock join, not a citation of it.

## 2.1 Power — the guard fires, as pre-registered

| situation | held-out anticipation frames | positive clip clusters | base rate | **C-POW** |
|---|---:|---:|---:|---|
| **intersection** | 1,271 | **44** | 0.02277 | OK |
| lane change | 1,016 | **36** | 0.01823 | **UNDERPOWERED — no verdict** |
| roundabout | 166 | **6** | 0.00290 | **UNDERPOWERED — no verdict** |

*The 40-cluster bar is the situation classifier's own, adopted here before measurement. Lane change
misses it by 4 clusters and therefore gets **no verdict**, only descriptive numbers. This is the
vacuity guard doing its job, not a result quietly downgraded after the fact.*

## 2.2 The headline: at intersections the loss is real, differential, and concentrated

Presence lift = P(at least one agent of that population in that band | anticipation frame) divided
by the matched **not**-in-situation baseline **on the same clips**. Paired clip-cluster bootstrap,
B = 2000.

| population / band | in situation | matched baseline | **lift [CI95]** | separated? |
|---|---:|---:|---|---|
| **CROSS / OFF_FRONT** | 0.0598 | 0.0059 | **10.127x [1.962, 24.561]** | **YES** |
| **CROSS / CROPPED_AWAY** | 0.0559 | 0.0090 | **6.192x [1.738, 13.155]** | **YES** |
| CROSS / IN_CROP | 0.0079 | 0.0026 | 3.065x [0.000, 8.535] | no |
| **ALL / CROPPED_AWAY** | 0.7899 | 0.5474 | **1.443x [1.226, 1.651]** | **YES** |
| **NEAR / CROPPED_AWAY** | 0.6349 | 0.4500 | **1.411x [1.086, 1.699]** | **YES** |
| NEAR / OFF_FRONT | 0.8458 | 0.7167 | 1.180x [1.042, 1.317] | yes |
| NEAR / IN_CROP | 0.5822 | 0.5285 | 1.102x [0.867, 1.332] | **no** |
| **ALL / OFF_FRONT** | 0.9174 | 0.8633 | **1.063x [0.961, 1.148]** | **no** |
| **ALL / IN_CROP** | 0.8301 | 0.7915 | **1.049x [0.917, 1.167]** | **no** |

**Read this from the bottom up — that is what makes it evidence rather than an artefact.**

1. **The band the model already sees carries NO differential signal.** `ALL/IN_CROP` 1.049
   [0.917, 1.167] and `NEAR/IN_CROP` 1.102 [0.867, 1.332], neither separated. *Whatever
   distinguishes an approaching intersection, it is not in the current crop's agent content.*
2. **`ALL/OFF_FRONT` is also a null** — 1.063 [0.961, 1.148], independently reproducing the sibling
   stream's 1.009 [0.970, 1.045]. **An undiscriminating "more is visible off-front" statistic is
   information-free, exactly as warned, and this instrument returns that null.**
3. **Against those nulls, `CROPPED_AWAY` separates at 1.443x for all agents and 6.192x for
   decision-relevant cross traffic.** The instrument is not reporting "bigger band, more stuff": it
   distinguishes bands, and the band that lights up is the one already inside the front camera's
   own pixels.

**Lane change is the honest counter-case, reported at equal prominence.** It is UNDERPOWERED
(36 clusters) and every point estimate runs the *other* way: `NEAR/CROPPED_AWAY` 0.628
[0.373, 0.873] and `NEAR/IN_CROP` 0.633 [0.383, 0.886], both separated **below 1**. Read
descriptively, **drivers change lane when the surroundings are emptier**, so a lane change is
preceded by *less* agent content everywhere. **The expectation that the peripheral loss is
concentrated in the PI's lateral situations is CONFIRMED for intersections and NOT confirmed for
lane changes.** No verdict is issued for lane change in either direction.

## 2.3 The recovery curve — the sharpest engineering finding in Part 1

Decision-relevant (P-CROSS) agent samples on intersection-anticipation frames, held-out.
**Today's crop misses 93.6 % of them** (only 6.37 % are `IN_CROP`). Of what is missed, **48.3 % is
recoverable by a wider crop alone**; the rest is genuinely off the front sensor.

| full HFOV | `f_eff` | recovered, of what is missed today | **decision-relevant content captured** |
|---:|---:|---:|---:|
| **51.4 deg (today)** | 266.0 | — | **6.4 %** |
| 56.0 | 240.7 | 0.7 % | 7.0 % |
| 60.0 | 221.7 | 1.4 % | 7.6 % |
| 65.0 | 200.9 | 2.0 % | 8.3 % |
| 70.0 | 182.8 | 3.4 % | 9.6 % |
| 75.0 | 166.8 | 8.2 % | 14.0 % |
| 80.0 | 152.5 | 10.9 % | 16.6 % |
| 90.0 | 128.0 | 18.4 % | 23.6 % |
| **100.0** | **107.4** | **31.3 %** | **35.7 %** |
| 110.0 | 89.6 | 38.1 % | 42.0 % |
| **120.5 (whole sensor)** | 73.2 | **47.6 %** | **51.0 %** |

> ### A wider crop buys essentially NOTHING below ~75 deg, and a lot above it.
> 51.4 -> 70 deg recovers **3.4 %** of what is missed. 51.4 -> 100 deg recovers **31.3 %** — a
> **9.2x larger** return. **A cautious intermediate widening (60, 65, 70 deg) is close to
> worthless**: it pays the full angular-resolution price and recovers almost none of the content.
> The decision is genuinely binary — stay at 51.4 deg, or go wide enough to matter.
> **On this axis the PI's ">= 100 deg" instinct is supported by the geometry: 100 deg captures
> 5.6x the decision-relevant content that 51.4 deg does (35.7 % vs 6.4 %).**

**Interval honesty.** The P-CROSS anticipation stratum holds **157 agent samples**. The
*content-share* CIs are consequently wide (`CROPPED_AWAY` 0.452 [0.208, 0.734]). The **decision**
statistic — presence lift 6.192 [1.738, 13.155] — is separated, and the recovery curve is a
deterministic function of stored per-sample geometry with no estimator at all. **Tier: CONFIRMED,
not DECISION-GRADE**, on n.

## 2.4 A correction to the number that motivated this brief

The brief's motivating figure is INHERITED from H2 section A.2: *"of agent-frames outside the model
crop, **12.65 %** are cropped-away while **11.69 %** are genuinely beyond it, so roughly half of
what an extra camera pass would buy is recoverable by a wider crop."*

**MEASURED here over all 2,650,584 agent samples on all 450 admitted clips:**

| band | this stream | H2 section A.2 |
|---|---:|---:|
| `IN_CROP` | **27.03 %** | (implied ~75.7 %) |
| `CROPPED_AWAY` | **12.33 %** | 12.65 % |
| `OFF_FRONT` | **60.64 %** | 11.69 % |

**The cropped-away figure reproduces (12.33 % vs 12.65 %, independent pipeline and clip sample).
The off-front figure does not — it is 5.2x larger here.** The two H2 numbers therefore cannot share
this denominator; theirs must be over a front-biased population. **Consequence for the brief's
claim: over the full agent population a wider crop recovers about ONE SIXTH of what is missed
(16.9 % cropped-away / 83.1 % off-front), not one half.**

**But "roughly half" survives exactly where it matters:** on the *decision-relevant* population at
intersections the crop-recoverable share is **48.3 %** (2.3). The efficiency claim is right for the
case that motivates it and wrong as a global statement. I am **not** asserting H2 is wrong — I could
not re-run their denominator — I am reporting that the two figures are not comparable as quoted, and
escalating it (section 10).

---

# 3. THE PRICE OF EACH INPUT SHAPE — MEASURED, spill-filtered

**Class: MEASURED (ours, dev box RTX 4060 8.6 GiB).** Tier: **DECISION-GRADE** for the relative
costs; the absolute ms are a 4060 number and are **not** an A40/Orin number.
Artifacts: `artifacts/shape_bench.json`, `artifacts/shape_bench_train_smallbatch.json`.
Code: `scripts/fov_shape_bench.py`, `scripts/shape_shim.py`.

**The circulating cost table was ESTIMATED. This replaces it with measurement.**

`px/deg` below is the **on-axis** angular resolution `f_eff * pi/180` — the true resolution at the
optical axis, not `width/HFOV` (which overstates it on a rectilinear canvas).

| shape | HFOV | true VFOV | `f_eff` | **px/deg** | tokens | infer ms/frame | **x today** | train ms/frame | **x today** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **256x256 @ 51.4 (TODAY)** | 51.4 | 51.4 | 266.0 | **4.643** | 256 | **7.21** | **1.00x** | **21.80** | **1.00x** |
| 256x256 @ 100 | 100.0 | 100.0 | 107.4 | **1.874** | 256 | 7.19 | **1.00x** | 21.80 | **1.00x** |
| 320x320 @ 100 | 100.0 | 100.0 | 134.3 | 2.343 | 400 | 11.66 | 1.62x | — | — |
| 384x384 @ 100 | 100.0 | 100.0 | 161.1 | 2.812 | 576 | 17.23 | 2.39x | 58.71 | 2.69x |
| 512x512 @ 100 | 100.0 | 100.0 | 214.8 | 3.749 | 1024 | 35.74 | 4.96x | 121.68 | 5.58x |
| 640x640 @ 100 | 100.0 | 100.0 | 268.5 | 4.686 | 1600 | 67.58 | **9.37x** | 226.53 | **10.39x** |
| **256x640 @ 100 (letterbox)** | **100.0** | **39.35** | 268.5 | **4.686** | **640** | **20.05** | **2.78x** | **66.76** | **3.06x** |
| *320x640 @ 100 (vertical-preserving)* | *100.0* | ***49.25*** | *268.5* | *4.686* | *800* | *~26.6 (EST)* | *~3.7x (EST)* | *~85 (EST)* | *~3.9x (EST)* |
| 256x512 @ 80 | 80.0 | **39.6** | 305.1 | 5.325 | 512 | 15.47 | 2.15x | — | — |
| 256x384 @ 60 | 60.0 | **39.9** | 332.6 | 5.804 | 384 | 11.22 | 1.56x | — | — |

**WDDM spill filter, applied and NOT waived.** At batch 16 the 384x384, 640x640 and 256x640 training
rows reported `peak_reserved` **7.32 / 10.03 / 8.08 GiB** on an **8.6 GiB** card and ms/frame jumped
**2.2x / 7.4x / 3.4x** for constant per-frame work. Windows does not OOM — it spills to host RAM —
so those rows were **refused as capacity claims** and re-measured at batch 1/2/4, where ms/frame is
flat (e.g. 640x640: 222.5 / 224.1 / 226.5 ms). **Every training number in the table above is from
the flat, non-spilled regime.** Peak reserved at batch 4: 256x256 **1.14**, 384x384 **2.09**,
512x512 **3.12**, 640x640 **4.68**, **256x640 2.20 GiB**.


⚠️ **These numbers were measured on an OTHERWISE-IDLE GPU, and that matters.** A re-run attempted
later while this stream's own feature extraction was using the card returned **294.9 ms/frame for
256x256** — 41x the idle figure. **Any shape benchmark taken under contention is invalid and none
appears here.** The table above is the idle-GPU measurement.

The italic `320x640` row is **ESTIMATED, not measured** — linear interpolation in token count
between the measured 640-token (20.05 ms) and 1024-token (35.74 ms) points. It is included because
it is the **vertical-preserving** variant: at 800 tokens it keeps a true VFOV of **49.25 deg**,
i.e. essentially today's 51.4 deg, which is the one real cost of the 256x640 letterbox.

## 3.1 Three results that decide the shape question

**(a) 100 degrees at today's input shape is FREE in compute — and costs 2.48x angular resolution.**
256x256 @ 100 deg is 7.19 ms vs 7.21 ms: identical, 256 tokens either way. The entire price of the
PI's ask *at fixed shape* is paid in resolution: **4.643 -> 1.874 px/deg**. That is the trade
Part 2 measures.

**(b) The letterbox is 3.4x cheaper than the square for the SAME field and the SAME resolution.**
256x640 and 640x640 have identical HFOV (100 deg), identical `f_eff` (268.5) and identical on-axis
resolution (4.686 px/deg). The square costs **9.37x** today; the letterbox costs **2.78x** —
**3.37x cheaper at inference, 3.39x cheaper in training**. The 100 degrees of *vertical* field the
square buys is what the letterbox declines.

**(c) The square 100-degree crop is ONE THIRD INVENTED PIXELS. The letterbox is ~zero.**
Replicate-padded row fraction, MEASURED per clip (`artifacts/geom.parquet`):

| arm | pad rows, rig A | pad rows, rig B |
|---|---:|---:|
| 256x256 @ 51.4 (TODAY) | **0.0 %** | **11.3 %** |
| 256x256 @ 70 | 3.6 % | 21.5 % |
| 256x256 @ 90 | 24.6 % | 27.7 % |
| **256x256 @ 100** | **32.0 %** | **32.8 %** |
| 256x256 @ 120.5 | 43.2 % | 43.8 % |
| **256x640 @ 100 (letterbox)** | **0.0 %** | **0.06 %** |

The sensor is **1080 rows tall**; a *square* crop wide enough for 100 degrees needs ~1590 rows, so a
third of its input is replicate-padding of rows the camera never captured. **A wide square crop is
the wrong shape for this sensor, and that is arithmetic, not an opinion.**

**A previously unreported asymmetry in the DEPLOYED input, MEASURED:** today's 51.4-degree crop
already replicate-pads **11.3 % of rows for rig B clips and 0 % for rig A**, because rig B's
principal point sits ~215 px lower and the crop centred on it overflows the bottom edge. **29.1 % of
the corpus is rig A** (INHERITED, H2 A.3), so roughly **71 % of training frames carry an 11 %
padded band that the other 29 % do not.** Flagged in section 10; it is a property of the deployed
cache, not of this study.

---

## 2.5 The powered secondary — and a POWERED result in the opposite direction

Part 1 **fits nothing** — it is pure geometry against a privileged label — so there is no estimator
for an unseen split to protect. Repeating it over all **450** admitted clips is therefore
legitimate, is declared, and buys the power the held-out side lacked. It clears the 40-cluster bar
for **both** of the PI's powered situations.

| situation | clusters (held-out) | clusters (all 450) | `ALL/CROPPED_AWAY` lift, all 450 | `CROSS/CROPPED_AWAY` lift, all 450 |
|---|---:|---:|---|---|
| **intersection** | 44 | **58** | **1.404x [1.210, 1.590]** sep. | **8.187x [3.107, 15.975]** sep. |
| **lane change** | 36 (under) | **44** | **0.759x [0.528, 0.997]** sep. **BELOW 1** | 0.484x [0.000, 1.674] not sep. |

`ALL/IN_CROP` remains a null in both (intersection 1.021 [0.893, 1.136]).

> ### The lane-change result is now POWERED, and it goes AGAINST the widening case. Reported here at
> full prominence.
> At 44 clusters, `ALL/CROPPED_AWAY` for lane-change anticipation is **0.759 [0.528, 0.997] —
> separated below 1.0**. There is **significantly LESS** peripheral agent content before a lane
> change than at baseline. **A wider crop does not help lane changes on this evidence; it is an
> intersection lever.** The pre-registered `NO DIFFERENTIAL LOSS` outcome was reachable and, for one
> of the PI's three situations, something stronger than it actually occurred.

---

# 4. PART 3 — THE CROSS-CORPUS PRICE, and a claim that turned out to be FALSE

**Class: MEASURED (ours; re-derived independently after a sub-investigation surfaced it).**
Tier: **DECISION-GRADE.**

## 4.1 The premise "we already train on comma2k19 + PhysicalAI" is FALSE for the deployed model

The brief and the retarget both treat the shared `F_REF = 266` geometry as load-bearing because the
flagship trains on a **comma2k19 + PhysicalAI mix**. That is a **stale docstring**, and it belongs
to a different model.

| leg | evidence | class |
|---|---|---|
| the docstring that says "mix" | `stack/scripts/train_flagship4b.py:3-4, 24-27` — a *usage example*, not a default | MEASURED |
| the default | `train_flagship4b.py:563` `--cache-dirs ... default=None` — no corpus is baked in | MEASURED |
| **`--data cached` DISCARDS every cache dir after the first** | `train_flagship4b.py:186-188`: `roots = [Path(c) for c in cache_dirs]` then `if data == "cached": roots = roots[:1]` | **MEASURED (re-read by me)** |
| the deployed run's exact command | `MODEL_REGISTRY.md:176` — `--data cached --cache-dirs /workspace/data/physicalai_phase0/_epcache` | **MEASURED (re-read by me)** |
| the run's OWN artifact | `taniteval/results/trainlogs/v1-speedjerk_config.json` -> `{"data": "cached", "cache_dirs": ["/workspace/data/physicalai_phase0/_epcache"]}` | **MEASURED (re-read by me)** |
| every other committed flagship4b trainlog | `nospeed-phase0`, `v2`, `v3enc` — all `data: "cached"`, all the single PhysicalAI cache | MEASURED |
| the only real `realmix` arm is a DIFFERENT model | `stack/scripts/pod_launch.sh:52-55` launches `train_worldmodel --config base250cam --data realmix --sim-frac 0.6`, run id `p0-sB01-realmix` | MEASURED |

> ### **`flagship4b-speedjerk-30k` — the deployed v1 — trains on PhysicalAI-AV ALONE, 100 %.**
> comma2k19 is an **out-of-distribution EVAL** corpus for it (`MODEL_REGISTRY.md:252`: comma2k19
> 0.849 vs floor 0.372, win **17.5 %**, i.e. it fails there), not a training corpus.
>
> **Consequence, and it is the single most decision-relevant thing in this document after
> section 2.3: the cross-corpus price of widening PhysicalAI's crop is FAR SMALLER than the brief
> assumes. There is no comma2k19 training mixture to break.**

**A second, independent contradiction to correct:** `Benchmarks & Eval/GEOMETRY_INTEGRITY_AUDIT.md:26`
states *"Flagship mix | comma2k19 0.40 / PhysicalAI-AV 0.60"*. Its own provenance row scopes it to
`p0-sB01-realmix` (base250cam @ 27k) — **not** to `flagship4b-speedjerk-30k`. Anyone quoting "60 %
PhysicalAI" about the deployed model is wrong: it is **100 %**. Escalated in section 10.

## 4.2 comma2k19's hard ceiling — and why "letterbox comma" is a trap

MEASURED by me from `stack/tanitad/data/calib.py` (`COMMA2K19_FOCAL_PX = 910.0`, 1164x874):

| quantity | value |
|---|---|
| comma2k19's **entire** horizontal field | `2*atan(582/910)` = **65.203 deg** |
| canonical square crop side | `focal_crop_size(910, 874, 1164, 256, 266)` = **874** px — **HEIGHT-CLAMPED** (ideal 875.8) |
| field the canonical crop retains | **51.302 deg**, `f_eff` = **266.54** |
| already discarded from comma today | **21.32 %** of its angular field (24.91 % of its columns) |

> **comma2k19 CANNOT SUPPLY 100 DEGREES AT ANY RESOLUTION. 65.2 deg is its sensor ceiling.**
> And `F_REF = 266` was *derived* from comma (`calib.py:14`) — so raising the canonical field is
> simultaneously a decision about comma's role.

**comma2k19 IS locally measurable** if the program wants this costed properly: 188 raw segments
under `C:/Users/Admin/tanitad-data/comma2k19/extracted/Chunk_1/`, plus a canonical 256 px eval cache
`C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f/` (90 episodes, `frames_u8
[300, 9, 256, 256]`). *(INHERITED from the sub-investigation, spot-checked but not re-enumerated
by me.)*

**The three options, costed:**

| option | what it costs | verdict |
|---|---|---|
| **(a) per-corpus geometry** — PhysicalAI wide, comma narrow | breaks the one thing `F_REF` exists for: a shared action->pixel scale. But **the deployed flagship does not mix corpora (4.1)**, so the breakage is confined to *future* mixed training and to cross-corpus **eval** comparability | **viable, and cheap today** |
| **(b) letterbox / mask comma to the wide canvas** | ⛔ **REFUSE.** It teaches a **corpus-identifiable border**. This model demonstrably exploits shortcuts — zeroing `v0` moves the imagined decode **x93.7** while the perceived decode is bit-exactly unchanged (INHERITED). A constant black band that is present iff the corpus is comma is exactly such a shortcut, and it would be invisible in aggregate metrics | **DO NOT DO THIS** |
| **(c) drop comma from the flagship mix** | **already true** — it is not in the mix (4.1). The cost is losing comma as an *OOD eval* anchor at matched geometry | **already paid; do not assume it is free for EVAL** |

**Recommendation on the cross-corpus axis: (a), with comma retained as an OOD eval corpus at its own
geometry and cross-corpus ADE comparisons explicitly re-baselined.** The loss is comparability of
existing cross-corpus numbers, not training data.

## 4.3 The re-cache claim is TRUE — and its corollary is a live silent-collision hazard

**Claim tested:** *"skip-hash `f09e44db` and the parity key `physicalai-train-e438721ae894` govern
WHICH episodes are selected, not HOW they are cropped, so a re-crop is a re-cache, not a
re-selection."*

**CONFIRMED.** `stack/tanitad/data/epcache.py:62-68` hashes `{"ids": [...], "params": {...}}` where
`ids` are clip ids and `params` for the parity build is
`{"size": 256, "n_stack": 3, "hz": 10, "calib": "ftheta_v2", ...}`
(`stack/scripts/build_pai_cache.py:85-86`). `f09e44db` is `sha256` over the sorted skip **clip ids**
(`stack/tanitad/lake/filtering.py:64`). **Both are pure selection. Episode-selection parity
survives a re-crop.**

> ### 🔴 **BUT: the cache key is PROVABLY BLIND to the crop geometry. MEASURED BY ME.**
> ```
> key with F_REF=266 : eafe5e4eb363
> key with F_REF=133 : eafe5e4eb363     IDENTICAL -> True
> ```
> Same sources, same `params`; only `calib.F_REF` differed — a *completely* different crop. Neither
> `F_REF`, nor the crop side, nor the centring convention, nor the projection, nor the `calib`
> module version enters the hash. The only geometry inputs are the scalar `size` and a **hand-typed
> string** `"calib": "ftheta_v2"`.
>
> **So a re-crop that keeps `size=256` and forgets to bump that hand-typed tag writes a directory
> literally named `physicalai-train-e438721ae894` containing DIFFERENT PIXELS — and it passes every
> guard**, because the parity content check hashes only `ep_%05d.pt` **basenames**
> (`stack/tanitad/data/parity.py:104-118`), and the manifest says so itself: *"a same-name file with
> different tensor bytes is not detected here"*.
>
> **And a non-square frame cannot be expressed in `params` at all** — there is one scalar `size`.
> A 640x256 rebuild hashes **identically** to a 256x256 build.
>
> ⇒ **This must be fixed BEFORE any re-cache, not after.** Derive the geometry portion of `params`
> from the frame object (`{h, w, f_ref, projection, centring}`) instead of a hand-typed generation
> tag. Escalation 2, section 10.

---

# 5. THE ENGINEERING COST OF A NON-SQUARE INPUT — scoped precisely

**Class: MEASURED (ours) for what I executed; INHERITED for the wider blast radius.**

I built and ran the frozen v1 trunk at 256x256, 384x384, 640x640 and **256x640**
(`scripts/shape_shim.py`). **C-FID: at the deployed shape the shim is BIT-IDENTICAL to the real
trunk — `max_abs_diff = 0.0`.** So the measurements in section 3 are the deployed weights, not an
approximation of them.

**What actually has to change, and one correction to the circulating estimate:**

| component | today | change needed | size |
|---|---|---|---|
| `encoder.ViTEncoder` `grid_hw` / `n_tokens` | one scalar, `grid_hw**2` | two dims `(gh, gw)` | **~3 lines** |
| `encoder.ViTEncoder.pos` | learned `[1, 256, 768]` = 196,608 params | `[1, gh*gw, 768]`; 640x256 -> `[1, 640, 768]` = 491,520 | ⚠️ **checkpoint-shaped — a hard warm-start break** unless resampled |
| `readout.SpatialGridReadout` | `assert hw*hw == n_tokens`; `AvgPool2d(hw//grid)` | non-square reshape + pooling | **~5 lines** |
| `models/imagination.py` (`sector_mask`, `advect`, `ImaginationField`) | square `grid_hw` in ~8 places | `(gh, gw)` | real but mechanical |
| `epcache` `params["size"]` | one scalar | `(h, w)` + geometry (see 4.3) | **must change anyway** |

> ### ⭐ **CORRECTION to the circulating estimate: a non-square input does NOT have to change
> `state_dim`, and therefore does NOT have to reshape every downstream head.**
> The concern raised was that a `grid_h x grid_w` readout moves `state_dim` 2048 -> 5120 and
> reshapes the operative predictor, both policies, the goal head and every grounding head. **That is
> true only for the `grid_w` route.** Pooling the (non-square) token grid **adaptively to the SAME
> 4x4 readout grid** keeps `state_dim = 2048` exactly, and I **MEASURED** it: my 256x640 and 640x640
> trunks both emit `[B, 2048]`, and at the square shape the adaptive pool is bit-identical to
> `AvgPool2d(4)` (`max_abs_diff = 0.0`).
>
> **So there are two designs, and the cheap one was being costed as if it were the expensive one:**
> - **adaptive-4x4** — `state_dim` unchanged, **every downstream head loads unchanged**, only
>   `encoder.pos` breaks. Cost: readout cells are no longer square in angle.
> - **`grid_w`** (what a sibling stream is implementing today) — spends readout cells
>   asymmetrically on a wide input, at the price of `state_dim` and a full head reshape.
>
> **Recommend prototyping on adaptive-4x4 first**, because it isolates the geometry question from a
> whole-stack reshape.


## 5.1 ⚠️ Settling the `state_dim` question with the sibling non-square stream — my exact evidence

A sibling stream implemented the non-square readout via `grid_h`/`grid_w` and reports `state_dim`
unchanged; I measured the **adaptive-4x4** route and also report `state_dim` unchanged. **These are
not the same claim and both can be true**, so here is exactly what I ran, so it can be checked
rather than arbitrated.

**What I measured** (`scripts/shape_shim.py`, reproducible in one command on either host):

```
PYTHONPATH=<stack> V1_TRUNK=<v1_trunk.pt> python3 shape_shim.py
  -> {'max_abs_diff': 0.0, 'bit_identical': True, 'shape': [4, 2048], 'state_dim': 2048}
     (256, 256) tokens  256 -> (2, 2048)
     (256, 640) tokens  640 -> (2, 2048)
     (640, 640) tokens 1600 -> (2, 2048)
```

Reproduced **independently on the dev box (RTX 4060) and on pod2 (A40)**, from two separately
extracted trunks.

**The mechanism, and why it keeps `state_dim` fixed.** `SpatialGridReadout` pools a `hw x hw` token
grid with `AvgPool2d(hw // grid)` and projects to `d_readout`, giving `grid * grid * d_readout`.
I replaced the fixed pool with `AdaptiveAvgPool2d((grid, grid))` on a `(gh, gw)` grid. The output is
`grid x grid` **whatever the input grid is**, so `state_dim = 4 * 4 * 128 = 2048` at 256x256,
256x640 and 640x640 alike — and at the square shape the adaptive pool is **bit-identical** to
`AvgPool2d(4)` (`max_abs_diff = 0.0`, which is the same check that proves the whole shim).

**The distinction that matters, and it is a design choice, not a bug:**

| route | `state_dim` | readout cells across a 100 deg width | downstream heads |
|---|---|---|---|
| **adaptive-4x4** (mine) | **2048, always** | 4 cells over 100 deg — each cell is **2.5x wider in angle** than it is tall | **all load unchanged** |
| **`grid_w`** (sibling's) | 2048 **only if `grid*grid_w*d_readout` is held at 2048** (e.g. `grid=4, grid_w=10, d_readout=51.2` is not integral; `grid=4, grid_w=10` at `d_readout=128` gives **5120**) | `grid_w` cells — angularly square cells | reshaped **iff** `state_dim` moves |

⇒ **Neither route needs to be retired.** They answer different questions: adaptive-4x4 buys a
**zero-blast-radius prototype** (which is what the FOV question needs *now*), `grid_w` buys
**angularly-square readout cells** (which is probably what a shipped wide model wants). **My
recommendation stands as written in section 7: prototype on adaptive-4x4 so the geometry question is
not entangled with a whole-stack reshape, then decide `grid_w` on its own evidence.** If the sibling's
`grid_w` config holds `state_dim` at 2048, the two are compatible and the only real question is
whether angularly-square cells beat angularly-wide ones — **which is a measurable ablation, not an
architectural argument.**

⚠️ **I did NOT modify the training path.** `stack/` is untouched by this stream. The shim lives in
this folder. *(Noted for the manifest: `stack/tanitad/models/{encoder,readout,imagination,fourbrain}.py`
and `config.py` were being edited by a sibling stream while I ran; my probe is deliberately
constructed from `ViTEncoder` + the checkpoint directly, bypassing `fourbrain.WorldModel`, so this
sweep is reproducible against the checkpoint rather than against a file in flight.)*

---

# 6. PART 2 — THE FOV x RESOLUTION x ASPECT SWEEP

**Class: MEASURED (ours) for the harness, its fidelity checks and its geometry ledger.**
**Tier of the adjudicated result: see 6.4 — the pre-registered vacuity guard governs it.**
Code: `scripts/fov_labels.py`, `scripts/fov_geom.py`, `scripts/shape_shim.py`,
`scripts/fov_extract.py`, `scripts/fov_eval.py`.

## 6.1 What was built, and why it is a valid paired contrast

Neither host is used for its episode cache: the sweep **rebuilds every arm from the raw mp4s**, so it
needs only mp4s + `calibration/` + the extracted v1 trunk and never touches a parity cache. It was
prototyped on the dev box's 500 locally decodable R0 clips and **finished on pod2's 760**. Labels come from `sc_situations.py` **unmodified**, on `build_episode`'s **exact** query
grid, so label index == frame-stack index with no alignment step. Split is **chunk-grouped**
(157 TRAIN / 343 HELD-OUT clips over 30 chunks).

**Each clip is decoded ONCE and all 11 arms are cropped from those same pixels**, so the sweep is
paired at the pixel level: identical clips, identical native frames, identical labels, identical
split — **only the crop geometry and the input shape differ.**

| arm | shape | HFOV | true VFOV | px/deg (on-axis) | tokens | what it isolates |
|---|---|---:|---:|---:|---:|---|
| `A_51_256` | 256x256 | 51.39 | 51.39 | 4.643 | 256 | **TODAY — the baseline** |
| `F70_256` | 256x256 | 70.0 | 70.0 | 3.191 | 256 | FOV, at zero compute cost |
| `F90_256` | 256x256 | 90.0 | 90.0 | 2.234 | 256 | " |
| `F100_256` | 256x256 | 100.0 | 100.0 | 1.874 | 256 | **the PI's ask, free in compute** |
| `F120_256` | 256x256 | 120.5 | 120.5 | 1.277 | 256 | the whole sensor |
| `M_match100` | 256x256 | 51.39 | 51.39 | 4.643 | 256 | **matched-degradation control** (F100's blur AND its padding, at today's field) |
| `R_blur100` | 256x256 | 51.39 | 51.39 | 4.643 | 256 | **INSTRUMENT-BLIND check** (F100's blur only, zero information added) |
| `A_51_384` | 384x384 | 51.39 | 51.39 | 6.964 | 576 | **resolution at today's field** |
| `F100_384` | 384x384 | 100.0 | 100.0 | 2.812 | 576 | resolution at 100 deg |
| `F100_640sq` | 640x640 | 100.0 | 100.0 | 4.686 | 1600 | 100 deg at today's angular resolution, **square** |
| `F100_640x256` | 256x640 | 100.0 | **39.35** | 4.686 | **640** | **the letterbox** — same width, same HFOV, same angular scale |

## 6.2 Bi-directional validation — both legs, both able to fail

**C-FID (fidelity).** The baseline arm rebuilt from raw mp4 must reproduce the repo's own canonical
crop. Two legs:

1. **The crop BOX must equal `calib.ftheta_crop_box` exactly, in integers.** ⭐ **This leg has teeth
   and it already caught a real bug in my own code:** deriving the crop offset from the *unrounded*
   half-side instead of the rounded box size disagrees with `calib` by **one pixel** whenever
   `cy - r` sits near a .5 boundary, which happened on a real clip. Fixed in `fov_geom.py`.
2. **The pixels**, with a stated tolerance: my crop runs on the GPU and `calib`'s on the CPU, and
   the `uint8` cast **truncates**, so a 1-ulp bilinear difference can flip a level.

**Result after the fix: box exact on 100 % of clips; max pixel difference 1; the fraction of pixels
differing by more than 1 is 0.0.** A clip failing either leg is refused, not warned about.

**C-NEG (a deliberately failing input).** Column-shuffled features, fitted and scored through the
identical pipeline, must land at chance. **The chance comparator itself is audited** by
`taniteval.rank_metrics.assert_chance_comparator`, which is unwaivable and which exists because this
program shipped a "chance" baseline that scored **1.726x chance**.

**The shim's own C-FID.** At the deployed shape the reshaped trunk is **bit-identical** to the real
trunk (`max_abs_diff = 0.0`), so every contrast is against the true deployed arm.

## 6.3 The geometry ledger the sweep produced — a result in its own right

Even before any AP, the per-clip geometry ledger (`artifacts/geom.parquet` -> section 3.1(c))
establishes two things that do not depend on the classifier at all:

- a **square** 100-degree crop is **32 % replicate-padding**; the **letterbox is 0.06 %**;
- today's deployed 51.4-degree crop already pads **11.3 % of rows on rig-B clips and 0 % on rig A**.

## 6.4 Adjudication — and the guard that governs it

**The sweep's universe is bounded by dev-box wall-clock, not by design.** Extraction runs at
~16 s/clip (11 arms x 3 distinct trunks + a full 1080p decode per clip), under contention with
other work on the same GPU. The pre-registered vacuity guard is what decides whether a verdict may
be emitted:

> **`REFUSED` — if a situation has fewer than 40 held-out positive clip clusters, or if the baseline
> arm does not clear chance, NO VERDICT IS EMITTED for it.**

**~170 clips are needed to reach 40 intersection clusters** (51 % of held-out clips carry an
intersection event, 69 % of clips are held-out). The adjudicated table below reports the state
reached; `scripts/fov_eval.py` is resumable and re-runs in ~2 minutes on whatever
`fov/feats/clip_*.npz` exists, so this completes without re-deriving anything.

<!-- SWEEP_RESULT -->
### RESULT: **`REFUSED` on all three situations — the vacuity guard fired on n.**

`artifacts/sweep_result.json`, produced by `fov_eval.verdict()`:

| situation | held-out rows | held-out positives | **positive clip clusters** | bar | **verdict** |
|---|---:|---:|---:|---:|---|
| intersection | 1,756 | 206 | **20** | 40 | **REFUSED** |
| lane change | — | — | **4** | 40 | **REFUSED** |
| roundabout | — | — | **2** | 40 | **REFUSED** |

**53 of 500 clips were extracted before the wall-clock ran out** (the run was still advancing at
handover; `sweep_result.json` is the last adjudication and re-running `fov_eval.py` updates it). Extraction ran at ~31.5 s/clip
under contention (11 arms x 3 distinct trunks + a full 1080p decode per clip, on a GPU shared with
other work); **~170 clips are needed for 40 intersection clusters.**

> ### ⛔ **READ THIS CORRECTLY. `REFUSED` IS NOT A NULL RESULT ABOUT FOV.**
> It means **no verdict was emitted** because the instrument did not reach the power it
> pre-registered as its minimum. **`sweep_result.json` does contain per-arm APs. They are NOT quoted
> anywhere in this document and they must not be quoted from the JSON either** — at 20 clusters they
> are noise, and quoting them is exactly the vacuous diagnostic this program's most recent fix was
> written to prevent. The guard did its job; the correct response is to finish the run, not to read
> the numbers early.

**What this does and does not leave standing:**

- **Part 1 (section 2) is unaffected.** It fits nothing, needs no classifier, and answers the FOV
  question directly at CONFIRMED tier.
- **Section 3's compute table is unaffected** — it is a timing measurement, not a fit.
- **What is genuinely NOT answered:** whether a wider or higher-resolution input actually improves a
  *model*. The three questions left open are exactly `F100_256 vs M_match100` (does 100 deg help
  once its own blur and padding are paid for), `A_51_384 vs A_51_256` (do more pixels help at
  today's field), and `F100_640x256 vs F100_640sq` (is the letterbox as good as the square).

> ### 🔴 ESCALATION 6 — **finish this sweep on a POD, not the dev box.**
> The harness is complete, staged and resumable: `fov_extract.py` skips any `clip_*.npz` that
> already exists, and `fov_eval.py` re-adjudicates in ~2 minutes on whatever has accumulated.
> **It needs only raw front-wide mp4s, `calibration/`, and `v1_trunk.pt` — no parity cache, no pod
> state.** On an idle A40 the encoder is several times faster than the contended 4060 and the run is
> ~1-2 GPU-hours for all 500 clips. **This is a finish-the-job ask, not a rebuild**, and it is the
> single highest-value follow-up in this document.
<!-- /SWEEP_RESULT -->

⚠️ **Whatever this sweep returns, its absolute APs are NOT comparable to the situation classifier's
pod3 numbers** — different universe, different corpus slice, TURN-half-only intersection label. Only
the **within-sweep** contrasts are valid, and they are valid because every arm shares clips, frames,
labels and split.

---

# 7. RECOMMENDATION

**Read the evidence class on each line. The recommendation rests on what is MEASURED, and it says
plainly where it is not.**

## 7.1 What the evidence supports

| # | recommendation | rests on | class / tier |
|---|---|---|---|
| **R1** | **Go to ~100 deg horizontal, or do not move at all.** An intermediate 60-70 deg widening is close to worthless: it recovers **3.4 %** of the missed decision-relevant content while paying the full angular-resolution price; 100 deg recovers **31.3 %** | section 2.3 recovery curve | **MEASURED, CONFIRMED** |
| **R2** | ⭐ **Do it with a WIDE, NOT SQUARE frame — `256 x 640`.** Same 100 deg, the **same on-axis angular resolution as today** (4.686 vs 4.643 px/deg), **2.78x** inference / **3.06x** training cost vs **9.37x / 10.39x** for `640x640` — **3.4x cheaper for an identical field and resolution** | section 3 (measured, spill-filtered) | **MEASURED, DECISION-GRADE** (relative cost) |
| **R3** | **Do NOT take 100 deg at 256x256.** It is free in compute and costs **2.48x** angular resolution (4.643 -> 1.874 px/deg) **and** makes a third of the input replicate-padding | sections 3, 3.1(c) | **MEASURED, DECISION-GRADE** |
| **R4** | **Per-corpus geometry (option a).** The premise that this breaks a comma+PhysicalAI training mix is **FALSE** — the deployed flagship trains on PhysicalAI **alone** | section 4.1 | **MEASURED, DECISION-GRADE** |
| **R5** | ⛔ **Refuse the letterbox/mask option for comma2k19.** A constant corpus-identifiable border is exactly the shortcut class this model exploits | section 4.2 | **REASONED from a MEASURED shortcut (INHERITED x93.7 figure)** |
| **R6** | 🔴 **Fix the cache key BEFORE any re-cache.** It is provably blind to crop geometry, and a non-square frame cannot be expressed in `params` at all | section 4.3 | **MEASURED, DECISION-GRADE** |
| **R7** | **Prototype on the adaptive-4x4 readout**, which keeps `state_dim = 2048` and leaves every downstream head loadable; only `encoder.pos` breaks | section 5 | **MEASURED, CONFIRMED** |

## 7.2 What the evidence does NOT support, stated as plainly

- **It does not support widening for LANE CHANGES.** At 44 clusters the peripheral content before a
  lane change is **significantly LOWER** than baseline (0.759 [0.528, 0.997]). **Widening is an
  intersection lever on this evidence.** (section 2.5)
- **It does not yet support "more pixels improve the model."** The compute price of resolution is
  measured; the *benefit* is what the sweep tests, and section 6.4 governs whether that number is
  quotable. **No claim that higher resolution helps the model appears in this document unless the
  sweep cleared its guard.**
- **It does not establish that ~half the periphery is crop-recoverable in general.** That holds for
  decision-relevant cross traffic at intersections (**48.3 %**); over all agents it is **~1/6**.
  (section 2.4)
- **The absolute ms figures are a 4060 number.** They are **not** an A40, Orin or Thor number. Only
  the *ratios* should travel.

## 7.3 The concrete proposal

**Adopt `256 x 640` at 100 deg horizontal as the candidate frame, and gate it on a retrained arm —
not on this shim.**

- **`f_eff` = 268.5** (vs 266 today) — angular scale is essentially **preserved**, which is the
  property `F_REF` existed to protect. The change is **field**, not scale.
- vertical field falls **51.4 -> 39.35 deg**. That is the real cost of the letterbox and it is the
  thing to watch: it trims sky and near-bonnet road.
  ⭐ **If that vertical loss is judged unacceptable, `320 x 640` buys it back**: true VFOV
  **49.25 deg** (essentially today's 51.4) at the same 100 deg and the same 4.686 px/deg, for
  **800 tokens** instead of 640 — an ESTIMATED ~3.7x inference cost vs the letterbox's measured
  2.78x, and still **~2.5x cheaper than the 640x640 square**. This is the fallback if a
  vertical-field ablation says the sky and near road matter.
- **tokens 256 -> 640; training ~3.06x.** On the flagship's 30k-step budget that is the price of the
  decision.
- **Episode-selection parity SURVIVES** (section 4.3) — it is a **re-cache**, not a re-selection —
  **but only if R6 is done first**, or the rebuild silently collides with `e438721ae894`.

**The cheapest discriminating experiment, pre-registered:** retrain one short arm at `256x640`/100
deg against a matched-budget `256x256`/51.4 deg control on the same episodes, and read the gate.
Both outcomes committed in advance: if the wide arm does not beat the control on the gate's primary,
**51.4 deg stands and the periphery needs the H2 second camera** — which section 2.2 already shows
carries a **10.1x** lift for off-front decision-relevant content, i.e. that is a good outcome too.

---

# 8. LIMITATIONS, STATED PLAINLY

1. ⛔ **The dev box does not hold the parity episode cache** (`14231cd29c74`, not `e438721ae894`).
   Part 2 rebuilds from raw mp4s and is a **self-contained paired experiment**; its absolute APs are
   never comparable to the situation classifier's pod3 numbers and are not quoted as such.
2. ⛔ **Part 2's universe is bounded by dev-box wall-clock**, not by design, and shared a contended
   GPU. The pre-registered vacuity guard (section 6.4) is what protects the conclusion; where it
   fires, no verdict is emitted.
3. ⚠️ **Every non-256 arm carries a one-directional handicap** (section 1.7): the frozen v1 weights
   were trained at 256x256/51.4 deg, so a resolution or aspect change is evaluated under a train/test
   shape shift that can only hurt it. **A win under that handicap is strong; a tie or loss is weak
   evidence.**
4. ⚠️ `obstacle.offline` is `scene:obstacles:autolabels:v2` — **machine labels**. Systematic misses
   of small or distant agents attenuate every band statistic in Part 1, and the attenuation is
   *toward understating* the periphery's content, i.e. against this document's own conclusion.
5. ⚠️ **Part 1's intersection label is the sibling stream's**, whose cross-traffic half covers only
   450 of 2,376 clips; and Part 2's is the **TURN half alone**. Section 4 V4 of that stream
   (2.415x [1.057, 7.931]) is what licenses the turn half standing alone, and it licenses it only to
   the strength of its own interval.
6. ⚠️ **The P-CROSS anticipation stratum is 157 agent samples.** The point estimates there are
   strong and the presence lift is separated, but the content-share intervals are wide. Tier
   CONFIRMED, not DECISION-GRADE.
7. ⚠️ **The rig mix in Part 1's universe is ~50/50 A/B**, not the corpus-wide 29.1 % rig A
   (the 26 locally available `obstacle.offline` chunks are not a random draw). Because every
   projection is per-clip this cannot bias the *geometry*, but the pooled band shares in section 2.4
   are shares over **this** clip sample.
8. ⚠️ **The compute table is a single-GPU inference/step-time model.** It excludes decode, ISP,
   dataloader and multi-GPU scaling. It is therefore optimistic on the absolute training cost of a
   wider frame and conservative on nothing.
9. ⚠️ **The cylindrical / equidistant-projection comparison at 100 deg was NOT run.** It was the
   lowest of the six revised priorities and the wall-clock went to the sweep and to the compute
   table. `fov_geom.py` already retains the real f-theta radial map, so this is a small follow-up,
   not a rebuild. **Declared as not done rather than quietly dropped.**

---

# 9. DELIVERABLE MANIFEST

**Everything below is in the repo working tree and STAGED (`git add`). Nothing was committed and
nothing was pushed.** Verified with `git ls-files --stage`, not a scoped `git status`.
Path: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-fov-crop-audit/`

| artifact | what it is | where it lives |
|---|---|---|
| `FOV_CROP_AUDIT.md` | this document — pre-registration, Part 1, the compute table, the cross-corpus finding, the recommendation | **repo** |
| `artifacts/band_stats_heldout.json` | ⭐ Part 1 PRIMARY: per-situation presence lift, content share and recovery curve, held-out | **repo** |
| `artifacts/band_stats_all.json` | Part 1 powered secondary over all 450 clips (the lane-change powered negative lives here) | **repo** |
| `artifacts/bands_summary.json` | the band run's own provenance: C-ALIGN residuals, rig split, the half-angle grid | **repo** |
| `artifacts/fov_bands_index.csv.gz` | per-clip band index (`*.parquet` is git-ignored in this repo) | **repo** |
| `artifacts/shape_bench.json` | ⭐ measured compute + memory per input shape, **with the WDDM spill verdict on every row** | **repo** |
| `artifacts/shape_bench_train_smallbatch.json` | the non-spilled training re-measurement that the spill filter forced | **repo** |
| *(a later `320x640` benchmark was DISCARDED, not staged)* | it ran while this stream's own extraction held the GPU and returned 294.9 ms/frame for 256x256 vs the idle 7.21 — **contended benchmarks are invalid and none is published** | — |
| `artifacts/labels_summary.json` | the sweep's label universe and chunk-grouped split | **repo** |
| `artifacts/geom.csv.gz` | ⭐ per-clip, per-arm geometry ledger: true VFOV, px/deg, **replicate-padded fraction by rig** | **repo** |
| `artifacts/sweep_result.json` | the sweep's adjudicated output — **verdict `REFUSED` on n**; contains per-arm APs that are **deliberately not quoted** (section 6.4) | **repo** |
| `scripts/crux.py` | ⭐ **RESCUED FROM A SINGLE DISK** — the per-clip f-theta projection machinery shared by the H2, situation-classifier and FOV streams | **repo** |
| `scripts/fov_bands.py` | Part 1: per-agent-sample band assignment + `req_half_px` (one scalar -> the whole recovery curve) | **repo** |
| `scripts/fov_band_stats.py` | Part 1 statistics: presence lift, content share, recovery curve, all clip-cluster bootstrapped | **repo** |
| `scripts/fov_geom.py` | ⭐ rectangular f-theta crop at arbitrary HFOV + the `blur_like` / `pad_like` confound controls | **repo** |
| `scripts/shape_shim.py` | ⭐ the minimal relaxation that runs the frozen v1 trunk at any shape; **`verify_identity()` proves it is bit-identical at the deployed shape** | **repo** |
| `scripts/fov_shape_bench.py` | the compute/memory benchmark **with the spill filter built in** | **repo** |
| `scripts/fov_labels.py` | situation labels on `build_episode`'s exact grid for the locally decodable clips | **repo** |
| `scripts/fov_extract.py` | the sweep: decode once, crop 11 ways, encode, **C-FID per clip** | **repo** |
| `scripts/fov_eval.py` | PCA-16 + ridge per arm, paired clip-cluster bootstrap, C-NEG, and the pre-registered `verdict()` | **repo** |

**Not in the repo, and deliberately so:**

| | where | why |
|---|---|---|
| `fov/feats/clip_*.npz` (per-arm frozen features, 53 of 500 clips and still advancing) | **dev-box scratch only** | derived from a gated corpus; rebuilt/resumed by `fov_extract.py`. ⚠️ Extraction was still running at handover — re-run `fov_eval.py` on the enlarged set, it is idempotent |
| `fov/bands/fov_bands.npz` (2.65 M per-agent-sample records) | **dev-box scratch only** | derived from gated `obstacle.offline`; rebuilt in ~51 s by `fov_bands.py` |
| `fov/labels/_LOCAL_ONLY_k2clip.json` | **dev box only, outside the repo** | 🔒 clip UUIDs may never enter a derived artifact |
| ~~`crux.py`~~ | ✅ **RESCUED** -> `scripts/crux.py`, staged | it existed in ONE place while THREE streams depended on it. Now in the repo. Escalation 4 is now only about **where it should finally live** (`stack/tanitad/data/`), not about whether it survives |

**Reproduction, end to end (dev box only, no pod):**

```
python fov_bands.py       <poses_dir> <sc_bundle> <bands_out>
python fov_band_stats.py  <bands_out> <sc_bundle> band_stats_heldout.json HELDOUT
python fov_band_stats.py  <bands_out> <sc_bundle> band_stats_all.json     ALL
python fov_shape_bench.py shape_bench.json
python fov_labels.py      <labels_out>
python fov_extract.py     <labels_out> <feats_out> --stride 3
python fov_eval.py        <feats_out> <labels_out> sweep_result.json
```

---

# 10. 🔴 ESCALATIONS — raised here, in the headline, not buried in a README

1. 🔴 **`GEOMETRY_INTEGRITY_AUDIT.md:26` and `train_flagship4b.py:3-4,24-27` both assert a
   comma2k19 + PhysicalAI flagship mix. The deployed v1 trains on PhysicalAI ALONE (100 %).**
   MEASURED three independent ways (section 4.1), including the run's own committed config JSON.
   The "0.40/0.60 mix" belongs to `p0-sB01-realmix` (base250cam @27k), a different model. **Any
   geometry, corpus or licensing decision premised on "we already train on both corpora" is
   premised on a false statement.** Belongs in `RETRACTION_LOG.md` under the **stale-prose /
   wrong-lineage** class — the same class as the `flagship4b-phase0` inversion. *(Flagged here
   rather than written by me into a shared program file mid-session.)*

2. 🔴 **The episode-cache key is PROVABLY BLIND to crop geometry, and the parity content check
   hashes only filenames. MEASURED: `F_REF = 266` and `F_REF = 133` produce the IDENTICAL key
   `eafe5e4eb363`.** A re-crop that keeps `size=256` and forgets to hand-edit the `"calib"` string
   writes a directory named `physicalai-train-e438721ae894` containing different pixels and **passes
   every guard**. A non-square frame cannot be expressed in `params` at all. **This must be fixed
   before any re-cache** — derive the geometry portion of `params` from the frame object. This is
   the **guard-that-cannot-fail** class.

3. 🔴 **The deployed input is rig-asymmetric and nobody has reported it.** Today's 51.4-degree crop
   replicate-pads **11.3 % of rows on rig-B clips and 0 % on rig A** (MEASURED, `geom.csv.gz`),
   because the principal-point-centred crop overflows the bottom edge for rig B. With ~71 % of the
   corpus on rig B, most training frames carry an 11 % band of invented pixels that the rest do not.
   This is a property of the **deployed cache**, not of this study, and it should be checked against
   the D-016 R1 acceptance criteria.

4. ✅→🔴 **`crux.py` is RESCUED, but it needs a permanent home.** The per-clip projection machinery
   that this stream, the H2 stream and the situation-classifier stream all depend on existed on **one
   disk, outside git** — three streams' headline numbers were unreproducible without it. **I have
   staged it verbatim at `scripts/crux.py` in this folder** (only a provenance header added), so it
   can no longer be lost. **What remains is a decision, not a rescue:** it belongs in
   `stack/tanitad/data/` next to `calib.py`, and it should not live in three incoming/ folders. That
   move is for the owner of the geometry layer.

5. **The H2 12.65 % / 11.69 % pair is not comparable as quoted** (section 2.4). The cropped-away
   figure reproduces; the off-front figure is 5.2x larger over the full agent population. The
   derived claim *"roughly half of what an extra camera pass would buy is recoverable by a wider
   crop"* is true for decision-relevant cross traffic at intersections (**48.3 %**) and false as a
   global statement (**~1/6**). H2's denominator should be published beside the numbers.

## What this unblocks

1. **The v5 / re-cache geometry decision** — R1–R3 give the frame, R6 gives the blocker that must
   clear first.
2. **The H2 sensor-activation case is STRENGTHENED, not weakened.** `CROSS/OFF_FRONT` lift is
   **10.127x [1.962, 24.561]** — the single largest effect in this document. Even at 120.5 deg a
   wider crop recovers only **47.6 %** of what is missed; **the other half is genuinely off the
   front sensor and only `cross_left`/`cross_right` can see it.** Widening and the second camera are
   complements, not alternatives.
3. **The sibling non-square stream** (`grid_w` in `config.py`/`readout.py`, in flight today) now has
   a measured cost table, a measured alternative (adaptive-4x4, `state_dim` preserved) and a
   bit-identical reference implementation to check against.
