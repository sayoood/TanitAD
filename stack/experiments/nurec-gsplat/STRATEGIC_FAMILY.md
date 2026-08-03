# STREAM C — making the STRATEGIC family scoreable

**All numbers MEASURED 2026-08-03 on Thor unless labelled otherwise. Artifacts in `results/`.**

---

## ⛔ First, a correction to the brief's premise

> *"on the 20 s night clip … the clip contains no junction-scale decision"*

**Half of that is REFUTED and half is CONFIRMED, and the difference decides the whole survey.**

Scene `00040136-e651-4abd-991d-0655ccda9430` (the night clip) — 202 poses, 20.0 s, 312.16 m,
mean 15.61 m/s:

| claim | verdict | evidence |
|---|---|---|
| the clip contains no junction | ⛔ **FALSE** | the ego is **inside a junction for 46 of its 202 poses (22.8 %)**, across **4** junctions (220, 222, 230, 239) |
| the clip contains no junction-scale **decision** | ✅ **TRUE, and now proved from the map** | at **every one of the four**, the ego's own lane has exactly **ONE** admissible continuation; max heading change through any of them is **2.58 deg** |

So the strategic family was not degenerate for lack of a junction. It was degenerate because
**there was nothing to choose**. `route_head_eq_logged = 1.0000` was a constant-predictor tie
against a label that was *correct* — `route_from_future_v21` returned ROAD FOLLOWING because the
map says road following was the only option. **The instrument was right; the scene had no branch.**

That reframes the search: nearly every urban clip has a junction. What is scarce is a **branch**.

---

## (a) Distance to the nearest junction, over the 202 poses

Two independent sources, deliberately not merged (`junction_probe.py`, run
`results/junction_00040136.json`):

**Source 1 — `map.xodr`** (219 roads / 26 junctions / 356 driving lanes / 86 junction-internal
roads, georeferenced `+proj=tmerc +lat_0=59.3487 +lon_0=17.9571`). Metric = clearance to the
nearest junction **drivable surface**, 0 when inside:

| min | p25 | median | p75 | max | poses inside |
|---|---|---|---|---|---|
| **0.000** | 1.116 | **19.963** | 88.829 | 140.822 | **46 / 202** |

**Source 2 — `clipgt/intersection_area.parquet`** (NVIDIA's own labels, 4 polygons, all
`is_complete=false`). Signed distance, negative inside:

| min | p25 | median | p75 | max | poses inside |
|---|---|---|---|---|---|
| **−7.314** | 13.430 | 64.563 | 140.505 | 236.596 | **22 / 202** |

The counts differ (46 vs 22) because the xodr surface is the full extent of the connecting roads
while the clipgt polygon is the core box — **they are not the same object and are not averaged.**
Two of the four polygons are entered: `FOUR_WAY` poses 154–171, `T_JUNCTION` poses 198–201.

**Cross-source agreement (the mandatory self-consistency control).** The clipgt ego track and the
geodetic map-frame track are fitted rigidly: **rms 0.1053 m, max 1.479 m over n=202**
(the clip window is `pose_record[49:251]`, not the first or last 202 — worth knowing, an earlier
pass took a different offset and got a 341.3 m path length instead of the true 312.16 m).
Transformed into the map frame, **100 % of every clipgt polygon's vertices land on an xodr
junction surface (median distance 0.000–0.004 m)**. The two sources agree on *where* the junctions
are.

**Controls, run before any number above was quoted:**

| control | result | verdict |
|---|---|---|
| C1 positive — a point on a junction lane centre must score 0 | max 0.000000 m over n=20 | PASS |
| C2 negative — the ego track shifted 500 m | min 342.79 m, 0 poses inside | PASS |
| C3 discrimination — midpoints of the 20 longest plain roads | median 50.184 m | PASS |

### The four traversals, and why none is a decision

| junction | poses | incoming | took | options **from the ego's lane** | options from its road | Δheading |
|---|---|---|---|---|---|---|
| 230 | 109–112 | road 67 lane −2 | 199 | **1** | 2 | −1.57° |
| 220 | 153–172 | road 65 lane −1 | 64 | **1** | 2 | −2.58° |
| 239 | 180–198 | road 189 lane −1 | 21 | **1** | 1 | −2.20° |
| 222 | 199–201 | road 190 lane −1 | 22 | **1** | 1 | −0.15° |

Junctions 230 and 220 do offer a second connecting road — but only from lane −2, and the ego is in
lane −1. That is a **lane-change decision taken ~40 m earlier**, not a junction decision, and it is
reported separately rather than folded in.

⚠️ **Instrument bug found and fixed here.** Resolving the incoming road by scanning backwards over
snapped poses reported `incomingRoad=20` for junction 239 and an option count of **0** — because
the real incoming connector, road 189, is **1.02 m long** and never wins a nearest-lane snap. A
zero reads as *"no continuation exists"*, which is false. The incoming road is now read from the
junction's own `<connection>` table. Regression test:
`test_resolve_incoming_survives_a_sub_metre_connector`.

---

## (b) Survey of all 1607 NuRec scenes

`nurec_hf_survey.py`. A USDZ is a zip whose members are all **STORED, never deflated** (39/39
verified), so an HTTP `Range` request pulls one member out of a 2 GB archive. Stage 1 read
`clipgt/egomotion_estimate.parquet` + `clipgt/intersection_area.parquet` from every scene.

> **1607 / 1607 scenes, 0 errors, 268.5 MB of traffic, 372 s** (≈ 15 300 scenes/h, 12 workers).
> That is **0.009 %** of the ~3 TB the naive download would have moved.

**Method validated end-to-end:** the range reader reproduced the local file's numbers exactly
(312.16 m, −43.96°, spans 154–171 / 198–201), and after the winning scene was downloaded in full
(1 835 661 788 bytes, md5 `2334a731cc03f72f518e55d5f81ae10e`), **every member the range reader had
pulled was md5-identical to the same member extracted from the complete archive.**

### Tiers (complete = ≥10 poses before entry AND ≥10 after exit)

| tier | definition | scenes |
|---|---|---|
| **T1** | complete traversal, **\|Δyaw\| ≥ 60°** — a real turn | **141** |
| T2 | complete traversal, 25° ≤ \|Δyaw\| < 60° | 84 |
| T3 | complete traversal, \|Δyaw\| < 25° — straight-through | 485 |
| T4 | traversal truncated by the clip boundary | 192 |
| T5 | intersection polygons present, never entered | 363 |
| T6 | no intersection polygon at all | 342 |

Distribution of the best complete turn: **55.88 % of scenes are exactly 0°**, p50 = 0.00,
p75 = 2.06, p90 = 53.49, p99 = 95.44, max 158.94.
**The night clip sits at the 75.6th percentile (2.27°) — it is a *typical* scene, not an unlucky
one.** ⇒ a strategic benchmark cannot be built by grabbing scenes at random; it must be selected.

**The scene list is banked, not just the count**: `results/junction_turn_scenes.tsv` — all
**225** T1+T2 scene ids with turn, category, pose span and context margins, ready to consume.
T1 category mix: `FOUR_WAY` 76, `T_JUNCTION_ASYMMETRICAL_PUBLIC_CONTROLLED` 13, `T_JUNCTION` 10,
**`ROUNDABOUT` 9**, `TWO_WAY_STOP` 9, `T_JUNCTION_ASYMMETRICAL` 7, `FOUR_WAY_STOP` 5, others 12.

⚠️ Those are candidates **by geometry**. Whether the ego's own lane had ≥2 options is a *map*
question and needs stage 2 per scene — on the 13 shortlisted, **2 of 15 scored scenes were refused
by the self-consistency control**, so budget for a similar refusal rate.

`ROUNDABOUT` and `CUL_DE_SAC` are present — relevant to the standing strategic-topology gap, since
PhysicalAI-AV itself ships no map.

### Stage 2 — lane-level decisions on 13 shortlisted scenes

The full `junction_probe.py` (map + clipgt + controls) on each:

| scene | ctl | align rms | traversals | lane-level decisions | max options **from the ego's lane** | max turn |
|---|---|---|---|---|---|---|
| **`7c72937c-c620-4776-9555-d57222c0081f`** | PASS | **0.0035 m** | 3 | **2** | **4** | 126.25° |
| `302c5c99-2762-4f5d-950e-1122f3e7c20a` | PASS | 0.027 m | 6 | 3 | 3 | 66.60° |
| `4a2db3b3-105b-4b03-9ddc-fe5900098887` | PASS | 0.087 m | 1 | 1 | 3 | 163.43° |
| `68abdefe-bd54-4b5e-95c6-a2e2eba6f51e` | PASS | 0.054 m | 4 | 3 | 2 | 114.22° |
| `b5d12506-227c-4461-b38e-d8ce2ec22247` (ROUNDABOUT) | **FAIL C3** | 0.038 m | 5 | 3 | 2 | 67.75° |
| *(9 more in `results/nurec_stage2_decisions.json`)* | | | | | | |

⚠️ The roundabout scene **fails control C3** (plain-road median clearance 4.955 m < 5 m threshold):
in a roundabout-dense map the "plain roads" are short stubs between junctions, so the
discrimination control is marginal. **Its numbers are not admissible** until C3 is re-specified for
that topology. Reported rather than quietly passed.

### ⭐ The recommended scene

**`7c72937c-c620-4776-9555-d57222c0081f`** — downloaded in full to
`thor:/home/nvidia/nurec_scenes/sample_set/26.04_release/7c72937c-c620-4776-9555-d57222c0081f/`
(usdz + reference mp4; `volume.nurec` 603 MB present, so it renders with the existing loader).

* 202 poses, 20.0 s, 127.55 m, mean **6.38 m/s** (urban), net heading change **+131.86°**
* **junction 149** (clipgt `FOUR_WAY`, 28-vertex polygon): entered pose 82, exited 143 — **85 poses
  of context before, 60 after**
* incoming road 35 lane −1 → **4 admissible continuations** (options span `{LEFT, UTURN}`), the ego
  took connecting road 48
* map-derived branch angle **+123.57°** vs realised **+123.53°** — **self-consistency error 0.04°**
* a second scoreable decision at junction 154 (2 options), and one single-option junction at 153
* 100 / 202 poses are *admissible* (a ≥2-option decision within 60 m ahead), `EFFECTIVE_N = 2`
* **Confirmed renderable**: loaded with the programme's own `nurec_loader.py` on Thor —
  **2,919,829 gaussians** (background 2,246,775 · road 419,829 · dynamic_rigids 252,490 ·
  dynamic_deformables 735), comparable to the night clip's 3.1 M. Unlike the night clip it has a
  substantial **dynamic_rigids** layer (`fourier_dim=20`), i.e. moving cross-traffic at the junction.

---

## (c) What the STRATEGIC family measures once a junction scene exists

Implemented in `strategic_gt.py`; emitted for the winner in `results/strategic_gt_7c72937c.json`.

### The thing that was actually missing: the **option set**

Not a route classifier — the harness already has one. What it lacked is *what the map admitted at
that moment*. Without it there is no way to separate a 1.0 that means **"chose correctly among
four"** from a 1.0 that means **"there was one road and everyone drove down it"**. Per pose the
label carries:

| field | meaning |
|---|---|
| `n_options` | admissible continuations **from the ego's own lane** — the metric's branching factor, hence its discriminability |
| `options[]` | every alternative, with its own map-derived class — the confusion set |
| `route_gt_class` | LEFT / STRAIGHT / RIGHT / UTURN of the branch the ego took |
| `dist_to_decision_point_m` | arc distance to the junction entry |
| `admissible` | **false unless `n_options ≥ 2` and the decision is within the horizon** |
| `event_id` | the resampling unit for the CI |

### Five rules the family must obey

1. **The class comes from the MAP, not the realised trajectory.** A route label read off the ego's
   own future yaw is circular — it cannot separate *"took the left branch"* from *"drifted left on
   a curving road"*. Here the class is the heading change of the *connecting lane* relative to the
   *incoming lane*.
2. **Only poses with `n_options ≥ 2` are scoreable.** Everything else is a constant-predictor tie
   and is excluded by the mask, not averaged in. On the night clip that is **0 of 202 poses**; on
   the winner it is **100 of 202**. This single rule is what makes the family able to discriminate.
3. **The value is per DECISION EVENT; the resampling CLUSTER is the SCENE.** Every pose
   approaching one junction carries the identical label, so a pose-level n overstates the sample by
   ~50×; and two events inside one clip share a map, a driver and a traffic culture, so the event is
   not the independent unit either. `taniteval.ci.episode_cluster_bootstrap(per_event, scene_id)`
   takes exactly this shape. On the winner: **100 admissible poses but `EFFECTIVE_N = 2` events.**
   ⇒ **a single scene cannot carry a strategic CI.** The 141 T1 scenes are the sample; at ~1–3
   scoreable events each the programme-level effective n is ~150–350.
4. **Report the confusion over the option set**, not a scalar accuracy — the strategic analogue of
   the manoeuvre confusion matrix that STREAM A is fixing.
5. **Never quote a strategic accuracy without `n_options` beside it.** A benchmark of mostly
   2-option junctions has a 50 % chance floor; the winner's junction 149 has a 25 % floor.

### The metrics

| metric | definition | scoreable when |
|---|---|---|
| `route_choice_accuracy` | model's chosen continuation == the one the ego took | `n_options ≥ 2` |
| `route_choice_confusion` | full option-set confusion, per branching factor | `n_options ≥ 2` |
| `route_class_accuracy` | LEFT/STRAIGHT/RIGHT/UTURN agreement, chance-corrected by the per-event option classes | `n_options ≥ 2` |
| `decision_lead_distance_m` | how far ahead the model's choice becomes stable and correct — *the* hierarchy signal: a strategic brain should commit before the tactical one can see the turn | `n_options ≥ 2` |
| `route_corridor_departure_rate` | already implemented; keep it, it works on single-option scenes too | always |
| `route_progress_rel` | already implemented | always |

**Implemented, not just specified**: `strategic_gt.score_strategic(events_by_scene, predictions)`
emits `route_choice_accuracy`, `route_class_accuracy`, `route_choice_confusion_gt_x_pred`,
`accuracy_by_branching_factor` and the `chance_floor`, each through
`taniteval.ci.episode_cluster_bootstrap` with the scene as the cluster. Its guards are driven with
input designed to make them fail: on a night-clip-shaped set (all single-option) it returns
**no accuracy at all** rather than a free 1.0, and a missing prediction scores **0** rather than
being dropped — dropping it would let an arm with no route head beat one that tries.
Tests: `test_score_strategic_*` in `stack/tests/test_xodr_junction_probe.py`.

`decision_lead_distance_m` is the one worth building the benchmark for: it is the only metric here
that a flat (non-hierarchical) policy cannot fake, because it asks *when* the choice was made, not
just whether it was right at the junction.

### Discrimination proved BEFORE the metric is quoted (`score_demo.py`)

Four synthetic arms scored against the **real** option sets of all 12 admissible surveyed scenes
(`results/score_demo.json`, estimator `episode_cluster_bootstrap`, cluster = scene):

| arm | route_choice_accuracy | 95 % CI | class acc |
|---|---|---|---|
| ORACLE | **1.0000** | [1.0000, 1.0000] | 1.0000 |
| RANDOM (uniform over the options) | 0.6667 | [0.3844, 0.9002] | 0.7333 |
| ALWAYS_STRAIGHTEST (what a flat model learns) | 0.5333 | [0.2500, 0.8125] | 0.6000 |
| NO_HEAD | 0.0000 | [0.0000, 0.0000] | 0.0000 |

Chance floor **0.4611**; **n = 15 events over 8 scenes** (4 of the 12 admissible scenes contribute
no scoreable event — every junction in them is single-option). **DISCRIMINATES: true** — ORACLE's
lower bound 1.0000 clears RANDOM's upper bound 0.9002.

⚠️ **And this is simultaneously the power analysis, which is the more useful result.** RANDOM's CI
is **0.516 wide**. A cluster bootstrap narrows roughly as 1/√n_scenes, so **a ±0.10 strategic
verdict needs ~100+ scenes with a scoreable branch.** That is precisely what the **141 T1 scenes**
are for. ⇒ **a shortlist cannot carry a strategic verdict, and neither can one hand-picked scene** —
including the winner. The winner is the scene to *render and drive*; the T1 set is the scene list to
*score on*.

### Known limits — stated per family with the reason and the n, as required

* **Two of 15 shortlisted scenes fail the self-consistency control** (`302c5c99` 98.21°,
  `d1a25a99` 179.86°) and are excluded. `302c5c99` re-enters one large junction three times, so the
  branch is genuinely ambiguous; `d1a25a99` has one event with a 180° flip. Refused, not published.
* `is_complete = false` on all four night-clip polygons and on most others: the labelled
  intersection area is clipped by the annotation region, so poses-inside is a **lower bound**.
* The xodr junction surface and the clipgt polygon are different objects; agreement is asserted on
  *location*, never by averaging the two counts.
* Junction 154 on the winner is truncated by the clip end — reported, never scored.

---

## ⚠️ Two traps worth propagating

**1. The OpenDRIVE reference line in this corpus is NOT the driven line.**
MEASURED on scene `7c72937c`, road 35: `laneOffset a = 10.495 m` at s=0, decaying to 0 by s=30.
The **reference line sweeps −112.18° → −71.67° (a 40.5° turn)** while the **lane centreline the car
actually drives runs dead straight at −112.43°**. Computing a branch angle from `planView` headings
gave **+51.49°** for a manoeuvre the ego drove at **+123.53°** and mislabelled it. Any route,
heading or curvature quantity must come from the *sampled lane centreline* (reference + laneOffset
+ inner lane widths), never from `ref_pose()[2]`.
Regression tests: `test_travel_heading_reads_the_driven_lane_not_the_reference_line`,
`test_travel_heading_diverges_from_the_reference_line_when_an_offset_exists`.

**2. Connecting-road centrelines overlap at a junction entry, so a nearest-lane snap picks the
wrong branch.** On `7c72937c` junction 149 the snap flip-flopped `36 → 46 → 47 → 48 → 59 → 60 → 44
→ 45 → 15 → 13`; the modal road was **15 (STRAIGHT)** while the ego actually drove **13 → 12** and
turned 163°. Resolve the branch **topologically** — the connecting road whose link lands on the
road the ego is on when it leaves — with polyline **coverage** as the independent cross-check
(the correct branch scores coverage 1.00 at 0.66 m mean; the next best scores 0.41 at 4.31 m).

---

## Evidence class

| claim | class |
|---|---|
| every number in (a), (b), (c) | **MEASURED (ours)** — artifacts listed in the manifest |
| "the night clip has no junction-scale decision" | **MEASURED**, from `map.xodr` option counts — was previously an assumption |
| "141 of 1607 scenes contain a real junction turn" | **MEASURED** over the full release, 0 errors |
| `decision_lead_distance_m` distinguishes hierarchical from flat policies | ⚠️ **HYPOTHESIS** — the metric is specified and computable, but has never been run against an arm |
| the roundabout scene's numbers | ⛔ **NOT ADMISSIBLE** — control C3 failed |
