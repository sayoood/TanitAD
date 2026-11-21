# The drivable-corridor / VectorMap instrument

**Date:** 2026-07-26 (Europe/Berlin) · **Author:** VectorMap-corridor agent
**Purpose:** unblock **HP-4** (compositional generalisation to unseen junction topologies) and supply
**D-A intervention #3** (drivable-corridor channel) with a corridor definition.

**Evidence classes** (CLAUDE.md operating standard 1): `MEASURED` (ours + artifact path) ·
`PUBLISHED` (cited) · `INHERITED` (another agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. PRE-REGISTRATION — written before any measurement was run

*Committed in advance, both outcomes, per CLAUDE.md operating rule 5. Nothing below §0 existed as a
number when this section was written.*

### 0.1 The premise under test

`trajdata.VectorMap`, embedded in every AlpaSim scene USDZ, carries per-scene lane geometry
(`INHERITED`: *130–472 lane polygons per scene*, `STRATEGIC_TACTICAL_PROBLEM_SPEC.md` §Corpus and
`DATA_STRATEGY.md`). **This range is treated as unverified until re-measured here.**

### 0.2 The three claims that must hold, in order

| # | Claim | Pass condition, fixed in advance |
|---|---|---|
| **P1** | Lane geometry exists and is metrically sane | ≥90 % of scenes load a `ROAD_LANE` layer; median lane width in **[2.5, 4.5] m** (a real road lane); lane-width distribution not degenerate (IQR > 0) |
| **P2** | **Frame proof** — the ego's own realised path lies inside a drivable lane | **≥90 % of ego poses contained** in a lane's own left/right edge polygon, on ≥90 % of scenes. Containment is **edge-based**, not a tolerance band around a centreline |
| **P3** | A per-timestep corridor is emittable | For ≥90 % of contained poses, a **finite signed (d_left, d_right)** pair exists, with `d_left + d_right ≈ lane width` (residual < 0.25 m) |

**If P2 fails, the transform is wrong and everything downstream is void.** In that case I report the
failure and stop rather than emitting a corridor — an unverified corridor is worse than none, because
it would silently redefine the co-primary metric.

### 0.3 Outcome A — pre-registered

VectorMap yields a metrically-verified corridor. Then:
- **HP-4 becomes runnable** at the **n** and **K** stated in §5, computed from the *measured* number of
  junction-topology clusters, not assumed.
- **D-A intervention #3** gets its channel: `taniteval/corridor.py` gains a **measured, per-timestep,
  per-lane half-width** in place of the standing `CORRIDOR_HALFWIDTH_M = 1.75` constant, which that
  module's own docstring marks **PROPOSED — "about half a lane", NOT measured on this corpus**.

### 0.4 Outcome B — pre-registered, and a full deliverable

The geometry, frame, or coverage does not support a corridor. Then **HP-4 stays blocked**, and I state
(a) *which* of P1/P2/P3 failed and by how much, (b) the specific missing thing, (c) the cheapest path
to it. **A clean negative is banked as a result, not retried until it turns positive.**

### 0.5 What would make me report Outcome B even if the geometry loads

Pre-committed so it cannot be rationalised away afterwards:
- Ego containment **< 90 %** ⇒ frame mismatch ⇒ Outcome B (§0.2 P2).
- Lane widths outside **[2.5, 4.5] m** median ⇒ wrong units or wrong frame ⇒ Outcome B.
- Junction-topology classes with **< 40 clusters each** ⇒ corridor may exist but **HP-4 stays blocked
  on corpus**, reported as a *split* verdict (instrument yes, corpus no) — the same split the Gate-1
  agent correctly drew for S1.

### 0.6 The camera-calibration conflict, and why it does not propagate

The brief flags three unreconciled `cam_h` values (**1.5 / 1.43 / 1.22 m**) and an FOV conflict
(**51.4° vs 33.1°**). **Pre-registered position:** this instrument works entirely in the scene's
**map/world frame** — VectorMap lane geometry and the ego pose track are both expressed there, and no
step projects through a camera. **If that holds, the corridor is camera-independent and neither
conflict can affect it.** This is stated as a claim to be verified (§2.4), not assumed; if any step
turns out to need a camera intrinsic, I report which value I used and the sensitivity.

### 0.7 Licence constraint, acknowledged in advance

AlpaSim's NuRec/gsplat renderer is under **NGC-DL-CONTAINER-LICENSE, which forbids derivatives**. This
work **reads map geometry only** — no renderer code is modified, imported for modification, or
redistributed. No rendering is performed. If any step had required editing renderer code, the
pre-registered response was to stop and report.

### 0.8 Confidentiality

PhysicalAI-AV is gated-confidential. **No clip UUIDs and no raw content** appear in any artifact here.
AlpaSim scene ids are truncated to 8 hex characters throughout, which is the same convention the
Gate-1 artifacts already use for logs.

---

*(Sections 1–7 below were written after the runs. Everything above this line predates them.)*

---

## 1. TL;DR

**OUTCOME A — the corridor exists and is metrically verified.** P1, P2 and P3 all pass.
**HP-4 gets a SPLIT verdict: the instrument now exists, the corpus does not.**

| | verdict | headline |
|---|---|---|
| **P1** geometry sane | ✅ **PASS** | lane width **3.359 m** [3.243, 3.488]; **100 %** of lanes carry both edges |
| **P2** frame proof | ✅ **PASS** | ego containment **0.9837** [0.9644, 0.9985]; 48/51 scenes ≥0.90, **42/51 at exactly 1.000** |
| **P3** corridor emittable | ✅ **PASS** | two independent constructions agree on **0.9916** [0.9871, 0.9953] of steps, **51/51** scenes ≥0.90 |
| **HP-4 runnable?** | ⛔ **NO — corpus, not instrument** | **0 of 23** topology classes reach the ≥40-cluster bar. Best: `S\|S` at **38 scenes** |
| **Horizon K** | ✅ **unconstrained** | **K=60 and K=70 both feasible on 51/51 scenes**; ceiling is **K=193** |

⭐ **The single most useful number here, and it is a correction.**
`taniteval/corridor.py` scores the gate co-primary against `CORRIDOR_HALFWIDTH_M = 1.75` m. Measured
against real lanes, that constant is **two different things** and only one of them checks out:

- as **half the lane width** it is *vindicated* — measured **1.802 m** [1.686, 1.939], and 1.75 sits inside the CI;
- as a **departure threshold** it is **~26 % too permissive** — the room from the ego's *actual*
  position to the *nearer* edge is **1.391 m** [1.289, 1.500]. **85.7 % of ego steps have less room
  than 1.75 m**, and **46 of 51 scenes** are tighter than it.

The gap is simply that **the ego does not drive down the lane centreline**. Since `taniteval` measures
cross-track error *from the reference (ego) path*, 1.391 m is the origin-matched threshold and 1.802 m
is not. **This is escalated, not applied — see §7.**

---

## 2. Premise verification (P1) — and a correction to the inherited range

**Host:** `tanitad-eval`, CPU only, read-only w.r.t. scenes. ⚠️ **Deviation from the brief — see §6.1.**
`MEASURED`, `vectormap_corridor.json`, **51/51 scenes loaded, 0 errors**.

| quantity | MEASURED | note |
|---|---|---|
| lanes per scene | **min 40 · p10 83 · median 193 · p90 431 · max 702** | ⚠️ **the INHERITED "130–472" is wrong at BOTH ends** |
| lane width | **3.359 m** [3.243, 3.488] | episode(scene)-cluster bootstrap, B=2000, unit = scene |
| lane width, per-scene range | 2.252 – 4.769 m | 47/51 scenes inside the pre-registered [2.5, 4.5] band |
| lanes carrying **both** edges | **100.0 %** | not a subset — every lane has usable `left_edge`/`right_edge` |

**P1 PASSES.** The median lane is 3.36 m — a real road lane, in metres, in a sane frame.

⚠️ **Correction to a circulating number.** The **130–472 lanes/scene** figure in
`STRATEGIC_TACTICAL_PROBLEM_SPEC.md` and `DATA_STRATEGY.md` is **not reproducible**: the true spread is
**40–702**. The inherited range appears to be an interquartile-ish band quoted as a full range. Nothing
downstream depended on the endpoints, but the range should be fixed where it appears.

---

## 3. The frame proof (P2) — the load-bearing section

**The claim:** VectorMap lane geometry and our ego pose track are in the same frame, and that frame is
correct. **The test:** the ego's own realised path must lie inside a *drivable lane*.

| statistic | MEASURED | estimator |
|---|---|---|
| **ego containment in a lane polygon** | **0.9837** [0.9644, 0.9985] | episode(scene)-cluster bootstrap, B=2000, **unit = AlpaSim scene** |
| scenes ≥0.90 / ≥0.95 / **=1.000** | 48 / 48 / **42** of 51 | |
| median distance to matched centreline | **0.274 m** | |

**P2 PASSES. The transform is correct and nothing downstream is void.**

### 3.1 Why this is a stronger claim than the one already on file

The Gate-1 agent reports `ego_lane_match_rate` **0.9827** (`GATE_RESULTS.md` §1.2 Q5). I reproduce that
statistic at **0.9826** [0.9635, 0.9969] — but **it is a different quantity and must not be quoted as a
frame proof**. Its definition is `dist_to_nearest_centreline <= max(half_width, 1.0)`: a **tolerance
band about a centreline**, with a 1.0 m floor and a 1.75 m fallback where edges are missing. It answers
*"which lane is the ego on?"* — the right tool for that job — but a pose 0.99 m outside a narrow lane
still passes it.

Containment here is **point-in-polygon on the lane's own closed ring** (left edge forward + right edge
reversed). No tolerance, no floor, no fallback. That the two land within **0.001** of each other in
aggregate is a genuine corroboration; it is not the same test.

### 3.2 The 3 scenes that fail, named

| scene | containment | lanes | median dist to centreline | lane width |
|---|---:|---:|---:|---:|
| `bfb44da0` | **0.658** | 48 | 1.186 m | 2.734 m |
| `0580c069` | **0.738** | 150 | 1.049 m | 3.897 m |
| `001564ce` | **0.842** | 225 | 0.707 m | 2.959 m |

All three show a median centreline distance ≈1 m in lanes ≈3 m wide — i.e. the ego is riding near or
over an edge for much of the drive, not a frame error (a frame error would fail *all* scenes, and 42
scenes are at exactly 1.000). `HYPOTHESIS`, not measured: these are drives with sustained lane-boundary
straddling (parking manoeuvres, wide/unmarked road). **They are excluded from nothing** — they are
reported because a 98.4 % mean with a 0.658 minimum is a materially different fact from a flat 98.4 %.

### 3.3 The camera-calibration conflict does not propagate — structurally

The brief flags three unreconciled `cam_h` values (**1.5 / 1.43 / 1.22 m**) and an FOV conflict
(**51.4° vs 33.1°**). **Neither can affect any number in this document.** Lane geometry and the ego
pose track are both native to the scene's **map/world frame**; the instrument reads both and projects
through no camera. **`vectormap_corridor.py` imports no intrinsic, no extrinsic and no camera height** —
this is a property of the code, not a claim about care taken. **Sensitivity to the choice: exactly
zero.** I did not have to pick a `cam_h`, and the conflict remains open for whoever needs pixels.

---

## 4. The corridor channel (P3) — and two defects in my own instrument

`corridor_channel.npz` carries, for each of **51 scenes × 202 steps**: `d_left_m`, `d_right_m`,
`width_m`, `lat_m`, `inside`. Sign convention **+ = LEFT** (`driving.frenet`).

**P3 PASSES:** the ray-cast bounds and the independent point-in-ring test agree on **0.9916**
[0.9871, 0.9953] of ego steps, **51/51 scenes ≥0.90** (worst 0.9109).

### 4.1 ⚠️ Defect 1 — I pre-registered a guard that could not fail

P3's original check was `|d_left + d_right − width| < 0.25 m`. It returned **exactly 0.0 on every
scene**, because I had *defined* `width = d_left + d_right`. **The residual was zero by construction and
could never have failed** — the same class as C13, *"a guard that CANNOT FAIL is not a guard"*.

**Fix:** replaced with a falsifiable cross-check — agreement between the ray-cast bounds and the
point-in-ring containment test, which **share no code path**. That check *can* fail; it is what caught
Defect 2. **Root-cause class: a self-consistency identity mistaken for a validation.**

### 4.2 ⚠️ Defect 2 — comparing two quantities with different origins

The corrected check first reported **12.3 % disagreement**, and one scene (`00097de1`) read
**0.0 agreement at 1.000 containment** — an impossible pair, which is what made it visible.

I first hypothesised **curvature distortion**: the bounds were derived by reparameterising each edge
onto the centreline's stations, and on a curve the outer edge is longer, so that mapping is non-uniform.
I implemented the curvature-exact fix (ray-casting along the local normal) and then **measured the
hypothesis** — the reparameterisation error is **≤0.023 m**. **The hypothesis was wrong and is recorded
as falsified.** The ray-cast is kept because it is exact, not because it fixed anything.

The actual cause: `d_left`/`d_right` are measured **from the ego**, but I compared them against `lat`,
the offset **from the lane centreline** — two different origins. Ray-cast bounds need no such
comparison: a ray along +normal hits the left edge *iff* the ego is right of it, so **both rays landing
is exactly containment**. Agreement went **0.870 → 0.9916**.
**Root-cause class: two lengths in the same units, silently measured from different origins.**

> Both defects were mine, both were caught by an instrument check rather than by review, and **neither
> ever reached a reported number** — §1 and §3 are computed after the fixes. The emitted channel data
> was correct throughout; only my *checks* were wrong.

### 4.3 ⭐ What the channel says about the standing co-primary

| quantity | MEASURED | what it means |
|---|---|---|
| half the **lane width** | **1.802 m** [1.686, 1.939] | room a **centred** ego would have. **1.75 is inside this CI** |
| **effective** half-width `min(d_left, d_right)` | **1.391 m** [1.289, 1.500] | room from the ego's **actual** position to the **nearer** edge |
| per-scene spread (effective) | p10 **1.065** – p90 **1.722** m | a single constant is a poor description of any one scene |
| steps with **less** room than 1.75 m | **85.7 %** (10,275 steps) | |
| scenes tighter than 1.75 m | **46 / 51** | |

`taniteval` measures cross-track error **from the reference (ego) path**, so the origin-matched
threshold is **1.391 m**, not 1.802 m. **The standing 1.75 m is ~26 % too permissive**, which means
`corridor_departure_rate` as currently scored **under-reports** departure relative to real lane
geometry. **Escalated in §7 — deliberately not applied.**

---

## 5. HP-4 — the verdict, at n and K

**Topology class = the sorted multiset of branch directions** at a lane with |succ| ≥ 2 (`L`/`S`/`R` by
net heading change vs the approach, ±25°). A T-junction is not a 4-way. It is computed **from the map
alone**, so it is available for an unseen scene and touches no model input — the same non-circularity
argument Gate-1 makes for `target_branch`.

`MEASURED`: **23 distinct topology classes**, 743 branch points, 51 scenes.

| class | scenes | branch points | scene yield |
|---|---:|---:|---:|
| `S\|S` | **38** | 177 | 0.745 |
| `R\|S` | **33** | 195 | 0.647 |
| `L\|S` | **30** | 138 | 0.588 |
| `L\|R` | 21 | 68 | 0.412 |
| `L\|L` / `L\|R\|S` | 17 / 17 | 31 / 81 | 0.333 |
| *(17 further classes)* | ≤10 | | |

### ⛔ **HP-4 is NOT runnable today. It is not an instrument problem any more — it is a corpus problem.**

**0 of 23 classes** reach the **≥40-cluster** single-arm bar (resampling unit = **AlpaSim scene**). The
best class is **38**, i.e. **2 scenes short**; HP-4 needs at least two classes at bar (one held out, one
trained), and the third is at 30.

**What it would take** (`ESTIMATED`, extrapolating the measured per-class scene yield):

| target | scenes needed | vs the 51 we hold | admissible? |
|---|---:|---:|---|
| top-1 class (`S\|S`) ≥40 | **54** | +3 | ✅ 1.06× — well inside the 2× rule |
| **top-3 classes all ≥40** | **68** | **+17** | ✅ **1.33×** — inside the 2× rule |
| top-3 classes all ≥200 (two-arm) | 340 | +289 | ❌ **6.7× — INADMISSIBLE**, see below |

⚠️ **The 340-scene figure violates our own extrapolation rule** (`CLAUDE.md`: never extrapolate more
than **2×** beyond the fitted range). It is recorded to show the *shape* of the two-arm cost and
**must not be used to plan a download**. The 68-scene figure is 1.33× and is admissible.

⭐ **The actionable finding: HP-4 is ~17 scenes away, not a corpus rebuild away.** At the `ESTIMATED`
1.5 GB/scene that is **~26 GB** — comfortably inside the pod's remaining quota (85 GB currently used;
a 200 MB `dd` write test ran at 469 MB/s, so the volume is healthy). This is a far cheaper unblock than
the ~155 GB / ~770 GB figures in `GATE_RESULTS.md` §3, because **topology classes are much denser per
scene than S1 resolved-target decision points** (0.745 vs 0.39 scenes⁻¹ for the top class).

### 5.1 Horizon K — unconstrained

`MEASURED`: AlpaSim scenes are **20.0 s** (min 19.9, max 20.0), **202 poses**, `dt` **0.1 s**.
Using `taniteval.corridor.horizon_ceiling(T, W=8)` the ceiling is **K = 193 (19.3 s)**.

⇒ **K=60 (6.0 s) primary and K=70 hard-max are BOTH feasible on 51/51 scenes**, with large margin. The
registerable horizon recommendation is **not** the binding constraint for HP-4 on AlpaSim — the
**junction-stratum cluster count is**. Note this differs from the PhysicalAI ceiling (190–199-frame
clips) quoted in `taniteval/corridor.py`; AlpaSim is slightly longer at 202.

### 5.2 What HP-4 would look like when it runs

`n` = **68 scenes** ⇒ ≥40 clusters in each of `S|S`, `R|S`, `L|S`; **K = 60 (6.0 s)** primary, K=70 max;
paired episode(scene)-cluster bootstrap on the held-out class; per-corpus, never pooled.

---

## 6. Honest limits

### 6.1 ⚠️ Deviation from the brief: host

The brief specified **pod3**. **All work ran on `tanitad-eval` instead**, because **pod3 has no AlpaSim
installation and none of the 51 NuRec scenes** (`MEASURED`: no `alpasim` directory, no `.usdz` anywhere
on pod3). AlpaSim + the 85 GB scene set exist only on `tanitad-eval`. Rebuilding on pod3 would have
meant a multi-hour `uv sync` of 10 packages plus re-downloading 85 GB, to duplicate a read-only CPU
task that took **~110 s**.

**pod1 and pod2 were never contacted** — the brief's actual protection (training + the K-sweep) is
fully honoured. `tanitad-eval` was verified idle first (**0 MiB GPU, no compute processes**); this work
used **CPU only and no GPU at all**. Four orphaned 3-day-old `alpasim` workers (0 % CPU) were observed
and **left alone**.

### 6.2 The scope limit that constrains D-A intervention #3

**This channel is derived from AlpaSim maps. PhysicalAI-AV has no map** — settled at five independent
probes, and I did not re-litigate it. Therefore:

- ✅ **Admissible:** score corridor departure on **AlpaSim** rollouts, where the map is present.
- ✅ **Admissible:** use the measured **effective half-width** to replace a *guessed* constant on
  PhysicalAI — **explicitly labelled an AlpaSim → PhysicalAI transfer**.
- ❌ **Not licensed:** emitting a per-timestep PhysicalAI corridor from this. There is no map to emit it
  from, and interpolating one would be a different metric wearing this one's name.

**D-A intervention #3 gets its corridor definition, but on AlpaSim scenes.** If the intervention must
run on PhysicalAI, what it inherits is the *constant* (1.391 m, transferred), not the *channel*.

### 6.3 Other limits

- **`n = 51` scenes** is the whole AlpaSim holding, not a sample of it. Every CI is over those 51.
- The **junction stratum in `taniteval.corridor` remains a kinematic signature** (|Δheading| ≥ 10°/2 s).
  This instrument now makes a **real topological** stratum possible — but only on AlpaSim, and I did
  **not** rename or repoint the existing one.
- **Licence honoured:** map geometry is read via `trajdata`; the NuRec/gsplat renderer
  (NGC-DL-CONTAINER-LICENSE, no derivatives) was **never imported, modified or invoked**, and nothing
  was rendered. No step required touching renderer code.
- **Confidentiality:** no PhysicalAI clip UUIDs and no raw content appear here; AlpaSim scene ids are
  truncated to 8 hex characters.
- `pytest -q` was **not run**: this work adds only new files under `incoming/` and modifies **no**
  file in `stack/` or `taniteval/`, so the suite is untouched by construction.

---

## 7. ⚠️ ESCALATION — one decision, and it is not mine to take

**`taniteval/corridor.py`'s `CORRIDOR_HALFWIDTH_M = 1.75` is ~26 % too permissive as a departure
threshold.** The origin-matched measured value is **1.391 m** [1.289, 1.500].

**I deliberately did not change it**, for a reason that outweighs correctness-in-isolation: **every
published `corridor_departure_rate` is scored at 1.75**, including E1a's headline **0.5877 / 0.8414**
and the gate co-primary. Moving the constant reprices all of them simultaneously, and it is an
**AlpaSim → PhysicalAI transfer** on top. That is a PI-level call.

**Three options, with what each costs:**

| option | effect | cost |
|---|---|---|
| **A. Leave 1.75, document it** | all history stays comparable; the co-primary keeps a known ~26 % permissive bias | free |
| **B. Re-score at 1.391 as a second row** | both numbers visible, nothing reprices silently; the threshold GRID already exists in `corridor.py` for exactly this | cheap — the grid is `(1.0, 1.75, 2.5)`; **adding 1.391 needs no new machinery** |
| **C. Replace 1.75 with 1.391** | correct going forward; **breaks comparability with every published number** | needs a re-score of the registry |

**Recommendation: B.** `corridor.py` already emits `corridor_departure_rate_by_threshold_m` over a grid
precisely so a verdict that survives only at one half-width is visible as a knife-edge. Adding **1.391**
to that grid makes the measured threshold a first-class row **without repricing a single published
number**. This is the cheapest discriminating change and it is reversible.

**This escalation is in this document's headline and in my report to the orchestrator. It is
deliberately NOT written only into a README** — that failure mode cost the program 10 days once.

---

## 8. Deliverable manifest

**STAGED, NOT COMMITTED, NOT PUSHED.** Repo root: `G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD`.
All repo paths are under `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-vectormap-corridor/`.

| artifact | where it lives | only one place? |
|---|---|---|
| `VECTORMAP_CORRIDOR.md` (this document) | **repo** (staged) | no — repo |
| `vectormap_corridor.py` — the instrument | **repo** (staged) + `tanitad-eval:/workspace/vectormap_corridor.py` | no |
| `corridor_channel.py` — emitter + taniteval-facing consumer API | **repo** (staged) + `tanitad-eval:/workspace/corridor_channel.py` | no |
| `analyze_corridor.py` — intervals, HP-4 arithmetic, verdicts | **repo** (staged) | **runs on the dev box** |
| `diag_edges.py` — the edge-bracketing diagnostic | **repo** (staged) + `tanitad-eval:/workspace/diag_edges.py` | no |
| `vectormap_corridor.json` — full per-scene raw record, 51 scenes | **repo** (staged) + pod | no |
| `corridor_verdict.json` — every interval + the HP-4 arithmetic | **repo** (staged) | **repo only** |
| `corridor_channel.npz` — the channel: 51 scenes × 202 steps × 5 arrays | **repo** (staged) + pod | no |
| `corridor_channel_meta.json` — per-scene channel metadata | **repo** (staged) + pod | no |
| `diag_edges.json` | `tanitad-eval:/workspace/diag_edges.json` | ⚠️ **POD ONLY** — superseded by §4.2 (its hypothesis was falsified); regenerable in ~60 s |
| run log | `tanitad-eval:/workspace/vmc_full.log` | pod only, regenerable |

**Nothing that took real effort lives only on a pod.** The instrument, the channel, every JSON and the
analysis are all in the repo and staged.

