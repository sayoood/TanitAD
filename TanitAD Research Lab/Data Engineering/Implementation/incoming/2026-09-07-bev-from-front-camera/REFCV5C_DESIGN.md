# refcv5c — the design (v2, CORRECTED after WP-A)

**Date** 2026-09-08 · **Owner** DataFlyWheel · **Status** DESIGN, corrected by measurement.
**PI direction:** *"a new version of refc, taking into account the proven assets of refcv5."*

---

## 0. ⛔ WHAT v1 OF THIS DESIGN GOT WRONG — read this before the rest

v1 gated everything behind a readout fix. **That premise was false, and it was my
error.**

| v1 claimed | MEASURED (WP-A) |
|---|---|
| "pooled to **4 readout columns**, 30°/bin, ±8 m at 30 m" | ⛔ **a v6/v7 fact quoted into a REF-C design.** REF-C runs `--image-hw 256 640` at stride 32 ⇒ **8×20 = 6.0°/column**, and `refc.py:2112` shows the anchor decoder **already cross-attends its 160 tokens** |
| "WP-A is the prerequisite" | ⛔ **WP-D is.** The address space is sufficient; the map it would index does not yet contain agents |
| "indexing into a 4-column map is indexing into almost nothing" | ⚠️ **over-claimed** (Master Mind withdrew it too): 4×4 still scores **5.0× its marginal control**. Supported form: **"loses about two-thirds of the available localisation."** |
| dense BEV assumed | ⭐ **SPARSE wins** — see §3 |

⭐ **The `fov_census` numbers themselves were fine. The error was the LAYER they
were attached to** — the same true-quantity-outside-its-scope defect this
programme has now logged at every layer it has looked at.

---

## 1. The thesis, unchanged and still the point

**refcv5-v2's diffusion planner sees the scene only through a `d_ctx = 64` token
added to a condition embedding.** DiffusionDrive — the paper REF-C derives from —
couples planner to perception **twice**, and we have only the second coupling.

MEASURED from the banked primary (`Library/papers/2411.15139`, quoted not recalled):
> **(1)** *"deformable spatial cross-attention to interact with BEV or PV features
> **based on the trajectory coordinates**"* · **(2)** *"cross-attention between the
> trajectory features and the **agent/map queries**"*
> Their ablation: *"spatial cross-attention is **vital** for accurate planning."*

⭐ **THE MECHANISM IS AN INDEX, NOT A SHARED LATENT.** The trajectory's own
waypoints are the query positions — each candidate path reads the feature map
**where it would drive**. Planner and perception share no embedding; **the
geometry is the address.** That is why the paper could run it on Transfuser with
BEV-only spatial CA, agent CA, **and no map** — ⭐ **published precedent for
exactly our configuration**, not an extrapolation to it.

---

## 2. ⛔ THE REAL BLOCKER — the map is empty, not coarse

MEASURED (WP-A, Master Mind, dev-box only):

* **Oracle ceiling** — prices the READOUT, since the input is built from the
  target: 16×40 **AP 0.4713 / 1.372 m** vs 4×4 **0.1583 / 2.669 m** ⇒ 2.98× AP.
  Every rung separated; replicate noise floor **≤0.0122 AP** against a 0.313 gap.
  ⭐ All-zero arm reads **0.01634 against a base rate of 0.016198** — the
  no-information value **exactly**, IoU and F1 identically zero.
* ⛔ **Real trunk — none of it transfers.** Fit AP 0.2049 → 0.0581 under pooling,
  but **test AP 0.027–0.034 against a 0.0325 marginal control**; raw pixels
  strictly below; `shuffled` indistinguishable from `pos_only`.

⇒ **The trunk carries addressable content it cannot generalise.** A coupling
built today would index into an empty room. **WP-D comes first.**

---

## 3. ⭐ DENSE vs SPARSE — ANSWERED, and it favours sparse

The token grid **is itself a lossy address space**. Under a *perfect* front-end
the full 16×40 reaches only **AP 0.471**, because a median of **4 BEV cells
(max 313)** share one token cell and only **242 of 640** token cells receive any
ground-plane cell. ⇒ a dense BEV indexed off it inherits a **quantisation floor
no training removes**, worst in the far field.

⭐ **Sparse pays no quantisation floor, and its seam already exists.**
`refc_agents.slot_features` passes continuous metric range and bearing — and its
original justification now reads as the right call for a second reason:
*"a token that hides range and bearing inside a learned embedding makes
cross-attention re-derive them from scratch."* It is simply gated off behind
`--agents off`.

⇒ **refcv5c couples through the SPARSE seam, not a dense BEV raster.** The dense
raster stays what it already is: a **probe** (P8) that measures whether the trunk
carries the scene.

---

## 4. Work packages, in the corrected order

| | package | state |
|---|---|---|
| **WP-D** ⭐ | BEV occupancy as an **auxiliary loss** — give the trunk a reason to carry generalisable agent content | ⛔ **PREREQUISITE.** Target exists (`bev_raster`, `[120,64]`, 60 m × ±16 m, 0.5 m); data local (145 parquets, 77 MB) |
| **WP-C** | agent seam ON (`--agents head`) at 108 M / 40 k | ⚠️ tiny-rig gate FAILED at 17 M / 500 steps; the shuffled arm relocated the cost to **auxiliary-task capacity competition on that rig**. ⛔ Whether it vanishes at scale is **UNTESTED** — a hypothesis, not a finding. **PI call (queue item 9)** |
| **WP-B** | waypoint-indexed cross-attention into the **sparse** agent tokens | after D and C — it is the coupling the paper calls vital |
| **WP-A** | ~~readout exposure~~ | ⛔ **CANCELLED.** REF-C is 8×20 and already cross-attends 160 tokens |
| **WP-E** | `speed_max_input` + pinned bucket ladder | ✅ shipped |
| **WP-F** | OUT: strategic vocabulary (6/15 tokens), map queries (no map), goal point (diagnostic-only) | named, not forgotten |

⭐ **INHERITED FROM refcv5 — the proven assets, carried not rebuilt:** P14 emitted-fan
ranking · 22-token tactical vocabulary · DDIM control-space sampler with its two
refusals · v0-conditioned anchors (n=117, `alat`) · P12 `a_star` ceiling fix ·
P5 decoder refinement · `--sel-refined --sel-score-emitted` · ego dropout 0.5.

---

## 5. ⛔ WHAT THE LABEL SIDE OWES, and its honest state

1. **The raster** — `[120,64]`, as held. ✅
2. **The occupancy base rate STAMPED IN the artifact** — **1.16 % of cells,
   median 0.45 %**. ⛔ Because an **all-zero predictor scores 98.84 %**, and the
   probe confirmed all-zero reads the base rate *exactly*. **Any consumer
   reporting accuracy has already been fooled**, so the number must travel with
   the data. ✅ measured, ⚠️ not yet stamped.
3. ⚠️ **THE THIRD VISIBILITY STATE — 2 of 3 MECHANISMS WORKING, NOT SHIPPED.**
   * ✅ **out-of-field** (`fov_mask`) and ✅ **beyond-range** (R_MAX = **178.5 m**,
     the p99 of 292,718 agent boxes — *a stated rule, not a fact*: 1 % of real
     agents sit beyond it and are treated as unobservable rather than free).
     Both empty-scene controls pass exactly.
   * ⛔ **OCCLUSION FAILS ITS CONTROL AND IS NOT SHIPPED.** Diagnosed twice:
     azimuth-binned ray-casting over a **Cartesian** grid leaks, because a cell's
     angular extent shrinks with range (row 0 spans ±89.1°, row 100 ±17.4°), so a
     discrete wall does not fill contiguous bins and rays pass between its cells.
     **Fix: shadow each occupied cell's SUBTENDED WEDGE** (`2·atan(half_cell/r)`,
     widened one bin), not its centre.
   * ⇒ **the 11.17 % figure is a LOWER BOUND.** It counts only out-of-field and
     beyond-range. Working occlusion makes it larger.
   ⚠️ And a near-field caveat that survives any fix: inside ~5 m one Cartesian row
   spans ~178° of azimuth, so **a consumer must not read near-field EMPTY as a
   cleared lateral corridor.**

---

## 6. Controls — the arm is inadmissible without them

`shuffled` seam (full degradation) · **ego-only floor** · raw-pixel floor ·
**constant/all-zero (must read the no-information value exactly — it already
does)** · **replicate**, against the rig's measured **14.3 %** false-positive rate
on `separated` · an `--agents off` twin so WP-C and WP-B are never read as one lever.
⚠️ Four metric families per arm with their `n`; **headway, time-gap and TTC are
where WP-C moved**, so they are where WP-B must be read.

## 7. What this design does not claim

* ⛔ Not that coupling (1) will help **here**. The ablation is PUBLISHED evidence
  on their stack (88.1 PDMS, NAVSIM, three cropped forward cams). We have one
  120° camera and no map.
* ⛔ Not that WP-C's exclusion will reverse at scale. **Untested.**
* ⚠️ Not that the BEV probe is on the path to better driving. It measures whether
  the trunk carries the scene; **WP-D promotes it to a trainer only because
  §2 showed the trunk does not.**


---

## ⛔ CROSS-STREAM CORRECTION (Master Mind, 2026-09-08) — one cited number is RETRACTED

⛔ **§ "All-zero arm reads 0.01634 against a base rate of 0.016198 — the no-information value
exactly" is RETRACTED** (`R-2026-09-07-ap-ties`, commit `405c149`). MEASURED from the banked
artifact: **0.016340211 vs 0.016197555 = +0.881 %**, not equal. **Mechanism:** a naive per-sample
AP **breaks ties by array order**, so a CONSTANT score is scored as though it had ranked the
positives first; an all-zero predictor is a constant score. Found by WP-D's own constant control
reading **+22.8 %** above its base rate — the no-information control scoring *higher* than the
value it defines. Root-cause class: **a control biased by the same ESTIMATOR it validates**.
⭐ **YOUR CONCLUSION IS UNAFFECTED and the ladder STANDS** — a 0.00014 bias is nothing against
0.4713 → 0.1583, and it applies equally to every arm. What is withdrawn is only the word
*"exactly"*, i.e. the claim that the control read its known value. ⚠️ But any FUTURE claim resting
on a **sub-1 % AP margin** — including the real-trunk column, where arms sit at 0.027–0.034 against
a 0.0325 marginal control — must use the tie-corrected metric.

## ✅ YOUR OCCLUSION WARNING DOES NOT REACH THE RUNNING WP-D ARM — it is POLAR, not Cartesian

You wrote: *"if WP-D starts before then, its negatives are two-state and its AP is optimistic."*
**WP-D started 2026-09-07 21:48Z and is training on Thor now.** Its negatives are **THREE-state**.

⭐ **Your diagnosis is right and it is a CARTESIAN-grid defect.** Azimuth-binned ray-casting over a
Cartesian grid leaks precisely because a cell's angular extent shrinks with range (your row 0
±89.1° vs row 100 ±17.4°), so a discrete wall does not fill contiguous bins. **WP-D's grid is
POLAR** — `PolarBEVSpec(n_az=20, n_rng=24)` — so **each column IS an azimuth bin by construction**;
`shadow_mask` walks one column at a time (`shadow[i1+1:, j] = True` beyond the nearest occupied
run) and **there is no angular-extent mismatch and no gap for a ray to pass between cells.** Your
subtended-wedge fix (`2·atan(half_cell/r)`) is the correct fix *for a Cartesian grid* and is not
needed here.
⭐ Note the shape: **polar was chosen for a different stated reason** — WP-A's quantisation floor
(median 4 BEV cells, max 313, per token cell) — and turns out to be right for a second reason
neither package anticipated. That is exactly what you observed about `slot_features`.

⚠️ **BOTH masks are LOWER BOUNDS, and WP-D says so in its own docstring**: it is 2-D (the join
drops `z`, so a low `stroller` shadows a `bus`), the shadow starts at a CELL boundary rather than
the true ray-exit range, and **only labelled agents occlude** — buildings, walls and vegetation
are not in `obstacle.offline` (10 classes, all dynamic agents), so a real urban scene is
**under-shadowed**. Same direction as your caveat, different cause. MEASURED on all 26,394
labelled frames / 905,512 boxes: **17.656 % of cells occluded, and 27.958 % of GT-OCCUPIED cells**
— i.e. a two-state target would assert that **more than a quarter of the agents in the grid are
free road**, which is why `--bev-aux-occlusion none` ships as the *named regression arm* rather
than the default.
✅ Your near-field caveat is ACCEPTED and is not covered by the polar grid: inside ~5 m one row
spans a very wide azimuth, so **a consumer must never read near-field EMPTY as a cleared lateral
corridor.** WP-D carries an `r_min_m` near-band IGNORE for the vertical-FOV part of this; the
lateral-corridor reading is a CONSUMER rule and belongs in whatever uses the map.

## ⭐ WP-A RAN AND DELIVERED — what was cancelled is a readout CHANGE, not the package

Minor but worth pinning so the register stays readable: `E-READOUT-CEILING-1` is **DONE** and
committed (`cb84c07`). The oracle ladder, the axis attribution, the replicate floor and the
real-trunk dissociation you cite as evidence for inverting the order **are WP-A's output**. What
is correctly cancelled is *changing the readout* — there is no four-column readout in REF-C to
change. Same distinction as *"rollable"* vs *"trained"*.

## What the label side owes, beyond items 1–3

1. ⭐ **The B1 EVAL join** (~10 MB) is the highest-value item and it is BLOCKING in-train eval for
   the WP-D panel. The trainer-side fix is already committed (`7dfb182`). ⚠️ Ship it as a **DATA
   file only** — ⛔ do NOT ship code to Thor before ~15:30 Berlin, or D1/D2 relaunch on different
   code than D0 ran.
2. **Item 3's wedge fix is still worth landing** for the Cartesian consumers (`bev_raster`, the P8
   probe), even though WP-D does not need it. Its absence is not blocking the running arm.
3. **`z` in the agent join.** WP-D's occlusion is 2-D *because the join drops `z`* — that is the
   single change that would turn both masks from lower bounds into estimates.
4. ⛔ **Not needed:** a dense Cartesian raster as a *component*. You settled that as SPARSE and
   `slot_features`; WP-D's polar target is a TRAINING-ONLY supervision signal, not a component,
   and it is removability-proven (136/136 shared params bit-identical, planner output identical
   but for one extra key).
