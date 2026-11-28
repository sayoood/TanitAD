# PRE-REGISTRATION — Stream E, SOTA_SCAN §11 items 1–3 (the three 0-GPU items)

**Written 2026-08-03 BEFORE any number below was computed.** Every branch is fixed here, including
the band the source scan left open and the INSTRUMENT-FAIL branch that a prereg without one cannot
adjudicate. Estimator everywhere: **paired episode-cluster bootstrap** over the val episodes
(`taniteval/taniteval/ci.py:261`), B = 2000, seed 0. ⛔ `overlapping_holdout_se` is refused.

Source: `Research/2026-08-03-sota-scan/SOTA_SCAN.md` §11 rows 1–3.

---

## Item 1 — `obstacle.offline` lead-agent ingest (SOTA §11 row 1)

**Standing before I start:** a sibling stream published
`Research/2026-08-03-longitudinal-distance-keeping.md` earlier today claiming this item is built and
admitted. **Duplicating it would be waste; taking its word for it would be `INHERITED` on a claim
that decides a gate.** So my registered action is *verification*, not re-implementation.

| branch | criterion | verdict I will write |
|---|---|---|
| **V-PASS** | the landed instrument imports, its tests pass, and its published D-LEAD-1 numbers are reproduced **bit-for-bit from the artifact JSON** | item 1 = DONE BY SIBLING, independently verified; my contribution is the verification + the unclosed half |
| **V-FAIL** | tests fail, or the note's numbers disagree with the artifact | raise as an instrument defect, quote both numbers, do not use the family |
| **V-GAP** | (expected, and it is named in their own §6) the eval path still reports the family UNAVAILABLE because `win["lead"]` is not built for the 40 val episodes | measure the size of the gap on the **actual banked val windows** and state it, rather than repeat it |

⛔ I will not edit their files. Any fix I find is escalated, not patched into a sibling's stream.

---

## Item 2 — progress-ratio, four families over the banked dumps, and open→closed predictivity (SOTA §11 row 2)

### 2a. The metric

`progress_ratio` = **along-track distance the arm covers ÷ along-track distance the human covers**,
per window, on the same waypoint grid the ADE uses. `progress_error = |1 − progress_ratio|`.
Rationale (PUBLISHED): Ego Progress alone is the strongest single closed-loop predictor in
[2605.00066](https://arxiv.org/html/2605.00066v1) (ρ = 0.83 vs No-Collision 0.45), and it is a
LONGITUDINAL quantity — the family that carries 88.7 % of our oracle gap.

### 2b. ⛔ Negative control FIRST — the instrument must discriminate before it is quoted

**D-PROG-1.** GT-as-arm (ratio ≡ 1 by construction) vs hold-`v0` CV, paired episode-cluster
bootstrap on `progress_error`.

* **PASS** — CI on Δ excludes 0 with the correct sign (CV worse) ⇒ admissible.
* **FAIL** — CI spans 0 ⇒ the gauge cannot move on our corpus; report NOT-APPLICABLE with its n and
  **stop**, do not score arms with it.

**D-PROG-2 (self-consistency, mandatory per the operating standard).** `progress_ratio` is computed
twice by two independent code paths — the module under test, and a from-scratch numpy reducer in the
runner. Agreement required to **1e-6**. A mismatch is **INSTRUMENT-FAIL**: no verdict is issued.
*(This control exists because a component-vs-family check once caught a 5347× curvature inflation in
an agent's own reducer.)*

### 2c. The registered test the scan asked for — and the band it left open

Kendall τ between the arm ranking by `ade_0_2s` and by `progress_error`, over the banked arms.

| τ | scan's text | my registered verdict |
|---|---|---|
| **τ < 0.7** | "pass" | progress-ratio is **non-redundant**; promote to gate emitter |
| **0.7 ≤ τ ≤ 0.9** | *unspecified — this is the gap I am closing in advance* | **INDETERMINATE on ranking alone.** The verdict is then decided **solely** by test 2d, which is the question the ranking is a proxy for. No promotion on 2c alone. |
| **τ > 0.9** | "fail" | progress adds nothing on our corpus; say so publicly instead of importing the field's framing |

⚠️ Registered in advance: **an arm-level τ over ~24 arms is a weak instrument** (the arms are not
independent — many are checkpoints of one lineage). Its result is reported, but it **cannot by itself
promote a metric**; 2d is the decision-grade test.

### 2d. ⭐ The decision-grade test — within-model, per-window, on OUR estimator

2605.00066 correlates **8 published scalars** across two benchmarks. We can do better on our own
data: three arms have **per-window open-loop AND per-window closed-loop trajectories on the same 881
windows / 40 episodes** (`…/2026-07-26-closedloop-artifact-rerun/raw_windows/clwin_*.pt`). So the
question "does the open-loop metric predict closed-loop failure" is askable **within one model, at
n = hundreds of windows, with an episode-cluster bootstrap on the correlation itself.**

For each arm and each open-loop metric `m` (ADE@2s, progress_error, speed_bias, along_bias,
cross-track, heading, curvature, yaw-rate), compute Spearman ρ(m, closed-loop ADE@2s) over windows,
with an **episode-cluster bootstrap CI on ρ**.

* **PASS for progress** — ρ(progress_error) CI excludes 0 **and** its point estimate exceeds
  ρ(ADE@2s) on ≥ 2 of the 3 arms ⇒ progress-ratio earns its place as a gate emitter, on our own
  evidence rather than on an imported n = 8.
* **PASS for ADE (the inconvenient outcome, committed in advance)** — ρ(ADE@2s) CI excludes 0 and is
  the **largest** of the metrics ⇒ **on our corpus ADE DOES predict closed-loop failure at the window
  level**, 2605.00066's headline does not transfer to a within-model reading, and I will write that
  as the result even though it contradicts the direction the scan recommends.
* **BOTH FAIL** — no open-loop metric's ρ CI excludes 0 ⇒ open-loop scoring is uninformative about
  closed-loop on our corpus; that is the strongest possible statement of the four-families case and
  must be reported as such.
* **INSTRUMENT-FAIL** — the closed-loop and open-loop window sets cannot be aligned window-for-window
  (`eid` mismatch, different n), or fewer than 10 episodes survive ⇒ no verdict.

⚠️ Registered in advance: window-level ρ and method-level ρ are **different quantities**. A positive
window-level ρ does **not** refute 2605.00066's method-level ρ = −0.36, and I will not claim it does.
What it does decide is **our** gate design, which is the only thing this measurement is for.

---

## Item 3 — ego-frame lane-graph raster from `map.xodr` (SOTA §11 row 3)

Render a coarse ego-frame BEV raster of the OpenDRIVE lane graph
(`Research/2026-08-02-nurec-xodr-map/`) — the conditioning surface Cosmos-Dreams consumes and the
strategic topology our hierarchy has never had.

### 3a. ⛔ Negative control FIRST

**D-MAP-1.** The raster must **discriminate**: a readout that sees only the raster must beat the
majority baseline at predicting the ego's own future route direction (left / straight / right at the
3 s horizon), on held-out poses.

* **PASS** — raster-only accuracy exceeds the majority-class rate by > 0.03 with an interval
  excluding the baseline ⇒ the raster carries strategic information and is worth wiring.
* **FAIL** — at or below majority ⇒ the raster as rendered is not informative; report the refutation
  and the reason (resolution, range, or the lane graph itself), and do **not** propose the trained
  probe.

### 3b. ⚠️ The estimator limit, registered in advance so it cannot be quietly ignored

Only **one** scene with a `map.xodr` is on this disk. **One scene = one cluster**, and an
episode-cluster bootstrap over a single cluster is not a valid interval. Therefore:

* if I obtain **≥ 5 scenes** (extraction from the NuRec bundles is CPU-only), the estimator is the
  **scene-cluster bootstrap** and the number is decision-grade;
* if I obtain **< 5 scenes**, I report a **contiguous-block bootstrap over disjoint time blocks**,
  **explicitly labelled as NOT the programme's decision-grade estimator**, and the finding is
  DIRECTIONAL only. It may not promote anything to a trained config.

### 3c. What this item may NOT conclude

⛔ It may not conclude anything about the strategic *brain*. It measures whether the **raster**
carries route information, not whether **our model** would use it. The trained probe is SOTA §11
row 4's ~1 GPU-h and is out of scope here.
⛔ `<speed max>` in this xodr must not be used as a target-speed label — the sibling probe MEASURED
it as mislabelled mph-for-km/h **and** inconsistent with the observed driving (their §6).

---

## Hosts, cost, and what would make this run invalid

Dev-box CPU + (for item 3 scene extraction only) **tanitad-thor CPU**. **0 GPU.** No training pod is
touched. If any number below turns out to have been computed on a training host, the run is void.

*Author: Stream E (PI priority 11). This file is written before the runners execute; the results file
is separate and may not edit this one.*
