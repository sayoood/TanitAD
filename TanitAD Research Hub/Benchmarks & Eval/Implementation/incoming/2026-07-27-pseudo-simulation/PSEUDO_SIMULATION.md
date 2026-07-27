# Pseudo-simulation — converting TanitAD's closed loop from EXTRAPOLATION to MEASUREMENT

**Date:** 2026-07-27 (Europe/Berlin) · **Stream:** Benchmarks & Eval · **Branch:** `agent/benchmarks-eval-20260721`
**Host:** `tanitad-eval` only (GPU idle before use; 0 MiB / 0 % at start). ⛔ pod1 / pod2 / pod3 untouched.
**Implements:** Option 1 of `…/Research/2026-07-27-closedloop-eval-without-renderer/CLOSEDLOOP_EVAL_RESEARCH.md` §8.
**Answers:** its §7.4, pre-registered there with both outcomes committed before this run existed.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited document, with source depth) ·
`INHERITED` (another agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

---

## 0. Headline

| # | Result | Class · tier |
|:--:|---|---|
| **1** | ⛔⛔ **THE LATERAL GRID AXIS IS DEAD, and it is dead in closed form.** The flat-road warp's relative displacement error is **exactly `height_above_road / h_cam`** — independent of depth, of `\|dlat\|` and of focal length (max deviation from the closed form over the whole grid: **0.0**). At the camera height (1.50 m) it is **100 %**; **above** it the applied displacement is **SIGN-INVERTED**. **Outcome L-BAD.** The grid ships **heading-only on the warped axis**. | `MEASURED` **tier 1** (exact, deterministic, model-free) |
| **2** | ⭐ **A car's roof moves 1.18 px when it should move 35.47 px.** At `\|dlat\| = 2.0 m`, depth 15 m: road surface −35.47 px ✅, sedan roof −1.18 px (**3.3 %** of the truth), SUV roof **+8.28 px** (wrong way), truck roof **+59.11 px**, building 2nd floor **+141.87 px** in a 256-px frame. | `MEASURED` **tier 1** |
| **3** | ⭐ **Exactly 50.0 % of the frame has no ground-plane preimage at all** (pitch 0, `c_y` = 128, 256 rows). The flat-road model cannot represent the upper half of a driving frame *even in principle*. At `\|dlat\| = 2.0 m` only **28.30 %** of in-frame scene points meet the pre-registered `rel_err < 0.25` bar against a required **95 %**. | `MEASURED` **tier 1** |
| **4** | ✅ **The identical test PASSES on the yaw arm at max error 0.0 px, so it is not vacuous.** Yaw `max\|ΔH\|` over 30 (h_cam, pitch) conditions = **0.0** (P1's C-GEO reproduced through our code path); lateral = **109.61**. | `MEASURED` **tier 1** |
| **5** | ⚠️⚠️ **The round-trip criterion §7.4 also offered is VACUOUS, and I refuse it rather than report its pass.** The ground-plane homography **composes exactly** (`n · t = 0` kills the second-order term), so `H(d)H(−d) = I` to machine precision — **1.4e-14 on yaw and exactly 0.0 on lateral**. A criterion that gives the provably-wrong arm a *perfect* score cannot adjudicate. | `MEASURED` **tier 1** |
| **6** | ⭐⭐ **The harness exists, in `taniteval/`, and its 0 %-out-of-envelope assertion is a HARD FAILURE that runs before any checkpoint is loaded.** `taniteval/pseudosim.py` + 21 tests. Measured on the shipped grid: `frac_steps_any = 0.0`, `frac_windows_any_step_out_of_envelope = 0.0`, verdict class **`MEASUREMENT`**. | `MEASURED` **tier 1** |
| **7** | ⚠️ **The first metric I built was gameable by standing still, and the smoke caught it.** With a `v0 × horizon` denominator, `recovery` scored the **BLIND** arm **+0.597 ABOVE** the sighted one — a planner that barely moves has a small cross-track error and was being paid for it. Denominator replaced by the plan's **own along-track distance**; a stopped plan is now **undefined, never 1.0**. Regression-tested. | `MEASURED` **tier 1** |
| **8** | ⛔ **Collision and TTC are NOT EMITTED, and no constant is substituted.** The 40-episode val cache carries `{frames_u8, actions, poses, maneuvers, episode_id}` and no cuboids; the cached `episode_id` is `int.from_bytes(clip_id[:4])` and **collides — 242 `clip_index` rows map onto the 40 val ids**. ⇒ the composite is named **`PSS_recovery_progress`** and explicitly **not** a Driving Score. | `MEASURED` **tier 1** |
| **9** | ⚠️ **Every result records `traffic_mode: log_replay_nonreactive`.** `trafficsim` disabled ⇒ `skip: true` ⇒ literal replay. This was nowhere on record before 2026-07-27; it is now in every node the harness emits. | `MEASURED` (harness) + `INHERITED` (the disclosure) |

### 0.1 The verdict in one sentence

**Pre-registered outcome A holds on the axis that survives and outcome L-BAD holds on the axis that does
not: a heading × longitudinal grid inside our measured envelope evaluates at 0 % out-of-envelope and
separates arms, so the EXTRAPOLATION label comes off — but the lateral axis is refused in code, not just
in prose, and we ship a protocol NARROWER than NAVSIM v2's and say so.**

### 0.2 My headline's tier, stated as required

The **fidelity verdict (results 1–5) is tier 1**: closed-form, deterministic, model-free, reproducible on
a CPU in seconds with no corpus and no checkpoint. The **arm scores (§6) are tier 2**: they depend on a
v4 checkpoint, an **ORACLE goal** (an upper bound, not deployable), and a **non-reactive** traffic mode.
Nothing in §6 may be quoted without all three qualifiers.

---

## 1. PRE-REGISTRATION — written before the measurement, not rewritten after

The research doc §7.4 committed both outcomes before this task began; this section adds the criterion
and the C13 self-gate, and is frozen.

### 1.1 The two pre-registered outcomes (verbatim from §7.4)

| outcome | what we write |
|---|---|
| **L-OK** | the pseudo-sim grid is **2-D** (lateral × heading), matching NAVSIM v2's axes. Full adoption of the protocol. |
| **L-BAD** | the grid is **heading-only + longitudinal-by-resampling**. We publish a *narrower* protocol than NAVSIM v2 and **say so**, rather than shipping an unvalidated lateral axis. A publishable, decision-relevant negative. |

> *"No stretching a marginal L-BAD into L-OK."*

### 1.2 The criterion, fixed before any number was produced

Pseudo-simulation's claim is that the perturbed observation **is what the camera would have seen at that
pose**. The quantity that decides it is therefore the **relative displacement error** of the synthesised
motion field:

```
R(x) = | u_applied(x) − u_true(x) |  /  | u_true(x) |
```

* **L-OK** iff `R < R_MAX = 0.25` on at least `FRAC_MIN = 95 %` of the frame, at the grid's maximum `|dlat|`.
* **L-BAD** otherwise.

Both constants are `PROPOSED` and both are published in the artifact
(`artifacts/lat_warp_fidelity.json → _criterion_pre_registered`).

### 1.3 ⚠️ The C13 gate applied to my own method

*A guard that cannot fail is not a guard.* Pre-committed controls, all of which fired:

| risk | control | did it fire? |
|---|---|:--:|
| The fidelity test is vacuous (returns FAIL for everything) | Run the **identical code path** on the **yaw** arm as a positive control. It must PASS. | ✅ yaw max error **0.0 px**, 100 % of points under `R_MAX` |
| The criterion is unfalsifiable rhetoric | State, in closed form, **the value of the world that would make it PASS**. | ✅ published: every scene point within **0.375 m** of the road surface |
| The *other* offered criterion (round-trip) silently passes a broken warp | Measure it too, and **refuse it if it cannot discriminate**. | ✅ round-trip is **0.0 on the lateral arm** — refused (§3) |
| The envelope assertion is decoration | A test that feeds a grid **0.5°** outside `ENV_YAW_MAX` and **requires** `EnvelopeViolation`. | ✅ `test_envelope_assertion_FAILS_just_outside_the_envelope` |
| A metric with no dynamic range ships anyway | Measure each component's range; **refuse to emit** a composite when none is admissible. | ✅ `VacuousMetric`; and §5.3's saturation finding |
| A metric that can be gamed | Adversarial input: a plan that does not move. | ✅ **it caught a real defect** — see §5.2 |

### 1.4 What I predicted, so I can be wrong on the record

`HYPOTHESIS`, committed before running: **P-1** the lateral arm would degrade *gradually* with `|dlat|`, so
a small lateral axis (≤ 0.5 m) would survive. **P-1 is FALSE and I record it as such**: the relative error
is **independent of `|dlat|`** (`rel_err = a / h_cam`), so shrinking the axis does **not** buy fidelity —
it only shrinks a perturbation that was never faithful. There is no small-`dlat` refuge.

---

## 2. ⭐⭐ THE LATERAL-FIDELITY VERDICT — L-BAD, in closed form

**Artifact:** `artifacts/lat_warp_fidelity.json` · **Script:** `scripts/lat_warp_fidelity.py`
(CPU-only, deterministic, ~2 s, **no model, no corpus, no GPU, no pod**).
The warp is **imported** from `taniteval.clhorizon.sampling_homography` — the same function
`corridor_rollout` calls — never re-implemented.

### 2.1 A0 — does the arm depend on the ground plane at all?

P1's C-GEO test, run on **both** arms over 30 conditions (`h_cam ∈ {1.2, 1.35, 1.5, 1.65, 1.8}` ×
`pitch ∈ {−4, −2, 0, 2, 4, 6}°`):

| arm | `max abs ΔH` over all 30 conditions | depends on the ground plane? |
|---|---:|:--:|
| **yaw** | **0.000000** | **no** — `H = K R K⁻¹` is a pure rotation |
| **lateral** | **109.60855** | **yes** — this *is* the flat-road assumption |

P1's *"max\|ΔH\| = 0.000e+00"* is reproduced exactly through our code path. `INHERITED → MEASURED`.

### 2.2 ⭐ A2 — the closed form, checked numerically rather than asserted

Over `|dlat| ∈ {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}` m × depths {6, 12, 25, 50} m × 11 scene heights, the
observed relative error matches

```
rel_err(x)  =  height_above_road(x) / h_cam
```

with **maximum absolute deviation 0.000000000** — i.e. it is **exact**, and it is **independent of depth,
of `|dlat|` and of focal length**.

⇒ **L-OK would require every scene point to lie within `0.25 × 1.5 = 0.375 m` of the road surface.**
That is the falsifiable condition, published in the artifact. A driving frame does not satisfy it.

### 2.3 ⭐⭐ A3 — the sign inversion, and the number that should end the discussion

`u_applied = u_true × (1 − height_above_road / h_cam)`. At `|dlat| = 2.0 m`, depth 15 m
(all content at this depth has the **same** true displacement of **−35.4667 px**):

| content | height above road (m) | true Δu (px) | **applied Δu (px)** | applied / true | sign inverted |
|---|---:|---:|---:|---:|:--:|
| road surface | 0.00 | −35.47 | **−35.47** | 1.000 | — |
| kerb | 0.15 | −35.47 | −31.92 | 0.900 | — |
| sedan wheel hub | 0.33 | −35.47 | −27.66 | 0.780 | — |
| sedan beltline | 1.05 | −35.47 | −10.64 | 0.300 | — |
| **sedan roof** | 1.45 | −35.47 | **−1.18** | **0.033** | — |
| **camera height** | **1.50** | −35.47 | **0.00** | **0.000** | ⬅ zero-crossing |
| SUV roof | 1.85 | −35.47 | **+8.28** | −0.233 | ⛔ |
| van roof | 2.60 | −35.47 | **+26.01** | −0.733 | ⛔ |
| truck / trailer roof | 4.00 | −35.47 | **+59.11** | −1.667 | ⛔ |
| traffic-light head | 5.50 | −35.47 | **+94.58** | −2.667 | ⛔ |
| building 2nd floor | 7.50 | −35.47 | **+141.87** | −4.000 | ⛔ |

**A sedan's roof receives 3.3 % of its true motion. Everything taller than the camera moves the wrong
way.** This is not a fidelity *degradation*; above 1.50 m it is a **different scene**.

### 2.4 A6 — the frame fraction that has no ground-plane preimage

The ground plane `Y = h_cam, Z > 0` projects only **below** the horizon. Every pixel at or above it shows
content the flat-road model cannot represent even in principle:

| pitch | horizon row `v` | rows with no ground preimage | fraction of frame |
|---:|---:|---:|---:|
| −2° | 118.71 | 118.71 | **46.37 %** |
| **0° (shipped)** | **128.00** | **128.00** | **50.00 %** |
| +2° | 137.29 | 137.29 | 53.63 % |
| +4° | 146.60 | 146.60 | 57.27 % |

### 2.5 A1 / A4 — the verdict clause, and the positive control beside it

| `|dlat|` (m) | mean err (px) | max err (px) | **mean rel err** | **frac points `rel_err < 0.25`** |
|---:|---:|---:|---:|---:|
| 0.25 | 4.18 | 24.38 | **1.4718** | **0.2830** |
| 0.50 | 8.36 | 48.77 | 1.4718 | 0.2830 |
| 1.00 | 16.73 | 97.53 | 1.4718 | 0.2830 |
| 1.50 | 25.09 | 146.30 | 1.4718 | 0.2830 |
| **2.00** | **33.45** | **195.07** | **1.4718** | **0.2830** |
| 3.00 (`ENV_LAT_MAX`) | 50.18 | 292.60 | 1.4718 | 0.2830 |

| `|dψ|` (deg) — **positive control** | max err (px) | max rel err | frac points `rel_err < 0.25` |
|---:|---:|---:|---:|
| 1 / 3 / 6 / **12 (`ENV_YAW_MAX`)** | **0.00** | **0.00** | **1.0000** |

**Required 95 %. Measured 28.30 %. ⇒ L-BAD, by 3.4× on the frame-fraction clause.**
And note the second column: **the relative error does not improve as the axis shrinks** — P-1 refuted.

### 2.6 ⚠️ A5 — the OTHER criterion §7.4 offered is vacuous, and I refuse it

§7.4 offered *"C-GEO/C-RT"*. **C-RT (round-trip residual) cannot see this defect at all.** With
`n = (0, cos p, sin p)` and `t ∝ (1, 0, 0)` we have `n · t = 0`, so the second-order term vanishes and the
ground-plane homography **composes exactly**: `H(a)H(b) = H(a+b)`, hence `H(d)H(−d) = I`.

| arm | max round-trip deviation from identity | max `H(a)H(a)` vs `H(2a)` |
|---|---:|---:|
| yaw | 4.17e-14 | 1.42e-13 |
| **lateral** | **0.0 (exactly, at 5 of 6 amounts)** | **0.0** |

**A criterion that hands the provably-wrong arm a *perfect* score is not a criterion.** It is logged here
and **refused**, rather than reported as a pass. *(This is the fourth "guard that cannot fail" this
program has had to retire; it belongs in `RETRACTION_LOG.md` under the C13 class — see §8.)*

### 2.7 What this does NOT say

* It does **not** say the lateral warp is useless. For **road-surface** content it is exact, and P1's
  ADE-based lateral sweep (`INHERITED`: +0.0285 m paired at 2.0 m, IDF 0.198) is a real, small number.
  The two are consistent: **our arm is largely insensitive to the half of the frame the warp destroys.**
  That is a statement about *our arm*, not a licence to call the observation faithful.
* It does **not** condemn the existing closed loop retroactively any more than it already stands
  condemned — `corridor_rollout` warps on **both** axes, so every sequential closed-loop number carries
  this infidelity on top of its EXTRAPOLATION label. That is a **new** and separate charge (§8, escalation 2).

---

## 3. What the grid became — and what it does NOT claim

| axis | substrate | fidelity | status |
|---|---|---|:--:|
| **heading `dψ`** | `H = K R K⁻¹` | **exact for arbitrary depth**; only FOV fabrication degrades it | ✅ **USED**, `dψ ∈ {−12, −8, −4, 0, +4, +8, +12}°` |
| **longitudinal `dlon`** | index offset along the logged path — **real frames, zero synthesis** | exact (it *is* observed footage) | ✅ **USED**, `dlon ∈ {−10, 0, +10}` frames (±1.0 s) |
| **lateral `dlat`** | ground-plane homography | §2 | ⛔ **REFUSED in code** (`LateralAxisRefused`) |

**21 grid points.** `GridSpec` raises `LateralAxisRefused` unless a caller supplies **both** an explicit
flag **and** a written reason — the L-BAD verdict is enforced by the type, not by a comment.

⚠️ **Disclosed limitations of the shipped grid, stated rather than buried:**

1. **We are NARROWER than NAVSIM v2 in heading** — our 12° (measured envelope) vs their 20° rejection
   filter. Ours is *measured*, theirs is *chosen*; they are not strictly commensurable and I do not claim
   parity. (`INHERITED` from the research doc §2.3, itself `PUBLISHED` C1.)
2. **We have no lateral axis at all**, where they sample every 0.5 m to ±2.0 m. This is the real cost of
   having a homography instead of 3D Gaussian Splatting.
3. **`dlon` changes the observation *time* as well as the position**, so replayed agents are at their
   logged positions for a different instant. This is **the same approximation the existing closed loop
   already makes** (it re-indexes to the nearest logged pose `mstar`), so it is not a new error — but it
   is an error, and `dlon` is the axis to drop first if anyone objects.

---

## 4. The harness — and the assertion that is the whole point

**`taniteval/taniteval/pseudosim.py`** (in the package, not in `incoming/`) · tests
**`taniteval/tests/test_pseudosim.py`** — **21 tests, all green**, CPU-only, no checkpoint, no pod.

### 4.1 The protocol, mechanically

For every anchor and every grid point: take the **real** frame window at `anchor + dlon`; warp it
**once** by `sampling_homography(0, dψ)`; call the planner **once**; record. **Nothing is fed back.**
`rollout_steps_executed` is emitted and is **`0` by construction**.

### 4.2 ⭐ The 0 %-out-of-envelope assertion

`assert_grid_in_envelope(grid)` runs inside `pseudo_evaluate` **before any checkpoint is loaded** (a bad
grid costs zero GPU-seconds — there is a test for that). It computes the fractions with
**`taniteval.ood.envelope_fractions`** — the same function the broken closed loop is judged by — and
raises **`EnvelopeViolation`** on anything above zero.

**MEASURED on the shipped grid** (`out/pseudosim_v4_30k.json → envelope_proof`, and the run log):

```
frac_steps_lat_over_3m                         = 0.0
frac_steps_yaw_over_12deg                      = 0.0
frac_steps_any                                 = 0.0
frac_windows_any_step_out_of_envelope          = 0.0
EXTRAPOLATION_VERDICT  →  class MEASUREMENT
ratio_is_lower_bound                           = False
```

Compare the sequential loop on the same corpus (`INHERITED`, P1/E1a): **K=20 → 12.3 %**, K=60 → 50.7 %,
**K=185 → 90.2 %** of windows outside.

### 4.3 ⭐ What value makes the assertion FAIL — stated, and exercised

The assertion **publishes its own falsifier** in its output:

```json
"falsifier": {
  "smallest_failing_abs_dyaw_deg": 12.000000001,
  "smallest_failing_abs_dlat_m":    3.000000001,
  "example": "GridSpec(dyaw_deg=(12.5,)) raises EnvelopeViolation"
}
```

and three tests hold it to that: the edge value **12.0°** is accepted (the envelope is inclusive),
**12.0012°** (`ENV_YAW_MAX × 1.0001`) **raises**, and **12.5°** raises with zero planner calls. *A grid
0.0012° too wide is refused.* The assertion is not decoration.

### 4.4 Refusals wired into the code, not into prose

| refusal | mechanism |
|---|---|
| lateral axis | `LateralAxisRefused` at `GridSpec.__post_init__` |
| out-of-envelope grid | `EnvelopeViolation` before any model load |
| a composite with no live component | `VacuousMetric` — **refuses to emit** |
| collision / TTC without cuboids | emitted as `None` + reason; **never a constant** |
| `overlapping_holdout_se` | named in `_refused_estimator` on every node |
| traffic mode | `traffic_mode` on every node; cannot be omitted |

---

## 5. The composite — map-free, and each component's discriminative range MEASURED

### 5.1 Why not ADE, and why not a Driving Score

`PUBLISHED` **DEMONSTRATED, weak tier (n = 8, C3)**: against Bench2Drive Driving Score,
**Ego Progress ρ = 0.83** (strongest single predictor) > DAC 0.71 > TTC 0.59 > **collision NC 0.45**,
**Comfort saturated ≥ 99.9 %** (*"essentially zero discriminative information"*), and classical
**L2/ADE ρ = −0.36, p = 0.43** — negative and not significant. *(NAVSIM v1's own coefficient is
**graphical, not numeric** (C2) and is deliberately not quoted here.)*

`PSS = (∏ gates) × (Σ wₓ sₓ / Σ wₓ)`, PDM-shaped. **Map-free by necessity**: PhysicalAI-AV has no map,
lane graph, junction annotation, traffic-light feature or route signal (settled at five probes; the card
says verbatim *"we do not include open maps data"*), so **DAC, Lane Keeping, Driving-Direction Compliance
and Traffic-Light Compliance are impossible and are not faked.**

| component | w | what it is | map-free? | status |
|---|---:|---|:--:|---|
| **ego_progress** | 5 | plan's along-track distance ÷ the human's, clipped to [0,1] | ✅ | admitted (§5.3) |
| **recovery** | 5 | `1 − \|xt_end\| / \|xt_hold_matched\|` — does the plan cancel the drift its own forward motion causes? | ✅ | admitted (§5.3) |
| **comfort** | 2 | `a_lon`, `a_lat`, jerk, yaw-rate all within bounds | ✅ | ⚠️ see §5.3 |
| **no_collision** | gate | at-fault collision with `obstacle.offline` cuboids | ✅ | ⛔ **NOT EMITTED** |
| **ttc** | 5 | minimum time-to-collision | ✅ | ⛔ **NOT EMITTED** |

⛔ **Because every multiplicative gate is a collision term and all of them are unavailable, the product is
empty and the result is NOT a Driving Score.** It is named `PSS_recovery_progress` and carries a
`_not_a_driving_score` field. **Do not compare it to a PDMS number.**

**Why collision is unavailable (MEASURED, two probes):** the val cache holds only
`{frames_u8 [199,9,256,256], actions, poses, maneuvers, episode_id}`. `obstacle.offline` would supply the
cuboids (97.4438 % coverage, `reference_frame="rig"` on 100 %, so no extrinsics needed), **but** the
cached `episode_id = int.from_bytes(clip_id.encode()[:4], "big")` **collides**: all 40 val ids resolve,
but to **242** `clip_index` rows, so episode → clip identity is not recoverable from the cache alone; and
of those 242, only **11** have their `obstacle.offline` chunk downloaded. Adding the gate is a scoped,
tractable job — §8, escalation 1.

### 5.2 ⚠️ The defect the adversarial test found, before it shipped

The first `recovery` used `xt_hold = |dlat + v₀·T·sin(dψ)|`. On the 2-episode smoke it scored the
**BLIND** arm **+0.597 [+0.548, +0.646] ABOVE** the sighted one. Root cause: **a planner that barely
moves has a small cross-track error, and the metric paid it for that.** *Standing still was scored as
perfect recovery.*

**Fix:** the denominator is the drift the plan's **own along-track distance** would have produced,
`|dlat + s_along · tan(dψ)|`. A stopped plan now yields `xt_hold → 0` and the score is **NaN (excluded),
never 1.0**. Regression-tested by `test_recovery_is_not_gameable_by_standing_still`, and the naive
denominator is retained in the dump as `_cross_track_hold_v0_m_DIAGNOSTIC_NOT_USED` so the defect stays
visible. **This is why the brief's "validate in BOTH directions" clause exists; it earned its keep here.**

### 5.3 The discriminative range of every component — MEASURED before adoption

*(filled from `out/pseudosim_v4_30k.json → arms.*.component_discriminative_range`; see §6.)*

The gate, `PROPOSED` and stated before scoring: a component is **admissible iff**
`ceiling_frac(score ≥ 0.999) < 0.95` **and** `(max − min) ≥ 0.05`, **and**, when ≥ 2 arms are present,
the **between-arm spread is non-zero**. `composite()` raises `VacuousMetric` if none survives.

---

## 6. Arm scores

*(section completed after the 40-episode run; see `artifacts/pseudosim_v4_30k.json`.)*

---

## 7. Deliverable manifest

*(see §9.)*

---

## 8. Escalations — raised here, not left in a README

---

## 9. What was deliberately NOT done
