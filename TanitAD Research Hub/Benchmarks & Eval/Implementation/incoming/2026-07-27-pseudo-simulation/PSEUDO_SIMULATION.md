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
| **10** | ⭐⭐ **THE INSTRUMENT PASSES ITS OWN GATE — 0 % out-of-envelope AND it discriminates.** `v4_oracle` vs the **identical** checkpoint on a destroyed image: **PSS +0.1882 [+0.1240, +0.2557], SEPARATED**, 15 442 rows / 40 episodes, paired episode-cluster bootstrap. Pre-registered condition **R-a**. ⇒ **the EXTRAPOLATION label comes off this surface.** | `MEASURED` **tier 2** (oracle goal, non-reactive) |
| **11** | ⛔⛔ **AND THE FIRST THING IT MEASURES IS UNFLATTERING: flagship-v4 30 k is INDISTINGUISHABLE FROM CONSTANT VELOCITY.** `PSS` **−0.0034 [−0.0138, +0.0078] n.s.**; `ego_progress` **+0.0055 [−0.0081, +0.0205] n.s.** — and that is the arm's **best case**, on an **ORACLE** goal built from its own future. Pre-registered as **R-c** *before* the numbers landed. | `MEASURED` **tier 2** |
| **12** | ⛔⛔ **On ERROR RECOVERY it is significantly WORSE than a planner that never steers.** Displaced by up to 12° of heading, `v4_oracle` cancels **6.3 %** of its own induced drift; `cv_holdv0`, which cancels **none by construction**, scores **7.8 %**. Paired **−0.0168 [−0.0332, −0.0008], SEPARATED** — ⚠️ small, and the interval nearly touches zero. **This is exactly the signal open-loop ADE cannot contain** (ρ = −0.36, p = 0.43). | `MEASURED` **tier 2**, ⚠️ small effect |
| **13** | ⚠️ **`comfort` is dead — at the FLOOR, opposite to the published finding, and for an arithmetic reason.** `|jerk| ≤ 8 m/s³` at `dt = 0.1 s` means an **8 mm** third-difference in the raw waypoints trips it: 0.0000 on both learned arms, 1.0000 on the analytic CV line. The gate dropped it from all three arms. **It is NOT retuned here** — retuning after seeing who fails is metric-shopping. | `MEASURED` **tier 1** (arithmetic) |

### 0.1 The verdict in one sentence

**Pre-registered outcome A holds — with one axis amputated on measured geometry.** A heading ×
longitudinal grid inside our envelope evaluates at **0.00 %** out-of-envelope against **12.26 %** for the
same corpus at the standing K = 20 and **90.24 %** at K = 185, and it **separates arms** (sighted vs
blind, +0.1882 [+0.1240, +0.2557]) — **so the EXTRAPOLATION label comes off, and TanitAD has a real
closed-loop instrument for the first time.** The lateral axis is refused **in code**, so we ship a
protocol NARROWER than NAVSIM v2's and say so. ⛔ **And the instrument's first verdict is that
flagship-v4 30 k, on an oracle goal, is statistically indistinguishable from constant velocity and
recovers from a heading perturbation slightly but significantly WORSE than not steering at all.**

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

### 1.5 The readings for §6, committed BEFORE the 40-episode numbers landed

Written while the run was still on arm 1, so the interpretation cannot be fitted to the result:

| # | if the run shows … | then I write … |
|:--:|---|---|
| **R-a** | `v4_oracle − v4_blind` PSS **separated** | the protocol has sensitivity to the perturbation it applies ⇒ arm scores admissible. **This is the gate on the instrument, and it is the only clause that licenses §6.** |
| **R-b** | `v4_oracle − v4_blind` **NOT separated** | ⛔ **no arm score is admissible** and §6 reports the failure instead of numbers. The whole protocol goes back on the bench. |
| **R-c** | `v4_oracle − cv_holdv0` **NOT separated** | ⭐ **the learned arm does not beat constant velocity under bounded perturbation.** That is a real, new, closed-loop-class finding consistent with the program's standing result that no arm beats hold-v₀ at cruising — and I report it at full prominence even though it is unflattering. |
| **R-d** | `comfort` saturates | drop it by the gate, exactly as the published cross-benchmark study found (≥ 99.9 % ceiling), and say the gate earned its keep. |

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

### 4.2a ⭐⭐ BEFORE vs AFTER — same corpus, same function, `MEASURED` on both sides

The sequential loop's fractions were `INHERITED`. **They are now `MEASURED` here**, recomputed from the
**committed** 30 k-gate per-window tensors with the *identical* `ood.envelope_fractions` the grid is
judged by, so the comparison is not my number against somebody else's.
**Artifact:** `artifacts/before_after_envelope.json` · **Script:** `scripts/before_after_envelope.py`.

⭐ **Reproduce-before-you-quote, passed:** the same script re-derives the committed corridor headline
numbers exactly — **v4 K=185 overall 0.6388 / junction 0.8432**, **REF-C base 0.5833 / 0.7027** — and
`assert`s on them, so nothing new is quoted from a tree that cannot reproduce the old.

| protocol | arm | K | n windows | steps outside | **windows with ≥1 step outside** | verdict class |
|---|---|---:|---:|---:|---:|---|
| sequential rollout | flagship-v4 30 k | 185 | 41 | 58.998 % | **90.24 %** | `EXTRAPOLATION` |
| sequential rollout | flagship-v4 30 k | **20** | 881 | 5.312 % | **12.26 %** | `EXTRAPOLATION` |
| sequential rollout | REF-C base 30 k | 185 | 41 | 54.581 % | **92.68 %** | `EXTRAPOLATION` |
| sequential rollout | REF-C base 30 k | **20** | 881 | 4.637 % | **10.10 %** | `EXTRAPOLATION` |
| ⭐ **pseudo-simulation** | **any** | — | 21 grid points | **0.000 %** | **0.00 %** | ⭐ **`MEASUREMENT`** |

*(The `INHERITED` "12.3 % at K=20" reproduces as **12.26 %** — confirmed, not merely repeated.)*
`GATE_PROTOCOL` §0.3 refuses K ≤ 20, so **every admissible sequential horizon is in the top four rows.**

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

**MEASURED, 40 episodes × 3 arms × 21 grid points = 15 981 evaluations per arm**
(`artifacts/pseudosim_v4_30k.json → arms.*.component_discriminative_range`; table generated by
`scripts/summarize_pseudosim.py`):

| component | arm | n | n NaN | min | max | mean | IQR | ceiling ≥0.999 | floor ≤0.001 | between-arm spread | **admissible** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `ego_progress` | `v4_oracle` | 15442 | 539 | 0.0000 | 1.0000 | 0.9462 | 0.0642 | 0.4009 | 0.0007 | **0.3464** | ✅ |
| `ego_progress` | `v4_blind` | 15442 | 539 | 0.0000 | 1.0000 | 0.5999 | 1.0000 | 0.4361 | 0.3178 | 0.3464 | ✅ |
| `ego_progress` | `cv_holdv0` | 15442 | 539 | 0.0000 | 1.0000 | 0.9407 | 0.0471 | 0.2690 | 0.0050 | 0.3464 | ✅ |
| `recovery` | `v4_oracle` | 13387 | 2594 | 0.0000 | 0.9990 | 0.0629 | 0.0000 | 0.0001 | 0.7556 | **0.0530** ⚠️ | ✅ (marginal) |
| `recovery` | `v4_blind` | 9186 | 6795 | 0.0000 | 0.9999 | 0.1159 | 0.1205 | 0.0001 | 0.5579 | 0.0530 ⚠️ | ✅ (marginal) |
| `recovery` | `cv_holdv0` | 13110 | 2871 | 0.0000 | 0.9995 | 0.0776 | 0.0450 | 0.0002 | 0.5565 | 0.0530 ⚠️ | ✅ (marginal) |
| `comfort` | `v4_oracle` | 15981 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **1.0000** | 1.0 | ⛔ range below `range_min` |
| `comfort` | `v4_blind` | 15981 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **1.0000** | 1.0 | ⛔ range below `range_min` |
| `comfort` | `cv_holdv0` | 15981 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | **1.0000** | 0.0000 | 1.0 | ⛔ **SATURATED at the ceiling** |
| `no_collision` | (all) | — | — | — | — | — | — | — | — | — | ⛔ **NOT COMPUTABLE** |
| `ttc` | (all) | — | — | — | — | — | — | — | — | — | ⛔ **NOT COMPUTABLE** |

⚠️ **`recovery`'s between-arm spread is 0.0530 against a `range_min` of 0.05 — it clears by 6 %.**
It is admitted, and it is reported as **marginal**. `v4_oracle`'s IQR is **0.0000**: three quarters of
its rows sit at the floor, so the component is carried by a tail. **Do not treat `recovery` as a
well-conditioned score on this arm** — treat it as evidence that the arm mostly does not recover at all.

#### ⚠️ `comfort` is dead — but at the FLOOR, not the ceiling, and the reason is arithmetic

The published finding is ceiling-saturation (≥ 99.9 % pass). **Ours is the opposite**, and it is not a
measurement artefact of small `n` — it follows from the discretisation:

```
j_k = (p_k − 3p_{k−1} + 3p_{k−2} − p_{k−3}) / dt³ ,  dt = 0.1 s  ⇒  dt³ = 1e-3
|j| ≤ 8 m/s³   ⟺   |third difference of position| ≤ 0.008 m  =  8 mm
```

**An 8 mm wiggle across three consecutive waypoints trips the clause.** Applied to a *raw* 20-waypoint
head output (256 anchors, no controller, no smoothing) it measures **waypoint quantisation**, not
comfort — which is why the analytic `cv_holdv0` straight line passes at exactly 1.0 while both learned
arms fail at exactly 0.0. The gate refuses it in **both** directions (`range below range_min` for the
learned arms, `SATURATED at the ceiling` for CV) and `composite()` drops it.

⛔ **It is NOT retuned here.** Retuning a bound after seeing which arms fail is metric-shopping. What a
usable comfort clause needs is stated instead: score the **controller-tracked** trajectory
(`clhorizon.wp_to_control` → bicycle), not the raw waypoints — i.e. the same object nuPlan/NAVSIM
actually measure. `PROPOSED` for whoever picks this up.

⚠️ **Anti-metric-shopping rule, fixed before the numbers landed.** If a component fails the gate, §6
reports **the pre-registered composite's result first** and only then discusses what a corrected
component would need. Any variant computed afterwards is labelled **post-hoc** and never becomes the
headline. The per-window dumps retain `ego_progress_raw_ratio`, `cross_track_end_m`,
`cross_track_hold_matched_m` and `along_track_end_m` precisely so a variant can be derived **with no GPU
and no re-run** — and therefore audited.

---

## 6. Arm scores

**Artifact:** `artifacts/pseudosim_v4_30k.json` (+ three `_perwindow_*.npz`) · **Script:**
`scripts/run_pseudosim.py` · **Tables generated by** `scripts/summarize_pseudosim.py`, not hand-typed.
40 val episodes · stride 8 · 21 grid points · **15 981 planner calls per arm, 0 rollout steps**.
`traffic_mode: log_replay_nonreactive` · goal provenance **ORACLE** (an upper bound, not deployable).

### 6.0 ⭐ The instrument's own gate, read BEFORE any arm score

Pre-registered as **R-a/R-b** in §1.5: does the protocol separate an arm that **can** see the
perturbation from the **identical** arm that cannot? `v4_blind` differs from `v4_oracle` in **exactly one
thing** — the image is zeroed. Same checkpoint, same oracle goal, same `v0`, same grid, same windows.
The heading perturbation is visible **only** in the image.

> **`v4_oracle − v4_blind` PSS = +0.1882 [+0.1240, +0.2557] · ⭐ SEPARATED · n = 15 442 rows / 40 episodes**
> **⇒ R-a fires. The protocol has sensitivity to the perturbation it applies, so §6.1–6.2 are admissible.**

Had this come back n.s., §6 would have reported the failure and no arm score at all (R-b). It did not.

⚠️ **A confound in this control, stated because it bounds the claim.** `v4_blind` is blind to the
**image** but still receives the **oracle goal** (`route` / `route_graded` / `vt_band`, minted from the
ego's own future). So the control isolates *"can it see the perturbation"* — the perturbation is
image-only — but it does **not** make the arm blind to the future. A cleaner control would blind both.
This matters for §6.3's residual, not for the gate above.

### 6.1 Scores

| arm | n evals | n eps | `ego_progress` | `recovery` | `comfort` | **PSS_recovery_progress** |
|---|---:|---:|---|---|---|---|
| `v4_oracle` | 15 981 | 40 | **0.9462** [0.9320, 0.9591] | **0.0629** [0.0449, 0.0816] | 0.0000 ⛔ dropped | **0.5622** [0.5496, 0.5725] |
| `v4_blind` | 15 981 | 40 | **0.5999** [0.4857, 0.7050] | **0.1159** [0.0913, 0.1426] | 0.0000 ⛔ dropped | **0.3749** [0.3076, 0.4368] |
| `cv_holdv0` | 15 981 | 40 | **0.9407** [0.9169, 0.9610] | **0.0776** [0.0604, 0.0959] | 1.0000 ⛔ dropped | **0.5705** [0.5558, 0.5844] |

`recovery` **defined fraction**: `v4_oracle` 0.8377 · `v4_blind` **0.5748** · `cv_holdv0` 0.8203 — the
blind arm is excluded on **42.5 %** of rows because it does not move far enough for the question to
mean anything. **That is the §5.2 fix working as designed**, not missing data.

### 6.2 Paired contrasts (episode-cluster bootstrap, B = 2000, identical rows, `overlapping_holdout_se` refused)

| contrast | `ego_progress` | `recovery` | **PSS** |
|---|---|---|---|
| `v4_oracle − v4_blind` | **+0.3464** [+0.2437, +0.4593] ⭐ SEP | **−0.0564** [−0.0811, −0.0314] ⭐ SEP | **+0.1882** [+0.1240, +0.2557] ⭐ SEP |
| **`v4_oracle − cv_holdv0`** | +0.0055 [−0.0081, +0.0205] **n.s.** | ⛔ **−0.0168** [−0.0332, −0.0008] ⭐ SEP | ⛔ **−0.0034** [−0.0138, +0.0078] **n.s.** |
| `v4_blind − cv_holdv0` | −0.3409 [−0.4501, −0.2420] ⭐ SEP | +0.0403 [+0.0211, +0.0589] ⭐ SEP | −0.1939 [−0.2595, −0.1332] ⭐ SEP |

### 6.3 ⭐⭐ Reading — and it is unflattering, which is why R-c was pre-committed in §1.5

**1. ⛔ The flagship v4 30 k arm is INDISTINGUISHABLE FROM CONSTANT VELOCITY on this instrument.**
`PSS` delta **−0.0034 [−0.0138, +0.0078]**, not separated; `ego_progress` **+0.0055 [−0.0081, +0.0205]**,
not separated. And this is the arm's **best case**: an **ORACLE** goal built from its own future.
**R-c fires exactly as pre-registered**, and it is consistent with the program's standing finding that
no arm beats hold-v₀ at cruising — now stated on a closed-loop-class surface that is a **MEASUREMENT**.

**2. ⛔⛔ On error recovery the learned arm is SIGNIFICANTLY WORSE than a planner that never steers.**
`recovery` **−0.0168 [−0.0332, −0.0008]**, separated. Displaced by up to 12° of heading, `v4_oracle`
cancels **6.3 %** of the drift its own forward motion causes; `cv_holdv0`, which by construction cancels
**none**, scores **7.8 %**. ⚠️ **The effect is small and the interval nearly touches zero (upper bound
−0.0008) — it is separated, not large.** But the direction is the finding: **there is no evidence the
arm steers back toward the reference path when displaced, and weak evidence it does slightly worse than
not trying.** **This is precisely the signal open-loop ADE cannot contain** (`PUBLISHED`, weak tier:
L2 vs closed-loop Driving Score ρ = −0.36, p = 0.43).

**3. The blind arm's residual `recovery` advantage (+0.0564) is a goal artefact, not a capability.**
`v4_blind` scores *higher* recovery than `v4_oracle` on the rows where it moves at all — but it keeps
the **oracle goal**, which encodes the ego's true future route (§6.0). A blind planner steering on a
true route label will sometimes curve the right way. ⚠️ **This is a confound in my control, not a
finding about blindness**, and it is why the gate in §6.0 is read on `PSS` (where progress dominates and
the blind arm collapses, −0.3409) rather than on `recovery` alone.

**4. ⚠️ Constant velocity is a strong baseline at a 2 s horizon, by construction.** Over 2 s most logged
driving *is* nearly constant-velocity, so `ego_progress` has little room to separate competent planners —
`cv_holdv0` scores 0.9407 against `v4_oracle`'s 0.9462. **The instrument's sensitivity (§6.0) and its
resolution between similar planners are different things, and only the first is demonstrated here.**
The direct remedy is §6.4.

**5. R-d fires:** `comfort` is dead in both directions and the gate dropped it from all three arms
(§5.3). The published expectation was ceiling-saturation; ours is floor-saturation, for the arithmetic
reason in §5.3. **The gate earned its keep — a hand-written suite would have shipped this clause.**

### 6.4 ⭐ A property of this protocol worth stating: the scoring horizon is DECOUPLED from the envelope

In the sequential loop, extending the horizon **is** what drives the ego out of the envelope — that is
the whole wound. In pseudo-simulation **no observation is ever re-synthesised after the grid point**, so
rolling the *plan* forward further in metric space costs **nothing** envelope-wise: the out-of-envelope
fraction stays `0` at any scoring horizon. NAVSIM v2 scores over **4 s**; we score over **2 s** only
because `FlagshipV4Head` emits 20 waypoints. **The cap is the head, not the protocol.** A longer-horizon
head would immediately get a longer-horizon MEASUREMENT out of this harness with no re-validation.

---

## 7. Deliverable manifest

All repo paths relative to the working tree on the dev box; everything **`git add`-ed, NOT committed,
NOT pushed**. Anything living in only ONE place is marked ⚠️.
Repo dir: `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-pseudo-simulation/`

| artifact | where it lives | md5-verified | what it is |
|---|---|:--:|---|
| **`taniteval/taniteval/pseudosim.py`** | **repo (the PACKAGE, not `incoming/`)** · `tanitad-eval:/workspace/_pseudosim/taniteval/` | ✅ `86a11c46…` | the harness: `GridSpec`, `assert_grid_in_envelope`, `pseudo_evaluate`, `score_windows`, `discriminative_range`, `composite`, `emit` |
| **`taniteval/tests/test_pseudosim.py`** | **repo** | — | **21 tests**, incl. the deliberately-failing envelope input and the standing-still adversary. Full `taniteval` suite: **498 passed**. |
| `PSEUDO_SIMULATION.md` | repo (this dir) | — | this report |
| `scripts/lat_warp_fidelity.py` | repo | — | §2 — C-GEO-LAT. **CPU only, no GPU, no pod, no corpus, no model.** ~2 s. |
| `artifacts/lat_warp_fidelity.json` | repo | — | §2 — A0…A6 + the computed verdict |
| `scripts/before_after_envelope.py` | repo | — | §4.2a — BEFORE/AFTER fractions + the reproduce-before-you-quote `assert` |
| `artifacts/before_after_envelope.json` | repo | — | §4.2a — 12.26 %/90.24 % vs 0.00 %, and the committed 0.6388/0.8432/0.5833 reproduction |
| `scripts/run_pseudosim.py` | repo · `tanitad-eval:/workspace/_pseudosim/run_pseudosim.py` | ✅ `171f2ffa…` | §6 — the 3-arm driver |
| `scripts/summarize_pseudosim.py` | repo | — | §5.3/§6 tables, **generated from the artifact, not hand-typed** |
| `artifacts/pseudosim_v4_30k.json` | repo · `tanitad-eval:/workspace/_pseudosim/out/` | ✅ `3a8f30fd…` | §6 — arm scores, ranges, paired deltas, envelope proof |
| `artifacts/pseudosim_v4_30k_perwindow_v4_oracle.npz` | repo · `tanitad-eval:…/out/` | ✅ `a45e216c…` | per-window dump — **the arithmetic-only path: any metric re-derivable with no GPU** |
| `artifacts/pseudosim_v4_30k_perwindow_v4_blind.npz` | repo · `tanitad-eval:…/out/` | ✅ `49cde496…` | per-window dump |
| `artifacts/pseudosim_v4_30k_perwindow_cv_holdv0.npz` | repo · `tanitad-eval:…/out/` | ✅ `35cfcd91…` | per-window dump |
| `artifacts/run.log` | repo · `tanitad-eval:/workspace/_pseudosim/out/run.log` | ✅ `0f03b0ca…` | §6 — the run log incl. the live envelope proof |
| ⚠️ `tanitad-eval:/workspace/_pseudosim/out/smoke.json` | **pod only** | — | the 2-episode smoke that caught the standing-still defect; superseded, deliberately not staged (its metric is the **pre-fix** one and staging it would put a retracted number in the repo) |

**Nothing that took real effort exists in only one place.** The one pod-only row is a superseded smoke.

⚠️ **A git-hygiene event, reported because it is the third instance of a known class.** This agent staged
(never committed) its deliverables mid-run. A **sibling stream then committed the whole index** as
`5a5a905` — *"⭐ THE CONTROLLER IS EXONERATED …"* — which **swept `taniteval/taniteval/pseudosim.py`,
`taniteval/tests/test_pseudosim.py`, `scripts/lat_warp_fidelity.py`, `artifacts/lat_warp_fidelity.json`
and `scripts/run_pseudosim.py` into a commit whose message is about the tactical head.** No work was
lost — the opposite: it is safely tracked — but the pseudo-simulation harness is now findable only under
an unrelated commit title. `CLAUDE.md` records this class twice already (`60265d3`, `3d41bd0`); this is
the third. **The "STAGE, NEVER PUSH" rule protects the agent's work but does not protect it from a
concurrent sibling's whole-index commit** — worth raising with the orchestrator (§8).

### 7.1 ⭐ What this unblocks

| stream | what was blocked | what unblocks now |
|---|---|---|
| **Benchmarks & Eval / gates** | `GATE_30K_RESULTS.md` §10.1's second branch closed with *"no admissible horizon holds the envelope"*, leaving **every** closed-loop gate quantity permanently stamped EXTRAPOLATION. | A **closed-loop-class surface that is a MEASUREMENT**, at 0 % out-of-envelope, with a co-primary-shaped emitter. Needs a `GATE_PROTOCOL` home — escalation 5. |
| **Architecture & Inference / hierarchy** | *"open-loop does not predict closed-loop"* was known but there was no admissible instrument to test a tactical/strategic change against. ADE is the field's **worst** predictor (ρ = −0.36, p = 0.43). | An **error-recovery** signal that ADE provably does not contain, on a bounded grid any arm can be run through in ~13 min/arm on one GPU. |
| **Data Engineering** | — | A precise, small, high-leverage ask: **store the full `clip_id` in the episode cache.** That single field turns on every safety metric (collision, TTC) for every future eval — escalation 1. |
| **Whoever writes the next `GATE_*.md`** | — | ⚠️ A **new** disclosure to carry: sequential closed-loop numbers are not only extrapolated, they were computed through a warp that sign-inverts the upper half of the frame — escalation 2. |

**Pods:** `tanitad-eval` only. pod1 (training), pod2 (owed controls), pod3 (classifier build) **untouched**.
GPU verified idle (0 MiB / 0 %) before use. All writes to `/workspace` (never `/root`, which is 99 % full
and silently truncates). No process was killed. Parity untouched — no episode re-selection, no corpus write.
🔒 No clip UUID or raw PhysicalAI content appears in any artifact.

---

## 8. ⭐ ESCALATIONS — raised here, not left in a README

| # | what needs a decision or a cross-stream change | owner |
|:--:|---|---|
| **1** | ⭐ **The collision gate needs an episode→clip join.** Two concrete blockers, both measured: (a) the val cache's `episode_id` is a 4-byte clip-id prefix and **collides 242 → 40**; (b) the matching `obstacle.offline` chunks are not downloaded (11 of 242 present locally). Fixing (a) needs the ingest to store the full `clip_id` — a **one-field change in the episode builder** that unblocks every safety metric this program will ever want. **Without it `PSS` can never become a Driving Score.** | Data Engineering |
| **2** | ⛔ **A NEW charge against the sequential closed loop, beyond EXTRAPOLATION.** `corridor_rollout` warps on **both** axes, so every sequential closed-loop number also carries the lateral infidelity of §2 — at K=20 the *lateral* p90 is 1.74 m (`INHERITED`, P1 §8), i.e. **the warp was sign-inverting the upper half of the frame on most windows**. `MODEL_REGISTRY.md`'s closed-loop rows need this note **in addition to** the EXTRAPOLATION stamp. | Model-registry agent |
| **3** | ⚠️ **`RETRACTION_LOG.md`, class C13 ("a guard that cannot fail").** Two entries from this run: (a) **C-RT was offered as a fidelity criterion and is provably vacuous** — it returns *exactly 0.0* for the arm under suspicion; (b) **the first `recovery` metric rewarded standing still** and was caught only by an adversarial input, not by review. Both are the same root cause: *a criterion was proposed without asking what value would make it fail.* | whoever maintains the log |
| **4** | ⚠️ **`ENV_YAW_MAX = 12.0` is still commented `# MEASURED` in two places** (`taniteval/corridor.py:109-110`, `stack/scripts/run_gate.py:613-614`) and P1's re-validation showed it is a **grid endpoint**, not a measurement (usable edge 15.47°). **This harness inherits that constant** — so if the comment is corrected, the shipped grid may widen to ~15°. Not applied here: both files are live and edited by sibling streams. | `corridor.py` / `run_gate.py` maintainers |
| **0** | ⛔⛔ **THE PROGRAM'S HEADLINE ARM IS NOT SEPARABLE FROM CONSTANT VELOCITY ON THE FIRST CLOSED-LOOP INSTRUMENT THAT IS A MEASUREMENT** (§6.3): `PSS` −0.0034 [−0.0138, +0.0078] n.s., `ego_progress` +0.0055 [−0.0081, +0.0205] n.s., **with an ORACLE goal**, and `recovery` −0.0168 [−0.0332, −0.0008] **worse than not steering**. This is tier 2 and small — but it is the kind of result that should decide what v5 optimises. **Not an agent's call to act on.** | **Sayed** / PI |
| **5** | ⭐ **The protocol change needs a home in `GATE_PROTOCOL.md`.** Pseudo-simulation is not a horizon `K`; §0.3's "refuse `K ≤ 20`" does not apply to it and must not be applied by analogy. A gate that wants a closed-loop-class MEASUREMENT should register `PSS_recovery_progress` on the `pseudo_simulation` surface with its grid and its traffic mode. **This will not happen by itself.** | PI / gate-card author |
| **7** | ⚠️ **Git hygiene, third instance of a logged class.** A sibling's whole-index commit `5a5a905` swept this stream's harness and artifacts into a commit titled about the tactical head, while this agent was still running. "Stage, never push" protects an agent's own work but **not from a concurrent sibling committing the index around it**. The mitigation that would actually work is a pre-commit check on foreign staged paths, not another rule in prose. | orchestrator / `AGENT_OPERATING_STANDARD.md` |
| **6** | ⚠️ **The lateral axis is refused, so the 3-D-reconstruction question is now the ONLY route to it.** If lateral perturbation matters (NAVSIM v2 thinks it does), the substrate must be depth-aware. Our own metric depth is available in principle (`obstacle.offline` cuboids, monocular depth), but AlpaSim's NuRec/gsplat packaging carries a derivative-forbidding licence. **A decision about whether lateral is worth a depth-aware warp belongs to the PI.** | **Sayed** |

---

## 9. Self-refutations, and what was deliberately NOT done

| # | what | status |
|:--:|---|---|
| 1 | **P-1 refuted by my own data**: I predicted the lateral error would shrink with `|dlat|` so a small axis would survive. It does not — the relative error is `|dlat|`-independent. Recorded in §1.4 rather than deleted. | corrected |
| 2 | I nearly reported the round-trip criterion as a **pass** for the lateral arm (it returns 0.0, which looks like success). It is a vacuity, not a pass. | corrected, §2.6 |
| 3 | The first `recovery` metric was **backwards** — it ranked blind above sighted. Caught by an adversarial input, not by reading the code. | corrected, §5.2 |
| 4 | **`SCENE_HEIGHTS_M` are standard vehicle/infrastructure dimensions, not measured from our corpus.** They make the closed form legible; **they do not enter it** — the verdict rests on `rel_err = a / h_cam`, which is exact, and on the 50 % above-horizon fraction, which is geometry. A corpus-grounded height distribution (from `obstacle.offline`) would sharpen §2.3's table but cannot change its sign. | disclosed |
| 5 | **The `dlon` axis changes observation time as well as position** (§3). Same approximation the existing closed loop makes, but it is an approximation. | disclosed |
| 6 | **No reactive traffic**, and the harness says so on every node. Deliberate: NAVSIM v2 does not have it either, and our own CAT-K probe measured NOT reactive. | deliberate |
| 7 | **The arm scores use an ORACLE goal** (route / route_graded / vt_band from the ego's own future). They are an **upper bound**, not deployable. `--goal-mode produced` is available on this checkpoint and was **not** run for time. | not done, flagged |
| 8 | **The v1 (`flagship4b-speedjerk-30k`) arm was not scored** — its plan step decodes conditioned on the *future action sequence* (`rollout_decode(…, fa, …)`), which does not exist for a perturbed state, so it is not a planner in the pseudo-simulation sense. REF-C **is** and would be the natural fourth arm. | not done, flagged |
| 9 | **No comparison to a published leaderboard.** This is an internal number on our own corpus, exactly as Option 1 promised. Option 2 (NAVSIM on OpenScene) remains the only route to a comparable figure. | out of scope |
| 10 | ⚠️ **My blind control is confounded**: `v4_blind` keeps the **oracle goal**, so it is blind to the image but not to its own future. The §6.0 gate is unaffected (it reads `PSS`, which the blind arm loses on progress by −0.3409), but the residual `recovery` gap of +0.0564 is **probably goal-driven and I do not claim it as a finding**. A doubly-blind control is the fix. | disclosed, §6.3 |
| 11 | ⚠️ **`recovery` clears its own range gate by 6 %** (spread 0.0530 vs `range_min` 0.05) and `v4_oracle`'s IQR is **exactly 0.0000**. The `−0.0168` result is separated but small and carried by a tail. **Reported as marginal, not as a headline effect size.** | disclosed, §5.3 |
| 12 | ⚠️ **Constant velocity is a strong baseline at 2 s by construction**, so §6's *non-separation* of `v4_oracle` from `cv_holdv0` is partly a statement about the horizon, not only about the arm. The instrument's **sensitivity** is demonstrated (§6.0); its **resolution between competent planners** is not. §6.4 is the remedy and it costs nothing envelope-wise. | disclosed, §6.3 |
