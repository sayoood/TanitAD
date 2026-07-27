# THE PI IS RIGHT, AND THE INSTRUMENT IS WORSE THAN "ADE-ONLY": **THE CLOSED-LOOP PRIMARY PAYS FOR LATERAL ERROR**

**Wall-clock date:** 2026-07-28 (Europe/Berlin) · **Stream:** Benchmarks & Eval ·
**Branch:** `agent/benchmarks-eval-20260721` · **Repo HEAD at start:** `84028f4`
**Hosts:** dev box only. Every number is CPU arithmetic over committed dumps.
⛔ **pod1 (TRAINING) and pod2 (small validation) were NOT touched — not even read.** pod3 and
`tanitad-eval` were not used: nothing here needs a GPU.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not
re-verified) · `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

> **The PI's demand:** *"You are always and only using ADE to assess the approaches; I think we need
> in addition the other metrics, especially the longitudinal and lateral control."*

---

## 0. Headline

| # | Result | Class · tier |
|:--:|---|---|
| **1** | ⛔⛔ **THE CLOSED-LOOP COMPOSITE REWARDS LATERAL DEGRADATION — IN BOTH DIRECTIONS, ON BOTH ARMS TESTED, AND AT A MAGNITUDE THAT DWARFS THE PROGRAM'S HEADLINE GAP.** Injecting a **2 m constant lateral offset** into `cv_holdv0` moves `PSS@twosided_v2` **+0.0581 [+0.0473, +0.0691] SEPARATED**; a 5° heading error moves it **+0.0747**; and a **zero-mean** lateral jitter (σ = 1 m), which cannot re-centre anything, moves it **+0.0303 [+0.0240, +0.0370] SEPARATED**. **8 of 8** injections tested are separated in the wrong direction. For scale, the entire published gap between `cv_holdv0` and the best learned arm is **−0.0090**. ⇒ **You cannot assess lateral control with the current primary, because the primary pays for lateral error.** | `MEASURED` **tier 1** |
| **2** | ⛔⛔ **THE MECHANISM IS THE SECOND ONE-SIDED CLAMP, IN THE OTHER WEIGHT-5.0 TERM.** `recovery = clamp(1 − xt_end/xt_hold, 0, 1)` has a **hard floor**, and the arms live on it: **55.65 % (`cv_holdv0`) … 92.19 % (`refc_xl_produced`)** of DEFINED rows sit at exactly 0, with the unclamped ratio reaching **34.1**. Rows already past the floor cannot be charged more, so any perturbation that helps a few rows raises the mean. **This is the `clamp_v1` failure class (retraction E-G, logged 2026-07-28) repeated in the term that was NOT audited.** | `MEASURED` **tier 1** |
| **3** | ✅ **THE SUITE THE PI ASKED FOR EXISTS, IS ADMITTED, AND ITS RANGE IS DEMONSTRATED IN BOTH DIRECTIONS.** `taniteval/taniteval/control.py` + 34 tests. **`lon_track`** MDE **±5 % speed error** (and σ = 0.25 m for a zero-mean jitter); **`lat_track`** MDE **±0.25 m** of lateral offset, **±0.01 m/m** of steering drift, **σ = 0.125 m** zero-mean; **`lat_heading`** MDE **±1.0°**. Every ladder straddles its null; ⛔ `admit()` REFUSES an axis validated on one side only. | `MEASURED` **tier 1** |
| **4** | ⭐ **AXIS PURITY IS MEASURED, NOT CLAIMED — AND IT COST ME THREE REWRITES.** On the zero-bias arm every axis' contamination ÷ signal is **≤ 0.0333**; on the real arm `lon_track` is **0.0097** and `lat_track` is **0.3477**. ⛔ **My first lateral form failed this test** — with a flat metre tolerance, halving the speed moved it **+0.1710** while a 1 m lateral offset moved it **−0.1379**: *contamination larger than signal* (**1.2398**). Both forms are published; the failing one is why the shipped one has a travel-widening corridor. | `MEASURED` **tier 1** |
| **5** | ⛔ **`comfort` DECIDED: WEIGHT 0.0, MEASUREMENT KEPT AND RENAMED — and the decisive control is the HUMAN.** Run through the identical arithmetic on the identical windows, **the human's own logged path fails the same bounds on 16.60 % of them** (jerk clause 5.96 %). Meanwhile `cv_holdv0` and `stand_still` — the two arms that do the LEAST — both score a perfect **1.0000**, and every learned planner floors (`v1_tactical_follow` **0.000375**, `v4_oracle` **0.0000**). It measures polyline smoothness and **rewards not driving**. ✅ Zeroing its weight is a **PROVABLE no-op**: 16 published `@clamp_v1` composites reproduce at **max\|diff\| = 0.000000**. | `MEASURED` **tier 1** |
| **6** | ⭐ **THE GATE HAD ONLY HALF ITS CLAUSE, FOR ITS WHOLE LIFE.** `discriminative_range` **computed** `floor_frac_le_0p001` and **never used it**, so a component pinned at its floor was admissible while the identical component pinned at its ceiling was refused. Now symmetric (`FLOOR_FRAC_MAX = 0.95`). `refc_xl_produced`'s `recovery` sits **2.8 points** from being auto-refused by it. | `MEASURED` **tier 1** |
| **7** | ⭐ **THE VALIDATED PLAN INVERTS THE OBVIOUS ORDER, AND THE EVIDENCE FORCES IT.** The highest-value item is not a training run and not a lateral experiment — it is **fixing `recovery`, at 0 GPU-h**, because until it is fixed *every* lateral result the program produces is measured on an axis that pays for the failure. Only **one** step in the plan costs GPU time, and it is **~29–49 GPU-h** (frozen encoder, from the MEASURED 0.662 share), not a GPU-week. | decision |
| **8** | ⚠️ **ONE STEP FAILED ITS OWN MDE TEST AND IS RE-SCOPED IN PLACE.** The systematic-scale MDE (5 %) **exceeds** the effect E-GOAL-3's realisable head would produce (≈2.2 % implied speed error), so that step may **not** be run against `lon_retime`-calibrated power; it is re-scoped onto `lon_time_rmse_s`, whose zero-mean-calibrated MDE (σ = 0.25 m ≈ **0.95 %** of mean travel) is below the effect. | ⛔ self-caught |

### 0.1 Tier and inherited qualifiers

**Tier 1** for everything in §1–§7 (deterministic CPU arithmetic over committed dumps; no model, no
checkpoint, no corpus). **Tier 2** for any statement about a named ARM, which inherits the panel's
four qualifiers verbatim: **oracle goal where used**, **non-reactive log replay**, **no collision or
TTC gate**, **`comfort` dropped by the gate**; plus the source stream's two (Block B arms are **plan
transforms, not trained planners**; two arms are **oracles**).
⛔ **Nothing here is a Driving Score** and nothing here is compared to a PDMS number.
⚠️ **`selected_frac` = 1.000 for every arm BY CONSTRUCTION** — there is no selection step on the
pseudo-simulation surface; every arm is scored on all 15,981 rows.

---

## 1. What was asked, and what the honest starting state was

The PI's premise needs one correction and then it gets *stronger*, not weaker.

**The correction.** The closed-loop primary is **not** ADE. It is
`PSS_recovery_progress@twosided_v2`, a PDM-shaped composite over
`{ego_progress: 5.0, recovery: 5.0, comfort: 2.0}` (`taniteval/taniteval/pseudosim.py`).
`ade_0_2s` is explicitly demoted to a diagnostic in `GATE_PROTOCOL` §0 and `heldout_gate`'s
`REFUSED_PRIMARY`.

**Why the concern is nevertheless right, and worse than stated.** Four facts, all MEASURED:

| # | fact | source |
|:--:|---|---|
| 1 | `comfort` carries weight 2.0 and is information-free — **§7** | MEASURED here |
| 2 | there is **no collision gate and no drivable-area term**; both are structurally unavailable | `pseudosim.COLLISION_UNAVAILABLE_REASON`, settled at five probes |
| 3 | the entire goal/selection line — Bar A, T3, E-GOAL-1→4 — is scored on **`ade_0_2s`**, and every headline from it is an ADE number | `V5_PLAN` §8 |
| 4 | the longitudinal/lateral primitives **existed** (`cross_track_m`, `along_track_progress_err_m`, `lateral.py`) but were **diagnostics, not ranked axes** — an intervention cut along-track RMS **8.799 → 1.557 m (5.65×)**, cut the longitudinal error share **76.9 % → 8.9 %**, raised the ±5 % distance hit-rate **15.00 % → 58.93 %**, and the composite **did not move** (`+0.0078 [−0.0110, +0.0260] n.s.`) | `…/2026-07-28-tactical-action-input/` |

⇒ **The remaining two-term composite is one fixed term (`ego_progress@twosided_v2`, repaired
2026-07-28) and one term that is broken in the same way the first one was.** That is §6, and it is
the finding.

---

## 2. The suite — four ranked axes on the pseudo-simulation surface

`taniteval/taniteval/control.py`. Every axis is computed on the **identical rows** the composite
scores (`human_chord > 0.5 m`, bit-identical to `score_windows`' `ego_progress` mask), with the
**identical estimator** (`ci.episode_cluster_bootstrap` / paired form, B = 2000, unit = val
episode). ⛔ `overlapping_holdout_se` appears nowhere.

| axis | kind | definition | why this shape |
|---|---|---|---|
| **`lon_track`** | LONGITUDINAL | `mean_k clamp(1 − \|t_err_k\| / T_TOL, 0, 1)`, `t_err = along_err / v_human` — **seconds off the human's schedule** | A *time* tolerance is scale-free across speeds (2 m at 5 m/s and 7 m at 17 m/s are charged alike) and it is what longitudinal control *is*. It is a **strict generalisation of `ego_progress`**, which reads the endpoint only and therefore cannot see a plan that arrives on time by the wrong route through time. |
| **`lat_track`** | LATERAL (placement) | `mean_k clamp(1 − \|XTE_k\| / corridor(s_k), 0, 1)` with `corridor(s) = D_TOL · max(s, S_MIN)/S_REF` | **XTE is arc-length-matched** — perpendicular distance to the human *polyline*, not to the human's position at the same tick — so longitudinal error cannot leak in. The corridor **widens with travel** because lateral error compounds (MEASURED: ×14.11 vs longitudinal ×3.20 over 0.5→2 s). Both choices were forced by measurement; see §4. |
| **`lat_heading`** | LATERAL (aim) | `clamp(1 − \|Δψ\| / PSI_TOL, 0, 1)`, Δψ against the human tangent **at the plan's own arc position** | A second, *different* lateral instrument: a pure translation moves `lat_track` and leaves `lat_heading` exactly unchanged; a steering drift moves both. The pair distinguishes *"in the wrong place"* from *"pointing the wrong way"*. |
| **`recovery`** | LATERAL (error recovery) | **IMPORTED** from `pseudosim.score_windows`, not reimplemented | the incumbent, carried so it is scored on the same ladders as everything else — which is how §6 was found |

**Every tolerance is `PROPOSED`, published, and swept.** `T_TOL_S = 1.0`, `D_TOL_M = 1.75`
(= `lateral.LANE_HALF_M`, itself PROPOSED because **no lane geometry exists in this corpus**),
`S_REF_M = 10.0`, `PSI_TOL_RAD = 0.2`. `S_REF_M` is chosen on **conditioning, stated**: at
5 / 10 / 20 / 40 the score's floor-saturation is **0.00 / 0.07 / 0.29 / 0.55** and its ceiling
**0.003 / 0.001 / 0.000 / 0.000**, so 10 is the best-conditioned point not near either bound.
The **`SUITE_ID` carries every one of them** (`control_v1@t1s_d1.75m_sref10m_psi0.2rad`), for the
same reason `pseudosim.metric_id` versions the progress term: a silent redefinition under a stable
name is this program's most-logged failure class.

### 2.1 ⛔ Two anti-gaming rules, each installed because a naive version FAILED here

1. **`lat_track` and `lat_heading` are UNDEFINED below `S_MIN_M = 2.0 m` of plan travel.**
   MEASURED, on the real panel, with the naive definition: **`stand_still` scored `lat_track`
   0.8455 — the highest value in the panel**, above `cv_holdv0` (0.4414) and `v1_tactical_follow`
   (0.4096). A plan that does not move has almost no cross-track error and was being *paid* for it.
   That is the identical defect `recovery`'s progress-matched denominator exists to remove
   (*"standing still is not recovery"*), reappearing one axis over. Pinned by
   `test_lat_track_is_not_gameable_by_standing_still`. ⭐ `lon_track` still charges the stopped plan
   hard (**0.2339**), which is the division of labour the two axes are for.
2. **`lat_bias_m` is reported SIGNED, beside the sign-blind `lat_track`.** A constant-sign lateral
   control can improve a one-sidedly biased planner by re-centring it; only the signed mean shows
   it. Pinned by `test_lat_bias_reveals_the_one_sidedness_lat_track_hides`.

---

## 3. ⭐ DEMONSTRATED DYNAMIC RANGE — the part the PI asked for

⛔ **A metric that cannot fail is not a metric.** Each axis is validated by injecting a *controlled*
degradation and measuring whether it separates, with a **paired episode-cluster bootstrap**
(B = 2000, n = 40 val episodes, 15,981 rows) at every rung of a **signed ladder that straddles the
null**.

### 3.1 The controls

| control | what it does | zero-mean? | why it exists |
|---|---|:--:|---|
| `lon_retime(k)` | resamples the plan along **its own polyline** at arc `k·s(t)` — **path preserved**, schedule scaled | no | the axis-pure longitudinal control |
| `lon_scale(k)` | scales the along-track component | no | the panel's own `v1_ego_half`/`v1_ego_double` construction, kept for comparability — ⛔ **it is NOT axis-pure**, see §4.2 |
| `lat_shift(d)` | constant lateral offset | no | ⛔ **directionally dangerous by construction**, which is why it is here |
| `lat_drift(r)` | `y += r·x` — a constant steering error | no | changes heading; an offset does not |
| `yaw_bias(°)` | rotates the whole plan | no | a *pointing* error rather than a *placement* error |
| `lat_jitter(σ)` | ⭐ per-row **zero-mean** lateral offset | **yes** | cannot re-centre a bias in expectation |
| `lon_jitter(σ)` | ⭐ per-row **zero-mean** along-track offset | **yes** | same |
| `yaw_jitter(σ°)` | ⭐ per-row **zero-mean** rotation | **yes** | ⛔ **added because the suite's own admission rule refused `lat_heading` without it** — every heading control I first wrote had a constant sign, and `lat_jitter` cannot substitute because a per-row *translation* leaves the terminal heading exactly unchanged. **The rule found a real gap in my control set.** |

### 3.2 ⭐⭐ The zero-bias reference arm — and why a real arm could not do this job

A ladder measures **the injection PLUS the arm's own bias**, and on a biased arm the two are not
separable. MEASURED: on `cv_holdv0` — which drives **straight** while the logged road curves —
`lat_track` is **non-monotone** at `yaw_bias +2° → +5°` (0.3661 → 0.3681) and at
`lat_drift +0.02 → +0.05` (0.3706 → 0.3719), because a small LEFT injection partially *follows* the
road on the left-curving windows. Reading that as *"the lateral metric is non-monotone"* would have
been a **false verdict about the metric, caused by the reference**.

⛔ **There is no laterally-unbiased arm in the panel**, so one is constructed: **`human_replay`**,
whose plan **IS the logged future path**, exactly — verified before use, `max |residual| =
1.53 × 10⁻⁵ m`, and the run **asserts** it rather than assuming it. Zero bias on every axis by
construction, so every injection is unambiguous. The two real arms are run on the identical ladders
and published beside it; **only `human_replay` decides admission**, and that is stated in the
artifact (`dynamic_range.admission_reference_arm`).

### 3.3 The demonstration

*(Full ladders in `raw/dynamic_range.json`; tables regenerate via `code/tables.py`.)*

**`lon_track` × `lon_retime`** — the schedule ladder, on the zero-bias arm:

| ×speed | 0.50 | 0.70 | 0.85 | 0.95 | **1.00** | 1.05 | 1.15 | 1.30 | 1.50 | 2.00 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `lon_track` | 0.4863 | 0.6925 | 0.8465 | 0.9489 | **1.0000** | 0.9490 | 0.8472 | 0.6948 | 0.4920 | 0.2383 |

⭐ **Symmetric to 1 × 10⁻⁴ at ±5 %** and monotone on both sides. **MDE = ±0.05 (a 5 % speed error),
both directions.** ⛔ Contrast with the term this generalises: `ego_progress@clamp_v1` scores the
2.00× rung **1.0000** — identical to a perfect plan (pinned by
`test_the_PUBLISHED_progress_term_cannot_see_what_lon_track_sees`).

**`lat_track` × `lat_shift`** — the placement ladder:

| offset (m) | −2.0 | −1.0 | −0.5 | −0.25 | **0** | +0.25 | +0.5 | +1.0 | +2.0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `lat_track` | 0.2129 | 0.4048 | 0.5919 | 0.7675 | **1.0000** | 0.7677 | 0.5922 | 0.4051 | 0.2133 |

⭐ **Symmetric to 2 × 10⁻⁴** across the whole ladder. **MDE = ±0.25 m.**

**`lat_heading` × `yaw_bias`**: 1.0000 → 0.9127 at ±1°, 0.8255 at ±2°, 0.5637 at ±5°, 0.1273 at
±10° — **symmetric to 4 decimal places. MDE = ±1.0°.**

**The zero-mean controls** (the ones that cannot re-centre anything):
`lat_jitter` σ = 0.125 / 0.25 / 0.5 / 1.0 / 2.0 m → `lat_track` 0.9066 / 0.8208 / 0.6946 / 0.5394 /
0.3758, monotone. **MDE = σ 0.125 m.** `yaw_jitter` **MDE = σ 0.5°**. `lon_jitter` **MDE = σ 0.25 m**
(≈ **0.95 %** of the 26.348 m mean 2 s human travel).

### 3.4 ⛔ The admission rule, and what it refuses

`control.admit()` raises `AxisNotDemonstrated` unless **all four** hold:

1. it separates at all *(else it is a C13 instrument)*;
2. it separates on **BOTH sides** of every two-sided ladder *(`clamp_v1` passed every adversary it
   had — `stand_still`, `v4_blind`, `v1_ego_half`, **all too slow** — and was blind above ratio 1.0
   for its whole life)*;
3. at least one **ZERO-MEAN** control separates *(a constant-sign control can move an aggregate by
   re-centring rather than degrading — MEASURED on this surface)*;
4. **monotone away from the null FROM THE MDE OUTWARD** *(inside the MDE the response is by
   definition not separated from zero, so its sign is not readable; the full-ladder monotonicity is
   still recorded, it just does not gate)*.

| axis | admitted? | on |
|---|:--:|---|
| `lon_track` | ✅ | `lon_retime` (±0.05), `lon_scale` (±0.05), `lon_jitter` (σ 0.25 m) |
| `lat_track` | ✅ | `lat_shift` (±0.25 m), `lat_drift` (±0.01 m/m), `yaw_bias` (±1°), `lat_jitter` (σ 0.125 m), `yaw_jitter` (σ 0.5°) |
| `lat_heading` | ✅ | `yaw_bias` (±1°), `lat_drift` (±0.01 m/m), `yaw_jitter` (σ 0.5°) |
| `ego_progress` | ✅ | `lon_retime` (±0.05), `lon_jitter` (σ 0.25 m) |
| `recovery` | ✅ **on the zero-bias arm** — ⛔ **and that is exactly the problem: see §6** | |

**The rule can fail, and is exercised failing**: four tests drive `admit()` to raise — one-sided
separation, no separation at all, no zero-mean control, and non-monotonicity.

---

## 4. Axis purity — MEASURED, and it cost me two rewrites

An axis that responds to everything is a longitudinal metric wearing a lateral name, and on this
corpus **ADE is 98.6 % longitudinal by squared-error energy**, so that is the default outcome unless
it is designed against and then *checked*.

⭐ **The ratio is arm-dependent, so it is published on BOTH arms** (`raw/cross_sensitivity.json`):

| arm | axis form | Δ under the OTHER axis' control | Δ under its own | contamination ÷ signal | dominates? |
|---|---|--:|--:|--:|:--:|
| `human_replay` | `lat_track` (SHIPPED) | +0.0000 | −0.5952 | **0.0000** | ✅ |
| `human_replay` | `lon_track` | −0.0171 | −0.5137 | **0.0333** | ✅ |
| `human_replay` | `lat_heading` | −0.0111 | −0.4363 | **0.0255** | ✅ |
| `cv_holdv0` | `lon_track` | −0.0044 | −0.4536 | **0.0097** | ✅ |
| ⭐ `cv_holdv0` | **`lat_track` — widening corridor (SHIPPED)** | **+0.0410** | −0.1178 | **0.3477** | ✅ ×2.9 |
| ⛔ `cv_holdv0` | **`lat_track` — FLAT tolerance (REJECTED)** | **+0.1710** | −0.1379 | **1.2398** | ⛔ **contamination EXCEEDS signal** |
| ⚠️ `cv_holdv0` | `lat_heading` | +0.0352 | −0.0097 | **3.6425** | ⛔ **see below** |

*(cross control = `lon_retime(0.5)`, path-preserving; own control = `lat_shift(−1 m)` / `lon_retime(0.5)` / `yaw_bias(5°)`)*

⭐ **On the zero-bias arm the purity is exact, not merely dominant:** `lat_track` under
`lon_retime` is **1.0000 at every rung from 0.5× to 2.0×**, with paired deltas of **3 × 10⁻⁹**.
⚠️ **Those rungs report `separated = True`, and that separation is ARITHMETIC, NOT EVIDENCE** — a
1 × 10⁻⁹ interval is at float64 resolution. Stated rather than quoted as a purity result.

⚠️⚠️ **`lat_heading`'s 3.6425 on `cv_holdv0` is NOT a contamination finding, and reading it as one
would be the fourth version of the same mistake.** `cv_holdv0` already *has* a large heading error
(`lat_heading` mean 0.3379), so it sits where a bounded score barely moves: its own control, a 5°
rotation, buys only **−0.0097**. The identical measurement on the zero-bias arm is **0.0255** —
**143× smaller** — because there the same 5° rotation buys **−0.4363**. This is §4.4's
*"a bounded score is not monotone in the raw error on a biased arm"* appearing in the purity
statistic rather than in a ladder. ⇒ **the zero-bias arm answers *"is the metric pure?"*; the real
arm answers *"how does it behave where we actually evaluate?"*. Both are published and neither alone
is the answer** — and the caveat ships inside the artifact, not only in this report.

### 4.1 ⛔ Self-refutation 1 — my first lateral form failed its own purity test

A flat metre tolerance made a **purely longitudinal** degradation move the lateral axis *upward* by
more than a genuine lateral offset moved it downward. The cause is real physics, not a coding bug: a
plan that drives half as far accrues less lateral deviation because lateral error compounds. The fix
is the travel-widening corridor, and its effect is measured: contamination **+0.1710 → +0.0410
(4.2×)** while a **0.25 m** offset now registers **−0.0339**, comparable to halving the speed —
where before it took a **1 m** offset to match.

### 4.2 ⛔ Self-refutation 2 — my first longitudinal CONTROL was not axis-pure either

`lon_scale` (an anisotropic scale of the along axis) **rotates every segment of a curved plan**. It
moved `lat_heading` by **−0.0371** while a genuine **5° `yaw_bias`** moved it by **−0.0069** — a
"longitudinal" degradation contaminating the lateral axis **5.4× harder than the lateral control
did**. Any lateral verdict read off that ladder would have been an artefact of the *control*.
`lon_retime` (path-preserving) is the control a purity claim may use; `lon_scale` is kept because it
is the panel's own `v1_ego_half`/`v1_ego_double` construction and must stay comparable. Pinned by
`test_lon_retime_preserves_the_PATH_and_lon_scale_does_not`.

### 4.3 ⛔ Self-refutation 3 — my heading reference was time-matched, and that is one-sided

Comparing the plan's terminal heading to the human's heading at the **same tick** makes a *slow plan
on the correct path* look mis-aimed, because at half the arc the road points elsewhere. MEASURED on
the zero-bias arm: `lon_retime(0.5)` drove `lat_heading` **1.0000 → 0.7713 (−0.2287)** while
`lon_retime(2.0)` left it at **exactly 1.0000** — a **one-sided longitudinal contamination of a
lateral axis**, which is the disease this module exists to cure. The reference tangent is now taken
on the human segment **closest to the plan's endpoint**, the same segment `signed_xte` already
chose. Contamination falls to **−0.0111, a 20.6× reduction**, with the `yaw_bias(±5°)` signal
unchanged at **−0.436**.

### 4.4 ⛔ Self-refutation 4 — a bounded score is not monotone in the raw error on a biased arm

My first test asserted that a zero-mean control must lower `lat_track` on **any** arm. It does not.
On an arm with a 1 m standing bias, zero-mean noise **raises** the mean score even though it raises
the mean error, because the score saturates at 0 and rows pushed further out are charged no more.
⇒ the guarantee a zero-mean control gives is about the **raw error** (Jensen: `E|b+ε| ≥ |b|`), not
about a bounded score — which is precisely why admission runs on the zero-bias reference. Pinned, in
the failing direction, by `test_a_BOUNDED_score_is_not_monotone_in_the_raw_error_on_a_BIASED_arm`.

---

## 5. The direction traps, all three, verified rather than assumed

The brief named three ways a controlled degradation moves the wrong way for a structural reason.
All three were live, and each is now instrumented rather than remembered.

| trap | what the suite does about it | MEASURED here |
|---|---|---|
| a **slowdown** raised the composite +0.1698 because a barely-moving plan scores `recovery = NaN` **by construction** | every ladder rung reports `axis_defined_frac` **and** `recovery_defined_frac` | ⭐ MEASURED on `cv_holdv0`: `lon_retime(0.5)` moves `recovery` **UP** 0.0776 → **0.0860** while its defined-frac moves **DOWN** 0.8203 → **0.8118** — **0.85 pp of rows left the denominator rather than being beaten.** The composite still falls (0.5492 → 0.3156) because `ego_progress` dominates it, but the `recovery` half moved the wrong way for exactly the structural reason the trap names, and it is now visible in the same row rather than inferable |
| a **constant-sign lateral drift** made an arm better by re-centring a one-sidedly biased planner | `ZERO_MEAN_CONTROLS`, and `admit()` **requires** one; `lat_bias_m` is emitted signed at every rung | the zero-mean controls reproduce the direction, so §6 cannot be dismissed as re-centring |
| **per-arm vs panel-wide gating flipped a verdict** (`comfort` 0.0004 → 0.2882, **720×**) | ⛔ `panel_gate()` is **PANEL-WIDE and says so in its own output**; the per-arm gate is **refused, not offered** | the refusal string ships in every emitted node |

---

## 6. ⛔⛔ THE FINDING: the composite pays for lateral error

### 6.1 The direction, paired, on real arms

Injected lateral degradation → Δ`PSS@twosided_v2` (paired episode-cluster bootstrap, B = 2000,
n = 40 episodes; `raw/recovery_onesided.json`). **Positive means the DEGRADED plan scores HIGHER.**

| arm | injection | Δ`recovery` | Δ`ego_progress` | **Δ`PSS@twosided_v2`** | sep |
|---|---|--:|--:|---|:--:|
| `cv_holdv0` | `lat_shift(+2 m)` | +0.1379 | −0.0008 | **+0.0581** [+0.0473, +0.0691] | ⛔ SEP |
| `cv_holdv0` | `lat_shift(−2 m)` | +0.1189 | −0.0008 | **+0.0501** [+0.0386, +0.0620] | ⛔ SEP |
| ⭐ `cv_holdv0` | **`lat_jitter(σ = 1 m)` — ZERO-MEAN** | +0.0717 | −0.0003 | **+0.0303** [+0.0240, +0.0370] | ⛔ SEP |
| `cv_holdv0` | `yaw_bias(+5°)` | +0.1784 | −0.0021 | **+0.0747** [+0.0618, +0.0872] | ⛔ SEP |
| `v1_tactical_follow` | `lat_shift(+2 m)` | +0.1190 | +0.0004 | **+0.0513** [+0.0409, +0.0610] | ⛔ SEP |
| `v1_tactical_follow` | `lat_shift(−2 m)` | +0.1137 | −0.0010 | **+0.0478** [+0.0346, +0.0607] | ⛔ SEP |
| ⭐ `v1_tactical_follow` | **`lat_jitter(σ = 1 m)` — ZERO-MEAN** | +0.0475 | −0.0002 | **+0.0205** [+0.0154, +0.0255] | ⛔ SEP |
| `v1_tactical_follow` | `yaw_bias(+5°)` | +0.1614 | +0.0016 | **+0.0707** [+0.0594, +0.0818] | ⛔ SEP |

**8 of 8 separated, all positive.** `ego_progress` — the term that *was* repaired — is flat to
±0.002 throughout, so the movement is entirely `recovery`.

⭐ **The zero-mean row is the one that closes the argument.** A per-row Gaussian offset with
expectation 0 cannot re-centre a bias, so the movement is not the re-centring artefact the program
already knows about. **It is the metric.**

**For scale**, using the panel's own published numbers: `v1_ego_v0 − cv_holdv0` is
**−0.0090 [−0.0160, −0.0023] SEP** — the whole realisable-arm gap. **Two metres of injected lateral
error is worth 6.4× that gap, in the wrong direction.**

### 6.2 The mechanism — the second one-sided clamp

`recovery = clamp(1 − xt_end/xt_hold, 0, 1)`. The clamp at 0 is a **hard floor**, and the arms live
on it (`raw/recovery_onesided.json`):

| arm | `recovery` mean | defined | **frac at floor 0** | unclamped ratio median | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| `refc_xl_produced` / `refc_xl_v0on` | 0.0259 | 0.8256 | **0.9219** | 1.367 | 3.686 | 13.0 |
| `refc_base_produced` / `refc_base_v0on` | 0.0293 | 0.8250 | **0.9160** | 1.373 | 3.827 | 16.0 |
| `refc_small_produced` | 0.0296 | 0.8253 | **0.9118** | 1.367 | 4.009 | 16.1 |
| `refc_xl_v0off` | 0.0366 | 0.8235 | **0.9008** | 1.359 | 4.156 | 34.1 |
| `v4_oracle` | 0.0629 | 0.8377 | **0.7556** | 1.114 | 3.312 | 8.3 |
| `v1_ego_v0` | 0.0653 | 0.8196 | **0.7287** | 1.095 | 5.228 | 28.4 |
| `v1_tactical_follow` | 0.0785 | 0.8423 | **0.6973** | 1.085 | 3.672 | 9.5 |
| `v4_blind` | 0.1159 | 0.5748 | **0.5579** | 1.042 | 9.252 | 19.9 |
| `cv_holdv0` | 0.0776 | 0.8203 | **0.5565** | 1.007 | 7.036 | 20.2 |

*(all 19 scorable arms in `raw/recovery_onesided.json`; the `refc` pairs are identical arms under two
names)*

Read the floor column: **on 55.65 %–92.19 % of scored rows the plan's cross-track error at 2 s is at
least as bad as not steering at all**, and `clamp` charges every one of them exactly 0. **The median
unclamped ratio exceeds 1.0 for EVERY arm in the panel** — the *typical* window is one the term
cannot grade. A perturbation that helps a minority of rows therefore raises the mean while the
damaged majority is absorbed by the floor.

⭐ **On the zero-bias `human_replay` arm, where nothing is saturated, `recovery` behaves perfectly**
— `lat_shift` gives 1.0000 → 0.8505 / 0.7277 / 0.5402 / 0.3157, symmetric to 4 dp and monotone.
⇒ **the precise claim is: `recovery` is correct only in the regime the arms are not in.**

⚠️ This is the **exact retraction class E-G logged 2026-07-28** — *"metric audited only in the
direction it was built to detect"* — in the term that was **not** audited when `ego_progress` was.
Its own detection heuristic names it: *"a metric defined by a one-sided `clamp`/`max`/`min` has an
entire half of its input domain with zero gradient — enumerate what lives there before adopting
it."* Here that half holds **55–92 %** of the data.

### 6.3 What was fixed now, and what is deliberately NOT

✅ **Fixed now (0 GPU-h, no published number changes):** `discriminative_range` gains the **missing
floor clause**. It had always *computed* `floor_frac_le_0p001` and never *used* it, so a
floor-pinned component was admissible while a ceiling-pinned one was refused —
asymmetric gating of a quantity with no preferred direction. `FLOOR_FRAC_MAX = 0.95`, symmetric with
`CEIL_FRAC_MAX`. **`refc_xl_produced`'s `recovery` sits 2.8 points below the new threshold** — it
would be auto-refused by a slightly stricter one, and that fact is now visible instead of invisible.

⛔ **NOT fixed here, deliberately, and escalated as E-1:** the *shape* of `recovery`. Unlike
`twosided_v2`, no bounded replacement is a **strict refinement** of the published term — every
candidate changes the value on rows the old term handled correctly. That is a **PI decision about
the metric's definition**, not an agent's, and it needs the same treatment `w` got: a shape family,
a sensitivity sweep, and a recorded choice. **Step 2 of the plan is exactly that work**, and it is
the highest-value item in the plan.

---

## 7. The decisions the brief demanded

### 7.1 `comfort` — **weight 0.0. Measurement kept, renamed to what it is.**

⛔ **Do not leave an information-free term carrying weight 2.0 and call the composite three-term.**
It no longer does: `COMPONENT_WEIGHTS = {ego_progress: 5.0, recovery: 5.0, comfort: 0.0}`, with
`COMPONENT_WEIGHTS_PUBLISHED_V1` frozen beside it and `WEIGHTS_ID = "w_ep5_rec5_comfort0"`.
`composite()` now emits `n_weighted_terms` and `components_zero_weighted` explicitly.

**The decisive control is the HUMAN.** The four clauses, identical arithmetic, identical 15,981
windows (`raw/comfort_audit.json`):

| what | all four clauses | a_lon | a_lat | **jerk** | yaw_rate |
|---|--:|--:|--:|--:|--:|
| `cv_holdv0` | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `stand_still` | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ⭐ **the HUMAN's own logged path** | **0.8340** | 0.9825 | 0.9150 | **0.9404** | 0.9812 |
| `v1_ego_v0` | 0.2882 | 0.9585 | 0.7570 | 0.2908 | 0.8881 |
| `refc_xl_produced` | 0.0054 | 0.5092 | 0.4436 | **0.0161** | 0.7990 |
| `v1_tactical_follow` | **0.0004** | 0.3622 | 0.6291 | **0.0004** | 0.8209 |
| `v4_oracle` | **0.0000** | 0.0000 | 0.0000 | **0.0000** | 0.1666 |

**Three reasons, each a measurement:**
1. ⛔ **It rewards not driving.** The two arms that do the *least* — a straight constant-speed line
   and a stationary plan — both score a **perfect 1.0000**, above real human driving (0.8340).
2. ⛔ **It is bimodal, not graded.** Every learned planner floors at ~0, driven by the **jerk**
   clause (`v1_tactical_follow`: jerk 0.0004 against a_lon 0.3622). Re-timing a plan at constant
   speed *smooths* it, which is why two arms differing **only in schedule** move **720×**
   (0.0004 → 0.2882). It measures the smoothness of a per-waypoint regression, not comfort.
3. ✅ **The human fails it too**, on 16.60 % of the same windows — so the bounds are not separating
   good driving from bad.
⚠️ **Class discipline:** the brief's *"100.0000 % violation over 1,708,288 candidates"* is
`INHERITED` and is a **fan-surface** number. On the **PSS panel** it is not constant but saturated at
both ends. Both are consistent — *information-free where it matters* — and I name the surface
difference rather than merging them.

✅ **The change is a PROVABLE no-op, and it is proved rather than argued.** `comfort` is dropped by
the panel-wide gate for every arm under every progress term, **and** a zero weight adds exactly
`0.0` to both numerator and denominator in `composite()`, so the value is bit-identical in both
branches. Re-running the 16 published `@clamp_v1` composites with the new weights:
**`max|diff| = 0.000000`, PASS** (`raw/repro_gate.json`). What is removed is a **false claim**, not
a number. A dedicated test drives the equality on a `comfort` array with *real range*, so it would
be admissible — and the composite is still identical to 0.0.

**What would REPAIR it, costed in §9 step 7:** the jerk clause is a property of independent
per-horizon waypoint regression, not of the vehicle. A jerk computed on a **fitted** trajectory (or
a continuous jerk-magnitude score instead of a binary AND of four bounds) is a real candidate — and
the **human's 0.9404 jerk pass-rate is the calibration target**, since a comfort clause that real
driving fails is mis-specified.

### 7.2 The missing gates — **stated, not faked; one is mechanical, the other is impossible**

| gate | status | why |
|---|---|---|
| **collision / TTC** | ⛔ **NOT COMPUTABLE today, but MECHANICALLY unblockable** | `obstacle.offline` covers **97.4438 %** of the corpus with rig-frame boxes. Two blockers, neither a research problem: (a) the cached `episode_id` is `int.from_bytes(clip_id[:4])` and **COLLIDES** — 242 clip_index rows map onto the 40 val `episode_id`s, so episode→clip identity is not resolvable from the cache alone; (b) the matching chunks are not downloaded. |
| **drivable area / DAC / lane-keeping / traffic-light** | ⛔ **IMPOSSIBLE, and settled** | PhysicalAI-AV has no map, lane graph, junction annotation or route signal — the card says verbatim *"we do not include open maps data"*, and `obstacle.offline`'s enum over **87,481 cuboids is 10 classes, all dynamic agents**. `egomotion` carries no lat/lon/GNSS, so OSM map-matching is impossible. Settled at five independent probes. |

**Decision: neither is faked, and the consequence is named in the output.** `MISSING_GATES` ships in
every emitted node, and `control_score()` carries `_not_a_driving_score` verbatim: with no collision
gate and no drivable-area term, **nothing computed on this surface is a Driving Score** and none of
it may be compared to a PDMS number. It scores **progress, recovery and control**.
⚠️ **A kinematic-feasibility term is computable today and is NOT a collision gate** — it must never
be named one, which is the over-claim the OOD verdict string once made.

### 7.3 The new axes are **not** silently added to the primary

`control.CONTROL_WEIGHTS` proposes `{ego_progress 5, recovery 5, lon_track 5, lat_track 2,
lat_heading 1}` and `control_score()` implements it — but **`pseudosim.COMPONENT_WEIGHTS` is not
changed to it**. Swapping the program's gate primary is a PI decision (E-2), and `control_score()`
**refuses** to include any axis whose dynamic range was not demonstrated, so the swap cannot happen
by accident. The `lon_track` weight of 5 is not arbitrary: **83 % of 2 s error is along-track** and
v4's 15k→30k regression was **100 %** longitudinal. `lat_track` enters at 2 because v1's tactical
**steering** is currently worth `+0.0006 [−0.0065, +0.0072] n.s.` against a straight line.

---

## 8. ⭐ THE VALIDATED PLAN

**The evidence forces an ordering that is not the obvious one.** The obvious plan is *"fix the
instrument, then attack longitudinal, then lateral"*, and that is broadly right — but the single
highest-value item in it is **not** a training run and **not** a lateral experiment. It is **Step 2,
at 0 GPU-h**, because until `recovery` is fixed *every* lateral number the program produces is
measured on an axis that **pays for the failure it is supposed to detect** (§6), and any
lateral-investment decision taken before it is uninterpretable.

**Cost basis, stated once.** `MEASURED`: encoder = **0.662** of a real `v4_loss_step` at 256×640.
`ESTIMATED` (V5_EVALUABLE §8.3, both biases named there): a full 30k × accum-8 v5 run on one A40 is
**≈113 GPU-h** at 256×640, **≈87** at 176×624, **≈72** at 128×576. A **frozen-encoder** decoder
retrain costs `(1 − 0.662) = 0.338×` a step **if features are cached** (⇒ **≈29 GPU-h** at 176×624;
≈38 / ≈24 at the other frames) and roughly `0.56×` if the encoder still runs forward (⇒ **≈49**).
⚠️ Both are `ESTIMATED`; the caching assumption is named because it is worth a **1.7×** swing.

### The steps

| # | step | cost | n | bar (pre-registered) | MDE (demonstrated) | what a REFUTE licenses |
|:--:|---|---|---|---|---|---|
| **1** | ⭐ **Re-adjudicate the existing 20-arm panel on the fixed axes.** Nothing new is computed that a GPU is needed for; the dumps exist. | **0 GPU-h**, ~5 min CPU | 15,981 rows / **40 episodes** | Does the realisable-arm ordering change when `lon_track`/`lat_track` are ranked instead of PSS? Pre-registered CONFIRM = at least one rank-1 change among realisable arms. | paired PSS half-width ≈ **0.010** at this n (from the panel's published CIs); `lon_track` ±5 % speed, `lat_track` ±0.25 m | ⇒ **the "nothing beats holding `v₀`" headline is instrument-independent**, and the program must stop attributing it to the metric. That is a real result and it retires a standing excuse. |
| **2** | ⛔⭐ **FIX `recovery`.** Design a bounded, monotone, *unsaturating* replacement; sweep the shape family exactly as `w` was swept; publish the sensitivity; let the PI choose. **Highest-value item in this plan.** | **0 GPU-h**, ~1 h CPU | 15,981 / **40** | **(a)** reproduces `clamp_v1` recovery where ratio ≤ 1 *(or states plainly that it cannot and why)*; **(b)** the injected-lateral ladder becomes **negative and separated** on `cv_holdv0` **and** `v1_tactical_follow`, under the **zero-mean** control; **(c)** no published *longitudinal* verdict flips. | **0.25 m** injected offset / **σ 0.125 m** zero-mean — both demonstrated, both far below the ±2 m at which the defect is currently worth +0.058 | ⇒ the perturbation-recovery construction is unsound on a grid where **`dlat` is identically 0** (recovery is driven **only** by `dpsi`, because the lateral grid axis is refused on measured geometry). The honest consequence is then to **drop `recovery` from the composite** and rank on `{ego_progress, lon_track, lat_track, lat_heading}` — which this suite already supports. **Either branch is a decision, not a stall.** |
| **3** | **Quantify the longitudinal lever without training.** The oracle longitudinal schedule is the ONLY thing that has ever beaten `cv_holdv0` (`v1_ego_oracle_lon − cv_holdv0` = **+0.0228 SEP** @clamp_v1, **+0.0438 SEP** @twosided_v2). Apply E-GOAL-3's *realisable* head schedule (RMS **0.7449 m**) to v1's path and score `lon_track`. | **0 GPU-h** (schedules are already computed) | 15,981 / **40** | `lon_track(realisable) − lon_track(v1_tactical_follow)` separated-better **and ≥ 50 %** of the oracle's `lon_track` gain. | ⛔ **RE-SCOPED — this step failed its own MDE test.** The `lon_retime` MDE is a **systematic 5 %** scale error; E-GOAL-3's head implies ≈**2.2 %** at the endpoint, **below it**. The step therefore runs against **`lon_time_rmse_s`**, calibrated by the **zero-mean** `lon_jitter` MDE of σ 0.25 m ≈ **0.95 %** of the 26.348 m mean travel — below the effect. **Stated because a step whose MDE exceeds its effect is not a step.** | ⇒ the schedule lever does not survive the drop from oracle to realisable, and **step 5 is not funded.** Consistent with, and a cheap test of, *"100 % of the residual is selection"*. |
| **4** | **Re-point the mid-run held-out gate at the fixed primary and prove it still stops the v4 failure shape.** | **0 GPU-h** — and it **protects ≈29.5 GPU-h**, the measured #1 v5 failure cause | the gate's own synthetic + probe surface | The gate must stop a run that decays on the new primary while `ade_0_2s` improves — the existing `test_a_run_that_decays_on_the_composite_while_ADE_IMPROVES_is_stopped` re-run against the new primary, **plus** its falsifier (the ADE control must NOT stop). | the gate's patience rule is 2 consecutive separated-worse probes; the probe-level MDE is the composite's own | ⇒ the new primary is **worse than the old one as a stop signal** and must not be adopted. ⛔ **This step is a precondition for step 5, not a parallel task.** |
| **5** | **The only funded training: a LONGITUDINAL-only intervention.** Freeze encoder + trunk, retrain the tactical decoder with a schedule-aware loss (along-track up-weight / a `lon_track`-shaped term). ⚠️ Contingent on steps 2–4. | ⭐ **≈29 GPU-h** (176×624, cached features) · **≈49** uncached · **≈38** at 256×640 — *not* a GPU-week | 15,981 / **40** on the gate surface; the mid-run gate fires at 10k | `lon_track` separated-better than `v1_tactical_follow` by **≥ the MDE**, **and** `PSS@twosided_v2` (post-step-2) not separated-worse, **and** `lat_track` not separated-worse. | ±5 % systematic / σ 0.95 % zero-mean, as above | ⇒ **do not fund the full v5 GPU-week on a longitudinal loss.** It also promotes *"the 2 s proposal surface is exhausted"* from a re-scoring result to a **training** result, which no experiment in the program has yet done. |
| **6** | **Lateral — and only now.** Re-adjudicate v1's steering (`v1_lat_straight − v1_tactical_follow`) and every lateral arm on `lat_track`/`lat_heading`. **No lateral training arm is funded at this stage.** | **0 GPU-h** | 15,981 / **40** | ⚠️ **The pre-registered expectation is REFUTE**: v1's steering is **+0.0006 [−0.0065, +0.0072] n.s.** against a straight line and **separated-HARMFUL once timed (+0.0100 SEP)**. CONFIRM would require a lateral arm to move `lat_track` by ≥ 0.25 m-equivalent without costing `lon_track`. | **0.25 m** offset · **1.0°** heading · **0.01 m/m** drift — all demonstrated on the zero-bias arm and **required to be re-demonstrated on the arm under test** | ⇒ **lateral investment stays unfunded, closed with an instrument that COULD have seen it.** ⭐ That is the whole point of ordering it here: the current negative verdict was reached on an axis that pays for lateral error, so it is not yet a real refutation. |
| **7** | **Optional, PI-gated: restore the missing gates.** (a) rebuild the val cache carrying `clip_id` so `episode_id` stops colliding, pull the `obstacle.offline` chunks, add a **collision gate**; (b) re-specify `comfort` against the human's **0.9404** jerk pass-rate. | **0 GPU-h**; engineering + download only | — | (a) a collision term with demonstrated dynamic range under an injected-degradation ladder, same four admission rules; (b) a comfort term the human passes at ≥ 0.95 and `stand_still` does **not** saturate. | to be registered when the step is scheduled | ⇒ (a) the cuboids do not resolve to val episodes even with `clip_id`, and **`PSS` can never be a Driving Score on this corpus** — which is a publishable limitation, not a gap; (b) no bound family separates driving quality here and `comfort` is **deleted**, not zero-weighted. |

### 8.1 ⭐ What we would learn if EVERY step failed

This is the case worth pre-committing to, because it is not a null result.

If **1** finds no ordering change, **2** finds no admissible recovery shape, **3** finds the
realisable schedule does not clear half the oracle's gain, **4** finds the new primary is a worse
stop signal, **5** finds a longitudinal loss does not move `lon_track`, and **6** confirms lateral is
worthless — then the program has established, on a **demonstrated instrument** rather than by
absence of evidence:

1. ⭐ **The 2 s planning surface is genuinely exhausted at this observation set**, on *both* of its
   physical axes independently, and *not* because the metric could not see it. Every prior
   "nothing beats `cv_holdv0`" statement was made on an instrument now known to be blind on the
   lateral axis and one-sided on the longitudinal one; after these steps it would be made on one
   that is neither.
2. ⭐ **The blocker is definitively the OBSERVATION SET, not the planner, the scorer, the fan or the
   loss** — which is the one hypothesis the program has never been able to isolate, because every
   previous refutation was confounded by the instrument. That licenses exactly one thing:
   **changing what the model sees** (horizon beyond 2 s, a corpus with a map, or a sensor set with
   agents), and it forecloses the four cheaper alternatives by measurement rather than by argument.
3. ⛔ **And it would retire a whole class of future spending**: no more re-scoring of the frozen
   2 s fan, no more goal-head accuracy work (already refuted at **16.7–33.6× collapse** once the
   consumer is trained), no anchor-set construction, and no map/route acquisition aimed at the
   **+2.9 %** cross-track axis.

Total cost of learning that: **0 GPU-h for six of the seven steps**, and ≈29–49 GPU-h for the one
that is contingent on the other six confirming.

### 8.2 ⚠️ Where I depart from the brief's suggested ordering, and why

The brief proposed *instrument → longitudinal → lateral*. I keep that spine but insert **step 2
(fix `recovery`) ahead of everything**, and I move **step 4 (re-point the gate) before the funded
training rather than alongside it**. Both are forced by §6: a primary that rewards lateral
degradation by **+0.037 to +0.073** cannot arbitrate a training run whose expected effect is a
fraction of that, and the mid-run gate is the *only* mechanism that stops a bad run — re-pointing it
at an unvalidated primary would reinstate the exact **≈29.5 GPU-h** loss it exists to prevent.

---

## 9. The panel on the new axes — three things ADE could not have said

*(`raw/axes_panel.json`; full table in `artifacts/tables.md` T8. **PANEL-WIDE gate: all five axes
admitted, none dropped.**)*
⚠️ **These are levels on a NEW metric id** (`control_v1@t1s_d1.75m_sref10m_psi0.2rad`) and are
**not** comparable to any published `PSS` value. Quote the suite id.

| arm | `lon_track` | rk | `lat_track` | rk | `lat_heading` | rk |
|---|--:|--:|--:|--:|--:|--:|
| ⚠️ `v1_ego_oracle_lon` *(ORACLE)* | **0.9830** | 1 | 0.3676 | 4 | 0.3181 | 7 |
| ⭐ `cv_holdv0` | **0.9347** | 2 | 0.3791 | 2 | 0.3379 | 2 |
| `refc_xl_produced` | 0.9322 | 3 | 0.3387 | 10 | 0.2676 | 14 |
| `v1_ego_v0` | 0.9306 | 5 | 0.3656 | 5 | 0.3229 | 3 |
| `v4_oracle` | 0.9256 | 8 | 0.3749 | 3 | 0.2949 | 8 |
| `refc_xl_v0off` | 0.8087 | 10 | 0.3373 | 15 | 0.2665 | 15 |
| `v1_tactical_oracle` | 0.7079 | 12 | 0.3564 | 8 | 0.3194 | 5 |
| `v1_lat_straight` | 0.7011 | 13 | 0.3650 | 6 | **0.3401** | **1** |
| ⛔ `v1_tactical_follow` | **0.7007** | 14 | 0.3557 | 9 | 0.3192 | 6 |
| `nospeed_tactical_oracle` | 0.6931 | 15 | 0.3571 | 7 | 0.3203 | 4 |
| ⚠️ `v4_blind` | 0.6073 | 16 | **0.5084** | **1** | **0.0807** | **16** |

**Three readings, none of them available from ADE or from `PSS`:**

1. ⭐⭐ **THE v1 FAMILY'S DEFICIT IS UNAMBIGUOUSLY LONGITUDINAL, AND ITS LATERAL BEHAVIOUR IS FINE.**
   `v1_tactical_follow` scores `lon_track` **0.7007** against `cv_holdv0`'s **0.9347** — a 0.234 gap
   on an axis whose MDE is 0.05 — while on `lat_track` it is **0.3557 vs 0.3791** and on
   `lat_heading` **0.3192 vs 0.3379**, gaps an order of magnitude smaller. **The whole v1 family sits
   at ~0.70 on `lon_track` and ABOVE every REF-C arm on both lateral axes.** The composite reported
   this as a single number (0.4242 vs 0.5492) that named no axis. ⇒ **the plan's step 5 targets the
   right axis, and now it is measured rather than argued.**
2. ⭐ **`v1_lat_straight` — v1's plan with its steering replaced by a STRAIGHT LINE — ranks FIRST on
   `lat_heading` (0.3401), above `cv_holdv0`.** That is the axis-resolved form of the published
   `+0.0006 [−0.0065, +0.0072] n.s.` result, and it is stronger than the published one: v1's learned
   steering is not merely *not better* than a straight line, it is **worse on the axis that measures
   steering**.
3. ⚠️ **AND THE PANEL EXPOSES A RESIDUAL LIMITATION OF MY OWN AXIS.** `v4_blind` ranks **1st on
   `lat_track` (0.5084)** while ranking **16th on `lon_track` (0.6073)** and **16th on `lat_heading`
   (0.0807)**. The travel-widening corridor removed most of the "drive less, deviate less" advantage
   but not all of it. ⇒ ⛔ **`lat_track` may never be quoted alone** — the mirror of the program's
   existing *"a recovery figure without its cross-track background is inadmissible"* rule:
   **a lateral figure without its longitudinal background is inadmissible.** `lat_heading`, which is
   travel-invariant, catches `v4_blind` immediately, which is exactly why there are two lateral axes.

---

## 10. Everything that is wrong with this work, stated by me

| # | limitation | status |
|:--:|---|---|
| 1 | ⛔ **No model was run and no checkpoint was scored.** Every number is arithmetic over dumps produced by other streams; their fidelity is `INHERITED` (the source reports publish a 2.2 mm cross-host port check). | by rule (pod1/pod2 forbidden) |
| 2 | ⛔ **The `recovery` defect is DIAGNOSED and INSTRUMENTED but NOT FIXED.** A replacement shape is a PI decision (E-1) and is step 2 of the plan, not this deliverable. Until it lands, **the primary still pays for lateral error.** | ⚠️ **the main caveat** |
| 3 | ⚠️ **All four tolerances are `PROPOSED`**, none is MEASURED on this corpus. `S_REF_M` is chosen on *conditioning*, which is a stated criterion but still a choice; the full grid is emitted (`block(sensitivity=True)`) so no verdict rests on one point, but **the levels in §9 move with the tolerance** and only rankings are robust. | disclosed, swept |
| 4 | ⛔ **`lat_track` STILL FAVOURS AN UNDER-TRAVELLING ARM, and the panel shows it: `v4_blind` ranks 1st on `lat_track` (0.5084) while ranking 16th on `lon_track` and 16th on `lat_heading`.** The travel-widening corridor cut the contamination **4.2×** (purity 1.2398 → 0.3477) but did not eliminate it. ⇒ **`lat_track` may never be quoted alone: a lateral figure without its longitudinal background is inadmissible**, the mirror of the program's existing cross-track-background rule. `lat_heading` is the travel-invariant partner and catches the case. | ⚠️ **disclosed, and it is a real residual** |
| 4b | ⚠️ **Purity ratios are ARM-DEPENDENT and I nearly mis-read my own.** `lat_heading` reads 0.0255 on the zero-bias arm and **3.6425** on `cv_holdv0` — not because the metric is contaminated but because `cv_holdv0` sits where a bounded score barely moves (its own control buys −0.0097 there vs −0.4363 on the clean arm). Both are published; a single-arm purity number is not a purity number. | disclosed (§4) |
| 5 | ⚠️ **`lat_heading` is defined only where the plan's final segment has length**, so a stopped or near-stopped plan is excluded rather than scored. Correct, but it means the axis' `defined_frac` (0.94–0.97 on real arms, **0.000** on `stand_still`) must be read beside the level. | deliberate |
| 6 | **The zero-bias reference arm is SYNTHETIC.** `human_replay` is not a planner; it exists to make an injection unambiguous. Admission is decided there, which is defensible but is a modelling choice, and I name it. | deliberate, published |
| 7 | ⛔ **I edited a sibling's test.** `stack/tests/test_heldout_gate.py::test_the_admitted_component_set_is_PINNED_at_the_first_probe` poisoned its pinned-range table using `comfort` as the marker, which stops producing a composite once `comfort` carries weight 0. I moved the marker to `recovery` (a weight-bearing component, and a strict subset of the real admitted set) **and strengthened the assertion**: the test now checks `weights_admitted == {"recovery"}`, so honouring the pin is *observable* rather than merely non-fatal. The guarantee the pin was written for — that per-arm admissibility cannot change the composite's definition mid-run (it flipped `cv − refc_base` **+0.1303 vs +0.0252** and changed the sign of `cv − v1`) — is unchanged and now better tested. ⇒ **E-3.** | ⛔ disclosed, deliberate |
| 8 | **No collision, no TTC, no map** — inherited and bounding. `PSS` and this suite are **not** Driving Scores. | inherited blocker |
| 9 | **2 s horizon, non-reactive log replay, lateral grid axis refused** — all inherited. The 2 s horizon is why `cv_holdv0` is a strong baseline *by construction*, and the refused lateral grid axis is why `recovery` is driven **only** by `dpsi` (relevant to step 2's REFUTE branch). | inherited |
| 10 | ⚠️ **The GPU-h figures are `ESTIMATED`**, derived from MEASURED step times; the frozen-encoder numbers add a **caching assumption worth a 1.7× swing**, stated in §8. They are single-GPU compute figures and **must not be quoted as a schedule**. | disclosed |
| 11 | ⚠️ **`lon_track` and `ego_progress` overlap by construction** — the latter is the endpoint-only, ratio-form special case. Their paired delta is therefore not an independent second read, and the composite weights in `CONTROL_WEIGHTS` double-count the longitudinal axis somewhat. Deliberate (longitudinal carries 83 % of 2 s error) but it is a choice, not a measurement. | disclosed |
| 12 | ⚠️ **`stack/`'s baseline moved under me**: the brief specified **1557 passed**; the tree at my start was **1576 tests** (concurrent siblings' additions). I report both so a count that grew on its own does not read as mine. | disclosed |

---

## 11. ⭐ ESCALATIONS — raised here, not left in a README

| # | what needs a decision or a cross-stream change | owner |
|:--:|---|---|
| **E-1** | ⛔⛔ **`recovery` IS ONE-SIDED AND THE PRIMARY REWARDS LATERAL DEGRADATION (+0.037 to +0.073 SEPARATED, including under a ZERO-MEAN control).** Nothing that touches the lateral axis can be adjudicated until a shape is chosen. It needs the `twosided_v2` treatment — a family, a sweep, a **recorded** choice — and it is **step 2 of the plan at 0 GPU-h**. ⚠️ **Any v5 gate run before it lands is gated on a metric that pays for the failure mode it exists to catch.** | **PI + `taniteval` maintainer — BEFORE any v5 gate** |
| **E-2** | ⭐ **A decision is owed on whether `lon_track` / `lat_track` / `lat_heading` enter the GATE PRIMARY.** `control.CONTROL_WEIGHTS` proposes it and `control_score()` implements it, but `pseudosim.COMPONENT_WEIGHTS` is deliberately **not** changed — swapping the primary is a PI call. My recommendation: **yes, after E-1**, because the primary's current two terms are one endpoint-ratio and one saturated clamp, and neither reads the *profile*. | **PI — before the v5 gate** |
| **E-3** | ⚠️ **I changed `stack/tests/test_heldout_gate.py`** (limitation 7). The pin's fixture had to move with the weight change; I moved it deliberately, strengthened the assertion, and recorded what it now guarantees. **A reviewer should confirm the strengthened form is what the pin's author intended.** | **`stack/` maintainer** |
| **E-4** | ⭐ **`discriminative_range` gained a FLOOR clause** (`FLOOR_FRAC_MAX = 0.95`), symmetric with the ceiling one it always had. It changes no published number *today*, but it **will** refuse a floor-pinned component in future — and `refc_xl_produced`'s `recovery` is **2.8 points** from it. Whoever runs the next panel should expect that and not read it as a harness bug. | **Benchmarks & Eval** |
| **E-5** | ⛔ **`comfort` now carries weight 0.0.** Provable no-op on every published number (`max|diff| = 0.000000`), but `panel_combine.py` and `recompute_panel.py` still iterate `PS.COMPONENT_WEIGHTS` for the gate and will keep reporting `comfort`'s inadmissibility — which is intended (transparency), not a leftover. **Any doc calling the composite "three-term" is now wrong.** | **Model-registry agent** |
| **E-6** | ⭐ **A NEW RETRACTION CLASS, offered for `RETRACTION_LOG.md` — I did not append it myself.** ⛔ **`A GATE THAT TESTS ONE END OF A BOUNDED QUANTITY`.** `discriminative_range` refused ceiling-saturated components and admitted floor-saturated ones for its whole life, while *computing* the floor statistic it never used. Sibling of E-G (`METRIC AUDITED ONLY IN THE DIRECTION IT WAS BUILT TO DETECT`) and E8 (`TWO-CONDITION GATE AUDITED AT ONE CONDITION`) — **all three are "the audit covered one side of a two-sided object", and this one is the case where the missing side was already being measured.** *Detection heuristic: grep any admissibility rule for a statistic it computes and does not use.* | **PI / RETRACTION_LOG owner** |
| **E-7** | ⚠️ **The collision gate is MECHANICALLY unblockable and nobody owns it.** Two steps: rebuild the val cache carrying `clip_id` (the current `episode_id` collides — 242 clip_index rows onto 40 val ids) and pull the `obstacle.offline` chunks. Neither is research. Until then **no TanitAD number is a Driving Score**, and that should be said in the paper rather than discovered by a reviewer. | **Data Engineering / PI** |

---

## 12. Deliverable manifest

Repo dir: `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-28-closedloop-control-suite/`
Everything `git add`-ed into the working tree. ⛔ **I did not commit and did not push.**
⚠️ marks anything living in only ONE place — **there is nothing in that state.**

| artifact | where it lives | what it is |
|---|---|---|
| `CLOSEDLOOP_CONTROL_SUITE.md` | repo (this dir) | this report |
| `code/run_control_suite.py` | repo | the whole suite run: axes panel, dynamic range, cross-sensitivity, recovery audit, comfort audit, reproduction gate. **No GPU, no model, no corpus.** Builds and **asserts** the zero-bias `human_replay` reference arm. |
| `code/tables.py` | repo | regenerates every table from the raw JSON — the tables are generated, not hand-typed |
| `raw/dynamic_range.json` | repo | ⭐ 22 (axis × control) cells × 3 arms, full ladders, paired CIs, MDEs, monotonicity, the admission verdict |
| `raw/recovery_onesided.json` | repo | ⛔ the floor-saturation census + the paired direction test on `PSS@twosided_v2` |
| `raw/comfort_audit.json` | repo | ⛔ four clauses × 20 arms **+ the HUMAN's own logged path** |
| `raw/cross_sensitivity.json` | repo | axis purity: every control × every rung, with the induced raw error, + the rejected flat-tolerance form |
| `raw/axes_panel.json` | repo | 20 arms on 5 axes, episode-cluster CIs, panel-wide gate, tolerance sensitivity |
| `raw/repro_gate.json` | repo | ✅ 16 published `@clamp_v1` composites at `max|diff| = 0.000000` with `comfort` zero-weighted |
| `raw/run_all.log` | repo | the run's own stdout |
| `artifacts/tables.md` | repo | the generated tables T1–T8 |
| **`taniteval/taniteval/control.py`** | repo | ⭐ **NEW MODULE** — the four axes, `signed_xte`, 8 controls, `dynamic_range`, `admit`, `panel_gate`, `control_score`, `MISSING_GATES` |
| **`taniteval/tests/test_control.py`** | repo | ⭐ **NEW — 34 tests**, 12 of which pin a FAILING value |
| `taniteval/taniteval/pseudosim.py` | repo | `FLOOR_FRAC_MAX` + the floor clause; `COMPONENT_WEIGHTS['comfort'] = 0.0`; `COMPONENT_WEIGHTS_PUBLISHED_V1`; `WEIGHTS_ID`; `COMFORT_STATUS`; `composite()` emits `n_weighted_terms` / `components_zero_weighted` |
| `stack/tests/test_heldout_gate.py` | repo | the pin's fixture moved off `comfort` and **strengthened** (E-3) |

**Reproduce everything, no GPU** (**451 s** for the six jobs on the dev box, 24 CPU):
```
python3 code/run_control_suite.py --n-boot 2000 \
  --in-dir <…/2026-07-27-pseudosim-arm-panel/artifacts> \
  --in-dir <…/2026-07-28-tactical-action-input/artifacts/pw> \
  --in-dir <…/2026-07-28-tactical-action-input/artifacts/blockA> \
  --out-dir raw
python3 code/tables.py raw > artifacts/tables.md
```

### 12.1 Suites

| suite | result | vs the brief's target |
|---|---|---|
| `taniteval/` | **697 passed, 0 skipped** (104.6 s) | ✅ 663 + **34 new**, **zero new skips**, zero failures |
| `stack/` | **1576 passed, 12 skipped** (106.9 s) | ✅ zero failures, **zero new skips**. ⚠️ The brief's baseline was **1557 passed / 12 skipped**; the tree had already advanced to **1576** from concurrent siblings before I started. I added **no** `stack/` tests. |

🔒 **Parity untouched** — no episode re-selected; all 20 arms share the identical 15,981 rows and row
identity is **asserted**, not assumed. No clip UUID or raw PhysicalAI content appears in any
artifact. ⛔ pod1, pod2, pod3 and `tanitad-eval` were not contacted.

---

## 13. Self-refutations

| # | what | status |
|:--:|---|---|
| 1 | ⛔ **My first lateral axis gave `stand_still` the panel's HIGHEST score (0.8455).** Caught by running the panel's own adversary through my new metric before believing it. Fixed with `S_MIN_M` and pinned. | corrected (§2.1) |
| 2 | ⛔ **My first lateral axis failed its own purity test** — flat tolerance, contamination **1.24×** the signal. Both forms are published so the rejection is checkable. | corrected (§4.1) |
| 3 | ⛔ **My first longitudinal CONTROL was not axis-pure** — an anisotropic scale rotates a curved plan and moved `lat_heading` **5.4× harder than a 5° rotation did**. `lon_retime` added; `lon_scale` kept for comparability with the panel's own probes. | corrected (§4.2) |
| 4 | ⛔ **My heading reference was time-matched and therefore one-sided** — a path-preserving slowdown cost **−0.2287** while a speed-up cost exactly **0**. Now arc-matched; contamination **20.6× smaller**. | corrected (§4.3) |
| 5 | ⛔ **My admission rule refused `lat_heading`, and it was right** — every heading control I had written was constant-sign, and `lat_jitter` is structurally unable to move a heading. `yaw_jitter` exists because the rule found the gap. | corrected (§3.1) |
| 6 | ⛔ **My first monotonicity test asserted something false** — a zero-mean control does *not* lower a bounded score on a biased arm (Jensen bounds the raw error, not the saturated score). Pinned as its own test rather than deleted. | corrected (§4.4) |
| 7 | ⚠️ **I nearly published a false "the lateral metric is non-monotone" verdict** from `cv_holdv0`, whose straight-line plan on a curving road makes a small left injection *corrective*. The zero-bias reference arm exists because of that near-miss, and the biased arms are published beside it. | corrected (§3.2) |
| 8 | ⚠️ **One of my own plan steps failed my own MDE rule** and is re-scoped in place rather than quietly kept (§8, step 3). | disclosed |
| 9 | ⚠️ **I did not hold anything to v1's 0.4271** — that is `wm_fidelity_ade_2s`, what the world model scores when handed the true actions, not a planning bar. | correct by construction |
| 10 | ⚠️ **The 3 × 10⁻⁹ "separations" in the purity cells are arithmetic, not evidence**, and are labelled as such rather than quoted as a purity result. | disclosed (§4) |
| 11 | ⛔⛔ **MY OWN RUN FAILED THE REPRODUCTION GATE, AND ONLY THE GATE CAUGHT IT.** My first full run put the synthetic `human_replay` arm into `arms`, where it voted on the **panel-wide** gate. It scores `ego_progress` ≈ 1.0 — **ceiling-saturated by construction** — so under the panel-wide rule it made `ego_progress` inadmissible **for every arm** and silently redefined the composite. The 16-arm reproduction gate went from `max\|diff\| = 0.000000` to **0.393900**. Nothing else in the pipeline would have noticed: the run completed, every table rendered, and every number was wrong. ⇒ ⭐ **a reproduction gate is not paperwork**, and a synthetic reference must never be allowed to vote. Fixed, the reason is in the code at the point of the fix, and the gate is back to **0.000000**. | ⛔ corrected |
