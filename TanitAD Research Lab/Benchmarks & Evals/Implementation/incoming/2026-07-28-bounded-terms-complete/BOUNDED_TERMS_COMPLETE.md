# THE THIRD CLAMP IS FIXED BY A LEVER THE SIBLING NEVER HAD — AND THE GUARD, REWRITTEN FROM ONE FAILURE, INDEPENDENTLY RE-DERIVES ALL FOUR

**Wall-clock date:** 2026-07-28 (Europe/Berlin) · **Stream:** Benchmarks & Eval ·
**Branch:** `agent/benchmarks-eval-20260721` · **Repo HEAD at start:** `31134bb`
**Hosts:** dev box only. Every number is CPU arithmetic over committed dumps. **0 GPU-h.**
⛔ **pod1 (TRAINING 23,000/30,000) and pod2 (small validation, arm B_wide) were NOT touched —
not even read.** pod3 and `tanitad-eval` were not used: nothing here needs a GPU.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not
re-verified) · `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

**Source:** `…/2026-07-28-recovery-twosided/RECOVERY_TWOSIDED.md` escalations **E-2** and **E-3**;
retraction classes **C45** (the clamp), **C46** (`comfort`), **C47** (shape vs density).

---

## 0. Headline

| # | Result | Class · tier |
|:--:|---|---|
| **1** | ⭐⭐ **`lat_heading` IS FIXED, AND THE LEVER THAT FIXED IT IS ONE `recovery` NEVER HAD.** `recovery`'s raw quantity is one number per row *by construction*; `lat_heading`'s was one number per row *by an implementation choice* — it read the terminal heading only, while `lon_track` and `lat_track`, the two bounded axes that **passed** the same audit, are means over 20 steps. Applying the identical per-step expression as a mean drops the panel's worst floor fraction **0.8429 → 0.0260 (32.4×)** with **no free parameter at all**. The shipped term `mean1_lin_q0p5` passes **10 of 10** injections on both arms including the **zero-mean** `yaw_jitter`, against the published term's **5 of 10**. | `MEASURED` **tier 1** |
| **2** | ⛔ **AND THE ZERO-PARAMETER ANSWER LOST ITS OWN ACCEPTANCE TEST, 9/10.** `mean1_lin_q0` — resolution change only, published shape verbatim — is what my rule *prefers* and what I expected to ship. It is correct in sign on all 10 cells and **not separated** on one (`cv_holdv0 × lat_drift(+0.05)`), so H1 disqualified it. ⇒ ⭐ **THE TWO LEVERS DO DIFFERENT JOBS: the RESOLUTION fixes the SATURATION (live_frac 0.157 → 0.746 at `q = 0`), the SHAPE fixes the POWER (9/10 → 10/10).** Reporting either alone would have been a half-repair. | `MEASURED` **tier 1** |
| **3** | ⛔⛔ **AND THE OBVIOUS RESOLUTION FIX DESTROYS THE AXIS'S REASON TO EXIST — CAUGHT BY A CONTAMINATION PANEL, NOT BY THE PASS RATE.** A plain mean over all 20 steps scores **10/10** and has the best live_frac in the sweep — and a **pure lateral TRANSLATION** moves it **−0.0222 SEPARATED**, *3.4× more than the smallest heading degradation does*. `lat_heading` exists only because *"an OFFSET moves `lat_track` and leaves `lat_heading` unchanged"*. **Mechanism:** step 0's plan tangent runs from the reference pose to waypoint 0, so translating the plan rotates that one segment enormously. ⇒ `mean1` averages the **plan-internal segments only**, and every `mean_*` term is disqualified. | `MEASURED` **tier 1** |
| **4** | ⛔ **MY PRE-REGISTERED C47 PREDICTION FAILED, AND THAT IS THE MOST USEFUL RESULT HERE.** I predicted the pass rate would be monotone in the reward bias (`cos` ≥ `lin` ≥ `share`), and committed both outcomes in advance. **MEASURED: `share` (bias 3.332) scores 10.0 mean-correct, `cos` (0.072) 10.0, `lin` (1.000) 9.4 — NOT monotone.** The unsaturating share form that scored **0/8** on `recovery` scores **10/10** here. ⇒ ⭐ **C47 IS BOUNDED, NOT REFUTED: what fails is a charge rate that collapses WHERE THE DATA LIVES, and the reward bias only predicts that when the median sits ABOVE the anchor.** `recovery`'s median ratio was **1.181** with 75.41 % of rows past the anchor; `lat_heading`'s median `u` is **0.9103** with 46.29 % past it. | `MEASURED` **tier 1** |
| **5** | ⭐⭐ **THE GUARD, REWRITTEN FROM THE `lat_heading` FAILURE ALONE, INDEPENDENTLY RE-DERIVES ALL FOUR KNOWN DEFECTS AND CLEARS ALL FOUR REPAIRS.** Gate `v2` refuses exactly `ego_progress@clamp_v1` (live **0.2461**), `recovery@clamp_v1` (**0.0781**), `comfort` (**0.0000**) and `lat_heading@term_lin_q0` (**0.1570**) — and admits `ego_progress@twosided_v2`, `recovery@twosided_v2`, `lon_track`, `lat_track` and the shipped `lat_heading` (**0.9975**). It was not tuned to any of them: the clause is one line, `live_frac ≥ 0.50`. | `MEASURED` **tier 1** |
| **6** | ⛔ **THE REASON `FLOOR_FRAC_MAX` MISSED IT GENERALISES, AND LOWERING THE CONSTANT DOES NOT FIX IT.** The pair (`FLOOR_FRAC_MAX`, `CEIL_FRAC_MAX`) tests **each end separately**, so a term **49 % floored AND 49 % ceilinged** clears both while having gradient on **2 %** of its rows — structurally invisible at *any* threshold. And 0.95 is a dead-component tripwire whose calibration only makes sense for a mean over 20 sub-samples; on a per-row term it is ~20× too loose. ⇒ the fix is the **two-sided `live_frac`** plus a published **`n_sub` row-resolution** field, in a **VERSIONED** gate. ⚠️ The published constant is **NOT lowered**: `recovery@clamp_v1` floors on 55–92 % of rows, so lowering it would drop that term panel-wide and silently redefine every published PSS value. | `MEASURED` **tier 1** |
| **7** | ✅ **REPRODUCTION GATE EXACT: `max\|diff\| = 0.000000` on the 16 published `@clamp_v1` composites** — with the gate versioned, `saturation` extended, `lat_heading` versioned and **its default flipped**. The published `lat_heading` expression is **BIT-identical on 20/20 arms**, the published axis id is still `lat_heading` and the published `SUITE_ID` string is unchanged. Third consecutive repair to meet this bar. | `MEASURED` **tier 1** |
| **8** | ⭐ **THE RANKING SURVIVES AND NO VERDICT FLIPS.** `cv_holdv0` ranks **1 among realisable arms** on the PSS composite **and** on the control composite, under both `lat_heading` terms; the whole v1 **tactical** family sits below every REF-C arm in all three panels. **0 of 10 paired contrasts flipped.** Both instrument guards **STRENGTHEN**: `v4_oracle − v4_blind` **+0.2353 → +0.2384**, `v1_ego_half − v1_tactical_follow` **−0.1310 → −0.1318**. | `MEASURED` **tier 2** |
| **9** | ⛔⛔ **`ego_progress` ABOVE r = 2 IS REAL, IS DEMONSTRATED, AND ITS PUBLISHED HEADLINE NUMBER WAS MISATTRIBUTED.** E-3 said *"still one-sided above ratio 2 — floored on 32.70 % of `v4_blind`'s rows"*. **MEASURED: of that 32.70 %, `r ≤ 0` contributes 31.78 pp and `r ≥ 2` contributes 0.93 pp — 97.2 % of it is the UNDER-travel side, which `twosided_v2` never claimed to touch.** The true over-side floor on a realisable arm peaks at **8.22 %** (`v1_lat_straight`). **But the defect is real anyway**, and one cell proves it: on an arm already over-travelling, a **ZERO-MEAN** `lon_jitter(σ = 2 m)` moves `ego_progress@twosided_v2` **+0.0104 [+0.0054, +0.0158] SEPARATED THE WRONG WAY** — while `clamp_v1`, `w = 0.5` and `w = 1/3` all get the sign right. | `MEASURED` **tier 1** |
| **10** | ⚠️ **AND I DID NOT SHIP THE `ego_progress` FIX, DELIBERATELY — no term in the grid scores 12/12.** `w = 0.5` corrects the sign but is **n.s.** (−0.0013 [−0.0039, +0.0014]); it also **halves the charge rate for over-travel**, and *"more permissive per unit"* is not automatically a fix. Changing `OVER_TRAVEL_WEIGHT` moves **every published PSS level and every ranking**. ⇒ **E-3 stays open with both outcomes measured and the exact knob published.** ⛔ **"The composite is sound end-to-end" may NOT be written.** | `MEASURED` **tier 1** |
| **11** | ⚠️ **`comfort`: PUBLISH THE MEASUREMENT, RETIRE THE SCORE-SHAPED NODE.** It is 100 % saturated on **20 of 20** arms (`max live_frac = 0.0000`), and its `observed_range = 1.0` is **1.0 for any binary array containing one of each value, at any mixture** — so `RANGE_MIN` cannot tell a balanced indicator from one that is 99.93 % constant (`v1_tactical_follow`: pass rate **0.0004**, range **1.0000**). It should not be emitted as `components.comfort` beside two real `[0,1]` scores; it should be emitted as a **rate**. It must **not** be deleted — C46 was found *because* the number was still on the page. | `MEASURED` **tier 1** |

### 0.1 Pre-registered outcome

**CONFIRM for `lat_heading` and the guard; REFUTE for my own C47 prediction; OPEN for
`ego_progress`.** The pre-registered CONFIRM was *"a term exists that is separated in the correct
direction on all 10 heading injections, on both arms, under the zero-mean control, while keeping
`live_frac ≥ 0.50` on every scorable arm"* — fired, and eight terms cleared it. The prediction
*"pass rate is monotone in the reward bias"* — **did not fire**, and §2.6 is what it taught.

### 0.2 Tier and inherited qualifiers

**Tier 1** for §1–§6 (deterministic CPU arithmetic over committed dumps; no model, no checkpoint,
no corpus). **Tier 2** for any statement about a named ARM, which inherits the panel's qualifiers
verbatim: **oracle goal where used**, **non-reactive log replay**, **no collision or TTC gate**,
**`comfort` dropped by the gate**; plus the source stream's two (Block B arms are **plan transforms,
not trained planners**; two arms are **oracles**).
⛔ **Nothing here is a Driving Score** and nothing here is compared to a PDMS number.
⚠️ **Nothing here is held to v1's 0.4271** — that is `wm_fidelity_ade_2s`, not a planning bar.

---

## 1. The defect, restated exactly

```python
dpsi_end   = wrap(psi_plan[-1] - psi_ref_at_matched_segment)   # ONE number per row
lat_heading = clamp(1 - |dpsi_end| / PSI_TOL_RAD, 0, 1)        # ⛔ the published term
```

`PSI_TOL_RAD = 0.2` rad = 11.46°. Both `|dpsi|` and the tolerance are non-negative, so
`1 - u ≤ 1` identically: **the ceiling clamp is provably never active** (MEASURED 0.0001–0.0023).
`clamp(1 - u, 0, 1)` **is** `max(1 - u, 0)`. One clamp, and it is the whole defect — the same
structure as `recovery@clamp_v1`.

`MEASURED`, `raw/density.json` (T1):

| | value |
|---|---|
| floor fraction, over DEFINED rows | **31.22 % (`v1_ego_half`) … 84.29 % (`v4_blind`)** |
| arms above the 50 % warning line | **6 of 20** (`v4_blind` + all five REF-C arms) |
| pooled non-probe median `u` | **0.9103** (median heading error **10.44°**) |
| pooled fraction with `u > 1` | **46.29 %** |
| p99 / max `u` | **11.70 / 15.7074** |
| ⚠️ `v4_blind` median `u` | **6.0696** — its plans point a median **69.55°** off |

⚠️ **`u_max = 15.7074` is structural, not an outlier:** it is `π / 0.2`, a plan pointing exactly
backwards. The raw quantity is a wrapped angle and is therefore **bounded**, unlike `recovery`'s
ratio which reached 34.1 — a difference that matters in §2.2.

⇒ **on nearly half the panel's rows the term is a constant with noise, and on `v4_blind` on five
sixths of them.** `pseudosim.FLOOR_FRAC_MAX = 0.95` refuses **none** of it (§3).

### 1.1 ⛔ THE OBVIOUS ALTERNATIVE, REFUTED BEFORE ANY SHAPE WAS WRITTEN

*"Just widen the tolerance."* `TOL_SENSITIVITY` already sweeps `psi_tol ∈ (0.1, 0.2, 0.4)`.
`MEASURED` (T2), floor fraction of the published shape:

| arm | 0.1 | **0.2 (published)** | 0.4 | 0.8 (**4×**) |
|---|--:|--:|--:|--:|
| `refc_xl_v0off` | 0.7620 | **0.5121** | 0.0670 | 0.0015 |
| ⛔ `v4_blind` | 0.9172 | **0.8429** | **0.7461** | **0.6120** |

⇒ at **four times** the published tolerance — 45.8°, at which point "tolerance" means nothing —
`v4_blind` is **still floored on 61.20 %** of its rows. **The tolerance moves the floor; it does
not remove the one-sidedness.**

---

## 2. ⭐ THE FIX — two levers, a rule, and a prediction that failed

### 2.1 ⭐⭐ The lever `recovery` never had: RESOLUTION

The audit that found this term also cleared `lon_track` and `lat_track` — and the difference
between them was in the audit table the whole time and was read as a footnote rather than as the
fix:

| axis | expression | `n_sub` | max floor |
|---|---|:--:|--:|
| `lon_track` | `mean_k clamp(1 − \|t_err_k\|/T_TOL, 0, 1)` | **20** | 0.0029 |
| `lat_track` | `mean_k clamp(1 − \|XTE_k\|/corridor(s_k), 0, 1)` | **20** | 0.0947 |
| ⛔ `lat_heading` | `clamp(1 − \|dpsi_end\|/PSI_TOL, 0, 1)` | **1** | **0.8429** |

A row of a mean-over-20 term saturates only if **all 20 sub-samples** do. `lat_heading` was the odd
one out, and nothing but implementation history made it so: the same per-step expression exists at
every step, with each step's plan tangent compared against the human tangent on the segment
`signed_xte` matched **to that step** (so the arc-matching correction is applied per step, not only
at the endpoint). `MEASURED`, published shape, **resolution alone, no parameter changed**:

| arm | terminal | mean over 20 | |
|---|--:|--:|:--|
| ⛔ `v4_blind` | **0.8429** | **0.0260** | **32.4×** |
| `refc_base_v0off` | 0.5087 | 0.2510 | 2.0× |
| `refc_xl_v0off` | 0.5125 | 0.2497 | 2.1× |
| `cv_holdv0` | 0.3638 | 0.1981 | 1.8× |
| `v4_oracle` | 0.4364 | 0.0121 | 36.1× |

**Panel max 0.8429 → 0.2510**, ceiling ≈ 0 throughout. Every arm clears the 0.50 rule.

### 2.2 ⛔ AND WHY A PURE SHAPE FIX WOULD HAVE BEEN A BAD TRADE HERE

The one-parameter budget family that fixed `recovery` transfers directly —
`g(u) = clamp(1 − (1−q)·u, 0, 1)`, `q` = *the score a plan exactly at the tolerance receives*,
`q = 0` **is** the published shape — but on the **terminal** resolution it has to fight `v4_blind`'s
median `u` of **6.07**. `MEASURED` min live_frac over scorable arms:

| `q` | 0 | 0.5 | 2/3 | 0.85 | 0.9 |
|---|--:|--:|--:|--:|--:|
| `term_lin_q*` | **0.1570** | 0.2536 | 0.3222 | 0.5250 | 0.7393 |
| `mean1_lin_q*` | 0.7464 | **0.9975** | 0.9977 | 0.9886 | 0.9791 |

⇒ on the terminal resolution nothing below **`q ≈ 0.85`** clears the rule, and at `q = 0.85` the
in-tolerance band is compressed to **15 %** of the range — the axis becomes nearly constant and its
weight-1.0 proposal would silently mean ~1/6 of what it says. **The resolution lever buys the same
gradient for free.**

### 2.3 ⭐ The pre-registered selection rule

Banked into `raw/injections_lat_heading.json ›
injections_lat_heading._selection_rule_PRE_REGISTERED` **before the panel was computed**, and
mirrored in `control.LAT_HEADING_TERMS`' docstring.

| rule | |
|---|---|
| ⚠️ **H0** | **ADDED MID-RUN AND DISCLOSED AS SUCH — it was NOT pre-registered.** DISQUALIFY any term for which a pure lateral **translation** moves the axis at least as much as a heading degradation: `max\|Δ\|` over `{lat_shift(±2), lat_jitter(1)}` must be `< min\|Δ\|` over the 5 heading injections, on **both** arms. ⛔ **Why adding it is admissible:** it is not a new preference, it is the axis's **own published defining claim** (`AXIS_META`), and it is the entire reason a second lateral axis exists. ⛔ **How it was found:** a `B = 40` smoke showed `mean` moving **−0.0222 SEPARATED** under `lat_shift(+2 m)`. `mean1` was built in response, and this rule is what forces the choice between them rather than leaving it to taste. |
| **H1** | DISQUALIFY any term whose **10** heading injections are not ALL separated in the CORRECT (negative) direction on **both** real arms, across both signs of every two-sided control **and** the ZERO-MEAN `yaw_jitter`. |
| **H2** | DISQUALIFY any term with `live_frac < 0.50` on any scorable arm. ⭐ Written **two-sided** on purpose: the one-sided pair is what missed this term. |
| **H3** | Among survivors PREFER the **smallest departure from the published term**, counted as **free parameters first**, family second: a resolution-only change introduces no parameter at all and beats any shape change. Within a resolution prefer the **LINEAR** family (affine on `u ≤ 1`) and the **smallest** surviving `q`. |
| **H4** | If H3's preferred class is empty, take the lowest C47 reward bias among survivors, and record that C47's law decided it. |

⛔ **The sibling's 8 injections could not be reused verbatim, and that is not a preference.**
`lat_shift` and `lat_jitter` **translate** the plan, and a translation leaves every plan tangent
exactly unchanged — `control._ctl_yaw_jitter`'s own docstring says so. A suite built from them is a
suite the candidate **cannot fail**. What is reused is the machinery: the same two real arms, the
same paired episode-cluster bootstrap (B = 2000, unit = val episode), the same both-signs and
zero-mean requirements. The four translation/re-timing controls **are** run — as the
**contamination panel** (§2.5), where they must move the axis *least*.

### 2.4 ⛔⛔ THE ACCEPTANCE TEST — 24 candidates, and the published term scores 5/10

`raw/injections_lat_heading.json`, T4/T5. ⛔ `overlapping_holdout_se` appears nowhere.
**Positive Δ means the DEGRADED plan scores HIGHER.**

| term | H0 purity | H1 | H2 min live_frac | reward bias | verdict |
|---|:--:|:--:|--:|--:|:--:|
| ⛔ `term_lin_q0` *(PUBLISHED)* | ✅ | **5/10** | **0.1570** | 1.000 | ⛔ H1 **and** H2 |
| `term_lin_q0p5` / `q0p6667` | ✅ | 10/10 | 0.2536 / 0.3222 | 1.000 | ⛔ H2 |
| `term_lin_q0p85` / `q0p9` | ✅ | 9/10 | 0.5250 / 0.7393 | 1.000 | ⛔ H1 |
| ⭐ `term_share_q0p5` | ✅ | 10/10 | 0.9977 | 3.332 | ✅ **survives** |
| `term_cos_q0p5` / `q0p75` | ✅ | 10/10 | 0.2443 / 0.3068 | 0.079 / 0.064 | ⛔ H2 |
| ⛔ every `mean_*` | **⛔** | 9–10/10 | 0.749–1.000 | — | ⛔ **H0** |
| ⚠️ `mean1_lin_q0` | ✅ | **9/10** | 0.7464 | 1.000 | ⛔ H1 |
| ⭐ **`mean1_lin_q0p5` ← SHIPPED** | ✅ | ✅ **10/10** | **0.9975** | 1.000 | ✅ |
| `mean1_lin_q0p6667` … `q0p9` | ✅ | 10/10 | 0.979–0.998 | 1.000 | ✅ (H3: larger `q`) |
| `mean1_share_q0p5` · `mean1_cos_q0p5` · `mean1_cos_q0p75` | ✅ | 10/10 | 0.918–1.000 | 3.332 / 0.079 / 0.064 | ✅ (H3: not linear) |

**H3 ⇒ `mean1_lin_q0p5`.** `mean1_lin_q0` (zero parameters) would have won it and did not survive H1.

**The shipped term, cell by cell — 10 of 10**, against the published term on the same rows:

| arm | injection | zero-mean | `term_lin_q0` (PUBLISHED) | **`mean1_lin_q0p5` (SHIPPED)** | |
|---|---|:--:|---|---|:--:|
| `cv_holdv0` | `yaw_bias(+5°)` | | −0.0097 [−0.0160, −0.0036] SEP | **−0.0322 [−0.0414, −0.0219]** | ✅ |
| `cv_holdv0` | `yaw_bias(−5°)` | | −0.0097 [−0.0154, −0.0035] SEP | **−0.0369 [−0.0452, −0.0275]** | ✅ |
| `cv_holdv0` | `lat_drift(+0.05)` | | ⛔ −0.0021 [−0.0054, **+0.0010**] n.s. | **−0.0111 [−0.0164, −0.0055]** | ✅ |
| `cv_holdv0` | `lat_drift(−0.05)` | | −0.0035 [−0.0066, −0.0002] SEP | **−0.0144 [−0.0193, −0.0091]** | ✅ |
| ⭐ `cv_holdv0` | **`yaw_jitter(σ=5°)`** | **yes** | −0.0146 [−0.0195, −0.0098] SEP | **−0.0337 [−0.0380, −0.0293]** | ✅ |
| `v1_tactical_follow` | `yaw_bias(+5°)` | | ⛔ −0.0046 [−0.0100, **+0.0007**] n.s. | **−0.0276 [−0.0330, −0.0221]** | ✅ |
| `v1_tactical_follow` | `yaw_bias(−5°)` | | ⛔ −0.0049 [−0.0138, **+0.0039**] n.s. | **−0.0352 [−0.0400, −0.0303]** | ✅ |
| `v1_tactical_follow` | `lat_drift(+0.05)` | | ⛔ −0.0025 [−0.0062, **+0.0011**] n.s. | **−0.0092 [−0.0120, −0.0062]** | ✅ |
| `v1_tactical_follow` | `lat_drift(−0.05)` | | ⛔ −0.0025 [−0.0071, **+0.0023**] n.s. | **−0.0134 [−0.0159, −0.0108]** | ✅ |
| ⭐ `v1_tactical_follow` | **`yaw_jitter(σ=5°)`** | **yes** | −0.0098 [−0.0144, −0.0051] SEP | **−0.0310 [−0.0351, −0.0270]** | ✅ |

⚠️ **AND THE PUBLISHED TERM'S FAILURE MODE IS NOT `recovery`'s, which I will not overstate.**
`recovery@clamp_v1` was **positive and separated** on 8 of 8 — it *paid* for degradation.
`lat_heading@term_lin_q0` is **correct in sign on all 10** and fails by being **unable to
separate** on 5 of them. The mechanism is the same (zero gradient on the saturated majority) but
the consequence measured here is **loss of power, not sign inversion**. Calling it "the metric
rewards the failure it exists to catch" would be a false analogy to the sibling.
⭐ The effect sizes tell the same story quantitatively: the shipped term moves **2.3–7.2×** further
per injection, on identical rows.

### 2.5 ⚠️ THE CONTAMINATION PANEL — where the obvious fix died, and what still leaks

`raw/injections_lat_heading.json › contamination_cells`, T6. These must move the axis **least**.

| control | `term_lin_q0` | **`mean1_lin_q0p5`** | ⛔ `mean_lin_q0` |
|---|---|---|---|
| `lat_shift(+2 m)`, `cv_holdv0` | −0.0001 n.s. | −0.0035 SEP | ⛔ **−0.0222 SEP** |
| `lat_shift(+2 m)`, `v1_tactical_follow` | +0.0012 n.s. | −0.0024 n.s. | ⛔ **−0.0213 SEP** |
| `lat_jitter(σ=1 m)`, `cv_holdv0` | +0.0000 n.s. | −0.0004 SEP | ⛔ −0.0158 SEP |
| ⚠️ `lon_retime(0.5)`, `cv_holdv0` | ⚠️ **+0.0306 SEP** | ⚠️ **+0.0361 SEP** | ⚠️ +0.0233 SEP |

**H0, as a ratio** — `max|Δ|` translation vs `min|Δ|` heading:

| term | `cv_holdv0` | `v1_tactical_follow` | pure? |
|---|---|---|:--:|
| `term_lin_q0` | 0.0002 vs 0.0021 | 0.0012 vs 0.0025 | ✅ |
| ⛔ `mean_lin_q0p5` | **0.0381 vs 0.0113** | **0.0365 vs 0.0095** | ⛔ **3.4× / 3.8× the wrong way** |
| ⭐ `mean1_lin_q0p5` | 0.0035 vs 0.0111 | 0.0024 vs 0.0092 | ✅ (3.2× / 3.8× margin) |

⚠️ **Two honest qualifications.** (a) The shipped term is **not exactly** translation-invariant —
two of its six translation cells are separated at |Δ| ≤ 0.0035. The plan tangents are exactly
invariant; the *matched reference segment* can shift by one index on a curving path. It is
**dominantly heading-sensitive**, not **immune**, and the published term is cleaner here.
(b) ⛔ **`lon_retime(0.5)` — a purely longitudinal, path-preserving degradation — RAISES this
lateral axis under EVERY term**, and under the shipped one it is **worse** (+0.0306 → +0.0361,
+18 %). This is an inherited defect (the source stream's own limitation), **this work does not fix
it**, and it is the one place where the new term is not strictly stronger. ⇒ **E-4.**

### 2.6 ⛔⛔ MY PRE-REGISTERED C47 PREDICTION FAILED — and this is the finding

Stated before the numbers: *"the pass rate should be MONOTONE IN THE REWARD BIAS: `cos` ≥ `lin` ≥
`share`. **Both outcomes are committed in advance:** if `share` passes here it BOUNDS C47 rather
than confirming it."* `MEASURED`, T7:

| family | mean reward bias | mean cells correct |
|---|--:|--:|
| `cos` | **0.072** | **10.00** |
| `lin` | **1.000** | **9.40** |
| ⛔ `share` | **3.332** | **10.00** |

**`pass_rate_is_monotone_in_reward_bias = False`.** The unsaturating share form —
`u⁻²` charge-rate decay, **0/8** on `recovery` — scores **10/10 on all three resolutions here**,
and `term_share_q0p5` is a full survivor of H0/H1/H2.

⭐ **The correct statement of C47 is therefore not "reward bias > 1 fails".** It is: *a shape fails
when its charge rate collapses **where the data lives**, and the reward bias only predicts that when
the median sits **above** the anchor.*

| | `recovery` | `lat_heading` |
|---|--:|--:|
| median of the raw quantity | **1.181** | **0.9103** |
| fraction past the anchor | **75.41 %** | **46.29 %** |
| `share_q0p5` charge rate at **its own median** | 0.2102 | **0.2740** |
| `share_q0p5` injections | ⛔ **0/8** | ✅ **10/10** |

On `recovery` three quarters of the rows sat in the decayed region; here more than half sit in the
share form's steepest region. **Same algebra, opposite verdict, and only the density differs.**

### 2.7 ⭐⭐ AND THE PARAMETER THAT FAILED ON THE SIBLING IS THE ONE THAT WINS HERE

`q = 0.5`, the even split, scored **7/8 and was DISQUALIFIED** on `recovery` two hours earlier,
where `q = 2/3` won. Here `q = 0.5` passes **10/10** and is selected. The two terms use the
**identical** one-parameter budget family on their own normalised quantity — pinned by
`test_the_parameter_that_FAILED_on_recovery_is_the_one_that_WINS_here`, which asserts the two
implementations agree to 1e-12 over a dense grid. ⇒ **an inherited constant is a hypothesis, and
the acceptance test decides — including when it decides in your favour.**

---

## 3. ⛔⛔ THE GUARD — why `FLOOR_FRAC_MAX = 0.95` missed it, and why a smaller number would miss the next one

### 3.1 Two structural reasons, not one

1. ⚠️ **RESOLUTION.** 0.95 is a *dead-component tripwire* whose calibration only makes sense for a
   term that is a **mean over 20 steps**, where a row-level floor fraction near 1 really does mean
   "dead". Applied to a **single-value-per-row** term it is ~20× too loose. Nothing in the code
   recorded which terms were which.
2. ⛔⛔ **IT TESTS EACH END SEPARATELY.** A term **49 % floored AND 49 % ceilinged** clears
   `FLOOR_FRAC_MAX` *and* `CEIL_FRAC_MAX` while having gradient on **2 %** of its rows. **No value
   of either constant can catch that case** — the pair of one-sided thresholds is structurally
   unable to express *"this term has no gradient left"*. That is the same *"audited on one side of
   a two-sided object"* class as `clamp_v1` itself, one level up. Pinned by
   `test_the_guard_ALSO_misses_a_term_that_SPLITS_its_saturation`.

### 3.2 What ships

| | |
|---|---|
| ⭐ `pseudosim.LIVE_FRAC_MIN = 0.50` | `live_frac = 1 − floor_frac − ceiling_frac`, the **two-sided** statistic the pair cannot express. Set to the same 0.50 as `SATURATION_WARN_FRAC` deliberately: C45's standing consequence was only ever printed as a warning, and a rule that is only a warning is a rule the next agent reads past. |
| ⭐ `pseudosim.GATE_VERSIONS` = `{v1, v2}` | **`v1` is the published gate and stays the DEFAULT.** `v2` = v1 **AND** `live_frac ≥ LIVE_FRAC_MIN`, and is **strictly stronger** — property-tested over 400 random shapes that it never admits what v1 refuses. |
| ⭐ `saturation(..., n_sub=…)` | publishes `live_frac` **and** the term's **row resolution** (`PER-ROW` vs `MEAN over 20 sub-samples`) beside the fraction, so the next reader does not have to rediscover why the threshold was wrong. `control.AXIS_N_SUB` declares it for every bounded axis. |
| ⭐ every node emits **both** verdicts | `admissible_v1` *and* `admissible_v2`, whichever version gates. A stronger reading that is invisible is a reading nobody adopts. |
| ⭐ `control.CONTROL_GATE_VERSION = "v2"` | the control surface — which has **no published composite to protect** and is the one being proposed for the gate primary — is held to the stronger rule. `panel_gate` also reports `admitted_under_gate_v1`. |
| `heldout_gate` probe records | now carry `gate_version` (pinned to `v1`, deliberately) and `components_admissible_under_gate_v2`. |

⛔ **`FLOOR_FRAC_MAX` is NOT lowered, and this is not timidity.** `recovery@clamp_v1` floors on
55.65–92.19 % of rows. Setting the constant to 0.50 would make it **inadmissible panel-wide**, drop
it from the composite and change **every published PSS value** — precisely the silent redefinition
this programme keeps logging. The gate is **versioned** instead. Pinned by
`test_the_published_gate_thresholds_are_NOT_LOWERED_and_here_is_why`.

### 3.3 ⭐⭐ THE GUARD RE-DERIVES ALL FOUR KNOWN DEFECTS — WITHOUT BEING TOLD ABOUT THREE OF THEM

`raw/audit.json`, T12. The clause was written from the `lat_heading` failure alone.

| term | `n_sub` | max floor | max ceil | **min live_frac** | gate v1 | **gate v2** |
|---|:--:|--:|--:|--:|:--:|:--:|
| ⛔ `ego_progress@clamp_v1` | 1 | 0.3178 | **0.5703** | **0.2461** | ✅ | ⛔ **REFUSED** |
| ✅ `ego_progress@twosided_v2` | 1 | 0.3270 | 0.1173 | 0.6704 | ✅ | ✅ |
| ⛔ `recovery@clamp_v1` | 1 | **0.9219** | 0.0002 | **0.0781** | ✅ | ⛔ **REFUSED** |
| ✅ `recovery@twosided_v2` | 1 | 0.0833 | 0.0007 | 0.9161 | ✅ | ✅ |
| ⛔ `comfort` | 1 | **1.0000** | **1.0000** | **0.0000** | ⛔ | ⛔ **REFUSED** |
| ✅ `lon_track` | 20 | 0.0029 | 0.1070 | 0.8930 | ✅ | ✅ |
| ✅ `lat_track` | 20 | 0.0947 | 0.0008 | 0.9051 | ✅ | ✅ |
| ⛔ `lat_heading@term_lin_q0` *(PUBLISHED)* | 1 | **0.8429** | 0.0023 | **0.1570** | ✅ | ⛔ **REFUSED** |
| ⭐ `lat_heading@mean1_lin_q0p5` *(SHIPPED)* | **20** | 0.0014 | 0.0010 | **0.9975** | ✅ | ✅ |

⭐ **Four refusals, four admissions, and the split is exactly the defect/repair split C45 and C46
established independently.** Note `ego_progress@clamp_v1` is caught on its **CEILING** (57.03 %),
the end nobody was looking at — a fourth confirmation from a direction the clause was not designed
for. The demonstration the brief asked for, on the real panel:

| check | result |
|---|:--:|
| `OLD_GUARD_ADMITTED_THE_BROKEN_TERM` | ✅ **True** |
| `GUARD_REFUSES_THE_BROKEN_TERM` | ✅ **True** *(8 arms inadmissible: `v4_blind` + all 7 REF-C arms)* |
| `GUARD_ADMITS_THE_FIXED_TERM` | ✅ **True** |

---

## 4. ⚠️ `ego_progress` ABOVE r = 2 — the attribution is wrong and the defect is real

### 4.1 ⛔ THE PUBLISHED HEADLINE NUMBER IS MISATTRIBUTED

E-3, published 2026-07-28: *"`ego_progress@twosided_v2` is itself still one-sided above ratio 2,
floored on **32.70 %** of `v4_blind`'s rows and 24.27 % of `v1_ego_double`'s."* The fraction is
right; **the mechanism attached to it is not.** `MEASURED` (T3), decomposing the floor by which
side produced it:

| arm | floor total | **from `r ≤ 0` (UNDER)** | **from `r ≥ 2` (OVER)** | median `r` |
|---|--:|--:|--:|--:|
| ⛔ `v4_blind` | 0.3270 | **0.3178 (97.2 % of it)** | **0.0093** | 0.9724 |
| `v1_ego_double` *(PROBE)* | 0.2426 | 0.0060 | **0.2342** | 1.9556 |
| `v1_lat_straight` | 0.0822 | 0.0000 | **0.0822** ← worst realisable | 1.0493 |
| `v1_tactical_follow` | 0.0805 | 0.0006 | 0.0798 | 1.0407 |
| `cv_holdv0` | 0.0141 | 0.0050 | 0.0091 | 0.9834 |

⇒ **`v4_blind`'s 32.70 % is 97.2 % UNDER-travel** — plans that end behind where they started — which
is the **published `clamp_v1` floor at `r ≤ 0`**, untouched by `twosided_v2` and never claimed by
it. The genuine over-side residual on a **realisable** arm peaks at **8.22 %**, not 32.70 %, and
`v1_ego_double`'s 23.42 % belongs to a **probe built to over-travel 2×** — it is that probe doing
its job. **A floor fraction quoted without decomposing which clamp produced it is not evidence
about that clamp.** ⇒ **E-5, a retraction-class item.**

### 4.2 ⛔⛔ AND THE DEFECT IS REAL ANYWAY — ONE CELL, UNDER A ZERO-MEAN CONTROL

`raw/injections_ego_progress.json`, T11. 4 over-travel injections × 3 arms, where the third arm is
`v1_ego_double` — **already over-travelling** (median `r` 1.956, 23.42 % already floored), so a
further over-travel lands almost entirely in the zero-gradient half. That is the decisive substrate,
and using a probe arm to test a *metric* is correct — it is not being ranked.

| progress term | correct | over-side floor max (realisable) | **`v1_ego_double × lon_jitter(σ=2 m)` — ZERO-MEAN** |
|---|:--:|--:|---|
| ⛔ `clamp_v1` | **2/12** | 0.0000 | −0.0013 [−0.0037, +0.0010] n.s. |
| ⚠️ **`twosided_v2`** *(SHIPPED)* | **11/12** | 0.0822 | ⛔⛔ **+0.0104 [+0.0054, +0.0158] SEPARATED THE WRONG WAY** |
| `twosided_asym_w0p5` *(floor at r=3)* | 11/12 | 0.0316 | ✅ sign −0.0013 [−0.0039, +0.0014] **n.s.** |
| `twosided_asym_w0p3333` *(floor at r=4)* | 11/12 | **0.0185** | ✅ sign −0.0013 [−0.0039, +0.0017] **n.s.** |
| `twosided_asym_w2` *(floor at r=1.5)* | 11/12 | 0.1748 | +0.0030 [−0.0010, +0.0073] n.s. |

⭐ **`twosided_v2` is the ONLY term in the grid whose zero-mean cell is separated in the WRONG
direction**, and a zero-mean jitter cannot re-centre a bias — so this is C45's mechanism, alive, in
the term repaired last night. On the same substrate the *constant* over-travel injections separate
correctly and enormously (−0.0973 to −0.1071), so it is specifically the **rows past the floor**
that leak.

### 4.3 ⚠️ WHY I DID NOT SHIP A FIX

**No term in the grid scores 12/12.** `w = 0.5` and `w = 1/3` correct the sign and remain **n.s.**;
they buy a later floor by **halving (or thirding) the charge rate for over-travel**, and *"more
permissive per unit"* is not automatically a fix — the brief's own warning. The trade is
arithmetically forced: a bounded `[0,1]` score with a fixed under-side **cannot** both charge
over-travel at rate 1 and keep charging past `r = 2`; the floor sits at `r = 1 + 1/w`. Pinned by
`test_twosided_v2_floors_at_ratio_2_and_the_w_grid_moves_that_floor`.

And `OVER_TRAVEL_WEIGHT` is **live in the PSS primary**: changing it moves **every published level
and every ranking**. ⇒ the knob, the grid and the failing cell are published and **E-3 stays open
as a PI decision**. `twosided_asym_w0p3333` was **added to `PROGRESS_TERMS` as a sensitivity
anchor only** — the default is untouched, pinned by
`test_the_new_w_anchor_does_not_touch_the_default_or_the_published_term`.

---

## 5. ⚠️ `comfort` — the decision, and why

**MEASURED** (T13, `raw/comfort.json`): `max live_frac over all 20 arms = 0.0000`. Every row of
every arm is at floor or ceiling. It is the AND of four bounds, i.e. a `{0,1}` indicator.

⛔ **`observed_range = 1.0` is an artefact and `RANGE_MIN` cannot see through it.** `max − min` is
**1.0 for any binary array containing one of each value, at any mixture**: `v1_tactical_follow` has
a pass rate of **0.0004** and an `observed_range` of **1.0000**, identical to a perfectly balanced
term. That statistic is what let `comfort` clear `RANGE_MIN` for its whole life. Pinned by
`test_comfort_is_100_percent_saturated_BY_CONSTRUCTION_and_range_is_a_lie`, which drives it at
0.1 %, 50 % and 99.9 % mixtures and asserts gate v1's verdict **depends on the mixture, not on the
(zero) gradient**.

> ### ⭐ DECISION: **PUBLISH THE MEASUREMENT, RETIRE THE NAME AND THE SCORE-SHAPED NODE.**
>
> **(a) It must not be emitted as `components.comfort`** beside two real `[0,1]` scores. A reader
> comparing `0.5492` (recovery) with `1.0000` (comfort) is comparing a **mean score** with a
> **pass rate**, and the programme has already published `observed_range = 1.0` for it twenty times
> as if it were range. A saturated diagnostic printed in score shape misleads exactly as a
> saturated score does — which is the argument that put it in the audit in the first place.
>
> **(b) It must not be deleted either.** ⛔ **C46 was found BECAUSE the number was still on the
> page.** A measurement that refutes its own term is the cheapest instrument in the suite: it is
> what showed the *human's own logged path* fails the bounds on 16.60 % of the same windows.
> Deleting it would remove the evidence and leave only the conclusion.
>
> ⇒ it is emitted as **`diagnostics.plan_smoothness_pass_rate`** — a **rate**, in its own units,
> carrying `COMFORT_STATUS` and its `live_frac = 0.0000` — and **gate v2 refuses it
> automatically**, so no future weight can be attached to it without the refusal being visible in
> the same record.
>
> ⚠️ **What would make it a score, and why that is not shipped:** the AND of four bounds discards a
> continuous margin that is already computed — `max over the four clauses of (observed / limit)` —
> which has real range. It is **named and not shipped** because C46's finding is that the **limits
> themselves** fail the human's own path; a margin against a bound we know is wrong is not a
> diagnostic. **Fix the bound first.**

---

## 6. ✅ THE REPRODUCTION GATE — exact

`raw/repro_gate.json`, T9. Every published `@clamp_v1` composite recomputed with the gate versioned,
`saturation` extended, `lat_heading` versioned and **its default flipped**:

| | |
|---|---|
| published `@clamp_v1` composites checked | **16** |
| **max \|diff\|** | ⭐ **0.000000** |
| verdict | ✅ **PASS** |
| published metric id, unchanged | `PSS_recovery_progress@clamp_v1` |
| published `lat_heading` expression **BIT-identical** | ✅ **20 of 20 arms** |
| published axis id, unchanged | `lat_heading` |
| published `SUITE_ID`, unchanged | `control_v1@t1s_d1.75m_sref10m_psi0.2rad` |
| new `SUITE_ID` | `control_v1@…_psi0.2rad+lath_rowmean_v2` |

⭐ **Backward-compatible by construction.** `SUITE_ID` and `lat_heading_axis_id` append the suffix
**only** when the term is not the published one, so every id emitted through 2026-07-28 keeps its
exact string — and a non-published term is *always* visible in the name. The same rule
`pseudosim.metric_id` uses.
⛔ **The synthetic reference arm was kept OUT of the gate vote.** `human_replay` is built and its
`max |residual| = 1.526 × 10⁻⁵ m` **asserted** before use, but it is never inserted into `arms` —
the sibling stream's gate caught it silently redefining the composite for every arm
(`max|diff| 0.000000 → 0.393900`).
⚠️ **Calibration note:** `human_replay` scores `lat_heading@mean1_lin_q0` = **0.9913**, not 1.0000 —
identical to its `ego_progress` (0.9914), so 0.991 is this surface's own noise floor rather than a
heading-specific artefact.

---

## 7. ⭐ THE RANKING STATEMENT AND THE INSTRUMENT GUARDS

`raw/panel.json`, T10. 20 arms · 40 val episodes · 15,981 rows per arm · row identity
`(ep_i, anchor, dlat, dyaw, dlon)` **ASSERTED across all 20, 0 refused**. ⛔ **PANEL-WIDE gate**;
the per-arm gate is **refused, not offered**.

> **Does `cv_holdv0` still rank first among realisable arms?** ⭐ **YES — in all three panels.**
> Rank **1** on the PSS composite (`PSS_recovery_progress@twosided_v2+rec_twosided_v2`), rank **1**
> on the control composite under the **published** `lat_heading`, and rank **1** under the
> **shipped** one.
>
> **Does the whole v1 family still sit below every REF-C arm?** ⭐ **YES**, in all three.
> ⚠️ *Definition, inherited verbatim from the sibling because it is easy to mis-read:* the claim is
> about the v1 **TACTICAL** family — `v1_tactical_follow`, `v1_tactical_oracle`, `v1_lat_straight`,
> `nospeed_tactical_oracle`. `v1_ego_v0` / `v1_ego_oracle_lon` are ego-**schedule** transforms and
> rank above every REF-C arm under the published term as well.
>
> **⭐ NO VERDICT FLIPPED.** All **10** paired contrasts keep their sign and their separation across
> the two `lat_heading` terms — unlike the sibling repair, which withdrew a 24-hour-old claim.
> One presentation-level move: `refc_xl_produced` ⇄ `refc_xl_v0on` swap ranks 3/4 on the control
> composite (levels 0.7268 vs 0.7269 published), which is a tie, not a finding.

⚠️ **THE INSTRUMENT GUARDS STRENGTHEN — the metric did not become permissive:**

| guard | PSS | CONTROL (published `lat_heading`) | CONTROL (shipped) | |
|---|---|---|---|:--:|
| `v4_oracle − v4_blind` | +0.2992 [+0.2217, +0.3829] SEP | +0.2353 [+0.1801, +0.2949] SEP | **+0.2384 [+0.1815, +0.3004] SEP** | ⭐ stronger |
| `v1_ego_half − v1_tactical_follow` | −0.1454 [−0.1758, −0.1149] SEP | −0.1310 [−0.1605, −0.1003] SEP | **−0.1318 [−0.1614, −0.1010] SEP** | ⭐ stronger |

⭐ **The PSS guards reproduce the sibling's numbers exactly (+0.2992 / −0.1454)** — which is the
positive control that this work did not touch the PSS composite. `lat_heading` is not in
`pseudosim.COMPONENT_WEIGHTS`, and the panel was recomputed to prove it rather than to assert it.

⛔ **Control-composite LEVELS are on a NEW suite id** and are not comparable to any published
control value: every scorable arm rises **+0.0145 to +0.0176** simply because a plan exactly at the
heading tolerance now scores 1/2 instead of 0 (`stand_still` moves **+0.0000** — its `lat_heading`
is undefined on every row, by the anti-standing-still mask). **Only orderings and paired deltas carry across.**

---

## 8. ⚠️ IS THE COMPOSITE SOUND END TO END? — the plain statement

**No, and here is exactly what remains.**

| term | status |
|---|---|
| ✅ `ego_progress@twosided_v2` | **over-travel charged, under-travel charged, admitted by gate v2** — but see the two residuals below |
| ✅ `recovery@twosided_v2` | fixed 2026-07-28, 8/8, live_frac 0.9161, admitted by v2 |
| ✅ `lat_heading@mean1_lin_q0p5` | **fixed here**, 10/10, live_frac 0.9975, admitted by v2 |
| ✅ `lon_track`, `lat_track` | audited, never defective, live_frac 0.89 / 0.91 |
| ⚠️ `comfort` | weight 0, 100 % saturated **by construction**, refused by v2, decision at §5 |

**What remains, and none of it is hidden:**

1. ⛔ **`ego_progress@twosided_v2` still fails one zero-mean cell** on an already-over-travelling
   arm (§4.2). No term in the grid scores 12/12. **This is the one open defect in a weighted term
   of the live primary.**
2. ⚠️ **`ego_progress`'s UNDER side (`r ≤ 0`) has never been audited at all.** It floors on
   **31.78 %** of `v4_blind`'s rows and no injection suite in this programme has ever tested it. It
   is the published `clamp_v1` floor and it predates both repairs. **Named here for the first
   time.**
3. ⚠️ **The shipped `lat_heading` still floors** on 0.14 % of rows and is **not exactly**
   translation-invariant (§2.5a).
4. ⛔ **`lon_retime(0.5)` raises `lat_heading` on every term**, +0.0361 under the shipped one —
   a longitudinal degradation improving a lateral axis, **inherited and not fixed** (§2.5b).
5. ⛔ **No collision gate, no TTC, no map.** `PSS` is not a Driving Score. 2 s horizon, non-reactive
   log replay, lateral grid axis refused — all inherited and bounding.

⇒ **The control axes are unblocked for the gate-primary decision** (the brief's condition was
`lat_heading`, and it is treated, guarded and versioned). **The composite is not sound end to end**,
and item 1 is the reason.

---

## 9. What ships

| file | change |
|---|---|
| `taniteval/taniteval/control.py` | ⭐ `LAT_HEADING_TERM_PUBLISHED/_DEFAULT/_DEFAULT_TARGET/_ALIASES`, `LAT_HEADING_TERMS` (**3 resolutions × 20 shapes = 60 terms**, + 1 alias; **24** of them swept end-to-end), `LAT_HEADING_RESOLUTIONS`, `LAT_HEADING_ANCHOR_GRID`, `_lath_linear/_share/_cos`, `lat_heading_from_err`, `lat_heading_axis_id`, `UnknownLatHeadingTerm`, `AXIS_N_SUB`, `CONTROL_GATE_VERSION = "v2"`; `residuals` emits `heading_err_rad_steps` `[n, K]`; `axes(lat_heading_term=…)` + `lat_heading_terminal` twin; `SUITE_ID(lat_heading_term=…)` **backward-compatible**; `axis_summary` / `panel_gate` / `dynamic_range` / `block` thread the term and the gate version |
| `taniteval/taniteval/pseudosim.py` | ⭐ `LIVE_FRAC_MIN`, `GATE_VERSIONS`, `GATE_VERSION_DEFAULT/_PUBLISHED`, `UnknownGateVersion`; `saturation(n_sub=…)` emits `live_frac` + `row_resolution` and warns on **combined** saturation; `discriminative_range(gate_version=…, n_sub=…)` emits `admissible_v1`/`admissible_v2` always; `twosided_asym_w0p3333` sensitivity anchor |
| `stack/tanitad/train/heldout_gate.py` | probe records carry `gate_version` (pinned `v1`) and `components_admissible_under_gate_v2` |
| `taniteval/tests/test_bounded_terms.py` | ⭐ **NEW — 31 tests**, 12 of which pin a FAILING value |
| `taniteval/tests/test_control.py` | ⚠️ **2 sibling tests updated** — see §11 limitation 1 |

**Suites:** `taniteval/` **757 passed, 0 skipped** (was 726 + **31 new**, **zero new skips**);
`stack/` **1576 passed, 12 skipped** — **unchanged**, zero failures, zero new skips.
🔒 **Parity untouched** — no episode re-selected; all 20 arms share the identical 15,981 rows and
row identity is **asserted**, not assumed. No clip UUID or raw PhysicalAI content appears in any
artifact — counts only.

---

## 10. ⭐ ESCALATIONS — raised here, not left in a README

| # | what needs a decision or a cross-stream change | owner |
|:--:|---|---|
| **E-1** | ⭐⭐ **THE SOURCE STREAM'S E-2 IS ANSWERED: `lat_heading` IS TREATED AND THE CONTROL AXES CAN ENTER THE GATE-PRIMARY DECISION.** `LAT_HEADING_TERM_DEFAULT_TARGET = "mean1_lin_q0p5"` was chosen by a rule fixed before the numbers; the full 24-term sweep is published and 7 other terms also pass. ⚠️ **The `q` is a PI decision I have taken provisionally**, exactly as the sibling did — one constant, pinned by a test. ⚠️ **And the WEIGHT must be re-derived, not inherited:** `CONTROL_WEIGHTS` puts `lat_heading` at 1.0, but the shipped term's live range is **half** the published one (`q = 0.5`), so weight 1.0 now means ~half of what the proposal assumed. | **PI + `taniteval` maintainer — BEFORE the gate-primary change** |
| **E-2** | ⛔⛔ **`ego_progress@twosided_v2` FAILS A ZERO-MEAN INJECTION on an already-over-travelling arm** (+0.0104 SEP, the wrong way). It is the **only open defect in a weighted term of the live primary**. No grid member scores 12/12; `w = 0.5` and `w = 1/3` fix the sign at the cost of halving the over-travel charge rate, and changing `OVER_TRAVEL_WEIGHT` moves **every published PSS level**. ⇒ **a PI decision with both outcomes measured** (T11). | **PI + Benchmarks & Eval** |
| **E-3** | ⚠️ **`ego_progress`'s UNDER side (`r ≤ 0`) has NEVER been audited** — floored on **31.78 %** of `v4_blind`'s rows, and no injection suite in this programme has tested it. It predates both repairs. A reversing plan at −1 m and at −10 m score identically. | **Benchmarks & Eval — next bounded-term pass** |
| **E-4** | ⛔ **`lon_retime(0.5)` RAISES `lat_heading` under EVERY term** (+0.0306 published → **+0.0361 shipped**, +18 %): a purely longitudinal, path-preserving degradation improving a lateral axis. Inherited, **not fixed here**, and the one place the new term is weaker. It is a *definedness/matching* artefact, not a clamp artefact — the matched reference segment moves when the plan's arc does. | **Benchmarks & Eval** |
| **E-5** | ⛔ **ONE PUBLISHED NUMBER IS MISATTRIBUTED AND SHOULD BE CORRECTED IN PLACE.** `RECOVERY_TWOSIDED.md` §8.3 / E-3 reads *"`ego_progress@twosided_v2` is still one-sided above ratio 2 — floored on 32.70 % of `v4_blind`'s rows"*. **97.2 % of that 32.70 % is the UNDER side (`r ≤ 0`), not `r ≥ 2`**; the over-side share is **0.93 pp**, and the worst realisable arm is **8.22 %**. The fraction is right, the mechanism attached to it is not. | **Doc-correction sweep + `RETRACTION_LOG` owner** |
| **E-6** | ⭐ **A NEW RETRACTION CLASS, offered for `RETRACTION_LOG.md` — I did not append it myself.** ⛔ **`A SATURATION FRACTION QUOTED WITHOUT DECOMPOSING WHICH CLAMP PRODUCED IT IS NOT EVIDENCE ABOUT THAT CLAMP`** (see E-5), **and its sibling: `A ONE-SIDED SATURATION THRESHOLD CANNOT SEE A TERM THAT SPLITS ITS SATURATION`** — 49 % floor + 49 % ceiling clears both 0.95 gates with gradient on 2 % of rows, at *any* threshold value. *Detection heuristic: report `live_frac = 1 − floor − ceiling` and the term's `n_sub` beside every bounded score; decompose any floor fraction by the region of the raw quantity that produced it.* | **PI / `RETRACTION_LOG` owner** |
| **E-7** | ⚠️ **C47 IS BOUNDED, and the bound should be written into the class rather than left in this report.** MEASURED: `share_q0p5` scored **0/8** on `recovery` (median ratio 1.181, 75.41 % past the anchor) and **10/10** on `lat_heading` (median `u` 0.9103, 46.29 % past it). The pass rate is **NOT** monotone in the reward bias here. The invariant that survives is *"the charge rate must not collapse where the data lives"*; the reward-bias proxy only predicts that when the median sits **above** the anchor. | **PI / `RETRACTION_LOG` owner** |
| **E-8** | ⚠️ **Downstream JSON consumers gain fields; nothing is removed.** `saturation` nodes now carry `live_frac`, `n_sub_per_row`, `row_resolution`; `discriminative_range` nodes carry `live_frac`, `admissible_v1`, `admissible_v2`, `_gate.gate_version`; `panel_gate` carries `gate_version` and `admitted_under_gate_v1`; `heldout_gate` probe records carry `gate_version` and `components_admissible_under_gate_v2`. | **`stack/` maintainer** |

---

## 11. Everything that is wrong with this work, stated by me

| # | limitation | status |
|:--:|---|---|
| 1 | ⛔ **I EDITED TWO SIBLING TESTS** in `taniteval/tests/test_control.py`. Each broke for a real reason and each was fixed by **preserving the original guarantee under the term it was written for** and adding the new one. `test_a_pure_offset_moves_placement_and_NOT_heading`: the `> 0.999` bar is preserved verbatim under `term_lin_q0`, the claim is re-asserted for the shipped term in the only form that survives a range change (offset response < ⅓ of the heading response), **and** I added the rejected `mean` resolution as a FAILING value so H0 cannot lose its teeth. `test_a_rotation_moves_heading`: the `−0.5` bar is preserved verbatim under `term_lin_q0` — it is arithmetically impossible for a `q = 0.5` term — and the DIRECTION is now asserted under **every one of the 61 terms** in the registry. ⇒ **a reviewer should confirm the strengthened forms are what the pins' authors intended.** | ⛔ disclosed, deliberate |
| 2 | ⛔ **H0 WAS ADDED MID-RUN, AFTER A SMOKE RUN, AND IS NOT PRE-REGISTERED.** It is the axis's own published defining claim rather than a new preference, and it *disqualifies* rather than promotes (it kills 8 candidates including the highest-live_frac ones), but it was written after I had seen `B = 40` numbers. Stated in the rule dict itself, not only here. | ⛔ **disclosed — the weakest methodological point in this work** |
| 3 | ⚠️ **`q` IS A CHOICE, not a measurement**, selected by rule from a 5-point grid over 3 families × 3 resolutions. The whole grid is published. No `q` is derived from data about what a heading error *costs*, because this surface has **no collision gate and no cuboids**. A cost-calibrated `q` is impossible here, not merely absent. | disclosed, swept |
| 4 | ⛔ **THE SHIPPED TERM STILL FLOORS**, on up to **0.14 %** of defined rows, and is **not exactly translation-invariant** — 2 of 6 translation cells separate at \|Δ\| ≤ 0.0035 (the published term's are ≤ 0.0012). It is *dominantly* heading-sensitive, not immune. | ⚠️ a real residual |
| 5 | ⛔ **I DID NOT FIX `ego_progress`**, deliberately (§4.3), and it fails one zero-mean cell. *"The composite is sound"* may not be written. | ⛔ open, escalated |
| 6 | ⛔ **`lon_retime(0.5)` contamination is WORSE under the shipped term** (+0.0306 → +0.0361). This is the only measured respect in which the new term is weaker than the published one. | ⛔ disclosed |
| 7 | ⚠️ **A `B = 40` SMOKE RUN DISAGREED WITH THE DECISION-GRADE RUN** on `term_lin_q0` (6/10 vs **5/10**) and on `mean_lin_q0` (9/10 both). Only `B = 2000` counts, and the disagreement is recorded rather than dropped — the sibling logged the same class of disagreement. | disclosed |
| 8 | ⚠️ **The 10 injections are a NEW suite, not the sibling's**, because the sibling's are structurally unable to move this axis (§2.3). That is necessary but it means the two repairs are **not** validated by literally the same experiment. The contamination panel *is* the sibling's four controls, so there is overlap. | disclosed |
| 9 | ⚠️ **Only 2 of the 20 arms carry the pass-rule injections.** The saturation census covers all 20; `v4_blind` is run as a reported stress substrate and is **not** part of the pass rule (requiring gradient at its median `u` of 6.07 would force `q > 0.835` and crush the band to 16 %). | disclosed |
| 10 | ⚠️ **`human_replay` scores 0.9913, not 1.0000**, under the mean resolution — a handful of degenerate rows where the reference path reverses (`max |per-step dpsi| = π`). Its `ego_progress` is 0.9914 on the same rows, so this is the surface's noise floor rather than a heading artefact — but the zero-bias reference is not exactly 1 and I will not round it up. | disclosed |
| 11 | ⛔ **No model was run and no checkpoint was scored.** Every number is arithmetic over dumps produced by other streams; their fidelity is `INHERITED`. | by rule (pod1/pod2 forbidden) |
| 12 | ⛔ **No collision gate, no TTC, no map.** Not a Driving Score; not comparable to PDMS. | inherited blocker |

---

## 12. Self-refutations

| # | what | status |
|:--:|---|---|
| 1 | ⛔⛔ **MY PRE-REGISTERED C47 PREDICTION FAILED.** I predicted the pass rate would be monotone in the reward bias and committed both outcomes in advance. `share` (bias 3.332) tied `cos` (0.072) at 10.0 and both beat `lin` (1.000) at 9.4. The form that scored **0/8** on `recovery` scores **10/10** here. ⇒ C47 is **bounded**, and the bound is the density, not the algebra. | ⛔ corrected (§2.6) |
| 2 | ⛔⛔ **THE ZERO-PARAMETER ANSWER I EXPECTED TO SHIP LOST ITS OWN ACCEPTANCE TEST.** `mean1_lin_q0` — resolution only, published shape verbatim — is what H3 prefers. **9/10.** Had I shipped it on the "it introduces no parameter, so it cannot be fitted" argument — which is the argument anyone would make — I would have published a term that still cannot separate a real steering drift. | ⛔ corrected (§0 result 2) |
| 3 | ⛔⛔ **MY FIRST RESOLUTION FIX DESTROYED THE AXIS'S REASON TO EXIST, and the pass rate would never have shown it.** `mean_lin_q0p5` scores 10/10 with the sweep's best live_frac — and a pure translation moves it 3.4× more than the smallest heading degradation. Only the contamination panel caught it, and only because the sibling's four "useless" controls were run anyway. | ⛔ corrected (§2.5) |
| 4 | ⛔ **I MISATTRIBUTED A NUMBER I INHERITED — and it was in a report I read as the method.** *"Floored on 32.70 % of `v4_blind`'s rows, still one-sided above ratio 2"* is 97.2 % the **under** side. I very nearly designed an over-travel fix against a number that was measuring something else. | ⛔ corrected (§4.1), escalated as E-5 |
| 5 | ⚠️ **I initially treated `FLOOR_FRAC_MAX = 0.95` as "just too loose" and was going to lower it.** That would have (a) broken the reproduction gate by dropping `recovery@clamp_v1` panel-wide and (b) **still missed the 49/49 case**, which no threshold on a one-sided statistic can catch. | ⛔ corrected (§3.1) |
| 6 | ⚠️ **I first wrote the acceptance suite with the sibling's 8 injections** before checking whether they can move this axis. They cannot — `control.py`'s own docstring says so, and it took reading the module rather than the report to see it. A suite the candidate cannot fail would have passed every shape including the defect. | ⛔ corrected (§2.3) |

---

## 13. Deliverable manifest

Repo dir:
`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-28-bounded-terms-complete/`
Everything `git add`-ed into the working tree. ⛔ **I did not commit and did not push.**
⚠️ marks anything living in only ONE place — **there is nothing in that state.**

| artifact | where it lives | what it is |
|---|---|---|
| `BOUNDED_TERMS_COMPLETE.md` | `repo:` this dir | this report |
| `code/run_bounded_terms.py` | `repo:` this dir | the whole run: density, the acceptance test, the guard demonstration, the reproduction gate, the panel, the `ego_progress` injections, the `comfort` evidence, the term audit. **No GPU, no model, no corpus.** Asserts row identity and the zero-bias reference; ⛔ md5-stamps the `ci.py`/`control.py`/`pseudosim.py` actually loaded (C44) |
| `code/tables.py` | `repo:` this dir | regenerates T1–T13 from the raw JSON — the tables are generated, not hand-typed |
| `raw/density.json` | `repo:` this dir | ⭐ the densities that decided the design + the `psi_tol` refutation + the `ego_progress` floor decomposition |
| `raw/injections_lat_heading.json` | `repo:` this dir | ⭐⭐ **THE ACCEPTANCE TEST** — 5 injections × 2 arms × 24 candidate terms, paired CIs, the stress substrate, the contamination panel, the pre-registered rule and the failed C47 prediction |
| `raw/guard.json` | `repo:` this dir | ⛔ what gate v1 admits and gate v2 refuses, all 20 arms |
| `raw/repro_gate.json` | `repo:` this dir | ✅ 16 published composites at `max\|diff\| = 0.000000` + `lat_heading` bit-identity |
| `raw/panel.json` | `repo:` this dir | the PSS and control panels, the ranking statement, 10 paired contrasts, the two instrument guards |
| `raw/injections_ego_progress.json` | `repo:` this dir | ⚠️ 4 over-travel injections × 3 arms × 5 progress terms — where E-2's single failing cell lives |
| `raw/comfort.json` | `repo:` this dir | ⚠️ the 100 %-saturation evidence and the publish/retire decision |
| `raw/audit.json` | `repo:` this dir | the full bounded-term audit on `live_frac`, v1 vs v2 side by side |
| `raw/run_all.log` | `repo:` this dir | the run's own stdout |
| `artifacts/tables.md` | `repo:` this dir | the generated tables T1–T13 |
| **`taniteval/taniteval/control.py`** | `repo:` | ⭐ the versioned `lat_heading` term, 3 resolutions × 8 shapes, per-step heading, `AXIS_N_SUB`, gate v2 on the control surface |
| **`taniteval/taniteval/pseudosim.py`** | `repo:` | ⭐ `live_frac`, the versioned gate, row-resolution reporting, the `w = 1/3` anchor |
| **`taniteval/tests/test_bounded_terms.py`** | `repo:` | ⭐ **NEW — 31 tests**, 12 pinning a FAILING value |
| `taniteval/tests/test_control.py` | `repo:` | 2 sibling tests updated and strengthened (limitation 1) |
| `stack/tanitad/train/heldout_gate.py` | `repo:` | `gate_version` + `components_admissible_under_gate_v2` in every probe record |

**Reproduce everything, no GPU** (**777 s** on the dev box, two banked decision-grade runs):
```
python3 code/run_bounded_terms.py --n-boot 2000 \
  --in-dir "<…/Benchmarks & Eval/…/2026-07-27-pseudosim-arm-panel/artifacts>" \
  --in-dir "<…/Architecture & Inference/…/2026-07-28-tactical-action-input/artifacts/pw>" \
  --in-dir "<…/Architecture & Inference/…/2026-07-28-tactical-action-input/artifacts/blockA>" \
  --out-dir raw
python3 code/tables.py raw > artifacts/tables.md
```
