# THE UNTESTED HALF PASSES ON EVERY REAL ARM AND IS ABSOLUTELY BLIND ON A REVERSING ONE — AND THE OPEN OVER-SIDE DEFECT IS A FORCED TRADE, NOT A MISSING SHAPE

**Wall-clock date:** 2026-07-28 (Europe/Berlin) · **Stream:** Benchmarks & Eval ·
**Branch:** `agent/benchmarks-eval-20260721` · **Repo HEAD at start:** `29038cb`
**Hosts:** dev box only. Every number is CPU arithmetic over committed dumps. **0 GPU-h.**
⛔ **pod1 (TRAINING 23,000/30,000) and pod2 (small validation, arm B_wide) were NOT touched —
not even read.** pod3 and `tanitad-eval` were not used: nothing here needs a GPU.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not
re-verified) · `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

**Source:** `…/2026-07-28-bounded-terms-complete/BOUNDED_TERMS_COMPLETE.md` escalations **E-2**
(the open zero-mean defect) and **E-3** (the never-audited under side); retraction classes
**C45** (the one-sided clamp), **C46** (`comfort`), **C47** (shape vs density, *as amended today*).

**Pre-registration:** `raw/PREREGISTRATION.json`, staged **before any under-side number existed**.

---

## 0. Headline

| # | Result | Class · tier |
|:--:|---|---|
| **1** | ⭐⭐ **THE UNDER SIDE PASSES ON EVERY REALISABLE ARM — FIRST AUDIT EVER — AND IS ABSOLUTELY BLIND ON A REVERSING PLAN.** 17 of 18 controls cleared a metric-independent direction check; across **all 18 terms × those 17 cells, ZERO are separated in the wrong direction**, and the published `clamp_v1` is **17/17** correct. ⭐ **And the single wrong-way separation in the entire 324-cell under-side panel sits on the ONE cell the direction rule REFUSED** (§2.2). ⛔ But on a plan reversing on **99.94 %** of rows, **DOUBLING the reversal — ground truth +13.59 m, 39.95 → 53.54 m — moves the score `+0.0001` [−0.0000, +0.0002] n.s., the WRONG SIGN**, and 5 m more reversal moves it `−0.0002` n.s. **Bit-identically under `clamp_v1` and `twosided_v2`.** | `MEASURED` **tier 1** |
| **2** | ⛔⛔ **AND THAT BLINDNESS IS ARITHMETICALLY UNFIXABLE — the only lever pays a plan that does not move, AND weakens the guard.** Any bounded `g` agreeing with `clamp_v1` on `[0,1]` has `g(0)=0` and is forced to `0` below — the exact structure of C47's impossibility proof for `recovery`. The only escape is a range budget `g(0)=p>0`. **MEASURED price:** `stand_still` rises **0.000000 → p exactly**, and at `p = 0.10` `v4_blind` gains **+0.0355** against `cv_holdv0`'s **+0.0059**, so `v4_oracle − v4_blind` on this axis falls **0.3464 → 0.3162 (−8.7 %)**. ⇒ **NAMED, PRICED, REFUSED — on the number, not the argument.** | `MEASURED` **tier 1** |
| **3** | ⛔⛔ **THE CELL THAT DEFINES "THE ONE OPEN DEFECT IN THE LIVE PRIMARY" IS NOT A VERIFIED DEGRADATION.** `v1_ego_double × lon_jitter(σ=2 m)`: the metric-independent ground truth `\|s_plan − s_human\|` moves **+0.0101 m [−0.0258, +0.0477] n.s.** on a **24.9 m** baseline error. Under my own pre-registered rule U-0 that control is **REFUSED**. ⚠️ **The defect is still real and I am not explaining it away** — a metric answering a perturbation the ground truth cannot detect with a **SEPARATED +0.0104** is spurious sensitivity of the wrong sign, and its mechanism is independently demonstrated in result 1 — **but its published magnitude was measured against a near-null perturbation, and that was never checked.** | `MEASURED` **tier 1** |
| **4** | ⭐⭐ **THE OVER-SIDE REPAIR IS A FORCED TRADE, AND THAT IS A THEOREM.** `twosided_v2` spends its **entire** remaining range on `[1, 2]` at rate 1, so any monotone bounded term identical to it on `[0, 2]` is identical **everywhere**. ⇒ **every repair of the zero-mean cell necessarily reduces the over-travel charge rate somewhere below `r = 2`.** "It halves the charge rate" is not a flaw of `w = 0.5`; it is **arithmetic**, and the previous stream's reason for escalating was therefore not a failure of search. Pinned by `test_preserving_the_published_charge_rate_below_r2_FORCES_the_r2_floor`. | `MEASURED` **tier 1** |
| **5** | ⭐ **AND THE TRICHOTOMY SAYS WHICH WAY EACH SHAPE MUST ERR.** **Convex** tail (the only way never to floor) ⇒ zero-mean jitter **REWARDED** by Jensen. **Linear** tail ⇒ it costs **EXACTLY nothing** (`E[g(r+δ)] = g(r)`), so such a term can only be n.s., never separated-correct. **Concave** tail ⇒ **charged** — but concavity with `g(1)=1, g'(1⁺)=−w` forces the floor at `r ≤ 1 + 1/w`, i.e. **a concave term charging at the published rate MUST floor by `r = 2`.** The frontier is `rate(1⁺) ≤ 1/(R−1)`. | `MEASURED` **tier 1** |
| **6** | ⭐ **TWO TERMS SCORE 12/12 AND BOTH ARE RESOLUTION CHANGES.** `mean\|sqrtlin_w0p3333` (failing cell **−0.0379 SEP**) and `mean\|hyp_w1` (**−0.0062 SEP**). ⛔ **No terminal-resolution term reaches 12/12** — 12 of 14 sit at 11/12 with the cell merely n.s., which is exactly what result 5 predicts for a linear or convex tail. | `MEASURED` **tier 1** |
| **7** | ⭐⭐ **THE BRIEF'S FREE LEVER TRANSFERS ON ONE SIDE AND INVERTS ON THE OTHER.** The 20-step mean cuts `v4_blind`'s floor **0.3270 → 0.0010 (327×)**, lifts its `live_frac` **0.6704 → 0.9990**, and multiplies the under-side charge by up to **3.5×** — and it makes the over-side wrong-way leak **5.7× WORSE** (`mean\|twosided_v2` **+0.0104 → +0.0590 SEPARATED**), while the resolution's lateral-purity margin collapses (`clamp_v1` **246× → 2.6×**, `twosided_v2` **58.5× → 7.7×**). ⇒ **the lever is not free here, and the disanalogy is definitional: heading error is equally meaningful at every step; cumulative displacement is not — progress given back is not progress.** | `MEASURED` **tier 1** |
| **8** | ⛔⛔ **NO CANDIDATE MAY SHIP: EVERY ONE OF THE NINE WEAKENS THE INSTRUMENT GUARD, AND ONLY `twosided_v2` KEEPS THE PUBLISHED RANKING CLAIM.** `v4_oracle − v4_blind` is **maximal at the shipped term (+0.2992)** and falls for all nine (`hyp_w1` +0.2957 … `mean\|clamp_v1` +0.2577). And *"the v1 tactical family sits below every REF-C arm"* is **TRUE only under `twosided_v2` and `mean\|twosided_v2`** — it is **FALSE under `clamp_v1` itself**. ⇒ **the zero-mean cell can be closed, and every measured way of closing it makes the metric more permissive.** The brief's own bar refuses all of them. | `MEASURED` **tier 2** |
| **9** | ✅ **REPRODUCTION GATE EXACT: `max\|diff\| = 0.000000` on the 16 published `@clamp_v1` composites**, with 9 new progress terms, a new control and a new per-step primitive in the library. Every new term is **BIT-identical** to `clamp_v1` for every `r ≤ 1` on all 20 arms. Published id `PSS_recovery_progress@clamp_v1` unchanged; default id unchanged. **Fourth consecutive repair to meet this bar.** | `MEASURED` **tier 1** |
| **10** | ⛔ **I FOUND A CORRECTNESS BUG IN MY OWN RUNNER — AND THE REPRODUCTION ASSERTION IS WHAT CAUGHT IT.** A memo keyed on `id(pw)` handed one plan's scores to another after CPython reused the freed address. The first decision-grade pass reported `clamp_v1` **7/12** and `twosided_v2` **8/12**. After the fix: **2/12** and **11/12** — the inherited values, and the failing cell reproduces at `abs_diff = 0.000000`. **A cache keyed on a reusable identity is a correctness bug, not a performance detail.** | `MEASURED` **tier 1** |
| **11** | ⚠️ **THE `lat_heading` WEIGHT: THE BRIEF'S PREMISE IS BACKWARDS, MEASURED.** *"The shipped term's live range is half the published one ⇒ weight 1.0 now means ~half."* Within-arm, yes (pooled sd **0.3359 → 0.2505**). But a composite aggregates **arm means**, and the shipped term's **BETWEEN-ARM span is 1.2047× LARGER** (0.2594 → 0.3125). ⇒ the influence-preserving weight is **0.83, not 2.0**. **Re-derived, escalated, NOT changed.** | `MEASURED` **tier 2** |

### 0.1 Pre-registered outcome

**CONFIRM on U-2 and U-6; ⛔ REFUTE on my own U-3; OPEN on E-2.**

| pre-registration | outcome |
|---|---|
| **U-1** *"a constant slowdown lowers `ego_progress`, separated, under both terms"* | ✅ **FIRED.** 18 slowdown cells (3 slowdowns × 3 arms × 2 terms): **18/18 correct in sign, 17/18 separated.** The single n.s. is `twosided_v2 × v1_tactical_follow × lon_retime(0.7)` (−0.0322), an arm whose median `r` is **above** 1 — see §2.3. **This had never been measured, only asserted in a docstring.** |
| **U-2** *"a further reversal is charged NOTHING"* | ✅ **FIRED, exactly.** +0.0001 n.s. for a doubled reversal; the same 42 rows score `0.00000000` at −1 m and at −30 m. |
| ⛔ **U-3** *"a zero-mean `lon_jitter` RAISES `ego_progress` on `v4_blind`, and more under `clamp_v1` than under `twosided_v2`"* | ⛔⛔ **DID NOT FIRE, IN BOTH CLAUSES.** MEASURED **−0.0070 SEP** (`clamp_v1`) and **−0.0135 SEP** (`twosided_v2`) — correct direction, and the magnitude ordering is the **reverse** of my prediction. ⇒ ⭐ **31.78 % under-floor is NOT enough to flip a zero-mean cell; the live 68 % dominates. It took ~100 % floor.** §2.4. |
| **U-4** *"`ego_progress` definedness is plan-invariant"* | ✅ **ASSERTED bit-exactly on all 324 under-side cells (18 terms × 3 arms × 6 injections), 0 violations.** ⇒ the *"a plan that barely moves scores `recovery = NaN`"* hazard the brief flagged **cannot touch this axis**: its NaN mask is a property of the LOGGED path. It touches the composite, and `defined_frac` is published on every composite cell. |
| **U-5** *"the same jitter is not rewarded on `cv_holdv0`"* | ✅ **FIRED**, −0.0246 (`clamp_v1`) / −0.0468 (`twosided_v2`), both SEP. |
| **U-6** *"the range budget is refused because it pays standing still"* | ✅ **FIRED**, and it also weakens the guard — a second, independent reason I had not predicted. |
| **U-7** *"the resolution lever has a small-denominator pathology"* | ✅ **FIRED**, and worse than predicted: it is not a curiosity, it **inverts the over-side verdict** (result 7). |
| **O-2** *"the over-side grid is a TRADE, not a search"* | ✅ **FIRED**, and was upgraded from a prediction to a **theorem** (result 4) and a **measured guard refusal** (result 8). |

### 0.2 Tier and inherited qualifiers

**Tier 1** for §1–§7 (deterministic CPU arithmetic over committed dumps; no model, no checkpoint,
no corpus). **Tier 2** for any statement about a named ARM, which inherits the panel's qualifiers
verbatim: **oracle goal where used**, **non-reactive log replay**, **no collision or TTC gate**,
**`comfort` weight 0**, plus the source stream's two (Block B arms are **plan transforms, not
trained planners**; two arms are **oracles**).
⛔ **Nothing here is a Driving Score** and nothing here is compared to a PDMS number.

---

## 1. The under side, restated exactly

```python
human  = |ref_path[-1] − ref_path[0]|          # the human's 2 s chord
ego    = x[:, -1]                              # the plan's along-track at 2 s
ratio  = ego / max(human, 1e-3)
score  = clamp(ratio, 0, 1)                    # clamp_v1  — floor at r ≤ 0, ceiling at r ≥ 1
score  = clamp(clamp_v1(r) − w·max(r−1,0), 0, 1)   # twosided_v2 — floor at r ≤ 0 AND r ≥ 2
```

⛔ **The two terms are BIT-identical for every `r ≤ 1`.** `twosided_v2` was built as a strict
refinement on the over side and **never claimed the under side**. So `r ≤ 0` — a plan that ends
**behind where it started** — has carried the published `clamp_v1` floor from the beginning, and no
injection suite in this programme has ever probed it. That is E-3, and this is its audit.

### 1.1 The density — and one correction to a number I was handed

`MEASURED`, `raw/under_density.json` (T1):

| | |
|---|---|
| ⛔ `v4_blind`'s under-floor | **31.78 %**, and **all of it is STRICTLY NEGATIVE** (`r = 0` exactly: 0.0000) |
| how far back it goes | median `r` over those rows **−0.134**, min **−0.657**, **81.9 %** of them below −0.10 |
| ⚠️ every other **scorable** arm | **≤ 0.41 %**, and for the `v1_ego_*` / `cv_holdv0` family the `r ≤ 0` rows are **exactly 0.0000**, not negative |
| ⭐ the shared artefact that explains those, **verified not assumed** | **0.4986 % of `cv_holdv0`'s defined rows carry an identically-zero plan** and **0.5893 % have `v0 < 0.05 m/s`** — the ego is stationary in the log, so a hold-`v0` plan is the zero vector and its along-track endpoint is exactly 0. A property of the CORPUS, not of any arm: the same 0.0050 appears on `cv_holdv0`, `v1_ego_half`, `v1_ego_double` and `v1_ego_v0`. |
| `stand_still` | **1.0000 at exactly `r = 0`** — the degenerate reference, used only to price the fix |

⇒ **the under floor is a real, single-arm phenomenon**: `v4_blind`'s plans genuinely reverse on
nearly a third of windows, and nothing else on the panel does.

---

## 2. ⭐⭐ THE UNDER-SIDE AUDIT

### 2.1 ⭐ The control the suite did not have — and why the floor was unreachable without it

⛔ **Neither existing longitudinal control can push a row that is already at `r ≤ 0` further
backwards.** `lon_scale` **multiplies** (`0 · k = 0`), and `lon_retime` re-samples along the plan's
**own arc** (a zero-length arc stays at zero). Pinned by
`test_lon_shift_is_the_control_the_under_side_audit_REQUIRED`.

⇒ **`lon_shift(d)`** was added to `control.CONTROLS`: a constant along-track offset, the exact twin
of `lat_shift`, whose zero-mean counterpart `lon_jitter` already existed. **The along axis simply
never had its constant-sign member, and that — not oversight alone — is why the under side went
unaudited.**

### 2.2 ⭐⭐ Every control's direction is VERIFIED, never assumed — and one cell failed

The brief's demand, made operational: a control counts as an under-travel **degradation** on an arm
only if it raises the **metric-independent** along-track endpoint error `|s_plan − s_human|`
(paired episode-cluster bootstrap, B = 2000, SEPARATED). `MEASURED`, T3:

| | |
|---|---|
| verified | **17 of 18** |
| ⛔ **REFUSED** | **`v1_tactical_follow \| lon_shift(−2 m)`** — ground truth **+0.0216 m [−0.2564, +0.3264] n.s.** |

⭐⭐ **AND THE ONLY WRONG-WAY SEPARATION IN THE ENTIRE 324-CELL UNDER-SIDE PANEL IS ON EXACTLY THAT
CELL.** `MEASURED`:

| term, on the REFUSED cell `v1_tactical_follow \| lon_shift(−2 m)` | Δ |
|---|---|
| ⛔ `twosided_asym_w2` *(a published SENSITIVITY anchor, never a default)* | ⛔ **+0.0268 [+0.0102, +0.0441] SEPARATED THE WRONG WAY** |
| ⚠️ `twosided_v2` *(SHIPPED)* | **+0.0037 [−0.0113, +0.0188] n.s.** — the only positive point estimate among the 18 terms besides the above |
| every other term | −0.0001 … −0.2008, correct sign |

**The re-centring hazard the rule was written for is real, it bit on a real arm, and the rule caught
it before it became a finding.** Mechanism: `v1_tactical_follow`'s median `r` is **1.0407**, so a 2 m
back-shift moves as many rows *toward* 1 as away from it — and the harder a term charges over-travel
(`w = 2` charges it twice as hard as the default), the more a back-shift *helps*.
⇒ ⭐ **without the direction rule this would have been published as a second wrong-way defect in a
weighted term. It is not one; it is a control that does not degrade.**

### 2.3 The acceptance test — 18 terms × 17 verified cells

`raw/inject_under.json`, T4/T5. **Δ > 0 means the DEGRADED plan scores HIGHER.**

| term | correct | ⛔ separated WRONG WAY |
|---|:--:|:--:|
| ⭐ `clamp_v1` *(PUBLISHED)* | **17/17** | **0** |
| ⚠️ `twosided_v2` *(SHIPPED DEFAULT)* | **14/17** | **0** |
| `twosided_asym_w0p5` · `w0p3333` · `hyp_w1` · `hyp_w0p5` · `exp_w1` · `exp_w0p5` · every `sqrtlin_*` | 17/17 | 0 |
| `twosided_asym_w2` | 10/17 | 0 |
| `mean\|clamp_v1` · `mean\|hyp_w1` · `mean\|sqrtlin_w0p3333` | 17/17 | 0 |
| `mean\|twosided_v2` | 16/17 | 0 |

⚠️ **The "WRONG WAY" column counts VERIFIED cells only.** Over the full 324-cell panel — including
the one refused cell — there is **exactly one** wrong-way separation, on `twosided_asym_w2`, on that
refused cell (§2.2).

> ### ⭐ THE UNDER SIDE PASSES.
> **Not one VERIFIED cell, on any term, at any resolution, is separated in the wrong direction.** The
> published `clamp_v1` is correct on every verified cell. `twosided_v2`'s 14/17 is a **power**
> shortfall, not a sign failure: the three n.s. cells are all on `v1_tactical_follow`, an arm whose
> median `r` is **above 1**, where slowing the plan down moves many rows *toward* the target and the
> two-sided term correctly nets out a smaller charge. ⚠️ That is `twosided_v2` behaving as designed,
> and it is why `clamp_v1` — which cannot see over-travel at all — looks "stronger" here. **It is
> not stronger; it is blind in the other direction, and §5 shows it 9-times separated the wrong way
> on the over side.**

### 2.4 ⛔⛔ MY OWN PRE-REGISTERED PREDICTION FAILED, AND IT IS THE MOST USEFUL RESULT IN THIS SECTION

Stated in `raw/PREREGISTRATION.json` before any under-side number: *"on `v4_blind` (31.78 %
under-floor) a **ZERO-MEAN** `lon_jitter(σ = 2 m)` **RAISES** `ego_progress` — the C45 mechanism on
the side nobody has audited — and **more under `clamp_v1`** than under `twosided_v2`."*
`MEASURED`, T5:

| term | `v4_blind × lon_jitter(σ=2 m)`, ZERO-MEAN |
|---|---|
| `clamp_v1` | **−0.0070 [−0.0131, −0.0014] SEP** — correct |
| `twosided_v2` | **−0.0135 [−0.0243, −0.0050] SEP** — correct, and **1.9× LARGER**, the reverse of my ordering |

⭐ **What it teaches, and it generalises past this term:** a floor fraction is **not** a sufficient
statistic for C45's mechanism. `v4_blind` is 31.78 % floored on the under side and the zero-mean
cell still moves the right way, because the **live 68 %** carries a **concave ceiling** at `r = 1`
that charges the same noise harder than the convex floor rewards it. **The two ends of one clamp
bias in OPPOSITE directions and which one wins is a fact about the arm's DENSITY, not about the
term.** ⇒ C45's leak needs the floor to hold a *majority near the kink*, not merely a large minority
— which is exactly what the reversing substrate supplies and no realisable arm does.

### 2.5 ⛔⛔ THE REVERSING SUBSTRATE — where it fails absolutely

`reverse_half = lon_scale(−0.5) ∘ v1_tactical_follow`: a plan travelling **backwards at half the
human's speed**. **99.94 %** of rows at `r ≤ 0`, median `r` **−0.5223**, mean ground-truth error
**39.95 m**. ⛔ **SYNTHETIC — never inserted into `arms`, never votes on a gate or a ranking**, the
same role `v1_ego_double` played for the over side. `MEASURED`, T6:

| injection | ground truth | mean `clamp_v1` | mean `twosided_v2` | paired Δ @`twosided_v2` |
|---|--:|--:|--:|---|
| base | 39.948 m | 0.000206 | 0.000206 | — |
| **`lon_scale(×2)` — DOUBLE the reversal** | **53.540 m (+13.59)** | 0.000340 | 0.000262 | ⛔ **+0.0001 [−0.0000, +0.0002] n.s.** |
| `lon_shift(−5 m)` | 44.899 m (+4.95) | 0.000000 | 0.000000 | −0.0002 [−0.0005, +0.0000] n.s. |
| ⚠️ `lon_jitter(σ=2 m)` *(control REFUSED: gt +0.03 m n.s.)* | 39.978 m | 0.003107 | 0.002165 | ⛔ **+0.0020 [+0.0006, +0.0037] SEP** |

⇒ ⛔ **the axis has NO GRADIENT LEFT.** A 34 % increase in the true error is answered with a
**+0.0001 point estimate of the wrong sign**, and the only cell that separates does so **upward**,
under a control that is not even a degradation. And T2 makes it concrete on **real rows**: 42 rows
of `v1_tactical_follow`, **held fixed**, pushed from a mean ratio of **−1.05 to −37.86** while their
ground-truth error grows **2.216 → 30.956 m** — and their score is
**`0.00000000` at every step, under `clamp_v1`, `twosided_v2`, `sqrtlin_w0p3333` and `hyp_w1` alike.**

---

## 3. ⛔ THE UNDER-SIDE FIX — named, priced, and refused on a number

### 3.1 The impossibility

Any bounded `g: ℝ → [0,1]` agreeing with `clamp_v1` on `[0,1]` has `g(0) = 0`; with `g ≥ 0` and
monotonicity that forces `g ≡ 0` on `(−∞, 0]`. **A strict refinement of the under side is
impossible** — the identical structure to C47's proof for `recovery`, reached independently here.
⇒ the only lever is a **range budget** `g(0) = p > 0`: `g(r) = p + (1−p)·clamp_v1(r)` above 0,
decaying to 0 below. `p = 0` reproduces the published term **bit-identically**. Pinned by
`test_the_under_side_floor_is_ARITHMETICALLY_UNFIXABLE`.

### 3.2 The price, `MEASURED` (T7, `raw/under_fix.json`)

| arm | `p = 0` | `p = 0.05` | `p = 0.10` | `p = 0.25` |
|---|--:|--:|--:|--:|
| ⛔ **`stand_still`** | **0.000000** | **0.050000** | **0.100000** | **0.250000** |
| ⛔ `v4_blind` | 0.599887 | 0.617629 | **0.635371** | 0.688597 |
| `cv_holdv0` | 0.940738 | 0.943701 | **0.946664** | 0.955553 |
| `v4_oracle` | 0.946246 | 0.948908 | **0.951570** | 0.959555 |

> ### ⭐ DECISION: **REFUSED — for two independent measured reasons, not one.**
>
> **(a) It pays a plan that does not move.** `stand_still` rises `0.000000 → p` exactly, on 100 % of
> its rows, for free. This programme has already paid for that trade once: the naive `v0`-based
> `recovery` denominator scored the **BLIND arm +0.597 ABOVE the sighted one** because a planner that
> barely moves has a small cross-track error. *Standing still is not progress.*
>
> **(b) ⭐ AND IT WEAKENS THE INSTRUMENT GUARD, WHICH I DID NOT PREDICT.** The budget is worth
> **6.0× more to `v4_blind` than to `cv_holdv0`** (+0.0355 vs +0.0059 at `p = 0.10`), because the
> budget's value is proportional to how much of the arm sits at the floor. ⇒ `v4_oracle − v4_blind`
> on this axis falls **0.346359 → 0.316199, −8.7 %**. **A fix whose benefit is concentrated on the
> worst arm is a fix that makes the metric more permissive.**
>
> ⇒ the under floor is **left open, and the honest instrument is to PUBLISH it**: `saturation`
> already emits `floor_frac` and `live_frac` beside every bounded score, and `raw/under_density.json`
> now decomposes the floor by the region of `r` that produced it — the E-6 heuristic, applied.

---

## 4. ⭐⭐ THE OVER SIDE — the trade is FORCED

### 4.1 The theorem the PI decision turns on

`twosided_v2` is `1 − (r−1)` on `[1,2]`: it spends its **entire** remaining unit of range there, at
rate exactly 1. So for a monotone bounded `g`:

> **If `g = twosided_v2` on `[0, 2]`, then `g(2) = 0`, hence `g ≡ 0` above 2 — `g` IS the
> defective term.**
> ⇒ ⭐ **Every repair of the zero-mean cell necessarily reduces the over-travel charge rate
> somewhere below `r = 2`.**

⇒ the previous stream's reason for escalating — *"`w = 0.5` and `w = 1/3` fix the sign but **halve
the charge rate**"* — is **not a property of those two candidates**. It is a property of **every
possible candidate**, and no wider search can avoid it. Pinned by
`test_preserving_the_published_charge_rate_below_r2_FORCES_the_r2_floor`, driven over the whole
registry: a term that keeps range above `r = 2` **must** differ from the published one below it.

### 4.2 ⭐ And the trichotomy says which way each shape errs

`test_the_over_side_trichotomy_is_arithmetic_not_a_preference`, driven at 200,000 exactly-zero-mean
draws away from any kink:

| tail | `E[g(r+δ)] − g(r)` for zero-mean `δ` | consequence for the failing cell |
|---|---|---|
| **CONVEX** — the only way never to floor (`hyp_*`, `exp_*`) | **> 0** by Jensen | the jitter is **REWARDED** |
| **LINEAR** (`twosided_*`) | **= 0 exactly** | it costs **NOTHING**; the cell can only be n.s. |
| **CONCAVE** (`sqrtlin_*`) | **< 0** | **charged** — the only class that can pass |

⛔ **And concavity buys the floor forward:** `g` concave with `g(1)=1`, `g'(1⁺) = −w` satisfies
`g(r) ≤ 1 − w(r−1)`, so it floors no later than the linear term at the same rate.
⇒ **`rate(1⁺) ≤ 1/(R_floor − 1)`**, and **a concave term charging at the published rate 1 must floor
by `r = 2`.** Pinned by `test_a_concave_term_charging_at_the_published_rate_MUST_floor_by_r2`.

### 4.3 ⛔⛔ AND THE FAILING CELL'S CONTROL IS NOT A DEGRADATION

The same direction rule, applied to the over side (T8):

| cell | Δ ground truth `\|s_plan − s_human\|` | admissible? |
|---|---|:--:|
| `cv_holdv0 \| lon_jitter(σ=2)` | +0.8509 [+0.7686, +0.9328] SEP | ✅ |
| `v1_tactical_follow \| lon_jitter(σ=2)` | +0.2290 [+0.1810, +0.2796] SEP | ✅ |
| ⛔ **`v1_ego_double \| lon_jitter(σ=2)`** | **+0.0101 [−0.0258, +0.0477] n.s.** | ⛔ **REFUSED** |

**Mechanism, and it is not subtle:** `v1_ego_double` over-travels ~2×, so its along-track error is
already **24.9 m on average (median 20.3 m)**; adding a **zero-mean** 2 m perturbation to a 25 m error raises `E|s+δ|` only at
second order. ⚠️ **I will not use this to make the defect disappear.** A sound bounded metric should
be **n.s.** on a perturbation the ground truth cannot resolve; `twosided_v2` answers it with
**+0.0104 SEPARATED**, and that is spurious sensitivity of the wrong sign whose mechanism §2.5
demonstrates independently. But **the magnitude that has been quoted as "the one open defect in a
weighted term of the live primary" was measured against a near-null perturbation, and nobody
checked.**

### 4.4 The grid, `MEASURED` (T9, `raw/inject_over.json`)

⛔ `overlapping_holdout_se` appears nowhere. 4 injections × 3 arms.

| term | correct | ⛔ WRONG WAY | over-side floor max | the failing cell |
|---|:--:|:--:|--:|---|
| ⛔ `clamp_v1` *(PUBLISHED)* | **2/12** | **9** | 0.0000 | −0.0013 n.s. |
| ⚠️ `twosided_v2` *(SHIPPED)* | 11/12 | **1** | 0.0822 | ⛔ **+0.0104 [+0.0054, +0.0158] SEP** |
| `twosided_asym_w0p5` / `w0p3333` | 11/12 | 0 | 0.0316 / 0.0185 | −0.0013 n.s. |
| `hyp_w1` / `hyp_w0p5` / `exp_w1` / `exp_w0p5` | 11/12 | 0 | **0.0000** | +0.0006 … −0.0004 n.s. |
| `sqrtlin_w1` / `w0p5` / `w0p3333` | 11/12 | 0 | 0.0822 / 0.0315 / 0.0185 | +0.0030 … −0.0016 n.s. |
| `sqrtlin_w0p25` | 9/12 | 0 | 0.0139 | −0.0015 n.s. |
| ⛔ `mean\|clamp_v1` | 3/12 | **9** | 0.0000 | −0.0236 SEP |
| ⛔ `mean\|twosided_v2` | 11/12 | **1** | 0.0626 | ⛔⛔ **+0.0590 [+0.0497, +0.0678] SEP** |
| ⭐ **`mean\|sqrtlin_w0p3333`** | ✅ **12/12** | 0 | 0.0132 | ✅ **−0.0379 SEP** |
| ⭐ **`mean\|hyp_w1`** | ✅ **12/12** | 0 | **0.0000** | ✅ **−0.0062 SEP** |

⭐ **The 12-of-14 pile-up at 11/12 is the trichotomy, visible in the data:** every linear and convex
terminal-resolution term lands with the zero-mean cell **n.s.**, exactly as §4.2 requires, and none
of them can do better *in principle*.
⚠️ **And the price of the late floor is large.** `sqrtlin_w0p3333` charges a 1.5× over-travel on
`cv_holdv0` **−0.0377** where `twosided_v2` charges **−0.3877** — **10× more permissive** on the
over-travel band where most of the panel actually lives.

---

## 5. ⭐⭐ THE RESOLUTION LEVER — it transfers on one side and INVERTS on the other

The brief's ⭐ free lever: `lat_heading` was one value per row **by implementation choice**, and a
20-step mean cut its worst floor **32.4×** with no free parameter. **`ego_progress` has the identical
structure** — `score_windows` reads only `x[:, −1]` over the human's 2 s chord, though the plan has
20 steps and the reference 20 matching poses. `PS.progress_ratios_per_step` publishes the per-step
form, reusing the **published** `PROGRESS_HUMAN_MIN_M = 0.5` verbatim so **no parameter is added**,
and preserving the published row mask **bit-for-bit** (pinned).

| | terminal (published) | 20-step mean | |
|---|--:|--:|---|
| ⭐ `v4_blind` floor | 0.3270 | **0.0010** | **327× better** |
| ⭐ `v4_blind` `live_frac` | 0.6704 | **0.9990** | |
| ⭐ under-side charge, `v4_blind × lon_shift(−5 m)` | −0.0667 | **−0.2339** | **3.5× more** |
| ⭐ under-side charge, `v4_blind × lon_shift(−2 m)` | −0.0214 **n.s.** | **−0.1173 SEP** | terminal **cannot separate it at all** |
| ⛔ the failing over-side cell | +0.0104 SEP | **+0.0590 SEP** | ⛔ **5.7× WORSE** |
| ⛔ lateral-purity margin, `cv_holdv0`, `clamp_v1` pair | 246× | **2.6×** | ⛔ **95× narrower** |
| ⛔ lateral-purity margin, `cv_holdv0`, `twosided_v2` pair | 58.5× | **7.7×** | ⛔ **7.6× narrower** |

⛔ **Why it inverts, and the reason is definitional rather than numerical.** `r_k = x_k / human_k`
has a **small denominator** at early `k`, so a constant along-track offset `e` enters step `k` as
`e / human_k` and explodes at the first defined step — the structural twin of the step-0 plan tangent
that killed the plain-mean `lat_heading` candidate (pinned by
`test_the_mean_resolution_has_a_SMALL_DENOMINATOR_pathology`). More mass lands near and past the
`r = 2` floor, so the clipping bias grows.

⭐ **And the deeper disanalogy, which is the transferable lesson:** `lat_heading`'s raw quantity —
heading error — is **equally meaningful at every step**, so averaging it is a pure resolution gain.
`ego_progress`'s raw quantity is **cumulative displacement**, and the axis is *defined* on the
terminal value: **progress made at 0.5 s and given back by 2 s is not progress.** ⇒ **the resolution
lever is not free when the quantity is cumulative.** That is the check the next bounded term needs,
and it is not "is there a per-step version?" but **"is the per-step version the same quantity?"**

---

## 6. ⚠️ THE CONTAMINATION PANEL

`raw/contamination.json`, T10. `ego_progress` is a **longitudinal** axis, so a purely **lateral**
degradation must move it **least**: `max|Δ|` over `{lat_shift(±2 m), lat_jitter(σ=1 m)}` must be
`< min|Δ|` over the verified longitudinal injections, on **both** real arms.

| term | `cv_holdv0` | `v1_tactical_follow` | pure? |
|---|---|---|:--:|
| `clamp_v1` | 0.0001 vs 0.0246 (**246×**) | 0.0004 vs 0.0086 (21.5×) | ✅ |
| ⭐ `twosided_v2` | 0.0008 vs 0.0468 (58.5×) | 0.0010 vs 0.0037 (**3.7×**) | ✅ |
| ⛔ **`hyp_w2`** | 0.0011 vs 0.0537 (48.8×) | **0.0007 vs 0.0001 (0.1×)** | ⛔ **FAILS** |
| `mean\|clamp_v1` | **0.0128 vs 0.0333 (2.6×)** | 0.0043 vs 0.0559 (13.0×) | ✅ |
| `mean\|sqrtlin_w0p3333` | 0.0152 vs 0.0469 (3.1×) | 0.0050 vs 0.0321 (6.4×) | ✅ |
| `mean\|hyp_w1` | 0.0239 vs 0.1713 (7.2×) | 0.0063 vs 0.0735 (11.7×) | ✅ |

⭐ **The panel earned its place again — it disqualified `hyp_w2`, which passed everything else**
(11/12 over, 13/17 under, zero floor). Its lateral response *exceeds* its smallest longitudinal one
by 7×, which for an axis whose entire purpose is along-track travel is disqualifying.
⚠️ **And it prices the resolution lever a second time:** the terminal terms' lateral leakage is
0.0001–0.0015; the `mean` terms' is **0.0043–0.0265**, up to **30× more**. They stay "pure" by the
ratio, but the margin collapses from **246× to 2.6×**.

---

## 7. ✅ THE REPRODUCTION GATE — exact

`raw/repro_gate.json`, T11.

| | |
|---|---|
| published `@clamp_v1` composites checked | **16** |
| **max \|diff\|** | ⭐ **0.000000** |
| verdict | ✅ **PASS** |
| every new term BIT-identical to `clamp_v1` for every `r ≤ 1` | ✅ **True, all 9 new terms × 20 arms** |
| published metric id, unchanged | `PSS_recovery_progress@clamp_v1` |
| default metric id, unchanged | `PSS_recovery_progress@twosided_v2` |
| `OVER_TRAVEL_WEIGHT`, unchanged | **1.0** |
| `PROGRESS_TERM_DEFAULT`, unchanged | `twosided_v2` |

⛔ **Both synthetic reference arms were kept OUT of the gate vote.** `human_replay` is built and its
`max |residual| = 1.526 × 10⁻⁵ m` **asserted** before use (its `ego_progress` scores **0.9914**,
reproducing the sibling stream's value exactly); `reverse_half` is built only as a metric-test
substrate. Neither is ever inserted into `arms`.

---

## 8. ⛔⛔ THE RANKING STATEMENT, THE GUARDS, AND WHY NOTHING SHIPS

`raw/panel.json`, T12/T13. 20 arms · 40 val episodes · 15,981 rows per arm · row identity
`(ep_i, anchor, dlat, dyaw, dlon)` **ASSERTED across all 20, 0 refused**. ⛔ **PANEL-WIDE gate.**

> **Does `cv_holdv0` still rank first among realisable arms?** ⭐ **YES — rank 1 under ALL TEN
> terms**, published, shipped and every candidate. This claim is robust to the whole grid.
>
> **Does the whole v1 tactical family sit below every REF-C arm?** ⛔ **ONLY under `twosided_v2`
> and `mean|twosided_v2`.** It is **FALSE under `clamp_v1` itself** (`v1_lat_straight` 0.7842 vs
> `refc_base_v0off` 0.7114) and false under every candidate. ⚠️ *Definition, inherited verbatim:*
> the claim is about the v1 **TACTICAL** family — `v1_tactical_follow`, `v1_tactical_oracle`,
> `v1_lat_straight`, `nospeed_tactical_oracle`; `v1_ego_*` are ego-**schedule** transforms.
> ⇒ **this is a property of the shipped term, not of the panel**, and any term change puts it at risk.
>
> **How many of the 10 paired contrasts flip?** ⭐ **ZERO under any candidate.** All 10 keep their
> sign and separation across every two-sided term. The **3** that flip against `clamp_v1`
> (`refc_{xl,base,small}_produced − v1_tactical_follow`) flip under **`mean|clamp_v1` too**, so they
> are attributable to the **SHAPE** (published → two-sided), not to the resolution — the known
> 2026-07-28 change, re-confirmed here.

⛔ **THE INSTRUMENT GUARDS REFUSE EVERY CANDIDATE:**

⚠️ **Which guard number is which.** The brief quotes `v4_oracle − v4_blind = +0.2384`; that is the
**CONTROL composite** under the shipped `lat_heading`. The table below is the **PSS composite**, the
object `ego_progress` is weighted in, where the same guard is **+0.2992**. ⭐ **My value reproduces
the sibling stream's +0.2992 exactly** — the positive control that this work did not disturb the PSS
composite. Both guards are the same contrast on different composites; a term change here moves the
PSS one.

| term | `v4_oracle − v4_blind` | `v1_ego_half − v1_tactical_follow` | verdict |
|---|---|---|:--:|
| ⭐ **`twosided_v2` (SHIPPED)** | ⭐ **+0.2992 [+0.2217, +0.3829]** | −0.1454 [−0.1758, −0.1149] | **the maximum** |
| `hyp_w1` | +0.2957 | −0.1850 | ⛔ weaker |
| `twosided_asym_w0p5` | +0.2903 | −0.1932 | ⛔ weaker |
| `mean\|twosided_v2` | +0.2882 | −0.1452 | ⛔ weaker on both |
| `sqrtlin_w0p5` / `w0p3333` | +0.2860 / +0.2846 | −0.2178 / −0.2335 | ⛔ weaker |
| `clamp_v1` | +0.2824 | −0.2687 | ⛔ weaker |
| ⭐ `mean\|hyp_w1` *(12/12)* | **+0.2800** | −0.1855 | ⛔ weaker |
| ⭐ `mean\|sqrtlin_w0p3333` *(12/12)* | **+0.2642** | −0.2358 | ⛔ **weaker by 11.7 %** |
| `mean\|clamp_v1` | +0.2577 | −0.2710 | ⛔ weaker |

> ### ⛔ DECISION: **NOTHING SHIPS AS THE DEFAULT, AND THE RULE THAT REFUSED IT WAS FIXED IN ADVANCE.**
> Pre-registered **A5**: *"DISQUALIFY any term that WEAKENS an instrument guard… a fix that makes
> the metric more permissive is not a fix."* **`twosided_v2`'s `v4_oracle − v4_blind` is the maximum
> over the entire grid** — necessarily, because it is the strictest term in it (fastest floor, full
> rate). Every repair of the zero-mean cell spends charge rate (§4.1), and spending charge rate is
> worth most to the **worst** arm.
> ⇒ ⭐ **THE ZERO-MEAN CELL CAN BE CLOSED, AND EVERY MEASURED WAY OF CLOSING IT MAKES THE METRIC MORE
> PERMISSIVE.** That is the answer to E-2 — not a missing shape, a **priced trade**, with the price
> now on the page in the programme's own guard units.
> ⇒ the two 12/12 terms' **SHAPES** (`sqrtlin_w0p3333`, `hyp_w1`) ship as **named, versioned,
> non-default** members of `PROGRESS_TERMS` with the full frontier published, and **E-2 remains a PI
> decision** — now with both outcomes measured, the theorem that forces the trade, and the guard cost
> of each point.
>
> ⚠️ **AND THE `mean` RESOLUTION IS DELIBERATELY *NOT* PROMOTED TO A REGISTRY ENTRY.** Only the
> primitive `progress_ratios_per_step` ships, with its pathology documented at the point of use; the
> composition is three lines in `code/run_egoprogress.py`. A named `PROGRESS_RESOLUTIONS["mean"]`
> would read as an endorsement, and the measurement says it is **half a defect** (§5): a registry
> entry is an invitation, and this one should be read before it is used.

---

## 9. ⚠️ THE `lat_heading` WEIGHT — the brief's premise is backwards

`raw/lat_heading_weight.json`, T14. E-1 asks for `CONTROL_WEIGHTS["lat_heading"] = 1.0` to be
**re-derived, not inherited**, because the shipped `mean1_lin_q0p5`'s live range is *half* the
published term's.

| | `term_lin_q0` *(published)* | `mean1_lin_q0p5` *(shipped)* | ratio |
|---|--:|--:|--:|
| pooled WITHIN-arm sd | 0.3359 | 0.2505 | 0.75× |
| pooled p05–p95 span | 0.9531 | 0.8360 | 0.88× |
| ⭐ **BETWEEN-arm span of means** | **0.2594** | **0.3125** | ⭐ **1.2047×** |
| ⭐ BETWEEN-arm sd of means | 0.0590 | 0.0715 | 1.21× |

⇒ ⛔ **The within-arm intuition is right and the conclusion drawn from it is wrong.** A composite
aggregates **arm means**, so the quantity a weight must be calibrated against is the **between-arm**
spread — and the shipped term's is **20 % LARGER**, not half. **Weight 1.0 today buys 1.20× the
influence the proposal assumed; the influence-preserving weight is `0.83`.**
⚠️ **MEASURED and ESCALATED, NOT CHANGED.** `CONTROL_WEIGHTS` is a PI decision, the control composite
is the object being proposed as the gate primary, and a weight change made by an agent inside an
unrelated repair is precisely the silent redefinition this programme keeps logging.

---

## 10. What ships

| file | change |
|---|---|
| `taniteval/taniteval/pseudosim.py` | ⭐ `_progress_hyperbolic` / `_progress_exp` / `_progress_sqrtlin` and **9 new versioned `PROGRESS_TERMS`** (7 → 16) (all strict refinements, none the default); `PROGRESS_UNDER_FLOOR_IS_UNFIXABLE`; `PROGRESS_OVER_SIDE_TRICHOTOMY`; `PROGRESS_HUMAN_MIN_M`; ⭐ `progress_ratios_per_step` (the resolution primitive, with its pathology documented at the point of use) |
| `taniteval/taniteval/control.py` | ⭐ **`lon_shift`** — the constant-sign along-track control the suite never had, plus its two-sided ladder. Its docstring carries the re-centring hazard and the direction-verification requirement |
| `taniteval/tests/test_egoprogress_terms.py` | ⭐ **NEW — 16 tests**, of which **9 drive a FAILING value** (the identical reversal scores, the standing-still price, the forcing theorem, the trichotomy, the small-denominator pathology, the direction flip, the two-sided saturation guard) |

**Suites:** `taniteval/` **773 passed, 0 skipped** (was 757 + **16 new**, **zero new skips**);
`stack/` **1576 passed, 12 skipped** — **unchanged**, zero failures, zero new skips. *(The single
pytest warning is pre-existing, in `test_clhorizon.py` via `ci.py:171`.)*
🔒 **Parity untouched** — no episode re-selected; all 20 arms share the identical 15,981 rows and row
identity is **asserted**, not assumed. No clip UUID or raw PhysicalAI content appears in any
artifact — counts only.

---

## 11. ⭐ ESCALATIONS — raised here, not left in a README

| # | what needs a decision or a cross-stream change | owner |
|:--:|---|---|
| **F-1** | ⛔⛔ **E-2 IS ANSWERED AS A PRICED TRADE, NOT AS A MISSING SHAPE — AND IT STAYS OPEN AS A PI DECISION.** Closing the zero-mean cell is possible (`mean\|sqrtlin_w0p3333`, `mean\|hyp_w1`, both 12/12) and **every** way of closing it (a) necessarily reduces the over-travel charge rate below `r = 2` (theorem, §4.1) and (b) **weakens `v4_oracle − v4_blind`, measurably, on all nine candidates**. The pre-registered bar A5 refuses all of them. **The PI is choosing a point on a frontier, not approving a fix.** | **PI + Benchmarks & Eval** |
| **F-2** | ⛔⛔ **ONE PUBLISHED NUMBER SHOULD BE QUALIFIED IN PLACE.** `BOUNDED_TERMS_COMPLETE.md` §4.2 / E-2 quotes *"+0.0104 [+0.0054, +0.0158] SEPARATED THE WRONG WAY"* as **the** open defect. The value reproduces exactly — **and its control moves the ground truth `+0.0101 m [−0.0258, +0.0477] n.s.` on a 25 m baseline**, i.e. it is a near-null perturbation. The defect stands; **the magnitude is not a degradation magnitude** and should not be read as one. | **Doc-correction sweep + `RETRACTION_LOG` owner** |
| **F-3** | ⚠️ **E-3 IS CLOSED AS AUDITED, WITH A RESIDUAL THAT CANNOT BE REPAIRED.** The under side **passes on every realisable arm** (0 wrong-way cells, any term, any resolution) and is **absolutely blind** once a plan reverses on essentially all rows. The blindness is arithmetically unfixable and the only lever pays standing still **and** weakens the guard. ⇒ **publish the under-floor fraction beside every value; do not fix the term.** | **Benchmarks & Eval — CLOSED, residual named** |
| **F-4** | ⭐ **A NEW ACCEPTANCE RULE, offered for adoption across every injection suite in the programme.** ⛔ **A CONTROL'S DIRECTION MUST BE VERIFIED AGAINST A METRIC-INDEPENDENT GROUND TRUTH BEFORE ANY VERDICT IS READ OFF IT.** It fired **twice** here on cells that would otherwise have been findings: `v1_tactical_follow \| lon_shift(−2 m)` (the only positive under-side estimate on the panel) and `v1_ego_double \| lon_jitter(σ=2 m)` (**the cell E-2 is built on**). Cost: one extra paired bootstrap per cell. | **PI / `taniteval` maintainer** |
| **F-5** | ⭐ **A NEW RETRACTION CLASS, offered for `RETRACTION_LOG.md` — I did not append it myself.** ⛔ **`A RESOLUTION CHANGE IS ONLY FREE WHEN THE PER-STEP QUANTITY IS THE SAME QUANTITY.`** The 20-step mean that fixed `lat_heading` with no free parameter **inverts** on `ego_progress`: floor 327× better, under-side charge 3.5× stronger, **over-side wrong-way leak 5.7× worse**, contamination margin 95× narrower. Heading error is equally meaningful at every step; **cumulative displacement is not — progress given back is not progress.** *Detection heuristic: before averaging a per-step form, ask whether the axis is defined on the terminal value or on the trajectory.* | **PI / `RETRACTION_LOG` owner** |
| **F-6** | ⭐ **AND ITS SIBLING, from my own bug:** ⛔ **`A MEMO KEYED ON A REUSABLE IDENTITY IS A CORRECTNESS BUG.`** `id(pw)` is reused by CPython after GC; my first decision-grade pass silently mixed two plans' scores and reported `clamp_v1` **7/12** where the truth is **2/12**. **The assertion that inherited cells must reproduce is what caught it** — inspection would not have. *Detection heuristic: every long run should re-derive at least one published number it did not compute.* | **PI / `RETRACTION_LOG` owner** |
| **F-7** | ⚠️ **`CONTROL_WEIGHTS["lat_heading"]` — E-1's premise is BACKWARDS and the owed re-derivation is done.** Between-arm span **1.2047× LARGER** under the shipped term, not half ⇒ influence-preserving weight **0.83**, not 2.0. Not changed by me. | **PI, BEFORE the gate-primary change** |
| **F-8** | ⚠️ **Downstream consumers gain names, nothing is removed.** `PROGRESS_TERMS` grows from 7 to 16 entries; `control.CONTROLS` / `LADDERS` gain `lon_shift`. Every default, id and published value is unchanged and the reproduction gate proves it. | **`stack/` + `taniteval` maintainers** |

---

## 12. Everything that is wrong with this work, stated by me

| # | limitation | status |
|:--:|---|---|
| 1 | ⛔⛔ **I SHIPPED A CORRECTNESS BUG INTO MY OWN DECISION-GRADE RUN AND PUBLISHED NOTHING FROM IT ONLY BECAUSE OF ONE ASSERTION.** The `id()`-keyed memo (result 10). Had I not required the inherited cells to reproduce, the report would have led with `twosided_v2` scoring 8/12 with **4** wrong-way cells. | ⛔ **disclosed — the most serious methodological event in this work** |
| 2 | ⛔ **THE PREFERENCE RULE A1–A7 WAS WRITTEN AFTER A `B = 40` SMOKE**, and says so inside `raw/PREREGISTRATION.json` rather than only here. The **acceptance** bar (U-0…U-7, O-1, O-2) was fixed before any number. | ⛔ disclosed, deliberate |
| 3 | ⚠️ **A `B = 40` SMOKE DISAGREED WITH THE DECISION-GRADE RUN**, and one of the disagreements was the bug and one was not: at `B = 40` `sqrtlin_w0p5`/`w0p3333` scored **12/12** on the over side and at `B = 2000` they score **11/12**. Only `B = 2000` counts; the disagreement is recorded rather than dropped. | disclosed |
| 4 | ⛔ **THE REVERSING SUBSTRATE IS SYNTHETIC**, and no realisable arm resembles it. Its role is to show that the under floor **has** a failure mode, not that any arm is in it. `v4_blind` at 31.78 % — the worst real case — does **not** leak (§2.4). | disclosed, by design |
| 5 | ⚠️ **ONLY 3 OF 20 ARMS CARRY THE UNDER-SIDE INJECTIONS** (plus the substrate). The density census covers all 20. Two of the three are the sibling suites' arms, chosen for comparability; `v4_blind` was added because it is the only arm with real under-floor mass. | disclosed |
| 6 | ⚠️ **THE GROUND TRUTH IS ITSELF A CHOICE.** `\|s_plan − s_human\|` is the along-track endpoint error — the quantity `ego_progress` is a bounded proxy for. It is **not** a safety ground truth, and a control that raises it is a *degradation of along-track fidelity*, not necessarily of driving. **There is no collision gate and no cuboids here**, so no better ground truth is available. | ⛔ disclosed — the weakest link in the direction rule |
| 7 | ⚠️ **THE UNDER-SIDE RANGE BUDGET'S DECAY BELOW `r = 0` IS ARBITRARY** (`p·(1+r)`, hitting 0 at `r = −1`). Its **price at `r = 0`** — the only number the decision turns on — is `p` for **any** such family, so the arbitrariness does not touch the conclusion. But the family was not swept. | disclosed |
| 8 | ⛔ **I DID NOT SHIP AN `ego_progress` FIX**, deliberately, and E-2 remains open. *"The composite is sound end to end"* may **not** be written. | ⛔ open, escalated (F-1) |
| 9 | ⚠️ **`sqrtlin_*` ARE A ONE-PARAMETER FAMILY I CHOSE**, `(1 − w·over)^0.5`. The exponent 0.5 was not swept; `a` and `w` trade off along the same frontier `rate(1⁺) = a·w`, `floor = 1 + 1/w`, so a sweep of `a` would move along the published frontier rather than off it — but that is an argument, not a measurement. | disclosed |
| 10 | ⛔ **No model was run and no checkpoint was scored.** Every number is arithmetic over dumps produced by other streams; their fidelity is `INHERITED`. | by rule (pod1/pod2 forbidden) |
| 11 | ⛔ **No collision gate, no TTC, no map.** Not a Driving Score; not comparable to PDMS. 2 s horizon, non-reactive log replay. | inherited blocker |

---

## 13. Self-refutations

| # | what | status |
|:--:|---|---|
| 1 | ⛔⛔ **MY PRE-REGISTERED U-3 FAILED IN BOTH CLAUSES.** I predicted a zero-mean jitter would RAISE `ego_progress` on `v4_blind`, more under `clamp_v1` than under `twosided_v2`. MEASURED: **−0.0070** and **−0.0135**, both correct-direction, and the ordering reversed. ⇒ **a floor fraction is not a sufficient statistic for C45's mechanism; the opposite end of the same clamp can outvote it.** | ⛔ corrected (§2.4) |
| 2 | ⛔⛔ **I EXPECTED THE RESOLUTION LEVER TO BE THE ANSWER AND IT IS HALF A DEFECT.** It is the single strongest result on the under side (327×, 3.5×) and it makes the over-side leak **5.7× worse** and the contamination margin **95× narrower**. Reporting only the under-side half would have shipped a term that is measurably worse where the open defect actually lives. | ⛔ corrected (§5) |
| 3 | ⛔⛔ **I FOUND AND FIXED A CORRECTNESS BUG IN MY OWN RUNNER MID-REPORT.** An `id()`-keyed memo. Caught by the inherited-cell assertion, not by reading the code. | ⛔ corrected (result 10) |
| 4 | ⛔ **I NEARLY REPORTED "E-2's CELL IS NOT A REAL DEGRADATION, SO THE DEFECT IS NOT REAL".** It is a refused control **and** a real defect: a metric must be n.s. where the truth is n.s. Both halves are stated (§4.3). | ⛔ corrected before publication |
| 5 | ⚠️ **MY FIRST JENSEN PREDICTION WAS TOO STRONG.** I predicted the convex, never-flooring shapes would be **REWARDED** (separated positive) on the failing cell. MEASURED: all four are **n.s.** (+0.0006 … −0.0004). The Jensen bias is real (the trichotomy test drives it) but **second-order and below detection at these σ**; the first-order effect is the floor **kink**. | ⛔ corrected (§4.2, §4.4) |
| 6 | ⚠️ **I ACCEPTED "THE SHIPPED `lat_heading` HAS HALF THE LIVE RANGE" AND IT IS BACKWARDS WHERE IT MATTERS.** Within-arm 0.75×; **between-arm 1.20×**. The weight should go **down**, not up. | ⛔ corrected (§9) |

---

## 14. Is the composite sound end to end? — the plain statement

⛔ **NO. Both jobs did not close, and here is exactly what remains.**

| term | status |
|---|---|
| ⚠️ `ego_progress@twosided_v2` — **over side** | ⛔ **STILL FAILS** its zero-mean cell (+0.0104 SEP). It is now known to be **closable at a measured price** and every closure weakens `v4_oracle − v4_blind`. **E-2 remains open as a PI decision (F-1).** |
| ⚠️ `ego_progress@twosided_v2` — **under side** | ⭐ **AUDITED FOR THE FIRST TIME AND IT PASSES ON EVERY REALISABLE ARM** — 0 wrong-way cells over 18 terms × 17 verified cells; the only wrong-way separation in the 324-cell panel is on the one **refused** control, under `twosided_asym_w2` (a sensitivity anchor, never a default). ⛔ **It is absolutely blind at `r ≤ 0`, unfixably**, and the residual is real but **not reachable by any realisable arm** on this panel (worst real case 31.78 %, does not leak). |
| ✅ `recovery@twosided_v2` | fixed 2026-07-28, 8/8, `live_frac` 0.9161, admitted by gate v2 — **not re-verified here** (`INHERITED`) |
| ✅ `lat_heading@mean1_lin_q0p5` | fixed 2026-07-28, 10/10, `live_frac` 0.9975 — ⚠️ **its WEIGHT is now measured and owed a PI decision (§9, F-7)** |
| ✅ `lon_track`, `lat_track` | audited, never defective — **not re-verified here** (`INHERITED`) |
| ⚠️ `comfort` | weight 0, 100 % saturated by construction, refused by gate v2 — `INHERITED` |

**What remains, and none of it is hidden:**

1. ⛔ **`ego_progress@twosided_v2` still fails one zero-mean cell**, and it is now a **priced trade**
   rather than an unsolved problem: closing it costs over-travel charge rate below `r = 2`
   (theorem) and costs guard strength (measured, all nine candidates). **PI decision.**
2. ⛔ **The under floor at `r ≤ 0` is permanent.** A plan reversing 1 m and one reversing 30 m score
   identically, forever, in any bounded strict refinement.
3. ⚠️ **The failing cell's control is a near-null perturbation**, so the *magnitude* of the one open
   defect has been over-read — the sign has not.
4. ⛔ **`lon_retime(0.5)` still raises `lat_heading`** (inherited E-4, not touched here).
5. ⛔ **No collision gate, no TTC, no map.** `PSS` is not a Driving Score.

⇒ **The under side is closed as audited with a named permanent residual. The over side is closed as
a measured, priced trade with the decision escalated. `ego_progress` is not sound end to end, and
item 1 is the reason.**

---

## 15. Deliverable manifest

Repo dir:
`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-28-egoprogress-complete/`
Everything `git add`-ed into the working tree and verified with `git ls-files --cached`.
⛔ **I did not commit and did not push.**
⚠️ marks anything living in only ONE place — **there is nothing in that state.**

| artifact | where it lives | what it is |
|---|---|---|
| `EGOPROGRESS_COMPLETE.md` | `repo:` this dir | this report |
| `code/run_egoprogress.py` | `repo:` this dir | the whole run: under density, the under-side audit with its direction rule, the range-budget pricing, the over-side grid, the contamination panel, the reproduction gate, the panel, the `lat_heading` weight. **No GPU, no model, no corpus.** Asserts row identity, the zero-bias reference, `ego_progress` definedness invariance and the inherited cells; ⛔ md5-stamps the `ci.py`/`control.py`/`pseudosim.py` actually loaded (C44) |
| `code/tables.py` | `repo:` this dir | regenerates T1–T14 from the raw JSON — the tables are generated, not hand-typed |
| ⭐ `raw/PREREGISTRATION.json` | `repo:` this dir | ⭐⭐ **staged BEFORE any under-side number was computed**; carries U-1…U-7, O-1…O-4, and the A1–A7 addendum with its post-smoke disclosure |
| `raw/under_density.json` | `repo:` this dir | ⭐ the under side's density per arm + the fixed-42-row identical-score demonstration |
| `raw/inject_under.json` | `repo:` this dir | ⭐⭐ **THE UNDER-SIDE AUDIT** — 6 injections × 3 arms × 18 terms, the ground-truth direction check, the reversing stress substrate |
| `raw/under_fix.json` | `repo:` this dir | ⛔ the range budget priced on `stand_still` and on `v4_blind` |
| `raw/inject_over.json` | `repo:` this dir | ⚠️ the over-side grid, the direction check that refuses E-2's control, the inherited-cell reproduction |
| `raw/contamination.json` | `repo:` this dir | ⚠️ purity, all 18 terms — where `hyp_w2` died |
| `raw/repro_gate.json` | `repo:` this dir | ✅ 16 published composites at `max\|diff\| = 0.000000` + strict-refinement bit-identity |
| `raw/panel.json` | `repo:` this dir | the 20-arm ranking, 10 paired contrasts, the two instrument guards under 10 terms |
| `raw/lat_heading_weight.json` | `repo:` this dir | ⚠️ the owed re-derivation |
| `raw/run_all.log` | `repo:` this dir | the run's own stdout |
| `artifacts/tables.md` | `repo:` this dir | the generated tables T1–T14 |
| **`taniteval/taniteval/pseudosim.py`** | `repo:` | ⭐ 10 new versioned progress terms, the impossibility and trichotomy constants, `progress_ratios_per_step` |
| **`taniteval/taniteval/control.py`** | `repo:` | ⭐ `lon_shift` + its ladder |
| **`taniteval/tests/test_egoprogress_terms.py`** | `repo:` | ⭐ **NEW — 16 tests**, 9 driving a FAILING value |

**Reproduce everything, no GPU** (**428 s** on the dev box):
```
python3 code/run_egoprogress.py --n-boot 2000 \
  --panel-terms "clamp_v1,twosided_v2,twosided_asym_w0p5,sqrtlin_w0p5,sqrtlin_w0p3333,hyp_w1,mean|clamp_v1,mean|twosided_v2,mean|sqrtlin_w0p3333,mean|hyp_w1" \
  --in-dir "<…/Benchmarks & Eval/…/2026-07-27-pseudosim-arm-panel/artifacts>" \
  --in-dir "<…/Architecture & Inference/…/2026-07-28-tactical-action-input/artifacts/pw>" \
  --in-dir "<…/Architecture & Inference/…/2026-07-28-tactical-action-input/artifacts/blockA>" \
  --out-dir raw
python3 code/tables.py raw > artifacts/tables.md
```
