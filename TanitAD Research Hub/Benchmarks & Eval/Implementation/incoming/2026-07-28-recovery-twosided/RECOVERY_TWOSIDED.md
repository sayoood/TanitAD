# THE SECOND ONE-SIDED CLAMP IS FIXED — AND THE INTUITIVE FIX ("MAKE IT NEVER SATURATE") IS THE ONE THAT FAILS

**Wall-clock date:** 2026-07-28 (Europe/Berlin) · **Stream:** Benchmarks & Eval ·
**Branch:** `agent/benchmarks-eval-20260721` · **Repo HEAD at start:** `588a031`
**Hosts:** dev box only. Every number is CPU arithmetic over committed dumps. **0 GPU-h.**
⛔ **pod1 (TRAINING) and pod2 (small validation) were NOT touched — not even read.**
pod3 and `tanitad-eval` were not used: nothing here needs a GPU.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (not
re-verified) · `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.

**Source:** `…/2026-07-28-closedloop-control-suite/CLOSEDLOOP_CONTROL_SUITE.md` escalation **E-1**,
plan step **2**; retraction classes **C45** (the defect) and **C46** (`comfort`).

---

## 0. Headline

| # | Result | Class · tier |
|:--:|---|---|
| **1** | ✅⭐ **THE ACCEPTANCE TEST PASSES 8 OF 8, ON BOTH ARMS, INCLUDING UNDER THE ZERO-MEAN CONTROL.** The identical injections that exposed the defect now all move the composite **DOWN and SEPARATED**: `lat_shift(+2 m)` **+0.0581 → −0.0276**, `lat_shift(−2 m)` **+0.0501 → −0.0339**, `yaw_bias(+5°)` **+0.0747 → −0.0124**, and the **zero-mean** `lat_jitter(σ=1 m)` **+0.0303 → −0.0082**, all `SEPARATED`, replicated on `v1_tactical_follow`. The metric no longer pays for lateral degradation. | `MEASURED` **tier 1** |
| **2** | ⛔⛔ **MY OWN PROVISIONAL FIX WAS REFUTED BY ITS OWN ACCEPTANCE TEST, AND THE REASON IS THE REAL FINDING.** I pre-registered a preference for the **unsaturating** `share` family (`xt_hold/(xt_hold+xt_end)`) on the obvious argument that a term with **no floor at all** cannot be blind. It floors on **0.0000** of rows on every arm — and it scored **0 of 8**. ⇒ ⭐ **"NEVER SATURATES" IS NOT THE PROPERTY THAT FIXES A SATURATING METRIC. THE PROPERTY IS THAT THE CHARGE RATE MUST NOT COLLAPSE WHERE THE DATA LIVES.** The share form's `\|dg/dr\|` decays like `r⁻²` (**1.00 at r=0, 0.25 at r=1, 0.0625 at r=3**) while the panel's median ratio is **1.18** — it rewards a near-perfect row **4.3×** harder than it charges a typical one. **A soft floor is still a floor.** | `MEASURED` **tier 1** |
| **3** | ⭐ **THE SHIPPED SHAPE IS `clamp(1 − r/3, 0, 1)`, CHOSEN BY A RULE FIXED BEFORE THE NUMBERS.** One interpretable parameter — `q` = *the score a plan that recovers nothing receives* — swept over two families at five anchors. `q = 0` **is** the published term. The pre-registered rule (R1 all-8-separated · R2 floor < 0.50 · R3 prefer the family that is affine-equivalent to the published term, smallest surviving `q`) lands on **`lin_q0p6667`**, `q = 2/3`. | `MEASURED` **tier 1** |
| **4** | ⛔ **AND THE MINIMUM-ASSUMPTION EVEN SPLIT — THE DIRECT ANALOGUE OF `OVER_TRAVEL_WEIGHT = 1.0` — FAILS HERE, 7/8.** `lin_q0p5` = *"let the published term run one unit negative, then rescale"* is exactly the argument that fixed `ego_progress`. Its 8th cell (`v1_tactical_follow` × `yaw_bias(+5°)`) is **correct in sign but n.s.**: **−0.0036 [−0.0075, +0.0002]**, stable at 5 bootstrap seeds. ⇒ **the sibling term's parameter cannot be ported — `recovery`'s ratio has a far heavier tail (median ≥ 1.004 on EVERY arm, p99 3.3–11.1, max 34.1) and the acceptance test, not the analogy, decides.** | `MEASURED` **tier 1** |
| **5** | ⛔ **A STRICT REFINEMENT IS PROVABLY IMPOSSIBLE, so this fix is `twosided_v2`'s *treatment* but not its *shape* — and the proof is a test, not a paragraph.** Any bounded `g: [0,∞)→[0,1]` agreeing with the published `1−r` on `[0,1]` has `g(1)=0` and, non-increasing and bounded below, **must be constant above 1** — i.e. it *is* the defect. The published term had already spent its entire range on the half of the domain where the plan beats hold. ⇒ the fix is a **RANGE-BUDGET** choice, not a slope choice. The attainable substitute — **affine-equivalence on `r ≤ 1`** — is delivered: `g = q + (1−q)·clamp_v1`, so no pair of under-recovering rows re-orders and the published value inverts exactly. | `MEASURED` **tier 1** |
| **6** | ✅ **REPRODUCTION GATE EXACT: `max\|diff\| = 0.000000` on 16 published `@clamp_v1` composites**, with the recovery term versioned and its default flipped, `clamp_v1` **bit-identical** to family member `lin_q0` over 5001 ratios, and the published metric id string **unchanged** (`PSS_recovery_progress@clamp_v1`) so no existing pin stops resolving. | `MEASURED` **tier 1** |
| **7** | ⭐ **THE RANKING SURVIVES — AND IT IS THE MOST IMPORTANT NEGATIVE OF THE WEEK.** `cv_holdv0` **still ranks first among realisable arms**, and the **whole v1 tactical family still sits below every REF-C arm**. **One** verdict flips: `v1_tactical_follow − refc_base_v0off` **−0.0322 SEP → −0.0159 n.s.** — *"the flagship's tactical head is separated-worse than a REF-C base decoder with its speed input deliberately zeroed"* is **WITHDRAWN**; it is now a tie. Both instrument guards get **stronger** (`v4_oracle − v4_blind` +0.2049 → **+0.2992**), so the metric did not become permissive. | `MEASURED` **tier 2** |
| **8** | ⛔⛔ **THE AUDIT FOUND A THIRD ONE-SIDEDLY CLAMPED TERM, AND IT SHIPPED 24 HOURS AGO AS "ADMITTED".** `lat_heading = clamp(1 − \|Δψ\|/PSI_TOL, 0, 1)` is a **single value per row**, not a mean over 20 steps, so it saturates at row level: **floored on 31.2 %–84.3 %** of defined rows, ceiling essentially never active (0.0001–0.0023) — one-sided in exactly the C45 shape. **Six of the 20 arms exceed 50 %**, and `FLOOR_FRAC_MAX = 0.95` **does not refuse it.** ⇒ **E-2.** | `MEASURED` **tier 1** |
| **9** | ⚠️ **`comfort` IS NOT A SATURATING SCORE — IT IS 100 % SATURATED BY CONSTRUCTION**, on every arm, at one end or the other (`cv_holdv0` ceiling **1.0000**, `v4_blind`/`v4_oracle` floor **1.0000**). It is a `{0,1}` indicator, so its `observed_range = 1.0` — the statistic that let it clear `RANGE_MIN` for its whole life — is an artefact of both values merely being *present*. C46 re-derived from a second direction, on a term the brief specifically required be audited **because it is still measured**. | `MEASURED` **tier 1** |
| **10** | ⚠️ **AND ONE RESIDUAL I DID NOT FIX AND WILL NOT HIDE: `ego_progress@twosided_v2` — last night's repair — is itself still one-sided above ratio 2**, floored on **32.70 %** of `v4_blind`'s rows and 24.27 % of `v1_ego_double`'s. Below the 50 % warning threshold, so not a C45 emergency; above zero, so not "sound". Named here rather than discovered later. | `MEASURED` **tier 1** |

### 0.1 Pre-registered outcome

**CONFIRM, with a self-refutation inside it.** The pre-registered CONFIRM was *"the injected-lateral
ladder becomes negative and separated on `cv_holdv0` AND `v1_tactical_follow`, under the zero-mean
control"* — fired, 8/8. The REFUTE branch (*"no admissible shape exists ⇒ drop `recovery` from the
composite"*) did **not** fire. ⛔ **But the shape I expected to win lost**, and the rule that decided
it was written down before the numbers (`raw/injections.json ›
injections._selection_rule_PRE_REGISTERED`).

### 0.2 Tier and inherited qualifiers

**Tier 1** for §1–§5 and §7–§8 (deterministic CPU arithmetic over committed dumps; no model, no
checkpoint, no corpus). **Tier 2** for any statement about a named ARM, which inherits the panel's
four qualifiers verbatim: **oracle goal where used**, **non-reactive log replay**, **no collision or
TTC gate**, **`comfort` dropped by the gate**; plus the source stream's two (Block B arms are **plan
transforms, not trained planners**; two arms are **oracles**).
⛔ **Nothing here is a Driving Score** and nothing here is compared to a PDMS number.
⚠️ **`selected_frac` = 1.000 for every arm BY CONSTRUCTION** — there is no selection step on the
pseudo-simulation surface; every arm is scored on all 15,981 rows.
⚠️ **Nothing here is held to v1's 0.4271** — that is `wm_fidelity_ade_2s`, not a planning bar.

---

## 1. The defect, restated exactly

```python
xt_end  = |y[:, -1] − ref_y[:, -1]|                  # cross-track at the plan's endpoint
xt_hold = |dlat + s_along · tan(dpsi)|               # the drift if it had NOT steered
r       = xt_end / xt_hold                           # what fraction of the error remains
recovery = clamp(1 − r, 0, 1)          # ⛔ the published term
```

⭐ **`recovery`'s clamp is one-sided in a stronger sense than `ego_progress`'s ever was.** Both
`xt_end` and `xt_hold` are absolute values, so `r ≥ 0` identically, so `1 − r ≤ 1` identically:
**the ceiling clamp is provably never active.** `clamp(1 − r, 0, 1)` **is** `max(1 − r, 0)`. There
is exactly one clamp and it is the whole defect.

`MEASURED`, `raw/saturation_census.json` — the floor holds the majority of every arm:

| | value |
|---|---|
| floor fraction, over DEFINED rows | **55.65 % (`cv_holdv0`) … 92.19 % (`refc_xl_produced`)** |
| median unclamped ratio | **> 1.0 for EVERY arm** (1.004 – 1.373); panel-pooled **1.181** |
| fraction of rows with `r > 1` | **75.41 %** panel-pooled |
| p99 / max ratio | 3.3 – 11.1 / **34.1** |

⇒ **on three quarters of the rows the plan ends further off the logged path than not steering at
all, and every one of them is charged exactly 0.** A perturbation that helps a minority therefore
raises the mean, and the damaged majority is absorbed. That is C45, and §3 is its consequence.

---

## 2. ⭐ THE SHAPE — one parameter, two families, a rule fixed before the numbers

### 2.1 ⛔ Why this is not `twosided_v2`'s shape, proved rather than asserted

`twosided_v2` could write *"the published term minus an over-travel charge"* because `clamp_v1(1) = 1`
left a whole unit of range underneath it. **`recovery`'s published term is already at 0 at `r = 1`.**

> **Claim.** Let `g: [0, ∞) → [0, 1]` be non-increasing with `g(r) = 1 − r` on `[0, 1]`. Then `g` is
> constant on `[1, ∞)`.
> *Proof.* `g(1) = 0`. For `r > 1`, monotonicity gives `g(r) ≤ g(1) = 0`, and the codomain gives
> `g(r) ≥ 0`. So `g(r) = 0`. ∎

⇒ **any bounded strict refinement of the published `recovery` term is the defect.** Pinned by
`test_a_STRICT_REFINEMENT_IS_IMPOSSIBLE_and_the_family_shows_it`, which *enumerates* the family
members agreeing with the published term below `r = 1` (exactly `{clamp_v1, lin_q0}` — the same
function twice) and asserts each is constant above it.

⇒ **the fix is a RANGE-BUDGET choice, not a slope choice.** The parameter is:

> **`q` = the score a plan that recovers NOTHING (`r = 1`) receives.**
> The recovery half `r ∈ [0,1]` gets the band `[q, 1]`; the divergence half `r > 1` gets `[0, q]`.
> `q = 0` **is** the published term — all budget on recovery, none on divergence.

⚠️ **There is no "over-recovery" side to be asymmetric about.** `r ≥ 0` identically, so the domain
has exactly two regions and **`q` is the only asymmetry there is**. Larger `q` = more budget spent
charging divergence, less resolution crediting recovery. That trade is the whole design space, and
its sensitivity is published (§6) rather than chosen.

### 2.2 The two families, at the same anchors

| family | `g(r)` | `g(0)` | `g(1)` | floors at | affine in the published term on `r ≤ 1`? | charge rate |
|---|---|:--:|:--:|---|:--:|---|
| **linear budget** `lin_q*` | `clamp(1 − (1−q)·r, 0, 1)` | 1 | `q` | `r = 1/(1−q)` | ⭐ **YES**: `g = q + (1−q)·clamp_v1` | **constant `1−q`** |
| **share** `share_q*` | `q / (q + (1−q)·r)` | 1 | `q` | ⭐ **never** | no (agrees only to 1st order at `r=0`) | decays like `r⁻²` |

At `q = 0.5` the share member is exactly **`xt_hold / (xt_hold + xt_end)`** — the fraction of the
total (hold + achieved) cross-track error attributable to the hold baseline. Parameter-free, elegant,
and §3 kills it.

### 2.3 ⭐ The pre-registered selection rule

Banked into `raw/injections.json` **before the panel was computed**, and mirrored in
`pseudosim.RECOVERY_TERMS`' docstring:

| rule | |
|---|---|
| **R1** | **DISQUALIFY** any shape for which the 8 injected lateral degradations are not **ALL separated in the CORRECT (negative) direction on BOTH real arms.** A shape failing this does not fix the defect. |
| **R2** | **DISQUALIFY** any shape whose recovery **floor fraction ≥ 0.50** on any scorable arm. C45: *"a term saturating on the majority of rows is not a metric; it is a constant with noise."* |
| **R3** | Among survivors **PREFER the LINEAR family** (affine-equivalent on `r ≤ 1` ⇒ no under-side re-ordering, published value exactly invertible). Within it, **smallest `q`** — minimum departure from the published term. |
| **R4** | If no linear member survives, take **`share` at `q = 0.5`** — the equal-budget, parameter-free member. |

*Why a rule at all:* the `w` decision published its sensitivity instead of choosing on the ranking it
liked. **A shape chosen after seeing the panel is a shape fitted to the panel.**

---

## 3. ⛔⛔ THE ACCEPTANCE TEST — and the self-refutation inside it

`raw/injections.json`, `artifacts/tables.md` T1. The **identical** 8 injections that exposed the
defect, on the **identical** 15,981 rows, with the **identical** estimator (paired episode-cluster
bootstrap, B = 2000, unit = val episode). ⛔ `overlapping_holdout_se` appears nowhere.
**Positive Δ means the DEGRADED plan scores HIGHER.**

### 3.1 The verdict, every shape

| shape | injections correct | max recovery floor frac | verdict |
|---|:--:|--:|:--:|
| ⛔ `clamp_v1` *(published)* | **0/8** | 0.9219 | ⛔ the defect |
| ⛔ `lin_q0p25` | 0/8 | 0.5884 | R1 **and** R2 |
| ⚠️ `lin_q0p5` *(the even split)* | **7/8** | 0.1372 | ⛔ R1 |
| ⭐ **`lin_q0p6667` ← SHIPPED** | ✅ **8/8** | **0.0833** | ✅ |
| `lin_q0p75` | ✅ 8/8 | 0.0532 | ✅ (R3: larger `q`) |
| ⛔ `share_q0p25` | 0/8 | **0.0000** | ⛔ R1 |
| ⛔ **`share_q0p5`** *(my provisional default)* | **0/8** | **0.0000** | ⛔ R1 |
| ⛔ `share_q0p6667` | 3/8 | 0.0000 | ⛔ R1 |
| ⛔ `share_q0p75` | 6/8 | 0.0000 | ⛔ R1 |

### 3.2 The shipped term, cell by cell — **8 of 8**

| arm | injection | zero-mean | Δ`recovery` | Δ`ego_progress` | **Δ composite** | |
|---|---|:--:|--:|--:|---|:--:|
| `cv_holdv0` | `lat_shift(+2 m)` | | −0.0633 | −0.0008 | **−0.0276 [−0.0401, −0.0151]** | ✅ SEP |
| `cv_holdv0` | `lat_shift(−2 m)` | | −0.0784 | −0.0008 | **−0.0339 [−0.0481, −0.0205]** | ✅ SEP |
| ⭐ `cv_holdv0` | **`lat_jitter(σ=1 m)`** | **yes** | −0.0179 | −0.0003 | **−0.0082 [−0.0122, −0.0044]** | ✅ SEP |
| `cv_holdv0` | `yaw_bias(+5°)` | | −0.0267 | −0.0021 | **−0.0124 [−0.0171, −0.0075]** | ✅ SEP |
| `v1_tactical_follow` | `lat_shift(+2 m)` | | −0.0641 | +0.0004 | **−0.0296 [−0.0443, −0.0162]** | ✅ SEP |
| `v1_tactical_follow` | `lat_shift(−2 m)` | | −0.0706 | −0.0010 | **−0.0344 [−0.0498, −0.0214]** | ✅ SEP |
| ⭐ `v1_tactical_follow` | **`lat_jitter(σ=1 m)`** | **yes** | −0.0195 | −0.0002 | **−0.0099 [−0.0168, −0.0044]** | ✅ SEP |
| `v1_tactical_follow` | `yaw_bias(+5°)` | | −0.0281 | +0.0016 | **−0.0111 [−0.0145, −0.0079]** | ✅ SEP |

`ego_progress` is flat to ±0.002 throughout — the movement is entirely `recovery`, as it was when the
sign was wrong.
⭐ **The zero-mean rows are the ones that close the argument in both directions.** A per-row Gaussian
offset with expectation 0 cannot re-centre a bias, so neither the old `+0.0303` nor the new `−0.0082`
is the re-centring artefact this program already knows about. **It was the metric; now it is fixed.**

### 3.3 ⛔⛔ SELF-REFUTATION — the unsaturating shape failed, and that is the finding

I pre-registered R4 expecting to use it: the `share` family has **no floor at all**, and a term that
never saturates cannot be blind. **It scored 0/8.**

`raw/injections.json › charge_rate` — the mechanism, MEASURED:

| shape | `\|dg/dr\|` @ r=0 | @0.5 | @1 | @2 | @3 | ⛔ reward bias *(near-perfect ÷ median row)* | injections |
|---|--:|--:|--:|--:|--:|--:|:--:|
| `clamp_v1` | 1.0000 | 1.0000 | 0.5000 | **0.0000** | **0.0000** | **∞** | 0/8 |
| `share_q0p25` | 2.9991 | 0.4800 | 0.1875 | 0.0612 | 0.0300 | **15.61** | 0/8 |
| `share_q0p5` | 0.9999 | 0.4444 | 0.2500 | 0.1111 | 0.0625 | **4.32** | 0/8 |
| `share_q0p6667` | 0.5000 | 0.3200 | 0.2222 | 0.1250 | 0.0800 | **2.41** | 3/8 |
| `share_q0p75` | 0.3333 | 0.2449 | 0.1875 | 0.1200 | 0.0833 | **1.88** | 6/8 |
| `lin_q0p5` | 0.5000 | 0.5000 | 0.5000 | 0.2500 | 0.0000 | **1.00** | 7/8 |
| ⭐ `lin_q0p6667` | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.1667 | **1.00** | **8/8** |

⭐ **The pass rate is monotone in the reward bias, across both families.** The panel's median ratio is
**1.18** and **75.41 %** of rows sit above 1.0, so a shape whose slope collapses by `r⁻²` charges the
*typical* row at a quarter of the rate at which it rewards a row that is already nearly perfect —
and an injection that pushes a few near-zero rows closer to zero then outweighs the many rows it
pushes further out.

⇒ ⭐⭐ **The property that fixes a saturating metric is not "no floor". It is "the charge rate must
not collapse where the data lives". A soft floor is still a floor.**

Reduced to two rows and pinned as a test
(`test_the_UNSATURATING_share_family_still_pays_for_lateral_degradation`): take `r = [0.4, 1.6]` and
degrade **both** to `[0.2, 2.4]` (mean ratio 1.00 → 1.30). `share_q0p5` goes **UP** (0.5495 → 0.5637);
`lin_q0p6667` goes **DOWN**; `clamp_v1` goes up harder. The share form is kept in `RECOVERY_TERMS`,
published and swept, **because it fails** — the rejection is checkable.

### 3.4 ⚠️ AND THE MINIMUM-ASSUMPTION EVEN SPLIT FAILED TOO

`lin_q0p5(r) = clamp(1 − r/2, 0, 1) = (clamp(1 − r, −1, +1) + 1)/2` — *"let the published term run one
full unit negative, then rescale to [0,1]"*. It charges divergence at exactly the rate it credits
recovery. **This is verbatim the argument that justified `OVER_TRAVEL_WEIGHT = 1.0`**, and I expected
it to be the answer.

**7/8.** The 8th cell — `v1_tactical_follow` × `yaw_bias(+5°)` — is **−0.0036 [−0.0075, +0.0002]**:
correct in sign, **not separated**. Stable at bootstrap seeds 0/1/7/123/2026 (`lin_q0p6667` is stably
separated at the same five). ⚠️ A **B = 60 smoke run read it as 8/8** — only the decision-grade
B = 2000 counts, and the disagreement is recorded rather than quietly dropped.

⇒ **the sibling term's parameter cannot be ported.** `ego_progress`'s ratio and `recovery`'s ratio
have very different tails: a floor at `r = 2` leaves **4.8–13.7 %** of rows uncharged here.
**The acceptance test, not the analogy, decides.**

---

## 4. ✅ THE REPRODUCTION GATE — exact

`raw/repro_gate.json`, T3. Every published `@clamp_v1` composite recomputed with the recovery term
versioned, its default flipped, and `discriminative_range` emitting saturation:

| | |
|---|---|
| published `@clamp_v1` composites checked | **16** |
| **max \|diff\|** | ⭐ **0.000000** |
| verdict | ✅ **PASS** |
| published metric id, unchanged | `PSS_recovery_progress@clamp_v1` |
| `clamp_v1` **bit-identical** to family member `lin_q0`, over 5001 ratios | ✅ |

⭐ **The versioning is backward-compatible by construction.** `metric_id` appends `+rec_<term>` **only
when the recovery term is not the published one**, so every id published through 2026-07-28 keeps its
exact string and no pin, log line or report stops resolving — while a non-published recovery term is
*always* visible in the name. Pinned by
`test_the_metric_id_carries_the_recovery_term_only_when_it_is_NOT_published`.

⚠️ **And `composite()`'s fallback is the PUBLISHED term, not the default.** A caller that hands in a
raw dict of arrays with no `_recovery_term` key is reproducing a pre-2026-07-28 number; naming it
after the new term would be exactly the silent redefinition being fixed. Pinned.

⛔ **The synthetic reference arm was kept OUT of the gate vote.** `human_replay` is built and its
`max |residual| = 1.53 × 10⁻⁵ m` **asserted** before use, but it is never inserted into `arms` — the
sibling stream's own gate caught it silently redefining the composite for every arm
(`max|diff| 0.000000 → 0.393900`) because it is ceiling-saturated on `ego_progress` by construction.

---

## 5. ⭐ THE PANEL UNDER BOTH TERMS, AND THE RANKING STATEMENT

**20 arms · 40 val episodes · 15,981 rows per arm · row identity `(ep_i, anchor, dlat, dyaw, dlon)`
ASSERTED across all 20, 0 refused.** ⛔ **PANEL-WIDE gate** (`ego_progress` + `recovery` admitted,
`comfort` dropped, under both terms). The per-arm gate is **refused, not offered**.

⛔ **These are levels on a NEW metric id** (`PSS_recovery_progress@twosided_v2+rec_twosided_v2`) and
are **NOT comparable to any published PSS value**: every arm rises by **+0.14 to +0.24** simply
because a plan that recovers nothing is now scored `2/3` instead of `0`. **Only orderings and paired
deltas carry across.** Full table: T4.

| rank (new) | arm | `rec@clamp_v1` (PUBLISHED) | `rec@twosided_v2` (NEW) | rank (pub) | recovery floor frac pub → new |
|--:|---|---|---|--:|---|
| 1 | ⚠️ `v1_ego_oracle_lon` *(ORACLE)* | 0.5943 | 0.8205 | 1 | 0.7173 → 0.0211 |
| 2 | ⚠️ `v4_oracle` *(ORACLE)* | 0.5362 | 0.7727 | 4 | 0.7556 → 0.0140 |
| **3** | ⭐ **`cv_holdv0`** | **0.5492** | **0.7721 [0.7407, 0.7999]** | **2** | 0.5565 → 0.0673 |
| 4 | `v1_ego_v0` | 0.5403 | 0.7645 | 3 | 0.7287 → 0.0344 |
| 5–9 | `refc_xl_produced` … `refc_small_produced` | 0.5259 … 0.5202 | 0.7452 … 0.7370 | 5–9 | 0.91–0.92 → 0.017–0.019 |
| 10 | `refc_xl_v0off` | 0.4724 | 0.6889 | 10 | 0.9008 → 0.0238 |
| 11 | `refc_base_v0off` | 0.4591 | 0.6726 | 11 | 0.8993 → 0.0247 |
| 12 | `v1_tactical_oracle` | 0.4305 | 0.6606 | 12 | 0.6922 → 0.0202 |
| **13 ⬆1** | `v1_lat_straight` | 0.4209 | 0.6590 | 14 | 0.5635 → 0.0503 |
| **14 ⬇1** | ⛔ `v1_tactical_follow` | 0.4242 | 0.6563 | 13 | 0.6973 → 0.0194 |
| 15 | `nospeed_tactical_oracle` | 0.4179 | 0.6491 | 15 | 0.6988 → 0.0198 |
| 16 | ⚠️ `v4_blind` | 0.3322 | 0.4742 | 16 | 0.5579 → 0.0833 |

### 5.1 ⭐ THE RANKING STATEMENT, EXPLICITLY

> **Does `cv_holdv0` still rank first among realisable arms?** ⭐ **YES.** Rank **1 → 1**. Rank 1
> overall remains `v1_ego_oracle_lon`, an **ORACLE**, exactly as before — unchanged by this work.
>
> **Does the whole v1 family still sit below every REF-C arm?** ⭐ **YES**, under both terms.
> ⚠️ *Definition, stated because I nearly mis-read my own check:* the claim is about the v1
> **TACTICAL** family — `v1_tactical_follow`, `v1_tactical_oracle`, `v1_lat_straight`,
> `nospeed_tactical_oracle`. `v1_ego_v0` and `v1_ego_oracle_lon` are ego-**schedule** transforms and
> rank **above** every REF-C arm under the published term as well, so folding them in would refute a
> claim nobody made. My first pass did exactly that and reported `False` for both terms.

**2 realisable arms moved** (`v1_lat_straight` ⇄ `v1_tactical_follow`, whose contrast is **n.s. under
both terms** — a presentation change, not a finding); **5 non-probe arms moved** (adding the
`v4_oracle` ⇄ `cv_holdv0` swap, also n.s. under both, and both oracles).

### 5.2 The contrasts — one verdict flips

| contrast | Δ under `rec@clamp_v1` | Δ under `rec@twosided_v2` | |
|---|---|---|:--:|
| `v1_ego_v0 − cv_holdv0` | −0.0090 [−0.0160, −0.0023] SEP | −0.0076 [−0.0143, −0.0011] SEP | |
| `v1_tactical_follow − cv_holdv0` | −0.1212 SEP | −0.1154 [−0.1399, −0.0896] SEP | |
| `refc_xl_produced − cv_holdv0` | −0.0230 SEP | −0.0269 SEP | |
| `v1_ego_v0 − v1_tactical_follow` | +0.1121 SEP | +0.1078 [+0.0830, +0.1310] SEP | |
| `v1_lat_straight − v1_tactical_follow` | −0.0020 n.s. | +0.0027 n.s. | |
| `refc_xl_v0on − refc_xl_v0off` | +0.0534 SEP | +0.0562 SEP | |
| ✅ `v4_oracle − v4_blind` *(G1 guard)* | +0.2049 SEP | ⭐ **+0.2992 SEP** | **+46 %** |
| ✅ `v1_ego_half − v1_tactical_follow` *(degradation guard)* | −0.1189 SEP | **−0.1454 SEP** | stronger |
| ⛔⛔ **`v1_tactical_follow − refc_base_v0off`** | ⛔ **−0.0322 [−0.0525, −0.0106] SEP** | ⚠️ **−0.0159 [−0.0396, +0.0089] n.s.** | **FLIP** |

**Three readings.**

1. ⭐ **The headline survives, and that is the consequential result.** Two metric repairs in two days,
   each large enough to move every level by 0.2, and *"a zero-parameter baseline beats every learned
   arm"* has not moved. It can no longer be attributed to the instrument being blind on the lateral
   axis — the instrument that says it now charges lateral degradation **8/8**.
2. ⛔ **One claim is WITHDRAWN.** *"A REF-C base decoder with its speed input deliberately zeroed
   scores above the flagship's deployed tactical head"* was **separated** under the published
   recovery term and is **n.s.** under the fixed one. It was published 24 hours ago. The point
   estimate keeps its sign; only the separation goes. **The v1-vs-REF-C ordering claim as a whole
   survives** — `v1_tactical_follow − refc_xl_produced` stays **−0.0886 SEP**.
3. ✅ **The metric did not become permissive.** Both instrument guards survive and get *stronger*;
   the G1 sighted-vs-blind gap grows 46 %.

---

## 6. ⚠️ SENSITIVITY OF THE RANKING TO THE SHAPE

T9, all nine shapes including the disqualified ones.

* ⭐ **ROBUST: `cv_holdv0` is rank 1 among realisable arms under EVERY shape tested** — both families,
  every anchor, including the published term. **The headline does not depend on the shape at all.**
* ⭐ **ROBUST: ranks 1–7 are identical under every shape.**
* ⚠️ **NOT robust: the full realisable order is not identical** — `v1_lat_straight` moves between
  #8 and #11 across the grid, and `refc_base_v0off` between #9 and #11. Their contrasts are n.s.
  ⇒ **do not read the deep ranking off a chosen `q`.**
* ⚠️ **The LEVELS move enormously with `q`** (`cv_holdv0` 0.5492 → 0.8228 from `q=0` to `q=0.75`).
  That is why the term is in the metric id.

⇒ **`q = 2/3` is reported as the primary because a rule fixed before the numbers selected it, not
because it wins anything.** `lin_q0p75` also passes 8/8 and gives the same rank-1 arm.

---

## 7. ⚠️ SATURATION BESIDE EVERY BOUNDED TERM — C45's standing consequence, made permanent

*"For every bounded term, report the FLOOR/CEILING FRACTION beside the score."* `discriminative_range`
**computed** `floor_frac_le_0p001` and **never surfaced it beside a level** — which is how a term
floored on 55–92 % of its rows was published twenty times. What ships now:

| where | what |
|---|---|
| `pseudosim.saturation(arr)` | ⭐ **NEW, public.** `floor_frac` / `ceiling_frac` / `saturated_frac` / `defined_frac` + a `SATURATION_WARNING` string that fires at **≥ 50 %** on either side |
| `pseudosim.discriminative_range` | every component node now carries `saturation` |
| `pseudosim.composite` | `component_saturation` for every admitted term, in the emitted node |
| `pseudosim.emit` | `components[k].saturation` beside every CI |
| `control.axis_summary` / `control.block` | every axis in `BOUNDED_AXES` carries it |
| `heldout_gate` probe records | `component_saturation` in every probe node — ⭐ **a probe that stops a GPU-week must show whether the term that stopped it had any gradient left** |

⚠️ **The WARNING (0.50) is deliberately stricter than the GATE (`FLOOR_FRAC_MAX = 0.95`).** Both
thresholds are asserted in one test so neither can drift into the other.

---

## 8. ⛔ THE AUDIT OF THE REMAINING TERMS — and it found a third one

*Two of two audited terms were one-sidedly clamped, so the composite is not sound until the rest are
checked.* `raw/term_audit.json`, T8. `comfort` is audited **despite carrying weight 0.0**, because
it is still MEASURED and a saturating diagnostic misleads exactly as a saturating score does.

| term | clamped side(s) | max floor frac over panel | worst arm | risk |
|---|---|--:|---|:--:|
| `ego_progress@clamp_v1` | BOTH | 0.3178 | `v4_blind` | ✅ no |
| ⚠️ `ego_progress@twosided_v2` | **FLOOR at r ≥ 2 — still one-sided above it** | **0.3270** | `v4_blind` | ✅ no *(but see below)* |
| ⛔ `recovery@clamp_v1` | FLOOR at r ≥ 1 *(ceiling never active)* | **0.9219** | `refc_xl_produced` | ⛔ **the defect** |
| ✅ **`recovery@twosided_v2`** *(shipped)* | FLOOR at r ≥ 3 | **0.0833** *(min 0.0140, `v4_oracle`)* | `v4_blind` | ✅ no |
| ⛔ `comfort` | **not a clamp — a `{0,1}` indicator** | **1.0000** | `v4_blind` | ⛔ **YES** |
| ✅ `lon_track` | per-step floor, **meaned over 20 steps** | 0.0029 | `v1_tactical_follow` | ✅ no |
| ✅ `lat_track` | per-step floor, **meaned over 20 steps** | 0.0947 | `refc_xl_produced` | ✅ no |
| ⛔⛔ **`lat_heading`** | **FLOOR at \|Δψ\| ≥ PSI_TOL — a SINGLE value per row, so it saturates at ROW level** | **0.8429** | `v4_blind` | ⛔ **YES** |

### 8.1 ⛔⛔ `lat_heading` is the third one-sided clamp, and it shipped yesterday as "admitted"

`lat_heading = clamp(1 − |Δψ| / PSI_TOL_RAD, 0, 1)`. Unlike `lon_track` and `lat_track` — which are
**means over 20 steps**, so a row saturates only if all 20 steps do — `lat_heading` is **one number
per row**. Its floor therefore bites at row level:

| | floor frac | ceiling frac |
|---|--:|--:|
| `v4_blind` | ⛔ **0.8429** | 0.0001 |
| every REF-C arm | **0.4999 – 0.5125** | 0.0011 – 0.0016 |
| `cv_holdv0`, the v1 family | 0.3122 – 0.3757 | 0.0010 – 0.0023 |

**Six of 20 arms exceed the 50 % warning threshold, the ceiling is never meaningfully active, and
`FLOOR_FRAC_MAX = 0.95` does not refuse any of them.** This is C45's shape exactly, in an axis the
control suite admitted 24 hours ago and proposed for the gate primary at weight 1.0.

⚠️ **It is NOT in the composite today** (`pseudosim.COMPONENT_WEIGHTS` is `{ego_progress 5,
recovery 5, comfort 0}`), so no published number is affected. It **is** in
`control.CONTROL_WEIGHTS`, the proposal behind escalation E-2 of the source stream. ⇒ **E-2 below.**
⛔ **`lat_heading` may not be quoted without its floor fraction**, and it must not enter a gate
primary until it is fixed the way `recovery` just was.

### 8.2 ⚠️ `comfort` — 100 % saturated by construction, which is worse than saturating

It is the **AND of four bounds**: a `{0,1}` indicator, so **every row is at floor or ceiling on every
arm**. `cv_holdv0` ceiling **1.0000**; `v4_blind` and `v4_oracle` floor **1.0000**. ⇒ its
`observed_range = 1.0` — the statistic that let it clear `RANGE_MIN` for its whole life — records
only that both values are *present*. This is C46 reached from a second direction, and it confirms
the weight-0.0 decision on grounds independent of the human-pass-rate argument.

### 8.3 ⚠️ The residual I did not fix

`ego_progress@twosided_v2` — **last night's repair** — is itself still one-sided above `r = 2`, and
floors on **32.70 %** of `v4_blind`'s rows and **24.27 %** of `v1_ego_double`'s. Below the 50 %
warning threshold on every arm, so it is not a C45 emergency and I did not re-open it inside this
change. It is **not zero**, so *"the composite is now sound"* would be an over-claim. ⇒ **E-3.**
*(`stand_still` floors it at 1.0000, but that is an adversary probe doing its job.)*

---

## 9. What ships

| file | change |
|---|---|
| `taniteval/taniteval/pseudosim.py` | `RECOVERY_TERM_PUBLISHED` / `_DEFAULT` / `_DEFAULT_TARGET` / `_ALIASES`, `RECOVERY_TERMS` (9 members + alias), `RECOVERY_HOLD_ANCHOR(_GRID)`, `recovery_from_ratio`, `UnknownRecoveryTerm`, ⭐ `saturation()` + `SATURATION_WARN_FRAC`; `score_windows(recovery_term=…)` emitting `recovery_raw_ratio` / `_recovery_term`; `metric_id(progress_term, recovery_term)` **backward-compatible**; `composite`/`emit` thread and publish both terms + `component_saturation`; `discriminative_range` surfaces `saturation` and skips `_`-prefixed provenance keys |
| `taniteval/taniteval/control.py` | `axes(recovery_term=…)`, `AXIS_META["recovery"].two_sided` **False → True** *(it was the bug's fingerprint)*, `BOUNDED_AXES`, `axis_summary`/`block` publish saturation and both term ids, `dynamic_range(recovery_term=…)` |
| `stack/tanitad/train/heldout_gate.py` | `RECOVERY_TERM`, `HeldoutGateConfig.recovery_term`, `PRIMARY_NAME` now `…@twosided_v2+rec_twosided_v2`, `PRIMARY_NAME_PUBLISHED_20260728_PROGRESS_ONLY` kept beside the 2026-07-27 one, `component_saturation` in every probe record |
| `taniteval/tests/test_recovery_term.py` | ⭐ **NEW — 29 tests**, 8 of which pin a FAILING value |
| `taniteval/tests/test_pseudosim.py` | ⚠️ **3 sibling tests updated** — see §11 limitation 1 |

**Suites:** `taniteval/` **726 passed, 0 skipped** (was 697 + **29 new**, **zero new skips**);
`stack/` **1576 passed, 12 skipped** — **unchanged**, zero failures, zero new skips.
🔒 **Parity untouched** — no episode re-selected; all 20 arms share the identical 15,981 rows and row
identity is **asserted**, not assumed. No clip UUID or raw PhysicalAI content appears in any artifact.

---

## 10. ⭐ ESCALATIONS — raised here, not left in a README

| # | what needs a decision or a cross-stream change | owner |
|:--:|---|---|
| **E-1** | ⭐⭐ **`recovery` IS FIXED AND THE V5 GATE IS UNBLOCKED — but the SHAPE is a PI decision that I have taken provisionally.** `RECOVERY_TERM_DEFAULT_TARGET = "lin_q0p6667"` was selected by a rule fixed before the numbers; the full sweep is published (§6) and `lin_q0p75` also passes. **The PI should ratify or override `q`; the code makes that a one-constant change and a test pins which member the alias resolves to.** ⚠️ **The gate primary's NAME changed** to `pseudosim_composite_PSS_recovery_progress@twosided_v2+rec_twosided_v2`. | **PI + `taniteval` maintainer — BEFORE the v5 gate** |
| **E-2** | ⛔⛔ **`lat_heading` IS THE THIRD ONE-SIDED CLAMP — floored on 31–84 % of rows, ceiling never active, and `FLOOR_FRAC_MAX = 0.95` does not refuse it.** It is **not** in the composite today, but `control.CONTROL_WEIGHTS` proposes it at weight 1.0 and the source stream's **E-2** asks the PI to adopt exactly that. ⇒ **the answer to that escalation must now be "not until `lat_heading` gets the treatment `recovery` just got".** The same one-parameter budget family applies directly. | **Benchmarks & Eval + PI — blocks the source stream's E-2** |
| **E-3** | ⚠️ **`ego_progress@twosided_v2` is still one-sided above ratio 2** (floor 32.70 % on `v4_blind`, 24.27 % on `v1_ego_double`). Below the warning threshold, so not urgent — but *"the composite is sound"* may not be written until it is re-examined with the same rule. | **Benchmarks & Eval** |
| **E-4** | ⛔ **ONE PUBLISHED VERDICT IS WITHDRAWN:** `v1_tactical_follow − refc_base_v0off` **−0.0322 SEP → −0.0159 n.s.** Any doc claiming *"a REF-C decoder with its speed input zeroed scores above the flagship's tactical head"* must be corrected. **Every PSS LEVEL published before today is also on a different metric id** and rises 0.14–0.24 under the new one; levels do not carry across, orderings and paired deltas do. | **Model-registry agent + doc-correction sweep** |
| **E-5** | ⭐ **A NEW RETRACTION CLASS, offered for `RETRACTION_LOG.md` — I did not append it myself.** ⛔ **`AN UNSATURATING METRIC CAN STILL PAY FOR DEGRADATION`.** MEASURED: `xt_hold/(xt_hold+xt_end)` floors on **0.0000** of rows and scored **0 of 8** injections correctly, while a shape that floors on **8.33 %** scored 8/8. The pass rate is monotone in the **charge rate at the median row**, not in the floor fraction. Sibling of C45 — C45 says *"enumerate what lives in the zero-gradient half"*; this says *"and measure the gradient where the data lives, because a rate that decays like `r⁻²` is a soft floor"*. *Detection heuristic: for any bounded score, compute `\|dg/dr\|` at the median of its own raw quantity and divide by `\|dg/dr\|` near the ideal — > 1 means the metric rewards polishing good rows more than it charges bad ones.* | **PI / RETRACTION_LOG owner** |
| **E-6** | ⚠️ **The saturation node now ships inside `heldout_gate` probe records and every `emit()`/`block()` node.** Downstream JSON consumers gain fields; nothing is removed. Whoever parses gate records should expect `component_saturation` and `recovery_term`. | **`stack/` maintainer** |

---

## 11. Everything that is wrong with this work, stated by me

| # | limitation | status |
|:--:|---|---|
| 1 | ⛔ **I EDITED THREE SIBLING TESTS** in `taniteval/tests/test_pseudosim.py`. Each broke for a real reason and each was fixed by **preserving the original guarantee under the term it was written for** and adding the new one: `test_composite_is_not_called_a_driving_score` (the id now carries both terms — I also added an assertion that the PUBLISHED id is still exactly emittable); `test_recovery_separates_a_recovering_planner_from_a_drifting_one` (the ORDERING claim is now asserted under **both** terms, the published *"hold scores 0"* value is still pinned under `clamp_v1`, **and** I added the defect itself in the failing direction — a plan 3× worse than hold scoring identically to hold); `test_recovery_is_defined_and_low_for_a_drifting_model_planner` (the `< 0.5` bar is preserved verbatim under `clamp_v1`, plus a new assertion that the two terms actually **disagree**). ⇒ **a reviewer should confirm the strengthened forms are what the pins' authors intended.** | ⛔ disclosed, deliberate |
| 2 | ⚠️ **`q` IS A CHOICE, not a measurement.** It is selected by a pre-registered rule from a 5-point grid over 2 families, and the whole grid is published — but no `q` is derived from data about what divergence *costs*, because **this surface has no collision gate and no cuboids**. A cost-calibrated `q` is impossible here, not merely absent. | disclosed, swept |
| 3 | ⛔ **THE SHIPPED TERM STILL FLOORS**, on **1.40 % (`v4_oracle`) – 8.33 % (`v4_blind`)** of defined rows across the 16 non-probe arms. The panel's **worst** floor fraction falls **0.9219 → 0.0833 (11.1×)** and its **best-case** arm falls 0.5565 → 0.0140 (39.7×) — reduced, not eliminated. Any shape bounded in `[0,1]` with a non-collapsing charge rate must floor somewhere; §3.3 is why the unbounded alternative is worse. | ⚠️ **a real residual** |
| 4 | ⚠️ **I did not touch the DENOMINATOR, deliberately.** `xt_hold = \|dlat + s_along·tan(dpsi)\|` is measured against the *perturbation*, while `xt_end` is measured against the *human's endpoint* (`ref_y[:, -1]`), which is not subtracted from `xt_hold`. On a curving window those are not the same origin, so `r = 1` is only approximately *"no better than hold"*. Changing it would confound the clamp fix with a definition change; it is named here as a separate question. | ⛔ **UNVERIFIED magnitude — flagged, not measured** |
| 5 | ⚠️ **`lon_retime(0.5)` still moves `recovery` the wrong way, and my fix does not cure it.** That trap is a **definedness** artefact — a barely-moving plan leaves the denominator (`xt_hold ≤ 0.10` ⇒ NaN) rather than being beaten — not a clamp artefact. The `defined_frac`/`saturation` reporting makes it visible in the same row; fixing it means changing the NaN mask, which is limitation 4's question. | inherited, now visible |
| 6 | ⛔ **No model was run and no checkpoint was scored.** Every number is arithmetic over dumps produced by other streams; their fidelity is `INHERITED`. | by rule (pod1/pod2 forbidden) |
| 7 | ⚠️ **The 8 injections are the ones that exposed the defect, by design** — same controls, same two arms, same levels. That makes the test a genuine *re-run* rather than a friendlier one, but it is **not an independent validation set**: a shape could in principle be tuned to these 8. The pre-registered rule and the fact that the *whole family* is published are the mitigation; a fresh injection set would be stronger. | disclosed |
| 8 | ⚠️ **Only 2 of the 20 arms carry injections.** The floor/saturation census covers all 20; the direction test covers `cv_holdv0` and `v1_tactical_follow`, inherited from the source stream's design. | disclosed |
| 9 | ⚠️ **A B = 60 smoke run read `lin_q0p5` as 8/8 and the B = 2000 run reads 7/8.** Only the decision-grade B counts, and I record the disagreement rather than dropping the run that disagreed. | disclosed |
| 10 | ⚠️ **My first ranking check reported `whole_v1_family_below_every_REFC = False` under BOTH terms** — a definitional error in *my* check (it swept `v1_ego_v0`/`v1_ego_oracle_lon`, ego-schedule transforms, into "the v1 family"), not a finding about the sibling's claim. Corrected before publication; the family is now written out explicitly in the code. | ⛔ corrected (§5.1) |
| 11 | ⛔ **No collision gate, no TTC, no map.** `PSS` is **not** a Driving Score and none of this may be compared to a PDMS number. 2 s horizon, non-reactive log replay, lateral grid axis refused — all inherited and bounding. | inherited blocker |

---

## 12. Self-refutations

| # | what | status |
|:--:|---|---|
| 1 | ⛔⛔ **MY PROVISIONAL DEFAULT WAS THE UNSATURATING SHARE FORM, AND ITS OWN ACCEPTANCE TEST SCORED IT 0/8.** It floors on **0.0000** of rows on every arm. Had I shipped on the "it can never be blind" argument — which is the argument anyone would make — I would have published a *second* metric that pays for lateral degradation, under a name announcing it was fixed. **Only running the 8 injections against my own candidate caught it.** | ⛔ corrected (§3.3) |
| 2 | ⛔ **The direct analogue of the `ego_progress` fix (`q = 0.5`, even split) also failed, 7/8.** I expected the sibling term's minimum-assumption parameter to port. It does not: `recovery`'s ratio tail is much heavier. | ⛔ corrected (§3.4) |
| 3 | ⚠️ **My first ranking check asked the wrong question** and would have reported the sibling stream's v1-vs-REF-C claim as false under its own published term. Caught because it disagreed with the published table under the term that produced it. | ⛔ corrected (§5.1) |
| 4 | ⚠️ **My first test fixture had no dynamic range**, so `composite()` refused (`VacuousMetric`) and two tests were passing on a *refusal* rather than on arithmetic. The fixture now spans recovery fractions **past 1.0**, i.e. it contains rows in the half the published term cannot see. | ⛔ corrected |
| 5 | ⚠️ **I initially claimed the share family "never re-orders two rows, it only changes paired deltas."** True per row, and **irrelevant** to a mean-over-rows metric — which is exactly why it failed. The claim is removed from the code, not softened. | ⛔ corrected |
| 6 | ⚠️ **A term audit that stopped at the composite's own terms would have missed `lat_heading`**, which carries weight 0 in the composite and weight 1 in the proposal the PI is being asked to adopt. Auditing "the remaining terms" had to mean the measured ones, not the weighted ones — which is the same argument that put `comfort` in the audit. | ⭐ the brief's instruction, vindicated |

---

## 13. Deliverable manifest

Repo dir:
`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-28-recovery-twosided/`
Everything `git add`-ed into the working tree. ⛔ **I did not commit and did not push.**
⚠️ marks anything living in only ONE place — **there is nothing in that state.**

| artifact | where it lives | what it is |
|---|---|---|
| `RECOVERY_TWOSIDED.md` | repo (this dir) | this report |
| `code/run_recovery_twosided.py` | repo | the whole run: injections, reproduction gate, panel, saturation census, term audit, shape sweep. **No GPU, no model, no corpus.** Asserts row identity and the zero-bias reference; ⛔ md5-stamps the `ci.py`/`control.py`/`pseudosim.py` actually loaded (C44) |
| `code/tables.py` | repo | regenerates T1–T9 from the raw JSON — the tables are generated, not hand-typed |
| `raw/injections.json` | repo | ⭐ **THE ACCEPTANCE TEST** — 8 injections × 9 shapes × 2 arms, paired CIs, + the pre-registered rule and the charge-rate mechanism |
| `raw/repro_gate.json` | repo | ✅ 16 published `@clamp_v1` composites at `max\|diff\| = 0.000000` |
| `raw/panel_both_terms.json` | repo | 20 arms under both recovery terms, 14 paired contrasts, the ranking statement |
| `raw/shape_sensitivity.json` | repo | the ranking at every point of the `q` grid, both families |
| `raw/saturation_census.json` | repo | ⚠️ floor/ceiling fraction of 8 bounded terms × 20 arms |
| `raw/term_audit.json` | repo | ⛔ the one-sidedness audit of every remaining term — where `lat_heading` was found |
| `raw/run_all.log` | repo | the run's own stdout |
| `artifacts/tables.md` | repo | the generated tables T1–T9 |
| **`taniteval/taniteval/pseudosim.py`** | repo | ⭐ the versioned recovery term, the shape family, `saturation()` |
| **`taniteval/taniteval/control.py`** | repo | `recovery_term` threading, `two_sided` corrected, saturation on every bounded axis |
| **`taniteval/tests/test_recovery_term.py`** | repo | ⭐ **NEW — 29 tests**, 8 pinning a FAILING value |
| `taniteval/tests/test_pseudosim.py` | repo | 3 sibling tests updated and strengthened (limitation 1) |
| `stack/tanitad/train/heldout_gate.py` | repo | `RECOVERY_TERM`, config field, versioned `PRIMARY_NAME`, per-probe saturation |

**Reproduce everything, no GPU** (**~145 s** on the dev box, three banked runs):
```
python3 code/run_recovery_twosided.py --n-boot 2000 \
  --in-dir <…/Benchmarks & Eval/…/2026-07-27-pseudosim-arm-panel/artifacts> \
  --in-dir <…/Architecture & Inference/…/2026-07-28-tactical-action-input/artifacts/pw> \
  --in-dir <…/Architecture & Inference/…/2026-07-28-tactical-action-input/artifacts/blockA> \
  --out-dir raw
python3 code/tables.py raw > artifacts/tables.md
```
