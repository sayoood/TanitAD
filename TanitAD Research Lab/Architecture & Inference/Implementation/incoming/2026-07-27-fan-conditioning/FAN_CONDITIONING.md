# FAN CONDITIONING — does a state-conditioned anchor set raise the ceiling, or the realised pick?

**Stream:** `2026-07-27-fan-conditioning` · **Host:** dev box, CPU only · **Date:** 2026-07-27
**Pre-registration:** `PRE_REGISTRATION.md`, written before any anchor set was fitted.
**Estimator:** paired episode-cluster bootstrap, `taniteval/ci.py`, **B = 2000**, unit = **episode**
(40). `overlapping_holdout_se` is never called.

---

## 0. HEADLINE — **VERDICT: REFUTE** on the briefed lever, and the redirect is measured

> ### ⛔ The lever two streams converged on does not exist. The fan's longitudinal vocabulary is **already exact**.
>
> **MEASURED, DECISION-GRADE, replicated on 4 independent fans.** The premise is true and the
> inference from it is false:
>
> - **TRUE:** the fan is *not* `v0`-conditioned. Candidate speed tracks ego speed at slope
>   **−0.129**; ground truth tracks it at **+1.0003**. The fan proposes the same ~17 m/s
>   distribution in every window.
> - **FALSE:** that this costs anything. **In 100.0 % of windows the fan already contains a
>   candidate within 0.5 m/s of the exact speed the car actually took** (mean gap **0.0525 m/s**,
>   max **0.4418**). Restricting the oracle to *only* speed-matched candidates changes it by
>   **+0.0000 m [0.0000, 0.0000]** — the best-in-fan candidate **is already speed-matched**.
>
> ⇒ **The marginal over-dispersion is real; the conditional coverage gap is not.** The fan is
> **wide, not mis-placed**. No reallocation of the longitudinal vocabulary can lower the ceiling,
> **by construction rather than by a null result.**
>
> ### ⭐ And the same instrument found what the missing information actually is.
>
> Giving the selector the **true 2 s goal position** — 2 of the 8 target scalars, nothing about how
> to get there — moves the realised pick **0.4714 → 0.2009 [0.1689, 0.2351]**, paired
> **−0.2705**, separated. That is **88.0 % of the fan's entire headroom**, and it **clears CONFIRM
> (0.4907) and STRONG (0.4271)** — the first thing in this program to do so.
> **Longitudinal-only goal: −31.4 %. Lateral-only: −5.8 %. Both: +88.0 %.** A pure interaction.
> ⚠️ **It is an ORACLE and therefore a bound, not a capability** — see §7.

| pre-registered outcome | returned |
|---|---|
| **CONFIRM** (realised < 0.4907) | ❌ best realisable conditioned pick **0.7968**, a **1.62×** miss |
| **PARTIAL** (ceiling moves, pick does not) | ❌ on the **real fan** the ceiling **cannot** move; on a *static* anchor set it moves, but the equal-storage control shows the gain is **anchor count, not conditioning** |
| **REFUTE** (neither moves) | ✅ **this one** |

**Tier: DECISION-GRADE** for the refutation (mechanistic, replicated ×4, both-directions
validated). **PROVISIONAL–CONFIRMED** for the goal-oracle bound (single corpus, oracle input).

---

## 1. Priority-1 was already executed by a sibling stream — reproduced, not repeated

The brief's priority-1 item (*"longitudinal admissibility filter… measure this FIRST"*) was run
overnight by `…/2026-07-27-fan-clip-local/`, which returned **REFUTE**: at `a_max = 2.5` the clip
deletes **85.4 %** of the fan and moves the as-trained selector by **+0.0020 m [−0.0016, +0.0076]**,
not separated.

**That result is reproduced here (§2) and is NOT the same experiment as this one:**

| | admissibility filter | conditioned anchor set |
|---|---|---|
| operation | **removes** candidates | **reallocates** candidates |
| effect on `oracle_in_fan` | can only **worsen** (0.1640 → 0.1683) | **can improve** |
| per-window proposals | 256 → 37.5 | stays **N** |

A subset operation cannot raise a ceiling. So the filter's REFUTE constrained only the *realised*
half and left the *ceiling* half genuinely open. **This stream closes it — and the answer is that
it was never open in the way both streams believed.**

## 2. S0 — GATE. Every committed number reproduced from raw before any new one was quoted

`raw/fanc_gate.json → S0_gate`, `_all_ok = true`.

| quantity | committed | reproduced |
|---|---:|---:|
| REF-C-XL `oracle_in_fan` / as-trained | 0.1640 / 0.4714 | **0.1640 / 0.4714** |
| REF-C-base / REF-C-small `oracle_in_fan` | 0.1914 / 0.2213 | **0.1914 / 0.2213** |
| v4 fan `oracle_in_fan` (v1 scorer) | 0.2505 | **0.2505** |
| v4 fan A0 as-trained / C2 | 0.8563 / 0.5645 | **0.8563 / 0.5645** |

Window identity across the two artifact families: `v0` is `allclose` between the REF-C fans and the
v5 reduced dump, on all 881 windows — the two families are the same deployment.

## 3. S1 — THE PREMISE IS TRUE: the fan is not `v0`-conditioned

`raw/fanc_gate.json → S1_premise`. **MEASURED.**

⚠️ **Quote the SLOPE, not the correlation.** `corr(v0, fan speed) = −0.974` looks like violent
inverted conditioning. It is an artifact: the envelope is nearly *constant*, so a tiny systematic
drift dominates a normalised statistic. The slope carries the physics.

| fan | anchors | **slope of candidate speed on `v0`** | GT's slope | fan mean speed | candidates within ±2 m/s of GT's speed |
|---|---:|---:|---:|---:|---:|
| REF-C-XL | 256 | **−0.1294** | +1.0003 | 16.12 m/s | **11.97 %** |
| REF-C-base | 128 | **−0.0686** | +1.0003 | 17.41 | 11.87 % |
| REF-C-small | 64 | **−0.1586** | +1.0003 | 16.39 | 14.46 % |
| **v4's own fan** | 256 | **−0.1442** | +1.0003 | 16.67 | — |

Per-`v0`-quintile mis-centring on REF-C-XL: at `v0 ∈ [0, 5.1)` GT needs **2.60 m/s** and the fan
offers a distribution centred on **17.11**; at `v0 ∈ [19.8, 36.5)` GT needs **28.34** and the fan
offers **13.98**. **The proposal set ignores ego speed.** The brief's premise is confirmed on all
four fans, including v4's.

> **Both converging streams observed this correctly. The error is in the next step.**

## 4. S5 — ⭐ THE DECISIVE MEASUREMENT: over-dispersion ≠ a coverage gap

`raw/fanc_coverage.json`. The question a state-conditioned anchor set must answer *yes* to:
**does the fan fail to offer the right speed?**

```
coverage gap   g_w = min_c | speed(c) − speed(GT_w) |
matched oracle = min ADE over ONLY candidates within ±tol of GT's speed
```

| fan | mean span | **mean gap** | **max gap** | **% windows w/ a candidate within 0.5 m/s** | `oracle` | **matched oracle (±1 m/s)** | paired Δ | sep |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **REF-C-XL** | 52.3 m/s | **0.0525** | 0.4418 | **100.00 %** | 0.1640 | **0.1640** | **+0.0000 [0.0000, 0.0000]** | ❌ |
| REF-C-base | 54.1 | 0.0879 | 0.9912 | 98.98 % | 0.1914 | 0.1924 | +0.0010 [0.0000, 0.0026] | ❌ |
| REF-C-small | 53.8 | 0.1154 | 1.0323 | 98.18 % | 0.2213 | 0.2227 | +0.0014 [0.0000, 0.0035] | ❌ |
| **v4's own fan** | 54.4 | **0.0532** | 0.4554 | **100.00 %** | 0.2505 | **0.2505** | **+0.0000 [0.0000, 0.0000]** | ❌ |

> ⇒ **THE BEST CANDIDATE IN THE FAN IS ALREADY A SPEED-MATCHED CANDIDATE, ON EVERY FAN, AT EVERY
> SIZE.** Discarding every candidate whose speed is more than **1 m/s** wrong — **91.6–93.7 %** of
> the fan — leaves `oracle_in_fan` **bit-identical**. There is nothing for a `v0`-conditioned
> anchor set to add.

At a tighter ±0.5 m/s the penalty becomes just-separated but negligible (+0.0025 / +0.0053 /
+0.0106 / +0.0021 m — **1.2 % of the ceiling**), which is the honest statement of the residual.

**The reconciliation with the 181 km/h plan:** the fan's *span* is ~53 m/s. A distribution that wide
covers the correct speed in every window **even though it is centred 14 m/s away**. Over-dispersion
and exact coverage are not in tension — they are the same fact seen from two directions. Stream 1
measured the span, Stream 2 measured the unreachable anchors; **neither measured the conditional,
which is what a conditioned anchor set would repair.**

### 4.1 ✅ Validation in BOTH directions — the instrument CAN return "conditioning is the lever"

Required by the brief; this program has shipped vacuous diagnostics.

**Fidelity.** `pick_nearest_to(GT)` must reproduce `oracle_in_fan` exactly → `max_abs_diff = 0.0`
on every configuration. ⚠️ **This control earned its place: it FAILED at 0.3655 m on first run**
and caught a real bug — the proximity metric was flat 8-dim L2 while scoring used mean-over-waypoint
L2. Different metrics, different `argmin`. Fixed in `fanc_common.pick_nearest_to`; the numbers above
are all post-fix.

**Shift response.** A rigid along-track shift of the whole fan must degrade the ceiling smoothly and
return the committed value at zero:

| shift (m/s) | −10 | −5 | −2 | −1 | **0** | +1 | +2 | +5 | +10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ceiling | 1.3365 | 0.8092 | 0.4835 | 0.3266 | **0.1640** | 0.3758 | 0.6512 | 1.3350 | 2.5083 |

**⭐ DELIBERATELY FAILING INPUT — `F_narrow`.** A fan whose speed spread is artificially compressed
by `k`, so it is *genuinely* speed-starved. If the instrument is real, conditioning must **help**
here:

| | coverage gap | % within 0.5 m/s | ceiling | after `v0` re-centring | **gain from conditioning** | sep |
|---|---:|---:|---:|---:|---|---|
| `k=1.0` — **the real fan** | 0.0525 | **100.0 %** | 0.1640 | 1.7875 | **+1.6236 [+0.9968, +2.4107]** *(hurts)* | ✅ |
| `k=0.5` | 1.1035 | 73.9 % | 2.3102 | 1.7986 | **−0.5116 [−0.9795, −0.0408]** *(helps)* | ✅ |
| `k=0.25` | 4.0191 | 36.1 % | 6.0203 | 1.5873 | **−4.4331 [−5.7116, −3.2593]** | ✅ |
| `k=0.1` | 6.6911 | 18.8 % | 8.4460 | 0.8596 | **−7.5863 [−9.1561, −6.0829]** | ✅ |

> **The apparatus reports "conditioning is the lever" — monotonically, separated, over four orders
> of starvation — whenever it is true. On all four real fans it reports the opposite.** The REFUTE
> is therefore a measurement, not an absence of power.

## 5. S4 — conditioning the **real** fan makes it worse, everywhere

`raw/fanc_recentre.json`. Three transforms that impose a state-conditioned longitudinal envelope on
the real per-candidate trajectories while preserving `N` and shape diversity. REF-C-XL:

| transform | resulting slope on `v0` | ceiling | Δ ceiling | realised (nearest to REF-C's own pick) | Δ vs as-trained | sep |
|---|---:|---:|---:|---:|---:|---|
| *(baseline — no transform)* | −0.129 | **0.1640** | — | 0.4714 | — | |
| `T_affine` (`a_max` 1.0) | +1.000 | 0.8860 | **+0.7221** | 1.0446 | +0.5732 | ✅ worse |
| `T_shift` (median → `v0`) | +1.006 | 1.7875 | **+1.6236** | 1.9704 | +1.4989 | ✅ worse |
| `T_scale` | +1.188 | 3.2933 | +3.1293 | 3.5305 | +3.0591 | ✅ worse |
| `NC` — same shift on **shuffled** `v0` | −0.013 | 1.6637 | +1.4998 | 1.8270 | +1.3556 | ✅ worse |

Every transform achieves its stated goal — slope goes from −0.129 to **+1.000**, i.e. the fan
becomes perfectly `v0`-conditioned — and **every one is separated-worse**, replicated on all three
REF-C fans. The mechanism is §4's: the fan already contains the right-speed candidate, and
re-centring moves a *different* candidate into that slot, one whose lateral shape was designed for a
different speed. **The fan's longitudinal and lateral components are coupled; conditioning the
longitudinal marginal breaks the coupling.**

## 6. S2 / S3 — the CoverNet question, answered on its own terms

`raw/fanc_anchors.json`. Static anchor sets fitted by k-means on train-fold GT trajectories,
**5 episode-disjoint folds**, `A_fixed(N)` vs `A_cond(N, B)` at **matched per-window proposal
count**.

### 6.1 The ceiling does move — and the equal-storage control shows why

| ceiling (oof) | B=1 *(fixed)* | B=2 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|---:|
| N=16 | 1.2051 | 0.8806 | 0.7363 | 0.6418 | **0.6070** |
| N=64 | 0.7041 | 0.6114 | 0.5662 | **0.5435** | 0.5549 |
| N=256 | 0.5515 | 0.5230 | **0.5189** | 0.5365 | 0.5549 |

Conditioning is **separated-better at matched N** nearly everywhere (N=16, B=16: **−0.5981
[−0.7158, −0.4900]**; N=256, B=4: −0.0327 [−0.0421, −0.0217]). **Negative controls NC1 (shuffled
`v0`) and NC2 (Gaussian noise) return POSITIVE deltas at every single cell** — conditioning on a
meaningless variable never helps, so the gain is genuinely `v0`'s.

> ⛔ **But the gain is anchor count, not conditioning.** At **equal total storage**,
> `A_cond(N,B)` vs `A_fixed(N·B)`:
>
> | | conditioned | one bigger fixed set | Δ | sep |
> |---|---:|---:|---|---|
> | `A_cond(16,16)` vs `A_fixed(256)` | 0.6070 | **0.5515** | **+0.0555 [+0.0308, +0.0809]** | ✅ **worse** |
> | `A_cond(16,4)` vs `A_fixed(64)` | 0.7363 | **0.7041** | +0.0322 [+0.0062, +0.0635] | ✅ worse |
> | `A_cond(64,4)` vs `A_fixed(256)` | 0.5662 | **0.5515** | +0.0147 [+0.0048, +0.0268] | ✅ worse |
> | `A_cond(64,2)` vs `A_fixed(128)` | 0.6114 | 0.6150 | −0.0036 [−0.0187, +0.0090] | ❌ |
>
> **`v0`-bucketing is never better and is usually separated-WORSE than spending the same anchor
> budget on one larger unconditioned set.** The matched-N gain is entirely the B× larger vocabulary.

**And the whole family is the wrong regime anyway:** the best conditioned static set reaches a
ceiling of **0.5189**, while the shipped learned fan reaches **0.1640** — **3.2× better**. A
CoverNet-style static anchor set, conditioned or not, is a large step *down* from what REF-C already
does. Its ceiling does not even reach the CONFIRM bar (0.4907), let alone its realised pick.

### 6.2 The realised pick — the pre-registered bar, missed by 1.62×

Reference = REF-C-XL's own selected trajectory (own ADE 0.4714), the best real reference available
without a GPU-week.

| realised (oof) | B=1 | B=2 | **B=4** | B=8 | B=16 |
|---|---:|---:|---:|---:|---:|
| N=256 | 0.8164 | 0.8002 | **0.7968 [0.6636, 0.9482]** | 0.8024 | 0.8096 |

**Best conditioned realised pick = 0.7968** vs **CONFIRM 0.4907** — a **1.62×** miss, with the
interval nowhere near the bar. Conditioning's contribution is separated but negligible:
**−0.0195 [−0.0309, −0.0078]**.

### 6.3 ⭐ The cleanest possible demonstration that a ceiling is not a win

The in-sample arm fits anchors on the *evaluation* windows themselves — maximum possible overfit,
the anchor set literally contains the answer:

| | ceiling | realised |
|---|---:|---:|
| `A_cond(256,4)` **in-sample** | **0.0000** | **0.4167** |
| `A_cond(256,4)` out-of-fold | 0.5189 | 0.7968 |

> **With a PERFECT ceiling — 0.0000, the answer is in the anchor set — the realisable rule still
> only reaches 0.4167.** 100 % of the residual is selection. This also discharges the "thin fitting
> data" objection in advance: even with infinite, perfectly matched anchors, the realised pick does
> not collapse.

**Analytic bound, pre-registered in §4 of `PRE_REGISTRATION.md` and confirmed:** a
nearest-to-reference rule is floored at its reference's own error. C2's reference is v1's world
model at **0.4271**, so **C2 on ANY anchor set can at best TIE the STRONG bar and can never beat
it.** The realised-pick question for that family was always about recovering the 0.1374 m
quantisation tax, and it needed **53.7 %** of it. Static anchor sets charge a **0.3254 m** tax —
**2.4× worse than the fan they would replace.**

## 7. S6 — ⭐ THE REDIRECT: the missing information is a 2-D GOAL, and it is worth 88 %

`raw/fanc_goal.json`, `raw/fanc_goal_decomp.json`. Same real fan, same windows, same
nearest-anchor rule — only the **reference** changes.

| arm | what it knows | ref's own ADE | **realised** | vs as-trained | sep | headroom recovered |
|---|---|---:|---:|---:|---|---:|
| `R_cv` | nothing (constant velocity) | 0.8377 | 0.8158 | +0.3443 | ✅ worse | −112 % |
| `R_goal_LONG_only` | true 2 s **along-track** endpoint | 0.6205 | 0.5679 | +0.0964 | ❌ | **−31.4 %** |
| `R_goal_LAT_only` | true 2 s **cross-track** endpoint | 0.5586 | 0.4891 | +0.0177 | ❌ | **−5.8 %** |
| **`R_goal2s`** | **the true 2 s POSITION (both)** | 0.2616 | **0.2009 [0.1689, 0.2351]** | **−0.2705** | ✅ **better** | **+88.0 %** |
| `R_goalfull` | the entire GT trajectory | 0.0000 | 0.1640 | −0.3075 | ✅ | 100 % |

**goal − no-goal, the contrast the brief demands as an arm:** **−0.6149 [−0.8189, −0.4159]**,
separated. Replicated: REF-C-base **+87.8 %**, REF-C-small **+91.3 %**.

Three readings, and the third is the one that matters:

1. **`R_goal2s` clears both bars** (0.2009 < 0.4271 < 0.4907) and lands within **0.037 m** of the
   fan's absolute ceiling. Two scalars close 88 % of a gap that **no** loss function (Bar A), **no**
   simulator (V5 §3), **no** clip (fan-clip), and **no** anchor conditioning (this stream) could move.
2. **It is a pure INTERACTION.** Either coordinate alone is *worse than doing nothing* (−31.4 %,
   −5.8 %, neither separated). Pinning one axis leaves the fan free to be wrong on the other. **A
   goal must be 2-D to be worth anything** — which is a real design constraint on any strategic head.
3. ⚠️ **It is an ORACLE and therefore a BOUND, not a capability.** `R_goal2s` is built from GT's own
   terminal position — 2 of the 8 target scalars — and cannot be evaluated at deploy time. The
   correct claim is: *an upper bound on goal conditioning is 0.2009*, and **any realisable goal
   predictor recovers some fraction of it.** This is recorded as a **HYPOTHESIS with a named
   missing instrument**, exactly as V5 §2.3 recorded the canary-gate.
   ⚠️ **And the instrument cannot come from this corpus:** PhysicalAI-AV carries **no map, lane
   graph, junction annotation, traffic-light feature or route/goal signal** (settled at five
   independent probes; the card says verbatim *"we do not include open maps data"*). **The goal
   input must come from AlpaSim or an external corpus** — which is the same conclusion the
   strategic-brain topology work reached, now with a number attached to what it is worth.

## 8. VERDICT, TIER, AND WHAT IT LICENSES

### 8.1 What is settled — DECISION-GRADE

1. **The longitudinal vocabulary is not the binding constraint, and cannot be.** 100 % speed
   coverage; matched oracle == unrestricted oracle at **+0.0000 [0.0000, 0.0000]**. Replicated on
   **4** fans across **3** anchor counts. Both-directions validated.
2. **`v0`-conditioning the real fan is separated-WORSE**, for every transform, on every fan.
3. **`v0`-bucketed static anchor sets are separated-worse than one larger unconditioned set at
   equal storage**, and the whole static family has a ceiling **3.2×** worse than what ships.
4. **The realised pick misses CONFIRM by 1.62×**, and misses it even with a *perfect* ceiling.
5. ⇒ **Per the pre-registration: the fan was not the binding constraint either.** For the **2 s
   open-loop, ADE-scored** selection surface, proposal generation and re-scoring are **both**
   exhausted. **I am reporting this as cleanly as a win and I am not re-scoping it.**

### 8.2 What I refuse to conclude

- **NOT** "the fan is fine, keep 256 anchors." This is **ADE**, which cannot see collision or TTC.
  The fan-clip stream's own scoping applies unchanged: a big fan may still be adversarial on a
  safety metric this instrument cannot compute.
- **NOT** "goal conditioning works." An oracle bound is not a capability. Nothing here trains,
  measures or costs a *realisable* goal predictor.
- **NOT** that any of this transfers past 2 s. Every number is a 2 s number.
- **NOT** that the static-anchor result grades REF-C's fan generator — it grades *static anchor
  sets*, and it grades them worse.

### 8.3 Threats to validity I could not remove

| threat | direction | mitigation |
|---|---|---|
| Anchor sets fitted on **val-fold GT** (~700 windows), not the 2376-episode parity train corpus | favours the conditioned arm at small B, starves it at large B | in-sample arm (§6.3) is the infinite-data bound and still fails; **starvation is reported per cell** (`anchors_starved`) |
| k-means fits under flat-8-dim L2, scores under mean-waypoint L2 | small, affects both arms equally | standard trajectory-clustering practice; fidelity control passes |
| The as-trained selector **cannot** be run on a counterfactual anchor set | unknown | **stated in advance** (`PRE_REGISTRATION.md` §5), not discovered late; a retrained selector is explicitly out of scope |
| 40 episodes | widens every interval | `MODEL_REGISTRY §1.2a`: half-widths shrink ×2.8–3.9 at 600 episodes. **The headline Δ is +0.0000 with a zero-width interval, so no n can rescue it.** |

## 9. RECOMMENDATION — what enters the v5 retrain and what does not

Gates `Project Steering/Gates/flagship-v5-retrain.PREP.md`.

**DOES NOT enter v5:**
- ⛔ **`v0`-conditioned / CoverNet-style anchor sets.** Measured worse at equal storage, worse on
  the real fan, and repairing a gap that does not exist. **Do not spend a GPU-week here.**
- ⛔ **Longitudinal admissibility filtering** (already refuted by the fan-clip stream; independently
  reproduced here).
- ⛔ **Any further re-scoring of the frozen 2 s fan.** Bar A closed discriminative, V5 closed
  simulative, this closes proposal-side. The surface is exhausted.
- ⚠️ **The S-2 redirect in `…/2026-07-26-v5-imagination-selection/V5_IMAGINATION_SELECTION.md`
  §7.1 — *"the fan's longitudinal admissibility is [the highest-value engineering task]"*, S-2
  re-scoped to *"constrain the proposal distribution"* — should be AMENDED to REFUTED.** The
  fan-clip stream already asked for this; this stream is the second, independent reason, and it
  refutes the *reallocation* form the filter could not reach. **Both halves of that row are now
  measured false and it is still standing.**

**SHOULD enter v5, and is the only funded direction this stream produces:**
- ⭐ **A 2-D goal / route input, supplied to the selector.** Bound: **0.4714 → 0.2009**, 88 % of
  headroom, clears both bars, replicated ×3. **It must be 2-D** — either coordinate alone is worth
  nothing (−31 % / −6 %).
- **The next experiment is not a retrain — it is a realisable goal predictor**, and it is cheap:
  measure what fraction of the 0.2705 m an *estimated* goal recovers as a function of goal error.
  A goal predictor with error `e` can be swept directly on the staged fan at **zero GPU** by
  perturbing `R_goal2s`. **That is the cheapest discriminating experiment and it is unowned.**
- ⚠️ **The corpus, not the architecture, is the blocker.** PhysicalAI-AV has no route/goal signal at
  five probes. This is a **hard dependency on AlpaSim or an external corpus** and should be on the
  critical path, not in a backlog.

## 10. ESCALATIONS — these must not sit in a file

1. 🔴 **The S-2 redirect (*"constrain the proposal distribution"*) in
   `…/2026-07-26-v5-imagination-selection/V5_IMAGINATION_SELECTION.md` §7.1 is now REFUTED by two
   independent streams and still stands.** ⚠️ It has *already propagated* — `Project Steering/
   V5_PLAN.md` and `Project Steering/Gates/flagship-v5-retrain.PREP.md` are both being edited by
   sibling agents in the current index. It needs an edit by the owner, not a note in an incoming
   folder. *(An orthogonality instrument sat unmerged for 10 days on exactly this failure mode.)*
2. 🔴 **The goal-oracle result should gate `flagship-v5-retrain.PREP.md` before any anchor-set work
   is scheduled.** It reverses the priority order the brief was written under.
3. 🟠 **`fanc_common.pick_nearest_to`'s metric-mismatch bug class** (proximity metric ≠ scoring
   metric) belongs in `RETRACTION_LOG.md`. It produced a stable, plausible, entirely wrong 0.3655 m
   and was caught **only** by the positive control. This is the same class as the fan-clip stream's
   `base_rank` bug and the hierarchy stream's retracted arm — **three in two days**.

### 10.1 For `RETRACTION_LOG.md` — root-cause classes

- **`MARGINAL-MISTAKEN-FOR-CONDITIONAL`** — a pathology measured on a marginal distribution (the
  fan's speed envelope) was inferred to be a capability gap in the conditional (per-window
  coverage). **Two independent streams made the same inference from the same true observation.**
  The check that settles it is one line: `min_c |speed(c) − speed(GT)|`. **New class.**
- **`CORRELATION-WITHOUT-SLOPE`** — `corr = −0.974` read as violent inverted conditioning; the slope
  is −0.129 against a GT slope of +1.000, i.e. *no* conditioning. A normalised statistic on a
  near-constant quantity is dominated by its own noise. **Quote the slope.**
- **`PROXIMITY-METRIC ≠ SCORING-METRIC`** — flat 8-dim L2 vs mean-over-waypoint L2 have different
  `argmin`s. Silent, stable, plausible, wrong. Caught only by a positive control.

## 11. DELIVERABLE MANIFEST

**Every artifact is in the repo and staged. Nothing lives in only one place. Nothing was
committed or pushed.**

| artifact | where | what |
|---|---|---|
| `FAN_CONDITIONING.md` | `repo:…/incoming/2026-07-27-fan-conditioning/` | this document |
| `PRE_REGISTRATION.md` | same | bars + failing-value proofs, written before fitting |
| `code/fanc_common.py` | same | loaders, anchor machinery, transferable rules, estimators |
| `code/fanc_gate.py` | same | S0 gate + S1 premise |
| `code/fanc_coverage.py` | same | **S5, the decisive instrument** + `F_narrow` positive control |
| `code/fanc_recentre.py` | same | S4 conditioning transforms on the real fans |
| `code/fanc_anchors.py` | same | S2/S3 static anchor-set grid, OOF + in-sample + NC1/NC2 |
| `code/fanc_goal.py` | same | S6 oracle-goal arm |
| `raw/fanc_gate.json` | same | S0/S1 raw |
| `raw/fanc_coverage.json` | same | coverage, shift response, positive control |
| `raw/fanc_recentre.json` | same | all transforms × 3 fans |
| `raw/fanc_anchors.json` | same | full N×B grid, both splits, both controls |
| `raw/fanc_anchors_perwindow.npz` | same | per-window vectors, so any interval is recomputable |
| `raw/fanc_goal.json`, `raw/fanc_goal_decomp.json` | same | goal arm + long/lat decomposition |

**Inputs consumed** (all pre-existing repo artifacts; **no pod contacted, no checkpoint loaded, no
episode opened, parity `e438721ae894` untouched**): `taniteval/results/fan_refc-{xl,base}-30k.pt`,
`…/2026-07-22-refc-small-30k/fan_refc-small-30k.pt`, `…/2026-07-26-v5-imagination-selection/raw/
{v5_v1_windows_reduced.pt, fan_last_along_v4.pt}`, `…/2026-07-26-bar-a-selector/raw/
bar_a_produced_windows.pt`.

🔒 No clip UUIDs or raw PhysicalAI content appear in any artifact; episodes are opaque integers
already present in committed dumps.

**Total compute: ~9 minutes of dev-box CPU.** No GPU. Zero pod load — pod1 (training), pod2 (arm
panel), pod3 and the eval pod (IDM v3) were never contacted.

**Suite green:** `taniteval` **559 passed** (2026-07-27, this stream added no files to `taniteval/`
or `stack/`; all new code lives in the hub folder above).
