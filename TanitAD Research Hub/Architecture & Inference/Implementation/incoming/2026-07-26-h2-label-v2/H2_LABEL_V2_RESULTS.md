# H2 · L2 — a BEHAVIOURAL sensor-need label: design, held-out validation, verdict

**Date:** 2026-07-26 (local, Europe/Berlin). **Author:** research engineer (H2 label-v2 stream).
**Pre-registration:** `PRE_REGISTRATION_L2.md`, this folder, written **before the builder was run**.
**Status:** decision doc. **CPU only — no GPU, no training, no model inference, no pod touched.**

**Evidence classes:** `MEASURED` (ours + path) · `PUBLISHED` (cited) · `INHERITED` (another agent/doc,
NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

# 0. VERDICT IN ONE BOX

> ## **GO — training can start.**
>
> **Held-out decision-relevance lift: `2.41× · 95 % CI [1.3998, 3.7041]`**
> paired episode-cluster bootstrap (ratio form, `taniteval.ci._draws`), **B = 2000**, seed 0,
> **1,415 episode-clusters**, 2000/2000 draws used, **zero chunk overlap with DEV**.
> `P(R2 | trigger⁺) = 12.84 %` (n = 1,192 frames) vs `P(R2 | trigger⁻) = 5.32 %` (n = 213,586).
> Risk difference **+0.0752 [+0.0211, +0.1441] — separated.**
>
> All four pre-registered PASS criteria are met: CI excludes 1.0 from above · point ≥ 1.5× ·
> n⁺ = 1,192 ≥ 200 · 138 trigger-positive clusters ≥ 40.
>
> **The E1 failure mode is excluded, not merely absent.** Leave-one-chunk-out: **16/16 still exclude
> 1.0**, lift range **2.15–2.77×**. τ\* was fixed by a **power rule that never reads the lift**, and
> it is **not** the maximum of the response curve (τ = 1.0 is). There is no argmax anywhere in the
> decision path.

### Where the fix came from — the 2×2, on identical CONFIRM frames

| | old response (`Δv ≤ −1 m/s`, base 22.5 %) | **new response** (brake ≥ 2 m/s², base 5.4 %) |
|---|---|---|
| **old trigger** (`L1_gate`, 3 m) | **1.10× [0.89, 1.32]** — reproduces the refutation | 1.29× [0.85, 1.82] |
| **new trigger** (`L2`, `a_req ≥ 0.5`) | **1.85× [1.38, 2.28]** ✅ separated | **2.41× [1.40, 3.70]** ✅ |

> **The trigger carries the fix (1.10 → 1.85), the response adds on top (1.85 → 2.41).**
> This matters for trust: **the new trigger is separated even under the *original, unamended*
> response.** The GO does not rest on the response redesign alone.

### Three caveats that are part of the verdict, not footnotes

1. **Adjusting for the ego's braking state at `t` removes the separation** — 1.35× [0.82, 2.05].
   A lead-time test favours the *mediator* reading over the *confounder* reading, but does not
   settle it. §6.
2. **At junctions the label is null** — 0.45× [0.00, 1.40] — and junctions are H2's headline
   situation. All the signal is off-junction (2.86× [1.65, 4.45]). §5.2.
3. **On the genuine off-front residual — the 36.4 % where a second camera is the *only* remedy —
   the label is NOT separated** (1.66× [0.78, 3.06]). Most of the signal sits in the
   recoverable-by-crop band. **This changes what to build first.** §5.3.

**One-line reading:** *`L1` failed because it measured geometric presence against a response that
fires on a quarter of all driving. Replace the proximity with a required-deceleration counterfactual
and the speed-difference with a brake application, and the signal is there — but it is a
front-periphery signal, not yet a cross-camera one.*

---

# 1. What was run, and on what

| | |
|---|---|
| Corpus | PhysicalAI-AV, 26 local chunks with `obstacle.offline` + calibration + `egomotion` |
| **DEV** (design, threshold, amendments) | 10 chunks / **905 clips** / 139,404 frames |
| **CONFIRM** (one look) | 16 chunks / **1,415 clips** / 214,778 frames |
| Overlap | **0 chunks, therefore 0 clips** — asserted in code before any statistic (`l2_confirm.py:44`) |
| Split rule | fixed in the pre-registration **before the builder ran**: prior-threshold-selection chunks → DEV; the rest by `index mod 3` |
| Estimator | **paired episode-cluster bootstrap**, ratio form, B = 2000, seed 0, resampling clips; both arms recomputed inside each draw. `episode_index`/`_draws` imported from `taniteval/taniteval/ci.py`. **`overlapping_holdout_se` used nowhere** |
| Geometry | per-clip `(cx, cy)` **and** per-clip 6-DoF extrinsics on every projection (two-rig corpus, rig A `cy` ≈ 543 / rig B ≈ 755) |
| Cost | **~11 CPU-minutes** total (build 26 chunks ≈ 5.5 min; DEV + CONFIRM + robustness ≈ 5 min). No GPU. No download. pod1/pod2/pod3 untouched |

**Pipeline validation — the rewrite reproduces the thing it replaces.** `L1_gate` is re-implemented
inside the same builder (ego straight-CV + agent realised track + centre distance ≤ 3.0 m) and
measures **1.73 %** of frames on CONFIRM against the published held-out **1.832 %**, with a lift of
**1.10× [0.89, 1.32]** against the published **1.16× [1.00, 1.33]**. *A pipeline that could not
reproduce the refutation it builds on would not be admissible.* (MEASURED, `l2_confirm.json`)

---

# 2. The label

### 2.1 What `L1` got wrong — three mechanisms, all removed

| # | `L1_gate` | why it destroys signal | `L2` |
|---|---|---|---|
| **M1** | conflict scored against the agent's **realised** track | the agent's realised track already contains **the agent's reaction to the ego** — a resolved conflict leaves no close approach, so the label deletes its own positives. *(The substrate agent spotted this trap for the ego and froze the ego's speed; it did not apply it to the agent.)* | agent extrapolated at **constant velocity** |
| **M2** | ego continued **straight** | at a junction the ego's route is a turn — a straight continuation invents conflicts with oncoming traffic and misses the crossing one | ego follows its **realised path**, speed **frozen**, braking applied along it |
| **M3** | trigger = centre distance ≤ 3 m | 3 m between two *centres* is one lane width; the held-out profile is monotone in `d` and crosses 1.0 at ≈3.5–4 m *because proximity grades into free-flow adjacency* | trigger = **required deceleration `a_req`** over real oriented footprints |
| **M4** ⭐ | response = `v(t+4) − v(t) ≤ −1 m/s`, **base rate 22.5 %** | **this is the one nobody named.** A response that fires on a quarter of ordinary driving cannot be separated by any trigger. *(MEASURED here: 22.45 % on CONFIRM, matching E1's 23.15 %)* | **a brake application ≥ 2 m/s²**, base rate **5.36 %** |

### 2.2 The definition, as implemented (`scripts/l2_label.py`, `scripts/l2_build.py`)

```
gap(t,h)   = ||p_a(t+h) − p_ego(t+h)|| − rho_ego − rho_agent          # oriented-rectangle radial extent
a_req(a,t) = min{ A ∈ {0,.5,1,1.5,2,2.5,3,4,5,6,8} : min_{h∈(0,4s]} gap > 0 }

   ego at horizon h : its REALISED PATH at arc length  min(v(t)h − ½Ah², v(t)²/2A)
                      (straight-continued past the end of the path — i.e. the case where it stopped)
   agent at h       : p_a(t) + v_a(t)·h,  v_a from a ±0.3 s central difference of its WORLD position

a_req_off(X,t) = max a_req over agents OUTSIDE the 51.4° encoder crop and INSIDE frustum X
a_req_seen(t)  = max a_req over agents INSIDE the crop

L2_trigger(X,t) = 1  iff  a_req_off(X,t) ≥ τ*   AND   a_req_seen(t) < τ*        [τ* = 0.5 m/s²]
R2(t)           = 1  iff  v(t) ≥ 3 m/s  AND  min_{h∈(0,4s]} ā_lon(t+h) ≤ −2.0 m/s²
L2_label(X,t)   = L2_trigger AND R2
```

`ā_lon` is a centred 0.5 s moving average of `d|v|/dt` from the ~100 Hz `egomotion` speed series.
*(The `ax` column is not used: its frame is not documented in `features.csv`, and deriving the
quantity we actually mean is cheaper than assuming.)*

**The second trigger clause IS the agent-removal counterfactual** — HOIST (arXiv:2312.02467) lifted
from objects to sensors. *With* the off-front agent the ego must brake at ≥ τ; **delete every
off-front agent and nothing the encoder can see requires it.** The off-front agent is the **binding**
constraint, so a second camera is the only remedy. `L1`'s clause (iv) was a distance-only shadow of this.

### 2.3 Non-circularity

The model receives the **256 px / 51.4° front crop** and **its own speed**
(`action_dim = 3`; INHERITED, `MODEL_REGISTRY`, flagship-v1 = `flagship4b-speedjerk-30k`).
`L2_trigger` is a function of 3D `obstacle.offline` boxes, per-camera calibration, cross-camera
frustum membership, and the ego's realised future path. **None is a model input and no clause is a
lookup over one** — the `route_target = _NAV_TO_ROUTE[nav_cmd]` ⇒ `route_skill = 0.0` failure cannot
recur here by construction. The `a_req_seen < τ` clause reads in-crop agents, but *from their 3D
boxes*; recovering that term from pixels is the capability under test, not a leak.

---

# 3. ⚠️ TWO AMENDMENTS, made on DEV, before CONFIRM was touched

Both are recorded here rather than by editing the pre-registration, exactly as that document
requires. Both were decided on the development split for a **stated structural reason**, and the
un-amended forms are reported as sensitivities in §5.4 so nothing is hidden.

### A1 — the free-flow precondition is **dropped from the definition and reported as a stratifier**

**Pre-registered:** `R2` additionally required `mean alon over [t−0.5, t] ≥ −0.5 m/s²` ("not already
braking"). It was there to make the response rare.

**The defect, MEASURED on DEV** (`l2_dev.json:A1_freeflow_diagnostic`): trigger frames satisfy the
free-flow clause on **68.2 %** of cases against **84.0 %** of all frames. **The clause preferentially
deletes trigger positives — because the ego is often already reacting by the time the conflict
geometry is measurable.** That is an anti-correlation built into the response definition: precisely
the trap the substrate agent identified for *conflict scored on a realised trajectory*, committed
one layer down, on the *response*. It also halves the base rate (2.25 % → 5.67 % without it), i.e.
halves the power.

**Amendment:** drop it from the definition; report the braking state as a **stratifier** (§6) so the
reader sees the confound rather than having it silently imposed.

⚠️ **Stated plainly: with the pre-registered response the CONFIRM lift is 0.87× [0.23, 1.62] and the
verdict would have been BOUND** (§5.4, S6). The mitigation is not rhetoric, it is the 2×2 in §0:
**the new trigger is separated (1.85× [1.38, 2.28]) even under the original, unamended `L1`
response**, which no amendment touches.

### A2 — conflicts that **braking cannot resolve** are excluded from the trigger

**The decision rule was written into `l2_dev.py` before it was run** (`A2_resolvable_diagnostic`):
compare `P(R2)` for the *unresolvable-only* population (`a_req = 8`, i.e. no deceleration up to
8 m/s² clears the conflict) against the *resolvable* population. If the unresolvable population's
response rate is **lower**, it is dilution and is excluded. Either way the number is published.

**MEASURED on DEV:**

| population | n frames | P(R2) |
|---|---|---|
| **unresolvable only** (`a_req = 8`) | 64 | **0.000 %** |
| **resolvable** (`0.5 ≤ a_req ≤ 6`) | 791 | **15.04 %** |
| no off-front conflict (background) | 134,337 | 5.14 % |

**Zero of 64, against a 5.1 % background.** The unresolvable population is not merely uninformative,
it sits *below* background — the signature of a geometry artifact (the ego driving *past* a static or
lateral object, which braking cannot and should not resolve). It is **24.7 % of the `a_req ≥ 2 m/s²`
population**, so leaving it in would have progressively poisoned the high-τ end of the curve.

**Amendment:** the trigger uses `max a_req over agents whose conflict braking can resolve`. Undoing
it costs 2.41× → 1.89× [1.11, 2.90] (§5.4, S3).

---

# 4. τ\* — chosen by a power rule that never reads the lift

> This clause exists because `L1`'s 3.0 m **was the argmax of a six-point lift sweep on 80 clips**.

**Rule, pre-registered:** τ\* = the **smallest** τ in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0} m/s² with, on
DEV, **n⁺ ≥ 200 frames and ≥ 40 trigger-positive episode-clusters.**

| τ (m/s²) | DEV rate | DEV n⁺ | DEV clusters⁺ | powered |
|---|---|---|---|---|
| **0.5** | **0.5703 %** | **795** | **93** | ✅ ⇒ **τ\*** |
| 1.0 | 0.4003 % | 558 | 71 | ✅ |
| 1.5 | 0.2374 % | 331 | 49 | ✅ |
| 2.0 | 0.1557 % | 217 | 36 | ✗ (clusters) |
| 2.5 | 0.1011 % | 141 | 26 | ✗ |
| 3.0 | 0.0717 % | 100 | 20 | ✗ |

*Smallest-that-is-powered, not best-performing.* **On CONFIRM the point estimate at τ\* = 0.5 (2.41×)
is not the maximum of the curve — τ = 1.0 gives 2.77×.** The operating point sits *below* the peak,
which is the observable signature that no argmax was taken.

---

# 5. The full response curve, coverage, strata, sensitivities

### 5.1 Response curve — CONFIRM (descriptive; only τ\* decides)

| τ (m/s²) | rate | n⁺ | clusters⁺ | P(R2\|+) | P(R2\|−) | **lift** | 95 % CI |
|---|---|---|---|---|---|---|---|
| **0.5 ⇐ τ\*** | **0.5550 %** | **1,192** | **138** | **12.84 %** | 5.32 % | **2.41×** | **[1.40, 3.69]** |
| 1.0 | 0.3380 % | 726 | 96 | 14.74 % | 5.33 % | 2.77× | [1.39, 4.68] |
| 1.5 | 0.1983 % | 426 | 63 | 10.80 % | 5.35 % | 2.02× | [0.67, 3.93] |
| 2.0 | 0.1089 % | 234 | 41 | 10.26 % | 5.36 % | 1.91× | [0.24, 4.81] |
| 2.5 | 0.0810 % | 174 | 31 | 12.07 % | 5.36 % | 2.25× | [0.14, 5.93] |
| 3.0 | 0.0605 % | 130 | 21 | 13.08 % | 5.36 % | 2.44× | [0.00, 7.40] |

**The profile is flat-to-broad, not spiky, and every point estimate is ≥ 1.9×.** It does not rise
with τ as I hypothesised; the honest reading is that τ selects *rarity*, not *severity of the
association*, and the CIs beyond τ = 1.0 are too wide to rank. DEV shows the same shape
(2.67 / 2.71 / 2.19 / 1.95 / 1.75 / 1.59), which is itself a reproduction.

**⭐ The response-severity axis is where the mechanism shows.** Holding the trigger at τ\* and
varying only the brake magnitude that defines `R2`:

| brake threshold | −1.5 m/s² | **−2.0 (used)** | −2.5 | −3.0 |
|---|---|---|---|---|
| **lift** | 2.31× [1.55, 3.05] | **2.41× [1.40, 3.69]** | **3.94× [1.96, 6.60]** | **6.27× [2.28, 12.18]** |

**Monotone increasing, and every one excludes 1.0. The harder the ego actually braked, the more
strongly `L2_trigger` predicted it** — that is what a real "the ego had to yield" mechanism looks
like, and it is not something a geometric artifact produces.

### 5.2 Coverage, class balance, strata (CONFIRM, MEASURED)

| quantity | value |
|---|---|
| **trigger rate** | **0.5550 %** of frames (1,192 / 214,778) |
| **label rate** (`L2_label` = trigger ∧ R2) | **0.0712 %** (153 frames) |
| episode-clusters trigger-positive | **138 / 1,415** (9.75 %) — **3.5× the n ≥ 40 bar** |
| episode-clusters label-positive | 31 / 1,415 |
| left camera / right camera | 0.3157 % / 0.2393 % |
| **both cameras at once** | **0.0000 %** — never co-fire in this corpus |
| response base rate | **5.36 %** (vs `L1`'s 22.45 % on the same frames) |
| triggering agent class | automobile **821**, person 122, rider 98, bus 87, heavy_truck 43, trailer 14, stroller 4, other_vehicle 3 |

*Class balance is physically sensible — dominated by vehicles, with a real pedestrian/cyclist tail.
Under `L1` at τ = 1.5 on a single chunk the triggering set contained **no automobiles at all**, which
was the first sign the old geometry was firing on box-size artifacts.*

| stratum | frames | n⁺ | trigger rate | **lift** | 95 % CI |
|---|---|---|---|---|---|
| **junction (in)** | 16,568 | 212 | **1.2796 %** | **0.45×** | **[0.00, 1.40]** ❌ |
| junction (out) | 198,210 | 980 | 0.4944 % | **2.86×** | [1.65, 4.45] ✅ |
| lane change (in) | 7,178 | 26 | 0.3622 % | 1.72× | [0.00, 6.95] — underpowered |
| lane change (out) | 207,600 | 1,166 | 0.5617 % | 2.48× | [1.41, 3.85] ✅ |

> ⚠️ **Junctions enrich the NEED (2.6× the trigger rate) and destroy the ASSOCIATION.** The most
> likely mechanism is the one E1 already recorded for the `< 3 m/s` regime: at a junction the ego is
> frequently already creeping or stopped, so "yields" and "is already yielding" are not separable by
> a brake-onset response. **This is the single most important open item**, because "entering the
> junction, activate the left camera" is the PI's own example of what H2 is for. §8, U-1.

### 5.3 ⚠️ The scope result that changes what to build

| trigger scope | n⁺ | **lift** | 95 % CI |
|---|---|---|---|
| **outside the 51.4° encoder crop** (primary, comparable to `L1`) | 1,192 | **2.41×** | [1.40, 3.70] ✅ |
| **outside the FULL 120.5° front field** (E0's genuine off-front residual) | 505 | **1.66×** | [0.78, 3.06] ❌ |

E0 measured (INHERITED) that **63.6 %** of `L1` positives are *recoverable by widening the crop* and
only 36.4 % are genuinely off-front. **The decision-relevant signal concentrates in the recoverable
band.** On the residual — where `cross_left`/`cross_right` are the *only* remedy — the label is not
separated at this n.

This does **not** overturn E0's compute finding (selective activation is still ~2.2× cheaper than
resolution-preserving widening, INHERITED). It says something narrower and more useful: **the
capability we can demonstrate today is "attend to the front periphery", and the cross-camera residual
is a smaller, harder, currently-unproven claim.** Building the head on the primary scope is
defensible; headlining "we learned when to switch on the side cameras" is not yet.

### 5.4 Sensitivities — all descriptive, none can move the verdict

| | n⁺ | lift | 95 % CI | reading |
|---|---|---|---|---|
| **PRIMARY** (ego path-speed, agent CV, resolvable, crop scope) | 1,192 | **2.41×** | [1.40, 3.69] | the verdict |
| **S3** unresolvable INCLUDED (undoes A2) | 1,520 | 1.89× | [1.11, 2.90] | A2 is worth ~0.5× |
| **S1** ego pure-CV straight (isolates **M2**) | 1,251 | 1.78× | [1.07, 2.70] | vs S3's 1.89× ⇒ **M2 contributes ≈ nothing** |
| **S2** agent REALISED future (isolates **M1**) | 463 | **3.93×** | [1.79, 6.78] | vs S3's 1.89× ⇒ **M1 is measured BACKWARDS** |
| **S4** residual scope | 505 | 1.66× | [0.78, 3.06] | §5.3 |
| **S6** response WITH the free-flow clause (undoes A1) | 1,192 | **0.87×** | [0.23, 1.62] | §3, A1 |
| **S7** response = the refuted `L1` one | 1,192 | 1.85× | [1.38, 2.28] | trigger works under the OLD response |

> **S1 and S2 must be read against S3, not against the primary** — the realised-future and pure-CV
> tables carry no resolvable-only variant, so they include the population A2 removes. S3 is the
> matched control.

**Two of my own design hypotheses are refuted by my own sensitivities, and I am reporting them as
measured:**

- **M2 (the path-preserved ego route) buys nothing.** 1.78× vs 1.89×. The elaborate arc-length
  machinery is not what made `L2` work.
- **M1 is backwards.** Extrapolating the agent at constant velocity was supposed to *recover*
  conflicts that the agent's reaction had erased. Measured, the **realised**-future agent gives a
  *higher* lift (3.93× vs 1.89×) on a third of the positives. The CIs overlap heavily
  ([1.79, 6.78] vs [1.11, 2.90]) so this is a **lead, not a result** — and adopting it now would be
  precisely the re-sweep the pre-registration forbids. Recorded for a future pre-registration.

**What actually did the work, then?** By elimination and by the 2×2: **M3 (required deceleration over
real footprints instead of centre distance) plus M4 (a brake application instead of a speed
difference), with A2 as a real but secondary contributor.**

---

# 6. ⚠️ The caveat that decides how far to trust this

**Adjusting for the ego's braking state at `t` removes the separation.** (MEASURED,
`l2_robustness.json`, Mantel–Haenszel pooled risk ratio, episode-cluster bootstrap B = 2000)

| adjustment | lift | 95 % CI | separated |
|---|---|---|---|
| **none — the pre-registered verdict** | **2.41×** | [1.40, 3.70] | ✅ |
| ego **speed** (6 bins) | **2.09×** | [1.19, 3.38] | ✅ |
| **braking state** at `t` (2 strata) | 1.35× | [0.82, 2.05] | ❌ |
| speed × braking state (12 strata) | 1.47× | [0.89, 2.20] | ❌ |

The imbalance is large: `P(already braking | trigger⁺) = 35.6 %` vs `14.1 %` for trigger⁻, and
`P(R2 | already braking) = 22.1 %` vs `2.6 %` in free flow — **an 8.5× response ratio on the
adjustment variable itself.**

**Is braking state a confounder or a mediator?** They demand opposite handling and make opposite
predictions about *order*, so the question is empirical.

**The lead-time test** (MEASURED, `l2_leadtime.json`): among the 138 trigger-positive CONFIRM clips,
32 have a hard brake onset within 4 s of the trigger onset.

| | value |
|---|---|
| trigger **strictly precedes** the brake | **65.6 %**, median lead **0.55 s**, mean 0.90 s |
| ego already braking at trigger onset | 34.4 % |
| **control** (no-trigger clips, random 4 s anchor) — already braking at anchor | 22.4 % |
| **clip-level rate of a hard brake in the 4 s window: trigger clips vs control clips** | **23.2 % vs 5.25 % ⇒ 4.4×** |

**Reading, stated as carefully as the evidence allows:** in two of three cases the trigger fires
*before* the brake, which is what a **mediator** looks like — the driver begins reacting before the
conflict geometry peaks, so conditioning on braking state blocks part of the very path being
measured (over-adjustment). But trigger frames are still enriched in already-braking states well
above the 22.4 % control base, so a density confound also operates. **Both channels are real.**

> **The defensible statement: the effect lies between ≈1.35× and ≈2.41×, and the lower end is not
> separated from 1.** The pre-registered verdict is the unadjusted number and it is GO — I am not
> retro-fitting the criterion — but a reader who wants one number for a slide should be given
> **2.09× [1.19, 3.38]** (speed-adjusted, still separated), not 2.41×.

**Corroborating that this is not one lucky chunk** — the exact failure that killed `L1`:

| check | result |
|---|---|
| **leave-one-chunk-out** | **16/16 still exclude 1.0**, lift range **2.15–2.77×** |
| per-chunk lift | 12/16 above 1.0, median **2.80×** |
| DEV → CONFIRM reproduction at τ\* | 2.67× [1.55, 4.07] → **2.41× [1.40, 3.69]** |

Two CONFIRM chunks report exactly 0.00× (1860 with n⁺ = 9; **2500 with n⁺ = 160**). Chunk 2500 is
**not** a degenerate response channel — its `R2` base rate is 4.67 % with 590 response frames and
`p05 alon_fut_min = −1.98`. It is a genuine high-speed (mean 14.1 m/s) chunk-scale null, and it is
the kind of heterogeneity that a 90-clip sample produces. Reported, not explained away.

---

# 7. C-EFF re-derived on a trigger that is now decision-relevant

| policy | left | right | either | **cams/frame** | saved vs always-on-7 |
|---|---|---|---|---|---|
| instantaneous | 0.3157 % | 0.2393 % | **0.5550 %** | **1.0055** | **85.64 %** |
| ±1 s hysteresis | — | — | 1.9201 % | 1.0192 | 85.44 % |
| ±2 s hysteresis | — | — | 3.0674 % | 1.0308 | 85.27 % |

**This is the unlock.** `H2_PHASE1_PLAN.md`'s superseding header states (correctly, at the time):

> *"the honest form is 'an off-front agent is geometrically proximate to the ego's path on 0.67 % of
> frames' — **not** 'the ego needed another camera on 0.67 % of frames'. The efficiency arithmetic is
> sound; the semantics of the trigger are open until a decision-relevant label exists."*

**A decision-relevant label now exists.** C-EFF may be stated in the stronger form **for the primary
(front-periphery) scope** — with §5.3's scope caveat and §6's adjustment caveat attached, and
`prov: "autolabel"` on the substrate.

---

# 8. Limitations, stated plainly

1. **§6 is the binding one.** Braking-state adjustment removes the separation; the lead-time test
   favours mediation but does not prove it. No observational design on this corpus can fully separate
   them — an interventional or matched-pair design would be needed.
2. **The GO depends on amendment A1** in the sense that the pre-registered response yields 0.87×.
   A1 was made on DEV for a stated structural reason, with the un-amended form published (S6). The
   defence is S7: **the new trigger is separated under the original response too.**
3. **Junctions are null (0.45× [0.00, 1.40])** — H2's headline situation. §5.2, U-1.
4. **The genuine off-front residual is not separated (1.66× [0.78, 3.06])** — the part that actually
   requires a second camera. §5.3, U-2.
5. **`obstacle.offline` is `scene:obstacles:autolabels:v2` — machine labels, not human GT.** Stamp
   `prov: "autolabel"`. Systematic misses of small/distant agents attenuate any lift; not excluded here.
6. **Agent velocity is a ±0.3 s finite difference of autolabelled box centres**, not a measured
   velocity. It is the noisiest input in the chain and it feeds M1, which §5.4 shows is the weakest link.
7. **`a_req = 8` conflates "needs ≥ 8 m/s²" with "not resolvable in the grid."** A2 removes the whole
   class; in a corpus with no collisions that is the right call, but it is a stated conflation.
8. **Frames with no annotated agent are not in the denominator** — inherited deliberately from the
   E0/E1 aggregation so the rates stay comparable. It makes the efficiency rates **conservative**.
9. **Strata are kinematic proxies**, not map labels — PhysicalAI-AV ships no HD map (three-leg
   absence, `H2_SUBSTRATE §B.3`).
10. **1,415 CONFIRM clips are not the 2,376-episode parity corpus.** **Nothing here re-selects
    training episodes; parity is untouched.**
11. **Predictability is still untested.** This establishes that the label is *decision-relevant*. It
    says nothing about whether a front-camera-only model can *predict* it — that is E3/U-1 of the
    plan doc, ~30 GPU-min, and it is the next gate, not this one.

---

# 9. Recommendation

**1. GO — build the head.** `L2_trigger` at τ\* = 0.5 m/s², **primary (out-of-crop) scope**, as a
**per-camera independent Bernoulli** — never a softmax over mixed axes (`H2_SUBSTRATE §C.1`; the
5-way maneuver softmax defect). Attach at **T3 `[B, 8, 2048]`** and route through the dynamics
(T6/T7), not off the static latent — the frozen-WM bank measured 3.649 m off T2 versus 0.599 m
through the dynamics (INHERITED, `H2_SUBSTRATE §F.3`). Keep the encoder frozen.

**2. Run E3 (encoder-feature decidability) BEFORE the head, not after.** ~30 GPU-min, no training.
The label is now worth predicting; whether it *is* predictable from the 51.4° crop is unanswered, and
a null there is cheaper to find now than after a pod-day.

**3. Class-balance the head deliberately.** 0.555 % positives, 138 of 1,415 episodes. Report
**PR-AUC against the base rate, never accuracy**, log routing entropy from step 0, and state the
asymmetric cost (a missed activation is a safety failure; a spurious one costs compute).

**4. Do NOT re-sweep τ, and do NOT adopt S2.** The agent-realised-future variant (3.93×) is a
**lead for a new pre-registration**, not a result. Adopting it here would be the exact error that
killed `L1`, one level up.

**5. Two escalations, raised rather than written into a doc** (the "10-day README" failure mode):
   - 🔴 **`H2_PHASE1_PLAN.md`'s superseding header needs updating in place.** Its statement that
     C-EFF cannot be given its strong form "until a decision-relevant label exists" is now
     **satisfied for the primary scope** — and its §3 E-sequence, which "restarts at the label", can
     resume at **E3**.
   - **`RETRACTION_LOG.md`** should carry §2.1's **M4** as a root-cause class in its own right:
     *"a null was attributed to the trigger when the RESPONSE variable was the unpowered half — a
     response with a 22 % base rate cannot be separated by any trigger."* E1's own robustness table
     (four flat response thresholds) contained the evidence and was read as *"the response is
     under-powered rather than mis-tuned"* — correct, and not acted on.

---

# 10. Deliverable manifest

**All artifacts are in the repo working tree at the path below. Nothing is on a pod or in a
worktree. Per instruction: no `git add`, no commit, no push was performed by this agent.**

`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-h2-label-v2/`

| file | what it is |
|---|---|
| `PRE_REGISTRATION_L2.md` | written before the builder ran; contains the split, τ-rule and PASS/BOUND criteria |
| `H2_LABEL_V2_RESULTS.md` | this document |
| `l2_dev.json` | DEV: response base rates, the **A1** and **A2** diagnostics, the τ\* power table, the DEV curve |
| `l2_confirm.json` | **the verdict** + full curve + coverage/balance + strata + speed-matched MH + 7 sensitivities + the 2×2 + per-chunk + C-EFF |
| `l2_robustness.json` | joint MH adjustment, confound balance, the zero-lift chunk diagnostics, **leave-one-chunk-out** |
| `l2_leadtime.json` | the mediator-vs-confounder discriminating test |
| `scripts/l2_build.py` | the label-table builder (footprint `a_req`, path-speed ego, CV agents, `L1` replication) |
| `scripts/l2_label.py` | **the label**: one definition, imported by both drivers so they cannot diverge |
| `scripts/l2_dev.py` · `l2_confirm.py` · `l2_robust.py` · `l2_leadtime.py` | the four analyses, in run order |

**Reused, not copied:** `2026-07-25-h2-e0-e1/scripts/_vendored_crux.py` (f-theta projection, per-clip
`(cx, cy)` + per-clip 6-DoF extrinsics, the encoder-crop test) and `h2e_stats.py` (the paired
episode-cluster bootstrap over `taniteval.ci`).

**Intermediate table is NOT in the repo** (26 chunk parquets, ~24 MB) and lives at
`…\scratchpad\l2tab\`. It rebuilds from the repo scripts in **~5.5 CPU-minutes**:

```
python scripts/l2_build.py <scratch>\l2tab 0036 0170 0174 0181 0617 0834 0840 0852 0868 0906 \
       0919 0928 0931 1573 1852 1860 1864 1870 1880 1900 2433 2498 2500 2503 2820 2838
python scripts/l2_dev.py && python scripts/l2_confirm.py \
       && python scripts/l2_robust.py && python scripts/l2_leadtime.py
```

**Data read (read-only, dev box):** `C:\Users\Admin\tanitad-data\physicalai\` —
`labels/obstacle.offline/*.zip` (26), `labels/egomotion/*.zip`,
`calibration/camera_intrinsics/*.parquet` + `sensor_extrinsics/*.parquet`,
`calibration/calibration/vehicle_dimensions/`. **No download. No pod touched. No GPU used. No
training job perturbed.**
