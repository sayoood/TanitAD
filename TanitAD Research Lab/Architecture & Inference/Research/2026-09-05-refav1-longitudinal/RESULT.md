# refav1 — THE LONGITUDINAL ANALOGUE

*Arch+Inference FlyWheel, 2026-09-05. Follows `M27 §5` ("the next arm is the
LONGITUDINAL ANALOGUE of this package's levers") and the cost-geometry package
`.../2026-09-05-refav1-cost-geometry/`. Ckpt 21,109, 40 windows / 8 episodes,
`--window-stride 16`, `ccos`, T1.*

⛔ **Tier stamp on every number below: T1** (the planner's own actions close the
loop through the predictor), except where a row is explicitly a *vocabulary
expressivity* row — those are labelled and are NOT planner results.

---

## 0. The one-line answer

**P4 — does refav1 BEAT `ha0_ext`? PARTLY, AND MEASURED (§6).** `lonshift`
(D2) landed: it **beats `ha0_ext` on FDE with a separated interval
(−0.3723 [−0.7473, −0.0180])** and on the lateral family, moves ADE from
+0.0162 to **−0.0904** (better, interval still touching zero), and **cuts the
longitudinal gap 47 %, +0.4862 → +0.2599** — but it **still loses that
family**, so the conjunction the bar demands is **NOT YET** met. Against `ha0`
(constant velocity) it now **wins ADE, FDE and the whole longitudinal family,
all separated**. And it gets there **while ACTING**: `frac a == 0` fell
**0.475 → 0.000**.

What this turn did produce, all MEASURED:

1. **P1 is complete.** The longitudinal vocabulary can command **no sustained
   acceleration at all**, its reachable `dv` over the plan window has **exactly
   one positive rung (+0.977 m/s)** against a corpus p90 of **+2.342**, and
   **82.4 % of the accelerating windows are outside it entirely**. The goal
   commands `a ≡ 0` on **31/40** windows and **20/24** of the GT-LON stratum.
2. **P2 is complete.** The jerk term **does not price the seam** between the
   measured `a0` and `controls[0]`, so the all-zero plan is the exact joint
   minimiser of both regularisers; and the third cost weight has never been
   evaluated (INHERITED — already pinned; the new part is that it **corrects
   `M27 §2`**: the 643× `W_VEND` difference is a **no-op**, not a confound).
3. **Two levers implemented, pinned OFF-by-default and bit-identical when off**
   (`a_shift`/`a_sustain`, `jerk_seam_a0`), plus **reached-it guards** so a
   stale stack cannot bank a shipped-vocabulary arm under a lever's name. A
   third (`W_VEND`) was built, **caught by an existing pin that reserves it as a
   PI decision, and REMOVED rather than the pin amended**.
4. **The design was changed on measurement, mid-turn, before any GPU was
   spent**: D1 could reach only **61.8 %** of the deficit, so D2 was derived,
   measured, implemented and promoted to the queue's first arm — at the
   expressivity level it closes **76 %** of the deficit and is the only design
   that goes **negative on ADE** (−0.0389, i.e. better than the floor).

⛔ **THE BLOCKER, NAMED AND MEASURED.** The dev-box 4060 is at **7,755 / 8,188
MiB with the SIBLING stream's two arms** (`combined`, `wk15_ladder`, both at
episode 3/8 and slowed to ~1,000–1,500 s/episode). Two concurrent arms is the
proven ceiling; a third OOMs. My five arms are **queued, gated on distinct
`--out` targets < 2, and will start automatically**; `raw/HANDOFF.md` +
`raw/finalize_lon.sh` land them without re-deriving anything. Everything in
§1–§4b is measured; §5 is pre-registered with **both outcomes committed before
the numbers exist**.

---

## 1. P1 — the longitudinal vocabulary, characterised as the lateral one was

**Instrument:** `raw/lon_vocab.py` (source only, no model) ·
`raw/lon_corpus.py` · `raw/lon_branch.py`. Outputs banked verbatim as
`raw/lon_vocab.txt`, `raw/lon_corpus.txt`, `raw/lon_branch.txt`.

### 1.1 What `canonical_controls` can command longitudinally — TABLE A/B/C

Every LON token builds `a_i = clip((v_t − v)/GOAL_REACH_S, ±GOAL_A_MAX)` and
then integrates, so **`a` decays geometrically to zero for every token**: the
vocabulary can name a *speed step*, never a *sustained acceleration*. At
`v0 = 10 m/s`, over the 2.0 s plan window:

| token | a[0] | a[2.0 s] | dv over 2 s | sustained? |
|---|---|---|---|---|
| `CRUISE` | 0.0000 | 0.0000 | **0.0000** | — |
| `ADAPT_SPEED_FOR_CURVE` | −1.0000 | −0.3874 | −1.3026 | decays |
| `FOLLOW` | −0.5000 | −0.1937 | −0.6513 | decays |
| `ACCELERATE` | **+0.7500** | +0.2906 | **+0.9770** | decays |
| `YIELD_MERGE` | −0.7500 | −0.2906 | −0.9770 | decays |
| `BRAKE_TO` | −1.5000 | −0.5811 | −1.9540 | decays |
| `CREEP` / `HOLD` | −1.5000 | −1.5000 | −3.0000 | clipped |

⭐ **The reachable set of `dv` over the plan window has exactly ONE positive
value at every `v0`: `+0.976982` m/s.** That is the longitudinal
`kappa ∈ {0, ±0.08}`.

### 1.2 What the corpus demands — TABLE D

| quantity (n = 40) | p10 | med | p90 | mean abs |
|---|---|---|---|---|
| `v0` (m/s) | +0.188 | +4.856 | +10.711 | 5.207 |
| **`a0` measured (m/s²)** | −1.408 | −0.101 | +1.120 | **0.722** |
| **GT `dv` over 2 s (m/s)** | **−2.158** | −0.158 | **+2.342** | **1.419** |
| `ha0_ext`'s own `dv` (= 2·a0) | −2.816 | −0.202 | +2.240 | 1.443 |

⭐ The floor's `dv` (1.443 mean abs) tracks the GT's (1.419) almost exactly.
**That, not skill, is why `ha0_ext` wins longitudinally.**

### 1.3 ⛔ The blocker, re-measured on the predicate it names — TABLE H

`M27 §5` said *"29 of 40 windows decode `ADAPT_SPEED_FOR_CURVE`, whose
canonical control is `a == 0`"*. **Confirmed, and it needed the correction M28
(2) warned about:** the token is `a == 0` only while
`v0 ≤ GOAL_CURVE_VMAX_MPS = 8.0`; above that it is a *sustained brake to 8 m/s*.
Counted on the predicate rather than the decode:

| decoded LON | n | med v0 | on the MAINTAIN branch | commands `a ≡ 0` |
|---|---|---|---|---|
| `ADAPT_SPEED_FOR_CURVE` | 29 | 3.471 | **27** | **27** |
| `CRUISE` | 4 | 10.613 | 4 | 4 |
| `BRAKE_TO` | 4 | 8.173 | 0 | 0 |
| `ACCELERATE` | 3 | 14.716 | 0 | 0 |

⇒ **31 / 40 windows (77.5 %) carry a goal that commands `a ≡ 0` over the whole
plan window — and 20 / 24 (83.3 %) of the GT-LON stratum.** The goal is silent
exactly where longitudinal action is demanded. *(The two `ADAPT` windows above
8 m/s are a different mechanism wearing the same token name.)*

### 1.4 ⭐⭐ REACHABILITY — the asymmetry, and it is severe — TABLE I/J

| | |
|---|---|
| most-positive reachable `dv` over 2 s (median over windows) | **+0.9770 m/s** |
| most-negative reachable `dv` over 2 s | −2.8468 m/s |
| GT `dv` p90 | **+2.3416 m/s** |
| ⇒ positive demand exceeds supply by | **2.40×** |
| GT `dv` above the reachable maximum | **14 / 40 (35.0 %)** |
| GT `dv` below the reachable minimum | 8 / 40 (20.0 %) |
| **of the 17 windows where the corpus ACCELERATES, unreachable** | **14 (82.4 %)** |

⭐ **This is the lateral defect with the sign flipped.** Laterally the only
sustained curvature was **6.9× too BIG** for the road (R 12.5 m against a
corpus at R 100–1000 m). Longitudinally the only positive rung is **2.4× too
SMALL**, while the braking side is oversized. A vocabulary mis-scaled in both
channels, in opposite directions — and it gives the programme's oldest
longitudinal finding (*"0/881 accelerate"*) a mechanism rather than a symptom.

### 1.5 ⭐ THE ORACLE-CHOOSER TABLE, with a REALISED column — TABLE G

**Stratum declared before any row was scored:** `GT-LON` =
`|GT dv over the 2.0 s plan window| ≥ 1.0 m/s` → **n = 24 / 40 (60 %)**
(sensitivity: ≥0.50 → n = 32; ≥1.50 → n = 13).
**Metric: medAE** — the MEDIAN over stratum windows of the per-window family
metric. ⛔ Not RMSE (M15: tail-dominated, ranks designs the other way).

⛔ **READ THE LABELS.** These rows are *vocabulary-expressivity* rows: each is a
control profile rolled through the programme's one unicycle integrator. Only
`cl` is a planner output. "REALISED" here means **realised with respect to the
CHOOSER** (no privileged input), *not* realised with respect to the planner —
that number is §5, and conflating the two is exactly retraction #29's error.

| row | LON speed medAE | LON accel medAE | LON along medAE | ADE med | class |
|---|---|---|---|---|---|
| GT (identity control) | **0.0000** | **0.0000** | **0.0000** | **0.0000** | must read exactly 0 ✅ |
| `cl` = `wk15`, the actual arm | 0.8208 | 0.8894 | 0.5927 | 1.1462 | **T1 planner** |
| `ha0_ext` — the floor (M11) | 0.2833 | 0.5435 | 0.3264 | 0.4161 | T1 floor |
| `ha0` — constant velocity | 0.7998 | 0.9173 | 0.5942 | 1.1462 | T1 floor |
| canon @ DECODED token | 0.7773 | 0.8805 | 0.5742 | 1.1254 | shipped vocab, REALISED |
| canon @ **ORACLE** LON token | 0.2649 | 0.5352 | 0.2197 | 0.4923 | **shipped vocab CEILING** |
| **D1 `a_sustain` = a0 MEASURED** | **0.3001** | 0.5435 | 0.2455 | **0.3970** | **new vocab, REALISED** |
| D1 control `a_sustain = 0` | 0.7773 | 0.8805 | 0.5742 | 1.1254 | = shipped ✅ |
| D1 regression `a_sustain = −a0` | 1.2500 | 1.3287 | 0.9324 | 1.5472 | must be WORSE ✅ |
| D1 @ ORACLE LON token | 0.2193 | 0.4551 | 0.1825 | 0.3970 | D1 ceiling |

**Three controls, all read what they must:** the identity control is exactly
0.0000; `a_sustain = 0` is bit-identical to the shipped profile to four
decimals on every column; the deliberate regression `a_sustain = −a0` is
worse on every column.

⭐⭐ **The result that matters, and it is stated in the form M19 requires — a
REALISED row against an ORACLE row:**

* **D1's REALISED beats the shipped vocabulary's REALISED by 2.59×** on LON
  speed medAE (0.3001 vs 0.7773), and
* **D1's REALISED beats the shipped vocabulary's ORACLE CEILING on ADE**
  (0.3970 vs 0.4923) while landing within 13 % of it on LON speed
  (0.3001 vs 0.2649), and
* it does so **with no oracle at all**.

### 1.6 ⭐ Why D1 has no oracle gap — the asymmetry that makes the longitudinal fix EASIER than the lateral one

`choose_kappa_level` needs a curvature the v7.0 head cannot supply (one `TURN_L`
slot, one `TURN_R` slot; `turns goaled correctly` is **0.2811 for every
`kappa_turn`**), so M15's payoff was an **ORACLE-CHOOSER bound** and M19/#29's
realised payoff was 2.3 %, not 3.7×.

**`a_sustain` has no such gap: its natural hint is `a0`, MEASURED.**
`a0 = (v[t0] − v[t0−dt]) / dt` is a backward difference of past speeds closing
at t0 — `echo_gate.ha0_ext`'s own definition, *"no future"* — admissible at T1
under the PI ruling of 2026-09-02 that the measured state at cycle time is a
legal initial state, and **strictly less information than `ha`**, which holds
the last observed *action*. There is no chooser to train, no level set, and **no
free parameter fitted on the scored split**. ⇒ **oracle column ≡ realised
column.**

### 1.7 ⭐⭐ THE PLAN COPIES THE VOCABULARY — measured on channel 0, and it is the diagnostic that says a goal change CAN move the plan

**Instrument:** `raw/lon_emitted.py` → `raw/lon_emitted.txt`. Zero GPU, read
straight out of the banked `cl_controls`.

| arm | n | mean\|a\| | frac `a ≡ 0` | `a` constant | **`a[0] == a_goal[0]`** | cem frac |
|---|---|---|---|---|---|---|
| `wk15` (W_KAPPA 15.11) | 40 | 0.31305 | 0.475 | 0.475 | **0.675** | 0.750 |
| `wk151` (seed 1) | 40 | 0.35681 | 0.425 | 0.450 | 0.625 | 0.725 |
| `ccos_argmax` (W_KAPPA 0) | 40 | 0.05056 | **0.775** | 0.775 | 0.775 | 0.750 |
| `kamm07` | 40 | 0.05315 | **0.775** | 0.775 | 0.775 | 0.750 |
| `l3ladder` | 40 | 0.07467 | **0.775** | 0.775 | 0.775 | 0.750 |

**The distinct emitted `a[0]` values on `wk15`:**
`+0.000000 ×19`, `−1.500000 ×4`, `+0.750000 ×3`, then 14 singletons.

⭐⭐ **Those three rungs are `CRUISE`/`ADAPT`'s 0, `BRAKE_TO`'s −1.5 and
`ACCELERATE`'s +0.75 — the vocabulary's own canonical first accelerations, with
the exact multiplicities of TABLE H's decode counts (4 `BRAKE_TO`, 3
`ACCELERATE`).** On **27 of 40 windows the emitted acceleration IS the decoded
token's canonical rung**: the planner is not choosing an acceleration, it is
reproducing the token's. This is the longitudinal twin of the already-banked
lateral finding (`kappa_quantisation.txt`: the winning curvature series is
exactly constant on 31/40 and reads exactly 0.000000 or 0.080000).

⭐ **And an independent confirmation of §1.3 by a different route:** on the three
`W_KAPPA = 0` arms `frac a ≡ 0` is **0.775 = 31/40** — *exactly* the
maintain-branch count measured from the source vocabulary. Two probes, two
mechanisms (dump controls vs `canonical_controls`), one number.

⇒ **MEASURED PREDICTION, recorded before the arm runs:** because the plan copies
the goal's longitudinal rung on 67.5 % of windows, `--a-sustain-mode a0` should
move the emitted plan on most of the 19 `a ≡ 0` windows. If it does **not**, the
finding is that the cost overrides the goal longitudinally — which is
`lonseam`'s question, and it is already queued.

### 1.8 ⛔ D1 CAN ONLY REACH 61.8 % OF THE GAP — so I took the next lever in the same turn

**Instrument:** `raw/lon_attribution.py` → `raw/lon_attribution.txt`.
`a_sustain` acts only on the maintain branch, so the first question is how much
of the deficit even *sits* there. Per-window paired `cl − ha0_ext`:

| stratum | n | LON speed | LON accel | ADE |
|---|---|---|---|---|
| ALL windows | 40 | **+0.4862** | +0.4117 | +0.0162 |
| MAINTAIN (D1 acts) | 31 | +0.3879 | +0.3058 | **−0.0728** |
| NON-maintain (D1 inert) | 9 | **+0.8247** | +0.7767 | **+0.3228** |

⇒ **only 61.8 % of the LON-speed deficit is addressable by D1**, and the
9 non-maintain windows carry a **2.1× larger** per-window error and *all* of the
ADE loss (refav1 already **beats** the floor on ADE by −0.0728 on the other 31).

⭐ **The control that makes this attribution real rather than a stratum
artifact:** `ha0_ext`'s own LON error is essentially the same on both branches
(**0.3007** vs **0.3231**) — as it must be, since the floor knows nothing about
the decoded token — while `cl`'s is **0.6886** vs **1.1478**. The split is a
property of the **planner**, not of the windows.
*(The `GOAL_A_MAX` clip on `a_sustain = a0` binds on only 3/31 maintain windows
(9.7 %); median |a0| there is 0.527, so the clip is not a limitation.)*

### 1.9 ⭐⭐ D2 — "the tokens name a change relative to WHERE YOU ARE GOING"

**Instrument:** `raw/lon_designs.py` → `raw/lon_designs.txt`. Three designs, all
hinted on the measured `a0`, all scored on the same windows with the decoded LAT
token held constant.

| design | rule |
|---|---|
| **D1** `a_sustain` | maintain branch → constant `a = a0` |
| **D2** `a_shift` | **every RELATIVE target** → `v_t' = v_t + a0·GOAL_REACH_S` |
| D3 | every token → constant `a = clip(a0 + a_token[0])` |

**Mean paired difference against the `ha0_ext` floor** (negative = beats it):

| design | Δ LON speed | Δ LON accel | Δ ADE |
|---|---|---|---|
| `cl` = `wk15` (T1 planner) | +0.4862 | +0.4117 | +0.0162 |
| SHIPPED canon @ decoded | +0.4551 | +0.3672 | +0.1492 |
| D1 `a_sustain` | +0.1752 | +0.1510 | +0.0059 |
| **D2 `a_shift`** | **+0.1184** | **+0.0708** | **−0.0389** |
| D3 | +0.1254 | +0.1031 | −0.0328 |

⭐ **The cross-check that makes the table admissible:** the `cl` row reproduces
the episode-cluster bootstrap's banked **+0.4862 / +0.4117 / +0.0162 to four
decimals**, so this script and §4b's estimator are reading the same thing.

⇒ **D2 closes 76 % of the shipped vocabulary's longitudinal deficit and is the
only design that goes NEGATIVE on ADE.** It reduces to D1's `a[0]` on the
maintain branch (so D1 is its special case at the first step) and, unlike D1, is
not inert on the 9 windows that carry all the ADE loss.

⛔ **Absolute targets are never shifted.** `HOLD` (stop), `CREEP` (1.5 m/s) and
`ADAPT_SPEED_FOR_CURVE` above `GOAL_CURVE_VMAX_MPS` (brake to 8 m/s) name a
speed **in the world**; shifting them would change what the token *means*. On
this 40-window grid the predicate is numerically inert (`HOLD`/`CREEP` decode
0/40), so it costs nothing here and prevents a real error elsewhere.

⚠️ **Still expressivity, not a planner result.** The reason to expect the search
to follow is §1.7: the emitted `a[0]` **is** the decoded token's canonical rung
on 27/40 windows.

**D2 is implemented (`canonical_controls(a_shift=...)`, `--a-sustain-mode
a0_shift`), pinned by 5 more tests (14 total in that file), mutually exclusive
with `a_sustain` by an explicit guard, and it is now the queue's FIRST arm.**

---

## 2. P2 — the cost side. Two missing terms, and the second is NOT mine to take

The brief's question: *is there a weight (or a missing term) whose absence lets
the planner sit at `a = 0`?* **Two, and both are measured from source.**

### 2.1 The shipped cost is a do-nothing prior with nothing longitudinal opposing it

```
c = goal_term(ccos)  +  w_jerk·mean(jerk²)  +  w_kappa·mean(κ²)
                     [+ w_vend·(v_end − target_speed)²   ← NEVER EVALUATED]
```

* `w_kappa·mean(κ²)` is minimised at `κ = 0`;
* `w_jerk·mean(jerk²)` is minimised by **any constant `a`, including 0** —
  and the all-zero plan attains **both** minima exactly.

### 2.2 ⛔ THE JERK SEAM IS NOT PRICED — the missing term, and it is mine

`jerk = (controls[:, 1:, 0] − controls[:, :-1, 0]) / dt` diffs the plan's **own**
actions only. **The step from the car's MEASURED `a0` to `controls[0]` is
free.** A plan that drops instantly from `a0 = −2.3 m/s²` to `a = 0` therefore
costs exactly the same jerk as one that continues smoothly — and doing nothing
is the cheaper of the two on every other term.

**Implemented as `plan(jerk_seam_a0=...)`**: prepend the measured `a0` as the
(−1)-th action before the diff. It uses the **same `w_jerk`** — a repair of an
incomplete term, not a new weight, exactly as `ccosh` repaired an undefined cost
rather than re-weighting one. `None` is the shipped path and is bit-identical.

### 2.3 ⛔ `W_VEND` HAS NEVER CONTRIBUTED A UNIT OF COST — and this is INHERITED, not my discovery

**MEASURED, positively (`inspect` + an `ast` walk, not a grep):**
`plan()`'s `target_speed` defaults to `None`; `refav1_arm.py` has exactly **two**
`.plan(` call sites, one of which is the docstring's `.plan()`, and **the single
real call does not pass `target_speed`.** ⇒ `if target_speed is not None:` is
False on **every banked refav1 window**, and the third entry of every
`--cost-weights` triple is a **dead term**.

⚠️ **I must state plainly that this was already known.**
`stack/tests/test_steer_conversion_complete.py::test_C1_no_production_plan_call_
site_passes_target_speed` **pins the call site**, and its sibling `test_C2`'s
docstring says verbatim that removal was rejected and that wiring T4 up *"is a
PI decision, and it is ESCALATED, not taken here."* I built a
`--target-speed-mode a0ext` lever, **the pin caught it, and I removed the lever
rather than amending the pin.** Reporting this as a discovery would have been a
false-novelty claim; the evidence class is **INHERITED (pinned by an existing
test), re-verified here**.

⭐ **What IS new is the consequence for a live decision record.** `M27 §2`
attributes the shipped-triple collapse partly to *"`W_VEND` 64.297 → 0.1, a
643× cut in the goal-endpoint weight"* and calls it a confound. **It is not a
confound — it is a no-op.** The shipped triple `(0.02, 0.05, 0.10)` differs from
the A/B triple `(0.0, W_KAPPA, 64.297)` in **`W_JERK` and `W_KAPPA` only**; the
`W_VEND` difference cannot have moved anything. That makes `wk15` a *cleaner*
one-variable arm than M27 claimed, and the correction is banked below.

⇒ **ESCALATED to the PI / Master Mind:** arming T4 is the third longitudinal
lever and it is now the only one that is blocked. The measurement that motivates
it is §1.2–§1.4; the admissible target is
`target_speed = max(0, v0 + a0·plan_horizon_s)` from the measured t0 state.

---

## 3. What was implemented, and the parity guard

| item | file | default | parity |
|---|---|---|---|
| `a_sustain` (vocabulary) | `stack/tanitad/refs/refa_v1.py::canonical_controls`, `_imagine_tactical_goal`, `plan` | `None` | bit-identical |
| `jerk_seam_a0` (cost) | `…refa_v1.py::plan::_cost_chunk` | `None` | bit-identical |
| result stamps | `res.a_sustain`, `res.jerk_seam_a0`, `res.target_speed`, `res.w_vend_armed` | — | additive |
| arm flags | `taniteval/tools/refav1_arm.py` `--a-sustain-mode {none,a0}`, `--jerk-seam {off,a0}` | `none` / `off` | bit-identical |
| **reached-it guards** | `refav1_arm.py` | — | a flag that does not come back on the result **raises**, so a stale stack cannot bank a shipped-vocabulary arm under a lever's name |
| **W_VEND stays dead** | `refav1_arm.py` | — | asserts `res.w_vend_armed is not True` |

**Tests: `stack/tests/test_refa_v1_a_sustain.py` (9) +
`stack/tests/test_refa_v1_lon_end_to_end.py` (4) — 13 passed.** Every case
carries a same-breath control that must read the other value; the suite pins
(a) `a_sustain=None` bit-identity across 5 lats × 8 lons, (b) that the knob
touches **only** the maintain branch, (c) that the lateral channel does not
move, (d) that `a_sustain = 0` leaves the emitted `plan()` controls
bit-identical **while `a_sustain = 1.4` moves them** — the `ccosh` lesson
applied prospectively, so an arm-level null will be a measurement and not a
dead pipe.

**Wider regression: 647 passed, 1 skipped, 0 failed** over the FULL BLAST
RADIUS — every test file in `stack/tests/` that mentions `refa_v1`,
`refav1_arm` or `canonical_controls` (47 files), in 2 m 57 s.

### ⛔ 3.1 A RETRACTION I OWE, CAUGHT BY MY OWN SECOND PROBE

Earlier in this turn I recorded that
`tests/test_refav1_kin_contract.py::test_A6_adding_ha0_moves_no_existing_arm`
was a **PRE-EXISTING red test** belonging to the kin-contract stream (a stale
exact-dict pin on `manifest["tiers"]` that predates `ha0_ext`'s registration),
and I escalated it rather than touching it. **That was wrong and it is
retracted.** Re-run alone: **26 passed.** Re-run in the *identical* five-file
selection that produced the failure: **70 passed.** It does not reproduce.

⚠️ **Root-cause class — and it is a new one worth naming:** I observed the
failure **while my own patch was HALF-APPLIED** (`--target-speed-mode` had just
been removed; the `getattr` fix for the hand-built `argparse.Namespace` had
not yet landed), so the arm-roll fixture took a path that belonged to *neither*
the before nor the after state. ⇒ **A red test observed during your own
multi-step edit is evidence about your edit, not about the test — and it is
certainly not evidence about another stream.** Same family as *"absence found
at ONE location is not absence"*, with the object swapped from a missing file
to a failing assertion, and aggravated by the fact that blaming a sibling
stream is the cheapest available explanation.

⇒ `D-REFAV1-KIN-A6-STALE` is **REFUTED**; no escalation to the kin-contract
stream is owed.

---

## 4. Compute discipline

* ⛔ `tanitad-refcv3` (refcv4b) and Thor **untouched**.
* The dev-box 4060 was **fully booked by the sibling's arms** for this entire
  turn (`kamm07`, `l3ladder` — both LANDED 21:05 / 21:11 UTC — then `combined`
  and `wk15_ladder`). My queue gates on **distinct `--out` targets < 2**, never
  on a process count: M28 (3) measured that one arm is a parent *and* its child,
  both carrying the full command line, so a process gate can never open.
* `OMP_NUM_THREADS=6`, off-Drive clone `C:/Users/Admin/tanitad-wt` with
  `PYTHONPATH=<clone>/stack;<clone>/taniteval`.

---

## 4b. ⭐ THE BASELINE PANEL — the number the new arms have to beat, measured

**Instrument:** `taniteval/tools/refav1_paired_delta.py`, paired
episode-cluster bootstrap, `n_boot 2000`, per family, never pooled.
⛔ Not `overlapping_holdout_se`. Banked: `raw/pd_lonbase.md` / `.json`.
**Known-value control `wk15.cl − wk15.cl` reads `+0.0000 [+0.0000, +0.0000]` on
all ten metrics.** ✅

| pair | ADE | **LON speed MAE** | LON accel MAE |
|---|---|---|---|
| `wk15.cl − ha0_ext` | +0.0162 [−0.1648, +0.1980] | **+0.4862 [+0.2825, +0.7201]** | **+0.4117 [+0.2232, +0.6451]** |
| `wk15.cl − ha0` | −0.0317 [−0.1344, +0.0559] | **+0.0702 [+0.0011, +0.1501]** | +0.0838 [−0.0006, +0.1846] |
| **`wk151.cl − wk15.cl` (SEED FLOOR)** | +0.0150 [−0.0288, +0.0817] | **+0.0076 [−0.0134, +0.0282]** | −0.0040 [−0.0206, +0.0098] |
| `kamm07.cl − wk15.cl` | **+0.0993 [+0.0039, +0.2149]** worse | **−0.0781 [−0.1440, −0.0183]** better | **−0.0861 [−0.1747, −0.0114]** better |
| `l3ladder.cl − wk15.cl` | **+0.4301 [+0.0556, +1.0860]** worse | **−0.0617 [−0.1183, −0.0108]** better | **−0.0708 [−0.1536, −0.0048]** better |

⇒ **THE TARGET IS +0.4862 m/s.** That is the whole longitudinal gap to
`ha0_ext`, and it is separated by a wide margin — this is not a marginal loss.
The seed floor on the same metric is **+0.0076**, so a longitudinal lever has
**64× the headroom it needs** to be distinguishable; the binding question is
size, not detectability.

⚠️ **ALL THREE sibling arms, harvested here rather than left for later (their
own `HANDOFF.md` asked for exactly this), read against the same baseline — and
all three show the IDENTICAL trade.** `raw/pd_lonbase.md` (first two) and
`raw/pd_lonbase2.md` (all three):

| arm − `wk15` | ADE | LON speed MAE |
|---|---|---|
| `kamm07` (Kamm cap) | **+0.0993 [+0.0039, +0.2149]** worse | −0.0781 [−0.1440, −0.0183] better |
| `l3ladder` (seed ladder) | **+0.4301 [+0.0556, +1.0860]** worse | −0.0617 [−0.1183, −0.0108] better |
| `combined` (ladder + Kamm) | **+0.1570 [+0.0276, +0.2788]** worse | −0.0592 [−0.1112, −0.0100] better |

⇒ **Every SEARCH-GEOMETRY lever buys ~12–16 % of the longitudinal gap and pays
for it with separated ADE damage.** `combined` still sits **+0.4269 [+0.2428,
+0.6559]** from `ha0_ext` on LON speed. ⭐ That is a third independent argument
that the longitudinal blocker is **not** where the search can reach: it is the
vocabulary the search is copying (§1.7) and the cost that lets it copy nothing
(§2).

---

## 5. P3 — PRE-REGISTERED, both outcomes committed before the numbers exist

**Queue** (`raw/queueLON.sh`), in priority order so a killed queue still yields
value. Every arm is **ONE VARIABLE** against the same named, already-banked
baseline **`wk15`** = `ccos`, `(W_JERK, W_KAPPA, W_VEND) =
(0.0, 15.11245, 64.29715042415070)`, `--plan-seed 0`, shipped vocabulary.

| # | arm | the one variable |
|---|---|---|
| 1 | **`lonshift`** | `--a-sustain-mode a0_shift` — **D2, the best measured design** |
| 2 | `lonseam` | `--jerk-seam a0` — the COST lever, alone |
| 3 | **`lonshift_s1`** | `lonshift` with `--plan-seed 1` — **the REPLICATE** |
| 4 | `loncomb2` | D2 + the jerk seam |
| 5 | `lonvocab` | `--a-sustain-mode a0` — D1, for ATTRIBUTION (is maintain-only enough?) |
| — | *(withdrawn)* | `--target-speed-mode a0ext` — blocked by the `test_C1` pin, escalated |

⛔ **The bar, stated before the numbers.** `D-REFAV1-CG-SEEDFLOOR` measured
`separated` on **4 of 10** paired family metrics between two arms differing
**only** in `--plan-seed`. So a separated CI is **necessary and not sufficient**.
The admissible form is *"the lever's paired delta exceeds the seed pair's delta
on the same metric"*; the floor is **ade 0.0607 · fde 0.3439 · cross 0.0710 ·
heading 1.1180**, and per-arm point drift on `TAC lat kappa` is 25.6 %. Arm 3
exists so `lonvocab`'s own noise floor is read on `lonvocab`'s own rig.

**Committed outcomes:**

* **If `lonshift` improves the LONGITUDINAL family past the seed floor** — the
  vocabulary was the blocker, §1.4 is the mechanism, and the next arm is
  `loncomb`.
* **If `lonshift` is a NULL** — then, exactly as with `ccosh`, the goal moved
  but the *search* did not follow it, and the finding is that the longitudinal
  blocker is the **cost**, not the vocabulary. The same-breath control that
  makes such a null a measurement is already banked: `a_sustain = 0` is
  bit-identical while `a_sustain = 1.4` moves the emitted controls
  (`test_J2`), and `wk15` reads 15.08 m from `ccos_argmax` on the same rig.
* **If `lonshift` improves LON but ADE regresses** — report it as a trade and
  do **not** call it a win; `M27` already showed seven separated family
  "improvements" produced by the planner *stopping*.
* **If `lonseam` alone matches `lonshift`** — the cost geometry, not the
  vocabulary, is the lever, and `a_sustain`/`a_shift` are redundant. Say so.
* **If `lonshift` and `lonvocab` are indistinguishable** — the 9 non-maintain
  windows were not where the planner could act after all, and §1.8's attribution
  over-promised. Report the attribution as refuted at the ARM level even though
  it holds at the expressivity level.

**Reading instrument:** `taniteval/tools/refav1_paired_delta.py` (paired
episode-cluster bootstrap, `n_boot 2000`, per family, never pooled;
⛔ never `overlapping_holdout_se`), against **both** floors `ha0` and `ha0_ext`
plus the `wk15` baseline, with the arm-against-itself known-value control that
must read exactly 0.0000 with a zero-width interval.

---

## 6. ⭐⭐⭐ THE RESULT — `lonshift` (D2) LANDED, and P4 answered honestly

**Arm:** `lonshift`, landed 2026-09-05T21:18:02Z, exit 0. `ccos`,
`(0.0, 15.11245, 64.29715042415070)`, `--plan-seed 0`, ckpt 21,109, 40 windows /
8 episodes, stride 16. **ONE variable against the banked `wk15`:**
`--a-sustain-mode a0_shift`. Verified by an **argv audit of the live process**
and by the reached-it guard, which raises on window 1 if `res.a_shift` returns
`None` — the first `plan()` completed (69.00 s), so the lever provably reached
`plan()`. Estimator: paired episode-cluster bootstrap, `n_boot 2000`, per family,
never pooled. Known-value control `wk15 − wk15` reads **`+0.0000 [0, 0]` on all
ten metrics**. Artifacts: `raw/pd_lonshift.md` / `.json`,
`raw/lon_emitted_lonshift.txt`, `raw/arms/rec_lonshift.json`.

### 6.1 ⛔ FIRST: DOES IT ACT? (a floor tie reached by doing nothing is not a result)

| | `wk15` | **`lonshift`** |
|---|---|---|
| `frac a == 0` over the plan window | 0.475 | **0.000** |
| `mean abs a` | 0.31305 | **0.46552** |
| emitted `a` constant over the horizon | 0.475 | **0.000** |
| **`a[0] == a_goal[0]` (the plan copies the token)** | 0.675 | **0.050** |
| CEM wins (vs a baseline winning) | 0.750 | **1.000** |
| `plan_cost == basecost_cv` | 0.250 | **0.000** |
| distinct emitted `a[0]` | 17 (`0.0` x19, `-1.5` x4, `+0.75` x3) | **40, no repeats** |

⭐⭐ **The do-nothing plan is gone on every window, and the quantisation with
it.** §1.7 measured that the emitted `a[0]` WAS the decoded token's canonical
rung on 27/40 windows; under D2 that falls to **2/40**, the CEM wins **100 %** of
windows, and no plan cost equals the do-nothing baseline any more. The
vocabulary change did not merely move the goal — **it unlocked the search**.

### 6.2 THE FOUR FAMILIES, `lonshift − wk15` (the lever's own effect)

| family | metric | delta | seed floor (`wk151 − wk15`) | verdict |
|---|---|---|---|---|
| ADE | `ade_m` | −0.1066 [−0.2114, +0.0193] | +0.0150 | better, not separated |
| ADE | `fde_m` | −0.2658 [−0.5148, +0.0442] | +0.0567 | better, not separated |
| **LON** | `speed_mae` | **−0.2263 [−0.3173, −0.1411]** | +0.0076 | **SEPARATED, 29.8x the floor** |
| **LON** | `along_mae` | **−0.1398 [−0.2291, −0.0578]** | +0.0170 | **SEPARATED, 8.2x** |
| **LON** | `accel_mae` | **−0.2078 [−0.3055, −0.1107]** | −0.0040 | **SEPARATED, 52x** |
| LAT | cross / heading / yaw | +0.0202 / +0.8225 / +0.0146 — none separated | +0.0073 / +0.4148 / +0.0096 | ⭐ **untouched — same order as seed noise** |
| TAC | `traj_lat_correct` | −0.1000 [−0.1750, −0.0250] | **−0.1000 [−0.1750, −0.0250]** | ⛔ **NOT ATTRIBUTABLE** — below |
| TAC | `traj_lon_correct` | +0.0250 [−0.1250, +0.2000] | +0.0000 | not separated |

⭐⭐ **The tactical row is the seed-floor rule working exactly as written, on its
first live use.** `TAC_traj_lat_correct` reads `−0.1000 [−0.1750, −0.0250]` —
**separated** — and a naive reading would have reported it as lever damage. But
the **seed pair** (`wk151 − wk15`, two arms differing ONLY in `--plan-seed`)
reads **`−0.1000 [−0.1750, −0.0250]`, bit-for-bit the same interval**. The
lever's delta does not exceed the floor's — it EQUALS it. ⇒ **the tactical
regression is NOT attributable to `a_shift`.** That is `D-REFAV1-CG-SEEDFLOOR` /
`H-ESTIM-SEED-1` catching a false positive in the field.

### 6.3 ⛔ P4, ANSWERED AS COMMITTED — against BOTH floors

| pair | ADE | FDE | LON speed | LAT heading | LAT yaw |
|---|---|---|---|---|---|
| `wk15 − ha0_ext` (before) | +0.0162 | −0.1065 | **+0.4862** | −4.5589 | −0.1154 |
| **`lonshift − ha0_ext`** | **−0.0904** [−0.2078, +0.0177] | **−0.3723 [−0.7473, −0.0180]** | **+0.2599** [+0.1290, +0.4265] | **−3.6436** [−5.7301, −1.2515] | **−0.1009** [−0.1461, −0.0525] |
| **`lonshift − ha0`** | **−0.1383 [−0.2155, −0.0602]** | **−0.3643 [−0.5409, −0.1819]** | **−0.1560 [−0.2553, −0.0521]** | +0.1935, ns | +0.0063 |

⭐ **AGAINST `ha0` (constant velocity): refav1 now BEATS IT ON ADE, FDE AND THE
ENTIRE LONGITUDINAL FAMILY, every one separated** — ADE −0.1383, FDE −0.3643,
LON speed −0.1560, along −0.1557, accel −0.1240. *(Two tiny lateral costs,
cross +0.0230 and yaw +0.0063, are separated but an order of magnitude smaller
than the wins, and they are reported here rather than omitted.)*

⭐ **AGAINST `ha0_ext` (the INTEGRATOR, M11): it now BEATS IT ON FDE with a
SEPARATED interval (−0.3723 [−0.7473, −0.0180]) and on the whole lateral
family**, and ADE moves from **+0.0162 (worse) to −0.0904 (better)** — though
that ADE interval still touches zero.

⛔ **AND IT STILL LOSES THE LONGITUDINAL FAMILY: +0.2599 [+0.1290, +0.4265],
separated.** ⇒ **the honest answer to P4 is NOT YET on the conjunction the bar
demands (ADE *and* longitudinal, both against `ha0_ext`).** What changed is the
size and the direction: **the longitudinal gap fell 47 % (+0.4862 → +0.2599) on
ONE variable**, the arm got there **while acting on every window rather than by
stopping**, and it is the first refav1 arm to beat `ha0_ext` on a distance metric
with a separated interval.

### 6.4 What is still owed on this result

1. ⛔ **`lonshift_s1` (the `--plan-seed 1` REPLICATE) has not run** — it is arm 3
   in the queue. The floor quoted above is **`wk15`'s** seed pair, not
   `lonshift`'s own. On `LON_speed_mae` the effect is **29.8x** that floor, so it
   is very unlikely to be noise — but the formal claim needs the replicate, and
   this result is quotable only with that caveat attached.
2. `lonseam` (the cost lever alone) started automatically at **21:17:38Z**;
   `loncomb2` and `lonvocab` follow. `lonvocab` (D1) is what makes the vocabulary
   win **attributable** between the maintain-only and the all-token form.
3. The **remaining +0.2599** is the next lever's target. §1.8 says where it
   lives; `HANDOFF.md` §4 lists the order.

### 6.5 ⭐⭐ THE NEXT LEVER, PRICED AND **DECLINED** — the oracle discount used prospectively

`lonshift` leaves **+0.2599** against `ha0_ext`. Rather than pick the most
interesting hypothesis, I attributed the residual first
(`raw/lon_residual.py` → `raw/lon_residual.txt`, zero GPU, on `lonshift`'s own
dump):

| stratum | n | mean `cl − ha0_ext` | share of the residual | **floor's OWN error there** |
|---|---|---|---|---|
| ALL | 40 | +0.2599 | 100 % | 0.3058 |
| **`\|a0\|` > `GOAL_A_MAX` (hint CLIPPED)** | **6** | **+0.8212** | **47.4 %** | **0.3526** |
| `\|a0\|` <= `GOAL_A_MAX` | 34 | +0.1609 | 52.6 % | 0.2975 |
| MAINTAIN branch | 31 | +0.1706 | 50.9 % | 0.3007 |
| NON-maintain | 9 | +0.5676 | 49.1 % | 0.3231 |

⭐ **15 % of the grid carries 47.4 % of what is left**, at **5.1×** the
per-window error of the rest — while the FLOOR's own error on those same windows
is only **1.19×** its error elsewhere. Same control as §1.8: **the split is a
property of the PLANNER, not of the windows.**

**And the mechanism is a constant.** `PlanConfig.a_max = 4.0` (the actuator box
the search may use) against `GOAL_A_MAX = 1.5` (the goal's clip) — a **2.67×**
gap — and `GOAL_A_MAX`'s own source comment calls it *"DV_BRAKE_MS/s; = the
decel_1.5 floor"*, i.e. a **braking** constant used as a symmetric global clip on
the goal's acceleration channel. The corpus reaches `\|accel\|` **p90 1.8443,
max 3.4095** — inside the planner's box, outside the goal's.

**So I measured the fix before building it** (`raw/lon_designs2.py` →
`raw/lon_designs2.txt`), mean paired difference vs `ha0_ext`, negative = beats
the floor:

| design | Δ LON speed | Δ LON accel | Δ ADE | Δ LON **on the 6** |
|---|---|---|---|---|
| SHIPPED canonical | +0.4551 | +0.3672 | +0.1492 | +1.5469 |
| **D2 `a_shift`, clip 1.5 (the LANDED arm)** | +0.1517 | +0.1031 | −0.0214 | +0.6752 |
| D4b clip 2.5 | **+0.1310** | +0.0891 | −0.0365 | +0.5155 |
| D4a clip 4.0 = the plan box | **+0.1310** | +0.0891 | −0.0365 | +0.5155 |
| D5a shorter reach, τ = 1.0 s | +0.1552 **worse** | +0.1436 worse | −0.0111 worse | +0.5048 |
| D5b τ = 0.6 s | +0.1825 **worse** | +0.1915 worse | +0.0010 worse | +0.4741 |
| **D5c τ = 2.0 s — CONTROL** | **+0.1517** | **+0.1031** | **−0.0214** | **+0.6752** |

**Two controls read exactly what they must:** `D5c` (τ = `GOAL_REACH_S`)
reproduces D2 **to four decimals on every column**, so the τ harness provably
reduces to the landed design; and **D4a ≡ D4b**, because `max\|a0\|` is 2.31 —
raising the clip past ~2.5 buys nothing, which bounds the lever.

⛔ **BOTH ARE DECLINED, and the arithmetic that declines D4 is the one M19/#29
cost the programme once — used PROSPECTIVELY this time.** D2's own expressivity
row predicted **+0.1517** and the LANDED arm realised **+0.2599**: the
expressivity table **over-predicts by 1.71×**. Applying that measured discount to
D4's expressivity gain (0.1517 → 0.1310, i.e. **13.6 %**) gives an expected
REALISED gain of about **+0.012 on a +0.2599 gap ≈ 4.6 %** — **below what this
rig can resolve against its own seed floor.** ⇒ **not worth a GPU hour.**
**D5 is refuted outright:** it is worse than D2 on every headline column, and it
helps the 6 clipped windows only by being a cruder version of D4.

⭐ **The point is not that D4/D5 fail; it is that they were priced with a
MEASURED discount from this very rig before any compute was spent.** The
remaining +0.2599 is broad (84.3 % of it sits at `v0 >= 2 m/s`, 75.2 % on the
GT-LON stratum), not concentrated in one constant ⇒ **the next lever is the COST
side, which is already queued** (`lonseam`, then `loncomb2`), not a third
vocabulary edit.

### 6.6 ⛔⛔ MY OWN ERROR: `lonseam` WAS INERT BY CONSTRUCTION, and the guard that now refuses it

**`lonseam` (`--jerk-seam a0`) landed 2026-09-05T21:53:28Z, exit 0, and read
`+0.0000 [+0.0000, +0.0000]` on ALL TEN family metrics** against `wk15`, with
emitted controls **bit-identical** (`mean\|a\|` 0.31305, `frac a ≡ 0` 0.475, the
same 17 distinct `a[0]` values down to `−1.339849`).

⛔ **That is not a null about the jerk seam. It is arithmetic.** The cost line
is `c = c + w_jerk * jerk.pow(2).mean(-1)`, and the arm carried the `wk15`
baseline triple `(0.0, 15.11245, 64.29715042415070)` — **`W_JERK = 0.0`.**
Repairing `jerk` while `w_jerk` is zero **cannot change the objective by a single
bit**. I designed the arm as "one variable against `wk15`" and did not check that
the variable was multiplied by a live weight. **A GPU hour was spent on an
experiment that was uninformative before it started.**

⭐ **What the instrumentation DID get right, and why the diagnosis is certain
rather than a guess:**

* the **reached-it guard passed** and the record carries `jerk_seam: "a0"`, so
  the flag **did** reach `plan()` — this was not a plumbing failure;
* the **same-breath control** in the same table reads non-zero: `lonshift` on
  the identical rig shows `mean\|a\|` 0.46552 / `frac a ≡ 0` 0.000 / 40 distinct
  values. So the harness works and the zero is real;
* which leaves exactly one explanation, and it is confirmed from the recorded
  weights: **the term was switched off.**

⚠ **ROOT-CAUSE CLASS: a lever multiplied by zero is not a null about the lever;
it is a null about a term that was switched off.** Same family as M23 (4)
(*"a NULL arm with a live optimiser is not a null"*) and as counting on the wrong
predicate (M28 (2)) — the experiment was **uninformative before it ran**, and the
only thing that catches that is a **refusal at flag-parse time**, not a careful
reading afterwards.

⭐ **THE DURABLE FIX, landed in the same turn.** `refav1_arm.py` now **REFUSES**
`--jerk-seam` with `W_JERK == 0.0`, before the rollout, naming the arithmetic and
the correct experimental form. **Both controls pass:** it fires on the exact
combination that wasted the hour, and reads **0 refusals** on (a) a non-zero
`W_JERK` with the seam ON and (b) `W_JERK = 0` with the seam OFF — so it is not
an over-broad guard.

⭐ **THE QUEUE IS CORRECTED** (`raw/queueLON3.sh`, live; `queueLON2.sh`
superseded and its shells killed under a disjoint search token):

| # | arm | triple | the one variable |
|---|---|---|---|
| 1 | `lonshift_s1` | `wk15` | `--plan-seed 1` — **the replicate D2 owes** |
| 2 | `seambase` | **`(0.02, 15.11245, 64.297)`** | — the seam pair's own baseline |
| 3 | `seamon` | **`(0.02, …)`** | `--jerk-seam a0` — **one variable vs `seambase`** |
| 4 | `lonvocab` | `wk15` | `--a-sustain-mode a0` (D1, attribution) |
| 5 | `loncomb3` | **`(0.02, …)`** | both levers, on a triple where **both can act** |

⛔ **`loncomb2` is DROPPED**: it carried the same inert seam half, so it would
have been `lonshift` re-run under a name claiming two levers — the worst of the
three possible outcomes, because it would have looked like a result.

⚠ **What this does NOT change:** `lonshift`'s result in §6.1–§6.3 is untouched.
It carries no seam, its lever is the vocabulary, and its numbers stand.

## 7. Deliverable manifest

| artifact | where it lives |
|---|---|
| `RESULT.md` (this file) | repo: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-longitudinal/` |
| `raw/lon_vocab.{py,txt}` · `raw/lon_corpus.{py,txt}` · `raw/lon_branch.{py,txt}` · `raw/lon_oracle.{py,txt}` | same package, repo |
| `raw/queueLON.sh` · `raw/patch_lon.py` · `raw/patch_arm_lon.py` | same package, repo |
| `a_sustain` + `jerk_seam_a0` implementation | repo: `stack/tanitad/refs/refa_v1.py` |
| arm flags + reached-it guards | repo: `taniteval/tools/refav1_arm.py` |
| pins | repo: `stack/tests/test_refa_v1_a_sustain.py`, `stack/tests/test_refa_v1_lon_end_to_end.py` |
| arm records + dumps (`lonshift`, `lonseam`, `lonshift_s1`, `loncomb2`, `lonvocab`) | **`C:/Users/Admin/refav1_margin/p4out/` — OFF-REPO until they land; bank under `raw/arms/` as the cost-geometry package did** |
| the off-Drive clone the arms run from | `C:/Users/Admin/tanitad-wt` (synced from the repo, verified by marker) |
