# WP-A RESULT — the v9 label release (Stage 2: route-checkpoint echo / leak study · Stage 3: build + validation)

*TanitAD Data FlyWheel · 2026-10-04. Pre-registration: `SPEC.md` (accepted by the Master Mind with D-WPA-1..8),
`SPEC_ADDENDUM_S2A1.md`, `SPEC_ADDENDUM_S3A1.md` — sha256 + UTC time in `raw/SPEC_SHA256.txt`, each recorded before
its numbers existed. Every number below is **MEASURED** from the raw JSON named beside it; the Stage-2 tables are
GENERATED (`code/s2_make_tables.py` → `raw/TABLES_s2_generated.md`), never hand-copied. Tier: label- and input-level
measurements on logged data; no model was trained; nothing here is a driving result. Intervals: episode-cluster
bootstrap over clips (B = 2000) — they answer "another draw of episodes" only.*

---

## Stage 2 — the ECHO / LEAK study for the route checkpoint (SPEC §9.E)

**Verdict, as registered: no RC variant is admissible as built — E2 (speed leak, against NAV) FAILS for every variant
and every pre-registered lever under the nonlinear model class.** E1 and E3-A1 read well; E2 decides. Diagnosis in
§S2.5. ⇒ Master Mind ruling (provisional, PI pending): road-geometry speed information is admissible, the reference
is the road-level route — and the registered follow-up **E2′/E3′ CONFIRMS RC-A50 with the noised training input**
under a CERTIFIED nonlinear instrument (§S2.7).

### S2.1 E1 — the trivial-planner floor (the "echo"), EVAL-DIAG vs refcv7's banked pick (`raw/s2_echo_leak.json`)
A constant-curvature arc tangent to the NOW heading through the checkpoint, driven at constant v0. On the route
package's 1,112-window grid (1,071 with every variant valid; 107 GT-turn / 574 straight windows, 38 turn episodes):

| arm | turn dir-correct | turn heading ≤ 15° | turn ADE m | straight ADE m | all ADE m |
|---|---|---|---|---|---|
| refcv7 E9 pick (banked) | 0.841 [0.750, 0.918] | 0.514 [0.427, 0.598] | 3.19 | 1.60 | 1.97 |
| TP · RC-A30 | 0.888 | 0.477 | 4.18 | 2.42 | 2.80 |
| TP · RC-A50 | 0.916 | **0.654** [0.531, 0.763] | 4.59 | 2.41 | 2.84 |
| TP · RC-A80 | 0.953 | 0.542 | 5.50 | 2.50 | 3.06 |
| TP · RC-B | **0.972** [0.926, 1.000] | 0.626 | 4.77 | 2.43 | 2.89 |
| TP · RC-A50 driven at the GT speed profile | 0.944 | 0.626 | 2.21 | 0.32 | **0.66** |
| CTRL deranged RC-A50 | 0.178 | 0.000 | 10.64 | 7.71 | 8.07 |
| CV straight | 0.000 | 0.000 | 9.64 | 2.60 | 3.92 |

* **The floor refcv8 must beat (R8-3):** a planner that merely aims at the checkpoint turns the right way on
  0.89–0.97 of GT-turn windows (refcv7 0.841) and holds the heading within 15° on 0.48–0.65 (refcv7 0.514). Its ADE
  is worse than refcv7 only because it keeps v0 — with the GT speed profile the same arc reaches **0.66 m** all-window
  ADE against refcv7's 1.97 m: the checkpoint's lateral content is strong; the speed choice is what is missing (the
  route package's B3 finding, seen from the input side).
* All eval139 windows (22,358 of 23,772 with log GT and every variant valid; 2,795 GT-turn windows): the same
  ordering — heading ≤ 15° A50 0.640, B 0.625, A80 0.547, A30 0.449; direction 0.90–0.94.
* ⚠️ **E1's registered control FAILED as worded:** "the deranged-RC TP must be worse than CV-straight on GT-turn
  heading ≤ 15°" is unsatisfiable — CV-straight scores a structural 0.000 there (a straight plan never matches a ≥ 30°
  turn). The deranged arm reads 0.000–0.028. The control's intent holds on every other reading (deranged worse than
  CV on all-window ADE, 8.0–8.3 vs 3.9 m; direction 0.15–0.21 vs 0.89–0.97 with the true RC). Reported, not
  re-worded.
* GT cross-check: log GT vs the bank's `gt` on 8,339 EVAL-DIAG slots: median 0.130 m, p95 0.65 m (pose/clock
  resampling); every EVAL-DIAG comparison uses the bank's own `gt`, as pre-registered.

### S2.2 E2 — speed leak (`raw/s2_levers_*.json`, `raw/s2_echo_leak.json`)
ρ = (R²(F) − R²(F0)) / (1 − R²(F0)), mean over τ = 1…6 s, 5-fold clip-grouped OOF; bar: `ρ(RC) ≤ ρ(NAV) + 0.03` AND
`ρ(NAV+RC) − ρ(NAV) ≤ 0.03`, under BOTH model classes. Decisive population: the seeded 600-clip TRAIN sample
(92,579 windows, 576 clips with every arm valid).

| instrument (trainS) | certified | ρ NAV | NAV+RC − NAV: A30 / A50 / A80 / B | ρ deranged (must be ≈ 0) | ρ O1 oracle | ρ target |
|---|---|---|---|---|---|---|
| OLS | yes | +0.040 | +0.010 / +0.006 / +0.009 / +0.010 — **PASS** | −0.003 | +0.329 | +1.000 |
| trees (registered) | **no** | +0.149 | +0.131 / +0.089 / +0.070 / +0.058 — **FAIL** | −0.059 | +0.381 | +0.997 |
| trees v2 (min leaf 200, l2 1.0; post-hoc repair) | **no** | +0.171 | +0.119 / +0.078 / +0.059 / +0.052 — **FAIL** | −0.044 | +0.382 | +0.997 |

* **Linear: no leak. Nonlinear: a clear leak** — the checkpoint adds 0.05–0.13 of the future-speed information v0
  lacks beyond what the nav args give. CLAUDE.md: a linear negative is not a negative about learnability, so the
  verdict is FAIL.
* **The tree instrument is not certified** (deranged control −0.044 to −0.059; bar |ρ| ≤ 0.01): it over-fits
  clip-specific feature values and reads useless features BELOW zero, so its positive increments are, if anything,
  UNDER-estimates — the FAIL is robust to the instrument's defect. The v2 repair (leaves must span more than one clip)
  did not certify it; that is reported, not hidden.
* On eval139 (132 clips) the trees fail their controls badly (deranged −0.13 to −0.21, target 0.91–0.94): 139 clips
  are too few for this model class, so eval139's tree readings are inadmissible; OLS passes there too.
* ⚠️ **The O1 positive control reads 0.28–0.38 against its registered ≥ 0.40 — FAILED as written** (a scalar oracle in
  a per-τ regression; D4's 0.57 was a one-hot over a different target). It still reads ~10× the RC's linear values and
  ~3× its tree increments, so the instrument detects a known oracle — the threshold was mis-set, not the instrument.
* Nav's own leak (diagnostic, SPEC §5.2; trainS, the first E2 run `raw/s2_echo_leak.json`): the time-based token +
  `t_next` recover **+0.089** (OLS) / **+0.267** (trees) against +0.039 / +0.152 for the spatial nav — the reason
  D-WPA-4 keeps time-to-turn out of the input is now measured.

**The pre-registered levers (E4 order), trainS, NAV+RC − NAV, trees v1 / v2:**

| arm | A30 | A50 | A80 | B |
|---|---|---|---|---|
| registered σ 8 m (x, y, ψ) | +0.131 / +0.119 | +0.089 / +0.078 | +0.070 / +0.059 | +0.058 / +0.052 |
| L1 σ 15 m | +0.106 / +0.090 | +0.083 / +0.071 | +0.069 / +0.059 | +0.062 / +0.052 |
| L2 no heading field | +0.117 / +0.105 | +0.083 / +0.072 | +0.067 / +0.059 | +0.048 / +0.043 |
| L3 noised (σ_along 2 m, σ_lat 0.75 m) | +0.043 / +0.036 | +0.050 / +0.043 | +0.050 / +0.045 | +0.040 / +0.036 |
| DIAG L1 + L2 | +0.099 / +0.088 | +0.077 / +0.066 | +0.067 / +0.057 | +0.047 / +0.043 |
| DIAG the heavy σ 25 m route itself (A50) | — | **+0.074 / +0.063** | — | — |

Every lever FAILS the 0.03 bar; the noised input comes closest (+0.036 to +0.050).

### S2.3 E3 — lateral leak (`raw/s2_echo_leak.json`; + addendum S2-A1)
| population | variant | p90 \|δ\| all (registered ≤ 1.0 m) | straight-support p90 (A1 ≤ 1.0 m) · share | curved-support p90 |
|---|---|---|---|---|
| eval139 | A30 / A50 / A80 / B | 3.86 / 3.63 / 3.02 / 5.61 — **FAIL** | 0.33 / 0.34 / 0.31 / 0.33 — **PASS** · 0.42–0.49 | 4.58–6.89 |
| trainS | A30 / A50 / A80 / B | 4.34 / 3.77 / 3.20 / 5.50 — **FAIL** | 0.33 / 0.30 / 0.26 / 0.27 — **PASS** · 0.37–0.44 | 4.84–6.45 |

* **E3 FAILED as registered; the failure is localised on curved-support windows**, where the σ = 25 m reference cuts
  corners by metres (the addendum's ESTIMATED mechanism, now MEASURED: curved p90 4.6–6.9 m vs straight 0.26–0.34 m).
* Where the reference is faithful, the checkpoint's offset from the road-level route is small: p90 0.26–0.34 m;
  windows with a lane-change-scale excursion have median 0.11–0.16 m vs 0.04–0.05 m without — the lane-level leak
  exists and is sub-lane in size. dev6 OOF ΔR² of the input over the heavy route: −0.04 to +0.05.

### S2.4 E4 — the decision rule, applied (`raw/s2_echo_leak.json → E4`)
No variant is eligible: all pass E3 via A1 (failure localised on curved support) and all FAIL E2. The pre-registered
consequence: the levers run (S2.2, all FAIL), and **the RC goes to the Master Mind as "not admissible as built",
with the numbers.**

### S2.5 Diagnosis (post-hoc; not a bar result)
* **The speed information is ROAD GEOMETRY, not the ego's lane-level choices.** The heavy σ = 25 m route — which has
  no lane-level detail at all — carries +0.063 / +0.074 on its own, i.e. 80–90 % of the σ = 8 m checkpoint's
  increment at 50 m (+0.078 / +0.089). Curvature ahead predicts slowing and accelerating; the nav args only cover
  ANNOUNCED junction turns, so NAV is too narrow a reference for "what a route legitimately implies about speed". A
  real navigation polyline carries the same curvature.
* **The noised checkpoint carries LESS speed information than the road-level route** (RC-A50 noised +0.043 / +0.050
  vs heavy route +0.063 / +0.074; RC-B noised +0.036 / +0.040).
* So the open question is not technical: *is road-geometry (curvature-ahead) information admissible inside a route
  input?* If yes, the right reference is a road-level route, not the nav args, and the noised checkpoint passes it.
  If no, no geometric route input passes — a turn-direction-only signal (the nav token) is the most the programme
  may feed.

### S2.6 What I recommend (for the Master Mind's decision)
1. **If road geometry is admissible:** **RC-A50 with the noised training input** (σ_along 2.0 m, σ_lat 0.75 m) and RC
   dropout ≥ 0.3. It has the highest heading-within-15° floor (0.640 all windows, 0.654 EVAL-DIAG), passes E3-A1
   (p90 0.30–0.34 m), and its noised speed increment (+0.043 / +0.050) sits below the road-level route's
   (+0.063 / +0.074). RC-B is the runner-up (best direction 0.972, lowest noised leak +0.036 / +0.040).
   ⚠️ That reading is post-hoc: it needs its own pre-registered bar (E2′: `NAV+RC_noised − NAV+RC_heavy ≤ 0.01`
   under a certified nonlinear instrument) before it is quotable as a pass.
2. **If not:** no RC; the nav token + spatial args remain the route input, and turn-direction cues come from the dense
   tactical labels.
3. **Either way, certify a nonlinear instrument first** (the cheapest open item): the trees' deranged control fails
   because ~171 autocorrelated windows per clip let them memorise clips. The next instrument: sub-sample one window
   per clip per 2 s and use 2,000+ train clips, then require |ρ_deranged| ≤ 0.01 before any reading.

### S2.7 E2′ / E3′ — the registered follow-up (`SPEC_ADDENDUM_S2A2.md`, sha256 `a44bd428…`, registered by the Master Mind 2026-10-04T12:59:38Z; `raw/s2_e2prime.json`)
Run after Stage 3 and the Stage-4 documents, as agreed. Master Mind ruling (provisional, to the PI): road-geometry
speed information is admissible in a route input; lane-level ego choice is not; the leak reference is the σ = 25 m
road-level route.

**Instrument — CERTIFIED before any reading, the registered one (no fallback needed):** all 4,369 train clips, one
window per clip per 2 s → 35,133 windows from 4,150 clips with every input valid; C1 deranged **−0.0045** (bar
\|ρ\| ≤ 0.01) · C2 target **0.997** (≥ 0.99) · O1 oracle **0.379** (≥ 0.20). Sub-sampling one window per 2 s across
4,000+ clips removed the clip memorisation that left Stage 2's trees uncertified.

| variant | ρ(NAV + HEAVY) | ρ(NAV + RC noised) | **E2′ Δ (bar ≤ 0.01)** | clean σ 8 m minus HEAVY (no bar) | OLS Δ (decides nothing) |
|---|---|---|---|---|---|
| **A50** | 0.2753 | 0.2404 | **−0.035 PASS** | +0.021 | +0.0004 |
| B | 0.2615 | 0.2351 | −0.026 PASS | +0.002 | +0.0096 |
| A30 | 0.2868 | 0.2327 | −0.054 PASS | +0.048 | +0.0020 |
| A80 | 0.2627 | 0.2401 | −0.023 PASS | +0.004 | −0.0011 |

ρ(NAV) on this population = 0.209. **The training noise is load-bearing:** the CLEAN A50 point carries +0.021 more
than the road-level route (it would FAIL the 0.01 bar); noised it carries 0.035 LESS. ⇒ the input must be fed with the
noise at training time (INTEGRATION §3.2).

**E3′ (clean checkpoint; both populations):**
| population | variant | straight-support p90 \|δ\| (bar ≤ 1.0 m) | LC-scale median (bar ≤ 0.5 m) · n | no-excursion median | curved-support p90 (no bar) |
|---|---|---|---|---|---|
| train (E2′ pop.) | **A50** | **0.318 PASS** | **0.137 PASS** · 3,697 | 0.051 | 5.48 |
| train | B / A30 / A80 | 0.312 / 0.349 / 0.288 PASS | 0.125 / 0.152 / 0.120 PASS | 0.048 / 0.053 / 0.047 | 6.55 / 5.81 / 4.80 |
| eval139 | **A50** | **0.340 PASS** | **0.157 PASS** · 2,214 | 0.051 | 5.52 |
| eval139 | B / A30 / A80 | 0.328 / 0.332 / 0.310 PASS | 0.147 / 0.141 / 0.142 PASS | 0.047 / 0.051 / 0.047 | 6.89 / 5.51 / 4.58 |

Curved support (descriptive, A50 train): within 10°-wide support-range bins the lane-change-scale windows sit at or
BELOW the no-excursion windows (10–30°: 0.38 vs 0.50 m; 30–60°: 0.47 vs 1.29 m; 60–90°: 0.78 vs 1.93 m), i.e. the
curved-support δ is the reference's corner-cutting, not lane choice.

**Decision (the registered rule): RC-A50 with the noised training input is CONFIRMED** (E2′ PASS and both E3′ bars PASS
on both populations); RC-B also passes, as the registered fallback. ⚠️ This rests on the Master Mind's provisional
ruling that road-geometry speed information is admissible; the PI's confirmation is pending. Optimism stamp unchanged:
the checkpoint is the ego's own future path, smoothed — optimistic on PhysicalAI by construction; privileged on NavSim.

---

## Stage 3 — build and validation (filled below as each split completes)

### S3.0 What was built (`raw/s3_validation_{eval139,train}.json`; releases listed in `LANDING_READY_WPA.txt ## WPA-S3`)
| split | clips | rows (one per camera frame) | trainer windows joined | npz md5 |
|---|---|---|---|---|
| train | 4,369 | 869,278 | 746,946 / 746,946 | `f63ece410b725febb8a5242cf2b01d3c` |
| eval139 | 139 | 27,664 | 23,772 / 23,772 | `6b5c7f207cffc3b7eb3cd527fd433599` |

Builder `stack/scripts/build_v9_labels.py` md5 `128846cb…`, v7 builder `s2_geom_emit_v7.py` `9a537ee2…`, base commit
`50efa52`, `--lc-labels off`. Sources and their md5s are in each `.manifest.json`. Lead pass from `join3d` (dev box;
838,130 train / 26,394 eval frames with agent data); SAM3 lane offsets: eval on the dev box, train on Thor (one
detached job under `cpu.lock`, 761 s, ended 2026-10-04 14:24:59 Berlin; no WP-A process left on Thor).

### S3.1 Validation against the bars registered in SPEC §9.V (the independent derivation `code/v9_independent.py` imports nothing from the builder)
| id | check | train | eval139 | verdict |
|---|---|---|---|---|
| V11 | join with the trainer's windows, \|Δnow\| | 746,946 / 746,946, 7e-15 s | 23,772 / 23,772, 0.0 s | **PASS** |
| V1 | windows with a lateral AND longitudinal label (single or partial) ≥ 95 % | **95.09 %** (710,256) [95.04, 95.14] | **94.57 %**; 95.58 % on the 136 clips the trainer scores tactically (3 G3-excluded clips out) | train **PASS** · eval **FAIL as registered** |
| V1 | every observed independent turn window labelled | 881 of 113,946 unlabelled (0.77 %) | 125 of 3,332 (3.75 %; incl. the reversing mask) | **FAIL as worded ("every")** |
| V2 | TURN side, variant b vs the independent tangent-heading derivation, recall / precision ≥ 0.95 | **0.980 / 0.988** (n 113,946) | 0.943 / 0.966 (n 3,332) | train **PASS** · eval **FAIL** |
| V2 | the same, excluding forward↔reverse cusp windows (post-hoc, S3.2) | 0.9885 / 0.9887 (31 cusp clips) | 0.9915 / 0.9816 (3 cusp clips) | PASS (post-hoc reading) |
| V2 | STOP recall / precision ≥ 0.95 (STOP-eligible windows) | **0.991 / 0.994** (n 28,169) | **0.999 / 0.998** (n 1,132) | **PASS** |
| V3 | turn t_start / t_end within ±0.5 s on ≥ 90 % | 0.985 / 0.989 | 0.950 / 0.983 | **PASS** |
| V3 | Δψ within ±3° · R_arc within ±10 % of the curvature-channel radius | 0.913 · 0.909 | 0.904 · 0.944 | **PASS** |
| V3 | t_stop ±0.3 s · d_stop ±1.0 m | 1.000 · 1.000 (median 0.04 s / 0.02 m) | 1.000 · 1.000 | **PASS** |
| V3 | nav d_next within ±2 m | 0.759 (n 395,938; median 0.14 m, p90 16.7 m; relative median 0.09 %) | 0.804 | **FAIL** — chord-arc vs speed-integral-arc drift over hundreds of metres; no relative tolerance was registered |
| V4 | analytic controls (circle → TURN R_arc / Δψ / t_start; constant deceleration → STOP t and d; straight; ramp; lead at 2 s; RC chord point; nav d_next; synthetic lane change) | `stack/tests/test_v9_labels_builder.py` | | **PASS** (20/20) |
| V5 | lane change vs the Alpamayo LC text: side-correct ≥ 0.80 on the 101 text clips; firing ≤ 0.10 on no-text clips | side-correct **14/101 = 0.139** (fires on 15, right on 14 = 0.933 when it fires); no-text firing 46/4,197 = **0.011** | 1 text clip | **FAIL** ⇒ LC ships never-positive (`--lc-labels off`), as pre-registered |
| V6 | one mutation per field must go RED (yaw sign, band shift, speed scale, mirrored map, lead removed, nav arc origin, RC y-flip, no module state) | tests | | **PASS** (all RED on the mutated input) |
| V7 | truncating full-band frames to 6 s changes ≤ 5 % of labels | — | lat-a 5.33 % · lat-b 7.27 % · lon 15.35 % (`raw/s3_validation_eval139_run1_h6.json`) | **FAIL** ⇒ `H_ABS_MIN = 8.0` (absence claims need the full band), as pre-registered |
| V8 | turn-commanded windows with no announced turn starting within 6 s ≤ 5 % (refcv7: 74.3 %) | (i) **22.2 %** (35,640 / 160,669) · (ii) excluding ≤ 30 m: 15.2 % · (iii) time token: 0.0 % | (i) 30.4 % · (ii) 24.2 % · (iii) 0.0 % | **FAIL as written** (the spatial token announces decelerating approaches early, D-WPA-4) |
| V8 | realised announced turns fed the matching token ≥ 90 % | **99.55 %** (125,029 / 125,591) | 99.34 % | **PASS** |
| V8 | realised turns the nav does not announce (curves, suppressed — L1's job) | 30,530 windows | 1,476 | reported |

### S3.2 What changed in the builder AFTER the first validation read (each disclosed; no bar moved)
1. **`H_ABS_MIN` 6.0 → 8.0 s** — the pre-registered consequence of V7's failure.
2. **Reversing mask (post-hoc).** V2's first eval read had 121 side inversions, all in 3 clips: 2 near-stationary
   G3-excluded clips and 1 stop-and-turn clip passing through forward↔reverse cusps (the log's WORLD-frame velocity
   points against the body heading; MEASURED). v9 has no reverse class and speed is unsigned, so ≥ 0.3 s of reversing in
   [NOW, NOW+8] now masks every action and geometry-goal cell (1,690 train / 146 eval windows). The validator gained
   D2's own stationary rule (band path < 5 m ⇒ no turn) and an independent cusp detector; all three V2 readings are in
   the JSON (`V2_turn_AS_FIRST_RUN_no_stationary_rule`: train 0.972 / 0.989, eval 0.912 / 0.968).
3. **All-masked-clip columns** — a builder bug caught by `test_reversing_window_is_masked` before it reached a release.
4. **`--lc-labels off`** — V5's pre-registered consequence; the SAM3 crossing stays as a diagnostic.

### S3.3 Class support (windows, clips) and goal coverage (`V10`, `V9`)
* **Lateral, variant a (default):** LANE_KEEP 549,452 (4,365 clips) · TURN_L 43,766 (772) · TURN_R 42,537 (797) ·
  LANE_CHANGE 0 (never positive) · partial {LK, LC_x} 74,868. Variant b: TURN_L 56,433 / TURN_R 56,566. eval139 (a):
  LK 18,211 · TURN_L 482 (14 clips) · TURN_R 1,491 (30 clips).
* **Longitudinal (train):** HOLD 8,269 · CREEP 8,867 · STOP 28,100 · FOLLOW 108,610 · DECELERATE 109,794 · ACCELERATE
  144,858 · KEEP 182,041 · partial {class, FOLLOW} 132,273. Every v9 class is populated (refcv7: 4 of 16 classes had
  zero windows, D1).
* **Goals, train (windows):** TURN_L 43,766 pos / 666,857 neg · STOP_POINT 28,100 / 659,303 · FOLLOW_LANE 549,452 /
  86,303 · YIELD_FOR_TURN_L/R 3,895 / 3,694 pos · TRAFFIC_LIGHT_REACT_RED 22,274 pos (12,669 by the stop-episode
  propagation on 184 clips; v8 had 14,022 in-band) · EVADE_IN_CORRIDOR 28,623 pos (v8: 9,414 — the NUDGE gate removed,
  D-WPA-6) · LANE_CHANGE_L/R 3,840 / 2,339 pos (VLM, VLM frames only) · SPEED_BAND never in the BCE. VLM tokens are
  labelled (positive or the PI's caption-absence negative) on 173,435 VLM-frame windows.
* **Against refcv7 (D1, the same 746,946 windows):** tactical labels on 23.22 % of windows → **95.09 %**; one anchor
  band per clip → per frame; nav "turn but no turn starts within 6 s" 74.3 % → 22.2 % (0.0 % for the time-rule
  diagnostic).

### S3.4 Stage 4 — the reader, the integration contract, the corpus manifest, the v7_labels fix
* Reader `stack/tanitad/data/v9_labels.py` (pure functions over an immutable, picklable `V9Release`; lookups refuse a
  clock mismatch > 1e-6 s); tests `stack/tests/test_v9_labels.py` (6/6, incl. a mutation that reintroduces a module
  global and must go RED). Smoke on the real eval release: 24 sampled windows join through `row_for_now` on the
  trainer clock; the pickle round-trip holds.
* `INTEGRATION.md` (tensor contract, dropouts, the NavSim mapping and the two reporting rows); `V9_SCHEMA.md` (all 132
  fields, checked mechanically by `code/s4_schema_check.py`, which goes RED on a mutated doc).
* Corpus manifest `raw/refcv8_corpus_manifest.json`: mask 693 ego-footprint boxes in 18 train clips (reproduces D3);
  drop list 8 (PI decision 8; Master Mind default keep + mask).
* v7_labels fix `code/fix/stack/tanitad/data/v7_labels.py` (`86f0c46e` → `abb1f64c`): the goal-negative policy travels
  on `V7Label` / `LabelManifest`; `stack/tests/test_v9_labels_v7_policy_isolation.py` passes 6/6 on the fix and goes RED
  on the unfixed tip module (the train label's LANE_CHANGE_L weight 0 → 1 after the eval load — D1 F2 exactly); the 23
  tip test files that import `v7_labels` pass against the fix (376 passed, 1 data-dependent skip).
