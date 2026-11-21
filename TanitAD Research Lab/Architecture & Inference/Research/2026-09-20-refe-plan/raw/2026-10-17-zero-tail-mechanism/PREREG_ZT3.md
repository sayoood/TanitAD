# PREREG ZT3 — the mechanism of REFe's DAC selection zeros, and which lever can reach it

**Written 2026-10-17 ~13:40Z, BEFORE any number below was computed.** Diagnostic agent for the Master Mind
(PI asked to start the PLAN-SELECTION lever). The sha256 of this file is in `PREREG_ZT3.md.sha256`, written
before the first analysis run. Any later change is an `Amendment n` appended at the end with its own time
and reason; the body is never edited.

**What was looked at before writing this (schema only, no outcome):** array keys/shapes of the C3 table,
P2's own table, the human seam, the tracked-states dump; one metric cache (a RANDOM-stratum token) for its
attribute names, polygon types and load time. The prior banked results quoted in the brief
(`D-REFE-ZEROTAIL-1/2`, `D-REFE-C3-SCORERBOUND-1`, `D-REFE-SELRULE-FIT`, `D-REFE-S1-READ`,
`H-REFE-AUXVETO-1`) were read as text. No per-token pick geometry, scene tag or scorer value of any zero
token was read.

## 0. Status, tier, estimator, admissibility

* **Diagnostic on navtest. It fits nothing, tunes nothing on navtest and decides no deployment.** Every
  threshold below is fixed here. A lever it suggests must be tested later on held-out data under its own
  pre-registration.
* **Tier T1**: NAVSIM v1.1 pseudo-simulation of one open-loop plan per token. All geometry is computed on
  the **LQR-tracked, kinematic-bicycle SIMULATED states** that NAVSIM's scorer judges (NAVSIM v1.1 source
  `navsim-3e8291bf…`, the harness's), plus the plan's own RAW interpolated states for the origin axis.
* **Estimator**: log-cluster bootstrap over the 136 navtest logs, 10,000 resamples, seed 20260927, using the
  landed `zero_tail.Boot` (imported, byte-identical copy, sha256 checked by the test). Shares are ratio
  estimators sum(indicator)/sum(1) over the target set; logs without target tokens contribute 0/0.
  **Every interval answers "another draw of LOGS" for this one checkpoint** — not another training run,
  and not another inference run (u400 and P2 select by deterministic argmax).
* **Admissibility (vision-only rule)**: the analysis uses map polygons, the human future and NAVSIM labels
  freely — they are ground truth for a diagnosis. Any LEVER proposed may use only cameras + nav command +
  ego status at inference.

## 1. Arms and target sets

| arm | role | source | rule |
|---|---|---|---|
| **A = u400** (SFT-4 arm B u400, deployed REFe) | **PRIMARY** | C3 table `raw/2026-10-11-c3-selection-headroom/table.npz` (1,058 tokens = all 529 u400 navtest zeros + 529 uniform non-zeroed, `tokens.json` stratum) | `navsim_v1_lane` |
| **B = P2** | **REPLICATION** | P2's own table `raw/2026-10-15-p2-zero-tail/p2_own_table/table.npz` (its 509 zeros) | `navsim_v1` |

**Target set T_X** (per arm): tokens with PDMS at the pick = 0 AND `zero_tail.cause_labels(NC_pick,
DAC_pick) == "DAC_only"` AND `zero_tail.sel_any(pdms) == True` (some proposal PDMS > 0). Classifier
imported, not reimplemented.

**Gates (stop and report if any fails):**
* G0 counts: A has exactly 529 zero tokens in the C3 table; B has 509 zero tokens and 445 SEL
  (`D-REFE-ZEROTAIL-2`).
* G1 instrument reproduction: the DAC recomputed by this package's per-step rule on simulated states equals
  the table's DAC for the pick P and the reference S on 100 % of T_A ∪ T_B.
* G2 identity: for arm A, this package's simulated P corners equal the banked tracked dump
  (`C:/Users/Admin/tmpauxv/tracked/`, `corners_sim_ego`) to ≤ 1e-3 m on every T_A token it covers.
* G3 control: the human path's DAC on every processed token; must read 1 on all (GAP-1: the human is never
  zeroed). A miss is reported with its tokens, it does not stop the run.

## 2. Reference trajectories per token

* **P** = the pick (`table['pick']`).
* **S** = the scorer's **best-ranked non-zero proposal**: argmax of the arm's selection score over
  proposals with PDMS > 0 (what a perfect zero-veto would pick). Selection score =
  `rl_val_eval.select` rule recomputed from `logits` (sigmoid; NC·DAC·(5EP+5TTC+2C)/12 [· p_lane]); the
  test asserts it reproduces `pick` on every table row.
* **H** = the human (`refe_p4_human_navtest_all.npz`, 8 poses at 0.5 s).
* Head order in `logits`: (NC, DAC, EP, TTC, C, DDC[, lane]).

## 3. Geometry definitions (all committed)

Per trajectory: NAVSIM interpolation → `PDMSimulator.simulate_proposals` → 41 states at 0.1 s; the 4 box
corners (FL, RL, RR, FR); **drivable polygons** = ROADBLOCK ∪ INTERSECTION ∪ DRIVABLE_AREA ∪ CARPARK_AREA
of `metric_cache.drivable_area_map`; a step is **non-drivable** iff fewer than 4 corners lie in a drivable
polygon (`pdm_scorer.py:284-286`).

* **t\*** = first non-drivable step (0.1 s units) of P's simulated states.
* **centre(t)** = mean of the 4 corners. **d\*** = |centre(t\*)| from the t0 rear-axle origin.
  **NEAR** iff d\* < 15 m, else **FAR**.
* **s\*_P** = arc length of P's centre path from step 0 to t\*. **L_X** = arc length of X's centre path over
  all 41 states.
* **side** at t\*: LEFT iff every out-corner is FL or RL; RIGHT iff every out-corner is FR or RR; else MIXED.
* **depth** = max over steps ≥ t\* of the max distance of an out-corner to the drivable union (shapely).
  **GRAZE** iff depth < 0.5 m.
* **Origin axis (own question Q6):** **TRACKER_MADE** iff P's RAW interpolated states pass the same rule
  (DAC = 1) while its SIMULATED states fail; **PLAN_INTRINSIC** iff the raw states already fail.
* **Δψ_X** = heading of X at the final state minus the initial heading (wrapped to (−180°, 180°]).
  **S_TURNS** iff |Δψ_S| ≥ 15°. **man(X)** = LEFT if Δψ_X ≥ +30°, RIGHT if ≤ −30°, else STRAIGHT.
* **e\*** = signed perpendicular offset of centre_P(t\*) from S's centre polyline (nearest segment; the last
  segment is extended beyond S's end), **positive = left of S's local heading**.
  **INSIDE** iff S_TURNS and sign(e\*) = sign(Δψ_S).
* **beyond** iff s\*_P > L_S.
* **lane membership**: LANE and LANE_CONNECTOR polygons of the same map containing a centre point.
  **ROUTE_DIFF** iff at some step k < t\* P's centre lies in ≥ 1 lane/connector polygon and NONE of those
  polygons is ever entered by S's centre (any step). **P_OFF_ROUTE** iff such a polygon is not in
  `metric_cache.route_lane_ids`.
* **junction (scene)**: **J** iff H's centre lies in an INTERSECTION polygon at any of its 41 states.
  **EXC_AT_JUNCTION** iff an out-corner of P at t\* is within 2 m of an INTERSECTION polygon.

### Pattern classes (Q3), mutually exclusive, first match wins

| # | class | definition |
|---|---|---|
| 0 | **S_STOP** | L_S < 1.0 m (the only non-zero option is a near-standstill) |
| 1 | **WRONG_MANOEUVRE** | L_P ≥ 5 m and L_S ≥ 5 m and man(P) ≠ man(S) |
| 2 | **CORNER_CUT** | S_TURNS and INSIDE and \|e\*\| ≥ 0.5 m |
| 3 | **WIDE_TURN** | S_TURNS and not INSIDE and \|e\*\| ≥ 0.5 m |
| 4 | **STRAIGHT_DRIFT** | not S_TURNS and \|e\*\| ≥ 0.5 m |
| 5 | **RUN_LONG** | \|e\*\| < 0.5 m and beyond |
| 6 | **ON_PATH** | \|e\*\| < 0.5 m and not beyond (P follows S within 0.5 m and still leaves) |

**Robustness, committed now:** the class shares are recomputed at |e\*| thresholds 0.25 m and 1.0 m and at
manoeuvre thresholds 20° / 45°; the report states whether the TOP class changes.

## 4. Questions

* **Q1 WHERE.** Shares of T_A by NEAR/FAR; t\* bins [0,1), [1,2), [2,3), [3,4] s; side; GRAZE; median s\*,
  d\*, depth.
* **Q2 SCENE.** Shares of T_A by command (navtest input export `driving_command` argmax: left/straight/right/
  unknown), J, v0 bin (`zero_tail.V0_BINS`), human-curvature bin (|Δψ_H| < 15°, 15–45°, ≥ 45°, from the seam's
  final heading). **Enrichment** = class share in T_A ÷ class share in navtest. Command, v0 and curvature
  base rates are exact over 12,146 tokens; J's base rate is estimated from the C3 strata (zero census
  529 + uniform random 529 of 11,617 non-zeroed, weight 11,617/529).
* **Q3 HOW.** Class shares (table above); descriptors L_P/L_S, e\*, Δψ_P − Δψ_S, ROUTE_DIFF, P_OFF_ROUTE,
  man(P) vs command, man(S) vs man(H).
* **Q4 DISCRIMINATION.** p_DAC = σ(logit_DAC). Medians of p_DAC(P), p_DAC(S); **DAC_INVERTED** iff
  p_DAC(P) ≥ p_DAC(S); rank of S in descending selection score (pick = 1); the term decomposition
  log s(P) − log s(S) = Δlog p_NC + Δlog p_DAC + Δlog W + Δlog p_lane (W = (5EP+5TTC+2C)/12) and the share
  of tokens whose largest positive term is each one.
* **Q5 LEVER COVERAGE.** Mechanical mapping committed here (Y = mechanism targets the pattern; P = partial;
  N = structurally cannot):

| lever | S_STOP | WRONG_MAN. | CORNER_CUT | WIDE_TURN | STRAIGHT_DRIFT | RUN_LONG | ON_PATH | TRACKER_MADE (axis) |
|---|---|---|---|---|---|---|---|---|
| P6 lane-rank twins (alternative-lane route intents) | N | Y | P only if ROUTE_DIFF | P only if ROUTE_DIFF | P only if ROUTE_DIFF | N | N | N |
| P6 trunk gradient (PDM DAC labels into the trunk) | P | Y | Y | Y | Y | Y | Y | P (needs simulated-state labels) |
| P7.1 map supervision (drivable raster aux head) | N | P | Y | Y | Y | Y | P (GRAZE below the head's 2 m cell) | N |
| proposal-side change (fan / trackability) | Y | P | P | P | P | P | P | Y |

  Reported per lever: **Y-share** (sum of Y cells) and **Y+P share** (upper bound), each with its CI.
* **Q6 (own) ORIGIN.** TRACKER_MADE share, and its cross-tab with the pattern classes.
* **Q7 (own) WITHIN-TOKEN DISCRIMINATION.** Within-token AUC of p_DAC separating DAC = 1 from DAC = 0
  proposals (Mann–Whitney, ties 0.5) on T_A tokens with a mixed DAC fan, vs the C3 random-stratum tokens
  with a mixed fan (comparison group). Paired log bootstrap of the difference of mean AUCs.

## 5. Committed readings (what each outcome means)

* **R1 (route vs geometry).** If **WRONG_MANOEUVRE ∪ ROUTE_DIFF ≥ 50 %** of T_A: lane-rank twins are a
  direct lever and the next experiment targets route choice. If the **same-route geometric classes
  (CORNER_CUT + WIDE_TURN + STRAIGHT_DRIFT + RUN_LONG + ON_PATH, ROUTE_DIFF = False) ≥ 50 %**: twins do not
  directly target the majority, and the lever is the scorer's boundary representation (P6 trunk DAC
  gradient / P7.1). Else MIXED.
* **R2 (origin).** TRACKER_MADE ≥ 25 % ⇒ an execution-aware lever earns its own arm; < 10 % ⇒ the plans
  themselves leave the road and the tracker is not the story.
* **R3 (head).** DAC_INVERTED ≥ 50 % ⇒ the DAC head is wrong on these tokens (representational lever);
  < 50 % ⇒ the head ranks S as more drivable but the product rule lets another term override it, and that
  term is named.
* **R4 (where).** NEAR ≥ 50 % ⇒ the boundary is in the camera's near field (a vision trunk can plausibly see
  it); FAR ≥ 50 % ⇒ anticipation problem.
* **R5 (within-token).** Mean within-token AUC on T_A ≤ 0.60 ⇒ the scorer is near-blind where it matters,
  whatever its pooled AUC.
* **Replication.** P2 "replicates" iff its top class equals A's and its share lies within ±15 points.
* **Next-experiment rule (committed):** the ONE next experiment targets the lever with the **largest
  Y-share** (Rule Zero 5), and is the cheapest test of it that (a) runs on held-out data, not navtest-fitted,
  (b) has both outcomes committed, (c) costs ≤ 1 dev-box GPU-day and 0 A40 unless the PI authorises more.

## 6. What is NOT done

No fitting, no threshold search, no pick change, no PDMS delta from a rule. No pod, no GPU, no commit.

---
## Amendment 1 (2026-10-17 ~13:55Z, written while the geometry dump runs, BEFORE any per-token class, share or
## scorer value of a target token was read; only G0 counts and 3 smoke-token DAC identities were seen)

1. **Q5 combination rule (was ambiguous).** A token's reach for a lever is the **class cell** if the token
   is PLAN_INTRINSIC, and the **TRACKER_MADE cell** if it is TRACKER_MADE. Sensitivity (reported, never
   deciding): the class cell for every token regardless of origin.
2. **Mutation control for the geometry pipeline:** `--mutate shift500` (all poses moved 500 m left, outside
   the metric cache's 100 m polygon radius) replaces the planned `shift50`, which could land on a parallel
   road and so is not a guaranteed-zero control. `--mutate empty_poly` stays.
3. **Target-set sizes now known (G0 run):** T_A = 343 tokens, T_B = 315 tokens. Population sizes only.

---
## Amendment 2 (2026-10-17 ~14:05Z, geometry dump still running; still no target-token class, share or scorer value read)

**PDMS at stake (added so levers are ranked by EFFECT, not only by count — Rule Zero 5).** For each class
and for each lever's reach set: Σ PDMS(S) over its T_X tokens ÷ 12,146 × 100 = the navtest PDMS points a
perfect zero-veto on those tokens would add while keeping the scorer's own ranking (a CEILING, not an
expected gain), with the same log bootstrap. Reported beside the count shares; the next-experiment rule in
§5 is applied to the PDMS-weighted Y-share, and the report states whether the count-weighted ranking agrees.
