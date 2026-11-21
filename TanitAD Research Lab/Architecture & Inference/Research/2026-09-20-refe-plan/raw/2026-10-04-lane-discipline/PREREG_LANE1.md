# LANE-1: lane discipline of REFe's final model, and a zero-training direction-aware selection (pre-registration)

> Written 2026-10-04 while `lane_census.py` runs, before any census output beyond its 6-token smoke test has been read.
> PI, after watching the video: *"The planner fan is not always following the lane and even if the fan is within the target
> lane, sometimes the selected trajectory is leaving the lane."* and *"we should check and validate the teacher itself, if it
> is considering drivable areas etc."*

## What is already MEASURED (before this census)

- **The benchmark cannot see it.** NAVSIM v1.1 sets `driving_direction_weight = 0.0` (`pdm_scorer.py:42`).
  - DAC counts any ROADBLOCK, INTERSECTION or CARPARK polygon as drivable, including the oncoming carriageway (`:251-258`).
  - There is no lane-keeping term.
- **Full navtest, harness CSV (`refe_navtest_final.csv`, pre-route-fix):** 527 of 12,146 picks have DDC < 1 (219 at 0.0, 308 at 0.5).
  - These picks average PDMS 58.0 and 68.1, against 84.1 for DDC = 1.
- **Scorer-label teacher, on the 3,137 held-out v5 sets (200,768 proposals; `teacher_label_census.json`):**
  - Drivable area is NAVSIM-faithful (`navsim_dac.violation`, 9.6 % violations), validated at 100.00 % agreement with NAVSIM on 25,600 proposals (`raw/2026-09-26-navsim-dac/`).
  - Driving direction comes from DriveRL's wrong-way categories 3/4 (2.66 %), not NAVSIM's oncoming-progress definition.
  - The solid-line crossing category (2) occurs **0 times in 200,768**.
  - No lane-keeping, lane-change or centre-line deviation leaf is stored.
  - ⇒ the scorer has never been shown a lane-keeping label.
- **Selection rule:** `navsim_v1` = NC·DAC·(5EP+5TTC+2C)/12. The scorer's DDC head is trained but never used to choose.
- **Trajectory teacher (DriveRL, `release/configs/driverl_teacher.yaml`):** its reward does carry `off_road -1.0`, `cross_lane 0.4`, `wrong_way 0.2` and `deviation_distance 0.75`, so the teacher whose rollouts are the proposal targets IS lane-aware by construction.

## Census (report-only, no pass/fail): `lane_census.py`, all 12,146 navtest tokens

Each of the 66 trajectories is scored by NAVSIM v1.1's own simulator and scorer on the token's metric cache: PDM-Closed, the 64 hypotheses and the human future.

**Control:** the reproduced pairwise PDMS of the pick must equal the harness CSV. Disagreements are counted and listed.

**Violations:**
- **V_onc:** NAVSIM DDC < 1, i.e. at least 2 m of centre progress outside every on-route lane within some 1 s window.
- **V_lane:** the footprint straddles a lane boundary for at least 1.0 s with the centre outside every intersection (`strad_s ≥ 1.0`).

The human's own rate of each is the base rate (legal lane changes and turns trip V_lane and V_onc too).

**Effect definitions**, on tokens where the human is clean of V:
- **FAN effect:** at least 50 % of the 64 hypotheses violate V.
- **SELECTION effect:** the pick violates V AND at least one hypothesis is clean of V with reproduced PDMS ≥ the pick's. A free clean alternative existed.

Reported with paired log-cluster bootstrap CIs (10,000 resamples, seed 20260927):
- on all tokens;
- on the **8,760 tokens the route fix leaves unchanged** (`routefix_changed_tokens.json`), where the table is the adopted model exactly.

**Head check:** the within-set AUC of the scorer's DDC head (sigmoid of logit 5) against NAVSIM V_onc, over sets that contain both classes.

## Lever L1 (zero training): choose argmax of navsim_v1 × p_DDC

- **Population for the decision:** the 8,760 route-fix-unchanged tokens.
- **Primary:** paired PDMS(L1) − PDMS(navsim_v1), from the reproduced harness score of each rule's pick.
- **ADOPT L1 iff BOTH:**
  - the PDMS lower bound > −0.10 (non-inferior);
  - the V_onc pick-rate difference has upper bound < 0.
- **Also reported:** V_lane, DAC, NC, EP, TTC and C rates of the picks; the same on all 12,146 tokens.
- **After an ADOPT read**, the L1 picks of the table are written as a seam and scored by the real W3 harness (CPU only), and the four families are reported. For the 3,386 route-fix-changed tokens, an L1 seam needs a model run and is queued behind the GPU lock.
- **L2 (hard gate: drop hypotheses with p_DDC < 0.5 unless none remain)** is reported only and is not adoptable from this read.

**If L1 fails, or V_lane is the larger effect:** the next lever is SFT-2. The pod labeller emits NAVSIM-faithful DDC and a lane-keeping label from the same map query `navsim_dac.py` already makes. The scorer is fine-tuned on them, and the selection rule multiplies them in.

The interval answers *another draw of logs* only: one checkpoint, deterministic inference.

## Departure (added 2026-10-04 ~09:30 local, after the first 1,180 census rows; the L1 rule and its bar are unchanged)

The registered lane proxy **V_lane failed its control**. NAVSIM's MULTIPLE_LANES footprint test, held for at least 1 s outside junctions, fires on:
- the **human 25.3 %**;
- **PDM-Closed 36.9 %**.

A 2.3 m-wide footprint in a ~3 m nuPlan lane polygon straddles under normal driving. V_lane is therefore reported as registered, but it is **not used to conclude anything**.

The distance to the metric cache's single route centreline was tried next and failed too: PDM-Closed reads more than 1 m on 15 % of drives even where the human keeps that lane.

The declared replacement is **`lane_census2.py`**. It measures distance from the vehicle centre to the centreline of the lane that contains it (nuPlan map), on non-junction lane steps:
- **lk10:** more than 1.0 m for at least 1.0 s;
- **lk15:** more than 1.5 m for at least 1.0 s.

These thresholds were fixed in the script before any hypothesis-level lk number was read, with the human and PDM-Closed as reported floors. It is analysed by `lane_analyze2.py` with the same effect definitions and estimator.

**ORACLE_mask_lk10** (drop violating hypotheses using the map) is an upper bound on what a lane-keeping label could buy, not a deployable rule.
