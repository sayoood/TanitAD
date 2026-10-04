# LANE-1 result: the two effects in the video, measured; the teacher validated

> PREREG_LANE1.md, landed e983685 before any census output; the departure (lane proxy) is declared there.
> PI 2026-10-04: *"The planner fan is not always following the lane and even if the fan is within the target lane,
> sometimes the selected trajectory is leaving the lane."* and *"check and validate the teacher itself."*

## Answer

1. **Both effects are real.** Selection is the larger one in every measure.
   - The map view of the video drive (`scene_01949_02501_plans20-25_dir.png`, Singapore, left-hand traffic) shows the fan cutting the inside of a right-hand bend toward the oncoming lane while the human holds the lane. That is real geometry, not the video's flat-ground camera projection.
2. **Why it was never penalised:**
   - NAVSIM v1.1 gives driving direction a weight of 0 and counts the oncoming carriageway as drivable; it has no lane-keeping term.
   - The scorer's direction label is a different event, and it has no lane-keeping label.
   - The selection rule never uses the direction head.
   - The scorer ranks lane keeping at chance (AUC 0.56) and direction weakly (0.62).
3. **The zero-training fix (L1) fails its registered bar.** The trained fix (SFT-2, NAVSIM-faithful labels) is pre-registered and chained on the pod.
4. **The teachers:**
   - The scorer-label teacher is faithful on drivable area, mostly wrong on direction, and blind to lanes.
   - The trajectory teacher drives at human level on direction and lanes, but leaves NAVSIM's drivable area where the human never does.

## Census (MEASURED; `lane_result.json`, `lane2_result.json`)

- **Setup:** NAVSIM v1.1's own simulator and scorer on each token's metric cache, for PDM-Closed, the final model's 64 hypotheses and the human future. All 12,146 navtest tokens, 0 errors.
- **Control:** the reproduced pairwise PDMS of the pick equals the harness CSV on 12,146 of 12,146 tokens, and the argmax reproduces the stored pick on 12,146 of 12,146.
- **Primary population:** the 8,760 tokens the adopted route fix leaves unchanged (126 logs), where the table is the adopted model exactly.
- **Estimator:** log-cluster bootstrap, 10,000 resamples, seed 20260927.

| violation | human | PDM-Closed | fan (mean of 64) | **selected plan** |
|---|---|---|---|---|
| enters an oncoming / off-route lane (NAVSIM DDC < 1) | 1.08 % [0.64, 1.62] | 2.96 % | 3.51 % | **2.83 % [2.03, 3.70]** |
| off the lane centre > 1.0 m for ≥ 1 s (lk10) | 11.74 % [9.28, 14.38] | 21.61 % | 15.47 % | **15.72 % [12.82, 18.79]** |
| off the lane centre > 1.5 m for ≥ 1 s (lk15) | 6.23 % [4.38, 8.23] | 6.34 % | 7.47 % | **7.42 % [5.45, 9.58]** |
| leaves the drivable area (DAC) | 0.00 % | 0.13 % | 11.50 % | **5.02 % [3.86, 6.27]** |
| registered proxy V_lane (footprint straddles lanes ≥ 1 s) | 23.85 % | 34.89 % | 28.32 % | 30.34 %: **control failed, not used** |

**Fan or selection?** Measured on tokens where the human is clean.
- **Fan effect:** at least half of the 64 violate.
- **Selection effect:** the pick violates although a clean hypothesis with at least the same true PDMS existed.

| violation | human-clean tokens | fan effect | **selection effect** | violating picks with a free clean alternative |
|---|---|---|---|---|
| oncoming / off-route | 8,665 | 1.28 % [0.83, 1.83] | **1.94 % [1.28, 2.66]** | **91.8 % [88.1, 95.4]** |
| lk10 | 7,732 | 3.71 % [2.80, 4.70] | **6.39 % [5.20, 7.62]** | 88.1 % [84.6, 91.3] |
| lk15 | 8,214 | 1.25 % [0.79, 1.80] | **2.30 % [1.73, 2.96]** | 96.9 % [93.8, 99.5] |

**Can the scorer see it?** Within-set AUC over sets containing both classes:
- direction head vs oncoming: 0.62 (1,471 sets);
- planner aggregate vs oncoming: 0.60;
- planner aggregate vs lk10: 0.56 (3,182 sets).

**Selection rules on the same proposals** (PDMS reproduced exactly as the harness scores a submission):

| rule | PDMS | vs current [95 % CI] | oncoming picks | lk10 picks |
|---|---|---|---|---|
| current (navsim_v1) | 84.46 | – | 2.83 % | 15.72 % |
| **L1: × direction head (registered)** | 84.36 | **−0.10 [−0.26, +0.05]** | 2.49 % (Δ upper bound < 0) | 15.41 % (−0.31 pp [−0.53, −0.08]) |
| L2: gate direction head ≥ 0.5 (reported only) | 84.44 | −0.02 [−0.06, +0.02] | 2.83 % | – |
| ORACLE: drop oncoming hypotheses (map; upper bound) | 84.46 | +0.00 [−0.14, +0.15] | 0.55 % | 15.59 % |
| ORACLE: drop lk10 hypotheses (map; upper bound) | 84.32 | −0.14 [−0.43, +0.17] | 2.44 % | 7.16 % |
| ORACLE: best true PDMS | 98.33 | +13.87 [+12.64, +15.08] | 2.85 % | – |

**L1 verdict: NOT ADOPTED.** The oncoming-pick reduction clears its bar, but the PDMS lower bound −0.26 misses the registered −0.10. DAC is unchanged (+0.05 pp [−0.08, +0.19]).

The oracle rows say what perfect labels could buy: oncoming picks to 0.55 % at no PDMS cost, and lane-centre violations below the human floor. The scorer cannot reach that because it was never shown the event.

## The teachers (MEASURED)

### Scorer-label teacher, 3,137 held-out sets, 200,768 proposals

From `teacher_label_census.json` and `teacher_vs_navsim_ddc_heldout.json`:
- **Drivable area:** NAVSIM-faithful, 100.00 % agreement with NAVSIM on 25,600 proposals (`raw/2026-09-26-navsim-dac/`).
- **Driving direction:** DriveRL's wrong-way category flags 2.66 %, NAVSIM 3.96 %.
  - The teacher label catches **31.2 %** of NAVSIM's violations.
  - **46.4 %** of its flags are NAVSIM violations.
  - Binary agreement of 95.86 % hides this, because violations are rare.
- **Solid-line crossing** (category 2): 0 of 200,768. Lane keeping, lane change and centre-line deviation: no leaf stored.

### Trajectory teacher (DriveRL; its rollouts are the proposal targets)

From `teacher_check_heldout_summary.json`: 3,137 held-out samples, 24 logs; the teacher's rollout, the human's logged future and the model's 64, each on NAVSIM's own simulation.

| check | teacher | human | model, mean of 64 | teacher − human |
|---|---|---|---|---|
| leaves the drivable area | 4.97 % [1.30, 10.31] | 0.00 % | 24.07 % | **+4.97 pp [+1.30, +10.31]** |
| enters an oncoming lane | 2.14 % | 2.68 % | 8.14 % | −0.54 pp [−1.84, +0.63] |
| off the lane centre > 1 m, ≥ 1 s | 11.70 % | 10.93 % | 13.35 % | +0.77 pp [−1.63, +3.10] |

- 96 of the teacher's 156 drivable-area failures come from one log (`2021.10.05.04.03.05_veh-50_01466_01790`); the rest of the logs run at about 2 %.
- On those 156 samples the model's hypotheses fail drivable area 60 % of the time, against 24 % overall: the fan inherits.
- The teacher's reward does carry lane terms (`driverl_teacher.yaml`: off-road −1.0, cross-lane 0.4, wrong-way 0.2, centre deviation 0.75).

## Four families (this is a selection / geometry analysis, not a new driving arm)

- **Lateral:** lane keeping (lk10, lk15) and direction, above.
- **Strategic:** route compliance (on-route lanes after NAVSIM's correction = the DDC test), above.
- **Tactical:** selection quality (fan vs selection effect, scorer AUCs), above.
- **Longitudinal:** not measured here. No new arm was adopted, so no seam was scored. SFT-2's stage 2 will run `families6.py` on its seam.

## What runs next

**SFT-2** (`eval/PREREG_SFT2.md`, landed af63313):
- The scorer is fine-tuned with NAVSIM's own direction label on every training set: 190,696 sets relabelled on the pod, validated 100.00 % against NAVSIM on 13,000 navtest trajectories.
- It then selects with L1.
- Chained on the pod behind SFT-1: gate, then the deliberate regression must turn G8 RED, then the run.

**Lane keeping (lk10):**
- It enters the labels as a 7th scorer output if SFT-2 leaves the lane-centre selection effect. The oracle says a perfect label halves lane-centre picks.
- The labels already exist: `lane_keep` is written beside `navsim_ddc` by the relabeller.

**The fan:**
- Drop or replace the teacher targets that fail NAVSIM's drivable area (trajectory-head fine-tune).
- This needs the PI's go-ahead: it is a new training of the proposal head, outside the scorer-only fine-tunes.
