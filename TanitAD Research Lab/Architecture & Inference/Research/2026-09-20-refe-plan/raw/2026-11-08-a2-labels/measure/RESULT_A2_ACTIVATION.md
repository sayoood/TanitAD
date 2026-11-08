# A2 behaviour labels: real-data measurement (MEASURED, dev box, CPU)

Scenes: **62** nuPlan **trainval** states, seeded (seed 20261109), navtest-disjoint by BOTH gate keys. navtrain identity UNVERIFIED (the navtrain scene-filter file is not on this box). Fan: 64 seeded proposals per scene = 8 speed profiles x 8 lateral profiles around the logged future (NOT REFe's fan). Rates below are therefore term-SENSITIVITY checks and false-positive floors, not training-label rates. Constants: `behaviour_terms.C`, version behaviour/1.

## Identity (the brief's bit-identical requirement)

* `score_plans`: pre-A2 tip vs overlay OFF vs overlay ON (PDM keys), byte-equal on every scene: **True** (includes a deliberate-regression arm that must break equality, and a repeat run).
* through the real worker path `online_labeller._job`, separate processes, tip vs OFF vs ON labels, n=62 scenes: tip==OFF **True**, OFF==ON labels **True**, ON repeated in a 2nd process **True**, OFF returns no behaviour arrays **True**.
* `q_comf == 1  <=>  NAVSIM c == 1` for every proposal on every scene: **True**.

## Per-term activation (over all proposals; `applicable` = term evaluated; `fired` = q < 1)

| term | applicable (prop. rate) | fired (prop. rate) | scenes where q varies across the 64 | PDM-Closed row: applicable / fired | logged-human row: applicable / fired |
|---|---|---|---|---|---|
| q_free | 0.369 | 0.276 | 24/62 | 23/62 / 0/62 | 22/62 / 17/62 |
| q_gap | 0.452 | 0.236 | 22/62 | 28/62 / 17/62 | 28/62 / 12/62 |
| q_osc | 1.000 | 0.059 | 31/62 | 62/62 / 0/62 | 62/62 / 0/62 |
| q_stall | 1.000 | 0.067 | 3/62 | 62/62 / 4/62 | 62/62 / 4/62 |
| q_comf | 1.000 | 0.141 | 38/62 | 62/62 / 2/62 | 62/62 / 0/62 |

The last two columns are the FALSE-POSITIVE FLOOR: a term that fires on PDM-Closed or on the logged human drive is measuring something the reference does too. `discriminating` is the D-RL-COLL-SPARSE-1 check: a term that is constant across the 64 proposals of a scene gives the listwise loss no gradient in that scene.

## Comfort: the pilot's problem, checked

Same simulated states, two detectors (`pilot_style_comfort` = raw 10 Hz differences as in the pilot's gym_env.py:331-348, reimplemented; `comfort_violations` = NAVSIM's own filtered channels). Proposals with >= 1 violated channel:

| set | NAVSIM-filtered | pilot-style raw |
|---|---|---|
| all proposals (3968) | 560 | 2031 |
| PDM-Closed row (62 scenes) | 2 | 14 |
| logged-human row (62 scenes) | 0 | 2 |

Violations per channel, summed over proposals (lon_acc, lat_acc, mag_jerk, lon_jerk, yaw_acc, yaw_rate):  NAVSIM [548, 0, 0, 296, 0, 0]   pilot-style [814, 0, 1428, 1993, 0, 0]

## Cost

`cost_probe.py` (12 scenes, warm, 3 calls each, medians, one process, a box shared with other jobs): `score_plans` OFF 0.42 s, ON 0.64 s, so the extension adds **0.23 s per set (p90 0.27 s), +54 %**. The table-free timings inside `a2_real_results.json` are cold/contended and not quotable. The M-LAB budget was 0.34-0.83 CPU-s per set on the pod (INHERITED): `--label-workers` must be re-sized with behaviour ON, re-measured on the pod.

## Per scene

| # | loc | v0 | free steps | lead steps | q_free app/fired | q_gap app/fired | q_osc fired | q_stall fired | q_comf fired |
|---|---|---|---|---|---|---|---|---|---|
| 0 | las_vega | 10.0 | 1.7 | 0.4 | 0.00/0.00 | 0.25/0.12 | 0.12 | 0.00 | 0.38 |
| 1 | sg-one-n | 7.4 | 9.0 | 0.0 | 1.00/1.00 | 0.00/0.00 | 0.00 | 0.00 | 0.25 |
| 2 | las_vega | 4.5 | 2.6 | 6.4 | 0.00/0.00 | 1.00/0.88 | 0.00 | 0.00 | 0.00 |
| 3 | sg-one-n | 0.0 | 9.0 | 0.0 | 0.00/0.00 | 0.00/0.00 | 0.00 | 1.00 | 0.00 |
| 4 | las_vega | 2.5 | 1.0 | 8.0 | 0.00/0.00 | 1.00/1.00 | 0.00 | 0.00 | 0.00 |
| 5 | sg-one-n | 9.2 | 8.9 | 0.1 | 1.00/0.75 | 0.12/0.00 | 0.12 | 0.00 | 0.25 |
| 6 | us-pa-pi | 9.0 | 9.0 | 0.0 | 1.00/0.75 | 0.00/0.00 | 0.12 | 0.00 | 0.25 |
| 7 | las_vega | 10.0 | 9.0 | 0.0 | 1.00/0.38 | 0.00/0.00 | 0.00 | 0.00 | 0.38 |
| 8 | sg-one-n | 4.6 | 1.2 | 0.2 | 0.00/0.00 | 0.12/0.12 | 0.12 | 0.00 | 0.12 |
| 9 | sg-one-n | 4.9 | 9.0 | 0.0 | 1.00/1.00 | 0.00/0.00 | 0.11 | 0.00 | 0.12 |
| 10 | las_vega | 0.0 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 11 | las_vega | 5.4 | 1.0 | 8.0 | 0.00/0.00 | 1.00/1.00 | 0.00 | 0.00 | 0.00 |
| 12 | las_vega | 11.0 | 0.1 | 6.4 | 0.00/0.00 | 1.00/1.00 | 0.12 | 0.00 | 0.38 |
| 13 | us-ma-bo | 10.6 | 9.0 | 0.0 | 1.00/0.12 | 0.00/0.00 | 0.12 | 0.00 | 0.38 |
| 14 | las_vega | 12.2 | 1.4 | 0.2 | 0.00/0.00 | 0.12/0.12 | 0.12 | 0.00 | 0.38 |
| 15 | las_vega | 11.8 | 9.0 | 0.0 | 1.00/0.75 | 0.00/0.00 | 0.12 | 0.00 | 0.38 |
| 16 | las_vega | 6.0 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.50 | 0.00 | 0.00 | 0.12 |
| 17 | sg-one-n | 6.0 | 7.0 | 2.0 | 1.00/1.00 | 1.00/1.00 | 0.12 | 0.00 | 0.12 |
| 19 | us-pa-pi | 0.0 | 9.0 | 0.0 | 0.00/0.00 | 0.00/0.00 | 0.00 | 1.00 | 0.00 |
| 20 | sg-one-n | 5.4 | 9.0 | 0.0 | 1.00/1.00 | 0.00/0.00 | 0.12 | 0.00 | 0.12 |
| 21 | sg-one-n | 5.3 | 6.5 | 0.0 | 1.00/1.00 | 0.00/0.00 | 0.00 | 0.00 | 0.12 |
| 22 | us-ma-bo | 9.6 | 9.0 | 0.0 | 1.00/0.75 | 0.00/0.00 | 0.12 | 0.00 | 0.25 |
| 23 | las_vega | 3.4 | 9.0 | 0.0 | 1.00/1.00 | 0.00/0.00 | 0.00 | 0.12 | 0.00 |
| 24 | las_vega | 4.1 | 0.6 | 1.0 | 0.00/0.00 | 0.38/0.38 | 0.00 | 0.00 | 0.25 |
| 25 | las_vega | 2.3 | 0.4 | 8.6 | 0.00/0.00 | 1.00/1.00 | 0.00 | 0.00 | 0.00 |
| 26 | sg-one-n | 4.5 | 9.0 | 0.0 | 1.00/0.75 | 0.00/0.00 | 0.12 | 0.00 | 0.12 |
| 27 | us-ma-bo | 7.9 | 9.0 | 0.0 | 1.00/0.25 | 0.00/0.00 | 0.12 | 0.00 | 0.25 |
| 28 | us-pa-pi | 7.3 | 5.1 | 3.9 | 0.38/0.25 | 0.88/0.88 | 0.12 | 0.00 | 0.25 |
| 29 | las_vega | 0.0 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 30 | sg-one-n | 5.8 | 9.0 | 0.0 | 1.00/0.75 | 0.00/0.00 | 0.12 | 0.00 | 0.12 |
| 31 | las_vega | 5.8 | 2.5 | 0.0 | 0.00/0.00 | 0.00/0.00 | 0.08 | 0.00 | 0.12 |
| 32 | us-pa-pi | 7.4 | 9.0 | 0.0 | 1.00/1.00 | 0.00/0.00 | 0.00 | 0.00 | 0.25 |
| 33 | las_vega | 0.0 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 34 | las_vega | 5.4 | 1.1 | 6.9 | 0.00/0.00 | 1.00/0.88 | 0.12 | 0.00 | 0.12 |
| 35 | las_vega | 7.5 | 0.9 | 2.9 | 0.00/0.00 | 0.50/0.25 | 0.12 | 0.00 | 0.12 |
| 36 | sg-one-n | 1.6 | 9.0 | 0.0 | 0.00/0.00 | 0.00/0.00 | 0.00 | 0.88 | 0.00 |
| 37 | las_vega | 8.6 | 5.0 | 4.0 | 0.50/0.38 | 0.88/0.88 | 0.12 | 0.00 | 0.25 |
| 38 | us-ma-bo | 5.0 | 9.0 | 0.0 | 1.00/1.00 | 0.00/0.00 | 0.08 | 0.00 | 0.12 |
| 39 | sg-one-n | 8.7 | 8.0 | 1.0 | 1.00/0.88 | 0.50/0.25 | 0.12 | 0.00 | 0.25 |
| 40 | us-pa-pi | 12.3 | 9.0 | 0.0 | 1.00/0.38 | 0.00/0.00 | 0.12 | 0.00 | 0.38 |
| 41 | us-pa-pi | 0.0 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 42 | las_vega | 10.7 | 0.5 | 1.6 | 0.00/0.00 | 1.00/0.23 | 0.12 | 0.00 | 0.38 |
| 43 | las_vega | 0.0 | 9.0 | 0.0 | 0.00/0.00 | 0.00/0.00 | 0.00 | 1.00 | 0.00 |
| 44 | las_vega | 9.1 | 0.2 | 2.8 | 0.00/0.00 | 0.75/0.75 | 0.12 | 0.00 | 0.25 |
| 45 | us-pa-pi | 0.0 | 3.0 | 0.0 | 0.00/0.00 | 0.00/0.00 | 0.00 | 0.00 | 0.00 |
| 46 | us-pa-pi | 4.1 | 1.9 | 2.4 | 0.00/0.00 | 1.00/1.00 | 0.00 | 0.00 | 0.12 |
| 47 | las_vega | 0.7 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 48 | sg-one-n | 6.6 | 9.0 | 0.0 | 1.00/0.75 | 0.00/0.00 | 0.12 | 0.00 | 0.12 |
| 49 | us-pa-pi | 0.0 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 50 | las_vega | 7.3 | 8.6 | 0.0 | 1.00/0.25 | 0.00/0.00 | 0.12 | 0.00 | 0.12 |
| 51 | las_vega | 0.0 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 52 | las_vega | 0.0 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 53 | las_vega | 0.1 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 55 | las_vega | 0.0 | 0.0 | 0.0 | 0.00/0.00 | 0.00/0.00 | 0.00 | 0.00 | 0.00 |
| 56 | las_vega | 0.0 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 57 | sg-one-n | 2.7 | 9.0 | 0.0 | 1.00/1.00 | 0.00/0.00 | 0.02 | 0.12 | 0.00 |
| 58 | sg-one-n | 1.4 | 1.0 | 0.0 | 0.00/0.00 | 0.00/0.00 | 0.00 | 0.00 | 0.00 |
| 59 | las_vega | 0.0 | 0.0 | 9.0 | 0.00/0.00 | 1.00/0.00 | 0.00 | 0.00 | 0.00 |
| 60 | las_vega | 10.7 | 0.8 | 0.6 | 0.00/0.00 | 0.25/0.25 | 0.12 | 0.00 | 0.38 |
| 61 | las_vega | 12.1 | 1.2 | 0.5 | 0.00/0.00 | 0.25/0.25 | 0.12 | 0.00 | 0.38 |
| 62 | las_vega | 3.0 | 0.4 | 7.5 | 0.00/0.00 | 1.00/0.88 | 0.00 | 0.00 | 0.12 |
| 63 | us-ma-bo | 7.9 | 0.8 | 8.2 | 0.00/0.00 | 1.00/1.00 | 0.12 | 0.00 | 0.25 |
