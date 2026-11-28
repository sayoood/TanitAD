# RESULT: P6-HN step 0 (0 GPU). Expert-proximate hard negatives exist in a third of navtrain held-out scenes at 0.5 m and in two thirds at most; the limit is hazard presence, and the failures sit on the cross-track axis the pre-registered ADE proximity under-counts

Date 2026-10-18. Pre-registration `PREREG_HN_CENSUS.md` (sha256 `5943731c1240d6a9a417ff09daa9e1b8db994a6356074c5590d05fde93301d4f`, re-verified at the end). CPU only, dev box, pod untouched, navtest table not opened.
Evidence class: **MEASURED (descriptive; nothing fitted)** for sections 1 to 4, **ESTIMATED** for section 5, **POST-HOC (not pre-registered)** for section 6. Tier: not applicable, no model is evaluated (PDM sub-score labels of plans on a navtrain DEV table); the four-family rule governs the P6-HN A/B, not this census.
Estimator for every interval: **log-cluster bootstrap over the 24 held-out logs, B = 10,000, seed 20260927**, answering *another draw of LOGS* only. One fan (u400's generator, ckpt 10075, NOT P2's and NOT P6's), one teacher rollout per key, no seeds: **nothing here is a lever effect.**

## 0. Verdict

* ⭐ **Supply.** 3,137 sets / 24 logs / 64 proposals. The fan is packed around the teacher plan: **99.4 % [99.0, 99.7]** of sets have a proposal within 0.5 m ADE (nearest-proposal median **0.089 m** [0.077, 0.104]; **23.4** [20.7, 26.1] of 64 within 0.5 m). Sets with **at least one expert-proximate hard negative (PDMS = 0)**: **33.7 % [23.3, 46.7]** at d = 0.5 m, **48.8 % [39.1, 60.3]** at 1 m, **60.5 % [51.5, 69.8]** at 2 m, **64.3 % [54.9, 73.2]** at 4 m. The ceiling is the share of sets with any hard negative in the 64 at all: **64.5 % [55.2, 73.5]**.
* ⭐ **Pre-registered reading (PREREG s6, anchor d* = 0.5 m, ADE, teacher): PARTIAL.** Point 0.337, and both CI ends (0.233, 0.467) sit inside the 0.10 to 0.50 band, so it does not straddle. The Lab's own twin-pair rule reads **PRESENT**: **34.7 % [21.6, 50.6]** of sets hold a DAC-0 / DAC-1 pair within 0.5 m mean distance. Against the logged **human** future (13 of 24 logs, 2,466 sets) the anchor reads **19.5 % [13.0, 26.9]**, also PARTIAL.
* ⭐ **Per set it is a heavy tail, not a floor.** At 0.5 m: 2,079 sets have 0 proximate hard negatives, 216 have 1, 154 have 2, 203 have 3 to 4, 485 have 5 or more (mean 2.09 [1.24, 3.22]). The pooled hard-negative rate among proximate proposals is **9.0 % [5.1, 14.9]** at 0.5 m and rises with distance to 17.0 / 23.5 / 34.8 / 41.6 % in the bands (0.5,1] / (1,2] / (2,4] / >4 m. **Only 19.0 % [13.3, 23.3] of ALL hard negatives lie within 0.5 m**, 49.8 % [39.6, 57.1] within 1 m, 81.2 % [73.3, 86.4] within 2 m. Hard negatives have median ADE **1.005 m** [0.871, 1.239] against 0.622 m [0.570, 0.681] for the rest; the nearest hard negative of a set that has one is at median **0.475 m** [0.362, 0.673] (2,024 of 3,137 sets have one).
* ⭐ **It is concentrated.** Half of the 1,058 sets with a proximate hard negative at 0.5 m sit in 4 of 24 logs; one log (`2021.10.05.04.03.05_veh-50_01466_01790`, 441 sets) alone holds 30.9 %. Per-log s_hn(0.5) runs from 0.00 to 0.80. Turns carry more: RIGHT **57.2 %** [45.7, 70.5] (278 sets, 12 logs), LEFT **38.5 %** [18.5, 58.8] (758, 18), STRAIGHT **29.0 %** [19.4, 39.9] (2,044, 24).
* ⭐ **The twin geometry is there wherever the hazard is.** 35.1 % [21.8, 51.2] of sets contain any DAC-0 proposal; of those sets **98.9 % [97.9, 99.7]** have a DAC-0 / DAC-1 twin pair within 0.5 m mean distance, and **77.3 % [71.6, 85.0]** of all DAC-0 proposals have a DAC-1 twin. So the supply limit is *whether the scene has a hazard*, not whether the fan can place a safe proposal beside it.
* ⭐ **Reach (ESTIMATED, gross).** A hard-negative term fed from the EXISTING fan reaches **0.29 [0.20, 0.40] to 0.32 [0.22, 0.44] PDMS** at d = 0.5 m (pre-registered axis), **0.42 to 0.46** at 1 m, against the Lab's unconstrained **0.86 to 0.94**. Arithmetic in section 5. On the post-hoc cross-track axis the same supply is **0.46 to 0.57**.
* ⚠️ **The axis matters, and the pre-registered one is the narrow one (POST-HOC).** The siblings' zero picks sit at median **1.6 to 1.9 m ADE** from the teacher (non-zero picks 0.7 m), so only **3 to 9 %** of them are "expert-proximate" at 0.5 m ADE. But the ADE gap is along-track lag (zero picks are 1.4 to 1.6 m BEHIND the teacher, path length 0.89 to 0.91 of it); the discriminating deviation is **cross-track: 0.39 to 0.42 m for zero picks against 0.13 to 0.14 m for non-zero picks**, and **62 to 69 %** of zero picks are within 0.5 m cross-track. With cross-track proximity the supply is **53.5 % [44.6, 61.6]** at 0.3 m and **60.2 % [50.5, 69.2]** at 0.5 m, i.e. it would read SUPPLY EXISTS. ⛔ This does not replace the committed reading; it says the P6-HN pre-registration must choose its proximity axis before the arm, and that "ADE <= 0.5 m" under-counts the lateral near-twins the Lab's mechanism is about.
* ⭐ **Does the data for hard-negative training exist on navtrain? Yes for mining, no for the offset bank.** Held-out: this 3,137-set table (PDM labels, proposals, teacher). Training-scale: **190,696 labelled on-policy training sets** and their queue (teacher plan per key) are archived on D: (section 7; the counts are INHERITED from the pod manifest and the archive was not opened here). The Lab's step-1 offset negatives (teacher +/- 0.15 to 0.60 m) **do not exist** and their yield cannot be read from the fan (section 8, N2).

## 1. What was measured, and what was missing

| item | value / status |
|---|---|
| table | navtrain held-out, 3,137 sets / 24 logs / 64 proposals x 20 poses (4 s); ranks 0 / 1 = 1,705 / 1,432 sets; all step 0, label_version 5, ckpt 10075 |
| proposals | `_tlane_work/ho/refe_heldout/sets/onpolicy_r*_hw*.jsonl` `traj` (float16 resolution) |
| labels | `s1_read_cache/labels/x/heldout_pdm/pdm_onpolicy_r*_hw*.jsonl` (nc, dac, ep, ttc, c, ddc, pdms) |
| expert (primary) | the banked **teacher plan** of the key, from the queue chunks (`teacher`, 20 x [x, y, yaw]); on navtrain this is the labels' reference candidate, **not** the logged human future |
| expert (secondary) | logged human future from the nuPlan DB for the **13 of 24 logs** whose DB is on disk (2,466 of 3,137 sets); the other 11 logs are not on this box (`raw/human_future_coverage.json`) |
| gates (before the hash) | 3,137 / 3,137 keys joined, 0 duplicate, 0 NaN; PDMS = 0 iff NC x DAC = 0 on all 200,768 proposals; the sets' own `navsim_dac.violation > 0` equals label `dac == 0` on **200,768 / 200,768** (permutation control 90.3 %, max 90.4 %) |
| gates (after) | G7 teacher-vs-human median ADE **0.82 m** (<= 5: PASS; q05/25/50/75/95 = 0.20 / 0.47 / 0.82 / 1.42 / 2.52); G8 class signs: LEFT median final y **+3.64 m**, RIGHT **-7.63 m** (PASS); K1 identity PASS; K2 wrong-expert control PASS (2.5 % vs 33.7 % at 0.5 m; nearest-proposal ADE median 3.07 m vs 0.089 m); unit tests 12 / 12; 6 / 6 deliberate-regression mutants turn their named test RED |
| **missing, named** | (a) the NAVSIM **driving command** for navtrain (only the `test` OpenScene metadata is on disk), so the by-command cut is the pre-registered proxy of expert-manoeuvre class; (b) the **teacher's own NAVSIM DAC**: `teacher_targets` carries only DriveRL's `dac.violation`, which flags 53 of the 19,284 PDM DAC-0 proposals and so cannot be used. The teacher's collision calculator flags 9 of 3,137 sets (0.38 % [0.00, 0.92] of the proximate-hard-negative sets at 0.5 m), so a collision-unsafe expert does not inflate the NC side; **whether the teacher itself leaves the drivable area is UNVERIFIED** (ZT3 measured human DAC = 1 on 1,224 / 1,224 navtest tokens, a human fact, not a teacher fact); (c) a human future for 11 of 24 logs |

## 2. Pre-registered results

Nothing below was tuned; the grid {0.5, 1, 2, 4} m and the definitions are those of the hash. All ADE unless marked FDE. Percentages carry the log-cluster interval.

### T0 base rates (all 64 x 3,137 proposals)

| quantity | value |
| --- | --- |
| hard-negative (PDMS = 0) rate, all proposals | 17.25 [11.86, 23.64] % |
| sets with >= 1 hard negative anywhere in the 64 | 64.5 [55.2, 73.5] % |
| sets with >= 1 DAC-0 proposal | 35.1 [21.8, 51.2] % |
| counts: HN / DAC-0 / NC-0 only / both | 34,629 / 19,284 / 15,345 / 1,224 |
| per-set HN rate q05/q25/q50/q75/q95 (point) | 0.000 / 0.000 / 0.062 / 0.266 / 0.688 |

### T1 SUPPLY, proximity = ADE to the teacher plan (PRIMARY)

| d (m) | sets with a proximate proposal % | **sets with >= 1 proximate HN %** | mean proximate / set | mean proximate HN / set | HN rate among proximate % | share of ALL HN that are proximate % |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 99.4 [99.0, 99.7] | **33.7 [23.3, 46.7]** | 23.4 [20.7, 26.1] | 2.09 [1.24, 3.22] | 9.0 [5.1, 14.9] | 19.0 [13.3, 23.3] |
| 1.0 | 100.0 [100.0, 100.0] | **48.8 [39.1, 60.3]** | 43.4 [40.8, 45.7] | 5.49 [3.45, 8.24] | 12.7 [7.7, 19.5] | 49.8 [39.6, 57.1] |
| 2.0 | 100.0 [100.0, 100.0] | **60.5 [51.5, 69.8]** | 58.1 [56.6, 59.4] | 8.96 [6.13, 12.56] | 15.4 [10.4, 21.9] | 81.2 [73.3, 86.4] |
| 4.0 | 100.0 [100.0, 100.0] | **64.3 [54.9, 73.2]** | 63.6 [63.3, 63.7] | 10.86 [7.47, 14.86] | 17.1 [11.7, 23.4] | 98.3 [96.9, 99.3] |

### T1b SUPPLY, proximity = FDE (sensitivity)

| d (m) | sets with a proximate proposal % | **sets with >= 1 proximate HN %** | mean proximate / set | mean proximate HN / set | HN rate among proximate % | share of ALL HN that are proximate % |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 96.0 [94.6, 97.2] | **13.0 [6.9, 21.3]** | 6.8 [5.4, 8.1] | 0.31 [0.15, 0.52] | 4.6 [2.1, 9.1] | 2.8 [1.5, 4.0] |
| 1.0 | 99.2 [98.9, 99.7] | **26.8 [17.2, 39.0]** | 16.8 [14.4, 19.3] | 1.25 [0.69, 1.97] | 7.4 [4.0, 13.0] | 11.3 [7.3, 14.5] |
| 2.0 | 99.9 [99.8, 100.0] | **41.0 [30.4, 54.2]** | 34.3 [31.5, 37.2] | 3.84 [2.28, 5.97] | 11.2 [6.4, 18.1] | 34.8 [25.6, 41.7] |
| 4.0 | 100.0 [100.0, 100.0] | **54.9 [46.1, 65.1]** | 51.9 [49.6, 54.0] | 7.33 [4.77, 10.67] | 14.1 [9.0, 20.9] | 66.4 [56.8, 73.4] |

### T1c sets with >= 1 proximate HN by sub-type (ADE), % [95 % CI over logs]

| d (m) | DAC-only | NC-only | both |
| --- | --- | --- | --- |
| 0.5 | 24.8 [14.4, 38.5] | 9.2 [5.9, 12.7] | 0.6 [0.1, 1.3] |
| 1.0 | 29.9 [18.3, 44.9] | 20.7 [14.2, 27.0] | 1.8 [0.9, 2.9] |
| 2.0 | 31.9 [20.0, 47.1] | 31.6 [22.0, 40.5] | 3.5 [1.8, 5.3] |
| 4.0 | 32.5 [20.3, 47.7] | 36.4 [25.6, 46.9] | 5.7 [2.8, 9.1] |

### T1d histogram of proximate-HN count per set (ADE; point; 3,137 sets)

| d (m) | 0 | 1 | 2 | 3-4 | 5+ |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 2079 | 216 | 154 | 203 | 485 |
| 1.0 | 1605 | 182 | 147 | 234 | 969 |
| 2.0 | 1238 | 135 | 140 | 262 | 1362 |
| 4.0 | 1120 | 136 | 126 | 251 | 1504 |

### T2 hard-negative rate by distance band, % [95 % CI over logs]

| band (m) | ADE: HN rate | ADE: n proposals | ADE: n HN | FDE: HN rate | FDE: n proposals |
| --- | --- | --- | --- | --- | --- |
| [0,0.5] | 9.0 [5.1, 14.9] | 73,368 | 6,571 | 4.6 [2.1, 9.1] | 21,310 |
| (0.5,1.0] | 17.0 [10.8, 24.3] | 62,659 | 10,660 | 9.3 [5.3, 15.4] | 31,548 |
| (1.0,2.0] | 23.5 [18.0, 28.9] | 46,246 | 10,873 | 14.8 [8.7, 22.5] | 54,743 |
| (2.0,4.0] | 34.8 [24.8, 42.8] | 17,117 | 5,952 | 19.9 [14.1, 26.1] | 55,073 |
| (4.0,inf] | 41.6 [22.1, 53.5] | 1,378 | 573 | 30.6 [23.7, 36.0] | 38,094 |

### T3 distance quantiles (ADE, m) [95 % CI over logs; exact 0.01 m histograms]

| population | n | q05 | q25 | q50 | q75 | q95 |
| --- | --- | --- | --- | --- | --- | --- |
| all proposals | 200,768 | 0.136 [0.121, 0.157] | 0.366 [0.328, 0.411] | 0.678 [0.616, 0.744] | 1.193 [1.092, 1.308] | 2.506 [2.261, 2.759] |
| nearest proposal per set | 3,137 | 0.031 [0.027, 0.037] | 0.058 [0.050, 0.070] | 0.089 [0.077, 0.104] | 0.136 [0.117, 0.155] | 0.253 [0.202, 0.301] |
| nearest HN per set (sets with an HN anywhere) | 2,024 | 0.100 [0.083, 0.136] | 0.249 [0.201, 0.328] | 0.475 [0.362, 0.673] | 0.978 [0.692, 1.239] | 2.197 [1.827, 2.811] |
| hard negatives, pooled | 34,629 | 0.264 [0.234, 0.322] | 0.588 [0.521, 0.720] | 1.005 [0.871, 1.239] | 1.717 [1.468, 2.074] | 3.187 [2.760, 3.616] |
| non-hard-negatives, pooled | 166,139 | 0.126 [0.113, 0.143] | 0.335 [0.304, 0.373] | 0.622 [0.570, 0.681] | 1.090 [1.003, 1.183] | 2.280 [2.095, 2.471] |

Sets with a hard negative anywhere in the 64: 64.5 [55.2, 73.5] %.  FDE: all proposals median 1.84 m, nearest-proposal median 0.134 m.

### T4 strata (expert-manoeuvre class from the teacher plan; teacher rank): sets with >= 1 proximate HN, ADE

| stratum | sets | logs | CI valid | HN rate % | d=0.5 % | d=1 % | d=2 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STATIONARY | 57 | 11 | point only | 19.5 [7.5, 26.7] | 26.3 [6.1, 38.9] | 68.4 [37.1, 90.0] | 75.4 [38.7, 100.0] |
| LEFT | 758 | 18 | yes | 18.2 [7.5, 30.3] | 38.5 [18.5, 58.8] | 49.2 [27.9, 70.5] | 54.7 [34.8, 74.6] |
| STRAIGHT | 2044 | 24 | yes | 15.5 [11.3, 20.3] | 29.0 [19.4, 39.9] | 44.5 [36.1, 54.0] | 59.2 [51.1, 66.7] |
| RIGHT | 278 | 12 | yes | 26.6 [17.4, 35.7] | 57.2 [45.7, 70.5] | 75.5 [65.9, 84.1] | 82.7 [74.9, 89.6] |
| rank0 | 1705 | 24 | yes | 16.8 [11.1, 23.4] | 33.8 [23.7, 46.1] | 47.0 [37.3, 58.9] | 59.4 [50.1, 68.8] |
| rank1 | 1432 | 24 | yes | 17.8 [12.6, 23.9] | 33.6 [22.3, 47.5] | 51.0 [40.4, 62.4] | 61.9 [52.4, 71.2] |

### T5 per-log (point; sorted by s_hn(0.5))

| log | sets | HN rate | s_hn 0.5 | s_hn 1 | s_hn 2 | s_hn 4 |
| --- | --- | --- | --- | --- | --- | --- |
| 2021.08.24.20.03.01_veh-45_00687_00787 | 76 | 0.410 | 0.80 | 0.87 | 0.92 | 0.92 |
| 2021.10.05.04.03.05_veh-50_01466_01790 | 441 | 0.357 | 0.74 | 0.84 | 0.88 | 0.88 |
| 2021.06.14.18.13.35_veh-26_01537_01717 | 50 | 0.330 | 0.54 | 0.90 | 0.96 | 1.00 |
| 2021.08.17.13.10.50_veh-08_00122_00295 | 57 | 0.289 | 0.49 | 0.65 | 0.67 | 0.67 |
| 2021.10.01.15.32.11_veh-28_00025_00097 | 38 | 0.181 | 0.45 | 0.55 | 0.58 | 0.76 |
| 2021.06.23.15.56.12_veh-16_00839_01285 | 237 | 0.134 | 0.42 | 0.55 | 0.66 | 0.70 |
| 2021.08.24.20.03.01_veh-45_00269_00428 | 84 | 0.222 | 0.35 | 0.52 | 0.62 | 0.70 |
| 2021.08.17.14.45.12_veh-42_01866_01999 | 132 | 0.091 | 0.33 | 0.34 | 0.35 | 0.35 |
| 2021.08.31.18.15.54_veh-40_00579_00980 | 259 | 0.261 | 0.32 | 0.54 | 0.72 | 0.81 |
| 2021.06.14.18.13.35_veh-26_04547_04710 | 155 | 0.072 | 0.28 | 0.33 | 0.36 | 0.37 |
| 2021.06.14.14.03.45_veh-38_01624_01811 | 22 | 0.223 | 0.27 | 0.64 | 0.64 | 0.64 |
| 2021.06.14.19.22.11_veh-38_02275_02455 | 176 | 0.067 | 0.24 | 0.35 | 0.45 | 0.45 |
| 2021.07.09.20.59.12_veh-38_03292_04331 | 438 | 0.122 | 0.23 | 0.43 | 0.61 | 0.65 |
| 2021.06.14.17.26.26_veh-38_05455_05749 | 211 | 0.066 | 0.23 | 0.29 | 0.44 | 0.48 |
| 2021.06.14.14.25.15_veh-26_04417_04531 | 24 | 0.026 | 0.21 | 0.42 | 0.46 | 0.54 |
| 2021.07.09.20.26.06_veh-35_02793_03289 | 304 | 0.138 | 0.20 | 0.37 | 0.52 | 0.53 |
| 2021.09.15.14.00.15_veh-28_01953_02255 | 134 | 0.123 | 0.15 | 0.32 | 0.40 | 0.41 |
| 2021.06.14.16.32.09_veh-35_04516_04698 | 32 | 0.074 | 0.12 | 0.38 | 0.62 | 0.66 |
| 2021.06.14.18.42.45_veh-12_01345_01523 | 20 | 0.114 | 0.10 | 0.50 | 0.50 | 0.50 |
| 2021.06.09.18.23.43_veh-35_03110_03179 | 67 | 0.104 | 0.06 | 0.31 | 0.57 | 0.78 |
| 2021.06.23.14.54.32_veh-16_00862_01000 | 36 | 0.084 | 0.06 | 0.31 | 0.44 | 0.53 |
| 2021.06.14.18.42.45_veh-12_04233_04472 | 83 | 0.079 | 0.05 | 0.29 | 0.58 | 0.65 |
| 2021.06.09.14.50.36_veh-26_00832_00905 | 21 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2021.08.31.12.21.30_veh-40_00661_00762 | 40 | 0.190 | 0.00 | 0.28 | 0.80 | 0.95 |

### T6 Lab step-0 twin pairs inside a set (mean pairwise L2 <= t), % or count [95 % CI over logs]

| t (m) | sets with >= 1 DAC0/DAC1 twin pair % | mean such pairs / set | DAC-0 proposals that have a DAC-1 twin % | sets WITH a DAC-0 that have a twin pair % | sets with >= 1 PDMS0/PDMS>0 pair % |
| --- | --- | --- | --- | --- | --- |
| t=0.3 | 33.1 [20.4, 48.7] | 12.3 [7.5, 17.5] | 53.1 [46.8, 63.8] | 94.2 [89.5, 96.9] | 58.9 [49.6, 68.6] |
| t=0.5 | 34.7 [21.6, 50.6] | 38.1 [22.5, 57.7] | 77.3 [71.6, 85.0] | 98.9 [97.9, 99.7] | 63.1 [54.2, 72.0] |

### T7 SECONDARY expert = logged human future (2,466 of 3,137 sets, 13 of 24 logs); gate G7 median ADE(teacher, human) = 0.820 m (<= 5: PASS)

| d (m) | human: sets with >= 1 proximate HN % | teacher, SAME sets: same quantity % | human: HN rate among proximate % | teacher same sets % |
| --- | --- | --- | --- | --- |
| 0.5 | 19.5 [13.0, 26.9] | 37.7 [25.4, 53.6] | 5.4 [2.9, 10.0] | 10.4 [5.7, 18.3] |
| 1.0 | 38.6 [28.1, 50.3] | 51.5 [39.8, 65.5] | 11.0 [5.7, 19.2] | 14.5 [8.4, 23.0] |
| 2.0 | 58.4 [47.5, 69.7] | 63.2 [52.3, 74.2] | 16.1 [9.7, 24.4] | 17.0 [10.9, 25.1] |
| 4.0 | 66.4 [55.3, 77.1] | 66.5 [55.3, 77.2] | 18.8 [12.1, 26.6] | 18.7 [12.2, 26.5] |

Nearest-proposal ADE median: human 0.223 m, teacher (same sets) 0.090 m.  teacher-vs-human ADE quantiles q05/25/50/75/95: 0.20 / 0.47 / 0.82 / 1.42 / 2.52 m.


## 3. Readings committed in advance (PREREG s6), applied

| reading | value | result |
|---|---|---|
| anchor s_hn(0.5), ADE, teacher | 0.337 [0.233, 0.467] | **PARTIAL** (band 0.10 to 0.50); point, lower and upper end all PARTIAL, no straddle |
| Lab twin-pair rule, t = 0.5 | 0.347 [0.216, 0.506] of sets | **PRESENT** (>= 0.10) |
| same anchor, FDE proximity | 0.130 [0.069, 0.213] | PARTIAL by the point; the lower end 0.069 would read SPARSE, so this one **STRADDLES** |
| same anchor, human expert (13 logs) | 0.195 [0.130, 0.269] | PARTIAL |
| monotone in d | 0.337 / 0.488 / 0.605 / 0.643 | yes, for ADE and for FDE |

**Consequence the table committed:** mine what exists AND build the offset bank for the scenes with none. Three things qualify it honestly: (i) it is one fan on 24 logs and one log supplies 31 % of the sets that carry the signal (leave-that-log-out, point only: (1,058 - 327) / (3,137 - 441) = 27.1 %); (ii) the FDE proximity (end-point match) is much stricter (13.0 %) because the fan's end points scatter 1 to 2 m along-track; (iii) the axis question of section 6.

**What the census adds to the Lab's picture.** The hard-negative rate near the teacher is low (9 % within 0.5 m) and climbs steadily with distance; 81 % of all hard negatives lie beyond 0.5 m. A term that sees only "within 0.5 m" sees a fifth of the negatives and, per scene, a heavy-tailed minority of scenes.

## 4. Controls and tests (PREREG s5)

K1 identity (expert := proposal 0): nearest ADE 0.0 on all sets, a proximate proposal at every d: PASS. K2 wrong expert (the teacher of another set of the same log): the proximate-hard-negative share falls from 33.7 % to 2.5 % [1.7, 3.4] at 0.5 m and from 48.8 % to 7.5 % at 1 m, and the nearest-proposal ADE median rises from 0.089 m to 3.07 m: the census reads the expert, not fan geometry. K3 and K4: `raw/test_hn_census.log` 12 / 12 PASS; `raw/mutants.json` 6 / 6 mutants RED on their named test (strict `<`, ADE over pose 0 only, hard negative = DAC-0 only, rank dropped from the join key, open low band edge, twin pair with the wrong label).

### K2 control: wrong expert (teacher of another set of the same log)

| d (m) | true expert: sets with proximate HN % | wrong expert % |
| --- | --- | --- |
| 0.5 | 33.7 [23.3, 46.7] | 2.5 [1.7, 3.4] |
| 1.0 | 48.8 [39.1, 60.3] | 7.5 [6.0, 9.1] |
| 2.0 | 60.5 [51.5, 69.8] | 16.7 [13.8, 19.4] |
| 4.0 | 64.3 [54.9, 73.2] | 30.2 [26.2, 34.0] |

Nearest-proposal ADE median: true 0.089 m, wrong 3.073 m.  K1 identity: nearest ADE max 0.0, proximate at every d: True.


## 5. Reach (ESTIMATED; arithmetic, not a measurement)

Inputs: kappa = 0.953 PDMS per pp of zero rate (= 4.15 / 4.36; MEASURED inputs, `D-REFE-GAP-CI`, carried in LAB-RUN-019 s4.3; **INHERITED**, not re-measured), Z = 4.36 pp (navtest u400, **INHERITED**), published relative cuts r = 0.208 (DA-WAM Tab. 3) and 0.227 (BeyondDrive Tab. 4) (**PUBLISHED, taken from LAB-RUN-019, not re-read against the PDFs here**). kappa x Z = 4.155 PDMS. The dev table's fan and the sibling scorers are transplanted into navtest units through these two constants: that is the ESTIMATED step. **Gross, not net** (`D-REFE-ZEROTAIL-4`: a gross reach can net about 0). **Levers on the same blind DAC head overlap** and do not add.

**C1, supply-limited reach = s_hn(d) x r x Z x kappa.** ASSUMPTION (HYPOTHESIS): the published methods served a hard negative to every scene and the effect scales linearly with the share of scenes served. At d = 0.5 m: 0.3373 x 0.208 x 4.155 = **0.29**, and 0.3373 x 0.227 x 4.155 = **0.32** (CI endpoints of s_hn: 0.20 to 0.44). Scaling the Lab's incremental-over-P6 range 0.55 to 0.94 by the same share gives **0.19 to 0.32** (0.55 x 0.337, 0.94 x 0.337).

**C2, coverage ceilings from the zero picks (three SFT-4 sibling scorers, rule navsim_v1, each read separately, NOT P2's scorer).** Zero-pick rates on this table are 3.28 % [1.35, 6.12] / 3.51 % [1.44, 6.32] / 3.16 % [1.32, 5.55] (n = 103 / 110 / 99). Ceiling = (share of zero picks that are themselves within d of the teacher) x 4.155. A ceiling, not an expectation.

* **within 0.5 m ADE:** 8.7 / 2.7 / 6.1 % of the zero picks, ceiling **0.36 / 0.11 / 0.25 PDMS**
* **within 1 m:** 19.4 / 13.6 / 25.3 %, ceiling **0.81 / 0.57 / 1.05**
* **within 2 m:** 62.1 / 56.4 / 61.6 %, ceiling **2.58 / 2.34 / 2.56**
* the other reading of coverage, "the zero pick's scene contains a proximate hard negative": 81.6 / 82.7 / 83.8 % at 0.5 m. That is **not** a reach (it would imply about 3.4 PDMS): it says the supply is concentrated exactly in the scenes where the scorer fails, which is the favourable half of the picture.

**Reading.** Fed from the existing fan at the Lab's offset scale, the term can act on about a third of scenes and touches **0.3 PDMS (ADE axis)**, about a third of the Lab's standalone number. The zero picks themselves are mostly NOT within 0.5 to 1 m ADE of the teacher, so the part of the Lab's 0.86 to 0.94 that depends on the term *generalising outward* from near-expert negatives to the scorer's actual failures is not supported by the ADE census; it is the unmeasured part of the estimate. Section 6 shows the lateral axis is more favourable.

Tables (generated, `raw/tables.md`):

### T8 reach C1 (ESTIMATED): supply-limited = s_hn(d) * r * Z * kappa, PDMS points, [CI endpoints of s_hn only]

| d (m) | r = 0.208 (DA-WAM) | r = 0.227 (BeyondDrive) |
| --- | --- | --- |
| 0.5 | 0.29 [0.20, 0.40] | 0.32 [0.22, 0.44] |
| 1.0 | 0.42 [0.34, 0.52] | 0.46 [0.37, 0.57] |
| 2.0 | 0.52 [0.45, 0.60] | 0.57 [0.49, 0.66] |
| 4.0 | 0.56 [0.47, 0.63] | 0.61 [0.52, 0.69] |

Unconstrained standalone (s = 1, the Lab's number): 0.86 / 0.94 PDMS.

### T9 reach C2 (ESTIMATED ceilings): where the sibling scorers' zero picks sit, each scorer read separately

| scorer | zero picks (n, rate %) | d (m) | zero picks within d of the teacher % | ceiling = coverage*Z*kappa (PDMS) | zero-pick sets that contain a proximate HN % |
| --- | --- | --- | --- | --- | --- |
| N-s0 | 103, 3.28 [1.35, 6.12] | 0.5 | 8.7 [3.6, 17.2] | 0.36 [0.15, 0.72] | 81.6 [60.0, 90.8] |
|  |  | 1.0 | 19.4 [9.6, 35.2] | 0.81 [0.40, 1.46] | 94.2 [86.4, 98.0] |
|  |  | 2.0 | 62.1 [50.5, 75.4] | 2.58 [2.10, 3.13] | 98.1 [91.3, 100.0] |
|  |  | 4.0 | 100.0 [100.0, 100.0] | 4.16 [4.16, 4.16] | 100.0 [100.0, 100.0] |
| N-s1 | 110, 3.51 [1.44, 6.32] | 0.5 | 2.7 [0.0, 7.0] | 0.11 [0.00, 0.29] | 82.7 [63.2, 90.3] |
|  |  | 1.0 | 13.6 [6.7, 26.2] | 0.57 [0.28, 1.09] | 94.5 [83.8, 99.1] |
|  |  | 2.0 | 56.4 [45.5, 70.0] | 2.34 [1.89, 2.91] | 97.3 [88.2, 100.0] |
|  |  | 4.0 | 99.1 [97.7, 100.0] | 4.12 [4.06, 4.16] | 100.0 [100.0, 100.0] |
| N-s2 | 99, 3.16 [1.32, 5.55] | 0.5 | 6.1 [0.0, 12.3] | 0.25 [0.00, 0.51] | 83.8 [60.4, 94.6] |
|  |  | 1.0 | 25.3 [10.4, 41.8] | 1.05 [0.43, 1.74] | 94.9 [84.6, 100.0] |
|  |  | 2.0 | 61.6 [46.2, 76.0] | 2.56 [1.92, 3.16] | 97.0 [87.0, 100.0] |
|  |  | 4.0 | 99.0 [97.5, 100.0] | 4.11 [4.05, 4.16] | 100.0 [100.0, 100.0] |


## 6. POST-HOC (written after the numbers above were read; ⛔ not pre-registered, no reading below changes section 3)

Two things in the census prompted questions the prereg did not ask, and the answers matter for how P6-HN is specified:

1. **Is the teacher itself a hard negative in the sets that carry a proximate hard negative?** Only by its collision calculator, and almost never (9 of 3,137 sets; 0.38 % [0.00, 0.92] of the proximate-hard-negative sets at 0.5 m). Its DAC is UNVERIFIED (section 1). The DriveRL collision calculator agrees with PDM NC == 0 on 14,097 proposals and disagrees on 2,574 (calculator only) and 2,472 (label only).
2. **Why do the zero picks sit 1.6 to 1.9 m from the teacher while hard negatives sit 0.2 to 0.4 m cross-track?** Because ADE is dominated by along-track lag. T12 splits the deviation on the teacher's own tangent. **Hard negatives are laterally displaced about 2 to 3x more than non-hard-negatives at the same ADE** (ADE band 0 to 0.5 m: cross-track median 0.20 vs 0.07 m; 0.5 to 1 m: 0.37 vs 0.13; 1 to 2 m: 0.31 vs 0.14) while their along-track offsets are about the same. **Zero picks are slower than the teacher, not faster** (signed along-track -1.4 to -1.6 m, path-length ratio 0.89 to 0.91 against 0.96 to 0.97 for non-zero picks) **and laterally off by 0.4 m (non-zero picks 0.13 m)**; 75 to 83 of about 100 are DAC-0 plans, 21 to 28 NC-0 only.

The same census with a **cross-track proximity** (mean absolute cross-track distance to the teacher path over the 20 poses, any speed; thresholds 0.3 / 0.5 / 1.0 / 2.0 m fixed before the run) is in T11: supply **53.5 % [44.6, 61.6]** at 0.3 m and **60.2 % [50.5, 69.2]** at 0.5 m, saturating at the 64.5 % any-hard-negative ceiling by 1 m; the hard-negative rate rises monotonically with cross-track distance (8.9 % within 0.15 m, 19.0 % to 0.3 m, 32.0 % to 0.6 m, 40.8 % to 1 m). **62 to 69 %** of the siblings' zero picks lie within 0.5 m cross-track of the teacher path (ceiling **2.6 to 2.9 PDMS**), 29 to 34 % within 0.3 m. C1 on this axis: **0.46 to 0.50** at 0.3 m and **0.52 to 0.57** at 0.5 m.

⚠️ **What this does and does not license.** It shows the committed ADE reading (PARTIAL, 0.3 PDMS) rests on a proximity axis that mixes speed into a lateral question. On the lateral axis the zero picks ARE near-twins of the teacher path, which is the ZT3 mechanism (ZT3's twin is the best non-zero proposal, median 0.29 m lateral, a different anchor), and the Lab's lateral-offset bank would address them. It does **not** make the supply "exist" for P6-HN: that needs this definition pre-registered, with both outcomes committed, before it decides anything (section 8, N3).

Tables (generated):

### T10 POST-HOC (not pre-registered): distance of the siblings' picks to the teacher plan (point; m)

| scorer | zero picks n | ADE zero picks q25/q50/q75 | ADE non-zero picks q25/q50/q75 | FDE zero picks q50 | FDE non-zero picks q50 | zero picks: DAC-0 / NC-0 only |
| --- | --- | --- | --- | --- | --- | --- |
| N-s0 | 103 | 1.04 / 1.68 / 2.28 | 0.39 / 0.70 / 1.30 | 4.62 | 1.89 | 75 / 28 |
| N-s1 | 110 | 1.29 / 1.89 / 2.33 | 0.39 / 0.73 / 1.34 | 5.22 | 1.91 | 83 / 27 |
| N-s2 | 99 | 1.00 / 1.63 / 2.33 | 0.40 / 0.69 / 1.28 | 4.49 | 1.85 | 78 / 21 |

POST-HOC Q1/Q2 (point unless a CI is shown): teacher flagged by its own collision calculator in 9 of 3,137 sets; its DriveRL `dac.violation` flags 0 sets, and the same field flags only 53 of 19284 PDM DAC-0 proposals (UNUSABLE as a teacher-DAC check); collision-calculator vs PDM NC==0 on proposals: both 14,097, calculator only 2,574, label only 2,472.  Share of proximate-HN sets (d=0.5) in which the teacher is collision-flagged: 0.38 [0.00, 0.92] %.  Concentration at d=0.5: 1058 sets with a proximate HN, half of them in 4 of 24 logs, top log 2021.10.05.04.03.05_veh-50_01466_01790 = 30.9 %.


### T11 POST-HOC (not pre-registered): the supply census with CROSS-TRACK proximity (mean abs cross-track to the teacher path over the 20 poses; speed ignored); thresholds fixed before the run

| c (m) | sets with a proximate proposal % | **sets with >= 1 proximate HN %** | mean proximate / set | mean proximate HN / set | HN rate among proximate % | C1 reach r=0.208 (PDMS) | C1 reach r=0.227 (PDMS) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 99.8 [99.6, 100.0] | **53.5 [44.6, 61.6]** | 48.9 [44.4, 52.6] | 5.70 [3.86, 7.66] | 11.7 [7.6, 16.8] | 0.46 [0.39, 0.53] | 0.50 [0.42, 0.58] |
| 0.5 | 99.9 [99.9, 100.0] | **60.2 [50.5, 69.2]** | 57.0 [54.7, 59.0] | 8.20 [5.53, 11.27] | 14.4 [9.6, 20.3] | 0.52 [0.44, 0.60] | 0.57 [0.48, 0.65] |
| 1.0 | 100.0 [100.0, 100.0] | **63.9 [54.4, 72.8]** | 63.0 [62.6, 63.4] | 10.57 [7.20, 14.55] | 16.8 [11.4, 23.1] | 0.55 [0.47, 0.63] | 0.60 [0.51, 0.69] |
| 2.0 | 100.0 [100.0, 100.0] | **64.5 [55.1, 73.5]** | 64.0 [63.9, 64.0] | 11.03 [7.58, 15.11] | 17.2 [11.9, 23.6] | 0.56 [0.48, 0.63] | 0.61 [0.52, 0.69] |

| cross-track band (m) | HN rate % | n proposals | n HN |
| --- | --- | --- | --- |
| (0.0,0.15] | 8.9 [5.5, 13.4] | 110,835 | 9,837 |
| (0.15,0.3] | 19.0 [13.1, 24.3] | 42,415 | 8,042 |
| (0.3,0.6] | 32.0 [23.5, 38.6] | 32,646 | 10,445 |
| (0.6,1.0] | 40.8 [31.3, 48.5] | 11,881 | 4,845 |
| (1.0,inf] | 48.8 [39.4, 57.0] | 2,991 | 1,460 |

| scorer | zero picks within c = 0.3 % | c = 0.5 % | c = 1.0 % | ceiling at c = 0.5 (PDMS) |
| --- | --- | --- | --- | --- |
| N-s0 | 29.1 [17.0, 46.7] | 62.1 [46.1, 76.6] | 94.2 [88.7, 97.6] | 2.58 [1.91, 3.18] |
| N-s1 | 30.9 [17.9, 50.0] | 64.5 [50.0, 79.2] | 94.5 [89.1, 97.8] | 2.68 [2.08, 3.29] |
| N-s2 | 34.3 [22.6, 50.9] | 68.7 [51.4, 87.9] | 94.9 [90.2, 98.5] | 2.85 [2.13, 3.65] |

### T12 POST-HOC: how hard negatives and zero picks deviate from the teacher path (teacher-tangent frame, m; medians q25/q50/q75 over proposals; point)

| population | n | mean abs along-track | mean abs cross-track | mean signed along-track (negative = behind the teacher) |
| --- | --- | --- | --- | --- |
| ADE band (0.0,0.5]: HN | 6,571 | 0.12 / 0.21 / 0.31 | 0.11 / 0.20 / 0.30 | -0.17 / 0.02 / 0.21 |
| ADE band (0.0,0.5]: nonHN | 66,797 | 0.12 / 0.22 / 0.34 | 0.03 / 0.07 / 0.16 | -0.22 / -0.01 / 0.20 |
| ADE band (0.5,1.0]: HN | 10,660 | 0.38 / 0.56 / 0.73 | 0.18 / 0.37 / 0.54 | -0.46 / 0.28 / 0.63 |
| ADE band (0.5,1.0]: nonHN | 51,999 | 0.52 / 0.64 / 0.79 | 0.05 / 0.13 / 0.31 | -0.66 / -0.26 / 0.61 |
| ADE band (1.0,2.0]: HN | 10,873 | 1.03 / 1.28 / 1.57 | 0.11 / 0.31 / 0.63 | -0.42 / 1.14 / 1.51 |
| ADE band (1.0,2.0]: nonHN | 35,373 | 1.09 / 1.28 / 1.54 | 0.06 / 0.14 / 0.31 | -1.39 / -1.06 / 1.13 |
| ADE band (2.0,4.0]: HN | 5,952 | 2.20 / 2.52 / 2.99 | 0.12 / 0.29 / 0.57 | 2.08 / 2.43 / 2.92 |
| ADE band (2.0,4.0]: nonHN | 11,165 | 2.17 / 2.45 / 2.90 | 0.07 / 0.17 / 0.38 | -2.74 / -2.25 / 2.00 |
| N-s0 picks: zero | 103 | 0.95 / 1.61 / 2.11 | 0.27 / 0.42 / 0.60 | -2.02 / -1.45 / -0.64; path-length ratio to teacher 0.80 / 0.91 / 0.95 |
| N-s0 picks: nonzero | 3034 | 0.27 / 0.60 / 1.23 | 0.04 / 0.14 / 0.34 | -1.05 / -0.23 / 0.29; path-length ratio to teacher 0.85 / 0.97 / 1.06 |
| N-s1 picks: zero | 110 | 1.16 / 1.78 / 2.21 | 0.26 / 0.39 / 0.61 | -2.11 / -1.63 / -0.84; path-length ratio to teacher 0.78 / 0.89 / 0.93 |
| N-s1 picks: nonzero | 3027 | 0.28 / 0.62 / 1.25 | 0.04 / 0.13 / 0.33 | -1.11 / -0.25 / 0.30; path-length ratio to teacher 0.85 / 0.96 / 1.06 |
| N-s2 picks: zero | 99 | 0.85 / 1.52 / 2.16 | 0.26 / 0.40 / 0.56 | -2.11 / -1.40 / -0.61; path-length ratio to teacher 0.77 / 0.89 / 0.95 |
| N-s2 picks: nonzero | 3038 | 0.27 / 0.59 / 1.19 | 0.04 / 0.13 / 0.33 | -0.99 / -0.23 / 0.28; path-length ratio to teacher 0.85 / 0.97 / 1.06 |

## 7. Does the data needed for hard-negative training exist on navtrain?

| need | exists? | where / what |
|---|---|---|
| proposals + per-proposal PDM safety labels + teacher plan, **dev scale** | **yes** | this table: 3,137 sets, 24 logs; PDM nc / dac / pdms; the fan is u400's generator, not P6's |
| proposals + labels + teacher plan, **training scale** | **yes, archived, not unpacked here** | `D:/Projects/TanitAD-artifacts/refe-pod-final-2026-10-04/onpolicy_sets_logs_full.tar.zst` (2.07 GB): **190,696 labelled on-policy training sets** (`onpolicy_sets_wc.txt`, a line count taken on the pod: **INHERITED, the archive was not opened here**); `onpolicy_queue.tar.zst` (1.82 GB): the queue chunks that carry the teacher plan; `refe_grow_test.tar.zst`, `refe_navtrain_banks_rest.tar.zst` and `../refe-onpolicy-2026-10-01/refe_train_grow.tar.zst`: the teacher target bank. **Their labels are simulator-calculator labels, not the S1 PDM labels.** If the sets are label_version 4 or 5 they carry `navsim_dac.violation`, which equals the PDM DAC on 200,768 / 200,768 held-out proposals; the collision calculator agrees with PDM NC == 0 on 14,097 and disagrees on 5,046 of 200,768. **Which label_versions the 190,696 carry is UNVERIFIED**, and the fan varies with the checkpoint at dump time |
| the labeller | **yes** | `refe/onpolicy_label_v4.py`, `refe/rl_tlane_label.py`; cost INHERITED and **not reconciled**: c = 0.34 to 0.36 CPU-s per set (`P6_GO_SIGNAL.md` C4, NAVSIM-equal labels) vs about 10.2 CPU-s per set for the full DriveRL teacher-plus-64 simulation (`eval/HANDOFF_RL_S1.md`) |
| the Lab's step-1 **offset bank** (teacher +/- 0.15 to 0.60 m) | **no** | not built; its hard-negative yield cannot be read from the fan (the fan's nearest proposals are 0.09 m ADE, mostly along-track variants; lateral 0.3 to 0.6 m offsets are a different population) |
| a **human** future for every navtrain scene | **partly** | nuPlan DBs present for 13 of the 24 held-out logs; the navtrain DBs were not retrieved from the pod (`data/navtrain_dbs` 164 GB, 2026-10-04 manifest) |

## 8. Next lever (Rule Zero: the census is a waypoint). Cheapest experiments first, all 0 GPU

* **N1, training-scale census (CPU, no new labels).** Unpack `onpolicy_sets_logs_full.tar.zst` (2.07 GB) and run the same code on the 190,696 training sets, with the proximity axis (ADE vs cross-track) **pre-registered first**. It removes the 24-log / one-fan limit that dominates every interval above. ESTIMATED size: about 11 GB uncompressed (59 KB per set x 190,696), parse at about 70 MB/s, plus decompression: it may pass the 8-minute line, so launch it detached with a HANDOFF.
* **N2, offset-bank yield pilot (CPU, creates labels, needs the labeller env).** Label the teacher laterally offset by +/- {0.15, 0.30, 0.45, 0.60} m on the 3,137 held-out sets (3,137 x 8 = 25,096 candidates; about 138 CPU-s at c = 0.35, ESTIMATED: 3,137 x 8 / 64 x 0.35) and read the share that is PDMS-zero and the share of scenes with at least one. This decides whether step 1 can supply the 66 % of scenes with no proximate hard negative at 0.5 m ADE. Not run here: it creates labels, which is outside this step.
* **N3, one definition before the arm.** The P6-HN pre-registration must fix proximity as **cross-track to the teacher path (and to the human path where it exists), speed excluded**, and say which of {fan-mined pairs, offset bank, both} feeds the term. This census supports a pair-mined variant too: wherever a DAC-0 proposal exists it has a DAC-1 twin within 0.5 m in 98.9 % of such sets, so the contrast needs no bank in about a third of scenes.
* **Do not run the P6-HN A/B before P6 reads** (Lab s4.4); it also needs a replicate arm (`H-ESTIM-SEED-1`) and the ZT3 / ZT4 read-out.

## 9. What would change these conclusions

A different fan (P6's) with a different density around the teacher; the training-scale census; a human-anchored census on all 24 logs; a teacher DAC check; and any one of the 24 logs: the top log alone moves s_hn(0.5) from 33.7 % to about 27.1 %. The intervals are *another draw of 24 logs* and exclude fan, teacher and training variance.

## 10. Draft BACKLOG status line (P6-HN)

`P6-HN | STEP 0 DONE 2026-10-18, 0 GPU (raw/2026-10-18-hn-census/RESULT.md; PREREG_HN_CENSUS.md sha256 5943731c): navtrain held-out table (3,137 sets / 24 logs, u400 fan, teacher-anchored): sets with >= 1 expert-proximate hard negative 33.7 % [23.3, 46.7] at 0.5 m ADE, 48.8 % at 1 m, 60.5 % at 2 m, ceiling 64.5 % = any hard negative; pre-registered reading PARTIAL; Lab twin-pair rule PRESENT (34.7 %; 98.9 % of sets with a DAC-0 have a twin). ESTIMATED gross reach from the existing fan 0.29 to 0.32 PDMS at 0.5 m ADE (Lab unconstrained 0.86 to 0.94); POST-HOC cross-track axis 0.46 to 0.57, zero picks sit 0.4 m cross-track / 1.5 m behind the teacher. Hard-negative data exists (190,696 labelled training sets archived; offset bank not built). NEXT: N1 training-scale census, N2 offset-bank yield pilot (CPU), N3 pre-register the cross-track proximity; A/B waits for P6.`

## 11. Integration escalations (Master Mind)

1. ⭐ **Land this folder** (`paths.json` lists every file as `[repo path, abs src, "raw"]`). Nothing is staged or committed; the package lives ONLY at `C:/Users/Admin/hncensus/` (dev box).
2. ⭐ **Register row proposed, not written** (the register is the Master Mind's): `D-REFE-HNCENSUS-1`, the supply bullet of section 0, MEASURED (descriptive), dev table, one fan, 24 logs; carry the POST-HOC flag on every cross-track number.
3. ⭐ **Decision for the P6-HN pre-registration:** the proximity axis (section 8, N3). The committed ADE reading and the post-hoc cross-track reading disagree on the band (PARTIAL vs SUPPLY EXISTS), and they disagree for a reason (speed enters ADE), not by noise.
4. The 11 missing nuPlan DBs and the teacher's NAVSIM DAC are the two gaps for a human-anchored or teacher-validated census.

## 12. Reproduce

```
VPY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
$VPY -I code/hn_load.py   --out raw/hn_joined.npz                 # 46 s
$VPY -I code/hn_gates.py  raw/hn_joined.npz raw/gates_data.json   # run BEFORE the prereg hash
$VPY -I code/hn_human.py  --joined raw/hn_joined.npz --out raw/human_future.npz
$VPY -I code/test_hn_census.py > raw/test_hn_census.log ; $VPY -I code/run_mutants.py --work <scratch> --out raw/mutants.json
$VPY -I code/hn_census.py --joined raw/hn_joined.npz --human raw/human_future.npz --out raw/census_results.json --per-set raw/per_set.csv   # 16 s
$VPY -I code/hn_posthoc.py  --joined raw/hn_joined.npz --per-set raw/per_set.csv --out raw/posthoc.json     # POST-HOC
$VPY -I code/hn_posthoc2.py --joined raw/hn_joined.npz --out raw/posthoc2.json                              # POST-HOC
$VPY -I code/render_tables.py raw > raw/tables.md
```
`raw/hn_joined.npz` (25.5 MB, sha256 in `inputs.json`) is a derived cache of banked inputs and is NOT in `paths.json`; rebuild it with `hn_load.py`.
CI convention note: strata, the human-expert subset and the zero-pick ratios reuse the SAME 24-log draws (`zero_tail.Boot` semantics), so a restricted subset sees a random number of its logs per draw; those intervals are slightly wider than a subset-only bootstrap would give.

## 13. Deliverable manifest

| artifact | where | copies |
|---|---|---|
| this report, prereg + sha256, code, raw JSON / CSV / logs, tables | `C:/Users/Admin/hncensus/` (dev box) | **ONE place** (mirror path when landed: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-hn-census/`) |
| landing list | `C:/Users/Admin/hncensus/paths.json` | one place |
| derived cache `raw/hn_joined.npz` | dev box only, rebuildable in 46 s | one place, not landed |
| nothing on a pod, nothing in a worktree, nothing staged, nothing committed or pushed | | |
