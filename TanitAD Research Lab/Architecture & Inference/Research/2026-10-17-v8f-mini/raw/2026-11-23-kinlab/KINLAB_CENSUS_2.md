# KINLAB census 2: `--dagger-label kin_v1` on RH bank s0, shard 0 of 4

**What ran:** `census_kinlab2.py` (CPU, one process, 170 s) on the KINLAB build (`tree/`, df250a00 + the kin_v1 change), over `shard_000of004.npz` (sha256 aefbb46c..., 533 MB) and `native_tracks_union.pt` (sha256 9022952e...), both pulled read-only and deleted afterwards (sources still on the pod). Records 492,478; ok 372,791 (tick 1: 208,340, tick 2: 164,451). Own state = `p10f[-1, 0:3]` of the record. Full numbers: `KINLAB_CENSUS_2.json`. **No constant was tuned on this census.**

**Gold on real records (same run, before the census):** on the first 4,000 ok records the speed_v1 labels (`label_bank`: slot, labels, status counts, census) and 1,000 human labels equal the df250a00 code bit for bit.

## Result: the relabelled share is 0.274, below the 0.5 mark. Reported, design unchanged.

| of ok records | tick 1 | tick 2 | pooled |
|---|---|---|---|
| ok | 208,340 | 164,451 | 372,791 |
| **relabelled (kin_v1 label)** | 76,599 (36.8 %) | 25,519 (15.5 %) | **102,118 (27.4 %)** |
| identity rule | 0 | 0 | 0 |
| refused_infeasible | 110,749 | 118,398 | 229,147 (61.5 %) |
| refused_path_exhausted | 9,509 | 9,935 | 19,444 (5.2 %) |
| refused_not_merged | 7,412 | 4,472 | 11,884 (3.2 %) |
| refused_nonfinite | 0 | 0 | 0 |
| speed_refused_infeasible (the reference itself refused) | 2,131 | 3,348 | 5,479 (1.5 %) |
| speed_path_exhausted (the reference itself refused) | 1,940 | 2,779 | 4,719 (1.3 %) |

Refused records keep their speed_v1 label (362,593 of 372,791 records hold a label: 102,118 kin_v1 plus 260,475 speed_v1 fallbacks); the 10,198 speed refusals keep the human label. The eligible index is unchanged.

**refused_infeasible by checker term** (not exclusive; pooled 229,147): accel 220,689, friction 173,878, curvature 22,730.

## Over the relabelled records (n = 102,118)
| quantity | p50 | p95 | p99 |
|---|---|---|---|
| merge time (s) | 1.0 | 3.6 | 5.2 |
| max lateral deviation of the label from the human polyline (m) | 0.043 | 0.429 | 0.769 |
| point distance \|P_kin[30] - P_speed[30]\| (m) | 0.339 | 1.725 | 3.053 |
| point distance \|P_kin[5] - P_speed[5]\| (first 1 s, m) | 0.320 | 0.732 | 0.961 |
| arc difference \|L_kin[30] - L_speed[30]\| (m) | 0.339 | 1.701 | 3.041 |

## Reading (diagnostic, `diag_refusal.py` on a stride of 30,000 ok records, `KINLAB_DIAG_REFUSAL.json`)
Of 18,410 refused_infeasible records in the sample, 17,733 carry the accel term, and in **98.6 %** of those the maximal chord acceleration sits on the step across the merge point (merge step +- 1). That is the interpretation flagged in `FACTS_KINLAB_2.md` (1): after the 1 s blend the integrated speed differs from the speed_v1 profile by about half the start gap `(a_o - a_ref0)`, and the tail returns to speed_v1's speed in one step. The phase-1 own-vs-human acceleration gap (p50 1.55 m/s2) makes that step exceed A_LIM for a large share of records. The refusal is the design's feasibility rule working as written, not a bug in the checker call; whether to keep the tail as written is the MM's decision. Mind that this is an observation about this reading of the brief's merge sentence, not a test of another one (no alternative was run).

Also: tick 2 relabels less than tick 1 (15.5 % vs 36.8 %), tracking the larger own-vs-human gap at tick 2 in phase 1 (accel p50 2.09 vs 1.16 m/s2).
