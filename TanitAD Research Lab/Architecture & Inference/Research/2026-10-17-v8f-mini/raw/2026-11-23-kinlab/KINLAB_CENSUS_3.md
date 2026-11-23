# KINLAB census 3: `--dagger-label kin_v1` after the MM's construction fix, RH bank s0 shard 0 of 4

**What ran:** the SAME script as phase 2 (`census_kinlab2.py`, CPU, one process) on the phase-3 tree, over the SAME `shard_000of004.npz` (sha256 aefbb46c..., re-pulled read-only, verified) and `native_tracks_union.pt` (sha256 90229524...). Records 492,478; ok 372,791. Outputs: `KINLAB_CENSUS_3.json`; phase 2 kept as `KINLAB_CENSUS_2_phase2.json/.md`. **No constant or rule was changed on this census.** Gold on the first 4,000 real ok records (speed_v1 label_bank, 1,000 human labels) equals df250a00 bit for bit.

## Result: relabelled share 0.6387 of ok records (bar, pre-committed by the MM: >= 0.5) on this bank (seed s0, shard 0 only; the other seeds' banks were not measured)
| of ok records | tick 1 | tick 2 | pooled | phase 2 pooled |
|---|---|---|---|---|
| ok | 208,340 | 164,451 | 372,791 | 372,791 |
| **relabelled (kin_v1)** | 136,233 (65.4 %) | 101,873 (61.9 %) | **238,106 (63.9 %)** | 102,118 (27.4 %) |
| identity rule | 0 | 0 | 0 | 0 |
| refused_not_merged | 40,383 | 38,189 | 78,572 (21.1 %) | 11,884 (3.2 %) |
| refused_path_exhausted | 16,820 | 11,812 | 28,632 (7.7 %) | 19,444 (5.2 %) |
| refused_infeasible | 10,833 | 6,450 | 17,283 (4.6 %) | 229,147 (61.5 %) |
| refused_nonfinite | 0 | 0 | 0 | 0 |
| speed_refused_infeasible (reference itself refused) | 2,131 | 3,348 | 5,479 (1.5 %) | same |
| speed_path_exhausted (reference itself refused) | 1,940 | 2,779 | 4,719 (1.3 %) | same |

refused_infeasible by checker term (not exclusive; pooled 17,283): accel 3,219 (phase 2: 220,689), curvature 14,103 (22,730), friction 4,870 (173,878). The infeasible refusals fell from 61.5 % to 4.6 %; the cost moved to **not_merged**, 3.2 % -> 21.1 % (the new speed condition: the integrated speed has to be within 0.05 m/s of the human step speed at a point where the lateral test also holds), and path_exhausted rose 5.2 % -> 7.7 % (later merges run the lookahead further along the logged path). Refused records keep their speed_v1 label (else the human label for the 10,198 speed refusals); the eligible index is unchanged.

## Over the relabelled records (n = 238,106), phase 3 | phase 2
| quantity | p50 | p95 | p99 | phase 2 p50 / p95 / p99 |
|---|---|---|---|---|
| merge time (s) | 2.2 | 5.4 | 6.0 | 1.0 / 3.6 / 5.2 |
| max lateral deviation from the human polyline (m) | 0.052 | 0.358 | 0.703 | 0.043 / 0.429 / 0.769 |
| point distance \|P_kin[30] - P_speed[30]\| (m) | 0.969 | 7.778 | 12.468 | 0.339 / 1.725 / 3.053 |
| point distance \|P_kin[5] - P_speed[5]\| (first 1 s, m) | 0.734 | 2.816 | 3.275 | 0.320 / 0.732 / 0.961 |
| arc difference \|L_kin[30] - L_speed[30]\| (m) | 0.958 | 7.779 | 12.463 | 0.339 / 1.701 / 3.041 |

## Reading (observations, nothing tuned)
* The bar is met at both ticks (65.4 %, 61.9 %). The 30,000-record refusal diagnostic is not run: it is required only if the share is below 0.5.
* The kin label now departs further from the speed_v1 label (arc difference at knot 30: p50 0.96 m, p95 7.8 m, against 0.34 / 1.70). Cause by construction: the integrated speed follows the rule from the own speed and merges later (p50 2.2 s) at the human's speed, whereas speed_v1's arc is rate-limited from v_o; the two arcs differ by the integrated speed offset accumulated until the merge. First-second point distance also grew (p95 2.8 m, from 0.73), because a(t) now pulls toward the human's speed after 1 s instead of replaying speed_v1's per-step accelerations. This is a property of the MM's rule, reported so the card can read it.
* not_merged (21.1 %) is now the largest refusal: the MM may want to know whether it is the speed band (0.05 m/s) or the lateral band that holds; not diagnosed (no refusal diagnostic was asked for at share >= 0.5).
