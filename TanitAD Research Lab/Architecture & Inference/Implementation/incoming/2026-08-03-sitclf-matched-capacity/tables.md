## The capacity ladder

| rung | params/head | lane_change AP-lift | roundabout AP-lift | intersection AP-lift |
|---|---:|---:|---:|---:|
| `ridge_pca16_w8` | 129 | 1.269 [1.075, 1.571] | 2.619 [1.893, 3.944] | 1.677 [1.454, 1.996] |
| `ridge_pca64_w8` | 513 | 1.483 [1.193, 1.957] | 3.831 [2.281, 7.101] | 1.401 [1.252, 1.602] |
| `ridge_pca256_w8` | 2,049 | 1.161 [0.964, 1.488] | 2.722 [1.683, 4.806] | 1.538 [1.353, 1.794] |
| `ridge_raw2048_w1` | 2,049 | 1.413 [1.145, 2.005] | 2.640 [1.750, 4.506] | 1.516 [1.343, 1.750] |
| `tf_pca16_d8` | 2,068 | 1.519 [1.145, 2.307] | 1.956 [1.447, 2.682] | 1.371 [1.222, 1.573] |
| `tf_pca16_d16` | 7,332 | 1.313 [1.086, 1.645] | 2.234 [1.432, 3.955] | 1.420 [1.272, 1.632] |
| `tf_pca16_d32` | 27,460 | 1.315 [1.065, 1.726] | 2.664 [1.500, 5.074] | 1.431 [1.250, 1.701] |
| `tf_pca16_d64` | 106,116 | 1.520 [1.245, 2.018] | 2.841 [1.911, 4.265] | 1.457 [1.293, 1.666] |
| `tf_pca16_d128` | 417,028 | 1.549 [1.258, 1.957] | 2.519 [1.774, 3.582] | 1.326 [1.186, 1.540] |
| `tf_pca16_d196` | 971,772 | 1.484 [1.199, 1.891] | 2.408 [1.713, 3.488] | 1.350 [1.214, 1.536] |
| `tf_pca16_d296` | 2,207,572 | 1.364 [1.126, 1.690] | 1.754 [1.068, 3.045] | 1.327 [1.187, 1.507] |
| `tf_pca64_d128` | 423,172 | 1.480 [1.227, 1.814] | 3.203 [2.078, 4.822] | 1.486 [1.325, 1.705] |
| `tf_pca16_d128_ep24` | 417,028 | 1.620 [1.311, 2.076] | 2.337 [1.591, 3.444] | 1.298 [1.154, 1.491] |
| `tf_pca16_d296_ep24` | 2,207,572 | 1.516 [1.278, 1.919] | 2.496 [1.650, 3.788] | 1.369 [1.225, 1.560] |

## Each rung against its OWN permuted-feature null (paired)

| rung | lane_change | roundabout | intersection |
|---|---|---|---|
| `ridge_pca16_w8` | +0.0844 [-0.6378, +0.5474]  | +1.8628 [+1.0980, +3.1490] **\*** | +0.6848 [+0.4281, +1.0108] **\*** |
| `ridge_pca64_w8` | +0.3408 [-0.1043, +0.8552]  | +2.9161 [+1.2764, +6.2895] **\*** | +0.3409 [+0.1248, +0.5479] **\*** |
| `ridge_pca256_w8` | -0.0464 [-0.5469, +0.3684]  | +1.6344 [+0.5029, +3.7594] **\*** | +0.4643 [+0.2071, +0.7408] **\*** |
| `ridge_raw2048_w1` | +0.2627 [-0.1846, +0.8573]  | +1.8131 [+0.8155, +3.6863] **\*** | +0.5201 [+0.3180, +0.7610] **\*** |
| `tf_pca16_d8` | +0.4839 [+0.0176, +1.2608] **\*** | +0.9326 [+0.3738, +1.6800] **\*** | +0.3676 [+0.1795, +0.5885] **\*** |
| `tf_pca16_d16` | -0.0181 [-0.6611, +0.3811]  | +1.3220 [+0.4457, +2.9774] **\*** | +0.4308 [+0.2264, +0.6732] **\*** |
| `tf_pca16_d32` | +0.3377 [-0.1609, +0.7543]  | +1.8381 [+0.6092, +4.2933] **\*** | +0.4582 [+0.2472, +0.7278] **\*** |
| `tf_pca16_d64` | +0.4777 [-0.0495, +1.0588]  | +2.0487 [+1.0704, +3.4482] **\*** | +0.5383 [+0.3473, +0.7542] **\*** |
| `tf_pca16_d128` | +0.3812 [-0.2977, +0.8638]  | +1.5687 [+0.7412, +2.6567] **\*** | +0.3535 [+0.1698, +0.5849] **\*** |
| `tf_pca16_d196` | +0.4413 [-0.0590, +0.9443]  | +1.5593 [+0.7850, +2.6282] **\*** | +0.3740 [+0.1852, +0.5804] **\*** |
| `tf_pca16_d296` | +0.3903 [+0.1043, +0.7129] **\*** | +0.8611 [+0.1953, +2.0674] **\*** | +0.3522 [+0.1577, +0.5539] **\*** |
| `tf_pca64_d128` | +0.4516 [+0.0997, +0.8112] **\*** | +2.2108 [+1.0782, +3.8191] **\*** | +0.4770 [+0.2784, +0.7009] **\*** |
| `tf_pca16_d128_ep24` | +0.4549 [+0.0388, +0.9376] **\*** | +1.3987 [+0.6318, +2.5085] **\*** | +0.3205 [+0.1368, +0.5378] **\*** |
| `tf_pca16_d296_ep24` | +0.4396 [+0.0216, +0.9173] **\*** | +1.5244 [+0.6605, +2.7870] **\*** | +0.4249 [+0.2443, +0.6377] **\*** |

## Each rung against the DEPLOYED rung (paired)

baseline = `tf_pca16_d128`

| rung | lane_change | roundabout | intersection |
|---|---|---|---|
| `ridge_pca16_w8` | -0.2795 [-0.5624, -0.0093] **\*** | +0.0999 [-0.7309, +1.1638]  | +0.3512 [+0.1269, +0.6049] **\*** |
| `ridge_pca64_w8` | -0.0656 [-0.5062, +0.4202]  | +1.3118 [-0.2112, +4.3937]  | +0.0747 [-0.1402, +0.2696]  |
| `ridge_pca256_w8` | -0.3873 [-0.8232, +0.0316]  | +0.2032 [-0.9536, +2.1146]  | +0.2118 [-0.0138, +0.4480]  |
| `ridge_raw2048_w1` | -0.1360 [-0.5778, +0.4947]  | +0.1206 [-0.8693, +1.8019]  | +0.1906 [+0.0035, +0.3661] **\*** |
| `tf_pca16_d8` | -0.0295 [-0.3843, +0.6138]  | -0.5637 [-1.4077, +0.0576]  | +0.0454 [-0.1753, +0.2625]  |
| `tf_pca16_d16` | -0.2357 [-0.5773, +0.1110]  | -0.2851 [-1.1483, +1.2779]  | +0.0945 [-0.1199, +0.3004]  |
| `tf_pca16_d32` | -0.2331 [-0.6001, +0.1302]  | +0.1452 [-0.8170, +2.0015]  | +0.1056 [-0.1399, +0.3652]  |
| `tf_pca16_d64` | -0.0280 [-0.3271, +0.3556]  | +0.3222 [-0.2551, +1.1490]  | +0.1311 [-0.0756, +0.3276]  |
| `tf_pca16_d196` | -0.0644 [-0.2943, +0.1662]  | -0.1108 [-0.6335, +0.4120]  | +0.0240 [-0.0961, +0.1373]  |
| `tf_pca16_d296` | -0.1848 [-0.5302, +0.1510]  | -0.7656 [-1.7685, +0.4663]  | +0.0016 [-0.1991, +0.1855]  |
| `tf_pca64_d128` | -0.0683 [-0.3325, +0.1539]  | +0.6835 [+0.0291, +1.6273] **\*** | +0.1598 [+0.0173, +0.3041] **\*** |
| `tf_pca16_d128_ep24` | +0.0714 [-0.1409, +0.3031]  | -0.1824 [-0.7492, +0.3793]  | -0.0279 [-0.1634, +0.0994]  |
| `tf_pca16_d296_ep24` | -0.0322 [-0.3039, +0.2732]  | -0.0235 [-0.6504, +0.7443]  | +0.0430 [-0.1432, +0.2095]  |

## Operating point at a 5 % alarm budget - PRECISION alongside recall

| rung | lane_change P / R (fires, tp) | roundabout P / R (fires, tp) | intersection P / R (fires, tp) |
|---|---|---|---|
| `ridge_pca16_w8` | 0.0182 / 0.0406 (3,907, 71) | 0.0432 / 0.1480 (3,912, 169) | 0.2490 / 0.1257 (3,550, 884) |
| `ridge_pca64_w8` | 0.0348 / 0.0778 (3,907, 136) | 0.0644 / 0.2207 (3,912, 252) | 0.1789 / 0.0903 (3,550, 635) |
| `ridge_pca256_w8` | 0.0323 / 0.0720 (3,907, 126) | 0.0578 / 0.1979 (3,912, 226) | 0.1870 / 0.0944 (3,550, 664) |
| `ridge_raw2048_w1` | 0.0312 / 0.0698 (3,907, 122) | 0.0432 / 0.1480 (3,912, 169) | 0.1955 / 0.0987 (3,550, 694) |
| `tf_pca16_d8` | 0.0471 / 0.1052 (3,907, 184) | 0.0314 / 0.1077 (3,912, 123) | 0.1676 / 0.0846 (3,550, 595) |
| `tf_pca16_d16` | 0.0302 / 0.0675 (3,907, 118) | 0.0368 / 0.1261 (3,912, 144) | 0.1575 / 0.0795 (3,550, 559) |
| `tf_pca16_d32` | 0.0325 / 0.0726 (3,907, 127) | 0.0570 / 0.1953 (3,912, 223) | 0.1600 / 0.0808 (3,550, 568) |
| `tf_pca16_d64` | 0.0345 / 0.0772 (3,907, 135) | 0.0654 / 0.2242 (3,912, 256) | 0.1927 / 0.0973 (3,550, 684) |
| `tf_pca16_d128` | 0.0253 / 0.0566 (3,907, 99) | 0.0532 / 0.1821 (3,912, 208) | 0.1608 / 0.0812 (3,550, 571) |
| `tf_pca16_d196` | 0.0371 / 0.0829 (3,907, 145) | 0.0488 / 0.1673 (3,912, 191) | 0.1391 / 0.0703 (3,550, 494) |
| `tf_pca16_d296` | 0.0230 / 0.0515 (3,907, 90) | 0.0478 / 0.1638 (3,912, 187) | 0.1653 / 0.0835 (3,550, 587) |
| `tf_pca64_d128` | 0.0197 / 0.0440 (3,907, 77) | 0.0787 / 0.2697 (3,912, 308) | 0.1780 / 0.0899 (3,550, 632) |
| `tf_pca16_d128_ep24` | 0.0282 / 0.0629 (3,907, 110) | 0.0557 / 0.1909 (3,912, 218) | 0.1372 / 0.0693 (3,550, 487) |
| `tf_pca16_d296_ep24` | 0.0259 / 0.0578 (3,907, 101) | 0.0596 / 0.2040 (3,912, 233) | 0.1721 / 0.0869 (3,550, 611) |

## Controls

### NEG_FEATURE - AP-lift of every rung fitted on features permuted ACROSS clips

| rung | lane_change | roundabout | intersection |
|---|---:|---:|---:|
| `ridge_pca16_w8` | 1.1846 | 0.7564 | 0.9922 |
| `ridge_pca64_w8` | 1.1421 | 0.9149 | 1.0596 |
| `ridge_pca256_w8` | 1.2077 | 1.0880 | 1.0734 |
| `ridge_raw2048_w1` | 1.1498 | 0.8267 | 0.9963 |
| `tf_pca16_d8` | 1.0351 | 1.0230 | 1.0037 |
| `tf_pca16_d16` | 1.3309 | 0.9121 | 0.9895 |
| `tf_pca16_d32` | 0.9777 | 0.8262 | 0.9733 |
| `tf_pca16_d64` | 1.0427 | 0.7927 | 0.9186 |
| `tf_pca16_d128` | 1.1673 | 0.9505 | 0.9723 |
| `tf_pca16_d196` | 1.0428 | 0.8491 | 0.9759 |
| `tf_pca16_d296` | 0.9735 | 0.8926 | 0.9753 |
| `tf_pca64_d128` | 1.0287 | 0.9918 | 1.0087 |
| `tf_pca16_d128_ep24` | 1.1650 | 0.9381 | 0.9775 |
| `tf_pca16_d296_ep24` | 1.0768 | 0.9713 | 0.9439 |

### NEG_LABEL - labels permuted across whole clusters (AP-lift)

| rung | lane_change | roundabout | intersection |
|---|---:|---:|---:|
| `ridge_pca16_w8` | 1.1710 | 1.0588 | 0.9530 |
| `tf_pca16_d128` | 1.0191 | 1.1035 | 0.9542 |

### SELF-CONSISTENCY - family AP-lift vs an independent recomputation

* `lane_change`: identical = **True** on 78,131 rows - family {'PEAK::tf_pca16_d128_ep24': 1.61993, 'tf_pca16_d128': 1.54854} vs recomputed {'tf_pca16_d128': 1.54854, 'PEAK::tf_pca16_d128_ep24': 1.61993}
* `roundabout`: identical = **True** on 78,249 rows - family {'PEAK::ridge_pca64_w8': 3.83099, 'tf_pca16_d128': 2.51922} vs recomputed {'tf_pca16_d128': 2.51922, 'PEAK::ridge_pca64_w8': 3.83099}
* `intersection`: identical = **True** on 71,010 rows - family {'PEAK::ridge_pca16_w8': 1.677, 'tf_pca16_d128': 1.32586} vs recomputed {'tf_pca16_d128': 1.32586, 'PEAK::ridge_pca16_w8': 1.677}

## Power

| situation | scorable rows | positives | clusters | clusters WITH a positive | base rate |
|---|---:|---:|---:|---:|---:|
| lane_change | 78,131 | 1,749 | 500 | 64 | 0.02238 |
| roundabout | 78,249 | 1,142 | 500 | 39 | 0.01459 |
| intersection | 71,010 | 7,032 | 500 | 230 | 0.09903 |

## Selected epochs (the optimisation read)

| rung | epochs trained | epoch* per fold |
|---|---:|---|
| `ridge_pca16_w8` | - (closed form) | lambda* [10000.0, 10000.0] |
| `ridge_pca64_w8` | - (closed form) | lambda* [1.0, 10.0] |
| `ridge_pca256_w8` | - (closed form) | lambda* [10000.0, 10000.0] |
| `ridge_raw2048_w1` | - (closed form) | lambda* [10000.0, 10.0] |
| `tf_pca16_d8` | 8 | [8, 7] |
| `tf_pca16_d16` | 8 | [8, 7] |
| `tf_pca16_d32` | 8 | [4, 1] |
| `tf_pca16_d64` | 8 | [5, 3] |
| `tf_pca16_d128` | 8 | [1, 4] |
| `tf_pca16_d196` | 8 | [3, 4] |
| `tf_pca16_d296` | 8 | [4, 1] |
| `tf_pca64_d128` | 8 | [3, 4] |
| `tf_pca16_d128_ep24` | 24 | [3, 8] |
| `tf_pca16_d296_ep24` | 24 | [2, 1] |

## THE FOUR FAMILIES (peak rung vs deployed rung)


### lane_change  (peak = `tf_pca16_d128_ep24`)

**TACTICAL** - n 78,131, positives 1749, base 0.02238
* `PEAK::tf_pca16_d128_ep24`: AP-lift **1.6199**, P@5% 0.0282 / R@5% 0.0629 (3,907 fires, 110 tp), lead 2.3 s (57/70 runs no alarm)
* `tf_pca16_d128`: AP-lift **1.5485**, P@5% 0.0253 / R@5% 0.0566 (3,907 fires, 99 tp), lead 2.4 s (59/70 runs no alarm)
* paired delta AP-lift: +0.0714 [-0.1409, +0.3031] 

**LONGITUDINAL** - not computable: target-speed accuracy, headway/time-gap/TTC (the arm emits a per-frame situation probability, not a trajectory); reported as regime strata:
* `decelerating`: n 16,793, pos 122, delta -0.4892 [-2.3843, +0.7249] 
* `steady`: n 38,722, pos 908, delta +0.0831 [-0.2929, +0.4473] 
* `accelerating`: n 22,616, pos 719, delta +0.1129 [-0.0662, +0.6949] 
* `low_speed_lt8`: n 56,105, pos 649, delta +0.0098 [-0.2880, +0.3059] 
* `cruise_ge8`: n 22,026, pos 1100, delta +0.1126 [-0.1034, +0.3814] 

**LATERAL** - not computable: heading, curvature, yaw-rate, cross-track error (the arm emits a per-frame situation probability, not a trajectory); reported as regime strata:
* `straight`: n 46,443, pos 577, delta +0.1562 [-0.4361, +0.8845] 
* `turning`: n 31,688, pos 1172, delta +0.0469 [-0.0917, +0.3307] 

**STRATEGIC** - UNAVAILABLE (n 78,131): no route/goal/map label exists on PhysicalAI-AV — settled at five probes (no map, lane graph, junction annotation or route signal; egomotion is clip-local metres with no GNSS), so there is no strategic target to score this arm against.

### roundabout  (peak = `ridge_pca64_w8`)

**TACTICAL** - n 78,249, positives 1142, base 0.01459
* `PEAK::ridge_pca64_w8`: AP-lift **3.8310**, P@5% 0.0644 / R@5% 0.2207 (3,912 fires, 252 tp), lead 2.3 s (21/40 runs no alarm)
* `tf_pca16_d128`: AP-lift **2.5192**, P@5% 0.0532 / R@5% 0.1821 (3,912 fires, 208 tp), lead 2.5 s (23/40 runs no alarm)
* paired delta AP-lift: +1.3118 [-0.2112, +4.3937] 

**LONGITUDINAL** - not computable: target-speed accuracy, headway/time-gap/TTC (the arm emits a per-frame situation probability, not a trajectory); reported as regime strata:
* `decelerating`: n 17,122, pos 244, delta +0.0893 [-0.6930, +0.7074] 
* `steady`: n 38,361, pos 499, delta +4.0424 [+0.9063, +9.6254] **\***
* `accelerating`: n 22,766, pos 399, delta +0.2336 [-2.0331, +4.2460] 
* `low_speed_lt8`: n 53,870, pos 1030, delta +1.2818 [-0.2836, +4.2890] 
* `cruise_ge8`: n 24,379, pos 112, delta +0.6503 [-0.1033, +3.0651] 

**LATERAL** - not computable: heading, curvature, yaw-rate, cross-track error (the arm emits a per-frame situation probability, not a trajectory); reported as regime strata:
* `straight`: n 47,841, pos 416, delta +2.9858 [+0.6988, +10.2504] **\***
* `turning`: n 30,408, pos 726, delta +0.7144 [-0.3883, +2.9361] 

**STRATEGIC** - UNAVAILABLE (n 78,249): no route/goal/map label exists on PhysicalAI-AV — settled at five probes (no map, lane graph, junction annotation or route signal; egomotion is clip-local metres with no GNSS), so there is no strategic target to score this arm against.

### intersection  (peak = `ridge_pca16_w8`)

**TACTICAL** - n 71,010, positives 7032, base 0.09903
* `PEAK::ridge_pca16_w8`: AP-lift **1.6770**, P@5% 0.2490 / R@5% 0.1257 (3,550 fires, 884 tp), lead 2.4 s (190/251 runs no alarm)
* `tf_pca16_d128`: AP-lift **1.3259**, P@5% 0.1608 / R@5% 0.0812 (3,550 fires, 571 tp), lead 1.6 s (206/251 runs no alarm)
* paired delta AP-lift: +0.3512 [+0.1269, +0.6049] **\***

**LONGITUDINAL** - not computable: target-speed accuracy, headway/time-gap/TTC (the arm emits a per-frame situation probability, not a trajectory); reported as regime strata:
* `decelerating`: n 15,895, pos 3097, delta +0.2770 [+0.0242, +0.5623] **\***
* `steady`: n 36,441, pos 2598, delta +0.1878 [-0.1133, +0.5834] 
* `accelerating`: n 18,674, pos 1337, delta +0.6360 [+0.1738, +1.2544] **\***
* `low_speed_lt8`: n 46,609, pos 6035, delta +0.3795 [+0.1759, +0.6091] **\***
* `cruise_ge8`: n 24,401, pos 997, delta +0.1959 [-0.1023, +0.7304] 

**LATERAL** - not computable: heading, curvature, yaw-rate, cross-track error (the arm emits a per-frame situation probability, not a trajectory); reported as regime strata:
* `straight`: n 47,598, pos 5860, delta +0.2777 [+0.0745, +0.4969] **\***
* `turning`: n 23,412, pos 1172, delta +0.3858 [+0.0459, +0.9073] **\***

**STRATEGIC** - UNAVAILABLE (n 71,010): no route/goal/map label exists on PhysicalAI-AV — settled at five probes (no map, lane graph, junction annotation or route signal; egomotion is clip-local metres with no GNSS), so there is no strategic target to score this arm against.
