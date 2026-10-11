n = 1018 rows, 11 logs.

**Amendment gates**

| gate | result | detail |
|---|---|---|
| C8 linear reproduces landed P5 (JSON, `seconds` stripped) | PASS | 0 mismatching fields; max float diff 0 |
| C8 linear reproduces landed P5 (NPZ, 13 arrays) | PASS | all array_equal |
| C2′ cubic knot max dev ≤ 1e-9 | PASS | 4.44e-16 |
| C2′ pchip knot max dev ≤ 1e-9 | PASS | 4.44e-16 |
| C2′ deliberate regression (`--mutate knot --interp cubic`) must FAIL | PASS (it failed) | max dev 0.257 m |

**PASS bar per arm (committed)**

| arm | §4 gates | arm (ii) verdict | (a) Δ sim8 Tc20s − T | (b) adverse guards | (c) acc median ratio | (c) heading-rate median ratio | C3 Tw | PASS |
|---|---|---|---|---|---|---|---|---|
| cubic | all PASS | CONFIRMED | -7.07 pp [-13.26, -1.83] | none | 2.321 / 1.371 = 1.69× | 8.978 / 5.110 = 1.76× | REFUTED | **PASS** |
| pchip | all PASS | CONFIRMED | -7.17 pp [-13.50, -1.83] | none | 2.497 / 1.371 = 1.82× | 9.151 / 5.110 = 1.79× | REFUTED | **PASS** |
| linear (= landed Tc20, reference) | all PASS | CONFIRMED | -7.07 pp [-13.26, -1.83] | none | 2.824 / 1.371 = 2.06× | 8.590 / 5.110 = 1.68× | REFUTED | FAIL c |

**Outcome: P5-S PASSED (primary, cubic)**

**Descriptive, cubic**

| paired contrast | Δ [95 % CI] |
|---|---|
| Tc20s-Tc20 sim8 | +0.00 pp [+0.00, +0.00] |
| Tc20s-Tc20 interp8 | +0.20 pp [+0.00, +0.49] |
| Tc20s-T interp8 | +12.48 pp [+4.37, +21.47] |

| ref | n | sim8 fail | interp8 fail | fail both | pass both |
|---|---|---|---|---|---|
| T | 1018 | 82 | 6 | 4 | 934 (91.75 %) |
| Tc20 | 1018 | 10 | 131 | 7 | 884 (86.84 %) |
| Tc20s | 1018 | 10 | 133 | 7 | 882 (86.64 %) |
| Tsel-s (DERIVED) | 1018 | 6 | 54 | 3 | 961 (94.40 %) |

Tsel-s learnability on its 82 selected rows (median per-row max): |acc| t20 2.456 vs Tc20s 3.665 m/s² (1.49×); |heading rate| t20 26.867 vs Tc20s 35.827 °/s (1.33×).

| label | max acc median | max acc p95 | max heading-rate median °/s | p95 °/s |
|---|---|---|---|---|
| t20 | 1.371 | 3.136 | 5.110 | 32.742 |
| Tc20 | 2.321 | 4.780 | 8.978 | 45.574 |
| Tw | 1.795 | 8.213 | 10.176 | 81.173 |

| arm (ii) guard, Tc20s − T | Δ [95 % CI] |
|---|---|
| sim8 | -7.07 pp [-13.26, -1.83] |
| ddc | +0.69 pp [+0.00, +1.99] |
| comfort | -0.10 pp [-0.41, +0.00] |
| lk10 | -2.46 pp [-5.58, +0.51] |
| mean_abs_along | -0.1481 [-0.1830, -0.1221] |
| mean_abs_cross | -0.0930 [-0.1256, -0.0534] |
| max_xy | -0.3998 [-0.4593, -0.3393] |
| heading_mae_deg | -0.9889 [-1.3718, -0.5051] |

**Descriptive, pchip**

| paired contrast | Δ [95 % CI] |
|---|---|
| Tc20s-Tc20 sim8 | -0.10 pp [-0.25, +0.00] |
| Tc20s-Tc20 interp8 | +0.20 pp [+0.00, +0.49] |
| Tc20s-T interp8 | +12.48 pp [+4.37, +21.47] |

| ref | n | sim8 fail | interp8 fail | fail both | pass both |
|---|---|---|---|---|---|
| T | 1018 | 82 | 6 | 4 | 934 (91.75 %) |
| Tc20 | 1018 | 10 | 131 | 7 | 884 (86.84 %) |
| Tc20s | 1018 | 9 | 133 | 6 | 882 (86.64 %) |
| Tsel-s (DERIVED) | 1018 | 5 | 54 | 2 | 961 (94.40 %) |

Tsel-s learnability on its 82 selected rows (median per-row max): |acc| t20 2.456 vs Tc20s 4.306 m/s² (1.75×); |heading rate| t20 26.867 vs Tc20s 36.018 °/s (1.34×).

| label | max acc median | max acc p95 | max heading-rate median °/s | p95 °/s |
|---|---|---|---|---|
| t20 | 1.371 | 3.136 | 5.110 | 32.742 |
| Tc20 | 2.497 | 5.353 | 9.151 | 45.598 |
| Tw | 2.284 | 11.065 | 10.543 | 84.020 |

| arm (ii) guard, Tc20s − T | Δ [95 % CI] |
|---|---|
| sim8 | -7.17 pp [-13.50, -1.83] |
| ddc | +0.69 pp [+0.00, +1.99] |
| comfort | -0.10 pp [-0.41, +0.00] |
| lk10 | -2.46 pp [-5.58, +0.51] |
| mean_abs_along | -0.1481 [-0.1830, -0.1221] |
| mean_abs_cross | -0.0930 [-0.1256, -0.0534] |
| max_xy | -0.3999 [-0.4594, -0.3393] |
| heading_mae_deg | -0.9891 [-1.3727, -0.5050] |

