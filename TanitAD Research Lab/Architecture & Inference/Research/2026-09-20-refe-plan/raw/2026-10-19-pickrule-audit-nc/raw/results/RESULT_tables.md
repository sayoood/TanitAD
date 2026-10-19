### Outcome

* **Final outcome: ROW-MISS** (PRIMARY source OOF, both seeds must agree). Source-sensitive (OOF vs BANKED outcome differ): **False**.
* BANKED: seed classes ['ROW-MISS', 'ROW-MISS'] -> **ROW-MISS**; VETO-RECOVERS **False** (recovering tau: none); committed next arm: NC calibration on the fast band (temperature or isotonic, out-of-fold), registered
* OOF: seed classes ['ROW-MISS', 'ROW-MISS'] -> **ROW-MISS**; VETO-RECOVERS **False** (recovering tau: none); committed next arm: NC calibration on the fast band (temperature or isotonic, out-of-fold), registered

### Gate and controls

| check | result | detail |
|---|---|---|
| C-COL[BANKED_s0] | PASS | {"ep_proposals_differing": 1011712, "other_columns_bit_equal": true} |
| C-COL[BANKED_s1] | PASS | {"ep_proposals_differing": 1011712, "other_columns_bit_equal": true} |
| C-COL[OOF_s0] | PASS | {"ep_proposals_differing": 1011712, "other_columns_bit_equal": true} |
| C-COL[OOF_s1] | PASS | {"ep_proposals_differing": 1011712, "other_columns_bit_equal": true} |
| C-ID1 | PASS | {"equals_epr1_picks_np": "15808/15808", "float32_torch_diagnostic_equals_banked": "15806/15808", "float64_rule_equals_banked_pick": "15808/15808", "planner_pin": {"file_sha256_equals_pin": true, "lines_sha256": "bcfaf349a155d881a3b35589bde3b1106f87d5e405d715be8e165625953c3574", "lines_sha256_expected": "bcfaf349a155d881a3b35589bde3b1106f87d5e405d715be8e165625953c3574", "status": "PASS", "v1w_line": "V1_W = (5.0, 5.0, |
| C-ID2[seed0] | PASS | {"abs_diff_vs_banked": {"new_pdms_x100": 0.0, "p2_pdms_x100": 0.0, "unsafe_rate_new": 0.0, "unsafe_rate_p2": 0.0}, "got": {"new_pdms_x100": 89.21936256518806, "p2_pdms_x100": 90.36484132879056, "unsafe_rate_new": 0.04061234817813765, "unsafe_rate_p2": 0.030364372469635626}, "picks_changed": 12314, "picks_changed_banked": 12314, "tolerance": 1e-09} |
| C-ID2[seed1] | PASS | {"abs_diff_vs_banked": {"new_pdms_x100": 0.0, "p2_pdms_x100": 0.0, "unsafe_rate_new": 0.0, "unsafe_rate_p2": 0.0}, "got": {"new_pdms_x100": 89.19121054696816, "p2_pdms_x100": 90.36484132879056, "unsafe_rate_new": 0.04048582995951417, "unsafe_rate_p2": 0.030364372469635626}, "picks_changed": 12292, "picks_changed_banked": 12292, "tolerance": 1e-09} |
| C-ID3 | PASS | {"tau0.5_changed_sets": 0, "tau0.5_fallback_sets": 0, "tau0.5_vetoed_pick_nc_sigmoid_ge_tau": true, "tau0_equals_unvetoed": true, "tau1.01_equals_unvetoed_all_fallback": true} |
| C-ID3-registered-reach-clause | FAIL | {"min_nc_sigmoid_at_p2_picks": 0.6286006861124068, "note": "INFORMATIONAL since Amendment 2 (amended after it failed; not a pre-registered pass)", "p2_picks_below_tau": {"0.3": 0, "0.5": 0, "0.7": 1, "0.9": 62}, "registered_text": "veto at tau = 0.5 changes the pick on at least one set and on fewer than all", "tau0.5_changed_sets": 0} |
| C-ID3b | PASS | {"non_fallback_vetoed_pick_nc_sigmoid_ge_tau": true, "of": 15808, "shift_logits": -3.0, "tau0.5_changed_sets_under_shift": 18, "tau0.5_fallback_sets_under_shift": 241} |
| C-PDMS | PASS | {"bit_equal_to_epr1_pdms_np": true, "source": "refe/planner.py:520 V1_W", "weights": [5.0, 5.0, 2.0]} |

### Q1 / Q2: overridden share (paired log-cluster bootstrap, B = 10,000, seed 20260927; CI = percentile 2.5 / 97.5)

| source / seed | n sets A | overridden | **Q1** | class | Q2 (P2 own NC-unsafe picks) | n | Q2 matched | Q1 - Q2 | median NC sigmoid at A picks |
|---|---|---|---|---|---|---|---|---|---|
| BANKED_s0 | 77 | 0 | **0.000 [0.000, 0.000]** | **ROW-MISS** | 0.000 [0.000, 0.000] | 118 | 0.000 [0.000, 0.000] (n 41) | 0.000 [0.000, 0.000] | 0.959 (q25 0.890, q75 0.978) |
| BANKED_s1 | 78 | 0 | **0.000 [0.000, 0.000]** | **ROW-MISS** | 0.000 [0.000, 0.000] | 118 | 0.000 [0.000, 0.000] (n 41) | 0.000 [0.000, 0.000] | 0.958 (q25 0.893, q75 0.978) |
| OOF_s0 | 82 | 0 | **0.000 [0.000, 0.000]** | **ROW-MISS** | 0.000 [0.000, 0.000] | 118 | 0.000 [0.000, 0.000] (n 36) | 0.000 [0.000, 0.000] | 0.960 (q25 0.888, q75 0.979) |
| OOF_s1 | 83 | 0 | **0.000 [0.000, 0.000]** | **ROW-MISS** | 0.000 [0.000, 0.000] | 118 | 0.000 [0.000, 0.000] (n 35) | 0.000 [0.000, 0.000] | 0.959 (q25 0.890, q75 0.979) |

Descriptive sensitivities (no authority): unsafe := NC label < 1, and the DAC analogue (DAC label 0, DAC sigmoid < 0.5).

| source / seed | Q1 (NC < 1) | n A | Q2 (NC < 1) | DAC Q1 | n A (DAC) | DAC Q2 | EP-led picks changed vs P2 |
|---|---|---|---|---|---|---|---|
| BANKED_s0 | 0.000 [0.000, 0.000] | 92 | 0.000 [0.000, 0.000] | 0.059 [0.014, 0.112] | 153 | 0.039 [0.012, 0.070] | 12466 |
| BANKED_s1 | 0.000 [0.000, 0.000] | 93 | 0.000 [0.000, 0.000] | 0.059 [0.014, 0.112] | 153 | 0.039 [0.012, 0.070] | 12458 |
| OOF_s0 | 0.000 [0.000, 0.000] | 97 | 0.000 [0.000, 0.000] | 0.070 [0.025, 0.119] | 172 | 0.039 [0.012, 0.070] | 12314 |
| OOF_s1 | 0.000 [0.000, 0.000] | 98 | 0.000 [0.000, 0.000] | 0.064 [0.025, 0.105] | 171 | 0.039 [0.012, 0.070] | 12292 |

### Q3: hard NC veto sweep on the EP-led composite (rows unchanged; x100 for PDMS; rates over all 15,808 sets; d = composite_tau minus P2 UNVETOED, paired)

P2 unvetoed: NC-unsafe pick rate 0.0075 [0.0052, 0.0102], DAC-unsafe pick rate 0.0228 [0.0162, 0.0309], pick-level PDMS x100 90.365 [89.150, 91.436]; P2's NC-unsafe 97.5th percentile (P2_hi) = 0.01017.

| source / seed | tau | NC-unsafe rate | DAC-unsafe rate | PDMS x100 | d NC-unsafe | d DAC-unsafe | **d PDMS x100** | fallback sets | picks changed by the veto |
|---|---|---|---|---|---|---|---|---|---|
| BANKED_s0 | 0.3 | 0.0097 [0.0065, 0.0137] | 0.0286 [0.0203, 0.0386] | 89.194 [87.944, 90.281] | 0.0023 [0.0003, 0.0046] | 0.0058 [0.0035, 0.0085] | **-1.171 [-1.483, -0.873]** | 0 | 0 |
| BANKED_s0 | 0.5 | 0.0097 [0.0065, 0.0137] | 0.0286 [0.0203, 0.0386] | 89.194 [87.944, 90.281] | 0.0023 [0.0003, 0.0046] | 0.0058 [0.0035, 0.0085] | **-1.171 [-1.483, -0.873]** | 0 | 2 |
| BANKED_s0 | 0.7 | 0.0096 [0.0064, 0.0135] | 0.0286 [0.0203, 0.0386] | 89.201 [87.959, 90.282] | 0.0022 [0.0003, 0.0044] | 0.0058 [0.0035, 0.0085] | **-1.164 [-1.471, -0.868]** | 1 | 8 |
| BANKED_s0 | 0.9 | 0.0094 [0.0062, 0.0132] | 0.0285 [0.0201, 0.0384] | 89.233 [87.995, 90.322] | 0.0019 [0.0001, 0.0041] | 0.0056 [0.0034, 0.0084] | **-1.131 [-1.443, -0.837]** | 37 | 146 |
| BANKED_s0 | none | 0.0097 [0.0065, 0.0137] | 0.0286 [0.0203, 0.0386] | 89.194 [87.944, 90.281] | 0.0023 [0.0003, 0.0046] | 0.0058 [0.0035, 0.0085] | **-1.171 [-1.483, -0.873]** | 0 | 0 |
| BANKED_s1 | 0.3 | 0.0098 [0.0065, 0.0138] | 0.0287 [0.0203, 0.0386] | 89.192 [87.939, 90.285] | 0.0023 [0.0003, 0.0047] | 0.0058 [0.0035, 0.0086] | **-1.173 [-1.491, -0.871]** | 0 | 0 |
| BANKED_s1 | 0.5 | 0.0098 [0.0065, 0.0138] | 0.0287 [0.0203, 0.0386] | 89.192 [87.939, 90.285] | 0.0023 [0.0003, 0.0047] | 0.0058 [0.0035, 0.0086] | **-1.173 [-1.491, -0.871]** | 0 | 2 |
| BANKED_s1 | 0.7 | 0.0097 [0.0065, 0.0135] | 0.0287 [0.0203, 0.0386] | 89.198 [87.950, 90.286] | 0.0022 [0.0003, 0.0044] | 0.0058 [0.0035, 0.0086] | **-1.166 [-1.480, -0.867]** | 1 | 8 |
| BANKED_s1 | 0.9 | 0.0094 [0.0063, 0.0133] | 0.0285 [0.0202, 0.0384] | 89.232 [87.990, 90.327] | 0.0020 [0.0001, 0.0042] | 0.0057 [0.0034, 0.0085] | **-1.132 [-1.448, -0.832]** | 37 | 147 |
| BANKED_s1 | none | 0.0098 [0.0065, 0.0138] | 0.0287 [0.0203, 0.0386] | 89.192 [87.939, 90.285] | 0.0023 [0.0003, 0.0047] | 0.0058 [0.0035, 0.0086] | **-1.173 [-1.491, -0.871]** | 0 | 0 |
| OOF_s0 | 0.3 | 0.0104 [0.0070, 0.0145] | 0.0300 [0.0212, 0.0404] | 89.219 [87.932, 90.362] | 0.0029 [0.0009, 0.0054] | 0.0071 [0.0044, 0.0104] | **-1.145 [-1.473, -0.840]** | 0 | 0 |
| OOF_s0 | 0.5 | 0.0104 [0.0070, 0.0146] | 0.0300 [0.0212, 0.0404] | 89.213 [87.921, 90.359] | 0.0030 [0.0010, 0.0055] | 0.0071 [0.0044, 0.0104] | **-1.152 [-1.487, -0.841]** | 0 | 2 |
| OOF_s0 | 0.7 | 0.0102 [0.0069, 0.0142] | 0.0300 [0.0212, 0.0404] | 89.232 [87.956, 90.365] | 0.0027 [0.0009, 0.0050] | 0.0071 [0.0044, 0.0104] | **-1.133 [-1.450, -0.833]** | 1 | 8 |
| OOF_s0 | 0.9 | 0.0100 [0.0067, 0.0140] | 0.0297 [0.0210, 0.0401] | 89.270 [87.992, 90.405] | 0.0025 [0.0007, 0.0047] | 0.0069 [0.0041, 0.0102] | **-1.095 [-1.425, -0.781]** | 37 | 145 |
| OOF_s0 | none | 0.0104 [0.0070, 0.0145] | 0.0300 [0.0212, 0.0404] | 89.219 [87.932, 90.362] | 0.0029 [0.0009, 0.0054] | 0.0071 [0.0044, 0.0104] | **-1.145 [-1.473, -0.840]** | 0 | 0 |
| OOF_s1 | 0.3 | 0.0105 [0.0071, 0.0145] | 0.0297 [0.0210, 0.0401] | 89.191 [87.907, 90.324] | 0.0030 [0.0010, 0.0054] | 0.0069 [0.0043, 0.0101] | **-1.174 [-1.503, -0.858]** | 0 | 0 |
| OOF_s1 | 0.5 | 0.0106 [0.0071, 0.0146] | 0.0297 [0.0210, 0.0401] | 89.185 [87.896, 90.320] | 0.0031 [0.0011, 0.0056] | 0.0069 [0.0043, 0.0101] | **-1.180 [-1.517, -0.862]** | 0 | 2 |
| OOF_s1 | 0.7 | 0.0103 [0.0071, 0.0143] | 0.0297 [0.0210, 0.0401] | 89.204 [87.930, 90.328] | 0.0028 [0.0010, 0.0050] | 0.0069 [0.0043, 0.0101] | **-1.161 [-1.478, -0.851]** | 1 | 8 |
| OOF_s1 | 0.9 | 0.0101 [0.0068, 0.0140] | 0.0295 [0.0208, 0.0398] | 89.246 [87.970, 90.369] | 0.0026 [0.0008, 0.0048] | 0.0066 [0.0040, 0.0099] | **-1.119 [-1.447, -0.800]** | 37 | 142 |
| OOF_s1 | none | 0.0105 [0.0071, 0.0145] | 0.0297 [0.0210, 0.0401] | 89.191 [87.907, 90.324] | 0.0030 [0.0010, 0.0054] | 0.0069 [0.0043, 0.0101] | **-1.174 [-1.503, -0.858]** | 0 | 0 |

Descriptive reference: the same veto applied to P2's own rows (no EP change).

| tau | NC-unsafe rate | DAC-unsafe rate | PDMS x100 | d PDMS x100 vs P2 unvetoed | fallback | changed |
|---|---|---|---|---|---|---|
| 0.3 | 0.0075 [0.0052, 0.0102] | 0.0228 [0.0162, 0.0309] | 90.365 [89.150, 91.436] | 0.000 [0.000, 0.000] | 0 | 0 |
| 0.5 | 0.0075 [0.0052, 0.0102] | 0.0228 [0.0162, 0.0309] | 90.365 [89.150, 91.436] | 0.000 [0.000, 0.000] | 0 | 0 |
| 0.7 | 0.0075 [0.0052, 0.0102] | 0.0228 [0.0162, 0.0309] | 90.365 [89.150, 91.436] | 0.000 [0.000, 0.000] | 1 | 0 |
| 0.9 | 0.0074 [0.0051, 0.0101] | 0.0229 [0.0162, 0.0309] | 90.367 [89.156, 91.433] | 0.002 [-0.017, 0.023] | 37 | 25 |

### VETO-RECOVERS, every tau (both seeds must satisfy both conditions)

| source | tau | (i) NC-unsafe rate <= P2_hi, seeds 0 / 1 | (ii) d PDMS lower CI >= 0, seeds 0 / 1 | rate points | d PDMS lower CI x100 | recovers |
|---|---|---|---|---|---|---|
| BANKED | 0.3 | [True, True] | [False, False] | [0.00974, 0.00981] | [-1.483, -1.491] | **False** |
| BANKED | 0.5 | [True, True] | [False, False] | [0.00974, 0.00981] | [-1.483, -1.491] | **False** |
| BANKED | 0.7 | [True, True] | [False, False] | [0.00962, 0.00968] | [-1.471, -1.48] | **False** |
| BANKED | 0.9 | [True, True] | [False, False] | [0.00936, 0.00943] | [-1.443, -1.448] | **False** |
| OOF | 0.3 | [False, False] | [False, False] | [0.01037, 0.0105] | [-1.473, -1.503] | **False** |
| OOF | 0.5 | [False, False] | [False, False] | [0.01044, 0.01056] | [-1.487, -1.517] | **False** |
| OOF | 0.7 | [False, False] | [False, False] | [0.01018, 0.01031] | [-1.45, -1.478] | **False** |
| OOF | 0.9 | [True, True] | [False, False] | [0.00999, 0.01006] | [-1.425, -1.447] | **False** |

### C-NULL (labels permuted within each set; a replicate FIRES iff its Q1 lower CI > 0.5; VOID iff >= 2 of 5 fire for any source / seed)

VOID: **False**; fire rate over all replicates: 0.000.

| source / seed | fires | Q1 per replicate (point [lo, hi]; n A) |
|---|---|---|
| BANKED_s0 | 0 of 5 | 0.000 [0.000, 0.000] n557; 0.000 [0.000, 0.000] n508; 0.002 [0.000, 0.007] n500; 0.000 [0.000, 0.000] n566; 0.000 [0.000, 0.000] n529 |
| BANKED_s1 | 0 of 5 | 0.000 [0.000, 0.000] n558; 0.000 [0.000, 0.000] n513; 0.002 [0.000, 0.007] n496; 0.000 [0.000, 0.000] n568; 0.000 [0.000, 0.000] n526 |
| OOF_s0 | 0 of 5 | 0.000 [0.000, 0.000] n542; 0.000 [0.000, 0.000] n523; 0.002 [0.000, 0.007] n510; 0.000 [0.000, 0.000] n546; 0.000 [0.000, 0.000] n496 |
| OOF_s1 | 0 of 5 | 0.000 [0.000, 0.000] n545; 0.000 [0.000, 0.000] n520; 0.002 [0.000, 0.007] n501; 0.000 [0.000, 0.000] n536; 0.000 [0.000, 0.000] n494 |
