### P4-T1 Official navhard two-stage EPDMS and zero-rates per arm (refcv7 step 50,400, inference seed 0; all 5,912 tokens; MEASURED, local devkit)

| arm | EPDMS (devkit) | dEPDMS vs G0 [95 % paired log-cluster CI] | DAC0 | dDAC0 pp [CI] | NC0 | dNC0 pp [CI] | EP mean | dEP [CI] | changed picks | unchecked points (picked plan / all candidates) |
|---|---|---|---|---|---|---|---|---|---|---|
| G0 (deployed pick) | 0.2022 | -- | 26.42 % | -- | 15.46 % | -- | 0.8388 | -- | 0 (0.0 %) | -- |
| G1 (own-map gate) | 0.2288 | +0.0266 [+0.0002, +0.0544] | 22.67 % | -3.76 [-5.27, -2.45] | 13.97 % | -1.49 [-2.27, -0.69] | 0.7738 | -0.0650 [-0.0754, -0.0544] | 2320 (39.2 %) | 6.7 % / 9.5 % |
| G1-DER (deranged mask) | 0.1753 | -0.0268 [-0.0604, +0.0079] | 35.39 % | +8.96 [+7.07, +10.91] | 13.11 % | -2.35 [-3.54, -1.25] | 0.6943 | -0.1445 [-0.1550, -0.1337] | 3888 (65.8 %) | 7.0 % / 9.5 % |
| G-ORC (GT drivable, PRIVILEGED) | 0.3162 | +0.1141 [+0.0843, +0.1437] | 14.45 % | -11.98 [-13.34, -10.52] | 14.78 % | -0.68 [-1.18, -0.17] | 0.8201 | -0.0187 [-0.0232, -0.0140] | 1326 (22.4 %) | 4.8 % / 9.5 % |

Seed floor (banked R7_A1 vs R7_A1_s1, same tokens): |dEPDMS| = 0.0065 (2x = 0.0131); |dDAC0| = 0.00 pp.

### P4-T2 Per stratum (P1' rule sets; per-token means; dDAC0 vs G0 paired log-cluster CI)

| arm | stratum | n | DAC0 | dDAC0 pp [CI] | NC0 | EP mean | per-token score mean | changed share |
|---|---|---|---|---|---|---|---|---|
| G0 (deployed pick) | S1 | 1140 | 100.00 % | -- | 15.35 % | 0.8761 | 0.0000 | 0.0 % |
| G0 (deployed pick) | S1b | 422 | 100.00 % | -- | 15.40 % | 1.0000 | 0.0000 | 0.0 % |
| G0 (deployed pick) | S2 | 914 | 26.26 % | -- | 100.00 % | 0.9800 | 0.0000 | 0.0 % |
| G0 (deployed pick) | rest | 3676 | 0.00 % | -- | 0.00 % | 0.7826 | 0.7052 | 0.0 % |
| G1 (own-map gate) | S1 | 1140 | 69.74 % | -30.26 [-34.34, -26.24] | 13.07 % | 0.7684 | 0.1825 | 55.6 % |
| G1 (own-map gate) | S1b | 422 | 91.71 % | -8.29 [-11.39, -5.64] | 11.85 % | 1.0000 | 0.0659 | 56.2 % |
| G1 (own-map gate) | S2 | 914 | 22.54 % | -3.72 [-6.20, -1.29] | 84.03 % | 0.9560 | 0.0755 | 41.8 % |
| G1 (own-map gate) | rest | 3676 | 3.78 % | +3.78 [+2.74, +4.84] | 1.03 % | 0.7156 | 0.6518 | 32.9 % |
| G1-DER (deranged mask) | S1 | 1140 | 70.18 % | -29.82 [-32.90, -26.68] | 11.84 % | 0.6940 | 0.1744 | 74.9 % |
| G1-DER (deranged mask) | S1b | 422 | 94.55 % | -5.45 [-7.32, -3.54] | 11.14 % | 1.0000 | 0.0434 | 74.4 % |
| G1-DER (deranged mask) | S2 | 914 | 36.43 % | +10.18 [+6.60, +14.03] | 68.60 % | 0.9223 | 0.1178 | 69.7 % |
| G1-DER (deranged mask) | rest | 3676 | 20.54 % | +20.54 [+17.83, +23.27] | 3.21 % | 0.6178 | 0.5175 | 61.6 % |
| G-ORC (GT drivable, PRIVILEGED) | S1 | 1140 | 42.46 % | -57.54 [-61.56, -53.64] | 12.98 % | 0.7892 | 0.3596 | 81.8 % |
| G-ORC (GT drivable, PRIVILEGED) | S1b | 422 | 86.97 % | -13.03 [-16.95, -8.97] | 11.85 % | 1.0000 | 0.0930 | 65.4 % |
| G-ORC (GT drivable, PRIVILEGED) | S2 | 914 | 15.43 % | -10.83 [-13.69, -7.87] | 91.25 % | 0.9749 | 0.0272 | 21.9 % |
| G-ORC (GT drivable, PRIVILEGED) | rest | 3676 | 0.08 % | +0.08 [+0.00, +0.23] | 0.14 % | 0.7796 | 0.7053 | 2.9 % |

### P4-T3 Gate statistics

| arm | deployed pick passes | fallback (no reach-kept candidate passes) | of which some candidate passes outside reach_keep |
|---|---|---|---|
| G1 (own-map gate) | 54.3 % | 379 | 0 |
| G1-DER (deranged mask) | 25.2 % | 532 | 0 |
| G-ORC (GT drivable, PRIVILEGED) | 75.9 % | 100 | 0 |

### P4-T4 Registered clauses (SPEC s4 literals)

```
{
 "G0_reproduces": true,
 "G1DER_delta": -0.026837610957282737,
 "G1DER_behaves(delta<=seed_floor)": true,
 "G1_clauses": {
  "dEPDMS_ge_0.020": true,
  "dEPDMS_ci_excludes_0": true,
  "dEPDMS_gt_2x_seed_floor": true,
  "DAC0_falls_ge_3pp": true,
  "DAC0_ci_excludes_0": true
 },
 "GORC_clauses_same_literals": {
  "dEPDMS_ge_0.020": true,
  "dEPDMS_ci_excludes_0": true,
  "dEPDMS_gt_2x_seed_floor": true,
  "DAC0_falls_ge_3pp": true,
  "DAC0_ci_excludes_0": true
 },
 "GORC_passes_same_literals": true,
 "verdict": "PASS",
 "committed_meaning": "PASS => the model's own map knows where it may drive and the selector ignores it: the gate enters refcv8's evaluation as an opt-in inference row on BOTH NavSim rows; a trained DAC-aware selector term becomes a named refcv8 lever."
}
```
