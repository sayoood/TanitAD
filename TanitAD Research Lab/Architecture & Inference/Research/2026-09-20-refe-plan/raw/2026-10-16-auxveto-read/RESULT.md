# RESULT: PREREG_AUXVETO read. Outcome row **AV-HARM**

> **Evidence class: MEASURED** (this read, `auxveto_read.py`, on the artifacts named in `AUXVETO_READ.json`). Tier: NAVSIM v1.1 non-reactive per-proposal table lookup, base = u400 (SFT-4 arm B).
> Prereg `eval/PREREG_AUXVETO.md` sha256 `ef9f17fa26a9d818…` (asserted by the script; landing commit `4f7e105`). Package `raw/2026-10-16-auxveto-read/`. NOT committed, NOT pushed.

## Headline

- **V-PRED (PRIMARY, A2 head, 10k updates, held-out IoU 0.675218): stratified navtest-population ΔPDMS (×100) = -0.788 [-1.496, -0.197].** The CI lies entirely below 0: `AV-HARM` fires on the unrounded numbers.
- **V-GT (ceiling of the pose-only raster route): +0.367 [+0.015, +0.638]**, lower bound 0.015132 > 0, so the prereg's AV-HARM branch reads *"the GT ceiling is positive"*. The margin of that is thin: the lower bound is +0.015 and the oracle O-DAC is +2.546 [+2.196, +2.880].
- **Validity: all five §9 checks PASS** (plus the population identity and the head-quality IoU identities). The read is VALID.
- The CI answers *another draw of logs* and is **blind to the head's training variance** (one head seed, H-ESTIM-SEED-1) and to the tracker.

## 1. The outcome, by the §6 decision list, applied mechanically to the unrounded numbers

```
V-PRED  point = -0.788243043366996
        lower = -1.4956563097381215
        upper = -0.19730761191846935
V-GT    point = 0.36702774726742615
        lower = 0.015132405135540827
        upper = 0.6384549778172475
```

| order | row | condition (prereg, verbatim) | evaluates to |
|---|---|---|---|
| 1 | **AV-GO** | V-PRED lower bound > 0 AND point ≥ +0.30 | lower -1.4957 > 0 is False |
| 2 | **AV-PROMISING** | V-PRED lower bound > 0 AND point < +0.30 | lower -1.4957 > 0 is False |
| 3 | **AV-HARM** **FIRES** | V-PRED upper bound < 0 | upper -0.1973 < 0 is True |
| 4 | **AV-CEILING-ONLY** | V-PRED not separated AND V-GT lower bound > 0 | not reached |
| 5 | **AV-NULL** | V-PRED not separated AND V-GT not separated above 0 | not reached |

**Row: AV-HARM.** V-GT branch (QUESTIONS.md Q7: "positive" = V-GT lower bound > 0): lower bound +0.015132 → *GT ceiling positive (V-GT lower bound > 0): the lever is raster quality (S1c)*. (The other reading, V-GT point > 0, is also True; both readings agree.)

**The row's committed next arm, verbatim from the prereg §6:**

> read V-GT: if the GT ceiling is positive, the lever is raster quality (S1c), else AV-NULL's next arm

## 2. Validity (§9). Any failure would make the read INVALID

| # | check | result | detail |
|---|---|---|---|
| 1 | B reproduces `pick` | PASS | 1058 / 1058 |
| 2 | V-PRED mask reproduces A2 IoU [0,40) 0.675218 ±1e-6 | PASS | 0.6752177378 (|diff| 2.62e-07) |
| 3 | zero control: all-drivable vetoes nothing, Δ = 0.0 exactly | PASS | pred grid: 0 vetoed, Δ 0.0; GT grid: 0 vetoed, Δ 0.0 |
| 4 | O-DAC's Z recovery ≥ every raster arm | PASS | O-DAC 343; V-PRED 30, V-GT 69, V-PRED-2k 28, V-PRED-400 66, V-GT-TRACKED 111 (recovery = Z tokens with NC×DAC > 0 at the pick, Q5) |
| 5 | y-flipped V-GT falls below V-GT's point | PASS | V-GT +0.3670; y-flipped -0.8704 |
| 0 | population identity (Z all NC×DAC = 0; N all > 0; 529 / 529) | PASS | Z 529, N 529 |
| 2b | 2k and 400-update masks vs `PRED_GEOMETRY.json` (reported) | PASS | 2k 0.634891 (banked 0.634891); 400 0.428047 (banked 0.428047) |

Test battery (written and run BEFORE any arm Δ was computed; re-run on the final script): `ZZTEST_PASS 43 22`; `3 passed in 162.36s (0:02:42)`. 22 deliberate-regression mutants, each turned a check RED (`test_auxveto_read.out`). Open details of the prereg were fixed in `QUESTIONS.md` BEFORE any outcome (15 items).

## 3. The arms: stratified navtest-population ΔPDMS (×100) vs B, and its two strata

Δ = (529/12,146)·Δ_Z + (11,617/12,146)·Δ_N. Paired log-cluster bootstrap, 123 logs, 10,000 resamples, seed 20260927, 95 % percentile CI. Margin 0.25 m.

| arm | role | **Δ PDMS (stratified) [95 % CI]** | Δ_Z / Δ_N | Z tokens recovered (NC×DAC > 0) of 529 | B's pick falsely vetoed on N (of 529) | every proposal vetoed (Z / N tokens) | picks changed (Z / N) |
|---|---|---|---|---|---|---|---|
| **V-PRED** | PRIMARY | **-0.788 [-1.496, -0.197]** | +4.69 / -1.04 | 30 | 61 | 73 / 32 | 66 / 29 |
| **V-GT** | ceiling, pose-only | **+0.367 [+0.015, +0.638]** | +11.81 / -0.15 | 69 | 2 | 12 / 0 | 147 / 2 |
| **V-PRED-2k** | head curve | **-0.543 [-1.231, +0.040]** | +4.31 / -0.76 | 28 | 55 | 74 / 37 | 51 / 18 |
| **V-PRED-400** | head curve | **-0.777 [-1.413, -0.250]** | +10.14 / -1.27 | 66 | 153 | 212 / 86 | 109 / 67 |
| **O-DAC** | oracle (never deployable) | **+2.546 [+2.196, +2.880]** | +58.45 / +0.00 | 343 | 0 | 68 / 0 | 348 / 0 |
| **V-GT-TRACKED** | reported only (§7) | **+0.794 [+0.628, +0.970]** | +18.23 / +0.00 | 111 | 0 | 15 / 0 | 236 / 0 |
| **V-GT-YFLIP** | control (§9.5) | **-0.870 [-1.615, -0.146]** | +7.66 / -1.26 | 52 | 217 | 277 / 149 | 87 / 68 |

## 4. The four metric families (§8). STRATEGIC is PAUSED

### LONGITUDINAL: Δ EP, Δ TTC (×100; stratified, paired CI)

| arm | Δ EP [CI] | Δ_Z / Δ_N | Δ TTC [CI] | Δ_Z / Δ_N |
|---|---|---|---|---|
| V-PRED | -1.120 [-1.889, -0.442] | +4.07 / -1.36 | -0.362 [-1.094, +0.351] | +0.00 / -0.38 |
| V-GT | +0.355 [+0.061, +0.595] | +10.44 / -0.10 | -0.131 [-0.533, +0.103] | +1.13 / -0.19 |
| V-PRED-2k | -0.605 [-1.292, -0.010] | +3.13 / -0.77 | -0.156 [-0.530, +0.048] | +0.57 / -0.19 |
| V-PRED-400 | -1.790 [-2.705, -1.021] | +7.80 / -2.23 | -0.657 [-1.409, -0.080] | +1.51 / -0.76 |
| O-DAC | +2.271 [+1.970, +2.555] | +52.14 / +0.00 | +0.074 [-0.011, +0.152] | +1.70 / +0.00 |
| V-GT-TRACKED | +0.683 [+0.537, +0.845] | +15.68 / +0.00 | +0.082 [+0.007, +0.157] | +1.89 / +0.00 |

### LATERAL: Δ DAC, Δ DDC, the Z-stratum recovery and the false-veto rate on N

| arm | Δ DAC [CI] | Δ_Z / Δ_N | Δ DDC [CI] | Z recovered: NC×DAC > 0 | (DAC = 1 only, POST-HOC variant) | false-veto rate on N |
|---|---|---|---|---|---|---|
| V-PRED | -0.673 [-1.496, +0.020] | +5.29 / -0.95 | -0.078 [-0.283, +0.041] | 30 / 529 = 5.7 % | 141 | 61 / 529 = 11.5 % |
| V-GT | +0.396 [-0.001, +0.694] | +13.23 / -0.19 | +0.049 [+0.009, +0.097] | 69 / 529 = 13.0 % | 183 | 2 / 529 = 0.4 % |
| V-PRED-2k | -0.517 [-1.215, +0.075] | +4.73 / -0.76 | +0.025 [+0.000, +0.056] | 28 / 529 = 5.3 % | 138 | 55 / 529 = 10.4 % |
| V-PRED-400 | +0.288 [-0.098, +0.655] | +10.78 / -0.19 | +0.037 [-0.517, +0.566] | 66 / 529 = 12.5 % | 170 | 153 / 529 = 28.9 % |
| O-DAC | +2.865 [+2.484, +3.223] | +65.78 / +0.00 | +0.086 [+0.024, +0.152] | 343 / 529 = 64.8 % | 461 | 0 / 529 = 0.0 % |
| V-GT-TRACKED | +0.914 [+0.714, +1.124] | +20.98 / +0.00 | +0.078 [+0.023, +0.138] | 111 / 529 = 21.0 % | 224 | 0 / 529 = 0.0 % |

### TACTICAL (selection quality): mean PDMS ×100 per stratum, Z / N / navtest-stratified [95 % CI]

| pick | Z / N / stratified [CI] |
|---|---|
| random proposal (expected over the 64) | 29.89 / 75.35 / 73.37 [71.42, 75.34] |
| B (u400 pick) | 0.00 / 92.30 / 88.28 [87.35, 89.11] |
| best of 64 (oracle max PDMS) | 83.12 / 98.84 / 98.16 [97.79, 98.54] |
| V-PRED | 4.69 / 91.26 / 87.49 [86.25, 88.60] |
| V-GT | 11.81 / 92.14 / 88.64 [87.58, 89.57] |
| V-PRED-2k | 4.31 / 91.53 / 87.73 [86.51, 88.85] |
| V-PRED-400 | 10.14 / 91.02 / 87.50 [86.27, 88.56] |
| O-DAC | 58.45 / 92.30 / 90.82 [89.85, 91.73] |
| V-GT-TRACKED | 18.23 / 92.30 / 89.07 [88.17, 89.89] |

### STRATEGIC

**PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` ITEM 25: the strategic layer is switched off until tactical and operative drive well). Not a pass, not omitted.

### Also reported (§8): Δ NC, Δ comfort, the all-vetoed share

| arm | Δ NC [CI] | Δ comfort [CI] | tokens with EVERY proposal vetoed, Z (of 529) / N (of 529) | proposals vetoed, Z / N (of 33,856 each) |
|---|---|---|---|---|
| V-PRED | -0.152 [-0.573, +0.058] | +0.000 [+0.000, +0.000] | 73 / 32 | 12495 / 5839 |
| V-GT | -0.004 [-0.032, +0.021] | +0.000 [+0.000, +0.000] | 12 / 0 | 11526 / 2286 |
| V-PRED-2k | -0.082 [-0.299, +0.046] | +0.000 [+0.000, +0.000] | 74 / 37 | 11207 / 4907 |
| V-PRED-400 | +0.078 [+0.026, +0.148] | +0.000 [+0.000, +0.000] | 212 / 86 | 22316 / 11225 |
| O-DAC | +0.004 [-0.027, +0.032] | +0.000 [+0.000, +0.000] | 68 / 0 | 18716 / 3727 |
| V-GT-TRACKED | +0.008 [-0.015, +0.030] | +0.000 [+0.000, +0.000] | 15 / 0 | 15264 / 2607 |

### Per-city split (descriptive, never gating; Q13). ΔPDMS ×100: Z-stratum mean / N-stratum mean / city-stratified; Z recovered

| city | n_Z / n_N / navtest tokens | V-PRED | V-GT | V-PRED-2k | V-PRED-400 |
|---|---|---|---|---|---|
| sg-one-north | 159 / 80 / 1914 | +5.52 / -2.51 / **-1.85**; rec 11 | +7.59 / -1.07 / **-0.35**; rec 14 | +6.32 / -2.56 / **-1.82**; rec 13 | +24.36 / -4.43 / **-2.04**; rec 48 |
| us-ma-boston | 146 / 164 / 3731 | +3.26 / -0.63 / **-0.48**; rec 6 | +16.82 / +0.03 / **+0.68**; rec 27 | +1.28 / -0.04 / **+0.01**; rec 2 | +3.90 / -1.05 / **-0.85**; rec 7 |
| us-nv-las-vegas-strip | 62 / 171 / 4064 | +4.84 / -0.61 / **-0.52**; rec 3 | +4.17 / +0.00 / **+0.06**; rec 3 | +0.00 / +0.00 / **+0.00**; rec 0 | +1.61 / -0.34 / **-0.31**; rec 1 |
| us-pa-pittsburgh-hazelwood | 162 / 114 / 2437 | +5.11 / -1.24 / **-0.82**; rec 10 | +14.36 / +0.00 / **+0.95**; rec 25 | +6.73 / -1.68 / **-1.12**; rec 13 | +5.07 / -0.79 / **-0.40**; rec 10 |

## 5. Reported, never gating: the margin curve and the head-quality curve

### Margin curve (veto iff signed distance < −m). Primary m = 0.25 m is the prereg's, fixed before any outcome

| arm | m (m) | Δ PDMS [CI] | Δ_Z / Δ_N | Z recovered | B falsely vetoed (N) | all-vetoed tokens Z / N |
|---|---|---|---|---|---|---|
| V-PRED | 0 | -0.186 [-0.698, +0.216] | +6.87 / -0.51 | 44 | 75 | 88 / 43 |
| V-PRED | 0.25 (primary) | -0.788 [-1.496, -0.197] | +4.69 / -1.04 | 30 | 61 | 73 / 32 |
| V-PRED | 0.5 | -0.270 [-0.791, +0.158] | +3.75 / -0.45 | 23 | 53 | 56 / 29 |
| V-PRED | 1 | -0.478 [-1.362, +0.110] | +2.78 / -0.63 | 17 | 30 | 40 / 14 |
| V-PRED | 2 | +0.015 [-0.070, +0.091] | +1.86 / -0.07 | 12 | 12 | 21 / 7 |
| V-GT | 0 | +0.888 [+0.513, +1.208] | +22.88 / -0.11 | 133 | 24 | 35 / 8 |
| V-GT | 0.25 (primary) | +0.367 [+0.015, +0.638] | +11.81 / -0.15 | 69 | 2 | 12 / 0 |
| V-GT | 0.5 | +0.165 [+0.002, +0.312] | +4.92 / -0.05 | 28 | 2 | 7 / 0 |
| V-GT | 1 | +0.050 [+0.007, +0.110] | +1.16 / +0.00 | 7 | 0 | 4 / 0 |
| V-GT | 2 | +0.008 [+0.000, +0.024] | +0.17 / +0.00 | 1 | 0 | 0 / 0 |

### Head-quality curve (the same head recipe at 400 / 2,000 / 10,000 updates, held-out)

| updates | held-out IoU [0,40) | Δ PDMS [CI] | Δ_Z / Δ_N | Z recovered | share of proposals with margin < −0.25 m |
|---|---|---|---|---|---|
| 400 | 0.4280 | -0.777 [-1.413, -0.250] | +10.14 / -1.27 | 66 | 49.5 % |
| 2,000 | 0.6349 | -0.543 [-1.231, +0.040] | +4.31 / -0.76 | 28 | 23.8 % |
| 10,000 | 0.6752 | -0.788 [-1.496, -0.197] | +4.69 / -1.04 | 30 | 27.1 % |
| GT (V-GT) | 1.0000 | +0.367 [+0.015, +0.638] | +11.81 / -0.15 | 69 | 20.4 % |

## 6. What the interval answers, and what it cannot

- **Estimator:** stratified (Z census 529 / 12,146 = 0.043553; N sample of 529 from 11,617 / 12,146 = 0.956447), paired log-cluster bootstrap over 123 C3 logs, 10,000 resamples, seed 20260927, weights fixed, 95 % percentile CI.
- **Its question:** *another draw of logs.* ⛔ It is **blind to the head's TRAINING variance** (the A2 head has one seed; H-ESTIM-SEED-1), and to the NAVSIM tracker. Raster inference variance is zero (deterministic).
- **Consequence for this row:** V-PRED's upper bound (-0.197) is below 0 for *this* head. A seed-1 head replicate is the one thing that could say whether −0.788 is a property of the recipe or of seed 0; the 400 / 2k / 10k points (-0.777 / -0.543 / -0.788) are three different trainings of one seed and are already not monotone.
- **Wall time:** drawing the 10,000 × 123 resample matrix 0.054 s; all 20 bootstrap evaluations 0.215 s; the whole read (gates, 7 arms, curves, bootstrap) 22.0 s, CPU only.
- **Transfer caveat (prereg §2):** the base is u400, not P2 or P6. This is a statement about the u400 scorer's failure set on C3, re-weighted to navtest. NAVSIM v1.1 non-reactive.

## 7. POST-HOC rows (NOT in the prereg; decide nothing)

- **Ties on the threshold.** On the 0.5 m GT grid a one-cell-wide non-drivable sliver has signed distance exactly −0.25 at its cells, so 200 V-GT proposals (32 tokens) sit exactly on the strict `< −0.25` threshold and are kept. Vetoing them instead (`≤`) moves V-GT from +0.367 [+0.015, +0.638] to +0.390 [+0.039, +0.669]: same sign, same branch. V-PRED has 0 ties. (`POSTHOC_TIE_SENSITIVITY.json`)
- **Pipeline cross-check.** V-GT's margins from the mask-derived signed distance agree with the label store's stored int8 signed distance to ≤ 0.050 m (the store's 0.1 m quantisation) on all 67,712 proposals; at m = 0.25 the veto sets differ on 916 proposals (1.35 %) and the picks on 34 tokens, which is what a ≤ 0.05 m quantisation of the field does to margins that cluster near the 0.25 m threshold. (`CROSSCHECK_VGT_STORED_SDF.json`)
- **Predicted-grid path cross-check.** The GT mask down-sampled to the head's 2 m grid, pushed through the predicted-grid path (2 m EDT, `PRED_GEOM`, bilinear), agrees with NAVSIM's DAC = 0 at the proposal level with κ 0.647 (0.5 m GT raster: κ 0.633); the same raster y-flipped reads κ 0.303, shifted one cell in x κ 0.472. So an axis or half-cell error in the 2 m path would have shown. (`POSTHOC_PRED_FRAME_CROSSCHECK.json`)

## 8. Reading (the read agent's; the mechanical decision above does not depend on it)

1. **At today's head quality the veto trades 30 recoveries (of 529 zeroed picks, Δ_Z +4.69) for 61 false vetoes of good picks (of 529, Δ_N -1.04),** and the net is negative. The DAC sub-score at the pick does not rise (-0.673 [-1.496, +0.020], the CI reaches 0): the head's raster is not a better judge of NAVSIM's DAC than the scorer's own picks.
2. **The head-quality curve does not rescue it.** 400 → 2k → 10k updates (IoU 0.428 → 0.635 → 0.675) give -0.777 / -0.543 / -0.788. The 400-update head vetoes 50 % of all proposals; its Z gain (+10.14) is consistent with the generic gain of moving ANY zeroed pick, whose B score is 0 by selection (the y-flipped GT arm also reads Δ_Z +7.66), and it is paid for in N.
3. **The ceiling of the pose-only route is small.** A perfect raster (V-GT) reads +0.367 [+0.015, +0.638], which is 14 % of the oracle DAC veto (+2.546 [+2.196, +2.880]); the same GT raster on NAVSIM's tracked states reads +0.794 [+0.628, +0.970] (reported only, §7). HYPOTHESIS for the gap V-GT → O-DAC: mostly the tracker (INHERITED support, not re-measured here: `AGREEMENT.json` finds 27.8 % of NAVSIM's DAC failures only in the tracked states, and V-GT-TRACKED is higher than V-GT); the gap V-PRED → V-GT is 1.16 PDMS. So the prereg's committed branch ("the lever is raster quality (S1c)") has a measured pose-only reference of about +0.37 PDMS at the prereg's margin: a head that only matched the GT raster would not be expected to read more under this rule. The margin curve says V-GT at m = 0 reads +0.888 [+0.513, +1.208] (descriptive; not a tuned setting).

## 9. Deliverable manifest (all under `raw/2026-10-16-auxveto-read/`; **single copy in the temp archive tree, not committed, not pushed**)

| file | what | where it lives |
|---|---|---|
| `auxveto_read.py` | the read: §3-§9 as code; `--gate`, `--run` | avr tree |
| `test_auxveto_read.py` | 43 checks incl. every §9 check + 22 deliberate-regression mutants | avr tree |
| `test_auxveto_read.out / test_auxveto_read_pytest.out` | test evidence on the final script | avr tree |
| `QUESTIONS.md` | 15 prereg details fixed BEFORE any outcome | avr tree |
| `GATE.json` | pre-read gates (checks 1, 2, 3) | avr tree |
| `AUXVETO_READ.json` | every arm, both strata, the stratified estimate + CI, §8 families, margin curve, head-quality curve | avr tree |
| `AUXVETO_PICKS.npz` | per-token picks and min margins of every arm (audit) | avr tree |
| `auxveto_read_run.log` | stdout of the run | avr tree |
| `RESULT.md / write_result.py` | this file and its generator | avr tree |
| `crosscheck_vgt_stored_sdf.py / CROSSCHECK_VGT_STORED_SDF.json` | POST-HOC pipeline cross-check | avr tree |
| `posthoc_tie_sensitivity.py / POSTHOC_TIE_SENSITIVITY.json` | POST-HOC tie sensitivity | avr tree |
| `posthoc_pred_frame_crosscheck.py / POSTHOC_PRED_FRAME_CROSSCHECK.json` | POST-HOC predicted-grid path cross-check | avr tree |
| `MANIFEST.sha256` | sha256 of every file above | avr tree |

Tree: `C:/Users/Admin/AppData/Local/Temp/avr/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-16-auxveto-read/`. The tree is the `git archive` of 4f7e105 plus `refe/` extracted from the same commit (the instrument imports `refe/aux_label_store.py`). Inputs used in place (not copied): `raw/2026-10-11-c3-selection-headroom/{tokens.json,table.npz}`, the label store `D:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest`, the head units under `D:/Projects/TanitAD-artifacts/scene_tap_probe{,_a2,_a3r}/units`, and the regenerable tracker shards `C:/Users/Admin/tmpauxv/tracked/` (sha256 in `AUXVETO_READ.json`).
