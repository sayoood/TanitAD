# PREREG_V8F_LISTEN2: result (executed 2026-10-25; zero training, CPU, M2b s0 and s1 banked checkpoints)

**Evidence class:** MEASURED (ours). Artifacts (this directory): `READ.json` (every number below), `READ_controls.json`, `runs/*.npz` (per-window arrays), `runs/means_*.npz` (the training-set means). **Tier of every number:** T1-class open loop, DIAGNOSTIC (donor and mean inputs are interventions on a fixed checkpoint, not driving). **Estimator of every interval:** paired episode-cluster bootstrap, B = 2,000, seed 0 (`taniteval/ci.py`, through the eval's own `paired` and macro-F1 reducer), 141 episodes on the O row (S 137, T-lat 139, T-lon 136: the label masks differ per row, as in LISTEN). **The question it answers:** "would another draw of EPISODES say this?". It does NOT answer "would another TRAINING RUN say this?" (the two banked training seeds, M2b s0 and M2b s1, are shown side by side for that; two seeds are not a seed-noise floor) and not "would another INFERENCE RUN say this?" (the planner is deterministic on CPU, so there is no inference variance). The PREREG is `PREREG_V8F_LISTEN2.md` (`prereg_commit` 067c8b74); nothing was staged or committed.

Sign convention (as in LISTEN): S and O rows are replaced minus clean in metres (larger is worse); T rows are clean minus replaced macro-F1 (a drop, larger is worse). "Separated" = the 95 % interval excludes 0.

## 0. The three outcomes (the committed tables, applied verbatim)

| experiment | outcome | condition as committed | what was measured |
|---|---|---|---|
| **E1 (PRIMARY)** route through the chain | **ROUTE FLOWS DOWN** | E1 > 0 and separated at BOTH seeds and above the donor-seed floor | T-lat F1 drop, donor nav at S only: s0 +0.442 [+0.353, +0.507], s1 +0.419 [+0.349, +0.483]; floor 0.0825. **Bar met at both seeds.** |
| **E2** is the grid's ordering a donor-swap artefact? | **PARTLY ROBUST** | ROBUST if all 4 rows agree at both seeds, PARTLY ROBUST if 2 to 3, NOT ROBUST if fewer | 3 of 4 rows agree at both seeds (S = nav, T-lat = S checkpoints, T-lon = ego); **the O row does not** (mean replacement ranks T code above ego at both seeds). **The top outcome, ROBUST, was MISSED.** |
| **E3** O's coherence | **INCOHERENCE** | ADE6(joint triple) - max single swap < 0 and separated at both seeds | s0 -3.835 [-4.959, -2.766] m, s1 -3.174 [-4.234, -2.221] m. **Bar met at both seeds.** |

Committed meaning, E1 ROUTE FLOWS DOWN (quoted): "T's lateral decision follows the route S derived from nav: nav -> S -> T is a working chain". Committed meaning, E3 INCOHERENCE (quoted): "O reacts to the CONTRADICTION between its inputs more than to a coherent foreign state". E2 PARTLY ROBUST carries the committed rule "the rows that disagree are reported with their cells" (section 2). **All controls pass** (section 4), so the reading is reported. The readings M1 to M7 the PREREG text left open (section 5) were written into `listen2.py` / `read2.py` before any LISTEN2 number was computed.

## 1. E1 (PRIMARY): does the route flow down? Donor nav injected at S only

**Intervention.** The donor window's nav token replaces S's own nav token; S's checkpoints therefore change; T and O then read S's checkpoints as usual, with their OWN nav, ego and limit tokens, O's age token and both scene paths untouched. This is the only path by which nav-at-S can reach T-lateral. The module asserts this per window with `torch.equal` (S's nav token is the donor's; T's and O's own tokens and scene paths are the clean pass's bit for bit): **asserted on 2,399 / 2,399 windows in each of the five donor runs** (s0 donor 0, s1 donor 0, s0 donor 1, and both identity runs).

### E1, per seed (read: T-lat F1 drop)

| seed | donor | T-lat clean F1 | T-lat F1 with donor nav at S only | **E1 = drop [95 % CI]** | n windows / episodes | separated | > 0 | above donor-seed floor (0.0825) | bar met |
|---|---|---|---|---|---|---|---|---|---|
| s0 | seed 0 | 0.8157 | 0.3738 | **+0.4419 [+0.3530, +0.5068]** | 2107 / 139 | yes | yes | yes | **yes** |
| s1 | seed 0 | 0.8292 | 0.4107 | **+0.4186 [+0.3489, +0.4827]** | 2107 / 139 | yes | yes | yes | **yes** |
| s0 | seed 1 (floor run) | 0.8157 | 0.4563 | +0.3594 [+0.2657, +0.4455] | 2107 / 139 | yes | yes | n/a | n/a |

**Donor-seed floor (M6).** |E1(s0, donor 0) - E1(s0, donor 1)| = |0.4419 - 0.3594| = **0.0825 F1**, applied to both seeds. E1 is 5.4 x (s0) and 5.1 x (s1) that floor; the second donor draw also separates on its own (+0.3594 [+0.2657, +0.4455]).

### Beside (descriptive, all from the same donor-nav-at-S arm; the PREREG marks these "descriptive")

| row (metric) | seed | clean level | change with donor nav at S only [95 % CI] | n windows / episodes | separated |
|---|---|---|---|---|---|
| S: checkpoint position error @ 50 m, replaced - clean (m) | s0 | 8.0122 | +4.1965 m [+2.9873, +5.5222] | 2319 / 137 | yes |
| S: checkpoint position error @ 50 m, replaced - clean (m) | s1 | 7.9384 | +3.9851 m [+2.8064, +5.1818] | 2319 / 137 | yes |
| T-lat: lateral macro-F1, clean - replaced | s0 | 0.8157 | +0.4419 F1 [+0.3530, +0.5068] | 2107 / 139 | yes |
| T-lat: lateral macro-F1, clean - replaced | s1 | 0.8292 | +0.4186 F1 [+0.3489, +0.4827] | 2107 / 139 | yes |
| T-lon: longitudinal macro-F1, clean - replaced | s0 | 0.4992 | +0.0057 F1 [-0.0377, +0.0491] | 2240 / 136 | no |
| T-lon: longitudinal macro-F1, clean - replaced | s1 | 0.5095 | +0.0050 F1 [-0.0447, +0.0437] | 2240 / 136 | no |
| O: ADE6, replaced - clean (m) | s0 | 2.1753 | +0.7264 m [+0.4259, +1.0812] | 2399 / 141 | yes |
| O: ADE6, replaced - clean (m) | s1 | 2.0546 | +0.8214 m [+0.5087, +1.1724] | 2399 / 141 | yes |
| O: ADE2 (beside), replaced - clean (m) | s0 | 0.3155 | +0.0306 m [+0.0173, +0.0464] | 2399 / 141 | yes |
| O: ADE2 (beside), replaced - clean (m) | s1 | 0.2914 | +0.0358 m [+0.0215, +0.0513] | 2399 / 141 | yes |

The fraction E1 / (T-lat x S-checkpoints swap), the PREREG's third "beside" read, with LISTEN's whole-checkpoint swap recomputed in the same run:

| seed | E1 (T-lat, nav at S only) | T-lat x S checkpoints (whole-checkpoint donor swap, recomputed) | fraction E1 / swap | T-lon: E1 | T-lon x S checkpoints | fraction (T-lon) |
|---|---|---|---|---|---|---|
| s0 | +0.442 [+0.353, +0.507] | +0.391 [+0.303, +0.467] | **1.13** | +0.006 [-0.038, +0.049] (not separated) | +0.041 [+0.003, +0.070] | 0.14 (both numerators are within noise: read as nothing) |
| s1 | +0.419 [+0.349, +0.483] | +0.406 [+0.337, +0.464] | **1.03** | +0.005 [-0.045, +0.044] (not separated) | +0.013 [-0.057, +0.081] (not separated) | 0.39 (both numerators are within noise: read as nothing) |

**What E1 says.** Injecting a foreign nav at S alone, with T's own nav untouched, takes T-lateral macro-F1 from 0.816 to 0.374 (s0) and from 0.829 to 0.411 (s1), separated at both seeds and several times above the donor-seed floor. By the committed table this is **ROUTE FLOWS DOWN**: T's lateral decision follows what S derives from nav, so nav -> S -> T is a working chain for the lateral decision, and the "PARALLEL PATH" narrowing of the hierarchy claim (to "S -> O" and "nav -> T") is **not** supported.

Beside it (descriptive, post hoc, not promoted): (a) the donor-nav-at-S arm moves T-lat by as much as the whole-checkpoint swap (fraction 1.13 / 1.03; the two intervals overlap at both seeds, so "at least as much" is a point-estimate reading, not a separated difference); so in this read the whole-checkpoint swap adds no separable effect beyond nav-at-S alone. (b) **T-longitudinal does not follow the route through S**: E1 on T-lon is +0.006 [-0.038, +0.049] F1 and +0.005 [-0.045, +0.044] F1, not separated at either seed (T-lon is carried by ego and the scene in LISTEN). (c) O's ADE6 under this arm is +0.726 [+0.426, +1.081] m (s0) and +0.821 [+0.509, +1.172] m (s1), which reproduces LISTEN's `nav@S` arm (+0.726 / +0.821 m; recomputed difference 6.7e-06 / 3.6e-06 m max over windows).

## 2. E2: is the grid's ordering an artefact of donor swapping? Training-set mean replacement

**Intervention.** Each input, one at a time, is replaced by the mean of that input over HBLEND's 9,582 training windows (M1: the input as the layer reads it at inference, at that layer's own tick; mean scene taken over the un-padded frames; mean ages as floats). The replacement goes through the same exactly-once shadow mechanism as LISTEN's donors. Mean-pass control: 9582 windows, window list identical to HBLEND's fit windows, traj30 max |diff| vs HBLEND 2.7e-05 m (s0) and 3.8e-05 m (s1).

### E2 verdict, per row (M3: largest SIGNED mean-replacement effect; the grid's largest-n input is LISTEN's, re-derived from LISTEN's `READ.json` and asserted equal to the PREREG)

| row | seed | grid's largest-n input | largest effect under mean replacement | its delta [95 % CI] | runner-up (delta) | intervals overlap | **agrees** | same cell as the grid's, under mean replacement | the grid's cell under donor swap (LISTEN) |
|---|---|---|---|---|---|---|---|---|---|
| S | s0 | nav | **nav** | +2.176 m [+1.209, +3.244] | scene (+1.773) | yes | **yes** | +2.176 [+1.209, +3.244] | +4.197 [+2.987, +5.522] |
| S | s1 | nav | **nav** | +2.140 m [+1.248, +3.119] | scene (+1.927) | yes | **yes** | +2.140 [+1.248, +3.119] | +3.985 [+2.806, +5.182] |
| T-lat | s0 | S checkpoints | **S checkpoints** | +0.363 F1 [+0.266, +0.475] | nav (+0.218) | yes | **yes** | +0.363 [+0.266, +0.475] | +0.391 [+0.303, +0.467] |
| T-lat | s1 | S checkpoints | **S checkpoints** | +0.329 F1 [+0.249, +0.406] | nav (+0.110) | no | **yes** | +0.329 [+0.249, +0.406] | +0.406 [+0.337, +0.464] |
| T-lon | s0 | ego | **ego** | +0.227 F1 [+0.163, +0.281] | scene (+0.150) | yes | **yes** | +0.227 [+0.163, +0.281] | +0.247 [+0.188, +0.305] |
| T-lon | s1 | ego | **ego** | +0.252 F1 [+0.156, +0.317] | scene (+0.216) | yes | **yes** | +0.252 [+0.156, +0.317] | +0.261 [+0.171, +0.328] |
| O | s0 | ego | **T code** | +4.146 m [+2.864, +5.535] | ego (+3.178) | yes | **no** | +3.178 [+2.740, +3.638] | +5.697 [+4.489, +6.979] |
| O | s1 | ego | **T code** | +3.181 m [+2.164, +4.299] | ego (+2.305) | yes | **no** | +2.305 [+1.972, +2.682] | +4.716 [+3.722, +5.806] |

**Rows agreeing at BOTH seeds (M4): S yes, T-lat yes, T-lon yes, O no -> k = 3 of 4 -> PARTLY ROBUST.** Per seed: s0 3 of 4, s1 3 of 4. ROBUST (all four rows) was the strongest committed outcome and it was **missed**: the only disagreement is the O row, where mean replacement ranks **T's code** above ego at both seeds (s0 +4.15 [+2.86, +5.53] m vs ego +3.18; s1 +3.18 [+2.16, +4.30] m vs ego +2.31). On O, ego and T code were already the two large cells with overlapping intervals in LISTEN, and they still overlap here at both seeds, so the swap of rank 1 and 2 on that row is within the interval width; the committed rule counts it as a disagreement regardless. The three agreeing rows pick their input at both seeds, and in S and T-lon the runner-up intervals overlap the winner (T-lat at s1 is the one row whose winner is clearly separated from its runner-up).

### E2, all cells (mean replacement; delta [95 % CI]; T rows in F1, S and O rows in m)

**Seed s0** (clean levels: S 8.012; T-lat 0.816; T-lon 0.499; O 2.175)

| row | scene | ego | nav | limit | S checkpoints | T code | tick ages | ALL inputs |
|---|---|---|---|---|---|---|---|---|
| S | +1.773 [+1.242, +2.348] | +1.226 [+0.805, +1.653] | **+2.176 [+1.209, +3.244]** | +0.002 [-0.027, +0.030] (ns) | not an input | not an input | not an input | +3.911 [+2.831, +5.055] |
| T-lat | -0.026 [-0.078, +0.009] (ns) | +0.032 [-0.031, +0.089] (ns) | +0.218 [+0.115, +0.320] | -0.030 [-0.073, -0.007] | **+0.363 [+0.266, +0.475]** | not an input | not an input | +0.499 [+0.420, +0.556] |
| T-lon | +0.150 [+0.087, +0.208] | **+0.227 [+0.163, +0.281]** | +0.084 [+0.043, +0.119] | +0.061 [+0.012, +0.111] | +0.020 [-0.007, +0.049] (ns) | not an input | not an input | +0.428 [+0.357, +0.489] |
| O | +0.152 [+0.093, +0.216] | +3.178 [+2.740, +3.638] | +0.023 [+0.013, +0.032] | +0.459 [+0.304, +0.632] | +0.158 [+0.106, +0.217] | **+4.146 [+2.864, +5.535]** | +0.005 [+0.001, +0.009] | +0.331 [+0.182, +0.484] |

**Seed s1** (clean levels: S 7.938; T-lat 0.829; T-lon 0.510; O 2.055)

| row | scene | ego | nav | limit | S checkpoints | T code | tick ages | ALL inputs |
|---|---|---|---|---|---|---|---|---|
| S | +1.927 [+1.347, +2.591] | +1.000 [+0.622, +1.417] | **+2.140 [+1.248, +3.119]** | +0.015 [-0.048, +0.110] (ns) | not an input | not an input | not an input | +3.971 [+2.850, +5.166] |
| T-lat | -0.004 [-0.038, +0.027] (ns) | +0.059 [+0.013, +0.106] | +0.110 [+0.043, +0.185] | -0.027 [-0.069, -0.002] | **+0.329 [+0.249, +0.406]** | not an input | not an input | +0.512 [+0.444, +0.564] |
| T-lon | +0.216 [+0.115, +0.287] | **+0.252 [+0.156, +0.317]** | +0.026 [-0.029, +0.071] (ns) | +0.139 [+0.021, +0.196] | -0.044 [-0.094, -0.007] | not an input | not an input | +0.439 [+0.341, +0.506] |
| O | -0.011 [-0.029, +0.005] (ns) | +2.305 [+1.972, +2.682] | +0.006 [-0.002, +0.015] (ns) | +0.298 [+0.209, +0.396] | +0.498 [+0.339, +0.687] | **+3.181 [+2.164, +4.299]** | +0.009 [+0.003, +0.015] | +0.528 [+0.376, +0.675] |

(ns) = not separated. Bold = the row's largest signed effect (the E2 argmax).

**Beside (descriptive, post hoc, not promoted).** (a) The O row's n > 1 from LISTEN survives without any donor: replacing ALL of O's inputs by their means costs +0.33 [+0.18, +0.48] m (s0) and +0.53 [+0.38, +0.68] m (s1), while replacing ego alone costs +3.18 [+2.74, +3.64] / +2.31 [+1.97, +2.68] m and T's code alone +4.15 [+2.86, +5.53] / +3.18 [+2.16, +4.30] m. So that excess is not a property of donor swapping. Caveat on the wording "no cross-input conflict": a mean token has no foreign window's content, but it still disagrees with the window's own other tokens (for O, the window's own kinematic prior and scene), so single-input mean cells are not conflict-free; only the ALL cell is. (b) Two sign-reversed cells, both separated: T-lat x limit (-0.030 / -0.027 F1) and, at s1 only, T-lon x S checkpoints (-0.044 F1): replacing that input by its mean slightly helps the macro-F1. They are small and not claims.

## 3. E3: O's coherence. Ego + S checkpoints + T code from ONE donor, together

**Intervention.** O receives the donor window's ego token, S checkpoints and T code together (`run_O(.., ('ego','ck','code'))`); the kinematic prior keeps the window's own `p10f` (asserted per window), and scene, nav, limit and tick ages stay O's own. **Comparator (M5):** the single swap with the largest MEAN ADE6 on the full set.

| seed | donor | mean ADE6 clean | joint triple | ego only | S checkpoints only | T code only | comparator (largest-mean single) | **E3 = joint - comparator [95 % CI]** | separated | < 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| s0 | seed 0 | 2.175 | 4.038 | 7.873 | 3.048 | 7.571 | ego | **-3.835 m [-4.959, -2.766]** | yes | yes |
| s1 | seed 0 | 2.055 | 3.596 | 6.770 | 3.949 | 6.187 | ego | **-3.174 m [-4.234, -2.221]** | yes | yes |
| s0 (floor run) | seed 1 | 2.175 | 3.922 | 8.182 | 2.892 | 7.712 | ego | **-4.261 m [-5.439, -3.198]** | yes | yes |

### The joint triple against the clean pass and against each single swap (ADE6, metres; paired; n = 2,399 windows / 141 episodes)

| seed | joint - clean | ego - clean | S ckpt - clean | T code - clean | joint - ego | joint - S ckpt | joint - T code | joint - per-window max of the three (DESCRIPTIVE, upward-biased comparator, M5) |
|---|---|---|---|---|---|---|---|---|
| s0 | +1.862 [+1.613, +2.128] | +5.697 [+4.489, +6.979] | +0.873 [+0.596, +1.193] | +5.395 [+4.210, +6.629] | -3.835 [-4.959, -2.766] | +0.990 [+0.601, +1.362] | -3.533 [-4.777, -2.370] | -8.241 [-9.543, -6.943] |
| s1 | +1.542 [+1.339, +1.733] | +4.716 [+3.722, +5.806] | +1.894 [+1.432, +2.397] | +4.132 [+3.138, +5.135] | -3.174 [-4.234, -2.221] | -0.353 [-0.858, +0.112] (ns) | -2.590 [-3.621, -1.597] | -6.575 [-7.727, -5.405] |

**What E3 says.** At both seeds the joint triple hurts O by 1.86 m (s0) and 1.54 m (s1) over clean, while the worst single swap (ego) hurts by 5.70 and 4.72 m; the difference, -3.83 [-4.96, -2.77] and -3.17 [-4.23, -2.22] m, is below 0 and separated at both seeds, with the second donor draw at s0 agreeing in sign and size. By the committed table this is **INCOHERENCE**: handing O a state from one donor in which three inputs agree is much less harmful than handing it that donor's ego alone, i.e. O reacts to the contradiction between its inputs more than to a coherent foreign state.

Limits on that reading, stated as they are (descriptive, not promoted): (a) the joint triple is not harmless: it is still separated from clean at both seeds, and it is WORSE than the S-checkpoint single swap at s0 (+0.99 [+0.60, +1.36] m, separated) while not separated from it at s1 (-0.35 [-0.86, +0.11] m). (b) "Coherent" is relative: the prior's `p10f`, scene, nav, limit and ages stay the window's OWN, so the triple is coherent among its three members, not with the rest of O's inputs. (c) The committed comparator is the WORST single swap; the bar is therefore met by any joint that is better than ego alone, which is a weaker statement than "the joint is better than every single swap".

## 4. Controls (all pass; the read was not stopped)

| control | committed condition | result | pass |
|---|---|---|---|
| 1. identity donor, s0 | gives exactly 0.0 | 8 non-clean rows bit-identical to the clean pass (max abs diff 0.0); 18 read cells, max abs delta 0.0, 0 separated | yes |
| 1. identity donor, s1 | gives exactly 0.0 | 8 non-clean rows bit-identical to the clean pass (max abs diff 0.0); 18 read cells, max abs delta 0.0, 0 separated | yes |
| 2. clean pass vs LISTEN's clean levels, s0 | <= 1e-4 | max abs level diff 1.1e-08 (S 0.0e+00; T-lat 0.0e+00; T-lon 0.0e+00; O 1.1e-08); per-window arrays vs LISTEN's clean arrays (float32 noise, not the committed control): largest 1.1e-04 on `clean/full:cons_pred`, all others <= 3.1e-05 | yes |
| 2. clean pass vs LISTEN's clean levels, s1 | <= 1e-4 | max abs level diff 9.5e-07 (S 9.5e-07; T-lat 0.0e+00; T-lon 0.0e+00; O 6.4e-09); per-window arrays vs LISTEN's clean arrays (float32 noise, not the committed control): largest 1.2e-04 on `clean/full:cons_pred`, all others <= 4.6e-05 | yes |
| 3. mean replacement with the window's OWN value, s0 | gives exactly 0.0 | 19 replacement rows bit-identical to the clean pass (max abs diff 0.0); 33 read cells, max abs delta 0.0, 0 separated | yes |
| 3. mean replacement with the window's OWN value, s1 | gives exactly 0.0 | 19 replacement rows bit-identical to the clean pass (max abs diff 0.0); 33 read cells, max abs delta 0.0, 0 separated | yes |
| 4. E1 invariance, run `s0` | T's and O's own nav tokens bit-identical to clean, asserted per window | 2399 / 2399 windows asserted with `torch.equal` | yes |
| 4. E1 invariance, run `s1` | T's and O's own nav tokens bit-identical to clean, asserted per window | 2399 / 2399 windows asserted with `torch.equal` | yes |
| 4. E1 invariance, run `s0_d1` | T's and O's own nav tokens bit-identical to clean, asserted per window | 2399 / 2399 windows asserted with `torch.equal` | yes |
| 4. E1 invariance, run `s0_id` | T's and O's own nav tokens bit-identical to clean, asserted per window | 2399 / 2399 windows asserted with `torch.equal` | yes |
| 4. E1 invariance, run `s1_id` | T's and O's own nav tokens bit-identical to clean, asserted per window | 2399 / 2399 windows asserted with `torch.equal` | yes |
| 5. training-set means, s0 (extra) | window list = HBLEND's fit windows; reproduces HBLEND's traj30 | n = 9582, ci/t/a_t/a_s equal = True, max abs traj30 diff 2.7e-05 m | yes |
| 5. training-set means, s1 (extra) | window list = HBLEND's fit windows; reproduces HBLEND's traj30 | n = 9582, ci/t/a_t/a_s equal = True, max abs traj30 diff 3.8e-05 m | yes |

Diagnostic (not a stop condition): LISTEN's own cells recomputed in the donor runs match LISTEN's arrays to 3.1e-05 (s0) and 2.3e-05 (s1) max abs; the E1 arm's S-row equals the `S x nav` cell bit for bit (0.0 / 0.0), as it must since S's step of the chain IS that cell.

**Instrument checks (the "built, tested, and unreachable" rule).** `pytest test_listen2.py`: 28 passed (`pytest_log.txt`). `run_mutations2.py`: the unmutated module is green and **all 14 source mutations go RED** (injection at T, at S+T; joint triple without the T code; mean ignoring the pad mask / taken at the wrong tick / divided by n+1 / T checkpoints in the wrong frame / zero ages / code not replaced / all-padded scene; own-value in the wrong frame; mean mode using own values; collector storing the clean trajectory as E1 or a single swap as E3) (`mutation_log.txt`); no survivor. The E1 injection point, the E3 triple and every mean cell are checked by captured reader inputs independent of the module's own assertions.

## 5. Readings of the PREREG text that the text leaves open (fixed in code before any LISTEN2 number; verbatim from `listen2.py`)

```
READINGS OF THE PREREG TEXT THAT THE TEXT LEAVES OPEN (all fixed here and in ``read2.py``, in code, BEFORE any LISTEN2 number was computed):
  M1  "the training-set mean of that input": the mean over the 9,582 HBLEND training windows (``train_windows(hist, start)`` of the 141 fit clips, the
      eval's cycle ages by window index) of the input AS THE LAYER READS IT at inference (``forward_infer``): per layer, at that layer's own tick, the
      token its producer builds (ego / nav / limit), the scene tokens, S's checkpoints in the layer's frame (``ck_for_t`` for T, ``ck_for_o`` for O),
      T's code, and the age pair. Every producer except the scene's is LINEAR, so the mean token of an input IS the producer of the mean input; the
      scene mean is taken token by token over the windows whose frame is not padded, and the replacement scene is not padded (pad all False).
  M2  mean ages: the mean (a_t, a_s) in ticks, fed as FLOAT tensors (the model only ever calls ``.float()`` on them).
  M3  E2's "largest effect": the argmax over the row's input columns of the SIGNED delta (the harm direction of the PREREG's sign convention). With
      delta(L, all) > 0 this equals R1's argmax of n = delta(L,X) / delta(L,all); it is used because 'all' may be undefined under mean replacement.
  M4  E2 verdict: a row AGREES iff its argmax equals the grid's largest-n input at BOTH seeds; k = the number of agreeing rows; ROBUST iff k = 4,
      PARTLY ROBUST iff k in {2, 3}, NOT ROBUST iff k < 2. The per-seed counts are reported beside.
  M5  E3 "max over the three single swaps": the single swap with the largest MEAN ADE6 on the full set (per seed); E3 = joint - that swap, paired.
      (A per-window maximum would be an upward-biased comparator and is reported only as a descriptive row.)
  M6  E1 "above the donor-seed floor": |E1(s0, donor 0) - E1(s0, donor 1)| of the same metric, applied to BOTH seeds (LISTEN's R3).
  M7  E1 outcome precedence: ROUTE FLOWS DOWN iff E1 > 0, separated and above the floor at both seeds; else PARALLEL PATH iff E1 is not separated
      at either seed; else SEEDS DISAGREE.
```

## 6. What these numbers do not say

* **Training variance.** The paired bootstrap resamples episodes with the checkpoints held fixed. E1 and E3 reach the same outcome at both banked training seeds (M2b s0 and s1), which is evidence against a one-seed artefact, but two seeds are not a noise floor and no replicate-of-the-lever arm exists here (the intervention is on inputs, not a training lever). E2's O-row disagreement is also the same at both seeds.
* **Donor choice.** The donor-seed floor is built from two donor draws on one checkpoint (s0). E1 at s0 changes by 0.0825 F1 between the two draws (0.4419 vs 0.3594); E3 by 0.43 m. Those are the only donor-variance reads taken.
* **Tier.** Everything is an intervention on a fixed checkpoint, scored open loop against the logged future (T1-class, DIAGNOSTIC). It says how the trained model uses its inputs, not how well it drives.
* **Four families.** The four-family rule governs evals of driving. This is an input-dependence diagnostic: it carries the layer rows only (S checkpoint error, T lateral / longitudinal macro-F1, O ADE6 and ADE2). The STRATEGIC family stays PAUSED under the PI ruling (ITEM 25): neither passed nor extended, and E1 does not lift it.
* **E1 is about the route, not about T-lon.** ROUTE FLOWS DOWN is a statement about T's lateral decision. T's longitudinal decision does not follow nav through S in this read (not separated at either seed).

## 7. The cheapest next lever (RULE ZERO: a result leaves the next arm behind)

None is run here, because none is pre-registered; each is CPU-only, about 1 minute per run on this harness, and the only blocker is a pre-registration (the Master Mind's call).
1. **E2 left one row open (O: ego vs T code).** The cheapest discriminating read: swap pairs (ego+code, ego+S checkpoints, S checkpoints+code) and the fully coherent four (also the prior's `p10f`) from one donor, to see whether the O row's ranking of ego against T code is a property of the layer or of which single token happens to be the most conflicting with the window's own prior.
2. **E1 settles the lateral chain, not the longitudinal one.** Inject the donor ego at S only (and at T only) and read T-lon, to see whether the longitudinal decision has any chain at all or is carried only by T's own ego and scene.
3. **A third seed of M2b** (a training run, not CPU-only) would turn "same outcome at two seeds" into a seed-noise estimate for E1 and E3. That needs compute the PI has not authorised here.

## 8. Fixes made while running the tests, and the manifest

The code under measurement (`listen2.py`, `read2.py`, `run_listen2.py`, `run_all2.sh`, `run_mutations2.py`, LISTEN's copied harness) was **not changed**; no donor seed, mean set, metric, comparator or bar was touched. Two bugs were found in `test_listen2.py` by the first run, both in the test, and fixed there:
1. `test_e3_joint_triple_hands_O_one_donors_ego_ck_and_code_only`: the hand-built shadow ran `model.run_o` with autograd on, whose kernel path differs by ~1e-6 from the `no_grad` path the harness uses, so `torch.equal` failed. The hand-built run is now under `torch.no_grad()`.
2. `_independent_means` called `.numpy()` on grad-tracking tensors (3 tests errored): now decorated with `@torch.no_grad()`.

| artifact | location (all under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/listen2/`) |
|---|---|
| this result | `LISTEN2_RESULT.md` (rendered from `READ.json` by `build_result2.py`) |
| every number | `READ.json`, `READ_controls.json` |
| per-window arrays | `runs/s0_d0.npz`, `s1_d0.npz`, `s0_d1.npz`, `s0_id.npz`, `s1_id.npz`, `s0_mean.npz`, `s1_mean.npz`, `s0_own.npz`, `s1_own.npz` |
| training-set means + controls | `runs/means_s0.npz`, `means_s1.npz`, `means_s0_control.json`, `means_s1_control.json` |
| tests and mutations | `pytest_log.txt` (28 passed), `mutation_log.txt` (14 of 14 RED), `test_listen2.py`, `run_mutations2.py` |
| memory / disk | `peakmem.log` (peak private commit 2.5 GB, working set 1.8 GB per run), `guard.log`, `run_all2.log` |
| code | `listen2.py`, `read2.py`, `run_listen2.py`, `run_all2.sh`, `guard.ps1`, `build_result2.py`; LISTEN's harness copies `listen.py`, `listen_read.py`, `run_listen.py`, `test_listen.py`, `run_mutations.py`; `code/` (LISTEN's tip export) |
| PREREG | `../grid_a2/PREREG_V8F_LISTEN2.md` |
| repo | nothing staged, committed or pushed; `scratchpad/listen/` and the repo untouched |

## CORRECTION (2026-10-25, MM; R72)

The STRATEGIC family is NOT paused for v8F. `PI-V8F-2026-10-17` (register): "R6 (strategic OFF) is superseded **for v8F**; the STRATEGIC family is un-paused for v8F". The label above quoted the superseded 2026-09-19 ruling. No number changes: the S row (S's checkpoint error under each intervention) is the strategic family's reading and is reported above; it is not paused.
