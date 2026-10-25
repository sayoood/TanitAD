# PREREG_V8F_LISTEN: result (executed 2026-10-25; zero training, CPU, M2b s0 and s1 banked checkpoints)

**Evidence class:** MEASURED (ours). Artifacts: `READ.json` (every number below), `runs/*.npz` (per-window arrays), `fig_listen_grid.png`. **Tier:** T1-class open loop, DIAGNOSTIC (donor inputs are interventions, not driving). **Estimator:** paired episode-cluster bootstrap, B = 2,000, seed 0 (`taniteval/ci.py`, through the eval's own `paired` and macro-F1 reducer). It answers "would another draw of EPISODES say this?"; it does not answer "would another TRAINING RUN say this?" (the two training seeds are shown side by side for that) nor "another inference run" (the planner is deterministic here). The PREREG is `3fdb1b61:Project Steering/PREREG_V8F_LISTEN.md`; nothing was staged or committed.

## 1. Both outcomes (the two committed tables, applied verbatim)

| table | outcome | condition met |
|---|---|---|
| Outcome (division of labour) | **NOT SHOWN** | P-UPPER fails at both seeds (T-lat's largest n is S checkpoints, not scene or nav); P-LOWER fails at both seeds (O's largest n is ego, not S checkpoints or T code); neither holds at any seed, so it is not SEEDS DISAGREE |
| Q-NAV (nav through T or around it) | **THROUGH T** | D = ADE6(nav at T only) - ADE6(nav at O only) is +0.461 m [+0.254, +0.687] at seed 0 and +0.510 m [+0.273, +0.793] at seed 1, separated at both |

Committed next step, outcome table, NOT SHOWN (quoted): "the report states that the levels' input dependence does not differ as the attention suggested, and the abstraction claim rests on the oracle and donor reads alone".
Committed next step, Q-NAV, THROUGH T (quoted): "the answer to the PI's question is yes: the trajectory follows nav mainly through T's decision".

**All five controls pass** (section 3), so the grid is reported. The reading rules R1 to R5 (how "largest n", "holds", the donor-seed floor, the headline precedence and Q-NAV are computed) were written into `listen_read.py` before any LISTEN number existed: the file's mtime is 10:28, the first grid run was written at 10:31 (`ls -l`, this directory).

### The largest cell per row per seed (R1: argmax of the unclipped n = delta(L,X) / delta(L,all))

| seed | row | largest-n cell (R1: argmax of unclipped n) | delta [95 % CI] | n | listens at this seed (delta > 0, separated, above donor floor) | in the hypothesis set |
|---|---|---|---|---|---|---|
| s0 | S | **nav** | +4.197 m [+2.987, +5.522] | 0.521 | yes | yes (scene or nav (P-UPPER)) |
| s0 | T-lat | **S checkpoints** | +0.3910 F1 [+0.3033, +0.4670] | 0.761 | yes | no (scene or nav (P-UPPER)) |
| s0 | T-lon | **ego** | +0.2466 F1 [+0.1882, +0.3054] | 0.671 | yes | (no hypothesis on this row) |
| s0 | O | **ego** | +5.697 m [+4.489, +6.979] | 4.087 | yes | no (S checkpoints or T code (P-LOWER)) |
| s1 | S | **nav** | +3.985 m [+2.806, +5.182] | 0.476 | yes | yes (scene or nav (P-UPPER)) |
| s1 | T-lat | **S checkpoints** | +0.4064 F1 [+0.3368, +0.4635] | 0.769 | yes | no (scene or nav (P-UPPER)) |
| s1 | T-lon | **ego** | +0.2610 F1 [+0.1709, +0.3277] | 0.684 | yes | (no hypothesis on this row) |
| s1 | O | **ego** | +4.716 m [+3.722, +5.806] | 3.585 | yes | no (S checkpoints or T code (P-LOWER)) |

P-UPPER and P-LOWER, per row and seed:

| hypothesis | seed | row | largest-n cell | in set | listens at seed | row holds | hypothesis holds at seed |
|---|---|---|---|---|---|---|---|
| P-UPPER | s0 | S | nav | True | True | True | False |
| P-UPPER | s0 | T-lat | S checkpoints | False | True | False | False |
| P-UPPER | s1 | S | nav | True | True | True | False |
| P-UPPER | s1 | T-lat | S checkpoints | False | True | False | False |
| P-LOWER | s0 | O | ego | False | True | False | False |
| P-LOWER | s1 | O | ego | False | True | False | False |

### What the grid says beyond the two committed tables (DESCRIPTIVE and POST HOC: not promoted, each would need its own pre-registration)

1. **The layers do differ in what they listen to; they differ along other lines than the two hypotheses drew.** Largest n per row, seeds 0 / 1: **S = nav** (0.52 / 0.48), then scene (0.40 / 0.43); **T-lat = S checkpoints** (0.76 / 0.77), then nav (0.34 / 0.41); **T-lon = ego** (0.67 / 0.68), then scene (0.50 / 0.49); **O = ego** (4.087 / 3.585), then T code (3.870 / 3.141). "NOT SHOWN" is a statement about P-UPPER and P-LOWER as written, not a statement that all four rows look alike.
2. **Attention is not dependence, shown in one cell.** T puts 97 to 100 % of its cross-attention on scene tokens (`D-V8F-ATTN-1`, INHERITED from the PREREG, not re-measured here), yet T-lat x scene is **-0.003 F1 [-0.059, +0.048] and +0.010 [-0.033, +0.051]**, not separated at either seed. The cell works as a probe: T-lon x scene is +0.185 / +0.187 F1, separated, and the deliberate-regression copy with a constant scene path reads T x scene as exactly 0.0 (control 3). T's lateral decision is carried by S's checkpoints, not by the image. Whether that is a real hierarchy or an echo of S's own nav read (S x nav n = 0.52 / 0.48) is **not tested here**.
3. **O barely reads the scene or its own nav token.** O x scene = +0.307 m [+0.214, +0.400] at seed 0 and +0.008 m [-0.015, +0.031] (not separated) at seed 1, against ALL = +1.394 / +1.315 m; O x nav = +0.015 m [+0.001, +0.030] and +0.011 m [-0.002, +0.024] (not separated at seed 1). The branch of the outcome table that sends O's scene to FLAT (PARTIAL, "if O's largest cell is the scene") was not reached, so FLAT is not triggered by this read; the descriptive fact is that O's scene cell is small.
4. **O's n exceeds 1, so n is not a share for that row.** ALL for O (+1.394 / +1.315 m) is **smaller** than ego alone (+5.697 / +4.716 m) or T code alone (+5.395 / +4.132 m). Swapping one conditioning token leaves the others from the window's own world (and the prior on its own `p10f`); swapping all gives a self-consistent donor world. HYPOTHESIS (untested): O reacts strongly to disagreement between its conditioning tokens, and the argmax-n rule (R1, fixed before data) therefore picks ego for O. On raw delta, ego and T code are both large with overlapping intervals (seed 0: +5.70 [4.49, 6.98] vs +5.40 [4.21, 6.63]; seed 1: +4.72 [3.72, 5.81] vs +4.13 [3.14, 5.14]). The verdict above is unchanged by this paragraph.
5. **"Listens" has no size threshold.** It is delta > 0, separated, and above the donor-seed floor. O x tick ages (+0.008 / +0.012 m, n 0.006 / 0.009) carries a LISTENS flag at both seeds; read the size from delta and n, not from the asterisk.
6. **Q-NAV beside.** Nav reaches the trajectory mostly through S and T: nav at S only +0.726 / +0.821 m (the route S -> T -> O, and S -> O), at T only +0.476 / +0.521 m, at O only +0.015 / +0.011 m, at all three +1.174 / +1.180 m. The committed table compares only T against O (THROUGH T); the S-only arm is larger than the T-only arm, so "through T's decision" understates that the route also passes through S's checkpoints.

### The cheapest next lever (RULE ZERO: a verdict must leave the next arm behind)

NOT SHOWN eliminates the two hypotheses as written; it does not leave nothing to do. Cheapest discriminating experiments, each CPU-only and about 1 to 2 minutes per run on the same harness (`listen.py` already supports single-cell swaps), **none run here because none is pre-registered**:
* **O coherence test:** swap pairs and triples of O's conditioning tokens (ego + S checkpoints + T code in the same donor window, the prior untouched) to see whether n > 1 is the inconsistency effect of point 4.
* **A second intervention family without cross-input conflict:** replace one input by the training-set mean token instead of a donor, to check that the rows' ordering is not an artefact of donor swapping.
* **T-lat echo test:** does T-lat's dependence on S checkpoints follow S's nav read (inject donor nav at S only and read T-lat), which separates a hierarchy from an echo.

What blocks them: only a pre-registration (the Master Mind's call) and the PI's go to spend the box; there is no compute or data blocker.

## 2. The full grid, per seed

Sign convention: S and O are donor minus clean (metres; larger is worse), T is clean minus donor (macro-F1 drop). n windows / episodes differ per row (S 2,319 / 137, T-lat 2,107 / 139, T-lon 2,240 / 136, O 2,399 / 141) because the label masks differ; the PREREG's "141 episodes" is the O row. The donor-seed floor is control 4 (seed 0, donor seeds 0 and 1), applied to both seeds (R3).

Clean level of each row's metric:

| row | clean level, seed 0 | clean level, seed 1 |
|---|---|---|
| S: checkpoint position error @ 50 m, donor - clean | 8.012 m | 7.938 m |
| T-lat: lateral macro-F1, clean - donor | 0.8157 F1 | 0.8292 F1 |
| T-lon: longitudinal macro-F1, clean - donor | 0.4992 F1 | 0.5095 F1 |
| O: ADE6, donor - clean | 2.175 m | 2.055 m |

### Seed 0 (M2b_s0, donor seed 0)

| row (metric) | input | delta (m or F1) | 95 % CI | separated | n windows / episodes | ratio n = d(L,X)/d(L,all) | donor-seed floor | per-seed flag (s0) | LISTENS (both seeds) |
|---|---|---|---|---|---|---|---|---|---|
| S | scene | +3.214 m | [+2.548, +3.971] | yes | 2319 / 137 | 0.399 | 0.166 | listens | **LISTENS** |
| S | ego | +1.347 m | [+0.888, +1.889] | yes | 2319 / 137 | 0.167 | 0.009 | listens | **LISTENS** |
| S | nav | +4.197 m | [+2.987, +5.522] | yes | 2319 / 137 | 0.521 | 0.369 | listens | **LISTENS** |
| S | limit | +0.073 m | [-0.014, +0.224] | no | 2319 / 137 | 0.009 | 0.049 | no | no |
| S | S checkpoints | not an input | | | | | | | |
| S | T code | not an input | | | | | | | |
| S | tick ages | not an input | | | | | | | |
| S | **ALL (row reference)** | **+8.057 m** | [+6.548, +9.750] | yes | 2319 / 137 | 1 by definition (defined) | 0.908 | | |
| T-lat | scene | -0.0031 F1 | [-0.0591, +0.0481] | no | 2107 / 139 | -0.006 | 0.0134 | no | no |
| T-lat | ego | +0.0824 F1 | [+0.0255, +0.1458] | yes | 2107 / 139 | 0.160 | 0.0358 | listens | **LISTENS** |
| T-lat | nav | +0.1729 F1 | [+0.0946, +0.2562] | yes | 2107 / 139 | 0.337 | 0.0328 | listens | **LISTENS** |
| T-lat | limit | -0.0252 F1 | [-0.0574, +0.0003] | no | 2107 / 139 | -0.049 | 0.0066 | no | no |
| T-lat | S checkpoints | +0.3910 F1 | [+0.3033, +0.4670] | yes | 2107 / 139 | 0.761 | 0.0375 | listens | **LISTENS** |
| T-lat | T code | not an input | | | | | | | |
| T-lat | tick ages | not an input | | | | | | | |
| T-lat | **ALL (row reference)** | **+0.5137 F1** | [+0.4347, +0.5727] | yes | 2107 / 139 | 1 by definition (defined) | 0.0272 | | |
| T-lon | scene | +0.1845 F1 | [+0.1131, +0.2559] | yes | 2240 / 136 | 0.502 | 0.0275 | listens | **LISTENS** |
| T-lon | ego | +0.2466 F1 | [+0.1882, +0.3054] | yes | 2240 / 136 | 0.671 | 0.0212 | listens | **LISTENS** |
| T-lon | nav | +0.0520 F1 | [+0.0190, +0.0837] | yes | 2240 / 136 | 0.142 | 0.0167 | listens | no |
| T-lon | limit | +0.0924 F1 | [+0.0266, +0.1518] | yes | 2240 / 136 | 0.252 | 0.0182 | listens | **LISTENS** |
| T-lon | S checkpoints | +0.0413 F1 | [+0.0032, +0.0704] | yes | 2240 / 136 | 0.112 | 0.0139 | listens | no |
| T-lon | T code | not an input | | | | | | | |
| T-lon | tick ages | not an input | | | | | | | |
| T-lon | **ALL (row reference)** | **+0.3674 F1** | [+0.2925, +0.4342] | yes | 2240 / 136 | 1 by definition (defined) | 0.0093 | | |
| O | scene | +0.307 m | [+0.214, +0.400] | yes | 2399 / 141 | 0.220 | 0.013 | listens | no |
| O | ego | +5.697 m | [+4.489, +6.979] | yes | 2399 / 141 | 4.087 | 0.310 | listens | **LISTENS** |
| O | nav | +0.015 m | [+0.001, +0.030] | yes | 2399 / 141 | 0.011 | 0.002 | listens | no |
| O | limit | +0.590 m | [+0.386, +0.820] | yes | 2399 / 141 | 0.423 | 0.150 | listens | **LISTENS** |
| O | S checkpoints | +0.873 m | [+0.596, +1.193] | yes | 2399 / 141 | 0.626 | 0.156 | listens | **LISTENS** |
| O | T code | +5.395 m | [+4.210, +6.629] | yes | 2399 / 141 | 3.870 | 0.141 | listens | **LISTENS** |
| O | tick ages | +0.008 m | [+0.002, +0.014] | yes | 2399 / 141 | 0.006 | 0.004 | listens | **LISTENS** |
| O | **ALL (row reference)** | **+1.394 m** | [+1.165, +1.623] | yes | 2399 / 141 | 1 by definition (defined) | 0.017 | | |

O, ADE2 beside (donor - clean, m):

| input | delta (m) | 95 % CI | separated |
|---|---|---|---|
| scene | +0.009 | [+0.003, +0.015] | yes |
| ego | +0.433 | [+0.301, +0.573] | yes |
| nav | -0.001 | [-0.003, +0.001] | no |
| limit | +0.392 | [+0.275, +0.523] | yes |
| S checkpoints | +0.041 | [+0.023, +0.061] | yes |
| T code | +0.464 | [+0.341, +0.600] | yes |
| tick ages | +0.001 | [+0.000, +0.001] | yes |
| ALL (row reference) | +0.051 | [+0.002, +0.102] | yes |

### Seed 1 (M2b_s1, donor seed 0)

| row (metric) | input | delta (m or F1) | 95 % CI | separated | n windows / episodes | ratio n = d(L,X)/d(L,all) | donor-seed floor | per-seed flag (s1) | LISTENS (both seeds) |
|---|---|---|---|---|---|---|---|---|---|
| S | scene | +3.632 m | [+2.874, +4.400] | yes | 2319 / 137 | 0.434 | 0.166 | listens | **LISTENS** |
| S | ego | +1.062 m | [+0.619, +1.586] | yes | 2319 / 137 | 0.127 | 0.009 | listens | **LISTENS** |
| S | nav | +3.985 m | [+2.806, +5.182] | yes | 2319 / 137 | 0.476 | 0.369 | listens | **LISTENS** |
| S | limit | +0.093 m | [+0.010, +0.217] | yes | 2319 / 137 | 0.011 | 0.049 | listens | no |
| S | S checkpoints | not an input | | | | | | | |
| S | T code | not an input | | | | | | | |
| S | tick ages | not an input | | | | | | | |
| S | **ALL (row reference)** | **+8.375 m** | [+6.692, +10.154] | yes | 2319 / 137 | 1 by definition (defined) | 0.908 | | |
| T-lat | scene | +0.0104 F1 | [-0.0330, +0.0507] | no | 2107 / 139 | 0.020 | 0.0134 | no | no |
| T-lat | ego | +0.0760 F1 | [+0.0237, +0.1343] | yes | 2107 / 139 | 0.144 | 0.0358 | listens | **LISTENS** |
| T-lat | nav | +0.2163 F1 | [+0.1268, +0.3194] | yes | 2107 / 139 | 0.409 | 0.0328 | listens | **LISTENS** |
| T-lat | limit | -0.0216 F1 | [-0.0564, +0.0042] | no | 2107 / 139 | -0.041 | 0.0066 | no | no |
| T-lat | S checkpoints | +0.4064 F1 | [+0.3368, +0.4635] | yes | 2107 / 139 | 0.769 | 0.0375 | listens | **LISTENS** |
| T-lat | T code | not an input | | | | | | | |
| T-lat | tick ages | not an input | | | | | | | |
| T-lat | **ALL (row reference)** | **+0.5286 F1** | [+0.4591, +0.5804] | yes | 2107 / 139 | 1 by definition (defined) | 0.0272 | | |
| T-lon | scene | +0.1869 F1 | [+0.1177, +0.2589] | yes | 2240 / 136 | 0.490 | 0.0275 | listens | **LISTENS** |
| T-lon | ego | +0.2610 F1 | [+0.1709, +0.3277] | yes | 2240 / 136 | 0.684 | 0.0212 | listens | **LISTENS** |
| T-lon | nav | +0.0517 F1 | [-0.0006, +0.1199] | no | 2240 / 136 | 0.136 | 0.0167 | no | no |
| T-lon | limit | +0.1405 F1 | [+0.0491, +0.2004] | yes | 2240 / 136 | 0.368 | 0.0182 | listens | **LISTENS** |
| T-lon | S checkpoints | +0.0130 F1 | [-0.0565, +0.0812] | no | 2240 / 136 | 0.034 | 0.0139 | no | no |
| T-lon | T code | not an input | | | | | | | |
| T-lon | tick ages | not an input | | | | | | | |
| T-lon | **ALL (row reference)** | **+0.3816 F1** | [+0.2818, +0.4491] | yes | 2240 / 136 | 1 by definition (defined) | 0.0093 | | |
| O | scene | +0.008 m | [-0.015, +0.031] | no | 2399 / 141 | 0.006 | 0.013 | no | no |
| O | ego | +4.716 m | [+3.722, +5.806] | yes | 2399 / 141 | 3.585 | 0.310 | listens | **LISTENS** |
| O | nav | +0.011 m | [-0.002, +0.024] | no | 2399 / 141 | 0.008 | 0.002 | no | no |
| O | limit | +0.365 m | [+0.240, +0.500] | yes | 2399 / 141 | 0.278 | 0.150 | listens | **LISTENS** |
| O | S checkpoints | +1.894 m | [+1.432, +2.397] | yes | 2399 / 141 | 1.440 | 0.156 | listens | **LISTENS** |
| O | T code | +4.132 m | [+3.138, +5.135] | yes | 2399 / 141 | 3.141 | 0.141 | listens | **LISTENS** |
| O | tick ages | +0.012 m | [+0.004, +0.021] | yes | 2399 / 141 | 0.009 | 0.004 | listens | **LISTENS** |
| O | **ALL (row reference)** | **+1.315 m** | [+1.122, +1.501] | yes | 2399 / 141 | 1 by definition (defined) | 0.017 | | |

O, ADE2 beside (donor - clean, m):

| input | delta (m) | 95 % CI | separated |
|---|---|---|---|
| scene | +0.000 | [-0.004, +0.004] | no |
| ego | +0.304 | [+0.213, +0.401] | yes |
| nav | +0.002 | [-0.001, +0.005] | no |
| limit | +0.209 | [+0.138, +0.294] | yes |
| S checkpoints | +0.107 | [+0.079, +0.138] | yes |
| T code | +0.286 | [+0.206, +0.367] | yes |
| tick ages | +0.001 | [-0.000, +0.002] | no |
| ALL (row reference) | +0.053 | [+0.003, +0.102] | yes |

## 3. Controls

**C2. Reproduction** (the clean pass vs the banked own-code eval of the same checkpoint; stop threshold 1e-3). MEASURED: `repro_s0.json`, `READ.json -> controls.reproduction`.

| seed | reference | max abs metric diff (S@50, ADE6, T-lat F1, T-lon F1) | max per-window diff (traj30 / ADE6 / goal_prob / ckpt) | argmax flips lat / lon | pass |
|---|---|---|---|---|---|
| s0 | banked box-2 GPU eval | 4.136e-05 | 5.34e-05 / 1.11e-05 / 1.49e-06 / 8.26e-05 | 0 / 0 | True |
| s0 | SFRESH CPU re-run | 4.136e-05 | 0.00e+00 / 0.00e+00 / 0.00e+00 / 0.00e+00 | 0 / 0 | True |
| s1 | banked box-2 GPU eval | 4.938e-05 | 9.92e-05 / 2.74e-05 / 2.00e-06 / 1.41e-04 | 0 / 0 | True |
| s1 | SFRESH CPU re-run | 4.938e-05 | 0.00e+00 / 0.00e+00 / 0.00e+00 / 0.00e+00 | 0 / 0 | True |

Metrics, mine vs banked (box-2) :

| seed | metric | mine | banked | abs diff |
|---|---|---|---|---|
| s0 | S_pos_err_50m | 8.012241 | 8.012200 | 4.14e-05 |
| s0 | ADE6 | 2.175306 | 2.175300 | 6.06e-06 |
| s0 | T_lat_macro_F1 | 0.815696 | 0.815696 | 0.00e+00 |
| s0 | T_lon_macro_F1 | 0.499155 | 0.499155 | 0.00e+00 |
| s1 | S_pos_err_50m | 7.938449 | 7.938400 | 4.94e-05 |
| s1 | ADE6 | 2.054566 | 2.054600 | 3.40e-05 |
| s1 | T_lat_macro_F1 | 0.829225 | 0.829225 | 0.00e+00 |
| s1 | T_lon_macro_F1 | 0.509529 | 0.509529 | 0.00e+00 |

**C1. Identity** (donor = self => every per-window array bit-identical to the clean pass, so every delta is exactly 0.0). MEASURED: `runs/s*_id.npz`.

| seed | cells compared | max abs diff vs clean | cells not identical | pass |
|---|---|---|---|---|
| s0 | 23 | 0.0 | 0 | True |
| s1 | 23 | 0.0 | 0 | True |

**C3. Structural zero** (deliberate regression: a copy whose O/T/S nav token is a constant must read the nav cells as exactly 0.0; a copy whose T scene path is constant must read T x scene as exactly 0.0; every unprobed cell of the same copy must stay alive, > 0). MEASURED: `runs/s*_navc.npz`, `runs/s*_tsoff.npz`.

| copy | probed cells -> max abs diff vs the copy's own clean | unprobed (alive) cells -> max abs diff | pass |
|---|---|---|---|
| s0/nav_const | S/nav = 0.0, T/nav = 0.0, O/nav = 0.0 | O/ego = 3.228e+01, T/ego = 5.000e+00 | True |
| s0/tscene_off | T/scene = 0.0 | T/ego = 6.000e+00, O/scene = 6.889e+00 | True |
| s1/nav_const | S/nav = 0.0, T/nav = 0.0, O/nav = 0.0 | O/ego = 2.988e+01, T/ego = 6.000e+00 | True |
| s1/tscene_off | T/scene = 0.0 | T/ego = 6.000e+00, O/scene = 2.134e+00 | True |

**C5. Consistency with the banked shuffles** (descriptive; plus the implementation check that O x T code at donor seed 1 IS `--tac shuffle`). MEASURED: `banked/`, `runs/s0_d1.npz`.

* s0, donor seed 1: max per-window absolute difference between my O x code ADE6 and the banked `--tac shuffle` ADE6 = **1.372e-05** m; mine O x code ADE6 mean 7.7118 m, banked tac-shuffle 7.7118 m, banked own-code 2.1753 m; banked delta +5.5365 m, mine +5.5365 m.
* s0, donor seed 0: O x T code delta = +5.3952 m (the report's cited range for the banked shuffles: +4.4 to +6.4 m ADE6).
* s1, donor seed 0: O x T code delta = +4.1320 m (the report's cited range for the banked shuffles: +4.4 to +6.4 m ADE6).

**C4. Donor-seed noise** (s0, donor seeds 0 and 1; a cell counts as listening only if |delta| exceeds this): per cell, |delta(donor 0) - delta(donor 1)|.

| row | scene | ego | nav | limit | S checkpoints | T code | tick ages | ALL |
|---|---|---|---|---|---|---|---|---|
| S (m) | 0.1663 | 0.0087 | 0.3691 | 0.0490 | n/a | n/a | n/a | 0.9078 |
| T-lat (F1) | 0.0134 | 0.0358 | 0.0328 | 0.0066 | 0.0375 | n/a | n/a | 0.0272 |
| T-lon (F1) | 0.0275 | 0.0212 | 0.0167 | 0.0182 | 0.0139 | n/a | n/a | 0.0093 |
| O (m) | 0.0131 | 0.3097 | 0.0019 | 0.1503 | 0.1559 | 0.1413 | 0.0037 | 0.0170 |

**All controls pass: True.**

Control 5 beside the cited banked range: the banked `--tac shuffle` deltas computed from `banked/` are +5.5365 m (seed 0) and +4.4198 m (seed 1), both at donor seed 1; ours at donor seed 0 are +5.395 m and +4.132 m, same sign and order of size. The seed-1 value is 0.29 m below the banked one; the donor-seed difference for that cell was measured only on seed 0 (0.141 m).

## 4. Q-NAV

| seed | arm (donor nav injected at) | ADE6 change vs clean (m) | 95 % CI | separated |
|---|---|---|---|---|
| s0 | S only (S -> T -> O) | +0.726 | [+0.426, +1.081] | yes |
| s0 | T only (T -> O) | +0.476 | [+0.261, +0.703] | yes |
| s0 | O only (the O x nav cell) | +0.015 | [+0.001, +0.030] | yes |
| s0 | S + T + O | +1.174 | [+0.759, +1.634] | yes |
| s0 | **D = ADE6(nav at T only) - ADE6(nav at O only), paired** | **+0.461** | [+0.254, +0.687] | yes |
| s1 | S only (S -> T -> O) | +0.821 | [+0.509, +1.172] | yes |
| s1 | T only (T -> O) | +0.521 | [+0.282, +0.812] | yes |
| s1 | O only (the O x nav cell) | +0.011 | [-0.002, +0.024] | no |
| s1 | S + T + O | +1.180 | [+0.769, +1.619] | yes |
| s1 | **D = ADE6(nav at T only) - ADE6(nav at O only), paired** | **+0.510** | [+0.273, +0.793] | yes |

## 5. Deviations and disclosures

* **D1 (numerical, no number changes).** `v8f_mini_eval._f1_reducer` raises an IndexError on this numpy when the bootstrap hands it float64 arrays; `listen_read.f1_reducer` wraps the eval's own reducer with an exact integer cast (the encoded values are integers).
* **D2 (resume).** A previous agent stopped the run on memory pressure; this agent finished it. The runs `s0_d0 s1_d0 s0_d1 s0_id s1_id s0_navc s0_tsoff` used 8 torch threads; `s1_navc`, `s1_tsoff` and the analysis used 6 (`LISTEN_THREADS=6`, now an env var in `run_listen.py`), all with `LISTEN_LRU=8`. Every control compares arrays inside one run, and the clean pass is bit-identical to SFRESH's CPU re-run, so the thread count can touch only float noise across runs.
* **D3 (cosmetic, after the numbers).** The figure's cell font, the Q-NAV bar-label rotation and the caption text were adjusted after the first render; no cell content, ordering, colour rule or threshold changed.
* **D4 (reading rules).** R1 to R5 are the previous agent's, fixed before data (see section 1).
* **Memory and disk.** Peak of this agent's processes (`peakmem.log`): the two collect runs **2,643 and 2,646 MB peak private commit (1,895 / 1,896 MB peak working set)**; the analysis **1,119 MB (518 MB)**. The previous agent's unfixed run was 3.1 GB private / 2.2 GB working set (inherited, not re-measured). Guard readings before each step (`guard.log`): commit headroom 8.7 to 11.8 GB (threshold 5), C: free 0.87 to 1.6 GB (threshold 0.7). Other agents' processes were running throughout and were not touched.
* **Tests.** `test_listen.py`: 20 passed (`pytest_log_final.txt`); `run_mutations.py`: 9 source-level mutations all RED, unmutated GREEN (`mutation_log.txt`, previous agent). The tests create `pt_tmp/` in this directory; nothing was deleted.

## 6. Manifest (everything under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/listen/`)

`LISTEN_RESULT.md` (this), `READ.json`, `fig_listen_grid.png`, `fig_listen_grid.py`, `listen.py`, `listen_read.py`, `run_listen.py`, `write_result.py`, `result_template.md`, `test_listen.py`, `run_mutations.py`, `runs/*.npz` (9), `repro_s0.json`, `mutation_log.txt`, `pytest_log.txt`, `pytest_log_final.txt`, `run_all.log`, `resume_runs.log`, `run_analyze.log`, `peakmem.log`, `guard.log`, `guard.ps1`, `resume_runs.sh`, `HANDOFF.md` (stale: superseded by this result), `banked/`, `code/` (tip export), `mock/` (figure test only), `pt_tmp/` (pytest scratch). Nothing was git-added or committed; landing this under `Research/` and updating `GOALS_AND_CLAIMS.md` (the abstraction claim, `D-V8F-ATTN-1`) is the Master Mind's integration step.
