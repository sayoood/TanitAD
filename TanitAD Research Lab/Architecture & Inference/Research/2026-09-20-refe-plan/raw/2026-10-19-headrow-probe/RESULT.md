# RESULT: HEADROW-PROBE (are the REFE head-row levers capacity-limited or feature-limited?)

**Agent:** TanitAD Architecture & Inference, task HEADROW-PROBE (brief `C:/Users/Admin/par18/probe.md`). **Written:** 2026-10-19 ~10:18. **Status:** stage-only; nothing committed or pushed.
**Evidence class and tier:** MEASURED (ours, artifact paths in `MANIFEST.md`), SELECTION tier (fit-side OOF, 5 folds grouped by log, f = 10 %, 117 logs, 15,808 sets). A design read about feature sufficiency, not a driving result: T0 / T1 do not apply. P2 and its trunk are IN-SAMPLE on these sets.
**Estimator:** paired log-cluster bootstrap, B = 10,000, seed 20260927, percentile 2.5 / 97.5. It answers "would another draw of LOGS say this?" and nothing else; two seeds sample head-fit noise only (the third variance, training-run to training-run, is not sampled).

| document | sha256 | written (2026-10-19) | relation to numbers |
|---|---|---|---|
| `PREREG_HEADROW_PROBE.md` | `1800705ef49333dffe0162ea5e51eb72fa1fe8fba57a0ea2f2869bf156762c55` | 06:43:27 | before any code or fit |
| Amendment 1 (additive outputs: the Master Mind addendum) | `f98e24deafdefe3501faecaa09a4fa328e2d00ef8658458678e09f6a034fd7f3` | 07:32 | before any outcome was computed |
| Amendment 2 (C2x, POST-HOC) | `662d228cb64dcfc7d10efaa58a37a88487ac59fd2daee4ba93b49c918754bc19` | ~09:29 | after the outcome tables, before any C2x number |
| Amendment 3 (C2y + index diagnostic, POST-HOC) | `dc58ff1ea5921aed5028027b3daa54c402f5b07f930e5551ef292d79632d4037` | ~09:41 | after C2x, before any C2y / index number |
| Amendment 4 (C2z, POST-HOC) | `aed50d2585dba8adba95e527112c3b6bcf4da5aed20ad14591d889a0ea5faf22` | ~09:54 | after C2y, before any C2z number |
| Amendment 5 (slot prior, POST-HOC) | `97194ebc7bde955f706a6ac9d0d097030eb8332beaf653ba31312712ae9b7fcb` | ~10:09 | after C2z, before any slot-prior number |

Amendments 2 to 5 are post-hoc and unregistered as results: each commits its reading rule before its own numbers exist, and **none changes, rescues or voids a per-row call, the outcome row or the next arm.** The decision rule, metrics and controls of the brief are exactly as registered.

## 0. For the Master Mind: calls, decisions, escalations (read this first)

**The calls (registered rule, unchanged):**

| row | call | basis |
|---|---|---|
| **NC (col 0)** | **FEATURE-LIMITED** | M - L upper CI < +0.01 on both seeds (-0.0014, -0.0009); L within 0.01 of P2 (-0.0008, -0.0009); C-POS GREEN |
| **DAC (col 1)** | **FEATURE-LIMITED** | M - L upper CI -0.0007, -0.0030; L - P2 +0.0004, +0.0005 |
| **TTC (col 3)** | **FEATURE-LIMITED** | M - L upper CI -0.0021, -0.0026; L - P2 -0.0006, -0.0004 |
| **EP (col 2)** | **INCONCLUSIVE** | M - L = +0.0054 [+0.0019, +0.0089] and +0.0063 [+0.0025, +0.0101]: lower CI < +0.01 (no CAPACITY) and seed-1 upper CI 0.0101 > +0.01 (no FEATURE by 0.0001); and clause (b) cannot hold (below) |

**Outcome row: MIXED.** NC, DAC and TTC close their head-row line; the lever for them is the trunk (P6, read about 25 Oct). EP takes its committed next arm: one re-run at f = 25 %. **No row is CAPACITY-LIMITED**, so no nonlinear-head or safety-row-MLP registration is triggered, and the "all four FEATURE" trunk-arm draft is NOT triggered (EP is INCONCLUSIVE; no outline written).

M is not merely "no better than L" on NC, DAC and TTC: it is significantly worse (-0.004 to -0.006 concordance, CIs excluding 0 on all six cells) with a lower train BCE than L in every row (NC 0.123 vs 0.138, DAC 0.155 vs 0.172, TTC 0.225 vs 0.249): the extra capacity fits the fit-side folds and does not transfer across logs.

**Decisions and escalations (each one is yours; I have not acted on any):**

1. **DECISION: spend the EP f = 25 % re-run?** It is the committed arm. Cost (ESTIMATED from the measured f10 fit seconds, EP L + M, 2 seeds: 49 + 58 + 284 + 305 = 696 s, about 11.6 min; x 2.5 sets, linear scaling assumed): **about 30 min of fit time, plus bank load, gate and metrics (~15 to 20 min), one CPU process, no GPU, no pod.** A second INCONCLUSIVE closes EP as undetermined (not re-run). **Two facts to weigh before spending it, both measured here:** (i) the EP FEATURE route is closed by construction: clause (b) needs L within 0.01 of P2's EP row, but P2's EP row was trained on the unconditional target and any conditional refit moves it by far more (+0.266 measured), so EP can only come out CAPACITY or INCONCLUSIVE at f25 too; (ii) the **slot prior** (section 3) reads **0.9794** on EP, above both probes (L 0.9616, M 0.9670), so the entire scene-dependent headroom above a feature-free lookup is at most 0.021 and M - L is +0.006 of it. If you run it, run it with the slot-table comparator registered first (zero cost) and with clause (b) restated against a conditional reference; otherwise EP's answer is "no evidence of a head-capacity lever; the row is dominated by a scene-independent component".
2. **ESCALATION: the registered C2 is not usable as written, for features of this kind.** It read RED in 8 of 8 cells and its own false-RED rate under the measured null is 15 % per cell, but the null it faces (a probe TRAINED on noise labels over P2's structured features) has a per-replicate RED rate of up to 0.375 (C2z). Any successor registration that reuses "lower CI of (concordance - 0.5) > 0" must replace it with a null calibrated on noise-trained fits of the same arm, and must report the slot prior beside every concordance. This probe did not need the control to decide a call (no CAPACITY was reached), so no call depends on it.
3. **ESCALATION: every head-row concordance in this probe (P2's, L's, M's) contains a scene-independent component.** A feature-free lookup of the mean label of each proposal SLOT scores NC 0.829, DAC 0.682, EP 0.979, TTC 0.8185 (section 3). The decision statistic (M - L) is a paired difference of two probes on the same features and is far less exposed than a level, but the levels (and any claim that a row "reaches 0.92") must be read with this. **For FASTNEG-CENSUS:** the slot index is exported (`proposal_index`); the slot is a candidate confounder of any speed stratum. I computed nothing by speed, as instructed.
4. **A-RES reads R-OVER** (section 2.4): the existing NC / DAC heads already rank EP-R1's new violators below the base pick in 92.8 % of the violating sets (AUC 0.770). Per the committed join rule, a head retrain alone cannot cure the EP-R1 leak, and any safety-row work must carry a pick-level floor on `pNC * pDAC` (the R0e-pick family) as its comparator. With NC and DAC FEATURE-LIMITED, the two reads agree: the heads are not the bottleneck; the pick rule's weighting is where the violators get through.
5. **INTEGRATION (nothing is merged or landed by me):** (a) `code/` (new files only; `epr1.py` is the landed parent, imported unchanged, sha `a751c63e...`, not re-staged) belongs in the repo next to `2026-10-18-ep-r1/code/`; (b) `raw/f10/oof_proposals_f10.npz` is **86,733,457 bytes**: decide whether it lives in git or on the D: artifact store; (c) the C2, C-POS, C2x, C2y and C2z per-cell tables (about 360 MB) are **single-place artifacts on `C:/Users/Admin/par18/probe/run/f10/`**; their statistics are staged, the tables are not. Recompute-able by rerunning the stages; sha256 in `MANIFEST.md`.
6. **Proposed GAC / BACKLOG rows** are in section 6; I have written nothing into `Project Steering/`.

## 1. The decision, in full

#### T1. Decision table (concordance over all 64 proposals, OOF; M - L; paired log-cluster bootstrap 10,000 / seed 20260927)

| row | seed | P2 ref (in-sample) | L | L - P2 | M | M - L [95 %] | CAPACITY clause (lo > +0.01) | FEATURE clause (hi < +0.01) |
|---|---|---|---|---|---|---|---|---|
| NC | 0 | 0.9165 | 0.9157 | -0.0008 | 0.9108 | -0.0049 [-0.0087, -0.0014] | no | yes |
| NC | 1 | 0.9165 | 0.9155 | -0.0009 | 0.9112 | -0.0044 [-0.0086, -0.0009] | no | yes |
| DAC | 0 | 0.9214 | 0.9218 | +0.0004 | 0.9183 | -0.0036 [-0.0067, -0.0007] | no | yes |
| DAC | 1 | 0.9214 | 0.9219 | +0.0005 | 0.9158 | -0.0061 [-0.0095, -0.0030] | no | yes |
| EP | 0 | 0.6956 | 0.9616 | +0.2660 | 0.9670 | 0.0054 [0.0019, 0.0089] | no | yes |
| EP | 1 | 0.6956 | 0.9613 | +0.2657 | 0.9676 | 0.0063 [0.0025, 0.0101] | no | no |
| TTC | 0 | 0.8812 | 0.8806 | -0.0006 | 0.8750 | -0.0056 [-0.0096, -0.0021] | no | yes |
| TTC | 1 | 0.8812 | 0.8808 | -0.0004 | 0.8747 | -0.0061 [-0.0099, -0.0026] | no | yes |

| row | CAPACITY (both seeds) | FEATURE M-L clause | L within 0.01 of P2 | C2 row RED | C-POS GREEN | base call | **final call** | reasons |
|---|---|---|---|---|---|---|---|---|
| NC | False | True | True | True | True | FEATURE-LIMITED | **FEATURE-LIMITED** | - |
| DAC | False | True | True | True | True | FEATURE-LIMITED | **FEATURE-LIMITED** | - |
| EP | False | False | False | True | True | INCONCLUSIVE | **INCONCLUSIVE** | M - L upper CI >= +0.01 on some seed and lower CI <= +0.01 on some seed |
| TTC | False | True | True | True | True | FEATURE-LIMITED | **FEATURE-LIMITED** | - |

Reading the table: CAPACITY needs the lower CI of M - L > +0.01 on both seeds; FEATURE needs the upper CI < +0.01 on both seeds AND |L - P2| <= 0.01 on the same reference. P2's reference is the stored logit of the row scored with the same all-64 concordance (in-sample for P2). C2 RED voids a CAPACITY call only, and C-POS not GREEN voids FEATURE calls only; no CAPACITY call existed to void, and C-POS is GREEN (below), so the base call is the final call in every row.

#### T2. Per row x arm x seed (OOF; SELECTION tier; intervals = another draw of LOGS only; P2 reference is IN-SAMPLE for P2)

| row | seed | arm | conc all-64 | conc top-8 | Brier | dPDMS x100 vs P2 | d unsafe-pick rate vs P2 | changed picks | train BCE |
|---|---|---|---|---|---|---|---|---|---|
| NC | - | P2 ref | 0.9165 [0.9074, 0.9267] | 0.7349 [0.6900, 0.7867] | 0.03788 [0.03289, 0.04313] | 0 | 0 (rate 0.03036) | 0 | - |
| NC | 0 | L | 0.9157 [0.9066, 0.9260] | 0.7329 [0.6869, 0.7830] | 0.03843 [0.03334, 0.04387] | +0.032 [-0.031, +0.089] | +0.0003 [-0.0003, +0.0009] | 0.105 | 0.13825 |
| NC | 0 | M | 0.9108 [0.9015, 0.9218] | 0.7438 [0.6966, 0.7893] | 0.04075 [0.03531, 0.04651] | -0.025 [-0.108, +0.061] | +0.0007 [-0.0001, +0.0016] | 0.208 | 0.12318 |
| NC | 1 | L | 0.9155 [0.9064, 0.9259] | 0.7293 [0.6830, 0.7785] | 0.03869 [0.03363, 0.04408] | +0.056 [-0.016, +0.132] | +0.0000 [-0.0007, +0.0007] | 0.113 | 0.13850 |
| NC | 1 | M | 0.9112 [0.9020, 0.9219] | 0.7375 [0.6905, 0.7841] | 0.03992 [0.03468, 0.04557] | +0.044 [-0.061, +0.140] | +0.0003 [-0.0006, +0.0014] | 0.207 | 0.12298 |
| DAC | - | P2 ref | 0.9214 [0.9053, 0.9373] | 0.6869 [0.6530, 0.7246] | 0.05449 [0.04386, 0.06724] | 0 | 0 (rate 0.03036) | 0 | - |
| DAC | 0 | L | 0.9218 [0.9060, 0.9374] | 0.6918 [0.6603, 0.7249] | 0.05532 [0.04457, 0.06825] | +0.074 [+0.002, +0.164] | -0.0005 [-0.0013, +0.0002] | 0.088 | 0.17248 |
| DAC | 0 | M | 0.9183 [0.9016, 0.9346] | 0.6935 [0.6608, 0.7280] | 0.05798 [0.04654, 0.07184] | +0.072 [-0.042, +0.183] | +0.0009 [-0.0004, +0.0024] | 0.207 | 0.15541 |
| DAC | 1 | L | 0.9219 [0.9062, 0.9375] | 0.6896 [0.6578, 0.7232] | 0.05521 [0.04454, 0.06810] | +0.093 [+0.017, +0.179] | -0.0007 [-0.0015, +0.0001] | 0.090 | 0.17280 |
| DAC | 1 | M | 0.9158 [0.8983, 0.9332] | 0.6795 [0.6403, 0.7197] | 0.05786 [0.04607, 0.07210] | +0.090 [-0.025, +0.217] | +0.0003 [-0.0010, +0.0016] | 0.191 | 0.15656 |
| EP | - | P2 ref | 0.6956 [0.6760, 0.7134] | 0.8977 [0.8836, 0.9097] | 0.04032 [0.03594, 0.04551] | 0 | 0 (rate 0.03036) | 0 | - |
| EP | 0 | L | 0.9616 [0.9576, 0.9651] | 0.9670 [0.9643, 0.9697] | 0.00951 [0.00884, 0.01028] | +0.509 [+0.301, +0.697] | +0.0032 [+0.0015, +0.0053] | 0.592 | 0.31372 |
| EP | 0 | M | 0.9670 [0.9626, 0.9705] | 0.9703 [0.9676, 0.9727] | 0.00867 [0.00790, 0.00951] | +0.432 [+0.177, +0.665] | +0.0053 [+0.0032, +0.0079] | 0.623 | 0.29945 |
| EP | 1 | L | 0.9613 [0.9571, 0.9649] | 0.9666 [0.9638, 0.9695] | 0.00945 [0.00877, 0.01023] | +0.477 [+0.267, +0.665] | +0.0037 [+0.0019, +0.0057] | 0.595 | 0.31349 |
| EP | 1 | M | 0.9676 [0.9638, 0.9708] | 0.9700 [0.9671, 0.9728] | 0.00862 [0.00792, 0.00938] | +0.565 [+0.348, +0.749] | +0.0034 [+0.0016, +0.0056] | 0.612 | 0.29910 |
| TTC | - | P2 ref | 0.8812 [0.8667, 0.8967] | 0.7008 [0.6647, 0.7470] | 0.07696 [0.06943, 0.08478] | 0 | 0 (rate 0.03036) | 0 | - |
| TTC | 0 | L | 0.8806 [0.8663, 0.8958] | 0.6998 [0.6652, 0.7432] | 0.07745 [0.07003, 0.08523] | +0.077 [-0.026, +0.164] | +0.0005 [-0.0003, +0.0015] | 0.186 | 0.24924 |
| TTC | 0 | M | 0.8750 [0.8599, 0.8907] | 0.7002 [0.6673, 0.7415] | 0.08531 [0.07720, 0.09364] | +0.004 [-0.144, +0.148] | +0.0008 [-0.0003, +0.0021] | 0.359 | 0.22541 |
| TTC | 1 | L | 0.8808 [0.8666, 0.8960] | 0.7008 [0.6665, 0.7433] | 0.07779 [0.07034, 0.08558] | +0.071 [-0.025, +0.157] | +0.0005 [-0.0002, +0.0014] | 0.192 | 0.24951 |
| TTC | 1 | M | 0.8747 [0.8599, 0.8905] | 0.6894 [0.6564, 0.7299] | 0.08278 [0.07513, 0.09072] | +0.133 [-0.030, +0.279] | +0.0006 [-0.0006, +0.0022] | 0.360 | 0.22196 |

#### T3. M - L on every metric (paired)

| row | seed | conc all-64 | conc top-8 | Brier | dPDMS x100 (M-L) | d unsafe-pick (M-L) |
|---|---|---|---|---|---|---|
| NC | 0 | -0.0049 [-0.0087, -0.0014] | 0.0109 [-0.0115, 0.0317] | +0.00231 [+0.00141, +0.00333] | -0.057 [-0.131, +0.023] | +0.0004 [-0.0004, +0.0013] |
| NC | 1 | -0.0044 [-0.0086, -0.0009] | 0.0082 [-0.0202, 0.0319] | +0.00123 [+0.00051, +0.00212] | -0.012 [-0.102, +0.070] | +0.0003 [-0.0005, +0.0011] |
| DAC | 0 | -0.0036 [-0.0067, -0.0007] | 0.0017 [-0.0178, 0.0211] | +0.00266 [+0.00167, +0.00388] | -0.003 [-0.153, +0.118] | +0.0014 [-0.0001, +0.0032] |
| DAC | 1 | -0.0061 [-0.0095, -0.0030] | -0.0101 [-0.0328, 0.0124] | +0.00265 [+0.00129, +0.00430] | -0.002 [-0.117, +0.114] | +0.0010 [-0.0003, +0.0024] |
| EP | 0 | 0.0054 [0.0019, 0.0089] | 0.0033 [-0.0001, 0.0066] | -0.00084 [-0.00119, -0.00052] | -0.078 [-0.267, +0.103] | +0.0021 [+0.0002, +0.0042] |
| EP | 1 | 0.0063 [0.0025, 0.0101] | 0.0034 [-0.0003, 0.0069] | -0.00083 [-0.00114, -0.00055] | +0.088 [-0.078, +0.246] | -0.0003 [-0.0018, +0.0014] |
| TTC | 0 | -0.0056 [-0.0096, -0.0021] | 0.0004 [-0.0188, 0.0165] | +0.00785 [+0.00572, +0.01005] | -0.073 [-0.222, +0.080] | +0.0003 [-0.0008, +0.0016] |
| TTC | 1 | -0.0061 [-0.0099, -0.0026] | -0.0114 [-0.0285, 0.0029] | +0.00499 [+0.00340, +0.00686] | +0.062 [-0.070, +0.192] | +0.0001 [-0.0009, +0.0014] |

**Pick-level reading (descriptive; the brief's rule does not use it).** NC, DAC, TTC: no pick-level change vs P2 is separated for either arm except the DAC L arm's dPDMS x100 +0.074 / +0.093 (lower CIs +0.002 / +0.017; unsafe-pick rate -0.0005 / -0.0007, CIs include 0). EP: all four cells raise dPDMS x100 (+0.43 to +0.57, CIs exclude 0) **and raise the unsafe-pick rate (+0.0032 to +0.0053, CIs exclude 0 in all four cells)**: the EP-R1 pattern (a progress gain bought with more unsafe picks) reproduces out of fold. M vs L at pick level is inside the interval on every row (T3). P2 is in-sample, so the sign of every pick-level number against P2 is conservative against the probe.

## 2. Controls

### 2.1 C-ID, reach proof, gate

The gate token (`raw/f10/gate_token.json`, sha256 `7d6e6b6797d738453c5600874f900ae6e08f14d56a5534bb3f29319cd365bbf4`, stage `gate`, 5 of 5 checks: G-COMMIT, G-DVB, G-BANK, G-LIVE, G-SUITE) was minted before any fit, and the three gate-bound files (`headrow_probe.py` `e09588a1...`, `test_headrow_probe.py` `8c14ef4f...`, `epr1.py` `a751c63e...`) are byte-identical to the bound values at the end of the run (re-hashed 10:14). The reach proof (G-LIVE, real batch of 512 sets): non-zero gradient on the probe parameters of every row and arm (L: `skip_w` 9.9 to 41.6; M: `W1` 8.3 to 39.1, `W2` 3.0 to 13.0) and **EXACTLY 0.0 on all six frozen rows and on the features**, with a zero control that reads exactly 0.0 on every parameter and the optimiser update moving the skip by 1.0e-3. The synthetic suite is 39 tests with a deliberate-regression arm per check (`raw/f10/test_headrow_probe.log`: 39 tests OK, re-run 10:14).

#### T4. Controls

**C-ID** GREEN: all 8 (row, arm) cells: z == z_stored under torch.equal on all 15808 sets, skip == P2's row, M output layer exactly 0, picks / concordance / Brier sums equal P2's exactly (max |z - z_stored| = 0.0).

**C2** (cold start, labels permuted within set, 5 replicates; RED iff any replicate has lower CI of (conc - 0.5) > 0):

| row | arm | RED | lower CI of (conc - 0.5): min over reps | max over reps | point (conc - 0.5) range |
|---|---|---|---|---|---|
| NC | L | True | +0.2127 | +0.2226 | +0.2332 .. +0.2422 |
| NC | M | True | +0.1241 | +0.1391 | +0.1501 .. +0.1710 |
| DAC | L | True | +0.2748 | +0.2790 | +0.3137 .. +0.3177 |
| DAC | M | True | +0.0991 | +0.1266 | +0.1331 .. +0.1573 |
| EP | L | True | +0.0555 | +0.0617 | +0.0721 .. +0.0791 |
| EP | M | True | -0.0192 | +0.0007 | +0.0010 .. +0.0170 |
| TTC | L | True | +0.2011 | +0.2088 | +0.2218 .. +0.2284 |
| TTC | M | True | +0.0456 | +0.0739 | +0.0928 .. +0.1119 |

**C2-DEC** (shuffled M - L, lower CI > +0.01 on any replicate): NC False, DAC False, EP False, TTC False

| row | C2-DEC replicate M - L [95 %] (5 replicates) |
|---|---|
| NC | -0.0641 [-0.1031, -0.0271]; -0.0786 [-0.1131, -0.0408]; -0.0811 [-0.1151, -0.0496]; -0.0706 [-0.1015, -0.0439]; -0.0873 [-0.1190, -0.0571] |
| DAC | -0.1700 [-0.2115, -0.1255]; -0.1597 [-0.2033, -0.1124]; -0.1698 [-0.2145, -0.1219]; -0.1789 [-0.2260, -0.1289]; -0.1806 [-0.2235, -0.1326] |
| EP | -0.0648 [-0.0847, -0.0435]; -0.0619 [-0.0797, -0.0420]; -0.0748 [-0.0962, -0.0508]; -0.0665 [-0.0856, -0.0445]; -0.0653 [-0.0853, -0.0418] |
| TTC | -0.1295 [-0.1813, -0.0794]; -0.1180 [-0.1590, -0.0760]; -0.1290 [-0.1691, -0.0894]; -0.1236 [-0.1670, -0.0807]; -0.1145 [-0.1633, -0.0651] |

**The C2 rule's false-RED rate under its null** (iid scores independent of the labels, same per-log concordance, same bootstrap, same rule; 200 trials x 5 replicates per row):

| row | trials with >= 1 RED | rate per cell [Wilson 95 %] | replicates RED | rate per replicate [Wilson 95 %] |
|---|---|---|---|---|
| NC | 30/200 | 0.150 [0.107, 0.206] | 32/1000 | 0.0320 [0.0228, 0.0448] |
| DAC | 34/200 | 0.170 [0.124, 0.228] | 37/1000 | 0.0370 [0.0270, 0.0506] |
| EP | 29/200 | 0.145 [0.103, 0.200] | 29/1000 | 0.0290 [0.0203, 0.0413] |
| TTC | 26/200 | 0.130 [0.090, 0.184] | 27/1000 | 0.0270 [0.0186, 0.0390] |

Mean per-cell false-RED rate 0.149; family rate over the 8 (row, arm) cells 1-(1-p)^8 = 0.724.

**C-POS** (planted nonlinear signal on the real features, cold start; GREEN iff M - L lower CI > +0.01 on both seeds): True

| seed | L conc | M conc | M - L [95 %] |
|---|---|---|---|
| 0 | 0.4575 [0.4393, 0.4774] | 0.8375 [0.8308, 0.8438] | 0.3800 [0.3589, 0.4009] |
| 1 | 0.4555 [0.4367, 0.4755] | 0.8375 [0.8314, 0.8433] | 0.3820 [0.3610, 0.4026] |

### 2.2 C-LEAK and C-NOISE

#### T5. C-LEAK (NC and DAC first) with the matched noise-row comparator

| row | seed | arm | d unsafe-pick vs P2 [95 %] | changed picks | matched noise sigma | noise d unsafe (2 noise seeds) | noise changed picks |
|---|---|---|---|---|---|---|---|
| NC | 0 | M | +0.0007 [-0.0001, +0.0016] | 0.208 | 0.1 | -0.0001; +0.0003 | 0.165 |
| NC | 0 | L | +0.0003 [-0.0003, +0.0009] | 0.105 | | | |
| NC | 1 | M | +0.0003 [-0.0006, +0.0014] | 0.207 | 0.1 | -0.0001; +0.0003 | 0.165 |
| NC | 1 | L | +0.0000 [-0.0007, +0.0007] | 0.113 | | | |
| DAC | 0 | M | +0.0009 [-0.0004, +0.0024] | 0.207 | 0.1 | +0.0001; +0.0001 | 0.156 |
| DAC | 0 | L | -0.0005 [-0.0013, +0.0002] | 0.088 | | | |
| DAC | 1 | M | +0.0003 [-0.0010, +0.0016] | 0.191 | 0.1 | +0.0001; +0.0001 | 0.156 |
| DAC | 1 | L | -0.0007 [-0.0015, +0.0001] | 0.090 | | | |

Flags: NC AMBIGUOUS, DAC AMBIGUOUS

(The C-NOISE grid is in `raw/f10/noise.json` and the tail of `raw/f10/tables.md` T5; not repeated here.)

The C-NOISE grid (P2's logit plus Gaussian noise, no fit, 5 sigmas x 2 seeds x 4 rows) is in `raw/f10/noise.json` and `tables.md` (T5 tail). C-LEAK flags: **NC AMBIGUOUS, DAC AMBIGUOUS** (no LEAK on either row: no M or L cell has a lower CI of d-unsafe above 0). Random re-picking alone moves the unsafe-pick rate by an amount of the same order as the arms do: the noise rows of NC and DAC up to sigma 0.3 read -0.0004 to +0.0013, the arms' changes -0.0007 to +0.0009 (and a noise row can produce a separated +0.0006 by itself: NC, sigma 0.03, seed 0). No arm's pick-level unsafe-rate change is distinguishable from re-picking.

### 2.3 What the registered C2 did, and the five post-hoc diagnostics it forced

The registered C2 (cold start, labels permuted WITHIN each set, scored against the real labels) read **RED in 8 of 8 cells** (concordance - 0.5 from +0.001 to +0.32), where its own false-RED rate under an iid null is 13 to 17 % per cell (family rate 0.72). Eight of eight is not chance, so the cause had to be found. Five post-hoc diagnostics (Amendments 2 to 5), each decided in advance by its own reading rule:

| diagnostic | what it keeps / destroys | result |
|---|---|---|
| **C2x** (labels of a random OTHER set) | keeps each slot's label marginal; destroys feature-to-label links at every level | 8 / 8 RED, concordance - 0.5 = +0.10 to +0.33 (T8a). Amendment 2's committed rule (<= 1 RED and all points within +-0.01): NOT met |
| **C2y** (entry-level permutation) | destroys the slot marginal too | 2 / 8 RED; points -0.09 to +0.10, BOTH signs (T8b). Amendment 3's committed rule (>= 2 RED: "the pipeline-artefact reading stands, the FEATURE calls carry it as a stated caveat"): **MET, literally** |
| **index diagnostic** (z = slot index) | a monotone function of the slot | 0.487 to 0.507: the labels do not depend on the slot index monotonically (T9) |
| **C2z** (8 noise-trained L replicates per row) | the C2 rule's calibration under the null it is meant to face | per-replicate RED rate NC 0.375, EP 0.250, DAC 0.125, TTC 0.125; spread -0.056 to +0.078 with sign changes; 3 of 32 replicates significantly BELOW 0.5 (T10). Amendment 4's rule (rate > 10 % in any row): **the C2 rule is miscalibrated; its REDs carry no information about a pipeline artefact** |
| **slot prior** (a feature-free table of the mean label of each slot) | the slot marginal only | NC 0.829, TTC 0.8185 >= 0.80: Amendment 5's committed reading **met**: the C2 / C2x REDs are explained by the slot prior (T11) |

**The quantitative check (arithmetic on the reported points, no new fit).** C2x trains on labels from another set, so the only thing the probe can learn from the features is the slot marginal, and it should reproduce the slot prior. It does, to the third decimal where the labels are unconditional: NC (L arm) +0.3268 vs the slot prior's 0.8294 - 0.5 = +0.3294; TTC +0.3186 vs +0.3185; DAC +0.1949 (L) / +0.1722 (M) vs +0.1818. (EP does not compare: cross-set permutation separates its labels from its safe mask.) The within-set C2 (+0.23 on NC, +0.31 on DAC) keeps each set's label mean, and P2's in-sample features encode each proposal's own label, so a probe that regresses the set mean from proposal features also orders proposals by their real labels: reading (a) of Amendment 2. **That second mechanism is consistent with C2x, C2y and C2z but was not tested directly.**

**How the caveat is carried (stated, not softened).** By Amendment 3's literal rule the pipeline-artefact reading "stands" and the FEATURE calls carry it as a caveat. Amendments 4 and 5 supply a mechanism for every control that fired (miscalibrated lower-CI rule; slot marginal; set-mean regression) and a measured number matching one of them, but **none of them tests the decision statistic itself (M - L on the real labels)**. The direct evidence on the pipeline is C-ID (bit-identity at init), the reach proof, C-POS (a planted nonlinear signal is found: M - L +0.38) and the export check that recomputes all 20 concordances from the staged tables (`raw/f10/export_verify.json`, 20 of 20 equal to `probe_result.json`; this reuses the same scoring function, so it checks the copy, not the function). **So: the three FEATURE calls stand under the registered rule, with the C2 caveat on them, and no number of this probe is quotable without sections 2.3 and 3.**

#### T8a. C2x (Amendment 2; cross-set label permutation; POST-HOC, descriptive)

| cell | concordance - 0.5 | lower bound | RED |
|---|---|---|---|
| dac/L | +0.1949 | +0.1513 | True |
| dac/M | +0.1722 | +0.1265 | True |
| ep/L | +0.1260 | +0.1034 | True |
| ep/M | +0.0973 | +0.0824 | True |
| nc/L | +0.3268 | +0.3108 | True |
| nc/M | +0.3279 | +0.3114 | True |
| ttc/L | +0.3186 | +0.3003 | True |
| ttc/M | +0.3275 | +0.3095 | True |

M - L per row: DAC -0.0227 [-0.0389, -0.0086]; EP -0.0287 [-0.0488, -0.0067]; NC +0.0011 [-0.0018, +0.0035]; TTC +0.0089 [+0.0070, +0.0109].  RED cells: 8/8; all points within +-0.01: False.

#### T8b. C2y (Amendment 3; entry-level label permutation; POST-HOC, descriptive)

| cell | concordance - 0.5 | lower bound | RED |
|---|---|---|---|
| dac/L | +0.0592 | +0.0035 | True |
| dac/M | +0.0973 | +0.0243 | True |
| ep/L | -0.0068 | -0.0384 | False |
| ep/M | -0.0623 | -0.0957 | False |
| nc/L | +0.0073 | -0.0194 | False |
| nc/M | -0.0393 | -0.0796 | False |
| ttc/L | -0.0530 | -0.1011 | False |
| ttc/M | -0.0918 | -0.1452 | False |

M - L per row: DAC +0.0381 [-0.0074, +0.0880]; EP -0.0555 [-0.0677, -0.0410]; NC -0.0467 [-0.1001, +0.0024]; TTC -0.0388 [-0.0654, -0.0159].  RED cells: 2/8; all points within +-0.01: False.

#### T9. Proposal-index diagnostic (Amendment 3; z = proposal index vs the real labels; no training)

| row | all-64 concordance [95 %] |
|---|---|
| NC | 0.5033 [0.5013, 0.5053] |
| DAC | 0.5036 [0.4995, 0.5076] |
| EP | 0.4873 [0.4867, 0.4880] |
| TTC | 0.5072 [0.5048, 0.5097] |

#### T10. C2z (Amendment 4; L arm, noise-trained probes, entry-level permutation, 8 replicates per row; POST-HOC, descriptive)

| row | RED replicates (lower bound > 0) | significantly BELOW 0.5 (upper bound < 0) | rate RED [Wilson 95 %] | (concordance - 0.5): min .. max |
|---|---|---|---|---|
| NC | 3/8 | 0/8 | 0.375 [0.137, 0.694] | -0.0183 .. +0.0565 |
| DAC | 1/8 | 0/8 | 0.125 [0.022, 0.471] | -0.0245 .. +0.0443 |
| EP | 2/8 | 1/8 | 0.250 [0.071, 0.591] | -0.0558 .. +0.0782 |
| TTC | 1/8 | 2/8 | 0.125 [0.022, 0.471] | -0.0439 .. +0.0388 |

Max per-row RED rate 0.375; above 10 %: True.

### 2.4 A-RES (zero training; adopted from `DRAFT_PREREG_NCDAC_RESOLUTION.md`)

#### T6. A-RES (zero training; EP-R1 f10 rows applied; P2 stored NC/DAC logits)

| seed | n V | n S | n D | frac(V, m >= 0) | median m (V) | AUC_V (safe > unsafe, top-8) | AUC all | share V new = best pNC*pDAC | frac(D, m >= 0) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 236 | 104 | 11889 | 0.0720 [0.0349, 0.1202] | -0.09036517340883636 | 0.7697 [0.7206, 0.8160] | 0.6695 [0.6460, 0.6956] | 0.0212 [0.0049, 0.0405] | 0.6292 [0.5729, 0.6765] |
| 1 | 237 | 103 | 11879 | 0.0717 [0.0349, 0.1195] | -0.09032984717805295 | 0.7703 [0.7221, 0.8163] | 0.6695 [0.6460, 0.6956] | 0.0211 [0.0049, 0.0403] | 0.6283 [0.5722, 0.6754] |

**Flag: R-OVER** ({"auc_V": {"0": 0.7696730147323033, "1": 0.7702750982493748}, "frac_V_m_ge0": {"0": 0.07203389830508475, "1": 0.07172995780590717}}).  Controls: {"C0_identity": true, "C1prime_shuffled_safe_flags": {"auc": {"B": 10000, "hi": 0.5085608686656398, "lo": 0.4798970968148301, "n_nan": 0, "point": 0.4939474461175081}, "covers_half": true}, "C2prime_independent_restatement": true, "C3_mutation_sign_of_m": {"flipped": 0.9279661016949152, "fraction": 0.07203389830508475, "red_as_required": true}}

Flag **R-OVER**: fraction(V, m >= 0) = 0.072 [0.035, 0.120] < 0.50 and AUC_V = 0.770 [0.721, 0.816] >= 0.70 on both seeds (|V| = 236 and 237 >= 100). All four controls passed (C0 identity: empty V and S, P2's picks bit for bit; C1' shuffled safe flags: AUC 0.494, interval [0.480, 0.509] covers 0.5; C2' independent restatement reproduces; C3 mutation of the sign of m turns the fraction to 0.928, RED as required). EP-R1's rows were applied, not fitted (pins verified before use), and are in-sample for both the EP row and P2.

## 3. The slot prior: what it is, what it shows, what it does not change

**Definition (Amendment 5).** For each row, the score of (set, slot j) is the mean label of slot j over the valid entries of the sets of the OTHER four folds (EP: safe entries). It uses no scene information, no feature and no training; it is scored with the registered concordance and the same bootstrap.

#### T11. Slot-prior diagnostic (Amendment 5; the leave-fold-out mean label of each proposal SLOT, no scene information, no training; POST-HOC, descriptive)

| row | SLOT PRIOR all-64 concordance [95 %] | P2 ref | P2 - slot prior | L seed 0 - slot prior | L seed 1 - slot prior | slot-marginal range (fold 0) |
|---|---|---|---|---|---|---|
| NC | 0.8294 [0.8129, 0.8451] | 0.9165 | +0.0871 [+0.0755, +0.1003] | +0.0863 [+0.0748, +0.0997] | +0.0861 [+0.0747, +0.0996] | 0.771 .. 0.985 |
| DAC | 0.6818 [0.6458, 0.7143] | 0.9214 | +0.2396 [+0.2025, +0.2827] | +0.2401 [+0.2028, +0.2831] | +0.2402 [+0.2029, +0.2832] | 0.792 .. 0.955 |
| EP | 0.9794 [0.9776, 0.9808] | 0.6956 | -0.2839 [-0.3028, -0.2666] | -0.0178 [-0.0211, -0.0148] | -0.0182 [-0.0217, -0.0150] | 0.557 .. 0.983 |
| TTC | 0.8185 [0.8009, 0.8362] | 0.8812 | +0.0627 [+0.0538, +0.0733] | +0.0621 [+0.0528, +0.0730] | +0.0623 [+0.0530, +0.0733] | 0.656 .. 0.965 |

Committed reading (Amendment 5): slot prior >= 0.80 on NC (0.8294) and TTC (0.8185) => C2 / C2x REDs explained by the slot prior: **True**.

**What it shows.** (1) The 64 proposals of a set are not exchangeable: slot j carries a scene-independent TYPE (what a slot means, presumably the planner's anchor vocabulary index, is INHERITED from the planner design and not verified here), the features encode it, and the label distribution depends on it (slot-marginal ranges in the last column). (2) **A feature-free lookup reaches 0.83 (NC), 0.82 (TTC), 0.68 (DAC) and 0.98 (EP) concordance.** The scene-dependent increment of P2 / L over the lookup is +0.087 (NC), +0.062 (TTC) and +0.24 (DAC), all CIs excluding 0; for **EP the lookup is above both probes** (L -0.018, M about -0.012 by arithmetic on the reported points: M - L closes about 30 to 35 % of the gap between L and the lookup). (3) The EP row is therefore almost entirely slot-determined; a CAPACITY call on EP would have been about decoding the slot, not about scene information, and a slot embedding delivers that with no head capacity.

**What it does not change.** The registered calls are on M - L and on L - P2, differences of two probes that read the same slot information from the same features; they do not move. What it changes is the reading: "feature-limited" here means *the cached 256-d features do not carry more of what the row needs than the linear row already extracts*, and the amount "carried" is, for NC and TTC, mostly the slot. The scene-dependent part is the increment in the last-but-two columns.

## 4. Outcome row and next arm

**Outcome: MIXED.** FEATURE rows NC, DAC and TTC close their head-row line (lever: the trunk, P6 about 25 Oct). EP: **INCONCLUSIVE -> re-run once at f = 25 %** (cost in section 0, item 1). Nothing else is committed. The probe never read the dev gate or navtest; a successor registration owns that.

## 5. Evidence classes and limits (stated now)

* **MEASURED (ours):** every number in the tables, from the artifacts in `MANIFEST.md`. **INHERITED, not re-verified:** EP-R1's row files (sha-pinned, verified before use), the bank and the ranked fit logs (sha256 re-verified at load by the landed `_setup`), `dt = 0.2 s` for the speed export (consistent with the first-point displacement of about 1.1 m), the meaning of a slot.
* **P2 and its trunk are in-sample** on these sets: the cached features may carry label information a flexible head can read and a linear head cannot. That inflates M - L toward CAPACITY, so a FEATURE call is conservative against it; no fit-side read can remove it, and no CAPACITY call was made on it.
* **One fraction (10 %), two seeds, one recipe** (M warm-started, weight-decayed, 20 epochs); no learning-curve exponent is fitted or quoted. C-POS tests only that this recipe sees a planted signal in M's own family.
* **Intervals answer "another draw of LOGS" only.** Two seeds sample head-fit noise; a separated interval is necessary and not sufficient for a lever effect (`H-ESTIM-SEED-1`).
* C2z ran the L arm only (M was not run, a stated limit). The within-set mechanism (set-mean regression) is not tested directly. The slot prior's leave-fold-out means use 4 of 5 folds, so each fold's prior is estimated from about 80 % of the sets.

## 6. Proposed GAC / BACKLOG rows (proposals only; nothing written into Project Steering)

| proposed row | text | evidence |
|---|---|---|
| GAC claim (ID by the Master Mind) | NC, DAC and TTC head rows are not capacity-limited on the cached 256-d S0 features: a hidden-256 GELU row does not beat a linear refit (M - L upper CI < +0.01, both seeds) and a linear refit equals P2's row (within 0.01). Status: SUPPORTED WITH CAVEATS (C2 caveat, section 2.3; P2 in-sample; one fraction, two seeds). Lever: the trunk. | `probe_result.json`, T1 |
| GAC claim | Head-row concordances include a scene-independent slot component: a feature-free slot lookup scores NC 0.829, DAC 0.682, EP 0.979, TTC 0.8185 (all-64, OOF over folds). Read every head-row concordance with it. MEASURED. | `slot_prior.json`, T11 |
| GAC claim | The registered C2 (lower-CI rule) is miscalibrated for noise-trained probes over P2's features: per-replicate false-RED up to 0.375 (L arm, 8 replicates per row). Do not reuse it. MEASURED. | `c2z.json`, T10 |
| BACKLOG | EP f = 25 % re-run (the committed arm), conditional on the slot-table comparator being registered first; cost about 30 min fit + 15 to 20 min load / gate / metrics, CPU only. Decision: Master Mind. | section 0, item 1 |
| BACKLOG | Successor registration "slot-aware head-row controls": a null calibrated on noise-trained fits of the same arm, the slot prior beside every concordance, and a slot-residual concordance (pairs compared after removing the slot marginal) as the scene-dependent read. 0 GPU. | sections 2.3, 3 |
| BACKLOG | Pick-level floor on `pNC * pDAC` as the comparator for any safety-row work (R0e-pick family); heads are not the bottleneck for EP-R1's violators (R-OVER). | section 2.4 |
| BACKLOG (note to FASTNEG-CENSUS) | The slot index is exported; condition on or report it beside any speed stratum. | section 0, item 3 |

## 7. Deviations and disclosures (all of them)

1. **Amendments 2 to 5 are post-hoc.** They add controls after the outcome tables were read, each with its reading rule committed before its numbers existed; none changes a call. Amendments 2 and 3 were first written with a wrong clock time (09:40 and ~09:45 instead of ~09:29 and ~09:41); I corrected the text and re-hashed BEFORE any C2x / C2y number existed (the shas above are the corrected ones).
2. **The first outcome was read from a premature `metrics` run.** The chain's `metrics` step waited for `c2_ttc_M_rep5.npz`, which a worker wrote early, so `metrics` ran at 09:20:46 with the TTC row's C2 incomplete. I read its call line at about 09:22, recognised the incompleteness, moved it aside (`probe_result_PREMATURE_c2_incomplete.json`, `metrics_premature.log`, `export_verify_premature.json`), waited for all 40 C2 cells, reran `metrics` at 09:26:44 and read the full tables at about 09:27. The calls are identical in the two runs (and every non-C2 number is byte-identical); only TTC's C2 flag could differ, and it cannot change a FEATURE call.
3. **Duplicate chain and duplicate cell.** An old `msys` bash child survived the kill of its parent and started a second chain; its export step crashed on the exclusive-create guard (`FileExistsError`) with no corruption, and I killed the stray tree by explicit PID. A second worker started `ttc/M/rep4` after the first had written it; killed by PID (the cell was written once, exclusive). No cell was overwritten.
4. **RAM waits.** The `psutil` guard waited repeatedly when the box's available RAM fell under 3 GB (other sessions' jobs); wall-clock figures for M cells therefore include waits in `main.log` (EP/M seed 1: 545 s in the log, 305 s of fit time in the table: the costing in section 0 uses fit seconds).
5. **EP clause (b) is unsatisfiable here** (section 0, item 1). I registered the rule as the brief stated it and report the consequence.
6. **Beyond the Amendment 5 text:** the table also reports the paired differences P2 - slot prior and L - slot prior (computed with the same estimator); the amendment committed the slot-prior concordance and the reading rule, not these differences.
7. **Files added after the gate** are not gate-bound and do not touch any fit behind a call: `export_oof.py`, `c2x.py`, `c2y.py`, `c2z.py`, `slotprior.py`, `run_c2_cells.py`, `gen_tables.py`, launch helpers. The gate-bound files were not edited after the token was minted.
8. **Operating standard:** CPU only (`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=8`), at most two of my Python processes at any time, no git add / commit / push, the read-only directories (`srttc_r1/`, `epr1/`, `epr1ub/`, `srttc_s0/`) not written, the full stack suite not run.

## 8. The Master Mind addendum (FASTNEG-CENSUS): files and the time the outcome was first read

* **File:** `raw/f10/oof_proposals_f10.npz` (86,733,457 bytes, sha256 `fc86c4c07444e3b94f7f8d04846410b88988abf6ab10bf3be55ab2f7f8785e8e`) with `oof_proposals_f10.meta.json` and `export_verify.json`. **It was written 2026-10-19 07:49:30, before any outcome was computed or read.**
* **Contents (15,808 sets, 64 proposals):** `oof_<row>_<arm>_s<seed>` for nc, dac, ep, ttc x L, M x seeds 0, 1 (the OOF logit of the replaced column `[15808, 64]`, copied byte for byte from the `main` stage's tables; sha256 of each input in the meta); `fold`; `token`, `log`, `step`, `rank`, `ckpt_step`, `set_index` (a SET is the tuple `(token, step, rank, ckpt_step)`: 8,922 tokens, 15,808 sets, 1 or 2 sets per token), `proposal_index` 0..63; `labels6 [15808, 64, 6]` (order `nc, dac, ep, ttc, c, ddc`); `p2_logits6 [15808, 64, 6]` (P2's frozen stored scores); `p2_top8`, `p2_pick_index` (P2's base pick under the frozen rule).
* **Speeds:** the bank units were dumped `--lean` and carry no trajectories (members of `C:/Users/Admin/srttc_s0/work/bank/*.sets.npz` and `*.fan.npz` checked; the `fan` proposals are empty arrays). The SOURCE TABLE does: `C:/Users/Admin/srttc_s0/work/sets/<log>.npz` (`traj3 [sets, 64, 20, 3]`, 117 stores, **USED**). `mean_speed_mps [15808, 64]` = path length through the ego origin / 4.0 s, and `p2_pick_mean_speed_mps [15808]`; a store row joined to a bank set only if `labels6` are exactly equal. `dt = 0.2 s` is INHERITED from `ref/model.py:94`. Verified: P2's pick and its speed equal on all sets (`export_verify.json`: ok).
* **Not done, by instruction:** no ratio, no stratum and no statistic by speed was computed or looked at.
* **First read of any outcome of this probe:** 2026-10-19 **about 09:22** (the per-row call line of the premature `metrics` run, written 09:20:46) and **about 09:27** (the full tables of the final run, written 09:26:44).
* **Adoption ledger** for `DRAFT_PREREG_NCDAC_RESOLUTION.md` (sha256 `fe83adb4938b05eae372feeb630d082e5b668edb864e1f8f788a4f283d44b7d3`): in the PREREG, section 5. The draft was subsumed and not run separately.

## 9. Manifest

See `MANIFEST.md` (every staged file with sha256 and size, the single-place artifacts listed separately). Staging root `C:/Users/Admin/par18/probe/out/`; the mapping to repo paths is in `C:/Users/Admin/par18/probe/out/paths.json`. Repo target root: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-headrow-probe/`.
