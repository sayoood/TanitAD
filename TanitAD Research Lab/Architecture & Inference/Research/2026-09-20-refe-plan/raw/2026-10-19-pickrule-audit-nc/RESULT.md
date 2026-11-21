# RESULT: PICKRULE-AUDIT-NC (Research Lab agent, 2026-10-19). 0 GPU, CPU, selection tier, SEEN scenes, P2 IN-SAMPLE

**Registration:** `PREREG_PICKRULE_AUDIT_NC.md` sha256 `28947063…53ea7`, sealed 2026-10-19T09:52:47+02:00 before any real-data stage; Amendment 1 `28e349b8…96522` (09:57:42, forecast sentence + implementation notes); **Amendment 2 `9dda8c58…f79d` (10:05:18, the C-ID3 reach clause; see section 4)**. Both amendments were sealed before any Q1, Q2 or Q3 value existed. Every stage refuses on a hash mismatch.

## 0. Headline

**Outcome: ROW-MISS** (both seeds, PRIMARY source OOF; the secondary BANKED source agrees, so the result is not source-sensitive). **The NC loss of an EP-led pick change is NOT a composition effect of the planner rule.** Of the **82 / 83** sets (seed 0 / 1) where the EP-led pick is NC-unsafe and P2's is not, **none has an NC sigmoid below 0.5**: Q1 = 0 / 82 and 0 / 83. The NC row did not flag those picks: their median NC sigmoid is **0.960** (q25 0.888, q75 0.979). **VETO-RECOVERS = False** at every tau in {0.3, 0.5, 0.7, 0.9}. Committed next arm (registered, `PREREG` section 7, ROW-MISS): **NC calibration on the fast band (temperature or isotonic, out-of-fold)**, to be registered.

**Three things the Master Mind must know before using this** (details in sections 3, 4, 6):
1. ⚠️ **A control was amended AFTER it failed.** The registered C-ID3 reach clause ("a veto at tau = 0.5 changes at least one of P2's picks") FAILED on a data fact: P2's picks never have NC sigmoid below **0.6286** (0 / 0 / 1 / 62 picks below tau = 0.3 / 0.5 / 0.7 / 0.9). I amended it (Amendment 2) before any quantity existed, kept the registered clause in the gate report as INFORMATIONAL (FAIL), and gated the veto's reach on a planted, label-free shift instead (C-ID3b). **If you rule that the registered clause stands, the prereg's own text makes this read INVALID**; every number needed to apply that ruling is in `result.json`.
2. ⚠️ **"Rows frozen, no training" was read as "this arm tunes nothing".** V1's rows do not apply to S0 (section 1), so the EP-led composite is EP-R1's c\* row, regenerated out-of-fold by the landed, sha-pinned `epr1.fit_row` (10 small deterministic CPU fits, ~7 s each). Control C-ID2 shows it is bit-identical to `D-REFE-HEADFIT-F10`'s banked read. The literal zero-fit reading (the banked FINAL rows, in-sample) is the SECONDARY source and reads the same.
3. ⚠️ **Q1 = 0 / 82 has a degenerate bootstrap interval [0.000, 0.000]** (no resample can create a positive numerator). Any estimator gives an upper bound far below 0.5 (exact binomial 97.5 % one-sided bound for 0 / 82: 0.044), so the class does not depend on this.

## 1. What was run

* **Data:** S0's bank, 15,808 sets of 64 proposals, 117 logs (f = 10 %), folds `rank mod 5`, seeds {0, 1}; units sha256-verified against the manifest `6296dfc0…`; loaded by the landed `epr1.load_fit_data`; the full `labels6` joined by content (C-JOIN: EP column, `safe`, `pdms`, logits, top8, log index all bit-equal to `FitData`). P2 checkpoint `model_final.pt` sha256 `ea31db22…` (`prepare_report.json`).
* **V1's rows do NOT apply to S0's `feat` (MEASURED, `raw/results/v1_applicability.json`):** RL scorers (R0d u1000, arms A and B) have a 7 x 256 `score_head`; P2's is 6 x 256. Of the 76 `score_q_mlp` / `score_dec` tensors they share by name and shape with P2, **0 are bit-equal** (max abs difference 0.029 / 0.031). RL re-trains the scoring decoder, so its 256-d pre-head feature is not P2's `feat`. (R0 u1000 B itself is not on the dev box; its shared base with R0d is INHERITED from `D-REFE-RL1-HEADWISE`.) Fallback per the brief: **EP-R1's f10 refit EP row** (`D-REFE-HEADFIT-F10`), c\* = lr 1e-3, anchor 0.1.
* **Composite R-EP** = P2's stored logits with column 2 (EP) replaced; the other five columns equal P2's bit for bit (C-COL, all four (source, seed)). PRIMARY = OOF row, SECONDARY = banked final rows.
* **Pick rule:** `planner.py` lines 558-561 reimplemented (float64, lowest index on ties); source pinned (lines' sha256 `bcfaf349…`, `V1_W` at line 520).
* **Estimator:** paired LOG-cluster bootstrap over 117 logs, B = 10,000, seed 20260927, identical draws everywhere (asserted equal to `epr1.boot_draws`). It answers "another draw of LOGS" only (`H-ESTIM-SEED-1`); the two seeds are the EP row's training seeds, not P2's.

## 2. The registered quantities

**Q1 / Q2** (x = sets; the share of NC-unsafe picks, NC label 0, whose OWN NC sigmoid is < 0.5):

| source / seed | n A (EP-led unsafe, P2 not) | overridden (sigmoid < 0.5) | **Q1** | class | Q2: P2's own NC-unsafe picks (n 118) | median NC sigmoid at A picks (q25, q75) |
|---|---|---|---|---|---|---|
| **OOF seed 0** | 82 | **0** | **0.000 [0.000, 0.000]** | **ROW-MISS** | 0.000 [0.000, 0.000] | 0.960 (0.888, 0.979) |
| **OOF seed 1** | 83 | **0** | **0.000 [0.000, 0.000]** | **ROW-MISS** | 0.000 | 0.959 (0.890, 0.979) |
| BANKED seed 0 | 77 | 0 | 0.000 [0.000, 0.000] | ROW-MISS | 0.000 | 0.959 (0.890, 0.978) |
| BANKED seed 1 | 78 | 0 | 0.000 [0.000, 0.000] | ROW-MISS | 0.000 | 0.958 (0.893, 0.978) |

Decision rule: both OOF seeds ROW-MISS -> **ROW-MISS**. Q2 (P2's reference) is 0 as well, and that was fixed by the gate before I looked at it: no P2 pick has an NC sigmoid below 0.6286, so Q2 is exactly 0 whenever P2 has an NC-unsafe pick (118 do). Q2 has no authority over the decision. Descriptive sensitivities, no authority: with unsafe := NC label < 1 Q1 is still 0 / 97 and 0 / 98; the DAC analogue (DAC label 0, DAC sigmoid < 0.5) is **0.070 [0.025, 0.119]** (n 172) and 0.064 [0.025, 0.105] (n 171): also ROW-MISS-like, DAC is not flagged either.

**Q3: hard NC veto sweep on the OOF composite** (rows unchanged; rates over all 15,808 sets; d = vetoed composite minus P2 UNVETOED, paired; the veto falls back to the unvetoed pick when no proposal is eligible). P2 unvetoed: NC-unsafe 0.0075 [0.0052, 0.0102] (118 sets), DAC-unsafe 0.0228 [0.0162, 0.0309], pick-level PDMS x100 90.365 [89.150, 91.436]; **P2_hi = 0.01017**.

| seed | tau | NC-unsafe rate | d NC-unsafe vs P2 | DAC-unsafe rate | d DAC-unsafe vs P2 | **d PDMS x100 vs P2** | fallback / picks changed |
|---|---|---|---|---|---|---|---|
| 0 | none | 0.0104 [0.0070, 0.0145] | +0.0029 [+0.0009, +0.0054] | 0.0300 | +0.0071 [+0.0044, +0.0104] | -1.145 [-1.473, -0.840] | 0 / 0 |
| 0 | 0.3 | 0.0104 | +0.0029 | 0.0300 | +0.0071 | -1.145 [-1.473, -0.840] | 0 / 0 |
| 0 | 0.5 | 0.0104 [0.0070, 0.0146] | +0.0030 [+0.0010, +0.0055] | 0.0300 | +0.0071 | -1.152 [-1.487, -0.841] | 0 / 2 |
| 0 | 0.7 | 0.0102 [0.0069, 0.0142] | +0.0027 [+0.0009, +0.0050] | 0.0300 | +0.0071 | -1.133 [-1.450, -0.833] | 1 / 8 |
| 0 | 0.9 | 0.0100 [0.0067, 0.0140] | +0.0025 [+0.0007, +0.0047] | 0.0297 | +0.0069 | -1.095 [-1.425, -0.781] | 37 / 145 |
| 1 | none | 0.0105 [0.0071, 0.0145] | +0.0030 [+0.0010, +0.0054] | 0.0297 | +0.0069 | -1.174 [-1.503, -0.858] | 0 / 0 |
| 1 | 0.5 | 0.0106 | +0.0031 | 0.0297 | +0.0069 | -1.180 [-1.517, -0.862] | 0 / 2 |
| 1 | 0.9 | 0.0101 [0.0068, 0.0140] | +0.0026 [+0.0008, +0.0048] | 0.0295 | +0.0066 | -1.119 [-1.447, -0.800] | 37 / 142 |

(All tau, both sources and the P2-with-veto reference rows: `raw/results/RESULT_tables.md`. Seed 1 at 0.3 / 0.7 in that file.)

**VETO-RECOVERS = False.** Condition (ii), the dPDMS lower CI >= 0 vs P2, fails at every tau and seed (lower CI between -1.52 and -1.42); condition (i), NC-unsafe rate <= P2_hi, holds only at tau = 0.9 for OOF (and at every tau for BANKED, whose NC-unsafe point 0.0097 already sits under P2_hi although its paired excess is +0.0023 [+0.0003, +0.0046]). No tau recovers. The registered next arm is therefore the ROW-MISS arm, not a veto.

## 3. Controls

| control | result |
|---|---|
| C-JOIN | PASS: all six bit-equality joins; label counts NC 0 / 0.5 / 1 = 70,603 / 8,343 / 932,766, DAC 0 / 1 = 117,136 / 894,576 |
| C-ID1 rule reproduction | **PASS: float64 rule == banked pick on 15,808 / 15,808 (100 %)**, == `epr1.picks_np` 15,808 / 15,808; planner lines 558-561 hash to the pin. Diagnostic: the float32 (torch sigmoid) evaluation agrees on 15,806 / 15,808 |
| C-ID2 composite identity | **PASS, difference exactly 0.0:** seed 0 new / P2 PDMS x100 89.21936 / 90.36484, unsafe 0.040612 / 0.030364, 12,314 picks changed; seed 1 89.19121, 0.040486, 12,292 changed: all equal `D-REFE-HEADFIT-F10`'s banked read |
| C-PDMS | PASS: bit-equal to `epr1.pdms_np`; weights `V1_W = (5, 5, 2)` of `refe/planner.py:520` |
| C-COL | PASS x4: columns 0, 1, 3, 4, 5 bit-equal to P2's; EP differs on all 1,011,712 proposals |
| C-ID3 identity clauses | PASS: tau 0 reproduces the unvetoed picks; tau 1.01 falls back everywhere and reproduces them; non-fallback vetoed picks have sigmoid >= tau |
| **C-ID3 registered reach clause** | **FAIL (informational since Amendment 2): 0 of 15,808 sets change at tau = 0.5 on P2's rows**; min NC sigmoid at P2's picks 0.6286 |
| C-ID3b reach on a planted shift | PASS: NC logits shifted by -3 -> 18 sets changed, 241 fallback, of 15,808; a no-op and a switch mutant of the veto each fail it |
| **C-NULL** (labels permuted within sets; fires iff Q1 lower CI > 0.5) | **0 of 5 replicates fired** for every (source, seed); VOID = False. Null Q1 0.000-0.002 on n 494-568. Synthetic false-RED rate of the rule under a true share 0.10 / 0.30: 0.000; at the 0.50 boundary 0.020-0.025 |
| C-REACH | `test_pickrule.py`: 20 tests OK, a RED mutation of every decision function and of the rule, veto, Q1 threshold, unsafe definition and PDMS weights is caught; planted COMPOSITION, ROW-MISS, MIXED and STRUCTURAL cases read as designed (`raw/logs/test_pickrule.log`) |

⚠️ **Honesty on the null:** it cannot fire here because essentially no pick has an NC sigmoid below 0.5 (only 2 of 15,808 EP-led picks do). Its POWER is shown on synthetic data (a planted structural artifact makes it fire 5 of 5), not on the real rows, where the bar is far from reachable.

## 4. Amendments and disclosures

* **Amendment 1** (before any real-data stage): withdrew the clause "VETO-RECOVERS can only hold if COMPOSITION does" from the forecast (a veto at tau 0.7 / 0.9 can remove a pick with sigmoid in [0.5, tau)); pinned implementation notes (float64 sigmoid, permutation recipe, what a firing null means, C-ID2 tolerance).
* **Amendment 2** (after the first gate stop, before any Q1 / Q2 / Q3): the C-ID3 reach clause. **This is a control changed after it failed.** The first gate report is `raw/results/gate_report_attempt1_C-ID3_FAIL.json`; the second `gate_report.json`. The consequence for Q2 (fixed to 0 by the gate number) was stated in the amendment before Q2 was computed.
* **Forecast** (prereg section 9, not a bar): ROW-MISS 0.55, MIXED 0.30, COMPOSITION 0.15. ROW-MISS is what was read.
* The registered outcome rests on **one rig** (P2 in-sample, one S0 bank, f = 10 %), the EP row's two training seeds and 117 logs. It is a SELECTION-tier label-lookup read, **not a navtest number and not V1**: V1's own 24 NC-worse picks live on navtest and were not re-examined.

## 5. POST-HOC, DESCRIPTIVE (written after `result.json` existed; no authority; no tau or exponent is chosen from it)

`raw/results/posthoc_diag.json`, OOF seeds 0 / 1 (the two agree).

* **D1.** At the 82 / 83 EP-led NC-unsafe picks the NC sigmoid is **lower than P2's pick in the same set in 89 % of sets** (median 0.960 vs 0.985) but stays high; the minimum over ALL EP-led picks is 0.477, over all P2 picks 0.629. The rule is not letting low-NC proposals through; it is choosing proposals whose NC row reads 0.89-0.98.
* **D2.** Their NC rank among the composite's own top-8 contenders (1 = lowest NC): mean rank **5.87**, against 5.93 over all sets; only 2 of 82 sit at rank 1. The NC row ranks them like any other pick.
* **D3.** A flag "NC sigmoid < tau" on EP-led picks is informative but small: tau = 0.7 flags 9 picks, 5 NC-unsafe (56 %); tau = 0.9 flags 182, 31 NC-unsafe (17 %, base rate 1.04 %). **What the tau = 0.9 veto does to those 31:** 9 are kept because no proposal is eligible, **13 are replaced by another NC-unsafe proposal**, 9 become NC-safe; and it turns 3 safe picks into NC-unsafe ones. This is why the veto barely moves the unsafe rate: the replacements are near-twins the same row also misreads.
* **D4. A soft NC power gate, `p_NC ** k` in the planner's product** (k = 1 is the rule; k reported for all of {1, 2, 4, 8, 16}, none chosen): NC-unsafe rate 0.0104 -> 0.0096 -> 0.0079 -> 0.0068 -> 0.0068 (P2 0.0075); d NC-unsafe vs P2 +0.0029 [+0.0009, +0.0054] at k = 1, **+0.0004 [-0.0009, +0.0021] at k = 4**, -0.0006 [-0.0020, +0.0006] at k = 8. dPDMS vs P2 gets WORSE with k: -1.145, -1.195, -1.248, -1.329, -1.641. **So an NC-only monotone change can restore the NC level but not the PDMS gap**, which is not an NC effect (it also carries P2's in-sample advantage: the comparator is in-sample, the composite out-of-fold). The unsafe-rate rise is larger on DAC (+0.0071 [+0.0044, +0.0104]) than on NC (+0.0029 [+0.0009, +0.0054]) at the same picks.

## 6. Next arm and what remains (Rule zero)

* **Committed (PREREG section 7, ROW-MISS): NC calibration on the fast band (temperature or isotonic, out-of-fold), registered.** D1-D4 bound what to expect: a monotone NC change restores the NC level (k = 4) but costs PDMS, and the replacements for a veto are NC-unsafe near-twins, so calibration alone is unlikely to be the lever; it still must be run as registered because D4 is post-hoc and unregistered.
* The larger term of the unsafe-pick rise is DAC (+0.0071 vs NC +0.0029), which is the arm `PRIVGEOM-DAC` (committed by the census) addresses; this result raises its priority relative to NC calibration, not its design.
* Blocked: nothing in this arm. The NC calibration arm and `PRIVGEOM-DAC` are 0 GPU (CPU).

## 7. Proposed GAC / BACKLOG text (for the Master Mind to land; I did not edit them)

**GAC, new row:** `| **D-REFE-PICKRULE-NC-1** (2026-10-19) | **PICKRULE-AUDIT-NC reads ROW-MISS: the NC loss of an EP-led pick change is NOT a composition effect of the planner rule.** PREREG sha256 28947063…53ea7 + Amendments 1 (28e349b8…) and 2 (9dda8c58…, a control amended AFTER it failed, disclosed). S0, 15,808 sets / 117 logs, P2 in-sample, paired log-cluster bootstrap 10,000 / seed 20260927. EP-led composite = EP-R1 c* row regenerated OOF (identity to D-REFE-HEADFIT-F10 exact); V1's rows do not apply to S0 (7-output RL head on a re-trained decoder, 0 / 76 tensors equal P2). Of 82 / 83 sets where the EP-led pick is NC-unsafe and P2's is not, 0 have an NC sigmoid < 0.5 (Q1 0.000, class ROW-MISS both seeds; BANKED rows agree); median NC sigmoid at those picks 0.960. VETO-RECOVERS False at every tau in {0.3, 0.5, 0.7, 0.9}: dPDMS lower CI -1.42..-1.52. C-ID1 100 % (15,808 / 15,808), C-NULL 0 / 5 fired. Post-hoc: a tau 0.9 veto replaces 13 of 31 flagged unsafe picks by another unsafe proposal; an NC power gate k = 4 restores NC (+0.0004 [-0.0009, +0.0021]) at dPDMS -1.25; DAC carries the larger unsafe-rate rise (+0.0071 vs +0.0029). MEASURED, SELECTION tier, one rig. raw/2026-10-19-pickrule-audit-nc/RESULT.md | MEASURED (selection tier) |` and in `H-REFE-RL1-NCRES` append: `UPDATE 2026-10-19 (PICKRULE-AUDIT-NC): composition is REFUTED at the S0 selection tier (Q1 0 / 82, 0 / 83); the unsafe EP-led picks are NC-sigmoid 0.89-0.98 picks the row misreads (row resolution at the top contenders, not the rule). The census's within-token AUC (no NC fast-band deficit) and this pick-level miss are different statistics: AUC averages all pairs, the pick sits among the top-8 near-twins.`

**BACKLOG:** mark `PICKRULE-AUDIT-NC` DONE (ROW-MISS). Add **`NC-CALIB-FASTBAND`** (0 GPU, committed ROW-MISS arm): register temperature / isotonic NC calibration, OOF on S0 f10, with the pick-level NC-unsafe rate and dPDMS vs P2 as endpoints, BOTH outcomes committed, and a replicate arm (H-ESTIM-SEED-1); cite `posthoc_diag.json` D3b / D4 as the expectation to beat (k = 4 NC power gate). Re-prioritise **`PRIVGEOM-DAC`** above it (DAC carries the larger share of the unsafe-pick rise). Fix before reuse: none from this arm.

## 8. Manifest

See `MANIFEST.md` (every file, sha256, where it lives). Staged under `C:/Users/Admin/par18/pickrule/out/`; `paths.json` is the landing list. Nothing is committed or pushed. Held only on the dev box (regenerable by `pickrule_run.py prepare`): `run/prepare/arrays.npz` (66 MB; the bank arrays and the four EP columns).
