# RESULT: EP-RELIAB-PICKS (Research Lab agent for the Master Mind, 2026-10-19)

**Outcome: EP-OVERPREDICTED, on both seeds, in both forms. VOID checks: none (every gating control passed; C-NULL fired on 1 of 5, VOID needs 2).**
**Dominant PDMS-loss term: DAC (share 0.63 / 0.60 of the loss, dominant in 98.8 % / 98.3 % of bootstrap draws; also dominant under Shapley).**
Evidence class: MEASURED. Tier: SELECTION on SEEN scenes (`D-L1-G0-1`), P2 IN-SAMPLE, label-lookup PDMS; T0-class, no capability claim, no eval-family claim (this is a diagnostic of a pick rule, not an eval of a model). 0 GPU, CPU only, 25 s.
Interval estimator: paired LOG-cluster bootstrap, 117 logs, B = 10,000, seed 20260927 (another draw of LOGS only; no training or inference variance, `H-ESTIM-SEED-1`). Seeds 0 / 1 are the two training seeds of EP-R1's c\* row (out of fold).
PREREG sha256 `e54067d4b16f598c1710064a7f01eafe825d19c01eadbe7124b86fa67e9a8e48` (sealed before any quantity existed) + Amendment 1 sha256 `9ed702999ae3e771392e76f75791cbc1c753a2a85c92c93f1254ebfa8be5a1ed` (C-POWER only, sealed before any real-data stage). Full tables: `raw/results/RESULT_tables.md`.

## 1. What was measured

| quantity (seed 0 / seed 1) | value |
|---|---|
| within-set slope of the EP label on the EP row's sigma, safe proposals (the scale gate) | **0.309 [0.297, 0.321] / 0.311 [0.299, 0.322]**: OFF-SCALE (bar: inside [0.9, 1.1]). The row's within-set spread is about 3.2 times the label's. |
| deciding form | the rank-free mass form D_M (no mapping fitted) |
| **D_M = M_p - M_e** (obs / pred EP mass at P2's picks minus at the composite's picks) | **+0.1349 [+0.1114, +0.1640] / +0.1331 [+0.1101, +0.1618]**: OVER on both seeds |
| M_e (composite's picks) / M_p (P2's picks) | 0.845 [0.834, 0.855] / 0.980 [0.962, 1.002] (seed 0) |
| predicted EP gain at the composite's pick vs P2's pick (the composite's own row) | **+0.1240 [+0.0991, +0.1537] / +0.1219 [+0.0972, +0.1513]** |
| realised EP gain (labels6) | **-0.0106 [-0.0156, -0.0042] / -0.0113 [-0.0165, -0.0048]** |
| Delta = predicted - realised (the other form, no authority) | +0.1346 [+0.1132, +0.1596] / +0.1332 [+0.1119, +0.1579]: OVER (no form sensitivity) |
| same predicted gain read from P2's OWN EP row | -0.0071 [-0.0097, -0.0042] / -0.0076 [-0.0102, -0.0046] |
| dPDMS x100 (composite - P2), reproduces the landed audit to 1e-9 | -1.1455 [-1.4730, -0.8398] / -1.1736 [-1.5035, -0.8579] |

12,314 / 12,292 of 15,808 sets change their pick.

**Attribution of dPDMS x100, sequential NC, DAC, EP, TTC, C (seed 0; seed 1 in the tables):** NC -0.389 [-0.598, -0.228], **DAC -0.721 [-1.020, -0.471]**, EP +0.076 [-0.221, +0.427], TTC -0.116 [-0.273, +0.037], C +0.004. Shares of the loss: DAC 0.629 [0.418, 0.920], NC 0.339 [0.214, 0.493], TTC 0.101, EP -0.066 [-0.405, 0.195]. The Shapley sensitivity (all 120 orders) keeps DAC the largest (-0.478), then EP -0.293 [-0.528, -0.009], TTC -0.224, NC -0.154. DAC is the dominant term in both attributions; the EP term is NOT the loss in the registered order (its sign and size depend on the order, because EP is valued at the composite's safety multipliers).

**Reliability at the picks (descriptive; observed / predicted mass; the safety rows are P2's):** deficit (risk) form at the composite's picks vs at P2's picks: NC 0.895 [0.664, 1.168] vs 0.782 [0.584, 1.027] (this reproduces the register's R53 number to 1e-9); DAC 0.947 [0.837, 1.084] vs 0.933 [0.787, 1.125]; TTC 0.925 [0.762, 1.089] vs 0.885 [0.708, 1.070]. Every interval contains 1 and every paired composite-minus-P2 difference contains 0. The extra unsafe mass the composite accepts is about the mass the rows predicted: DAC observed +113 vs predicted +113.6 over 15,808 sets, TTC +114 vs +95.5, NC +51.5 vs +37.4. EP (positive form): 0.845 at the composite's picks vs 0.980 at P2's picks (paired -0.135 [-0.164, -0.111]); P2's own EP row at P2's picks reads 0.989 [0.981, 0.997].

## 2. Reading (what the numbers say, and the limits on saying it)

The replacement HYPOTHESIS (`D-REFE-PICKRULE-NC-1` correction, R53) has two parts and both are supported at the selection tier:
1. **The extra safety risk is estimated about correctly:** no safety row is detectably miscalibrated at either pick population, and the accepted extra unsafe mass is close to the predicted one. "About correctly" means not detectably miscalibrated; the intervals are wide (NC 0.66 to 1.17).
2. **The EP gain is predicted but not delivered:** the composite's EP row claims +0.12 and the labels show -0.011. The claim is concentrated where the row picks: its reliability is 0.98 at P2's picks and 0.845 at its own.

So the loss is not the EP lever delivering and the rule mis-spending it; it is a safety cost paid for an EP gain that is not there. DAC is where most of it lands.

**Limits, all stated in the prereg and none softened:**
- **P2 in-sample biases the realised gain down, hence both statistics toward OVER.** To remove the call the realised EP at P2's picks would have to be about 0.145 EP units lower out of sample per changed set (Delta's lower bound 0.113 x 15,808 / 12,314). P2's own EP row is almost exactly calibrated at its own picks (0.989), so I judge an optimism of that size implausible (ESTIMATED, not measured).
- **A selection effect alone also gives OVER.** An argmax over 64 proposals of an over-dispersed row (slope 0.31) lands on the proposals where the row over-claims most, with no other defect. This arm does not separate "slot prior" from "selection on noise"; it shows the over-claim is real and large. (Over-dispersion alone would imply a realised gain near 0.31 x 0.124 = +0.04, not the measured -0.011; the difference may be P2's in-sample optimism, which this arm cannot size.)
- The EP row is about 98 % slot prior (`D-REFE-HEADROW-PROBE-1`); this arm does not decompose the predicted gain into slot and scene parts.
- One P2, one S0 bank, two EP-row training seeds, 117 logs. The interval does not cover another training run.
- The landed PDMS formula has DDC at weight 0 and no TLC. It reproduces the tabulated PDMS bit for bit, but whether the planner rule's weights match the formula the labels are scored with is the committed next arm for a non-OVER outcome, so it was not audited here.
- Forecast check (committed before any number): scale OFF about 0.6 (happened), OVER about 0.55 (happened), dominant term TTC about 0.35 / DAC about 0.25 (it was DAC; TTC's share was 0.10).

## 3. Controls (all gating controls PASS; `raw/results/gate_report.json`, 36 tests `raw/logs/test_epreliab.log`)

C-ID (P2 vs P2): Delta, D_M, dPDMS, every attribution and Shapley term read exactly 0.0, n_changed = 0 (15 intervals). C-ID-RULE: rule reproduces the bank's pick on 15,808 / 15,808, planner pin PASS. C-ID-LANDED: dPDMS equals the landed audit's `result.json` to 1e-9, both seeds. C-ID-R53: the NC reliability equals NC-FASTCAL's `posthoc_diag.json` to 1e-9. C-PDMS: bit-equal to `epr1.pdms_np`. C-ATTR: contributions telescope to the per-set float64 total within 1e-12; Shapley efficiency within 1e-12; float64 total equals the landed float32 dPDMS (difference 5.5e-8, bar 1e-4). C-REACH: a planted label shift d moves Delta by exactly d x n_changed / N, a planted sigma shift likewise, on a synthetic bank and (as an identity of the arithmetic) on the real arrays; RED mutants fail. C-POWER (Amendment 1, neutral label field, real picks and real log structure): +0.10 reads OVER, -0.10 UNDER, 0 reads WITHIN [0, 0], for both forms and both seeds. C-SCALE: planted ON and OFF slopes, offset invariance. C-REACH-DECISION: branch table plus 20 RED mutations, all caught. Independent re-derivation with no shared code (`code/independent_check.py`): max difference 1.8e-15 / 7.2e-16 over predicted gain, realised gain, Delta, M_e, M_p, D_M, slope, dPDMS and the five sequential terms.

**C-NULL (powered, empirical):** picks shuffled within token, 5 replicates: **fired on 1 of 5** (replicate 4, UNDER, both upper bounds just below 0: -0.00004 / -0.00035; a chance event at the rule's own rate); VOID needs 2. E-NULL (200 more, descriptive): the registered rule's empirical any-fire rate on the real structure is 0.040 (nominal about 0.05; synthetic worst cell 0.074), P(2 or more of 5 fire) at that rate 0.015. The real D_M sits 35.8 / 35.6 null standard deviations from the null mean (null sd 0.0038), empirical two-sided p = 0.005 (the floor of 200 replicates).

**Deviations, all before any real-data quantity:** (a) Amendment 1: C-POWER as registered depended on the real baseline (a strongly OVER baseline could stop a planted -0.10 from reading UNDER); it was replaced by a label-independent neutral bank, found from the design while writing the tests. (b) The first run of the test suite had one over-strict assertion (shuffled picks must fire 0 of 5; the registered rule tolerates 1); corrected to the registered "fewer than 2". No library code changed. (c) Minimal detectable effect (descriptive): half-width of the Delta interval 0.023, of D_M 0.026.

## 4. Committed next arm, and what I propose

Committed (by outcome EP-OVERPREDICTED): EP-led head changes are CLOSED as a lever on these features. The EP lever moves to scene-specific EP (trunk / P6), and the Master Mind drafts a pessimistic-EP pick-rule registration (EP shrunk toward its slot prior).

Ready-to-register sketch for that arm (the Master Mind's call; 0 GPU, same arrays, `epreliab_lib` reusable unchanged): `sigma_shrunk = lam * sigma_comp + (1 - lam) * sigma_slot`, where `sigma_slot` is a lookup of the EP label's mean by proposal index fitted on training folds only and `lam` is fixed in the registration (not tuned on the read); the quantities are dPDMS against P2 and the same Delta / D_M / (R) block, so a pass needs both a recovered PDMS and a closed over-claim. A second, independent observation for the registration: DAC carries most of the loss, so a DAC-side lever (PRIVGEOM-SAFE) is a candidate to pair with it; that is my suggestion, not part of the committed table.

**Proposed GAC text.** New row `D-REFE-EPRELIAB-1` (2026-10-19), MEASURED (selection tier, SEEN scenes, P2 in-sample): "EP-OVERPREDICTED. The EP-led composite (EP-R1 c\* row, out of fold) predicts an EP gain of +0.124 [+0.099, +0.154] / +0.122 [+0.097, +0.151] at its picks against P2's and realises -0.011 [-0.016, -0.004] / -0.011 [-0.017, -0.005]. The EP row is OFF the label's scale (within-set slope 0.31); the rank-free mass form decides: D_M +0.135 [+0.111, +0.164] / +0.133 [+0.110, +0.162], OVER on both seeds, and Delta agrees. The safety rows are not detectably miscalibrated at either pick population (NC 0.895 [0.664, 1.168], DAC 0.947 [0.837, 1.084], TTC 0.925 [0.762, 1.089]). The dPDMS loss (-1.15 / -1.17 x100) is dominated by DAC (share 0.63 [0.42, 0.92] / 0.60 [0.39, 0.88]; dominant in 98 % of draws). C-NULL 1 of 5 (not void). Limits: P2 in-sample biases toward OVER; selection on an over-dispersed row also gives OVER. PREREG sha256 `e54067d4…`. `raw/2026-10-19-ep-reliab-picks/RESULT.md`." And in `H-REFE-RL1-NCRES`, `D-REFE-PICKRULE-NC-1`, `D-REFE-HEADROW-PROBE-1`: the R53 replacement HYPOTHESIS is promoted to supported at the selection tier (both parts), with the limits above.
**Proposed BACKLOG text.** Close `EP-RELIAB-PICKS` (outcome EP-OVERPREDICTED). Open `EP-PESSIMISTIC-PICK` (MM drafts the registration; sketch above) and `EP-SCENE-SPECIFIC` (trunk / P6; the slot-prior share of the over-claim is the number it must beat). Keep `RULE-WEIGHT-AUDIT` queued but unpromoted (the committed branch for a non-OVER outcome did not trigger).

## 5. Deliverable manifest

See `MANIFEST.md` (every file, bytes, sha256, location). Staged, never committed or pushed: `C:/Users/Admin/par18/epreliab/out/` with `paths.json` (`[repo path, abs source, "raw"]`) under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-ep-reliab-picks/`. Inputs read, never copied or modified: `C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz` (sha256 `29130bcd…`), the landed audit's `result.json`, NC-FASTCAL's `posthoc_diag.json`. Single-copy items: none (all outputs are in the staged package; the dev-box working copies are regenerable).
