# PREREG NC-FASTCAL: does a speed-conditional (per-stratum) Platt calibration of P2's NC logit recover the EP-led composite's NC-unsafe pick rate and PDMS against P2's rule? (Research Lab agent for the Master Mind, 2026-10-19)

**Written BEFORE any quantity of this arm exists.** No `labels6` array, no logit array, no calibration parameter, no pick and no interval of this arm has been opened or computed. The sha256 of this file is recorded in `PREREG_NC_FASTCAL.sha256` before the first real-data stage runs, and every stage refuses to start unless the file (and every amendment) still hashes to its sealed value. A change comes only by a written amendment (`PREREG_NC_FASTCAL_AMENDMENT<k>.md`) saved and hashed before the number it governs.

- Tier: SELECTION, on SEEN scenes (`D-L1-G0-1`), P2 IN-SAMPLE. No dev table, no navtest, no GPU. 0 GPU, CPU only (`CUDA_VISIBLE_DEVICES=-1`).
- Design source: the Master Mind's brief for task NC-FASTCAL, and `MM_RULING_PICKRULE.md` section 4a (the committed ROW-MISS arm of `PREREG_PICKRULE_AUDIT_NC.md` section 7): the calibration must be SPEED-CONDITIONAL, because a monotone transform of one global NC logit cannot reorder confident errors. The prior is low; the arm runs because it is cheap and committed.
- The design (data, strata, calibration form, quantities, decision, controls) is FIXED by the Master Mind. Choices of mine are limited to the implementation details of sections 3 to 7 and are marked **[impl]**; none of them is chosen from any number.

## 0. Disclosures (what I had seen before sealing)

1. I read the landed `RESULT.md` of PICKRULE-AUDIT-NC (this arm's data is its data) and the GAC rows `D-REFE-PICKRULE-NC-1` and `D-REFE-FASTNEG-CENSUS-1`. So I know, for the **uncalibrated** composite and for P2: P2's NC-unsafe pick rate 0.0075, DAC-unsafe 0.0228, PDMS x100 90.365; the OOF EP-led composite's NC-unsafe rate 0.0104 / 0.0105 (seeds 0 / 1), paired excess over P2 +0.0029 / +0.0030, dPDMS x100 about -1.15 / -1.17 with lower CI about -1.47 / -1.50; the NC sigmoid at its 82 / 83 newly NC-unsafe picks has median 0.960.
2. I also read the audit's POST-HOC, DESCRIPTIVE section 5 (no authority there): a soft NC power gate `p_NC ** k` restored the NC level (k = 4: d NC-unsafe vs P2 +0.0004 [-0.0009, +0.0021]) while dPDMS vs P2 got WORSE with k (-1.2 to -1.6). That is a number about a different, monotone NC change on the same rows. It is the reason for my forecast (section 9), and it is not a value of any quantity below.
3. I have NOT opened `arrays.npz`, `speeds.npz`'s `code` array or any NC label for this arm. The only things opened before sealing are file hashes, the code I reuse, `strata.json` (label-free; cut points and row counts are published in the census), and the landed audit's `RESULT.md`.
4. Nothing was computed on synthetic data before sealing, except reading the audit's own synthetic false-RED estimates.

## 1. Question

`D-REFE-PICKRULE-NC-1` read ROW-MISS: at an EP-led pick, the NC row scores the newly NC-unsafe picks confidently safe (median sigmoid 0.960), and no hard NC veto recovers. `D-REFE-FASTNEG-CENSUS-1`: the fast-speed band (stratum Q4) holds 49.7 % of all unsafe rows. The committed next arm asks the cheapest remaining question about the NC row alone: **if the NC logit is re-scaled and shifted separately inside each speed stratum, fitted only on training folds, does the EP-led composite stop losing NC-unsafe picks and PDMS against P2's rule?**

## 2. Data (identical to PICKRULE-AUDIT-NC; READ ONLY)

- S0 bank, f = 10 %: 15,808 sets of 64 proposals, 117 logs, folds = `rank mod 5` (the arrays' `fold` and `log_fold`), `label_order = (nc, dac, ep, ttc, c, ddc)`, rule `navsim_v1`, ONE P2 checkpoint. Arrays: the landed audit's `C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz` (keys `z6` [N,64,6] float32 = P2's stored logits, `lab` [N,64,6] float32 = `labels6`, `top8`, `log_idx`, `fold`, `log_rank`, `log_fold`, `log_names`, `ep_OOF_s0`, `ep_OOF_s1`). The file's sha256 must equal `arrays_npz_sha256` of the audit's `prepare_report.json`; the audit's C-JOIN already proved the labels, logits, `safe`, `pdms` and top8 bit-equal to the landed `epr1.FitData`.
- **EP-led composite = EP-R1's c\* row (lr 1e-3, anchor 0.1, 10 epochs, batch 256 sets), regenerated OUT OF FOLD, seeds {0, 1}**: the arrays' `ep_OOF_s0` / `ep_OOF_s1`, produced by the audit with the landed `epr1.fit_row` / `apply_row` (the audit's C-ID2 proved them equal to `D-REFE-HEADFIT-F10`'s banked read). Not re-fitted here. The banked in-sample rows (the audit's SECONDARY source) are NOT used in this arm.
- **Pick rule = the audit's reimplementation of `planner.py:558-561`** (`pickrule_lib.rule_aggregate`, float64, first argmax). Pinned landed modules: `pickrule_lib.py` sha256 `555c0a3468acd94adeeeb10f47527644a2b9d49722f554fed2da1fb29668eaeb`, `epr1.py` `a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8`, `fastneg_lib.py` `0c0b399b37a47ee3e17f9d5484bc7f9521ff7dbe6bea37e8c7a38042668667ca`; each is hash-checked at load.
- **Bootstrap:** paired LOG-cluster bootstrap over the 117 logs, B = 10,000, seed 20260927, the SAME draws for every quantity (`fastneg_lib.boot_draws(117, 10000, 20260927)`, asserted equal to `epr1.boot_draws`), percentile 2.5 / 97.5, every rate a ratio of per-log sums, every difference computed on the same resampled logs. It answers "would another draw of LOGS say this?" and nothing about another training run of P2 or of the EP row (`H-ESTIM-SEED-1`). The Platt fit has no seed; the two seeds differ only in the EP column.

## 3. Strata (label-free, landed)

The census's `strata.json` (sha256 `67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6`) with its `speeds.npz` (sha256 `433fd274abdae0de12b027ca27a4c1178fa64d60c25cdb48eb9acae22ba103de`; array `code` [15808, 64] uint8): stratum `S` (every proposal of a set whose P2 pick has v_b < 0.5 m/s: 195 sets, 12,480 rows), and `Q1..Q4` by r = v_p / v_b with cuts 0.9076 / 1.0453 / 1.2228 (249,808 rows each). Stratum codes 0 = S, 1..4 = Q1..Q4. **Structural join checks (label-free, gating, C-JOIN-STRATA):** `speeds.npz`'s `key_log` equals the arrays' `log_names[log_idx]` set by set; its `pick` equals `top8[:, 0]`; `code` recomputed from `r` and the `strata.json` cut points equals the stored `code`; the rows per stratum equal `strata.json`'s `rows_per_stratum`. The strata are a function of speeds and of P2's pick only; they never depend on a calibration, an EP column or a label.

## 4. Calibration (PRIMARY): per-stratum Platt scaling of the NC logit

For fold k in 0..4, the TRAINING sets are all sets with `fold != k` and the HELD-OUT sets are `fold == k` (grouped by log, so the held-out logs never enter that fold's fit). For each stratum s in {S, Q1, Q2, Q3, Q4}:

- **Rows:** every proposal in a training set whose stratum code is s. **Inputs:** x = P2's NC logit `z6[..., 0]` cast to float64. **Targets:** t = the NC label `lab[..., 0]` in {0, 0.5, 1}, used as a SOFT target (the NC head is a soft-BCE head; 0.5 is not rounded). **[impl]**
- **Model:** `u = a_s x + b_s`. **Loss:** mean BCE-with-logits `mean( softplus(u) - t u )` over the stratum's training rows, no regularisation, no weights.
- **Solver [impl]:** deterministic Newton's method with the exact gradient and Hessian from (a, b) = (1, 0), Armijo backtracking (c = 1e-4, halving, at most 60), at most 100 iterations, converged when max |gradient| < 1e-10. The problem is convex in two parameters, so this is the same optimum an L-BFGS or closed-form solve reaches; a unit test checks it against `scipy.optimize.minimize(method="L-BFGS-B")` on planted data.
- **Fit failure rule [impl, fixed now]:** if a stratum has fewer than 1,000 training rows, or the Hessian is singular, or the solver does not reach max |gradient| < 1e-8, or the result is non-finite or |a| or |b| exceeds 50, that (stratum, fold) keeps the IDENTITY (a = 1, b = 0) and the event is reported in the result headline. The decision rule is NOT changed by a failure.
- **Application:** for every proposal of a held-out set of fold k with stratum code s, `z_NC' = a_{s,k} x + b_{s,k}` in float64. The five other columns are P2's (columns 0 is replaced; 1, 3, 4, 5 untouched; column 2 is the composite's OOF EP).
- **Descriptive companion, no authority (impl):** the same procedure with ONE stratum (every proposal in stratum 0), "global Platt". It is what the C-NULL replicates are expected to approximate.

## 5. Arms compared (each seed s in {0, 1}; the pick of each arm is the planner's float64 first argmax)

- **P2:** the rule on P2's stored logits (= the bank's banked pick; control C-ID1).
- **U_s (uncalibrated composite):** P2's logits with column 2 replaced by `ep_OOF_s{s}`. Reported BESIDE the result.
- **C_s (calibrated composite, PRIMARY):** U_s with column 0 replaced by the OOF per-stratum Platt NC logit of section 4.
- Descriptive, no authority: **G_s** (U_s with the global-Platt NC column), and **PC** (P2 with the per-stratum-calibrated NC column and P2's own EP, seed-independent).

## 6. Quantities and decision (FIXED by the Master Mind)

Per seed, for the arm X in {C_s, U_s, G_s, PC}, all against P2's pick on the same sets, paired, same draws:

- **dNC** := NC-unsafe pick rate of X minus that of P2 (a pick is NC-unsafe iff `lab[..., 0] == 0` at the pick; an NC = 0.5 pick is not NC-unsafe).
- **dDAC** := DAC-unsafe pick rate of X minus P2's (`lab[..., 1] == 0` at the pick). Reported; no authority over the decision.
- **dPDMS** := pick-level PDMS x100 of X minus P2's (`NC * DAC * (5 EP + 5 TTC + 2 C) / 12` of `labels6` at the pick, DDC at weight 0; `pickrule_lib.pdms_from_labels`). Label-lookup PDMS at the selection tier, not a navtest number.
- Also reported: the three levels, the number of picks changed against P2 and against U_s.

**Decision (fixed now), PRIMARY arm C_s, both seeds:**

- **RECOVERS** iff, on BOTH seeds, **(i)** the UPPER bound (97.5th percentile) of dNC(C_s) is `< +0.001` AND **(ii)** the LOWER bound (2.5th percentile) of dPDMS x100 (C_s vs P2) is `>= 0`.
- **NO-RECOVER** otherwise (a NaN bound is NO-RECOVER). The two conditions are also reported separately per seed (descriptive, so the reader can see which one binds). The uncalibrated composite U_s is reported beside the result with the same quantities; it has no authority over the decision.
- **Final label:** `VOID` if the C-NULL control fires on 2 or more of its 5 replicates (section 7); otherwise RECOVERS or NO-RECOVER.

**Committed next arm (by outcome, fixed now):**

| outcome | next arm |
|---|---|
| RECOVERS | a 0-train pick-rule registration for the dev gate (it reads SEEN scenes, `D-L1-G0-1`) |
| NO-RECOVER | NC calibration closes; the line goes to PRIVGEOM-SAFE (already queued) and the trunk |
| VOID | no next arm from this read; the null's firing is reported and the instrument is repaired first |

## 7. Controls (every one is built so it can fail; every reach check is stated on a PLANTED input, never on a property of the real data: MM ruling 1 of the audit)

- **C-JOIN-STRATA (gating):** section 3.
- **C-ID (gating):** (a) the identity calibration (a = 1, b = 0 in every stratum and fold) reproduces the uncalibrated composite U_s: the NC column is bit-equal to P2's and the picks equal U_s's on 15,808 / 15,808 sets, both seeds; (b) P2's rule on `z6` equals the bank's banked pick (`top8[:, 0]`) on 15,808 / 15,808 sets; (c) the uncalibrated composite's quantities (dNC, dDAC, dPDMS, the levels) equal the landed audit's `result.json` (`analysis.composites.OOF_s{seed}.sweep.none`) to 1e-12.
- **C-REACH (gating, planted):** (a) on a **synthetic planted bank** built inside the test and the gate (random logits, planted stratum codes, a planted per-stratum parameter table, the real `apply_calibration` code path), a planted shift `b = -3` in ONE stratum changes the NC logit on exactly the proposals of that stratum and nowhere else, moves the pick on at least one set and on fewer than all sets, and leaves every set that holds no proposal of that stratum unchanged; (b) the same planted parameter table applied to the REAL arrays through the same function changes the NC logit exactly on the planted stratum's proposals (a statement about the transform, not about picks), and the number of real picks it moves is reported INFORMATIONALLY (not a gate: it is a data property). A RED mutant of `apply_calibration` that ignores the stratum, or writes the wrong column, or uses another fold's parameters, must make (a) or the fold check FAIL (`test_ncfastcal.py`).
- **C-FOLD (gating, planted):** the fit of fold k does not read the labels of fold k: flipping every held-out label of fold k on a planted bank leaves fold k's parameters bit-identical; a RED mutant that fits on all sets changes them.
- **C-NULL (powered, `RETRACTION_LOG` R51):** the stratum codes are permuted WITHIN TOKEN (see below), 5 replicates, permutation seeds `20261019 + 7919 * rep`. The permuted codes are used both to fit and to apply; everything else (rows, EP columns, labels, folds, the bootstrap draws) is untouched; the full decision rule of section 6 is recomputed for the replicate (both seeds). A replicate FIRES iff it reads RECOVERS. **Report how often RECOVERS fires. VOID iff it fires on 2 or more of the 5.** *Within-token permutation [impl]:* the sets are grouped by `key_token` (`speeds.npz`); for each token in sorted order the codes of all its sets are pooled in row-major order and permuted with `rng.permutation(m * 64)` from one `default_rng(20261019 + 7919 * rep)` consumed token by token; a token with a single set is permuted within that set. This keeps every token's multiset of strata (scene-level stratum, e.g. the standstill scenes) and destroys the proposal-level link between speed ratio and stratum, so a firing null would mean the recovery is not speed-conditional. *Power:* the bar of RECOVERS is far from the real data (section 9), so the real-data null is expected to be silent for a weak reason; its POWER is shown on a planted bank in `test_ncfastcal.py` (a planted speed-conditional miscalibration makes the true strata fire and the permuted strata not; a planted globally-miscalibrated bank makes the permuted strata fire, which the branch table maps to VOID). Pre-declared descriptive companion with no authority: for the 5 replicates and for the true strata, how often condition (i) ALONE holds on both seeds.
- **C-FIT (reported):** every (stratum, fold) fit's convergence flag, iteration count, parameters and held-out BCE before and after. A fit failure falls back to identity (section 4) and is named in the headline.
- **C-REACH-DECISION (gating):** every decision function of sections 6 and 7 (the two conditions, the both-seeds `all`, NaN handling, the strict `< 0.001` and the `>= 0`, the VOID count, the next-arm map) is exercised by a branch table, and a RED mutation of EACH (comparison flipped, bound swapped, `all` replaced by `any`, `and` by `or`, a constant changed) must make the table FAIL (`test_ncfastcal.py`).

## 8. Descriptive diagnostics (pre-declared, no authority, no threshold chosen from them)

- Per (stratum, fold): a, b, training rows, training BCE before and after; pooled held-out BCE before and after, per stratum (does Platt improve the held-out NC log-loss at all?).
- At the picks of U_s on the held-out folds: mean raw NC sigmoid, mean calibrated NC sigmoid and the empirical mean of the NC label at the pick (winner's-curse reading: is the row overconfident at the SELECTED proposals specifically?), and the same for P2's picks.
- For C_s: the audit's Q1-style numbers: the number of sets where C_s's pick is NC-unsafe and P2's is not, the share of those whose OWN calibrated NC sigmoid is `< 0.5`, and the median calibrated NC sigmoid at them.
- G_s and PC on the section 6 quantities.

## 9. Forecast (a prediction, not a bar)

HYPOTHESIS, committed before any number: **NO-RECOVER, with probability about 0.97.** Reasons: (1) condition (ii) asks the OUT-OF-FOLD composite to beat P2's IN-SAMPLE pick-level PDMS with a lower CI of 0 or more, while the uncalibrated gap is about -1.15 x100 (disclosure 1) and an NC-only change did not close it in the audit's post-hoc D4 (disclosure 2); condition (i) asks the NC-unsafe excess to have an upper CI below +0.001, while the audit's strongest NC-only monotone change had +0.0021; (2) per-stratum Platt is fitted on ALL proposals, whereas the confident errors live in the selected tail, so an average re-scaling should barely move the selected proposals' sigmoids. I put about 0.15 on condition (i) alone holding on both seeds, about 0.01 on condition (ii) holding on both seeds. I put no weight on a firing C-NULL beyond the roughly zero that RECOVERS' own unreachability gives.

## 10. Limits stated in advance

- SELECTION tier, SEEN scenes, P2 IN-SAMPLE: the comparator's PDMS is optimistic on these sets and the dPDMS of an out-of-fold composite against it is biased AGAINST the composite. Condition (ii) inherits that bias and is fixed by the brief, not softened here. A NO-RECOVER on (ii) therefore does not say the calibrated composite is worse than P2 out of sample.
- Label-lookup PDMS, not a closed-loop or navtest number. Tier: T0-class selection read. No capability claim.
- One P2, one S0 bank, one f = 10 % corpus, two EP-row training seeds, 117 logs: the interval answers "another draw of LOGS" only (`H-ESTIM-SEED-1`); the seeds are a floor on the EP row's training variance, not on P2's, and the calibration itself has no seed.
- The NC = 0.5 entries (8,343 proposals) are fitted as soft targets and are neither safe nor unsafe in the pick-level NC-unsafe rate.
- The strata are those of the CENSUS, which depend on P2's pick (v_b); a deployed planner has no P2 pick before it picks. The strata here are a diagnostic partition of the S0 bank, not a deployable gate; a RECOVERS would register a 0-train pick-rule on the dev gate (section 6), which would have to define a deployable stratum first.
- Platt on all proposals calibrates the average, not the selected tail; the diagnostics of section 8 say whether that is the reason.
