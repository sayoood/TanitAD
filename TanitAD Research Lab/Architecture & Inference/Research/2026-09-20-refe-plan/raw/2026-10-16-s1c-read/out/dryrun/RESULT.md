# RESULT: S1c auto-read. PREREG_S1C row **O3 NO GAIN**; AUXVETO arm V-S1C row **AV-HARM**

> ⛔ **DRY RUN ON S1's `H1-s0` ARTIFACTS: every row below is PLUMBING. It is NOT an S1c result and must not be quoted as one.** The arm read is S1's 400-update head, the 'comparator' is itself (a ZERO control), and no S1c number exists.


> **Evidence class: MEASURED** (this read, `s1c_autoread/1`, on the artifacts named in `RESULT.json`). Tier: **T0-like** for the K2 / budget reads (head perception on teacher-forced navtest frames, never driving performance); the NAVSIM tier (table lookup, base u400) for the veto. Arm `H1-s0`, repo `Sayood/tanitad-refe-s1aux`, forward device **cpu**. Read 2026-10-16T08:33:57Z. Nothing committed, nothing pushed.

## Headline

- **K2 (bar 0.80):** P1 [0, 40) m = **0.4957** on the 1,058 C3 tokens (CI [+0.4553, +0.5395]) and **0.5609** on the natural navtest population (ESS 575.4; CI [+0.5217, +0.5988]). The C3 read governs K2: **K2 FIRES**.
- **Budget contrast vs S1's `H1-s0` (paired, same frames, same log resamples):** Delta_C3 = **+0.0000** [+0.0000, +0.0000], Delta_NAT = **+0.0000** [+0.0000, +0.0000]; separated (CI excludes 0 and |Delta| > F): C3 **none** (F 0.035), NAT **none** (F 0.023).
- **PREREG_S1C row: O3 NO GAIN.** Stamp: *separated, single seed, not yet a lever claim*.
- **AUXVETO V-S1C (stratified navtest-population dPDMS x100): -0.8756 [-1.5017, -0.3061]** vs V-GT +0.3670 [+0.0151, +0.6385]: row **AV-HARM**; validity all 6 checks PASS.
- **STRATEGIC: PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` ITEM 25): not a pass, not omitted, not extended to the other three families.

## 1. Validity (PREREG_S1C section 2 and 4 INVALID row)

| check | result |
|---|---|
| C1 step0 held-out `aux_eval` IoU = S1 H1-s0's 0.1586791569036157 +-1e-3 | **holds**: 0.1586791569036157 (|diff| 0.0) |
| the pod's run-check (`run_check.json`) | ok |
| the arm's evidence is the registered arm (token profile `refe_aux_s1`, argv, start/done/eval tags, lr 0 at the end, data audit, head config) | consistent |
| forward: the scorer logits of the same pass are bit-identical to the banked C3 table | PASS (1058/1058 tokens) |
| INSTRUMENT: the comparator `H1-s0` read in the same call reproduces S1's banked navtest block (286 numeric leaves) and the SHIFT_LOCATE natural read | PASS, max |diff| 0 on cpu (tol 1e-09) |

## 2. K2 and the perception reads (navtest, 1,058 C3 tokens; navtest governs)

| read | H1-s0 | (comparator = self) |
|---|---|---|
| P1 IoU [0, 40) m, all frames | 0.4957 [+0.4553, +0.5395] | 0.4957 [+0.4553, +0.5395] |
| P1 IoU [0, 20) m, all frames | 0.5442 [+0.5088, +0.5822] | 0.5442 [+0.5088, +0.5822] |
| P1 IoU [20, 40) m, all frames | 0.4466 [+0.3984, +0.4981] | 0.4466 [+0.3984, +0.4981] |
| P1 IoU [40, 72) m, all frames | 0.4003 [+0.3428, +0.4595] | 0.4003 [+0.3428, +0.4595] |
| P1 [0, 40) on the u400-DAC-zero tokens / the others | 0.4000 / 0.5538 | 0.4000 / 0.5538 |
| P1 [0, 40) NATURAL navtest population (nat_p1; weights fixed) [+0.5217, +0.5988] | **0.5609** | 0.5609 [+0.5217, +0.5988] |
| P2 near-boundary SDF MAE (m) | 3.4401 | 3.4401 |
| P3 pooled DAC AUC (head) vs the scorer's | 0.7718 vs 0.7874 | 0.7718 vs 0.7874 |

**C4 init floor** (trained minus the seed-0 INITIAL head, same frames): P1 [0, 40) +0.3118 [+0.2755, +0.3514]; 95 % lower bound > 0.

Natural population: the 1,058 C3 frames re-weighted by IPF (200 it.) to navtest's own city mix (all 12,146 keys) x the failure stratum (529 / 12,146); ESS 575.4; margins {"BOS": 0.3072, "LV": 0.3346, "PIT": 0.2006, "SG": 0.1576}, failure stratum 0.0436. Per-city natural-weighted P1 (H1-s0): BOS 0.493 (n 310, 45 logs), LV 0.694 (n 233, 44 logs), PIT 0.461 (n 276, 21 logs), SG 0.359 (n 239, 13 logs).

Held-out navtrain P1 [0, 40) (24 logs; informational, navtest governs): update400: H1-s0 0.6307 [+0.5474, +0.6964].

aux_eval IoU by tag (the trainer's pooled held-out read, all cells): {"step0": 0.1586791569036157, "update400": 0.6071109087152684}; aux_bce first -> last [0.6219482663273811, 0.43245788022875786].

## 3. The budget contrast (S1c minus S1's `H1-s0`)

| population | Delta | paired 95 % CI (primary: weights fixed) | separated? (F) | sensitivity: weights re-fitted per resample |
|---|---|---|---|---|
| C3 (1,058 tokens) | +0.0000 | [+0.0000, +0.0000] | **none** (0.035) | n/a |
| natural navtest population | +0.0000 | [+0.0000, +0.0000] | **none** (0.023) | [+0.0000, +0.0000] (class unchanged) |

*Estimator:* paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927, models held fixed, logs resampled. It answers **"would another draw of LOGS say this?" ONLY**: it is blind to training variance (ONE seed, `H-ESTIM-SEED-1`: every S1c-vs-S1 difference is **separated, single seed, not yet a lever claim**; F is the cheap proxy for 400-update seed noise, the variance at 10,000 updates is unknown) and to inference variance (the head is deterministic: ASSERTED).

## 4. PREREG_S1C section 4, applied mechanically

| condition | evaluates to |
|---|---|
| valid (C1 holds, run-check ok, evidence consistent) | True |
| O1: P1_C3 >= 0.80 and P1_NAT >= 0.80 | False |
| O2: Delta_C3 and Delta_NAT both separated and > 0 (and not O1) | False |
| O3: Delta_C3 or Delta_NAT not separated | True |
| O4: Delta_C3 or Delta_NAT separated and < 0 | False |

**Row: O3 NO GAIN.** Remaining gap to the bar `0.80 - P1`: C3 +0.3043, NAT +0.2391 (worse population: C3).

**The row, verbatim from `eval/PREREG_S1C.md`:**

> **Rule:** Delta_C3 or Delta_NAT not separated (CI includes 0 or |Delta| <= F) on either population, and C1 holds
>
> **Meaning:** the training budget is **not** the lever for this tap at >= 80,000 rows; the representation (frozen trunk) or the label coverage limits the head
>
> **NEXT lever, executed in the same run of work:** **close the budget lever for the visual tap**; go to the trunk lever (head on P6's trained trunk / graft; PI decision) and the label-coverage lever. A null from ONE seed closes nothing alone: if the PI wants it closed, the replicate (seed 1) is the price
>
> **What blocks it:** PI: trunk lever; P6's checkpoint

MM note: Rule Zero applies. The O2 / O3 label-coverage lever is a 0-A40 CPU label build and does not wait for the PI (PREREG_S1C section 9, item 3); the replicate (`H1u10k-s1`), `N-u10k` and the trunk lever are NOT pre-authorised.

## 5. AUXVETO arm V-S1C (amendment A1), applied mechanically

| validity (section 9; any failure => INVALID) | result | detail |
|---|---|---|
| 1_B_reproduces_pick | PASS | {"n_equal": 1058, "of": 1058} |
| 0_population_identity | PASS | {"Z": 529, "N_sample": 529} |
| 2_A1_mask_iou_equals_harness_P1 | PASS | {"iou_veto_read": 0.4957035913034702, "iou_harness_P1_0_40": 0.4957035913034702, "abs_diff": 0.0, "tol": 1e-06} |
| 3_zero_control | PASS | {"pred_grid_40x32": {"ok": true, "n_vetoed_proposals": 0, "delta_pdms_x100": 0.0, "picks_equal_B": true}, "gt_grid_160x128": {"ok": true, "n_vetoed_proposals": 0, "delta_pdms_x100": 0.0, "picks_equal_ |
| 4_odac_bounds_rasters | PASS | {"odac_recovery": 343.0, "max_arm": 69.0} |
| 5_yflip_mutation_below_vgt | PASS | {"delta_vgt": 0.36702774726742615, "delta_vgt_yflip": -0.8704397232662655} |

Instrument identity (the tool reproduces the banked `AUXVETO_READ.json` for V-GT and O-DAC and V-PRED): PASS, max |diff| 0.

| decision list (section 6, read top-down, V-S1C in V-PRED's role) | evaluates to |
|---|---|
| AV-GO: V-PRED lower bound > 0 AND point >= +0.30 | False |
| AV-PROMISING: V-PRED lower bound > 0 AND point < +0.30 | False |
| AV-HARM: V-PRED upper bound < 0 | True |
| AV-CEILING-ONLY: V-PRED not separated AND V-GT lower bound > 0 | False |
| AV-NULL: V-PRED not separated AND V-GT not separated above 0 | False |

**Row: AV-HARM**. GT ceiling positive (V-GT lower bound > 0): the lever is raster quality (S1c). Stamp: *separated, single head seed, not yet a lever claim*.

**The row's committed next arm, verbatim from `eval/PREREG_AUXVETO.md` section 6:**

> read V-GT: if the GT ceiling is positive, the lever is raster quality (S1c), else AV-NULL's next arm

*Note (QUESTIONS Q9): amendment A1 adds no separate next-arm cell for V-S1C; the cell of the row V-S1C lands in is quoted. An item in it that names "S1c" is the arm just read (satisfied by this read); the remaining items are the committed next arms.*

### Arms (stratified navtest-population delta vs B, x100; Delta = 529/12,146 Delta_Z + 11,617/12,146 Delta_N)

| arm | dPDMS | Delta_Z / Delta_N | Z recovered (NC x DAC > 0) of 529 | B's pick falsely vetoed on N | every proposal vetoed (Z / N) |
|---|---|---|---|---|---|
| V-DRYRUN-H1S0 | -0.8756 [-1.5017, -0.3061] | +7.84 / -1.27 | 50 | 100/529 | 148 / 48 |
| V-GT | +0.3670 [+0.0151, +0.6385] | +11.81 / -0.15 | 69 | 2/529 | 12 / 0 |
| O-DAC | +2.5456 [+2.1961, +2.8800] | +58.45 / +0.00 | 343 | 0/529 | 68 / 0 |
| V-GT-YFLIP | -0.8704 [-1.6152, -0.1464] | +7.66 / -1.26 | 52 | 217/529 | 277 / 149 |
| V-PRED | -0.7882 [-1.4957, -0.1973] | +4.69 / -1.04 | 30 | 61/529 | 73 / 32 |

### The four families (AUXVETO section 8; per family, never pooled)

- **LONGITUDINAL:** dEP -1.5321 [-2.4101, -0.7870]; dTTC -0.6409 [-1.3969, -0.0566].
- **LATERAL:** dDAC -0.0158 [-0.4875, +0.3854]; dDDC +0.3165 [+0.0197, +0.7708]; Z-stratum recovery 50/529 (DAC-only variant 155); false-veto rate on N 100/529.
- **TACTICAL (selection quality, PDMS x100, stratified):** random 73.37, B 88.28, best-of-64 98.16, V-DRYRUN-H1S0 87.40, V-GT 88.64, O-DAC 90.82.
- **STRATEGIC: PAUSED** (PI 2026-09-19, ITEM 25).
- Also: dNC -0.2095 [-0.6725, +0.0992]; dcomfort +0.0000 [+0.0000, +0.0000].

Margin curve (reported, never gating): 0 m: dPDMS -0.9239 [-1.5445, -0.3511]; 0.25 m: dPDMS -0.8756 [-1.5017, -0.3061]; 0.5 m: dPDMS -0.8623 [-1.5033, -0.2793]; 1 m: dPDMS -0.7650 [-1.4273, -0.1511]; 2 m: dPDMS -1.0651 [-2.0612, -0.2105].

The CI answers *another draw of logs*; it is blind to the head's TRAINING variance (one seed, `H-ESTIM-SEED-1`) and to the tracker. V-GT-TRACKED is reported-only in the AUXVETO read (section 7) and is not re-read here.

## 6. The four families for S1c itself (PREREG_S1C section 8)

- **LONGITUDINAL / LATERAL / TACTICAL: NOT COMPUTED as paired reads.** There is no budget-matched N arm (arm B's scorer trained 25x the updates at batch 8), so a difference is not attributable to the head. An `N-u10k` control (~4.2 A40-h) would unlock them and is a PI decision. The V-S1C table above is the NAVSIM-tier family read of the HEAD's raster, which needs no N arm (the base is the banked u400 scorer).
- **Arm B's pick, DESCRIPTIVE (levels on the 3137 held-out sets, 24 logs; BUDGET-CONFOUNDED: the S1c arm's scorer pair trained 25x the updates of an N arm (10,000 x 8 rows vs 400 x 64 rows), so a level difference is NOT attributable to the head; levels only, no paired difference, no interval, no classification (PREREG_S1C section 8)):**

| arm | pick PDMS x100 | DAC | EP | TTC | NC | DDC | zero picks |
|---|---|---|---|---|---|---|---|
| H1-s0 | 90.709 | 97.545 | 84.666 | 96.780 | 99.091 | 96.047 | 105 |
| N-s0 | 90.763 | 97.609 | 84.708 | 96.780 | 99.091 | 95.983 | 103 |
| N-s1 | 90.541 | 97.354 | 84.295 | 97.004 | 99.123 | 96.286 | 110 |
| N-s2 | 90.893 | 97.514 | 84.841 | 97.004 | 99.315 | 95.681 | 99 |
- **STRATEGIC: PAUSED** (PI 2026-09-19, ITEM 25): not a pass, not omitted.

## 7. Provenance

- PREREG_S1C body sha256 `317584a3f7e82ee8...`; amendments: none. PREREG_AUXVETO body (pre-A1, CRLF form) `ef9f17fa26a9d818...`, A1 `2b26ccadd2095711...`; amendments: ['## Amendment A1 (2026-10-16 ~09:30 Berlin, Master Mind; written BEFORE S1c has trained and before any arm of this prereg has been read): arm V-S1C'].
- Estimator pins: paired log-cluster bootstrap, 10000 resamples, 95 %, seed 20260927; proxy_eval.boot's draw scheme. Answers 'another draw of LOGS' only.
- Heads read (aux_drv.pt sha256): H1-s0 `34a70829cd75...`, H1-s1 `3385b50b0a6c...`, H1-s2 `5ee9ab6c2352...`. Token: `PASS_d257647b7858.json` (commit d257647b7858).
- Fetch: 11 files, 107027270 B, every one verified (size + LFS sha256 / git blob id). Not fetched: model_sft_B.pt (1.29 GB; listed only), scorer_states_last.pt.
- Files: `RESULT.json`, `nat_p1_*.json`, `perception_navtest_*.json`, `perception_heldout_*.json`, `veto.json`, `desc_selection.json`, `forward_*/` (head_pred npz, identity.json, run.json), `fetch_manifest.json`, `autoread.log`.
