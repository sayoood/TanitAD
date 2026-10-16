# PREREG_S1D (DRAFT): does the NUMBER OF TRAINING LOGS limit the S1 drivable-area head? (dose-response at S1c's budget, then the extra-log arm it would justify)

> ⚠️ **DRAFT, NOT IN FORCE.** Drafted by the S1d-scope agent on 2026-10-16 (~21:45 Berlin) from `SCOPE.md`; every threshold below is written BEFORE any S1d number exists. The Master Mind finalises and registers it, or amends it, before any arm launches; after registration nothing changes except by a dated amendment appended at the end. Companion of `SCOPE.md`. Parent: `eval/PREREG_S1C.md` (K2, P1, the estimator, the seed policy), `eval/PREREG_AUX_DRIVABLE_S1.md`. Tier: **T0-like** (head perception on teacher-forced frames; never driving performance). STRATEGIC family: **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25) throughout.
> Evidence classes: MEASURED (artifact named) / INHERITED / ESTIMATED / HYPOTHESIS.

## 0. What changed between the brief and this draft (read first)

The brief asked for "x2 and/or x4 label logs". **Neither is constructible inside navtrain**: S1c's head already drew **64,146 frames from 1,161 of the 1,167 train logs** and the label store holds all 1,192 (`W/sim_s1c_draws.json`; `W/SCOPE.md` section 0, MEASURED). The "~100 label logs" regime belongs to the A2 probe (123 navtest logs, ~98 per fold unit). What exists: (i) **+1,167 nuPlan-val logs** outside navtrain (labels buildable on the CPU, pixels on no disk we hold, no head-only trainer path) = a true **x2 in logs**; (ii) no x4. So the draft registers the question the data CAN answer first: **a dose-response by REMOVING logs with the unchanged trainer**, which bounds what x2 could add, and registers the x2 arm (minority-city version) as a conditional follow-up whose launch it gates.

## 1. Question

S1c (`H1u10k-s0`, 10,000 updates x 8 rows) read K2 FIRES: navtest P1 IoU [0,40) **0.7230** [0.6865, 0.7591] on the 1,058 C3 tokens and **0.7638** [0.7341, 0.7910] on the natural navtest population, against the 0.80 bar; remaining gap **0.0770 (C3), 0.0362 (natural)**; held-out navtrain P1 0.8210 (`raw/2026-10-16-s1c-read/out/RESULT.md`, MEASURED). Per city (natural-weighted): LV 0.860, BOS 0.713, PIT 0.703, SG 0.632.

**Q: with the SAME head, frozen u400 trunk, labels, learning rates, budget (10,000 updates x batch 8 = 80,000 draws) and natural sampling as S1c, does HALVING the number of distinct training logs (584 of 1,167, per-city stratified) lower navtest P1 [0,40) enough that DOUBLING the logs could plausibly close the remaining gap to K2 on BOTH the C3 set and the natural navtest population?**

Hypotheses (HYPOTHESIS until read):
* **HS1D-1 (the claim under test):** the head is log-limited; halving the logs costs at least the remaining gap on both populations (row L1).
* **HS1D-0 (what I expect, stated so a miss is no surprise):** halving costs little (|Delta50| <= F, row L3), because S1c's held-out unseen-log P1 is already 0.821 and the navtest shortfall is carried by the minority cities (a composition effect, `SHIFT_LOCATE.md`), so the lever is city coverage and sampling, not log count.

## 2. Arms, comparators, controls

| | |
|---|---|
| **Stage 1 arm (the only one required)** | **`H1u10k-q50-s0`**: S1c's argv verbatim (`PK/raw/2026-10-16-s1c-build/ARGV_DIFF.json["H1u10k-s0"]`, 59 tokens) with exactly three differences: the arm name `H1u10k-q50-s0` (given to the launch gate, as S1c's `H1u10k-s0` was), the `--out/--dump-scores` paths, and **`--heldout-logs` = `W/dose/heldout_logs_S1D_q50.txt`** (607 lines = the 24 real held-out logs + 583 dropped train logs; sha256 `c804ff3258368b806af05b9fcafdd502cb7f975b6702b98ee5828d2c3ca257cc`). `--epochs` is NAMED `1.0` (the default, now must-name). Seed 0, H1 (feed off), profile = S1c's. ~4.3 A40-h. |
| Training pool of q50 | 584 logs (LV 366 / BOS 112 / PIT 58 / SG 48), **92,714 rows / 52,798 frames** (53.0 % / 53.1 % of S1c's pool); 0.863 draws per row at 80,000 draws. Selection: seed 20261016, per city by sha256(seed\|log), nested; list sha256 `7d4ec620...`. |
| **Comparator of record** | S1c's `H1u10k-s0` (x1): same seed, same init, same frozen trunk, same labels, same eval frames. Contrast Delta50(x) = P_S1c(x) - P_q50(x), paired on the same navtest frames. |
| Stage 2 arm (CONDITIONAL, section 5) | `H1u10k-q25-s0`: 292 logs, 47,426 rows / 27,056 frames, **`--epochs 1.69` NAMED** (at the default 1.0 the run would stop silently at 5,929 updates: MEASURED arithmetic, `W/dose/dose_logs.json`), sha256 `3874c65d...`. Run only if stage 1 reads L1 or L2. |
| Stage 3 arm (CONDITIONAL, section 6) | `H1u10k-x2m-s0`: S1c's pool + the **463 minority-city nuPlan-val logs** (125 BOS / 134 PIT / 204 SG; ~30.8k frames, ESTIMATED), same budget. NOT launchable today (pixels, trainer path); registered now so its bars cannot move. |
| **C1 identity (free)** | step-0 held-out `aux_eval` IoU = S1c's **0.1586791569036157** (+-1e-3). A mismatch means the trunk, data or init differ: not paired, no claim. The head init depends on the seed only. |
| C2 zero control | G12 weight-0 control exactly 0.0 (a launch-gate check already required). |
| C3 schedule (S1c's control name; not the C3 token set) | run-check: update N of N at lr 0.0, every eval tag and dump. **Plus the data event must read `train_sets` = 92,714 for q50 (47,426 for q25) and `heldout_sets` = 3,137**; any other number is INVALID (this is the check that the subset file did what it should). |
| C4 init floor | A2-style: trained minus the seed-0 INITIAL head on the same frames, 95 % lower bound > 0. |
| Deliberate-regression arm (instrument) | the same readout run with the q50 head's predictions y-flipped must read far below q50 (the AUXVETO `V-GT-YFLIP` pattern); a readout that cannot see that is not read. |

## 3. Reads (all on navtest; navtest governs, as in S1c)

* **P1 [0,40) m, all frames** on (a) **C3** (1,058 tokens) and (b) the **natural navtest population** (SHIFT_LOCATE weights: failure stratum 4.36 %, navtest's own city mix, ESS reported). **K2 is decided on the point estimate against 0.80 on BOTH populations**, as S1c's rule words it; the interval is context.
* **Dose contrast** Delta50(C3), Delta50(NAT) with the paired log-cluster 95 % CI (`head_fit_pairs`); gaps G_C3 = 0.0770, G_NAT = 0.0362 (S1c, MEASURED).
* Secondary, no decision: P1 bands, per-city P1, held-out navtrain P1 (24 logs), P2, P3, the 5,000-update eval (is the order of arms the same mid-run?).
* **Reference floor** F = 0.035 (C3), 0.023 (NAT): the largest pairwise difference between S1's three H1 seeds at 400 updates (`PREREG_S1C.md` section 3). A proxy only; the variance at 10,000 updates is unknown (HYPOTHESIS that it is no larger).

## 4. Committed outcomes and the NEXT lever for each (Rule Zero: no outcome ends at a verdict)

Read top-down. "Separated" = paired CI excludes 0 **and** |Delta| > F. With one seed and one log subset it is **necessary, not sufficient** (section 7). If the two populations disagree, the **C3 read governs** (as S1c section 4), both are reported.

| outcome | rule | what it means | NEXT lever, executed in the same run of work | what blocks it |
|---|---|---|---|---|
| **INVALID** | C1 fails, or the run-check refused, or `train_sets`/`heldout_sets` differ from section 2 C3, or the readout instrument fails its regression arm | no S1d read | find the cause (drift of trunk / data / environment / subset file); a re-run is a Master Mind / PI decision (~4.3 A40-h) | PI |
| **A0 ANOMALY** | P_q50(C3) >= 0.80 **and** P_q50(NAT) >= 0.80 | K2 reached with HALF the logs: S1c's miss was NOT a log-count limit; it is run-to-run variance or draw luck | replicate **S1c** (`H1u10k-s1`, 4.3 A40-h; one-line gate change already named in PREREG_S1C section 5) before anything else; if it also misses, the S1c-vs-q50 order is noise and the head's level is a seed lottery: hand the better head to S2 with that stamp | PI: A40 spend |
| **L1 LOG-LEVER LIVE** | Delta50 separated and > 0 on BOTH populations **and** Delta50(C3) >= 0.0770 **and** Delta50(NAT) >= 0.0362 | halving the logs costs at least the remaining gap; if returns diminish (HYPOTHESIS, tested by stage 2), the next doubling gains at most Delta50, which could close K2 | (1) **build the extra-label store now** (CPU, 0 A40: `W/build_extra_labels.sh`, ~15-45 min, tested); (2) run **stage 2 (q25)** to test concavity: gain(q25->q50) = P_q50 - P_q25 must exceed Delta50, else Delta50 is not an upper bound and the x2 forecast is stamped UNBOUNDED; (3) escalate the pixel acquisition for the 463 minority logs (a `--list` pass first) and the head-only trainer path to the PI; (4) replicate pair (section 7) | PI: pixels (a download, ~26 GB ESTIMATED, UNVERIFIED), A40; build agent: head-only path |
| **L2 LOG-LEVER REAL BUT SHORT** | Delta50 separated and > 0 on both populations but below the gap on at least one | doubling the logs cannot, even optimistically, close the gap on its own; it can contribute | rank the remaining gap `0.80 - P1` by lever: (a) **city-balanced draws** (zero new data; a sampler flag = a trainer change), (b) a **20,000-update** arm (within 2x of the fitted top, no extrapolation beyond), (c) the trunk lever on P6's checkpoint; the x2 minority arm only as a COMBINED arm with (a) | A40; trainer change; P6 checkpoint |
| **L3 LOG-LEVER FLAT** | Delta50 not separated on either population (CI includes 0 or \|Delta50\| <= F), C1 holds | at fixed 80,000 draws the number of distinct logs is not what limits the head | **close the log lever for the visual tap** (single seed, single subset: a null from one draw closes nothing alone; the price of closing it is the q50 subset replicate `q50b`, seed 20261017, 4.3 A40-h); go to (a) city-balanced draws and (c) the trunk lever; build NO extra labels | PI: A40 for the replicate and for (a) |
| **L4 FEWER LOGS BETTER** | Delta50 separated and < 0 on either population | not expected; subset or seed luck | defect hunt first (C1, C2, schedule, the per-eval trajectory, the kept-list sha), then the subset replicate `q50b` as the cheapest arm that separates "subset luck" from "log count" | PI: A40 |

K2 for the arm itself is always reported with its population named; a q50 or q25 arm is not expected to clear it (A0 is the only row that says what clearing means).

## 5. Stage 2 (conditional): concavity of the dose curve

Run `H1u10k-q25-s0` only if stage 1 reads L1 or L2. Committed reads: gain_a = P_q50 - P_q25 and gain_b = Delta50 = P_S1c - P_q50, each with a paired CI. **CONCAVE** (gain_a > gain_b, with gain_a - gain_b outside the CI of gain_b): Delta50 is an admissible optimistic bound for the next doubling and L1/L2 stand as written. **NOT CONCAVE** (gain_a <= gain_b): the x2 forecast is **UNBOUNDED**, Delta50 is a lower bound only, and the x2 arm is justified on L1 alone with that stamp. No exponent is quoted from three points (CLAUDE.md: below R^2 0.80 no quotable exponent; use the matched ratio).

## 6. Stage 3 (conditional, registered now): the minority-city extension `H1u10k-x2m-s0`

Launched only after L1 and after the PI has decided the pixels and the head-only path. Pool = S1c's pool + the 463 minority-city extra logs (token policy S1D-TOK: stride 60 from row 30, `W/make_extra_yaml.py`); budget 10,000 x 8, everything else S1c's. Predicted pool shares (ESTIMATED): LV 50.4 / BOS 14.6 / PIT 15.1 / SG 19.9 % (navtest natural: 33.5 / 30.7 / 20.1 / 15.8).

| outcome | rule | NEXT lever |
|---|---|---|
| **X-INVALID** | C1/run-check/data-event/pre-launch audit fails | find the cause |
| **X-O1 K2 CLEARED** | P_x2m(C3) >= 0.80 **and** P_x2m(NAT) >= 0.80 | replicate seed 1; hand the head to S2; the veto read (an AUXVETO amendment registers the arm, as A1 did for V-S1C) |
| **X-O2 PARTIAL** | Delta_x2m vs S1c separated and > 0 on both, bar missed | rank by remaining gap: city-balanced draws on the enlarged pool; a 20k budget; trunk lever |
| **X-O3 NO GAIN** | Delta_x2m not separated on either | minority-city label coverage is closed for the visual tap at +30.8k frames; go to city-balanced draws and the trunk lever |
| **X-O4 WORSE** | Delta_x2m separated and < 0 | defect hunt (pixel decode of the new logs, calibration, label alignment), then the registered replicate |

**Pre-launch place audit (committed):** for the 1,058 C3 tokens, the share with a training frame within 2 m (position + heading) is 0.775 for S1's pool (`SHIFT_LOCATE.md` d1, MEASURED). Recompute it for pool + extras. If it rises by more than **0.05**, the arm's reading is stamped **"place-coverage confounded"** (it still counts; the stamp travels with every number).

## 7. Seed policy (H-ESTIM-SEED-1)

* **ONE arm, ONE seed (0), ONE log subset** per stage-1 and stage-2 level. Until a replicate exists every Delta is stamped **"separated, single seed, single log subset, not yet a lever claim"**. The paired log-cluster bootstrap resamples navtest LOGS with the models held fixed; it is blind to (i) training variance and (ii) **subset variance (would another random half of the logs say this?)**, which this design adds. F is the cheap proxy for (i) at 400 updates only.
* The sufficient form is a replicate that varies what the CI cannot: **`H1u10k-s1`** (S1c's seed 1, prices training variance at 10,000 updates; already named in PREREG_S1C) and **`q50b`** (seed 20261017 subset, prices subset variance). Neither is pre-authorised; each is 4.3 A40-h and goes to the PI queue on the outcome, with the Master Mind's recommendation then. **L1 needs both to become a lever claim; L3 needs `q50b` to be closed.**
* A K2 verdict on an arm (P1 vs 0.80) is an absolute read of one trained head against a fixed bar and needs no replicate to be reported; "the number of logs is the lever" does.

## 8. Estimator

**Paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927** (`proxy_eval.boot`'s draw scheme; ratios of sums on the same resamples; `eval/aux_s1_readout.py`, the S1c pairing extended to the q arms). It answers **"would another draw of navtest LOGS say this?" ONLY.** Not another training run, not another log subset (section 7), not another inference run (the head is deterministic: ASSERTED). The natural-population read re-weights the same predictions with the SHIFT_LOCATE weights (the S1c autoread driver, arm names extended).

## 9. Parity, and how each item is checked (mechanically, not by assertion)

| invariant | check | status |
|---|---|---|
| **navtest eval tokens untouched** | (1) every label store the arm reads has `manifest.leakage == {token_intersection: 0, log_intersection: 0}` against `navtest.json` (12,146 tokens, 136 logs), enforced by the builder's `--navtest`; (2) the navtest list pin is the CR-stripped sha256 `3ba1c96488f157fd...` that S1c's chain and E10 already check; (3) the C3 token table is the banked one (identical sha to S1c's autoread) | dose arms: store unchanged (sha `4a17db67...`), leak 0/0 MEASURED; extras: smoke store leak 0/0 MEASURED on 241 keys, full build re-checked at its manifest |
| **train / validation log disjointness** | `make_dose_logs.py` asserts kept U real-held-out = {} and kept U navtest-logs = {} (both **0** MEASURED), the file contains all 24 held-out logs; at run time **G7: `train_rows_in_heldout_logs` = 0 and `len(ho_idx)` > 0**, and C3 above pins `train_sets`/`heldout_sets` | MEASURED for the subsets |
| **evaluation set identical across arms** | `ho_idx` is built from the `--heldout` SETS directory, not from `--heldout-logs` (`scorer_finetune.py:663-668`); the sets dir md5s equal S1c's (`MD5SUMS.txt`) and `heldout_sets` reads 3,137 | source MEASURED; run-time check committed |
| **extras disjoint from navtrain, navtest, held-out** | `inventory_db.json` computes the non-navtrain list against the yaml `log_names` block (1,192) and `navtest.json` (136): 1,167 logs, 0 navtest; re-asserted by the gate from the extra manifest before X-arms | MEASURED |
| **no place leak** | the section 6 audit; the `test/`-dir non-navtest logs (1,213) are NOT proposed for exactly this reason | committed |

## 10. Invalid-run and kill rules

* The launch gate (arm classes registered, `--heldout-logs` pinned by sha256 per class, `--epochs` must-name, the schedule-reach check), the run-check and C1-C4 decide validity. A run that ends early, whose last row is not update N of N at lr 0, or whose eval tags/dumps are incomplete banks nothing.
* Operational check (not outcome-bearing): `s_per_update` at accum 1 within S1c's 1.24-1.8 s and train hours within 3.4-4.7.
* No kill rule on the trajectory (S1c's reasoning: the arm is cheap enough to finish).

## 11. Not computed, non-claims

* LONGITUDINAL / LATERAL / TACTICAL: NOT COMPUTED as paired reads (no budget-matched N arm; `PREREG_S1C.md` section 8). **STRATEGIC: PAUSED**, never a pass, never omitted, never extended.
* No claim about the veto, E1-E3, or driving performance. Down-sampling is assumed symmetric to adding (HYPOTHESIS): nuPlan-val logs may differ from navtrain logs in place coverage, which is why stage 3 exists and carries the place audit.
* The A2 probe dose-response (dev box, in-domain, ~25-100 logs) is a different regime and is NOT part of this prereg's decision rule.

## 12. Open decisions for the Master Mind (not made here)

1. Accept the reframing (no x2/x4 inside navtrain) and register stage 1 as the dose arm. 2. Approve the gate work: arm class(es) `H1u10k-q50` (and `-q25`), per-class `--heldout-logs` pin, named `--epochs`, `parse_arm` in `aux_s1_readout.py` / `aux_navtest_forward.py`, tests (a smaller job than S1c's eleven patches; HYPOTHESIS: one build-agent session). 3. The A40 slot: after P6 (~7 days) or a PI reorder. 4. Whether to run the dev-box probe dose-response now (different regime, ~5 h shared GPU, a probe flag). 5. **Do not build the extra labels now**; run `build_extra_labels.sh` only on row L1. 6. Whether city-balanced draws (a trainer change, the lever the per-city table points at) should be drafted in parallel as `PREREG_S1E`.
