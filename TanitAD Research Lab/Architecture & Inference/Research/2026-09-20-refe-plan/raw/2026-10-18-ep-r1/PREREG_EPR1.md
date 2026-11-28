# PREREG (DRAFT FOR REGISTRATION by the Master Mind; nothing run): EP-R1, a head-only retrain of P2's EP output as a CONDITIONAL expectation, with a pairwise ranking term among safe contenders

**Written 2026-10-18 (EP-R1 build agent), BEFORE any EP-R1 number exists.** No fit-set concordance, no pick-level read, no c\*, no dev-gate value and no navtest value has been computed by anyone for this arm.
The S0 bank was opened for HEADERS ONLY (member names, dtypes, shapes, plus the 6-string `label_order` schema array). No feature, logit or label value was read.
This file's sha256 is in `PREREG_EPR1.sha256`. Changes after registration are appended as `Amendment n`; the body is never edited.
**Needs from the Master Mind (headline):** (1) register this file and land the package; (2) decide whether EP-R1 shares SR-TTC's single navtest open (section 8, recommended) or needs its own; (3) the dev-gate and confirm appliers `epr1_dev.py` / `epr1_confirm.py` are NOT built here (section 11); (4) S0 is unfinished and nothing here runs before `S0_DONE`.

## 0. What was read, and what was not

| read | what was used | class |
|---|---|---|
| `GOALS_AND_CLAIMS.md` rows `D-REFE-EP-1`, `D-REFE-EP-2`, `D-REFE-SELRANK-1`, `D-REFE-FREEPOINT-1`, `D-REFE-SAFEOBJ-R0D` | text only (mirror tip `8ebe82d`) | register (INHERITED) |
| `raw/2026-10-18-ep-selection/` `RESULT.md`, `DIAGNOSIS.md`, `code/ep_common.py`, `ep_rules.py`, `ep_fit.py` | the diagnosis, the pre-registered gates F1-F3, the PDMS formula `NC*DAC*(5EP+5TTC+2C)/12` (reproduces the table, max abs diff 0.0) | MEASURED there |
| `raw/2026-10-18-sel-rank/` `PREREG_SELRANK_ARM.md`, amendments 1-2, `MM_DECISIONS_R1.md`, `S0_HANDOFF.md`, `code/srttc_r1.py`, `srttc_r1_gate.py`, `launch_r1.ps1`, `test_srttc_r1*.py` | the template this arm mirrors | design |
| one S0 bank unit `L1b95fda46f.sets.npz` | `.npy` headers of every member; `label_order` = `[nc, dac, ep, ttc, c, ddc]` (so **EP = column 2, NC = 0, DAC = 1, TTC = 3, C = 4**); `feat` float16 [479,64,256], `logits` float32 [479,64,6], `top8` int16 [479,8], `labels6` float32 [479,64,6] | MEASURED (headers) |

**Not read:** any value of `feat`, `logits`, `labels6`, `top8`; any dev-gate or navtest value for a fitted object; `S0_SUMMARY.json`. S0 had banked 15 units of the planned set when this was written.

## 1. What this rests on

| fact | value | where | class |
|---|---|---|---|
| P2's EP head is an UNCONDITIONAL expectation: violators are scored 0 in the label (EP is exactly 0 on 98.4 % of NC/DAC-violating proposals; the rest are NC = 0.5). `E[NC*DAC*EP]` 0.7216 vs mean pEP 0.7186 | held-out table, 1,705 tokens / 24 logs | `D-REFE-EP-1`, `DIAGNOSIS.md` (b) | MEASURED (SELECTION tier, descriptive) |
| the rule multiplies `pNC*pDAC` on top (`planner.py:558-561`), so violation risk is charged twice: the effective EP term is 0.70 vs a true 0.96 on the longest plans; bias of pEP on NC=DAC=1 plans -0.043 (shortest) to -0.168 (longest) | same | same | MEASURED |
| 1,698 of 1,705 tokens hold at least one safe proposal; best safe EP 0.978 vs the pick's 0.851; picking the best-safe-EP proposal would give +8.07 [6.23, 10.12] PDMS x100 (99.6 % of the oracle gain) | same | `D-REFE-EP-1` | MEASURED; an ORACLE over safe proposals, **not a reach** |
| within-token rank power on NC=DAC=1 plans: plan length 0.904, `pEP/(pNC*pDAC)` 0.791, raw `pEP` 0.363 | same | `DIAGNOSIS.md` (b) | MEASURED |
| four zero-training rules over P2's six outputs ELIMINATED AT FIT; best C4: PDMS +0.61 [+0.19, +1.20] in-sample, +0.51 [+0.13, +1.01] nested out-of-fold, but TTC -0.70 [-1.34, -0.08] separates adversely; moved picks gain about +3 and new violations cost about -50 | same | `D-REFE-EP-2` | MEASURED (fit set) |
| EP head w/o its violation signal, navtest pool bound | +0.801 [0.500, 1.103] = 13.6 % of the 5.875 pool; with it +1.573 | `D-REFE-SELRANK-1` | MEASURED, a ceiling not a reach |
| TTC contender concordance 0.722 [0.686, 0.759] on navtest | | `D-REFE-SELRANK-1` | MEASURED |
| the navtest look was left UNSPENT by EP-1 | | `D-REFE-EP-2` Master Mind note | register |

**Reading that motivates the lever.** The EP defect is a labelling defect: the label folds the violation event into progress, and the rule then charges that event a second time. The lever removes the event from the label (condition on NC = DAC = 1) so the existing rule `NC*DAC*(5EP+5TTC+2C)/12` reads each quantity once. This is a design hypothesis, not a result.

## 2. Hypothesis and the single lever

**H-EPR1.** Retraining only the EP output row of P2's `score_head` to predict EP *given NC = DAC = 1*, with a pairwise ranking term among safe contenders, raises P2's navtest PDMS by at least the bar in section 7 with no NC/DAC/TTC component separated adverse.

**One lever (single factor):** the EP output row `score_head.weight[2,:]`, `score_head.bias[2]` (257 parameters), initialised from P2's checkpoint. **Frozen, and PROVED frozen** (non-zero gradient on this row; exactly 0.0 on the other five rows and on the features; section 6, C1/C6): the trunk, the visual context, the proposal fan, the NC/DAC/TTC/C/DDC rows, and the selection rule `NC*DAC*(5EP+5TTC+2C)/12` (`refe/planner.py:558-561`). **The deployed rule is the planner's existing rule, now reading a conditional EP**: the double charge is removed by the label, not by a new rule.

**Consequence.** The fan is frozen, so the 64 proposals per token are bit-identical to P2's tables. Dev gate and confirm need NO harness scoring: the PDMS of the new pick is the lookup `pdms[token, new_pick]`, after SR-TTC's fan-frozen proof (C4).

## 3. Target and loss (exact; each is a named parameter of `epr1.py`)

For a set with proposals i = 1..64, banked labels `NC_i, DAC_i, EP_i` (float32, clipped to [0, 1]):

* **safe_i = (NC_i >= 1.0) and (DAC_i >= 1.0)** (`SAFE_MIN = 1.0`; NC = 0.5 is NOT safe). This is the EP-1 "geometry-safe" definition.
* **Conditional target:** `y_i = EP_i` on safe_i. Unsafe proposals carry NO target for this row.
* **Anchor** = mean over SAFE proposals (all 64 positions of every set in the minibatch) of `BCE_with_logits(z_ep_i, y_i)`. The denominator is the number of safe entries in the minibatch (not 64 x sets), so the anchor's weight does not depend on the batch's safe fraction.
* **Pair term** over the FROZEN top-8 contenders (membership = the bank's `top8`, computed once by P2's own rule, never refreshed; K = 8): pair (i, j) is admitted iff **safe_i and safe_j and y_i > y_j**. Loss `softplus(-(z_ep_i - z_ep_j))`, mean over the minibatch's admitted pairs (clamped at 1). Ties in EP (many are saturated at 1.0) carry no pair.
* **Total** = `pair_w * pair + anchor_w * anchor`, `pair_w = 1.0`, `anchor_w` from the grid.

Why a mean over safe entries and over pairs: the same reason as `MM_DECISIONS_R1.md` row 5. Why the pair term is restricted to the frozen top-8: it mirrors SR-TTC (the bank stores exactly that membership) and it is where the rule's margins are small (median top-two log-margin 0.0015, `DIAGNOSIS.md` (c)).

**Stated limit (design, not hidden).** (a) After the fit the pick can leave the frozen top-8, so the pair term trains the ranking only inside the OLD contender set; the anchor, which sees every safe proposal of all 64, carries the level correction outside it. (b) The EP logit of an UNSAFE proposal receives no gradient from either term, so the row can extrapolate on violators; the rule's `pNC*pDAC` factor is the only guard. This is the **violator-leak risk (outcome D-LEAK, control C7)** and the reason the pick-level read in section 5 exists.

## 4. Data, folds, bank verification (identical to SR-TTC R1; nothing re-selected)

* **FIT set = S0's bank at the nested size f in {10, 25, 50, 100} %, and f MUST EQUAL the f at which SR-TTC R1 wrote its c\*** so that both rows are fitted on the identical sets and logs (the joint secondary needs this). EP-R1 does not decide any S0 scale-up; that belongs to SR-TTC's A2.4.
* Fit-log selection: `ranked_fit_logs.tsv` (sha256 `a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1`), key `sha256("SR-TTC/A2/v1" + "\n" + log)`, S_f = first-fraction <= f, folds `rank mod 5`, minimum 20 logs per fold (A2.1, A2.2). Replacements only via an explicit `--replacements-file`; more than 5 % stops.
* Bank verification: S0_DONE present; every unit's sha256 equals the S0 manifest BEFORE `np.load`; `label_order` equals `[nc, dac, ep, ttc, c, ddc]`; stored `top8` equals the frozen rule applied to the stored logits; P2's EP row reproduces the stored EP logits to 1e-2.
* PathGuard: the fit never reads the dev-gate (held-out) or navtest tables (forbidden tokens, AST-checked), exactly as `srttc_r1.py`.
* **Parity:** the physicalai corpus (`physicalai-train-e438721ae894`, skip-hash `f09e44db`) is not touched; nothing here selects episodes.

## 5. Frozen decisions (every open point, settled before any number)

| # | point | decision (frozen) | parameter |
|---|---|---|---|
| 1 | grid | lr {1e-4, 3e-4, 1e-3} x anchor {0.03, 0.1, 0.3} = **9 configs**, K = 8, `pair_w` = 1.0. Same as SR-TTC | `GRID_LR`, `GRID_ANCHOR` |
| 2 | optimiser | Adam (0.9, 0.999), no weight decay, no schedule, **10 epochs, 256 sets per batch**; not tuned | `--epochs 10 --batch-sets 256` |
| 3 | seeds | **two seeds {0, 1}**, averaged inside selection; both required downstream (C3). One training seed per arm is a NAMED LIMIT (H-ESTIM-SEED-1) | `--seeds 0,1` |
| 4 | **c\* criterion** | **c\* = argmax over the 9 configs of the MEAN over seeds {0, 1} of the POOLED out-of-fold SAFE-CONTENDER CONCORDANCE of the EP row.** Concordance counts, over the admitted pairs of section 3 (safe_i, safe_j, y_i > y_j, both in the frozen top-8), `1[z_i > z_j] + 0.5 * 1[z_i == z_j]`, pooled over all out-of-fold sets, divided by the number of pairs. This is the EP analogue of `MM_DECISIONS_R1.md` row 2 | `--criterion pooled_oof` |
| 5 | fold-mean concordance | REPORTED for every config with `criteria_agree`; never switches c\* | built in |
| 6 | ties for c\* | first in grid order (lr ascending, then anchor ascending); tied configs listed | built in |
| 7 | **PICK-LEVEL OOF READ (new, disclosure)** | For every config x seed, and for c\* with a paired log-cluster bootstrap (10,000 resamples, seed 20260927): out-of-fold mean PDMS x100 of the pick under the planner rule with the EP column replaced, versus P2's pick on the same sets (**IN-SAMPLE for P2**, which trained on these sets: biased in P2's favour, so conservative against the new row); the unsafe-pick rate of both; PDMS from `NC*DAC*(5EP+5TTC+2C)/12` on the banked labels. **It never selects c\*, never gates anything.** It exists to (a) expose the violator leak (D-LEAK) and (b) decide how a dev failure is READ (section 8) | `pick_level` in the result |
| 8 | grid-edge disclosure | if c\* has lr = 1e-3 (the grid maximum) it is reported as `c_star_at_grid_edge`. Section 8 row E1 commits the consequence. The grid is never extended inside this arm (adding values would change a registered grid) | built in |
| 9 | LC read | the A2.4 learning-curve read, statistic Delta = c(100 %) - c(50 %) of the pooled OOF concordance of the EP row, prefixes {25, 50, 100} %, same bootstrap, delta = 0.010, both-rule `flat`. **It has no authority over S0 scale-up**; it only decides how a dev failure is read (A2.6 analogue) | `epr1.py lc` |
| 10 | P2's own row on the OOF pairs | reported, labelled **IN-SAMPLE for P2**; not a reference for any bar | built in |
| 11 | does a CPU head fit need a launch-gate PASS? | **YES** (CLAUDE.md, PI 2026-09-26; no size exemption). `epr1_gate.py` mints a token bound to {landed commit, argv sha256, S0 bank-manifest sha256, sha256 of `epr1.py`}; `launch_epr1.ps1` refuses without a matching one. Checks G-COMMIT, G-DVB, G-PATH, G-BANK, G-LIVE (one real-bank optimiser step; no outcome statistic), G-SUITE (`test_epr1.py` in full form on a clean `git archive` export: 0 failures, 0 skips, at least 46 tests), each with a deliberate-regression arm. A new trainer flag without a G-DVB entry is refused by the gate | `epr1_gate.py` |
| 12 | stages | `run` (R1), `lc`, `c2` (the shuffled-label negative control fit at c\*), each its own token | CLI |

## 6. Controls (each is a gate; a miss voids the arm, it is not a result)

* **C0 identity.** lr = 0 leaves the row bit-identical and reproduces P2's pick (`top8[:,0]`) on every set; pooled concordance equals P2's reference exactly; pick-level Delta is exactly 0.0.
* **C1 reachability** (the "built, tested, unreachable from its caller" class): argparse -> `Options` -> `RowHead` EP row -> the terms in `total_loss` -> `.backward()` -> `fit_row`'s `opt.step()`. Non-zero gradient on the EP row (weight and bias) for the full loss, the pair term alone and the anchor alone; **exactly 0.0** on all six frozen rows and on the features; the zero-weight control reads exactly 0.0; a leaky loss reading another column reads NON-zero (the probe can read a leak). **Mutation arms, each must go RED:** delete the pairwise line; remove the feature `detach`; cut `opt.step()`; **drop the safe mask of the anchor; drop the safe mask of the pair term.** Checked by the suite (`test_epr1.py`) and at launch by G-LIVE on a real bank unit.
* **C2 negative control.** The same fit at c\* with EP labels shuffled among the SAFE proposals of each set (`epr1.py c2`) must sit at chance (OOF concordance within 0.06 of 0.5 on the synthetic rig; on the real bank reported) and must give a dev gain <= 0 when its rows are applied at the dev gate.
* **C3 replicate** (H-ESTIM-SEED-1). Two fit seeds, same flags; a separated interval from one seed is necessary, not sufficient; both seeds must clear the bar.
* **C4 fan-frozen proof.** Inherited from SR-TTC's forward: re-dumped proposals equal the table's, max abs pose diff 0.0 m. EP-R1 changes no part of the fan.
* **C5 estimator.** Paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (`zero_tail.Boot`; `boot_draws` mirrors it and a test pins the equality). It answers **"another draw of LOGS" only**; the two fit seeds answer head-fit noise; inference is deterministic.
* **C6 conditional-target identity (new).** (i) The gradient on the EP row is **bit-identical** when the EP labels of UNSAFE proposals are replaced by any other values. (ii) On a batch where every proposal is unsafe the gradient on the EP row is **exactly 0.0**. (iii) Positive control: changing the EP labels of SAFE proposals changes the gradient. Mutations (anchor mask, pair mask) go RED on (i) and (ii).
* **C7 violator-leak disclosure (new).** For c\*, both seeds: out-of-fold unsafe-pick rate of the new pick vs P2's; mean new `sigmoid(z_ep)` on unsafe vs safe proposals of the out-of-fold sets. Reported; consumed by section 8.

## 7. Reach (ESTIMATED; arithmetic) and the bars (frozen at registration)

**Ceilings (MEASURED, not reaches):** EP head without its violation signal, navtest pool: +0.801 [0.500, 1.103] (`D-REFE-SELRANK-1`). Best safe-EP oracle on the held-out table: +8.07 (`D-REFE-EP-1`), a different construction (an oracle over safe proposals on all tokens) and not comparable. The only measured anchor for what a rule over the existing outputs reaches is EP-1's C4: +0.51 [+0.13, +1.01] out-of-fold on the held-out table, TTC -0.70.

| f (share of the 0.801 ceiling captured) | navtest pool gain |
|---|---|
| 0.10 | 0.08 |
| 0.25 | 0.20 |
| 0.50 | 0.40 |

`f` has no measured anchor on this rig. The +0.30 bar below needs f of about 0.37 of that ceiling **if the ceiling is the right one**; the EP-1 oracle suggests more headroom, the SELRANK ceiling less. **f near 0 is a live outcome and is committed as one (rows D-FLAT, E-NOTPROVEN).** Minimum detectable effect on navtest: about 0.17 for about 1,000 changed picks (`PREREG_SELRANK_ARM.md` section 6, ESTIMATED); EP-R1 may change more picks, which narrows it, so the count is reported, not used.

**Dev-set reuse, disclosed.** The dev gate is P2's own held-out table (24 logs, 3,137 sets / 1,705 tokens). EP-1 read it descriptively and fitted four zero-training rules on it. The EP-R1 row is fitted on S0's bank and never sees it, and no threshold below was tuned on it, but the DESIGN of this arm used its diagnosis. It is therefore a clean out-of-sample read of the *fitted object* and not a virgin set. EP-1's C4 numbers on it are a reference, not a bar. SR-TTC reads the same set: two arms, two separately registered bars, no selection between them.

**Bars.**

* **Fit-side (R1):** no bar. R1 produces c\* and the disclosures of section 5. It is not a result about H-EPR1.
* **Dev gate (read ONCE, after c\* and its sha256 are written):** the held-out pre-head features of SR-TTC's gate forward, the two rows of c\* (seeds 0 and 1), the planner rule, PDMS by lookup. Paired out-of-sample total gain **>= +0.30 PDMS x100 on BOTH seeds**, and C0, C1, C2, C4 pass. 24 logs cannot support a lower-bound rule. Reported alongside (not bars): NC, DAC, EP, TTC, C, DDC components per seed, changed picks, the unsafe-pick rate, and the JOINT number (EP-R1 row + SR-TTC's c\* TTC row of the same seed).
* **Navtest confirm (read ONCE; PRIMARY = EP-R1 alone vs P2's pick):** total paired gain point **>= +0.30** AND lower bound **> 0** AND no sub-score NC, DAC, TTC, C or DDC with an interval wholly below 0 AND no longitudinal or lateral family component separated adverse (`families6`, unpaired, conservative) AND the EP component not wholly below 0. Both seeds must clear. TACTICAL reported; **STRATEGIC PAUSED** (PI 2026-09-19, item 25), never a pass; distance-keeping UNAVAILABLE until lead-agent tracks reach `families6.py`.
* **Navtest SECONDARY (pre-registered, read on the same look, NOT used to decide the primary):** the JOINT arm, EP-R1 row + SR-TTC's c\* TTC row of the same seed and the same fraction, against P2's pick, and against EP-R1 alone, same estimator and same four families. If SR-TTC has no c\* row at the time, the secondary is **not computed, with that reason named**.

## 8. Outcomes, committed in advance, each with its next arm (Rule Zero)

**Where the navtest look comes from.** The look is the one EP-1 left unspent. `D-REFE-EP-2`'s Master Mind note held it for the SR-TTC-retrained scorer. **Recommended and assumed here: ONE navtest open for the whole REFE head programme**, which reads SR-TTC alone, EP-R1 alone, and the joint, each against its own bar registered before the open. Three registered readings, no selection among them, multiplicity disclosed. If the Master Mind instead gives EP-R1 its own open, only the "shared" clause changes. **Navtest opens for EP-R1 iff EP-R1 alone passes the dev gate.**

| id | trigger (all pre-registered) | meaning | next arm (named, with its cost) |
|---|---|---|---|
| **A** (build void) | any of C0, C1, C6 fails, or the gate does not mint | the instrument is wrong | fix and re-gate; no fit number is read until the gate passes |
| **E1** (grid edge) | c\* has lr = 1e-3 | the grid may not reach the level shift (Adam moves a bias at most about lr per step; 10 epochs of about 50 steps at f = 10 %) | **EP-R1b**: a NEW registration extending lr to {3e-3, 1e-2} at the same anchors and epochs, CPU, minutes. The dev gate is NOT read on the edge c\* before EP-R1b has run, unless the Master Mind records why |
| **D-PASS** | dev gain >= +0.30 on both seeds, C0-C2 and C4 pass | the head ranks safe contenders out of sample | navtest confirm (section 7) |
| **D-TRADE** | dev gain < +0.30 on a seed AND on that seed EP component > 0 and TTC component < 0 | EP-1's trade again: progress bought, TTC paid | (1) **EP-R1t**, CPU, minutes, NEW registration: admit a pair only if `TTC_i >= TTC_j` in the banked labels, so the row is never taught to prefer a faster plan that is worse on TTC. (2) Read the JOINT dev number already reported: if it clears +0.30 on both seeds a NEW registration **EP-R1J** names the pair the primary and spends the look. Nothing is promoted post hoc |
| **D-LEAK** | OOF unsafe-pick rate of the new pick exceeds P2's with the paired interval wholly above 0 on both seeds, whatever the dev gain | the conditional row extrapolates on violators | **EP-R1u**: a NEW registration adding an unsafe-proposal anchor with target 0 at weight u in {0.03, 0.1, 0.3}, which interpolates between this arm (u = 0) and P2's unconditional label (u = 1), CPU, minutes |
| **D-FLAT** | dev gain < +0.30, not D-TRADE, not D-LEAK, LC call FLAT | head/feature limited; more fit data of this kind is not the lever | **feature-vs-capacity probe**, CPU, minutes: a 2-layer EP probe on the cached features with the same folds. If its concordance stays <= the EP row's, the features do not carry EP-given-safe and the lever moves to the trunk (a conditional EP gradient into the trunk, the P6 pattern) |
| **D-DATA** | dev gain < +0.30, LC call RISING or INCONCLUSIVE, f < 100 % | DATA-LIMITED, no verdict on H-EPR1, **not a refutation** | the probe above (CPU) AND re-fit at the next nested size, which is SR-TTC's S0 increment (cost already named there); dev gate read #2, disclosed, the maximum |
| **E-ADOPT** | confirm primary clears (section 7), both seeds | the double charge was a binding constraint; one seed pair, direction not yet a lever claim | (i) the joint secondary; (ii) a third seed before any registry claim; (iii) hand the conditional EP row to the EPO/SAFEOBJ arms as their EP term |
| **E-SMALL** | lower bound > 0 but point < +0.30, or one seed misses | real but under the registered materiality | stack with the joint secondary already read; third seed; the changed-pick count sets the next dose |
| **E-NOTPROVEN** | point > 0, lower bound <= 0 | below the instrument's resolution | same as E-SMALL |
| **E-TRADE** | EP separated up AND TTC, NC or DAC separated adverse (primary fails by rule) | EP-1's trade survives the training-side fix | the **joint secondary decides the next arm, not the verdict**: if the joint meets the primary bar it is recorded as a SECONDARY-SUPPORTED PAIR LEAD and, because the look is spent, needs a confirmation set not yet opened (UNVERIFIED whether one exists on this rig; named blocker). Either way run **EP-R1t** (CPU) |
| **E-MECH** | EP component not separated up AND PDMS gain <= 0 | the double charge is not what holds the pick short on navtest | descriptive diagnosis on the already-open navtest table (no second confirm): the rule's margin terms by token class, as `DIAGNOSIS.md` (c) did on the held-out table |
| **E-REFUTED** | confirm upper bound < 0 | selection got worse | log the cause class (violator leak vs contender-set mismatch vs fit-to-navtest transfer), then the D-LEAK and D-FLAT probes |
| **J1 / J2** (joint secondary) | joint meets / misses the primary bar | **J1:** the pair behaves as the EP-1 trade analysis predicted. **J2:** TTC ranking does not repair the EP trade | J1: third seed on both rows; J2: the trunk-level lever (a joint conditional-EP + TTC gradient into the trunk) |

No row ends at a verdict. Rows D-TRADE, D-LEAK, E1 and E-TRADE name arms that are NEW registrations; none is run inside this arm.

## 9. Cost (ESTIMATED; arithmetic)

* **Fit: CPU only, minutes to tens of minutes.** The fitted object is 257 parameters on cached features; 9 configs x 2 seeds x 5 folds x 10 epochs of about 50 minibatches at f = 10 %.
* **No new GPU.** Sharing, not adding: S0 (bank) is SR-TTC's; the held-out pre-head features are SR-TTC's gate forward; the navtest pre-head features are SR-TTC's S3 forward (4.86 GPU-h ESTIMATED, `PREREG_SELRANK_ARM.md` section 4). **If SR-TTC's dev gate fails, S3 does not run, and EP-R1's confirm has no navtest features** (named blocker: EP-R1 would then need its own S3 of about 4.86 GPU-h, a PI/MM GPU decision).
* Memory: `feat` is 32 KB per set in float16, 0.5 GB at f = 10 %, 5.7 GB at f = 100 %.

## 10. What this does not claim

* Nothing about driving quality beyond T1 NAVSIM v1.1 PDMS (self-action open loop, not closed loop).
* Nothing about the zero tail (P6 and SAFEOBJ own it).
* Not a claim that progress is free: a progress-leaning gain moves the longitudinal family away from the human (best-of-64 drives at 1.177 x the human's progress, `PREREG_SELRANK_ARM.md` section 8) and is reported as such.
* The bars inherit SR-TTC's dev floor (+0.30, from the training-seed range 0.33-0.44 over nine SFT-4 scorers). A head-only fit on cached features may have a much smaller seed floor; the inherited number is conservative and INHERITED, not re-measured.
* "Safe" is geometry-safe (NC = DAC = 1), not TTC-safe: TTC stays an unconditional output and keeps its own double charge. SR-TTC fits TTC unconditionally too, so the joint secondary does not remove that one. Unaddressed here, named.

## 11. Not built in this package (flagged, not silently dropped)

* **`epr1_dev.py` and `epr1_confirm.py`** (apply c\*'s rows to the held-out / navtest pre-head features, pick by the frozen rule, PDMS by lookup, paired bootstrap, `families6`, C2 and C4 reads). They need SR-TTC's held-out and navtest pre-head files, which do not exist yet. The reusable parts are built and tested: `apply_row`, `picks_np`, `aggregate_np`, `boot_draws`, `paired_delta`. Their semantics are fixed by sections 7 and 8.
* Real-bank execution of anything. Nothing was run on the real bank beyond the header probe.
