# PREREG: HEADROW-PROBE. Are the REFE head-row levers capacity-limited or feature-limited? (CPU, fit-side OOF, f = 10 %)

**Written 2026-10-19 (Architecture and Inference agent HEADROW-PROBE), BEFORE any probe code exists and BEFORE any probe number exists.** This file's sha256 is in `PREREG_HEADROW_PROBE.sha256`, written before the first fit. Changes after registration are appended as `Amendment n` (own file, own sha256, written before the first fit if they touch the design); this body is never edited.

**Design authority:** the Master Mind's brief `C:/Users/Admin/par18/probe.md` (section 2 below reproduces its design as fixed, and section 3 lists the choices this registration had to make where the brief is silent), plus the two additions that reached this agent with the brief: (i) EP-R1ub finished NONE-FEASIBLE (`raw/2026-10-19-ep-r1ub/RESULT.md`), so NC and DAC are reported FIRST; (ii) the draft `DRAFT_PREREG_NCDAC_RESOLUTION.md` (sha256 `fe83adb4938b05eae372feeb630d082e5b668edb864e1f8f788a4f283d44b7d3`) is SUBSUMED here, item by item, in section 5.

## 0. What was read, and what was not

| read | used for | class |
|---|---|---|
| `probe.md` (the brief) | the fixed design | design |
| `raw/2026-10-19-ep-r1ub/RESULT.md` and `DRAFT_PREREG_NCDAC_RESOLUTION.md` | the reason NC/DAC come first; the adoption ledger | design |
| landed `epr1.py` (sha256 `a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8`, equal to the blob at tip `1afc54f6fa5e`) and landed `srttc_r1.py` (sha256 `b841347d90c298ec597f10afbbd284d68258df55c98cbac8cbd9a0a444cc5d0b`, equal to the blob at the same tip) | loaders, PathGuard, folds, sha256 bank verification, the frozen rule, the bootstrap, the label conventions of the EP and TTC rows | code |
| `C:/Users/Admin/epr1/run/f10/epr1_result.json` | the pins for "the IDENTICAL fit logs": bank manifest sha256 `6296dfc0b03905038fbf132f82009f6b6d7e8d61fa9abfb65c9164f487a5f2a1`, ranked logs sha256 `a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1`, 117 logs, 15,808 sets, fold sets 3068 / 4014 / 3612 / 2505 / 2609; the two EP-R1 f10 rows `ep_row_seed{0,1}.npy` for the auxiliary read A-RES (full sha256 values in section 4.4) | MEASURED (ours, that path) |
| the bank unit headers (array names, dtypes, label value sets of one unit) | the label conventions in section 3 | data format, no outcome |

**Not read, and not to be read by this package:** any dev-gate (P2 held-out table) value, any navtest value. The PathGuard of the landed code refuses those path tokens. **No probe number exists at the time of writing**; no concordance, Brier, pick-level or loss value of any L or M arm has been computed.

## 1. The question

Head-only refits of P2's `score_head` rows on S0's frozen features do not move the pick:
* EP-R1 leaked (`D-REFE-HEADFIT-F10`): pick-level OOF -1.15 PDMS, unsafe-pick rate +0.010.
* SR-TTC was flat (`D-SRTTC-DEV-F10`): dev +0.10 / +0.08 against a +0.30 bar, LC FLAT.
* RL-1's composite lost NC with base safety heads (`H-REFE-RL1-NCRES`).
* EP-R1ub (NONE-FEASIBLE, 18 of 18 cells): the unsafe-pick rate ROSE although the unsafe-proposal anchor pushed the EP row's sigmoid on unsafe proposals down to P2's level or below. The leak is therefore not the EP row over-scoring proposals it knows are unsafe; it is fast proposals that the NC / DAC rows score as safe. **NC and DAC are the rows that matter most and are reported first.**

The next lever depends on ONE question. Do the cached 256-d features (`feat [S,64,256]`, the input of `score_head`) carry what each row needs, so that a linear head is the limit (then a nonlinear head would help)? Or do they not, so that the trunk is the limit (then P6, or a gradient into the trunk)?

## 2. The fixed design (from the brief; implemented as written, changed only by an amendment written before any number exists)

* **Rows:** NC (column 0), DAC (column 1), EP (column 2), TTC (column 3) of P2's `score_head`. Reporting order: **NC, DAC, EP, TTC.**
* **Inputs:** S0's bank `feat [S,64,256]` fp16 (the input of `score_head`); labels `labels6` of the bank, in each row's registered label convention (section 3.1).
* **Data:** f = 10 %, the IDENTICAL fit logs as SR-TTC R1 and EP-R1 at f10 (117 logs, 15,808 sets), 5-fold grouped-by-log CV (fold = rank in key order mod 5), 2 seeds {0, 1}. Fit-side OOF only. **No dev table, no navtest.**
* **Arms per row:** (L) a linear row refit from P2's row; (M) a 2-layer MLP probe, hidden 256, GELU, no dropout. Both: Adam lr 1e-3, weight decay 1e-4, **20 epochs**, BCE on the row's (soft) label, the SAME batches and seeds.
* **Metrics per row x arm x seed** (paired log-cluster bootstrap over the 117 fit logs, **10,000 resamples, seed 20260927**, percentile 2.5 / 97.5, the landed `boot_draws` / `paired_delta`):
  1. OOF pairwise concordance within sets over **all 64 proposals**, and within the **frozen top-8** (bank membership);
  2. the Brier score;
  3. the pick-level OOF read with ONLY that row replaced in the planner rule (`planner.py:558-561`, rule `navsim_v1`: NC x DAC x (5 EP + 5 TTC + 2 C) / 12 over sigmoids), the other five columns frozen at P2's stored logits: dPDMS x100 vs P2's pick and the change in the unsafe-pick rate.
* **Decision rule per row (fixed):**
  * **CAPACITY-LIMITED** iff the M - L concordance (all 64) has **lower CI > +0.01 on BOTH seeds**.
  * **FEATURE-LIMITED** iff its **upper CI < +0.01 on BOTH seeds** AND the L arm's concordance is **within 0.01 of P2's reference**.
  * Otherwise **INCONCLUSIVE.**
* **Controls:**
  * **C-ID (identity):** M with its output layer zeroed and a linear skip initialised to P2's row reproduces P2's row bit for bit.
  * **C2 (POWERED, RETRACTION_LOG R51):** 5 replicate label shuffles per arm. RED iff the shuffled concordance minus 0.5 has a lower CI > 0 on ANY replicate. A per-seed point <= 0 rule is NOT allowed. Report the control's false-RED rate under the null it is meant to pass.
  * **C-LEAK:** for NC and DAC, report the M arm's unsafe-pick-rate change (the D-LEAK analogue).
* **Outcome rows, each with its committed next arm:**
  * CAPACITY on EP or TTC -> a nonlinear-head registration with a powered C2, on a fresh dev read where one remains (EP-R1's dev read is UNSPENT; SR-TTC's is consumed).
  * CAPACITY on NC / DAC -> a safety-row MLP registration, the direct test of `H-REFE-RL1-NCRES`.
  * FEATURE on all rows -> the head-row line closes; the lever is the trunk (P6's end-to-end scorer training, read ~25 Oct), and an EP-conditional gradient into the trunk is drafted as the next A40 arm.
  * INCONCLUSIVE -> re-run at the next nested size (f = 25 %) ONCE, with the cost named.
* **Hard limits:** CPU only (`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=8`), at most 2 Python processes, `psutil` available RAM >= 3 GB before every fit (a script waits otherwise), no full stack suite, no landed file edited, no lock touched, `srttc_r1/`, `epr1/`, `epr1ub/`, `srttc_s0/` read only, kill by explicit PID only, LF files, stage and never push.

## 3. What the brief leaves open, fixed here (each is a choice, not a fact; all are fixed BEFORE any number)

### 3.1 Row label conventions ("each row's registered label convention")

| row | label `y` | valid entries (BCE, Brier, concordance pairs) | source of the convention |
|---|---|---|---|
| NC (col 0) | `labels6[..., 0]` in {0, 0.5, 1}, soft | all 64 proposals of every set | the bank; no conditioning |
| DAC (col 1) | `labels6[..., 1]` in {0, 1} | all 64 | the bank; no conditioning |
| EP (col 2) | `labels6[..., 2]` in [0, 1], soft | **SAFE proposals only**: NC >= 1.0 AND DAC >= 1.0 on the clipped labels (landed `safe_np`; NC = 0.5 is NOT safe). A concordance pair needs BOTH members safe. | EP-R1 / PREREG_EPR1: EP is a CONDITIONAL expectation (EP given NC = DAC = 1) |
| TTC (col 3) | `labels6[..., 3]` in {0, 1} | all 64 | SR-TTC R1 `srttc_r1.py`: `y3 = labels6[..., 3]`, anchor on all 64 |

Concordance: pair (i, j) is admitted iff both entries are valid and `y_i > y_j`; concordant iff `z_i > z_j`; ties in the score count 0.5 (landed convention). Pooled concordance = sum of concordant weights / sum of admitted pairs over the OOF sets; the unit of the bootstrap is the LOG (per-log numerators and denominators). "Within the frozen top-8" restricts both members to the bank's `top8` indices of that set. Brier = mean over valid entries of (sigmoid(z) - y)^2 in float64.

### 3.2 The arms

* **Parameterisation (the landed one).** The new logit of the replaced column is `z = z_stored + feat . (skip_w - p2_w) + (skip_b - p2_b) + mlp(feat)`, where `z_stored` is the bank's stored P2 logit and `mlp` is absent for L. With `skip = P2's row` and the mlp output zero, the added terms are EXACTLY 0.0, so `z == z_stored` bit for bit (C-ID). Features are cast to float32 and DETACHED (the trunk and the other five rows are frozen and PROVED frozen, section 6).
* **L** = the skip only, initialised to P2's row (weight[r,:], bias[r]).
* **M** = the skip (initialised to P2's row, so M CONTAINS L's function class and starts at P2's function) PLUS the hidden branch `mlp(feat) = W2 . GELU(W1 . feat + b1) + b2` with W1 [256,256], b1 [256], W2 [1,256], b2 [1]; **no dropout, no input normalisation** (P2's row reads the raw features). **W2 = 0 and b2 = 0 at init** (this is the "output layer zeroed" of C-ID); W1 and b1 use the PyTorch default Linear init (uniform within +-1/sqrt(256)) drawn from a `torch.Generator` seeded `7919*seed + 104729*row + 1299709*fold + 11`. *Reading of the brief:* the C-ID text ("M with its output layer zeroed and a linear skip initialised to P2's row") fixes that M carries a linear skip; a pure MLP could not reproduce P2's row. M - L therefore isolates what the hidden branch adds on top of the same linear start.
* **Training (both arms identical):** minibatches of **256 sets** (the landed default), shuffled each epoch by a `torch.Generator` seeded `seed*1_000_003 + len(train_idx)` (the landed convention; the order depends on the training index and the seed only, so L and M, and all four rows, see the SAME batches), **20 epochs**, `torch.optim.Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)` (coupled L2 on ALL trainable tensors, skip included), loss = mean over the valid entries of the minibatch of `BCE_with_logits(z, y)`. No pairwise term, no unsafe anchor, no schedule, no early stopping. `torch.use_deterministic_algorithms(True)`, 8 threads.
* **OOF:** for each fold k, train on the sets of the logs with fold != k, score the sets with fold == k; the OOF logit table `[15808, 64]` per (row, arm, seed) is assembled and SAVED (npy, sha256 in the result), so every metric is recomputable from it.
* **P2 reference** per row: the same metrics on the stored P2 logit, IN-SAMPLE for P2 (P2 trained on these sets), biased upward and in P2's favour at the pick.

### 3.3 Statistics and the estimator

* Every interval is the 95 % percentile interval of the **paired log-cluster bootstrap** (the landed `boot_draws(117)`, B = 10,000, seed 20260927; the SAME draws for every statistic, so every difference is paired). **It answers "would another draw of LOGS say this?" ONLY.** Head-fit noise (the W1 init, the batch order) is sampled by the two seeds, which is why the decision needs BOTH; inference is deterministic. The interval does not see training variance (`H-ESTIM-SEED-1`).
* Single-arm intervals are the ratio sum(num)/sum(den) over the resampled logs; differences use the landed `paired_delta` (same denominators).
* **"Within 0.01 of P2's reference"** (the FEATURE clause) = |point(L, all-64 concordance) - point(P2 reference, all-64 concordance)| <= 0.01 on BOTH seeds (two-sided; the signed value is reported). P2's reference is in-sample, so a low L can fail the clause for a reason that is P2's bias; that makes FEATURE harder to call, never easier.
* Pick-level read: the new logit replaces ONLY column r of P2's stored logits (`z6[..., r]`), the landed `picks_np` (stable descending argsort of the frozen rule over all 64 proposals, lowest index on ties) gives the pick, `pdms_np` / `safe_np` of the labels give its PDMS and safety. dPDMS x100 = 100 x (mean PDMS of the new pick - mean PDMS of P2's pick) over the OOF sets; d-unsafe = unsafe-pick rate of the new pick - P2's.

### 3.4 Controls, exactly

**C-ID (identity), on the real bank, all four rows, L and M, both through the same constructor and the same scoring path the arms use:**
(a) with zero training steps, `z == z_stored` under `torch.equal` on ALL 15,808 sets (not allclose); (b) the picks equal P2's picks on all sets, and the per-log concordance numerators / denominators and the Brier sums equal the P2 reference EXACTLY; (c) for M, `W2`, `b2` are all exactly zero, the skip equals P2's row exactly. Also recorded (documentation, not bit-for-bit): max |feat . p2_w + p2_b - z_stored| over <= 2048 sets for each of the four rows, which must be <= 1e-2 (the landed row-consistency tolerance) or the run aborts. **C-ID RED aborts the run before any fit.**

**C2 (label-shuffle negative control; POWERED).** *Why it is cold-started:* an arm that starts at P2's row carries P2's REAL signal (P2 trained on these sets), so "shuffled concordance - 0.5" would be RED by construction and the control would carry no information. C2 therefore trains each arm from a COLD start (skip = 0, bias = 0, stored logit not used, the mlp branch as in M, W2 = b2 = 0) on labels that are PERMUTED within each set among the valid entries of that row (each set's label multiset and valid mask unchanged; the EP row permutes among the safe proposals), scores the OOF logit against the REAL labels, and reads the all-64 concordance. Everything else (folds, batches, epochs, optimiser, scoring, bootstrap) is the arm's code path.
* 5 replicates per (row, arm): shuffle seeds `20261019 + 1000*rep + 10*row` for rep = 1..5; training seed 0 for all replicates (the replicates differ in the shuffle). 4 rows x 2 arms x 5 reps x 5 folds = 200 fits.
* **RED (cell) iff, on ANY of its 5 replicates, the lower 95 % bound of (concordance - 0.5) is > 0.** A per-seed point <= 0 rule is not used. A **RED C2 of a row (either arm) VOIDS that row's CAPACITY call** (a pipeline that manufactures concordance from shuffled labels could manufacture an M - L gap); it does not void a FEATURE call (a manufactured gain can only push toward CAPACITY).
* **C2-DEC (same fits, the decision statistic's own null):** RED iff, on any replicate, the lower 95 % bound of the shuffled (M - L) all-64 concordance is > +0.01 (the CAPACITY threshold). Same consequence.
* **The control's false-RED rate under the null it is meant to pass** is MEASURED (not asserted): for each row, 200 trials; each trial draws 5 replicates of iid N(0,1) scores (numpy `default_rng(20261019 + 7*trial + rep)`) independent of the labels, scores them with the SAME per-log concordance and the SAME bootstrap and applies the SAME rule; the rate of trials with >= 1 RED and the per-replicate RED rate are reported with Wilson 95 % intervals. Expected for a calibrated one-sided 97.5 % bound: about 2.5 % per replicate, about 12 % per (row, arm) cell with 5 replicates. **The family false-RED rate over the 8 cells is reported as 1 - (1 - p)^8 with the measured p, because the voiding rule can fire somewhere on chance alone.**

**C-POS (the control's mirror image: POWER; added by this registration, section 4.1).**

**C-LEAK (NC and DAC; the D-LEAK analogue).** For each of NC and DAC and each seed: the M arm's d-unsafe-pick-rate vs P2 with its interval, and the L arm's, and M - L. Flag per row: **LEAK** iff the lower bound of d-unsafe (M vs P2) > 0 on BOTH seeds; **CLEAN** iff the upper bound <= 0 on both seeds; otherwise **AMBIGUOUS**. Because P2's pick is in-sample, every flag is read against the matched noise-row control C-NOISE (section 4.2); the flag itself uses no comparator.

### 3.5 Launch discipline

* **A launch gate** (PI 2026-09-26 and ruling R5: a shuffled-label fit IS a fit): a stage `gate` mints `gate_token.json` bound to the sha256 of the code files, the landed parent blobs, the full config dict (every lever above), the bank manifest, the ranked logs, the P2 row and checkpoint, with checks G-COMMIT (parents equal their pinned blobs), G-DVB (every lever of the config equals the value the built model / optimiser reports: hidden 256, GELU, no Dropout module, lr, wd, epochs, batch, skip equals P2's row, W2 = b2 = 0), G-BANK (every bank unit sha256-verified BEFORE np.load; the f10 fit logs equal EP-R1's), G-LIVE (the reach proof of section 6 on a real batch), G-SUITE (the package's synthetic tests, including a RED mutation arm per check). Every fitting stage REFUSES to start without a token whose bindings match. Deviation from the landed per-package gate files: this gate is a stage of the same entry point, not a separate file.
* **RAM:** before every fit a script checks `psutil.virtual_memory().available >= 3 GB` and waits (re-checking every 20 s) otherwise.
* **Evidence class and tier:** every number is MEASURED (ours, the paths in the manifest), SELECTION tier (fit-side OOF), a design read about feature sufficiency and not a result about driving. T0 / T1 do not apply (no closed-loop claim).

## 4. Added by this registration (beyond the brief; each is conservative or descriptive, none can create a CAPACITY or FEATURE call the brief's rule would not)

### 4.1 C-POS: the probe's power (it can veto a FEATURE call)

A null result from M is only evidence of FEATURE-LIMITED if M could have found a nonlinear signal had one existed. C-POS plants one on the REAL features. Teacher (label-independent, a function of the f10 features only): `x_std = (feat - mean) / std` per dimension over all f10 proposals; `g0 = w . GELU(3 * V x_std / sqrt(256))` with V [32,256] ~ N(0,1), w [32] ~ N(0,1)/sqrt(32), drawn from `numpy default_rng(20261019)`; the LINEAR part is removed by ordinary least squares of `g0` on `[x_std, 1]` over all proposals (so a linear head has nothing left to find); `g` is standardised; the planted label is `y = sigmoid(2 g)` (soft, in (0,1)); all 64 entries valid. Both arms are fitted from a COLD start (P2's row says nothing about a planted signal) with the SAME protocol, seeds {0, 1}, 5 folds, and scored OOF on the planted label. **C-POS GREEN iff the M - L all-64 concordance has lower CI > +0.01 on BOTH seeds** (the CAPACITY criterion applied to a known nonlinear signal). **C-POS RED VOIDS every FEATURE-LIMITED call (those rows become INCONCLUSIVE with the reason "probe underpowered")**; CAPACITY calls are unaffected. Limits: the teacher is in M's own function family, so GREEN proves the pipeline can detect a capacity gap of that kind, not that every real structure is learnable.

### 4.2 C-NOISE: the matched noise-row comparator for C-LEAK (pre-registered; it was post hoc in EP-R1ub)

P2's pick is in-sample, so ANY change of a row moves the pick and may move the unsafe-pick rate. For each of the four rows: P2's stored logit of that column plus iid Gaussian noise (sigma in {0.01, 0.03, 0.1, 0.3, 1.0}, noise seeds {0, 1}, numpy `default_rng(20261019 + 100*row + 10*i_sigma + seed)`), on all 15,808 sets, with the same pick-level estimator (no fitting; not OOF). Report changed-pick fraction, d-unsafe [CI], dPDMS [CI] per cell. For each arm and seed the matched comparator is the noise sigma whose seed-mean changed-pick fraction is nearest (absolute difference) to the arm's OOF changed-pick fraction. Descriptive: it never enters a call.

### 4.3 How control outcomes attach to the call (not in the brief; fixed now)

| control | RED means | effect on a row's call |
|---|---|---|
| C-ID | the arm is not the identity at init | the whole run aborts before any fit |
| gate / reach proof | a frozen row or the features receive a gradient, or the probe parameters receive none | no fit is launched |
| C2 or C2-DEC (that row, either arm) | the pipeline can manufacture concordance (or an M - L gap) from shuffled labels | CAPACITY for that row is VOID -> INCONCLUSIVE ("C2 RED") |
| C-POS | the probe cannot see a planted nonlinear signal | every FEATURE call is VOID -> INCONCLUSIVE ("probe underpowered") |

### 4.4 A-RES: the zero-training NC/DAC resolution read (adopted from the draft; section 5)

* **Reference rows (applied, not fitted):** `C:/Users/Admin/epr1/run/f10/ep_row_seed0.npy` (sha256 `f775753f0817d53607b0044e9d33f6d2c5f5b01b73a8d20567d6394ca393fa95`) and `ep_row_seed1.npy` (sha256 `636f6270f52ef462fb6a1a56a990aa8c683f3c30a3394091b77d10f439852fa8`), verified against these pins and against `final_rows` of `epr1_result.json` BEFORE use; applied to the EP column with the landed `apply_row` and the landed `picks_np` on all 15,808 sets. They were fitted on all fit-set sets (in-sample for the EP row, and P2 is in-sample too): disclosed.
* **Sets per seed:** V = {P2's pick safe, the new pick unsafe}; S = {P2's pick unsafe, the new pick safe}; D = {the picks differ, both safe}.
* **Statistics (paired log-cluster bootstrap, same draws):** with `s = log(pNC * pDAC)` from P2's stored logits (float64, columns 0 and 1), `m = s[new pick] - s[P2's pick]`: (1) the fraction of V with m >= 0 and the median m on V, and the same two on D; (2) the within-top-8 AUC of `s` for SAFE versus UNSAFE contenders (AUC = P(s_safe > s_unsafe), ties 0.5, pairs within a set), on the V sets and on all sets that have both classes among their top-8; (3) the share of V sets whose new pick is the top-8 contender with the highest `pNC * pDAC`.
* **Flag (descriptive; it never changes a per-row call of the probe):** **R-NONE** if |V| < 100 on a seed (the read is void for that seed); **R-RES** iff fraction(V, m >= 0) >= 0.50 AND AUC_V <= 0.60 on both seeds; **R-OVER** iff fraction(V, m >= 0) < 0.50 AND AUC_V >= 0.70 on both seeds; **R-MIXED** otherwise (point estimates decide, as in the draft; the intervals are reported).
* **Controls:** C0 (an "EP row" equal to P2's row gives empty V and S and P2's picks bit for bit); C1' (shuffle the safe flags among each set's top-8 contenders: the AUC statistic must read 0.5, i.e. the interval of AUC - 0.5 covers 0); C2' (the statistics reproduce under an independent brute-force re-statement on a 500-set subsample); C3 (a mutation arm per statistic, namely the sign of m, the unsafe mask, the top-8 membership, must go RED).
* **How it joins the probe's rows (committed now):** NC/DAC CAPACITY + R-RES -> consistent; the safety-row MLP registration is the arm. NC/DAC FEATURE + R-RES -> the heads cannot see the violators and the features cannot either: the lever is the trunk (P6). **Any R-OVER -> the heads DO rank the new violators lower and the rule's weighting outweighs them: the safety-row MLP registration (if called) must carry a pick-level floor on `pNC * pDAC` (the `R0e-pick` family) as its comparator, and a head retrain alone cannot cure the leak.** R-MIXED / R-NONE: no additional inference.

## 5. Adoption ledger for `DRAFT_PREREG_NCDAC_RESOLUTION.md` (sha256 `fe83adb4938b05eae372feeb630d082e5b668edb864e1f8f788a4f283d44b7d3`; subsumed, NOT run separately)

| draft item | decision | reason (one line) |
|---|---|---|
| Hypothesis H-NCDAC-RES and the zero-training Part A read | ADOPTED as A-RES (4.4) | it separates RES / OVER / FEAT cheaply and the brief has no such read |
| Reference rows = the hashed EP-R1 f10 rows, applied with `apply_row` / `picks_np` | ADOPTED | no new fit; the sha256 pins exist |
| Sets V / S / M | ADOPTED (M renamed D, to avoid the arm M) | same definitions |
| Statistic 1 (m, fraction, median) | ADOPTED | as drafted |
| Statistic 2 (within-top-8 AUC, safe vs unsafe) | ADOPTED with the direction fixed: AUC = P(s_safe > s_unsafe) | the draft's thresholds only read right for that direction |
| Statistic 3 (share of V whose new pick has the highest pNC*pDAC) | ADOPTED | as drafted |
| Outcomes R-RES / R-OVER / R-MIXED / R-NONE and their thresholds | ADOPTED as a FLAG | thresholds kept; it cannot override the brief's outcome rows |
| Control C0 (EP row = P2's gives empty V and bit-identical picks) | ADOPTED | as drafted |
| Control C1 (random replacement pick gives AUC 0.5) | REJECTED as written, REPLACED by C1' (shuffle the safe flags among the top-8) | the AUC is within the top-8 and does not depend on the pick; a random pick would not give 0.5 for m either |
| Control C2 (independent re-statement) | ADOPTED as C2' (named so, to avoid the label-shuffle C2) | as drafted |
| Control C3 (mutation arm per statistic) | ADOPTED | as drafted |
| Part B: NC / DAC head-only retrain with a pairwise top-8 ranking term, selected on OOF contender AUC at a pick-level constraint | REJECTED as part of this package | conflicts with the brief's fixed BCE-only L / M arms, and it is a downstream registration; the NC/DAC CAPACITY row names the successor, and Part B's pairwise term is noted as a comparator candidate for it |
| Cost: CPU seconds, no GPU, no new bank | ADOPTED | A-RES is seconds; the probe is minutes to an hour |
| Navtest form of the read stays with the PI | ADOPTED as a scope limit | it opens rows beyond RL-1's look |
| Limit: P2 in-sample (conservative for R-RES) | ADOPTED and EXTENDED | the trunk's features are also in-sample for P2, which can inflate M - L toward CAPACITY (section 7) |
| "An OOF replication of Part A (a gated fit) if inconclusive" | REJECTED here | needs a gated EP-R1 refit; a follow-up, not a part of this probe |
| (from EP-R1ub RESULT section 4, not the draft) the noise-row control | ADOPTED PRE-REGISTERED as C-NOISE (4.2) | it was post hoc and unregistered there |

## 6. Reach proof and RED mutation arms (the "built, tested, unreachable from its caller" class)

On a REAL batch (the first 512 sets of the fit data with contrast for the row) and for each of the four rows and each of L, M, with ALL six rows of P2's head as leaf parameters feeding the non-target columns and the features as a leaf requiring gradient (the landed probe construction):
* **full loss:** non-zero finite gradient on the probe parameters (skip_w, skip_b; for M also W2, b2); **EXACTLY 0.0** on all six frozen rows and on the features. For M at the zero-initialised W2 the gradient on W1, b1 is EXACTLY 0.0 by construction (dL/dh = W2^T g = 0) and that is asserted as an identity; at a perturbed W2 (nonzero) W1, b1 receive a non-zero gradient (positive control).
* **zero control:** the loss multiplied by 0.0 gives EXACTLY 0.0 everywhere (a probe stuck at non-zero cannot pass).
* **valid mask (EP, C6 analogue):** replacing the labels of UNSAFE entries leaves the gradient bit-identical; an all-unsafe batch gives EXACTLY 0.0; changing the labels of SAFE entries changes the gradient (positive control).
* **update:** one real optimiser step on the real fit path moves the probe parameters and no frozen tensor.
* **Mutation arms (each must go RED):** (m1) remove the `detach` on the features; (m2) add the loss of another column (leak); (m3) drop the EP safe mask; (m4) stop the gradient into the hidden branch; (m5) change the weight to 0 on the BCE term (the full-loss check must catch it); (m6) a skip initialised to zeros instead of P2's row (C-ID must catch it); (m7) a non-zero W2 init (C-ID must catch it); (m8) a bootstrap that resamples sets instead of logs (the estimator test must catch it); (m9) concordance without the tie 0.5 (the brute-force test must catch it); (m10) the pick-level read applied to the wrong column (the column test must catch it).

## 7. Outcome mapping and its combination rule (fixed now)

Calls are made PER ROW. Then:
* every row with **CAPACITY** takes its committed arm: EP or TTC -> a nonlinear-head registration with a powered C2 on a fresh dev read where one remains (EP-R1's UNSPENT; SR-TTC's consumed); NC or DAC -> a safety-row MLP registration (the direct test of `H-REFE-RL1-NCRES`), carrying A-RES's pick-level comparator if A-RES reads R-OVER;
* if **all four rows are FEATURE-LIMITED** -> the head-row line closes at the fit; the lever is the trunk (P6, read ~25 Oct); an EP-conditional gradient into the trunk is drafted (outline in RESULT.md) as the next A40 arm;
* every row with **INCONCLUSIVE** is re-run ONCE at f = 25 % (the next nested size), only for those rows, with the cost named in RESULT.md from the measured f10 wall-clock (the sets scale about 2.5x); a second INCONCLUSIVE closes the question as undetermined at the data we have, it is not re-run;
* a mixture (some CAPACITY, some FEATURE) is reported as exactly that: the FEATURE rows close their head-row line, the CAPACITY rows proceed.
The probe NEVER reads the dev gate or navtest: a successor registration owns that.

**Limits stated now (so they cannot be discovered afterwards).** (i) P2 is IN-SAMPLE on these sets, and so is its TRUNK: the cached features of the bank's own training sets may carry label information a flexible head can read and a linear head cannot, which inflates M - L toward CAPACITY and which no fit-side read can remove; a CAPACITY call here is a necessary, not a sufficient, case for a nonlinear head on unseen scenes (the successor registration's dev read is what tests that), whereas a FEATURE call is conservative against it. (ii) The intervals answer "another draw of LOGS" only; two seeds sample head-fit noise; with two seeds the rule cannot be calibrated, which is why it needs both. (iii) One fraction (f = 10 %); a learning-curve exponent is neither fitted nor quoted. (iv) The C2 voiding rule can fire on chance (rate measured in the result, about 12 % per cell nominal). (v) M is warm-started and weight-decayed by the brief's recipe; a different recipe could move M - L, and C-POS tests only that this recipe sees a planted signal.

## 8. Amendment protocol

Any change to sections 2 to 7 is written as `PREREG_HEADROW_PROBE_AMENDMENT<n>.md`, hashed before it is acted on, and, if it touches the design, BEFORE the first fit. Nothing is amended after a number exists, except to correct a transcription that no number depends on.
