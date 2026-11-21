# PREREG PICKRULE-AUDIT-NC: does the NC loss of an EP-led pick change arise in how the planner rule COMPOSES the rows, or in the NC row's own ranking? (Research Lab agent for the Master Mind, 2026-10-19)

**Written BEFORE any quantity of this arm exists.** Nothing below was computed on a label array, a pick or a composite. The sha256 of this file is recorded in `PREREG_PICKRULE_AUDIT_NC.sha256` before the first real-data stage runs, and every stage refuses to start unless the file still hashes to it. A change comes only by a written amendment (`PREREG_PICKRULE_AUDIT_NC_AMENDMENT<k>.md`) saved and hashed before the number it governs.

- Tier: SELECTION, on SEEN scenes (`D-L1-G0-1`), P2 IN-SAMPLE. No dev table, no navtest and no GPU is read or used. 0 GPU, CPU only.
- Design source: the Master Mind's brief for task PICKRULE-AUDIT-NC, and the committed next arm of `PREREG_FASTNEG_CENSUS.md` section 6 (row NO DEFICIT): "a 0-GPU pick-rule audit on S0: the unsafe-pick rate of an EP-led composite as a function of the NC / DAC veto threshold, with the rows unchanged".
- Rows are FROZEN in this arm: it tunes no row, chooses no hyperparameter and reads no row's parameters as a degree of freedom (see section 3 for the exact meaning).

## 0. Disclosures (what I had seen before sealing)

1. I read the GAC rows `D-REFE-RL1-HEADWISE`, `D-REFE-RL1-MECH`, `H-REFE-RL1-NCRES`, `D-REFE-FASTNEG-CENSUS-1`, `D-REFE-EPR1UB-1`, `D-REFE-HEADFIT-F10` at the tip `993a78cf`. From them I know EP-R1's banked OOF pick-level numbers (seed 0: PDMS x100 89.219 vs P2 90.365, unsafe rate 0.0406 vs 0.0304; 12,314 of 15,808 picks changed), which are the identity target of control C-ID2. That is a number about a composite I regenerate; it is not a value of Q1, Q2 or Q3.
2. I read the census's `inputs_report.json` label value counts (NC = 0: 70,603, NC = 0.5: 8,343, DAC = 0: 117,136 of 1,011,712 proposals). Descriptive; no pick is involved.
3. I opened NO `labels6` array, NO logit array and NO pick of the bank before sealing. The only checkpoint tensors I opened are structural (shapes and bit-equality of the RL scorers vs P2's modules, section 3.1).
4. The numeric false-fire rate of the interval rule (section 8, C-NULL) was computed on SYNTHETIC data only (`scratch/fp_rate.py`).

## 1. Question

`D-REFE-RL1-HEADWISE`: the zero-training head-wise composite V1 (RL EP + comfort heads on the deployed scorer's other heads) lost NC on navtest, NC -0.124 [-0.237, -0.016]. `D-REFE-HEADFIT-F10` / `D-REFE-EPR1UB-1`: a refit EP row raised the unsafe-pick rate in two more arms, and an anchor that pushed the EP row's unsafe scores down did not stop it. `D-REFE-FASTNEG-CENSUS-1`: the NC row has NO fast-band ranking deficit (F minus Q1-Q3 within-token AUC +0.0323 [-0.0044, +0.0654] / [-0.0041, +0.0653]); DAC does (-0.0718). So the NC loss should arise in how `planner.py:558-561` COMPOSES the rows. This arm tests it.

Two readings of one event, "an EP-led pick change lands on an NC = 0 proposal that P2's pick was not":
- **COMPOSITION:** the NC row FLAGGED that proposal (its own NC sigmoid < 0.5) and the product rule picked it anyway, because a high EP term outweighed a low NC factor. The row had the information; the rule spent it.
- **ROW-MISS:** the NC row did NOT flag it (sigmoid >= 0.5). The row lacked the information at that proposal, whatever the rule does.

## 2. Data (identical to FASTNEG-CENSUS / HEADROW-PROBE)

- S0's bank `C:/Users/Admin/srttc_s0/work/bank/*.sets.npz` (READ ONLY), every unit sha256-verified against `MANIFEST.sha256.json` BEFORE `np.load` (manifest sha256 `6296dfc0b03905038fbf132f82009f6b6d7e8d61fa9abfb65c9164f487a5f2a1`), `label_order = (nc, dac, ep, ttc, c, ddc)`, rule `navsim_v1`, ONE checkpoint (`D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt`).
- f = 10 %: the 117 logs selected by the landed `epr1.select_logs(rows, 10, None)` from `ranked_fit_logs.tsv` (sha256 `a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1`); 15,808 sets of 64 proposals; folds = `rank mod 5`; training seeds {0, 1}.
- Loaders: the landed `epr1.py` (sha256 `a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8`, pin checked at load) `load_manifest`, `load_ranked_logs`, `select_logs`, `load_p2_row`, `load_fit_data` (integrity checks: shapes, finiteness, labels in [0, 1], stored top8 = the frozen rule applied to the stored logits, P2-row consistency <= 1e-2). The full `labels6 [N, 64, 6]` is read from the SAME verified units, in the same unit order and with the same log masks, and joined by content: its EP column, its `safe_np` and its `pdms_np` must equal `FitData`'s `y`, `safe` and `pdms` bit for bit on all 15,808 sets (control C-JOIN).
- Estimator helpers: the landed census `fastneg_lib.py` (sha256 `0c0b399b37a47ee3e17f9d5484bc7f9521ff7dbe6bea37e8c7a38042668667ca`, pin checked at load): `boot_draws`, `ratio_ci`, `diff_ci`.
- Bootstrap: paired LOG-cluster bootstrap over the 117 logs, B = 10,000, seed 20260927, the SAME draws for every quantity (`epr1.boot_draws(117, 10000, 20260927)`), percentile 2.5 / 97.5, every ratio a ratio of per-log sums, every difference computed on the same resampled logs. This answers "would another draw of LOGS say this?" and nothing about another training run of P2 (`H-ESTIM-SEED-1`).

## 3. Rows

**R-P2** = P2's frozen `score_head` outputs, i.e. the bank's stored logits `[N, 64, 6]` (columns NC, DAC, EP, TTC, C, DDC).

### 3.1 Do V1's rows apply to S0's `feat`? NO, so the fallback applies (MEASURED before sealing)

V1 = RL R0B's EP and comfort heads + the deployed scorer's NC, DAC, TTC, DDC and lane (`PREREG_RL1_B_COMPOSITES.md` section 1). It applies to S0 only if S0's `feat` is the input of V1's `score_head` and V1's scorer decoder is P2's. It is not:
- RL's scorer is a 7-output head: the R0d u1000 states carry `score_head.weight (7, 256)` (and V1 was scored with `navsim_v1_lane`, which needs the 7th output, `planner.py:553-557`); P2's `score_head.weight` is `(6, 256)` (MEASURED: `scorer_states_u1000.pt` of the local R0d arm vs `model_final.pt`; the loader `epr1.load_p2_row` itself refuses a 7-output head: "a 7-output SFT-4 head is not P2").
- RL trains `score_q_mlp`, `score_dec` and `score_head` (`scorer_finetune.py` docstring: "the score loss reaches only `score_q_mlp`, `score_dec`, `score_head`"), so its 256-d pre-head feature is the output of ITS decoder, not P2's. MEASURED: of the 76 `score_q_mlp` / `score_dec` tensors both arms (A and B) of R0d u1000 share by name and shape with P2, **0 are bit-equal** (max abs difference 0.029 / 0.031).
- Evidence class: MEASURED for R0d u1000 A and B (`C:/Users/Admin/par18/rl/r0d_dl/r0d/scorer_states/scorer_states_u1000.pt`). R0 u1000 B itself (the navtest R0B) is not on the dev box; that it shares R0d's base (SFT-4 arm B u400) is INHERITED from `D-REFE-RL1-HEADWISE`, not re-verified. The conclusion does not need it: R0B's head is 7-output by the rule `navsim_v1_lane` V1 was scored with.

Therefore, per the brief, the EP-led composite is built from **EP-R1's f10 refit EP row** (`D-REFE-HEADFIT-F10`). This is a DIFFERENT EP-led change from V1's, with the same mechanism class (a refit EP row, other columns frozen) and it is the population in which `D-LEAK` was observed.

### 3.2 R-EP: the EP-led composite

R-EP = R-P2 with column 2 (EP) replaced and columns 0, 1, 3, 4, 5 equal to P2's (bit for bit, control C-COL). Two sources for the replaced column, both at EP-R1's registered c\* = (lr 1e-3, anchor 0.1, 10 epochs, batch 256 sets, Adam, betas (0.9, 0.999), eps 1e-8, no weight decay), seeds {0, 1}:

- **PRIMARY, `OOF`:** the 5-fold grouped-by-log out-of-fold EP logit, regenerated by the landed `epr1.fit_row` + `epr1.apply_row` exactly as `epr1.oof_run` does (train logs = every log not in the fold, in key order; seed `s`; `torch.use_deterministic_algorithms(True)`, 8 threads). It is the object whose pick-level read `D-REFE-HEADFIT-F10` banked. **Control C-ID2 requires it to reproduce that banked read to 1e-9** (new / P2 PDMS x100, new / P2 unsafe rate and the number of changed picks, per seed, from `raw/2026-10-18-ep-r1/raw/f10/epr1_result.json`), which proves it is the same object, not a lookalike.
- **SECONDARY, `BANKED`:** EP-R1's banked FINAL rows `ep_row_seed{0,1}.npy` (sha256 `f775753f…` / `636f6270…`; fitted on ALL 15,808 sets, hence IN-SAMPLE on S0), applied as `stored + feat . (w - p2_w) + (b - p2_b)`. It trains nothing at all in this arm. It is reported with the same quantities and the same rules, **has no authority over the outcome**, and a disagreement between the two sources is disclosed as SOURCE-SENSITIVE, not resolved by choosing.

**What "rows are frozen, no training in this arm" means here.** The arm tunes nothing: no row, threshold of a row, seed or hyperparameter is chosen from anything it reads. The PRIMARY EP column is a deterministic REGENERATION of a row that EP-R1 already registered (same code, same pins, same seeds, same folds), checked by C-ID2. If the Master Mind wants the literal zero-fit reading, SECONDARY is it; the choice of which one carries the outcome was made here, before any number, because only PRIMARY has a banked pick-level read to be identical to and because it is out-of-fold (a deployable-like EP column), whereas BANKED is in-sample.

## 4. Pick rule (reimplemented exactly)

`planner.py` at the tip (git blob `6fb6598a2ecf230153a4d7adfb790afc30728070`, file sha256 `91437cd09362ff872a861c6166b3c00eee98ce9b98c4e85b6a95ca133e29e211`), lines 558-561 (their sha256 `bcfaf349a155d881a3b35589bde3b1106f87d5e405d715be8e165625953c3574`, LF):

```
        if self.rule == "navsim_v1":
            # NAVSIM v1 PDMS: NC x DAC x (5 EP + 5 TTC + 2 C) / 12 -- driving direction does not enter
            w1 = torch.tensor(self.V1_W, device=p.device, dtype=p.dtype)
            return p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()
```
with `p = score.sigmoid()` (line 551) and `V1_W = (5.0, 5.0, 2.0)` (line 520); `_pick` is `argmax` of that aggregate. My `rule_aggregate(z6)` is the same expression, same operation order, evaluated in **float64** (the dtype of the bank's own pick, `epr1.aggregate_np`), ties to the LOWEST index (`np.argmax`; equal to the stable descending argsort the bank used). A float32 evaluation (the planner's runtime dtype) is run as a DIAGNOSTIC and its agreement with the bank is reported; the float64 evaluation is the one the controls and the outcome use.

**Veto at tau** (Q3 only): eligible set `E_tau = {j : sigmoid(z_NC[j]) >= tau}` (strict `<` excludes); the pick is the first argmax of the same aggregate over `E_tau`; **if `E_tau` is empty the pick is the UNVETOED rule pick** (the veto never leaves a set undefined; the number of such fallback sets is reported per tau). Rows are unchanged by the veto. tau in {0.3, 0.5, 0.7, 0.9}.

## 5. Definitions (labels from `labels6`; label convention as `PREREG_FASTNEG_CENSUS_AMENDMENT1.md` A1.3)

- **NC-unsafe pick** := the picked proposal has `labels6[..., 0] == 0`. **DAC-unsafe pick** := `labels6[..., 1] == 0`. (An NC = 0.5 pick is neither. Descriptive sensitivity with no authority: unsafe := label < 1; and `epr1`'s "unsafe" = not (NC >= 1 and DAC >= 1), used only by C-ID2.)
- **Pick-level PDMS** := `NC * DAC * (5 EP + 5 TTC + 2 C) / 12` of the picked proposal from `labels6` (DDC at weight 0; no TLC). The weights are NAVSIM v1's `(5, 5, 2)/12`: they are `REFePlanner.V1_W = (5.0, 5.0, 2.0)` at `refe/planner.py:520` (comment there: "navsim @ 3e8291b `pdm_scorer.py:38-42`"), applied as in `aggregate` lines 558-561, and equal `epr1.pdms_np` (docstring: "formula MEASURED to reproduce the table, ep_common.pdms_formula"). Control C-PDMS: my implementation equals `epr1.pdms_np` bit for bit on every proposal of the bank. ⚠️ This is a LABEL-lookup PDMS at the selection tier on SEEN scenes, not a navtest PDMS and not an EPDMS.
- P2's pick = `rule_aggregate` argmax on R-P2. The EP-led pick = the same on R-EP.

## 6. Quantities (each per seed, per source; point + percentile CI from the paired log-cluster bootstrap of section 2)

- **Q1 (overridden share, EP-led):** over the sets `A` where the EP-led pick is NC-unsafe AND P2's pick is NOT NC-unsafe (`NC label != 0`): the share of those EP-led picks whose OWN NC sigmoid `sigmoid(z_NC[pick])` is `< 0.5`. A ratio of per-log sums (numerator: |A_i with sigmoid < 0.5|; denominator: |A_i|). Also reported: |A|, the numerator, the median NC sigmoid at those picks.
- **Q2 (reference, P2):** the same share over ALL sets where P2's own pick is NC-unsafe (NC label 0). Descriptive companions, no authority: the matched Q2 (P2 unsafe AND EP-led pick not), and the paired contrast Q1 - Q2 on the same draws.
- **Q3 (veto sweep):** for tau in {none, 0.3, 0.5, 0.7, 0.9}, rows unchanged, on R-EP: (a) NC-unsafe pick rate, (b) DAC-unsafe pick rate, (c) pick-level PDMS x100, each with its own CI; and its paired difference to P2's UNVETOED pick (same sets): d(NC-unsafe rate), d(DAC-unsafe rate), dPDMS x100, with CIs; plus the number of fallback sets and the number of sets whose pick the veto changed. Descriptive companion rows (no authority): the same sweep applied to R-P2 itself ("P2 + veto"), and the DAC analogue of Q1 (DAC sigmoid < 0.5).

## 7. Decision rule (fixed now)

**Per seed and source:** Q1 lower CI > 0.5 -> `COMPOSITION`; Q1 upper CI < 0.5 -> `ROW-MISS`; otherwise `MIXED` (no special case for a small |A|: the interval carries it, |A| is reported).
**Outcome (PRIMARY source only):** `COMPOSITION` iff BOTH seeds read COMPOSITION; `ROW-MISS` iff BOTH read ROW-MISS; otherwise `MIXED` (two seeds that disagree are MIXED, as in the census). SECONDARY's class is reported beside it and carries no authority.

**VETO-RECOVERS** (separate, reported for every tau, never chosen post hoc): at tau, for seed s, let `R` = the NC-unsafe pick rate of R-EP vetoed at tau, and `P2_hi` = the 97.5th percentile of P2's UNVETOED NC-unsafe pick rate. tau RECOVERS iff, on BOTH seeds, (i) `R <= P2_hi` (the composite's rate is within P2's interval from above) AND (ii) the lower CI of dPDMS x100 (vetoed R-EP minus unvetoed P2, paired) is `>= 0`. `VETO-RECOVERS` iff at least one tau in the grid recovers. The list of recovering tau (possibly empty) is reported as is; no tau is named as "the" answer in this arm.

**Committed next arm (by outcome, fixed now):**

| outcome | next arm |
|---|---|
| COMPOSITION + VETO-RECOVERS | a 0-train pick-rule registration: a hard NC veto at a tau named by a rule fixed IN that registration (not by this arm), for the dev gate |
| COMPOSITION, no recovery | a soft gating rule (multiplicative NC power, `p_NC^k` in the product), registered |
| ROW-MISS | NC calibration on the fast band (temperature or isotonic, out-of-fold), registered |
| MIXED | report both shares and stop |

## 8. Controls (each must hold or the read is INVALID; every one is built so it can fail)

- **C-ID1 (rule reproduction):** `rule_aggregate` + first-argmax on P2's stored logits equals the bank's banked pick (`top8[:, 0]`) on **15,808 / 15,808 sets** (100 %), and also equals `epr1.picks_np`. The float32 diagnostic's agreement is reported. Source pin: lines 558-561 of the tip `planner.py` hash to the value of section 4 (checked against the copy the run is given; if that copy is absent, the check is reported NOT RUN, never PASS).
- **C-ID2 (composite identity):** the PRIMARY R-EP, no veto, reproduces EP-R1's banked pick-level read at c\* for both seeds: `new_pdms_x100`, `p2_pdms_x100`, `unsafe_rate_new`, `unsafe_rate_p2` to 1e-9 and `oof_picks_changed` exactly (from `epr1_result.json`).
- **C-ID3 (veto identity and reach):** veto at tau = 0 reproduces the unvetoed picks on 100 % of sets, bit for bit; veto at tau = 1.01 (every set falls back) reproduces them too; veto at tau = 0.5 changes the pick on at least one set and on fewer than all (a switch is not a veto); for every non-fallback set the vetoed pick has NC sigmoid >= tau.
- **C-COL:** R-EP columns 0, 1, 3, 4, 5 equal R-P2's bit for bit; column 2 differs on >= 1 proposal.
- **C-JOIN:** the label join of section 2 (EP column, `safe`, `pdms` bit-equal to `FitData`).
- **C-PDMS:** section 5.
- **C-NULL (powered, `RETRACTION_LOG` R51):** within each set the 64 proposals' LABELS (all six columns, one permutation per set) are permuted; the rows, the picks and the sigmoids are NOT touched. Q1 is recomputed. 5 replicates per (source, seed), permutation seed `20261019 + 7919 * rep`. A replicate FIRES iff its Q1 LOWER CI > 0.5 (the COMPOSITION bar, an interval rule, not a point rule). The false-fire rate is reported. **VOID iff the null fires on >= 2 of 5 replicates for any (source, seed).** The false-RED rate of this rule under a TRUE null was computed on synthetic data BEFORE registering (R51's demand): with 117 logs and |A| in {60, 100, 150}, a true share of 0.10 or 0.30 fires 0.000 of 1,000 simulations at every |A|, and a true share of exactly 0.50 (the worst case, the boundary) fires 0.025 / 0.024 / 0.020 (ESTIMATED, `scratch/fp_rate.py`). A permuted-label share is expected near the base rate of low-NC-sigmoid picks, far below 0.5, so its expected false-RED rate is ~0. The control's POWER (that it CAN fire when a structural artifact exists) is shown in the test-suite by a planted case in which every pick has NC sigmoid < 0.5 independently of labels.
- **C-REACH (code reachability):** the decision functions are exercised by a branch table; a RED mutation (comparison flipped, bound swapped, seed-agreement `all` replaced by `any`, a threshold constant changed) of EACH of them must make the table FAIL (`test_pickrule.py`).

## 9. Forecast (a prediction, not a bar)

HYPOTHESIS, committed before any number: **ROW-MISS or MIXED more likely than COMPOSITION.** Reason: P2's NC row is a soft-BCE head with a 7 % unsafe base rate, and the census found no fast-band ranking deficit for it; a proposal with NC sigmoid < 0.5 has an aggregate multiplied down by more than half, and the EP-led change moves picks within 64 proposals that mostly have NC sigmoid near 1. I put roughly 0.55 on ROW-MISS, 0.30 on MIXED, 0.15 on COMPOSITION. I put no weight on VETO-RECOVERS beyond: it can only hold if COMPOSITION does, because a veto at tau <= 0.5 cannot remove a pick whose NC sigmoid is >= 0.5.

## 10. Limits stated in advance

- SELECTION tier, SEEN scenes, P2 IN-SAMPLE: P2's picks are optimistic on these sets and the dPDMS of an out-of-fold composite against them is biased AGAINST the composite; the VETO-RECOVERS condition (ii) inherits that bias and is fixed by the brief, not softened here.
- Label-lookup PDMS, not a closed-loop or navtest number. Tier: T0-class selection read. No capability claim.
- One P2, one S0 bank, two EP-row training seeds: the interval answers "another draw of LOGS" only (`H-ESTIM-SEED-1`); the two seeds are a floor on training variance of the EP row, not of P2.
- The NC = 0.5 entries (8,343 proposals) are neither safe nor unsafe in Q1 (a pick with NC 0.5 is not "NC-unsafe"); the `label < 1` sensitivity is reported, with no authority.
- The sigmoid threshold 0.5 is the brief's, not calibrated; an NC row that ranks well but is miscalibrated above 0.5 on its unsafe picks reads ROW-MISS here although a temperature fix would flag it. That case is the committed ROW-MISS next arm (calibration), not a contradiction.
- V1's own picks are NOT re-examined: they live on navtest (no S0 equivalent), and this arm does not open navtest.
