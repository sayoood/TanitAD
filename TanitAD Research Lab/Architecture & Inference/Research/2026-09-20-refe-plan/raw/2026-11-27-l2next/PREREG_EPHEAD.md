# PREREG_EPHEAD: re-fit C1's EP (progress) read-out alone on tracked EP, selected by 5-fold log-level cross-validation (plan L2 next arm, `H-REFE-SAFEHEAD-1` successor)

REFe-line agent, 2026-11-27. Modelled on `PK/raw/2026-11-27-safehead/PREREG_SAFEHEAD.md`; its estimator, bars, controls and outcome structure are reused, and every deviation is listed in 9.
**State at freezing (disclosed):**
- The code is written and tested on SYNTHETIC data only: `refe/test_safehead_next.py` (+ the unchanged `test_safehead.py`). Every number those tests print is synthetic.
- **No EP-HEAD head has been trained on real data.** No navtrain out-of-fold number, no held-out number, no navtest pick, score or switch count of any EP-HEAD head exists. The real cache `F:/safehead_cache/` was never opened by this agent (it is the data agent's, and it is read-only for this agent).
- The facts about the cache in 2.3 are **INHERITED** (SAFE-HEAD card 2.3, DATA_CONTRACT.md v1). The controls in 6 re-measure each of them at run time.
- Parent outcomes, which are facts at freezing: **L2 SAFE-HEAD = NULL-HELDOUT** (both seeds' held-out rule chose step 0) and **L3 EP-GEOM = NULL-HELDOUT**. The diagnostic `PK/raw/2026-11-27-safehead/diag_veto/NOTE.md` (descriptive, zero training, MEASURED T0) is the only SAFE-HEAD-family navtrain number this card has seen: on the fit split the safety head cuts safety misses 69 -> 41, on the 6 held-out logs 18 -> 20, and every pick change costs EP (-2 to -4 points).

## 1. Question
C1 is REFe's best (MODEL_REGISTRY 14.6: **C1 92.5548 [91.7982, 93.2579]**, T1, 12,146 tokens). SAFE-HEAD re-fitted the four SAFETY rows and the held-out rule (6 logs) could not tell a gain from noise. The diagnostic says the agg32 argmax sits among near-tied proposals and that moving the pick pays in EP.
Does C1's navtest tracked PDMS rise if ONLY row 2 of `score_head` (**EP**, the progress read-out) is re-fitted on on-policy navtrain proposals with the tracked SOFT EP label, when the saved step is chosen by a selection rule that uses **all 35 navtrain logs** (5-fold log-level cross-validation) instead of 6?
The trunk, the proposals, the scoring decoder, the safety and comfort read-outs and the planner rule all stay C1's.

**Motivation, disclosed as SNOOPED.** `PLAN_REFE_9455.md` 1.2 and the L3 card: EP is the dominant share of C1's selection regret (2.47 points of the oracle-of-64 decomposition, navtest populations). The SAFE-HEAD diagnostic above is a navtrain read. Navtest is snooped for this question. **Nothing in this card is fitted on navtest.** The hyperparameters in 3 are SAFE-HEAD's, set a priori. The one data-driven choice, the saved step, is made by out-of-fold navtrain PDMS (4.3). Navtest is read once, under this card.

## 2. Arms, anchor and data
| arm | what | role |
|---|---|---|
| **C1** | checkpoint `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4` (registry 14.6); its banked seam pick | comparator |
| **EPH_s0** | C1 with `score_head` row 2 trained per 3, `--seed 0`, `--select cv5`; the CV rule's step | **the decision arm** |
| **EPH_s1** | the same recipe, `--seed 1` (the data order) | the training-seed replicate R (`H-ESTIM-SEED-1`) |
| **EPH_RED** | the same recipe, seed 0, EP labels permuted across ALL tokens (`--permute-labels-seed 20261127`) | CTRL-RED |

### 2.1 The row, from code
- `score_head = nn.Linear(256, 6)` (`model.py:563`); order NC, DAC, EP, TTC, C, DDC (`train.py:269`, `planner.py:547-549`). **Row 2 = EP.**
- EP-HEAD trains **row 2 alone**: 256 + 1 = **257 parameters**, in one optimiser group `row2`. Rows 0, 1, 3, 4, 5 are leaves outside the optimiser; their gradient is the zero control. `--rows 2` is only accepted together with `--select cv5` (the trainer refuses otherwise).
- The selection rule of the C1 seam (`navsim_v1`, `planner.py:558-561`) is NC x DAC x (5 EP + 5 TTC + 2 C) / 12, so **EP is in the aggregate with weight 5/12 and a changed EP logit CAN move a pick** (unlike DDC in SAFE-HEAD).

### 2.2 The anchor (a table read; no new seam) - identical to SAFE-HEAD 2.2
- C1's 64-proposal tracked table `full_C1_s{0,1}of2.jsonl` (sha256 s0 `9d1239a2...`, s1 `77588849...`), the banked dump `stage2_p5c_c1.proposals.npz` (`b258e50a...`) and seam `stage2_p5c_c1.npz` (`8df306ff...`); 12,146 tokens, 136 logs. PDMS(any pick) = the table's `tr.pdms` at that proposal.
- ⚠️ The two table shards exist on E: only (not in git, HF UNVERIFIED). Banking them is an MM item (SAFE-HEAD card 11) and a precondition of this read.

### 2.3 The data (SAFE-HEAD 2.3, `DATA_CONTRACT.md` v1; loader `refe/safehead_data.py`)
- **Navtrain cache: 2,272 tokens in 35 logs** (`split = train` 29 logs, `split = heldout` 6 logs). Per token: `s` [64,256] (the exact input of `score_head`), C1's `logits` [64,6], `pick`, `props`, and the tracked labels `tr_nc, tr_dac, tr_ddc, tr_ttc, tr_ep, tr_c, tr_pdms` [64].
- ⛔ **THE EP LABEL IS A REBUILT METRIC CACHE (INHERITED, DATA_CONTRACT.md; EP-GEOM card 3).** Against the harness on 57-60 navtest picks NC, DAC, TTC, DDC and C agree exactly, **EP differs on 6 of them (max 0.0108)** and PDMS by at most 0.0045 (mean |d| 0.0008). SAFE-HEAD therefore refused any EP/PDMS label for a fitted or selected quantity. **EP-HEAD cannot avoid that: its label IS `tr_ep`, and its selection IS `tr_pdms`.** The deviation is deliberate and bounded:
  - the fit target is the cache's soft EP; the selection ranks five saved steps by the cache's PDMS (approximate by <= 0.0045 on the audited picks);
  - **every navtest number is the harness-exact table's** (2.2), never the cache's, so a cache-vs-harness bias can only make the head fit a slightly different EP and can never inflate a reported score;
  - the risk is named: a systematic bias of the cache EP would be learned by the head. It is UNVERIFIED at freezing (10) and is **audited at run time by a descriptive M5 row** (the head's pooled navtest rank-correlation with the harness-exact `tr.ep`, vs C1's).
- **Folds (new).** The 35 logs, sorted by NAME, get fold = index mod 5 (7 logs per fold). The assignment is a function of the log names only, not of the seed. Each token inherits its log's fold. `safehead_data.cv_folds` is the one definition.
- **Sealing** is SAFE-HEAD's: the trainer accepts only the SEALED index (sha256 per file, re-hashed at load); the token binds it.
- The held-out logs are held out from the HEAD fit **in the 4.3 sense only inside each fold**; in the final fit all 35 logs train the head (4.3). This is the change from SAFE-HEAD and the reason a 6-log rule is no longer the bottleneck.
- Navtest `s` cache: the same tool on the 12,146 navtest tokens; it carries no labels.

## 3. The training recipe (every value fixed now; `refe/safehead_train.py`, `refe/safehead_gate.py` `FAMILIES["ephead"]`)
- **Init:** C1's `score_head.weight[2]` and `.bias[2]`. The trainer refuses unless C1's head on the cached `s` reproduces the cached logits (CTRL-FEAT, max and p99 <= 1e-4).
- **Parameters:** one weight row [256] + one bias (257), group `row2`.
- **Loss (ONE):** quality-focal BCE, `mean |y - sigmoid(z)|^gamma * BCE(z, y)` over (token, proposal), gamma **2.0**, on the **soft** label `tr_ep` in [0, 1] (Li et al. 2020 quality form). All 64 proposals enter. No class weight, no other term. (`TERM_OF_ROW[2] = "qfl_ep"`.)
- **Optimiser:** Adam lr **3e-4**, betas (0.9, 0.999), eps 1e-8, weight decay 0. **Schedule:** batch **32 tokens** (x 64 proposals), **2,000 steps**, constant lr, an epoch permutation from `numpy.default_rng(seed)`, partial batch dropped. **Saved steps 0, 250, 500, 1000, 2000.** Step 0 is C1's head exactly.
- **Determinism:** CPU, `torch.set_num_threads(1)`, `use_deterministic_algorithms(True)`. G-CKPT proves bit-identical repeats, including the CV report.
- **Seeds:** 0 (decision), 1 (R), RED seed 0 with permutation seed 20261127 (a permutation of ALL 2,272 tokens' label rows; the OUT-OF-FOLD evaluation of RED uses the TRUE labels).
- **Cost:** seconds of CPU per fit (ESTIMATED: the synthetic 2,000-step run takes ~10 s); a run = 5 fold fits + 1 final fit. No GPU.

## 4. The read (`safehead_read.py`, generalised per 9; table lookup exactly as SAFE-HEAD 4)
**4.1 Logits.** `L = banked_logits + s_nt @ (W - W0)[2].T + (b - b0)[2]` on **row 2 only** (`safehead_data.head_logits`). The untrained head returns the banked logits bit for bit. NC, DAC, TTC, C, DDC are never touched.
**4.2 Selection.** `k = argmax agg32(L)`, float32 numpy (the form PREREG_P5C_NEXT AMENDMENT_1 validated: 36,438 / 36,438). No float64 anywhere (`D-REFE-SRULE-1`).
**4.3 The cross-validation rule (the only selection; pre-registered here).**
- For each fold k in 0..4: a FRESH model (the same seed) is trained with the 3 recipe on the tokens of the other four folds, and at every saved step its head scores the fold-k tokens (the read's delta arithmetic, 4.1). So every one of the 2,272 tokens is scored at every saved step by a head that never saw its log.
- **Out-of-fold PDMS(step)** = the POOLED mean over all 2,272 tokens of the cache's `tr_pdms` at `argmax agg32` of that token's out-of-fold logits, x 100. Step 0 is C1's own head and is a column of the same table.
- **The chosen step is the ARGMAX of out-of-fold PDMS; ties go to the EARLIEST step** (exact float equality). No margin, no tolerance.
- **If the chosen step is 0 the outcome is NULL-CV**: EPH_s0 is C1 and no navtest decision is read.
- **The final head** is trained on ALL 35 logs (all 2,272 tokens) with the same recipe and seed; the saved head at the chosen step is the decision head. (Training is prefix-deterministic: G-CKPT proves the step-2 head of a 4-step run equals a 2-step run's. So "trained to the chosen step" and "trained to 2,000 and the chosen step saved" are the same head.)
- EPH_s1 applies the same rule to its own run. A seed-1 rule that chooses step 0 gives Delta_1 = 0, so R fails (5).
- The rule is computed by the trainer (`cv_select.json`), frozen before navtest is touched. It uses the cache's `tr_pdms` (2.3, approximate) and ONLY ranks steps.
- ⚠️ What this rule does and does not do: it sees EP, safety and comfort through PDMS (that is the point: SAFE-HEAD's safety-only rule could not see the EP cost of a pick change). It has no noise margin, so a spurious +0.001 can choose a non-zero step; that only OPENS the navtest read, whose bars (5) are the decision. The descriptive per-log counts `[improved, worsened, n_logs]` per step are written for the reader and are not part of the rule.
**4.4 Score.** PDMS(EPH) = the table's `tr.pdms` at `k`; sub-scores are `tr` at `k`.

**Tier, estimator, population.** **T1**: NAVSIM v1.1 PDMS, LQR-tracked, the anchor table. All 12,146 navtest tokens, 136 logs. **Estimator:** paired log-cluster bootstrap, `eval/p5c/p5c_read.py` `Boot`, **10,000 draws, seed 20260927**, 95 % percentile, `full_set` means; the read refuses any other `Boot`. **Which variance each interval answers:** the bootstrap answers *"another draw of LOGS"*; EPH_s1 answers *"another training run"*; inference variance is zero (a deterministic argmax over fixed logits). The CV selection noise is part of the training pipeline and is covered by R.

## 5. Measures, bars and the committed reading
**Measures.** M1 Delta0 = PDMS(EPH_s0) - PDMS(C1) with its CI. M2 sub-score deltas NC, DAC, TTC, EP, C (judged) and DDC (reported) x100, each with a CI. M3 Delta1 for EPH_s1 with sub-scores. **M4 the four families** (binding, Sayed 2026-08-02), each paired EPH vs C1 on the same tokens with CI and n, or a per-family reason and n:
- LONGITUDINAL and LATERAL: `p5c_read_prep.py convert/score` on the 8-pose seams `seam_{EPH0,EPH1,C1pick}.npz`, then `p5c_families.py --pairs EPH0:C1pick EPH1:C1pick` via `run_fam_chain.sh`. EP is the lever most likely to move LONGITUDINAL (progress); the family is read in full, not assumed.
- TACTICAL: `p5c_tactical.py` command adherence plus selection quality from the table (regret to the oracle, top-1 rate, zeroed picks while a proposal > 0 exists).
- STRATEGIC in full (PI 2026-11-16): `instruments/strategic_family.py read`.
- Every separated-adverse family component is named in the verdict.
**M5 (descriptive only):** the switch rate; safety-fail-with-alternative counts, fixed and created; pooled per-row navtest AUC and **the head's pooled rank-correlation with the harness-exact `tr.ep`, C1 vs EPH (the 2.3 audit)**; the direct-form pick agreement; the per-step out-of-fold table (PDMS, safety-miss rate, EP, switch rate, per-fold PDMS, per-log counts).
**F** = |PDMS(EPH_s0) - PDMS(EPH_s1)|, the training-seed floor of this rig (one draw, 1 df). F_REFe = **0.1158** (`PREREG_C1R_READ.md:35`). **Bar P2 = 2 x max(F, F_REFe).**

**Bars.**
- **P1** CI-lo(Delta0) > 0.
- **P2** Delta0 >= 2 x max(F, F_REFe).
- **G** no separated-adverse judged sub-score on EPH_s0: for none of NC, DAC, TTC, EP, C is CI-hi(EPH_s0 - C1) < 0.
- **R** Delta1 > 0 and no separated-adverse PDMS or judged sub-score on EPH_s1.
- **Every control in 6 PASSES.** Otherwise INVALID and every number is descriptive.

**Outcomes and next arms, in precedence order (Rule Zero: each non-ADOPT names its next arm).**
1. **INVALID.** A control FAILED. Next: fix that control under a dated amendment and re-read.
2. **NULL-CV.** Seed 0's CV rule chose step 0. EPH_s0 is C1; navtest is not read for a decision. **Next arm: DATA-SCALE** (8.1).
3. **ADVERSE.** CI-hi(Delta0) < 0, or G fails. Not adopted; the verdict names each adverse sub-score. **Next arm: DATA-SCALE (8.1) with those sub-scores carried into its guard.**
4. **ADOPT** = P1 ^ P2 ^ G ^ R. EPH_s0's row 2 becomes C1's EP read-out via `safehead_data.export_full` (asserts every non-head tensor bit-identical), plus a SPEC amendment the MM lands, and a registry row. The goal status is printed as written: **BELOW GOAL** if the point is below 94.55. **Next arm: the composition EPH + SAFE-HEAD/L1 tracked set (8.2), own card.**
5. **SEPARATED-SMALL** = P1 ^ G, but P2 or R fails. Labelled *"separated, single seed"* or *"inside the seed floor"*. Not adopted. **Next arm: DATA-SCALE (8.1) with 3 seeds.**
6. **NULL.** The CI of Delta0 contains 0. Not adopted. **Next arm: DATA-SCALE (8.1).**
- **INCOMPLETE** (at `confirm`): a family output is missing. A work item, never a result.

## 6. Controls (each can FAIL; any FAIL = INVALID)
- **CTRL-ID.** C1's untrained head through THIS read on the cached navtest `s`: logits bit-identical to the banked dump; picks equal the banked seam picks on **12,146 / 12,146**; per-token PDMS exactly the table's, mean 92.5548 to 5e-5. **On the CV side the trainer FAILS the run unless step 0's out-of-fold picks equal C1's cached pick on all 2,272 tokens.**
- **CTRL-DET-NT, CTRL-FEAT, CTRL-FEAT-NT, CTRL-S, CTRL-SET, CTRL-T2, CTRL-SHA**: as SAFE-HEAD 6 (the navtest cache is the banked forward; C1's head reproduces the cached logits <= 1e-4; the cache's pick is the planner rule on its logits; 12,146 tokens / 136 logs; harness equals the table at the new picks to 1e-6; every pin and each run's own gate token). CTRL-SHA also binds each head's `index_sha256` to the sealed train index and the card's sha256.
- **CTRL-REACH**, read from each run's own `metrics.jsonl` of the FINAL fit (all 2,000 steps): non-zero gradient on row 2; **EXACTLY 0.0** on rows 0, 1, 3, 4, 5; rows 0, 1, 3, 4, 5 bit-identical to C1 in every saved head (`load_head` refuses otherwise). **Mutations, MEASURED in `test_safehead_next.py`:** a trained row removed from the optimiser, and a frozen row (NC) entering the loss, each turn G-LIVE RED.
- **CTRL-CV (new).** (a) The folds in `built.json` equal sorted-log-name index mod 5 for every log and each log is in exactly ONE fold (G-DVB re-derives them from the sealed index; the mutation `sh_cv_token_folds` turns G-DVB RED); (b) `cv_select.json` has 2,272 tokens, 35 logs, 5 folds; (c) the final fit used all 2,272 tokens (`built.n_train`); (d) the chosen step in `run.json` equals the argmax of `cv_select.json` with ties earliest.
- **CTRL-RED.** EPH_RED's out-of-fold PDMS at step 2000 must read **below its own step 0**, AND its CV rule must choose **step 0**, AND its step-0 entries must equal EPH_s0's. A path whose training or selection ignores the labels FAILS this. No navtest number of EPH_RED is used.
- **CTRL-LEAK.** No train or held-out log is a navtest log and no token is shared; every held-out log is in `sft_heldout_logs.txt`, no train log is, and the two are disjoint. The trainer refuses at load; the read re-checks.
- **CTRL-CODE.** `CODE.sha256` lists every EP-HEAD/SH-MLP file; the launch token binds the code tree by commit and tree digest.
- **CTRL-SH-UNCHANGED.** The SAFE-HEAD output of the refactored trainer is bit-identical to the landed one (`test_safehead_next.py` t01, probe `d33a6eca...`, MEASURED against `e1eb6bf3`), and the three SAFE-HEAD profiles' fingerprints are unmoved.

## 7. The launch gate (CLAUDE.md, PI 2026-09-26: head-only training is training)
- **Profiles:** `refe_ephead_s0`, `refe_ephead_s1`, `refe_ephead_red` (new dicts, runner `safehead`), produced by `safehead_gate.py` `_next_profiles()` and registered through the unchanged `PROFILES.update(SHG.profiles())` in `launch_gate.py`: **`launch_gate.py` is NOT patched**. Each pins its argv exactly (`--rows 2 --select cv5`, every lever, the seed, the permutation seed for RED only) and requires the four data flags, bound by sha256 in the token.
- **Checks:** G-HYG (config refuses undeclared attributes; reads <= parser dests); **G-DVB** (every parser dest has an entry, **new flags `--mlp-hidden` and `--select` have theirs**, each built lever equals the argv, the parameter count equals the argv's implication (257), live terms = `qfl_ep`, the CV folds equal the re-derived ones, init = C1's sha256, data sha256 = manifest); G-LIVE (5-step real-data smoke: `qfl_ep` finite, row 2 has gradient, EXACTLY 0.0 elsewhere, row 2 moved, other rows bit-identical); G-EVAL (the loader builds the trainer's head <= 1e-4, the untrained head is bit-exact with picks n/n); G-CKPT (round trip, two identical runs bit-identical incl. the CV report, prefix property); G-SUITE (`test_safehead.py --fast`, `test_safehead_next.py --fast`, `test_launch_gate.py --static --no-banked`); G-CLOCK NA.
- **Deliberate-regression arms (each must turn its check RED):** the eleven SAFE-HEAD arms plus `sh_cv_token_folds` (G-DVB). The reachability mutations and `sh_cv_token_folds` are MEASURED in `test_safehead_next.py --fast` (13/13 PASS, synthetic); the full arm sweep runs in its full mode (see `regression/REGRESSION_VERDICT.md`). The trainer's own refusal of a missing or mismatched token and of any arm in a launch is unchanged.
- **No existing profile moved:** the 40 pre-SAFE-HEAD whole-profile fingerprints and the three SAFE-HEAD ones are pinned (t04 of `test_safehead_next.py`, MEASURED).

## 8. Committed next arms (specified now; none is computed under this card)
- **8.1 DATA-SCALE (on every non-ADOPT outcome; a PI decision).** The same recipe on an EXPANDED navtrain cache (more logs than 35): the CV rule, the 5 folds by sorted name and the bars unchanged. It needs (a) more navtrain logs run through C1's forward and the tracked PDM labeller (storage: `s` is 64 x 256 x 4 B = 64 KB per token plus `props`; the 2,272-token cache is ~150 MB of `s`), (b) the label rebuild's EP parity audit extended, (c) a new card. **Blocked on a PI decision (data and storage) and on the data agent.** Sizing from this card's own run: the CV report's per-fold PDMS spread and the per-log counts bound how many logs are needed.
- **8.2 After ADOPT: composition** with SAFE-HEAD's confidently-safe set / L1's tracked set (`TRK+EPH`), own card.
- **Parallel arm already carded: SH-MLP** (`PREREG_SHMLP.md`, SAFE-HEAD 8.1).

## 9. Deviations from the SAFE-HEAD card (the complete list)
1. Trained row: 2 (EP) alone, soft label `tr_ep`, instead of rows 0, 1, 3, 5 with 0/0.5/1 labels.
2. Selection: 5-fold log CV over all 35 logs on out-of-fold PDMS (cache `tr_pdms`), instead of 6 held-out logs on the safety-miss rate. Consequence: the cache's rebuilt EP/PDMS enter a fitted and a selected quantity (2.3).
3. Final fit on all 2,272 tokens instead of the 29 train logs.
4. NULL-CV replaces NULL-HELDOUT; the next arm is DATA-SCALE, not SH-MLP.
5. CTRL-RED reads out-of-fold PDMS and the chosen step; CTRL-CV and CTRL-SH-UNCHANGED are new.
6. **The read script must be generalised before launch** (NOT frozen by this card's `CODE.sha256`, which lists the trainer, gate and tests): `safehead_read.py` -> (a) head logits through `safehead_data.head_logits` with the rows taken from the head's meta; (b) the decision step from `cv_select.json` when it exists; (c) the profile names `refe_ephead_*` and the arm labels EPH0/EPH1; (d) CTRL-REACH over rows from meta; (e) CTRL-RED and CTRL-CV per 6; (f) M5 per 5. **No bar, outcome rule, estimator or control threshold may change in that edit**; it is frozen under a dated CODE amendment BEFORE any head is trained, and a diff that touches a bar is an INVALID launch.

## 10. UNVERIFIED at freezing (each is checked by a control, not assumed)
- The cache EP's agreement with the harness beyond the 57-60 audited picks (INHERITED); the head may learn a cache bias. Audited by the M5 correlation row; a sizeable gap is a finding, not a reason to move a bar.
- Whether 2,000 steps at 3e-4 moves a 257-parameter row far enough; the saved-step grid and NULL-CV catch the case where it does not.
- The CV rule's power: 35 logs, 7 per fold, per-log PDMS differences are noisy; it can only pick a different saved step (cannot touch navtest), and R re-runs the whole recipe including the selection.
- That EP is a lever at all: the 2.47-point EP share is oracle-informed (an upper bound on what a re-ranking could win, not an expectation).

## 11. Integration and banking (MM)
- Land `PK/raw/2026-11-27-l2next/` and `land_at_PK_refe/*` at `PK/refe/`: **changed** `safehead_data.py`, `safehead_train.py`, `safehead_gate.py`; **new** `test_safehead_next.py`; **test pins** `test_profile_pins.patch` (8 tests, 43 -> 49 profiles; the two that broke last time, `test_rep_gate` unlisted count and `test_scorerft_profile` allowed set, are patched the way commit 5eb0ae09 did). **`launch_gate.py` is untouched.**
- ⚠️ `safehead_data.py`, `safehead_train.py`, `safehead_gate.py` are the files the SAFE-HEAD `CODE.sha256` froze. Their landed copies change under this patch; the SAFE-HEAD run is finished (NULL-HELDOUT) and its outputs are unaffected (CTRL-SH-UNCHANGED); the SAFE-HEAD card's hashes remain the record of what ran.
- Bank the two C1 table shards to HF with sha256 verification (the only copy of the comparator).
