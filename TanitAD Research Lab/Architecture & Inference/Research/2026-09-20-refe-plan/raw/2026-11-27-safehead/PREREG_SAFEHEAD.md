# PREREG_SAFEHEAD: boundary-focused training of C1's safety heads, head-only, detached (plan L2, `H-REFE-SAFEHEAD-1`)

REFe-line agent, 2026-11-26 (the folder is dated 2026-11-27, as the brief names it). **Frozen at 2026-11-26T22:10Z, before any SAFE-HEAD number exists.** At freezing:
- The code is written and tested on SYNTHETIC data only: `refe/test_safehead.py` 14/14 and `test_safehead_read.py` 6/6 PASS. Every number those tests print is synthetic.
- **The real cache is being built by a separate data agent** (`F:/safehead_cache/`, `DATA_CONTRACT.md` v1). This agent has never opened it. No SH head has been trained on real data, and no held-out or navtest SH pick, score or switch count exists.
- The facts about the cache in 2.3 are **INHERITED** from the MM's note of 2026-11-26, which relays the data agent's validation. The controls in 6 re-measure each of them.
- The only real-data computations are C1's own, already-registered numbers, re-read through this card's arithmetic as a plumbing check (2.2).

## 1. Question
C1 is REFe's best (MODEL_REGISTRY 14.6: **C1 92.5548 [91.7982, 93.2579]**, T1, 12,146 tokens).
Does C1's navtest tracked PDMS rise if ONLY its four safety read-outs are re-fitted on on-policy navtrain proposals with tracked labels and a boundary-focused loss? Rows 0, 1, 3 and 5 of `score_head` are re-fitted. The trunk, the proposals, the scoring decoder, the EP and comfort read-outs and the planner rule all stay C1's.

**Motivation, disclosed as SNOOPED.**
- `PLAN_REFE_9455.md` 1.2: on SCORER-FT C's navtest tables, about 2.9 points of selection regret are safety misses that are "unsafe as planned".
- `D-REFE-HEADATTR-1`: C1's safety heads have pooled AUC 0.90–0.96 but are confidently wrong at the pass/fail boundary.
- 1.3: every progress-ward lever so far (S-RULE, SCORER-FT T and P) bought EP with safety.
- These are all navtest populations, so navtest is snooped for this question.
- **Nothing in this card is fitted on navtest.** The hyperparameters in 3 were set a priori, without any SH number. The one data-driven choice, the saved step, is made on 6 held-out navtrain logs (4.3). Navtest is read once, under this card.

## 2. Arms, anchor and data
| arm | what | role |
|---|---|---|
| **C1** | the checkpoint `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4` (registry 14.6); its banked seam pick | comparator |
| **SH_s0** | C1 with `score_head` rows 0, 1, 3, 5 trained per 3, `--seed 0`; the held-out rule's step | **the decision arm** |
| **SH_s1** | the same recipe, `--seed 1` (the data order) | the training-seed replicate R (`H-ESTIM-SEED-1`) |
| **SH_RED** | the same recipe, seed 0, labels permuted across train tokens (`--permute-labels-seed 20261127`) | CTRL-RED, read on held-out only |

### 2.1 The row → term mapping, from code
- `score_head = nn.Linear(dec_width = 256, n_score_components = 6)`: `refe/model.py:563`, `:96`. Its input `s` [64, 256] is the output of `score_q_mlp` → 4 × `CrossBlock` (`model.py:706-709`).
- The order is `train.py:269` `ScorerBank.COMPONENTS = ("collision", "drivable_area", "progress", "ttc", "comfort", "ddc")`, and `planner.py:547-549` states the same order. So **0 NC · 1 DAC · 2 EP · 3 TTC · 4 C · 5 DDC**.
- SH trains **rows 0, 1, 3, 5 = NC, DAC, TTC, DDC**: 4 × (256 + 1) = **1,028 parameters**. Rows 2 (EP) and 4 (C) are never parameters, and no other tensor of C1 is loaded.
- ⚠️ **DDC is selection-inert and score-inert.**
  - The C1 seam's rule `navsim_v1` (`planner.py:525`, `:558-561`) is NC × DAC × (5 EP + 5 TTC + 2 C) / 12, so DDC does not enter it.
  - The tracked PDMS on the anchor table is the same formula: MEASURED max |difference| 0 over 400 tokens × 64 proposals, with DDC excluded. With DDC included the difference is 1.0.
  - So training row 5 cannot change a pick or a score. It is trained as the brief specifies and reported only.
  - **The PLAN's "DDC 0.15" safety regret is unreachable by this lever.**

### 2.2 The anchor (a table read; no new seam)
- **Table:** C1's own 64-proposal tracked table, `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s{0,1}of2.jsonl` (`PREREG_P5C_NEXT.md` 4).
  - Shard sha256: s0 `9d1239a250c021b89480b0079f2d66ac7b319a3eaee73c80a3deac208b89b42e`; s1 `77588849ce5a3080e5ec941f71ba1b52f380fdc04d406849416e8c73c99a99d0`.
  - 12,146 tokens, 136 logs. Per proposal: `tr` (NAVSIM v1.1, LQR-tracked) nc / dac / ddc / ttc / c / ep / pdms.
  - Its read `full_C1_read.json` has CTRL_T PASS: the table at the pick equals the harness CSV, max 1.06e-11 on PDMS.
  - ⚠️ **The shards are on E: only** (40 MB each, not in git, HF UNVERIFIED). Banking them is an MM item (11).
- **The banked dump:** `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.proposals.npz`, sha256 `b258e50a722101c090861c6f1f0c26d05b38de966fffeb84f0ea17c3dce95007`. It holds the proposals [12146, 64, 8, 3], float32 logits [12146, 64, 6] and the pick.
- **The seam:** `stage2_p5c_c1.npz`, sha256 `8df306ff940e50d6699942b9e61836c8193b4ed5ef21c32ec0d755859d349995`.
- **Plumbing check, MEASURED 2026-11-26 with this card's code.** These are C1's numbers, not SH's:
  - agg32 on the banked logits reproduces the banked pick on **12,146 / 12,146** tokens;
  - the table's pick equals the dump's on every token;
  - the seam poses equal `proposals[pick]`;
  - the table at the pick gives **PDMS 92.55477798611808**, which is registry 14.6's 92.5548.
- So **PDMS(any pick) = the table's `tr.pdms` at that proposal**, and its sub-scores are `tr` at that proposal.

### 2.3 The data (`DATA_CONTRACT.md` v1, `PK/raw/2026-11-27-safehead-data/`; loader `refe/safehead_data.py`)
- **Navtrain cache:** **2,272 tokens in 35 logs.**
  - `split = train` covers 29 navtrain logs; `split = heldout` covers 6 logs from `splits/sft_heldout_logs.txt`.
  - The layout is one npz per log, with arrays stacked over its tokens and a `token` array. `index.csv` holds `token, log, split, file, sha256`.
  - Per token: `s` float32 [64, 256], the exact captured input of `score_head`; C1's `logits` [64, 6]; C1's `pick`; `props`; `props8`; and the tracked labels `tr_nc, tr_dac, tr_ddc, tr_ttc, tr_ep, tr_c, tr_pdms` [64].
- **The labels are a REBUILT metric cache** (contract caveats, INHERITED):
  - against the harness on 57–60 navtest picks, **NC, DAC, TTC, DDC and C agree exactly**;
  - **EP differs on 6 of them (max 0.0108)**, and PDMS by at most 0.0045.
- ⛔ **So this card uses ONLY the safety labels `tr_nc, tr_dac, tr_ttc, tr_ddc` for any fitted or selected quantity.**
  - That covers the loss (3), the held-out rule (4.3) and CTRL-RED (6).
  - The cache's `tr_ep` and `tr_pdms` are printed as *descriptive, approximate* only.
  - The validator's PDMS-identity check is reported and never gates.
  - Every **navtest** score comes from the anchor table (2.2), never from the cache.
- The held-out logs are held out from the HEAD fit, not from C1's own training: navtrain is C1's training distribution.
  - No train log is among the 24 SFT held-out logs, and no navtrain log or token is a navtest one (CTRL-LEAK).
- **Navtest `s` cache:** the same tool on the 12,146 navtest tokens. It carries no labels.
- **Sealing:** the trainer and the read accept only a **SEALED** index, written by `safehead_data.py seal` into the run directory with a sha256 per file. Every file is re-hashed against the seal at load. The launch token binds the sealed index by sha256.

## 3. The training recipe (every value fixed now; `refe/safehead_train.py`, `refe/safehead_gate.py` `REGISTERED`)
- **Init:** C1's `score_head.weight` / `.bias`. The trainer refuses unless C1's head on the cached `s` reproduces the cached logits (CTRL-FEAT): **max and p99 ≤ 1e-4**. The data agent's CTRL-LOGIT read 7.6e-6 (INHERITED).
- **Parameters:** one weight row [256] plus one bias per trained row, in four optimiser groups `row0, row1, row3, row5`. The EP and C rows are leaves outside the optimiser; their gradient is the zero control.
- **Loss (ONE):** quality-focal BCE (Li et al. 2020; for 0/1 labels it is the focal loss), `mean |y − σ(z)|^γ · BCE(z, y)` per row over (token, proposal), then the mean of the 4 rows.
  - γ = **2.0**. Targets are the tracked labels (NC and DDC may be 0.5).
  - All 64 proposals of a token enter. There is no class weight and no other term.
- **Optimiser:** Adam with lr **3e-4**, betas (0.9, 0.999), eps 1e-8, weight decay **0**.
- **Schedule:** batch **32 tokens** (× 64 proposals) per step, **2,000 steps**, constant lr. The order is an epoch permutation from `numpy.default_rng(seed)`, and a partial batch is dropped.
- **Saved steps:** **0, 250, 500, 1000, 2000.** Step 0 is C1's head exactly.
- **Determinism:** CPU, `torch.set_num_threads(1)`, `use_deterministic_algorithms(True)`. G-CKPT proves bit-identical repeats.
- **Seeds:** **0** (decision) and **1** (R). SH_RED uses seed 0 and permutation seed 20261127.
- **Cost:** seconds of CPU per run (ESTIMATED; the synthetic 2,000-step run takes ~10 s). No GPU.

## 4. The read (`safehead_read.py`)
**4.1 Logits.** `L = banked_logits + s_nt @ (W − W0)[rows].T + (b − b0)[rows]` on rows 0, 1, 3, 5 only. `banked_logits` are the dump's float32 logits and `s_nt` is the cached navtest `s`.
- The untrained head returns the banked logits bit for bit.
- EP and C are never touched.
- The deployed head `s @ W.T + b` differs from this by float rounding of `s` only; the direct-form pick agreement is reported (M5).

**4.2 Selection.** `k = argmax agg32(L)`. This is the planner's navsim_v1 in float32 numpy, in the exact form `PREREG_P5C_NEXT` AMENDMENT_1 validated (36,438 / 36,438). There is no float64 anywhere: that is the `D-REFE-SRULE-1` dtype trap.

**4.3 The held-out rule (the only selection; SAFETY LABELS ONLY).**
- For each saved step, compute the **held-out safety-miss rate**: the share of held-out tokens whose pick fails NC, DAC or TTC (label < 1) while some proposal passes all three.
  - The pick is `argmax agg32` of 4.1's logits, with the held-out tokens' own cached logits as the base.
  - DDC is excluded because it is inert (2.1). The cache's EP and PDMS are not used (2.3).
- **The chosen step is the LOWEST; ties go to the EARLIEST step.**
- ⚠️ Progress is not seen by this rule. A step that buys safety with EP is caught on navtest by G (EP is judged there, from the harness-exact table) and by P1 on PDMS.
- It is computed by the trainer at the end of each run (`heldout_select.json`), on the 6 held-out logs only, and is frozen before navtest is touched.
- SH_s1 applies the same rule to its own run.
- If seed 0's rule chooses **step 0**, the outcome is **NULL-HELDOUT**: SH_s0 is C1, and navtest is not read for a decision.

**4.4 Score.** PDMS(SH) = the table's `tr.pdms` at `k`. Sub-scores are `tr` at `k`.

**Tier, estimator, population.**
- **T1:** NAVSIM v1.1 PDMS, LQR-tracked, the anchor table.
- **Population:** all 12,146 navtest tokens, 136 logs.
- **Estimator:** paired log-cluster bootstrap, `eval/p5c/p5c_read.py` `Boot`, **10,000 draws, seed 20260927**, 95 % percentile, on `full_set` means. It is the C1r / TRK estimator, and the read refuses a `Boot` with any other seed or draw count.
- **Which question each variance answers:**
  - The bootstrap answers *"would another draw of LOGS say this?"*.
  - Training variance is SH_s1's question.
  - Inference variance is zero: the selection is a deterministic argmax over fixed logits.

## 5. Measures, bars and the committed reading
**Measures.**
- **M1** `Δ0 = PDMS(SH_s0) − PDMS(C1)`, with its CI.
- **M2** Sub-score deltas NC, DAC, TTC, EP, C (judged) and DDC (reported only), each ×100 with its CI.
- **M3** `Δ1 = PDMS(SH_s1) − PDMS(C1)`, with sub-scores.
- **M4 — the four families** (binding, Sayed 2026-08-02). Each is paired SH vs C1 on the same tokens, with its CI and n, or with a per-family reason and n where it cannot be computed.
  - **LONGITUDINAL and LATERAL:** `p5c_read_prep.py convert/score` on the 8-pose seams `seam_{SH0,SH1,C1pick}.npz`, then `p5c_families.py --pairs SH0:C1pick SH1:C1pick`. This is the 8-pose like-for-like form of `run_srule_families.sh`; the transparent arm is built from the 8 poses, not the registered 20-pose form, and is labelled so.
  - **TACTICAL:** `p5c_tactical.py` command adherence, plus selection quality from the table: regret to the oracle, top-1 rate, and the share of zeroed picks while a proposal > 0 exists.
  - **STRATEGIC in full** (PI 2026-11-16): `instruments/strategic_family.py read`.
  - Every separated-adverse family component is named in the verdict.
- **M5 (descriptive only):**
  - the switch rate;
  - safety-fail-with-alternative counts, fixed and created (NC/DAC/TTC fail at the pick while a proposal passing all three exists);
  - pooled per-row navtest AUC, C1 vs SH;
  - the direct-form pick agreement;
  - the held-out per-step table.
- **F** = |PDMS(SH_s0) − PDMS(SH_s1)|, this rig's training-seed floor (one draw, 1 df). F_REFe = **0.1158** (`PREREG_C1R_READ.md:35`, one draw). **Bar P2 = 2·max(F, F_REFe).**

**Bars.**
- **P1** CI-lo(Δ0) > 0.
- **P2** Δ0 ≥ 2·max(F, F_REFe).
- **G** No separated-adverse judged sub-score on SH_s0: for none of NC, DAC, TTC, EP, C is CI-hi(SH_s0 − C1) < 0.
- **R** Δ1 > 0, and no separated-adverse PDMS or judged sub-score on SH_s1.
- **Every control in 6 PASSES.** Otherwise the outcome is INVALID, and every number is descriptive.

**Outcomes and next arms, in precedence order (Rule Zero: each non-ADOPT names its next arm).**
1. **INVALID.** A control FAILED. Next: fix that control under a dated amendment and re-read.
2. **NULL-HELDOUT.** Seed 0's held-out rule chose step 0. **Next arm: SH-MLP (8.1).**
3. **ADVERSE.** CI-hi(Δ0) < 0, or G fails. **Not adopted.** The verdict names each adverse sub-score. **Next arm: SH-TOPK (8.2)**, with those sub-scores carried into its guard.
4. **ADOPT** = P1 ∧ P2 ∧ G ∧ R.
   - SH_s0's head becomes C1's scorer read-out, via `safehead_data.export_full`, which asserts every non-head tensor is bit-identical, plus a SPEC amendment the MM lands.
   - It gets a registry row. The goal status is printed as written: **BELOW GOAL** if the point is below 94.55.
   - **Next arm: L3 EP-GEOM inside the SH-scored safe set (8.4).**
5. **SEPARATED-SMALL** = P1 ∧ G, but P2 or R fails. Labelled *"separated, single seed"* or *"inside the seed floor"*. **Not adopted.** **Next arm: SH-MLP (8.1) with 3 seeds.**
6. **NULL.** The CI of Δ0 contains 0. **Not adopted.** **Next arm: SH-MLP (8.1).**
- **INCOMPLETE** (at `confirm`): a family output is missing. That is a work item, never a result.

## 6. Controls (each can FAIL; any FAIL = INVALID)
- **CTRL-ID.** C1's untrained head, run through THIS read on the cached navtest `s`, must give:
  - logits bit-identical to the banked dump;
  - picks equal to the banked seam picks on **12,146 / 12,146**;
  - per-token PDMS exactly the table's at C1's pick, with mean = 92.5548 to 5e-5.
  - The held-out side is the trainer's: step 0 must reproduce the cached picks on every held-out token, or the run FAILS.
- **CTRL-DET-NT.** The navtest cache is the banked forward: the cached navtest picks equal the banked picks on 12,146 / 12,146, and max |cached − banked logits| ≤ 1e-4. This licenses the table as SH's comparator: same 64 proposals, same order.
- **CTRL-FEAT.** C1's head on the cached float32 `s` reproduces the cached logits: **max and p99 ≤ 1e-4**. It applies to the navtrain cache (the trainer refuses) and to navtest, where the banked logits are the target (CTRL-FEAT-NT). A wrong tensor is off by O(1), and an `s` stored at float16 FAILS (MEASURED, `test_safehead.py` t07).
- **CTRL-S.** The cache validator requires argmax agg32(cached logits) = the cached pick on every token, in float32.
- **CTRL-REACH**, read from each run's own `metrics.jsonl`, all 2,000 steps:
  - non-zero gradient on every trained row;
  - **EXACTLY 0.0** on EP and C;
  - EP and C rows bit-identical to C1 in every saved head; `load_head` refuses otherwise.
  - Mutation: a row removed from the optimiser, or EP entering the loss, turns the gate's G-LIVE RED. This is MEASURED in `test_safehead.py` t09, as deliberate-regression arms `sh_reach_row_dropped` and `sh_reach_ep_in_loss`.
- **CTRL-RED.** SH_RED's final step (2,000), on the held-out logs, must read below the untrained head on BOTH:
  - (a) the AUC of each of NC, DAC and TTC, and
  - (b) the held-out safety-miss rate: RED's must be HIGHER. Safety labels only.
  - The two runs' step-0 entries must be identical.
  - A path whose training or read ignores the labels FAILS this. No navtest number of SH_RED is used.
- **CTRL-LEAK.** No train or held-out log is a navtest log (the 136 from the anchor table) and no token is shared. Every held-out log is in `sft_heldout_logs.txt`, no train log is, and train and held-out are disjoint. The trainer refuses at load, and the read re-checks.
- **CTRL-SHA.** The pins must equal:
  - the anchor table shards, dump and seam (2.2);
  - every decision head against its run's `heads.sha256`;
  - each head's init = C1's sha256;
  - each head's `index_sha256` = the sealed train index;
  - the card's own sha256.
  - Each run must have run on its own arm's gate token (`run.json`).
  - A gate SMOKE head is refused.
- **CTRL-SET.** 12,146 tokens and 136 logs. The tokens of the table, the dump, the seam and the navtest cache are equal.
- **CTRL-T2** (at `confirm`): the harness pass over the new seams equals the table at the new picks to 1e-6, for SH0, SH1 and C1pick.
- **CTRL-CODE.** `CODE.sha256` lists every SAFE-HEAD file. The launch token binds the code tree by commit and tree digest.

## 7. The launch gate (CLAUDE.md, PI 2026-09-26: head-only training is training)
- **Profiles:** `refe_safehead_s0`, `refe_safehead_s1`, `refe_safehead_red`, in `refe/launch_gate.py`'s PROFILES through `refe/safehead_gate.py` (runner `"safehead"`).
  - Each pins its argv exactly: the flag set, every lever, the seed, and the permutation seed for RED only.
  - Each requires the four data flags, which are bound by sha256 in the token.
- **Checks:**
  - **G-HYG:** the config refuses undeclared attributes; namespace reads ⊆ parser dests; config fields = parser dests.
  - **G-DVB:** every parser dest has an entry, **and a new flag without one is REFUSED**. Each built lever equals the argv (seed, steps, lr per optimiser group, batch, γ, rows, save steps, permutation). The Adam constants and the loss are the registered ones, the init sha256 is C1's, and the data sha256 values equal the manifest.
  - **G-LIVE:** a real-data smoke (5 steps): every declared term finite, every trained row with gradient, EXACTLY 0.0 on EP and C, every trained row moved, EP and C bit-identical.
  - **G-EVAL:** the read's loader builds the trainer's head (≤ 1e-4), its change equals the trainer's on the trained rows, EP and C are untouched, and the untrained head is bit-exact with picks n/n.
  - **G-CKPT:** save/load round trip, two identical runs bit-identical, and the step-2 head of a 4-step run equal to a 2-step run's.
  - **G-SUITE:** `test_safehead.py --fast` and `test_launch_gate.py --static --no-banked`.
  - **G-CLOCK:** NA, because a sealed label cache has no label clock.
- **Deliberate-regression arms:** each check's arm turns it RED. This is MEASURED, 11 arms, `test_safehead.py` t13.
- **The trainer's own refusal.** A real run starts only if `TANITAD_SH_GATE_TOKEN` names a PASS token that `gate_token.verify_token` MATCHes for this argv, tree and data. It also refuses under any regression arm. Both are MEASURED, t08 and t14.
- **No existing profile moved:** all 40 pre-existing whole-profile fingerprints are pinned in `test_safehead_prefp.json`, MEASURED t04.

## 8. Committed next arms (specified now; ⛔ none is computed under this card)
- **8.1 SH-MLP (capacity).** On each safety row, add a residual correction `z_r = C1_r(s) + MLP_r(s)` with MLP 256 → 64 → 1 (GELU) and a zero-initialised last layer, so step 0 is C1 exactly.
  - 66,052 parameters.
  - The same data, loss, optimiser, steps, held-out rule, seeds, read and bars.
  - It needs its own card and gate profile. If it is also NULL, the boundary is not shallowly decodable from `s`, and the next lever is 8.3.
- **8.2 SH-TOPK (selection-local).** SH's recipe, with the loss restricted to each token's top 8 proposals by C1's own agg32 on the cached logits, fixed at load.
- **8.3 Level 2: train `score_dec` + `score_q_mlp` + `score_head` on cached `visual_ctx`.** This is **BLOCKED** on storage (3.93 MB per token; about 3,000 tokens is E:'s limit, `FEASIBILITY_L2.md` c) and a GPU. It is a PI decision.
- **8.4 After ADOPT:** L3 EP-GEOM inside the SH-scored confidently-safe set, fitted on navtrain held-out.
- **Future compositions, named only:** **TRK+SH** (L1's tracked set scored by SH's head) and **ENS+SH** (L1b's two-seed ensemble with SH's safety rows). Each needs its own card after its parent cards read.

## 9. Run plan (the exact steps are in `HANDOFF.md`)
1. The data agent finishes the cache. Then `safehead_data.py check`, followed by `seal` of the train index and of the navtest index into the run directory.
2. `safehead_read.py prep`: write the navtest log list and re-check the anchor sha256 values.
3. For each arm s0, s1 and red: `launch_gate.py run` (a PASS token), then the trainer under that token. CPU, seconds.
4. `safehead_read.py read`, then the families chain (`run_fam_chain.sh`; CPU, harness venv), then `safehead_read.py confirm`, which gives the FINAL verdict.
5. Update `GOALS_AND_CLAIMS.md` `H-REFE-SAFEHEAD-1` in the same turn as the verdict (`gac_row_SAFEHEAD.md`).

## 10. UNVERIFIED at freezing (each is checked by a control, not assumed)
- The cache's file layout as built. The loader implements `DATA_CONTRACT.md` v1 (per-log npz plus a `token` array), plus per-token and stacked forms. It refuses anything else loudly (`safehead_data.py check`, the HANDOFF's first step).
- The held-out token count. The cache has 2,272 tokens in 35 logs; the split between train and held-out tokens is not known to this agent, and log slices hold 4 to 381 tokens (contract).
  - With 6 held-out logs the safety-miss rate is a coarse selector.
  - Its noise can only pick a different saved step: it cannot touch navtest, and R re-runs the whole recipe, selection included.
- Whether 2,000 steps at 3e-4 moves a 1,028-parameter head far enough. The held-out step grid, and NULL-HELDOUT, catch the case where it does not.
- The 2.9-point "unsafe as planned" bound is oracle-informed. It sizes the lever and is not an expectation.

## 11. Integration and banking (MM)
- Land this folder at `PK/raw/2026-11-27-safehead/`, and `land_at_PK_refe/*` at `PK/refe/`:
  - 5 new files: `safehead_data.py`, `safehead_train.py`, `safehead_gate.py`, `test_safehead.py`, `test_safehead_prefp.json`;
  - plus `launch_gate.py.patch`: 3 hunks, +12 lines, no existing line changed;
  - plus `test_profile_pins.patch`: 7 existing tests pin the post-tip profile set or count, as every new profile card has had to update (precedent: `refe_p5c_c1r`). This patch adds the 3 SAFE-HEAD names, moves 40 → 43, and moves `test_rep_gate` 19 → 22. The 7 files are `test_div_gate`, `test_dose_gate`, `test_field_p7_profile`, `test_rep_gate`, `test_s1zsm_gate`, `test_scorer1p6_gate` and `test_vdac_gate`. MEASURED: 4 of them failed on the patched tree without it.
- `stack/` is not touched: REFe's gate is `PK/refe/launch_gate.py`.
- Bank the two C1 table shards (2.2) to HF with sha256 verification. They are the only copy of this card's comparator.
