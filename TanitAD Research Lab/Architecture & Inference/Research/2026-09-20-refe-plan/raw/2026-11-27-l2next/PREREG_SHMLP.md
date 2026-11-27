# PREREG_SHMLP: a residual MLP on C1's four safety read-outs, head-only, detached (SAFE-HEAD card 8.1, `H-REFE-SAFEHEAD-1` successor)

REFe-line agent, 2026-11-27. **This card is SAFE-HEAD's section 8.1, made executable.** `PK/raw/2026-11-27-safehead/PREREG_SAFEHEAD.md` (sha256 `3e2c69b463cfd5954ad98b0d89f8c07e4ebd940fe76e040846856512200253d3`) applies VERBATIM to every item not listed in 2; only the differences are written here, so that a reader can diff the two cards by reading one page.
**State at freezing (disclosed):**
- The code is written and tested on SYNTHETIC data only: `refe/test_safehead_next.py` and the unchanged `test_safehead.py`. Every number those tests print is synthetic.
- **No SH-MLP head has been trained on real data**, and no held-out or navtest SH-MLP pick, score or switch count exists. The real cache `F:/safehead_cache/` was never opened by this agent.
- Parent fact at freezing: **SAFE-HEAD = NULL-HELDOUT** (both seeds' held-out rule chose step 0). The diagnostic `PK/raw/2026-11-27-safehead/diag_veto/NOTE.md` (descriptive, zero training, MEASURED T0): on the fit split the LINEAR head cuts safety misses 69 -> 41 (seed 0; 43 for seed 1), on the 6 held-out logs 18 -> 20 / 21; every pick change costs EP.
- The diagnostic's own reading is that DATA, not capacity, is the measured bottleneck ("capacity (SH-MLP, card 8.1) is not the measured bottleneck"), and that SH-MLP's **expected failure mode is a wider train/held-out gap**. This card is committed in spite of that reading, because SAFE-HEAD 8.1 names it, because it is a CPU-seconds run, and because a NULL here is what licenses the DATA-SCALE spend. It is read against the diagnostic table (6, CTRL-GAP).

## 1. Question
Is C1's safety boundary (the pass/fail edge of NC, DAC, TTC) shallowly decodable from the 256-d post-decoder feature `s` with a NON-linear read-out? Same population, bars and comparator as SAFE-HEAD: does C1's navtest tracked PDMS (**92.5548 [91.7982, 93.2579]**, MODEL_REGISTRY 14.6, T1) rise if the four safety read-outs gain a residual MLP, everything else frozen?

## 2. What differs from PREREG_SAFEHEAD (the complete list)
1. **Head form (SAFE-HEAD 8.1, verbatim): on each safety row r in {0 NC, 1 DAC, 3 TTC, 5 DDC}, `z_r = C1_r(s) + MLP_r(s)`**, `MLP_r: 256 -> 64 -> 1` with **GELU** (exact erf) between, and the **last layer zero-initialised** (weight and bias EXACTLY 0.0), so step 0 is C1 exactly.
   - C1's linear `W, b` are **constants** (not parameters, no gradient, bit-identical in every saved head: `load_head` refuses a schema-/2 head whose linear part moved).
   - **Trainable parameters: 4 x (256 x 64 + 64 + 64 + 1) = 4 x 16,513 = 66,052** (= SAFE-HEAD 8.1's figure; G-DVB re-derives it from the argv and compares it with the optimiser's count).
   - The first layer is initialised by PyTorch's default `nn.Linear` init under `torch.manual_seed(seed)`, rows in order 0, 1, 3, 5. **The seed therefore moves BOTH the data order and the hidden initialisation** (SAFE-HEAD's seed moved only the data order); R is the replicate of both.
   - DDC is still selection-inert and score-inert (SAFE-HEAD 2.1); its MLP is trained and reported only.
2. **One new flag: `--mlp-hidden 64`** (the profile pins it; default 0 reproduces SAFE-HEAD bit for bit, MEASURED, `test_safehead_next.py` t01 against the landed tree `e1eb6bf3`). It has its G-DVB entry (built value, parameter count, zero-init last layer).
3. **Head file schema `safehead.head/2`** carries the MLP arrays; the read applies them through `safehead_data.head_logits` (base + MLP, torch float32 exact-erf GELU, the trainer's own arithmetic). The untrained head returns the banked logits bit for bit.
4. **Profiles:** `refe_shmlp_s0`, `refe_shmlp_s1`, `refe_shmlp_red` (new dicts; `launch_gate.py` is NOT patched). Argv = the SAFE-HEAD argv + `--mlp-hidden 64`.
5. **Everything else is SAFE-HEAD's, unchanged:** the cache (2,272 tokens, 35 logs; 29 train / 6 held-out), the sealed index, the loss (quality-focal BCE, gamma 2.0, 0/0.5/1 safety labels, mean over the 4 rows), Adam lr 3e-4 betas (0.9, 0.999) eps 1e-8 wd 0, batch 32, **2,000 steps**, constant lr, saved steps 0/250/500/1000/2000, determinism (CPU, one thread), seeds 0 (decision) and 1 (R), RED = seed 0 + `--permute-labels-seed 20261127`, **the held-out rule (4.3: lowest held-out safety-miss rate over the saved steps, safety labels only, ties earliest)**, the read (4), the anchor table, the estimator (paired log-cluster bootstrap, 10,000 draws, seed 20260927), the measures M1-M5, the four families, the bars **P1 / P2 = 2 x max(F, 0.1158) / G / R**, and the controls CTRL-ID, CTRL-DET-NT, CTRL-FEAT(-NT), CTRL-S, CTRL-REACH, CTRL-RED, CTRL-LEAK, CTRL-SHA, CTRL-SET, CTRL-T2, CTRL-CODE.
6. Controls that CHANGE or are NEW:
   - **CTRL-REACH (MLP form).** Non-zero gradient on every trained row's MLP group (the sum over its four tensors) at every one of the 2,000 steps; **EXACTLY 0.0** on rows 2 and 4 (leaves outside the optimiser); the MLP last layer **exactly 0 at step 0** (G-DVB `mlp_last_layer_abs_sum_init == 0.0`, and the step-0 head's read is bit-exact with n/n picks); rows 2 and 4 and the linear part bit-identical to C1 in every saved head. ⚠️ Step 1's first-layer gradient is exactly zero BECAUSE the last layer is zero (by construction); the group's gradient is non-zero from step 1 through the last layer's. **Mutations, MEASURED:** `sh_reach_row_dropped` and `sh_reach_ep_in_loss` turn G-LIVE RED; `sh_mlp_off` (the flag ignored) turns G-DVB RED; `sh_mlp_nonzero_init` turns G-DVB and G-EVAL RED.
   - **CTRL-RED (unchanged text)**: RED's step 2000, on held-out, below the untrained head on (a) AUC of NC, DAC, TTC and (b) a HIGHER held-out safety-miss rate; step-0 entries identical.
   - **CTRL-GAP (new, DESCRIPTIVE, never gating).** The diagnostic `diag_veto/veto_diag.py` is re-run, zero training, on each SH-MLP head at its chosen step AND at step 2000, on the fit split and the 6 held-out logs. The reading to write beside `NOTE.md`'s table: fit-split misses (C1 69), held-out misses (C1 18), PDMS and EP of the pick, switch rate. A fit-split fall with a held-out rise (the diagnostic's pattern, expected to be larger here) is reported as **the capacity-does-not-transfer finding**, which is what the committed next arm answers.
   - **CTRL-SH-UNCHANGED (new).** SAFE-HEAD's output through the refactored trainer is bit-identical to the landed one (t01), and the three SAFE-HEAD profile fingerprints are unmoved (t04).

## 3. Cost (ESTIMATED, not measured on real data)
CPU seconds to minutes per run (4 MLPs of 256 -> 64 -> 1 over 32 x 64 rows per step; the synthetic 600-step run is seconds). No GPU. Memory: the cache's `s` is ~150 MB.

## 4. Outcomes and next arms, in precedence order (SAFE-HEAD 5 with the next arms re-pointed; Rule Zero)
1. **INVALID.** A control FAILED. Next: fix that control under a dated amendment and re-read.
2. **NULL-HELDOUT.** Seed 0's held-out rule chose step 0: SH-MLP_s0 is C1, navtest is not read for a decision. **Next arm: DATA-SCALE** (the same recipe on an expanded navtrain cache; a PI decision on data and storage; the L2-deep arm 8.3 stays blocked on storage and a GPU). This is the outcome the diagnostic predicts.
3. **ADVERSE.** CI-hi(Delta0) < 0, or G fails. Not adopted; the verdict names each adverse sub-score. **Next arm: SAFE-HEAD 8.2 SH-TOPK** with those sub-scores in its guard.
4. **ADOPT** = P1 ^ P2 ^ G ^ R. The head becomes C1's scorer read-out via an `export_full` extended to schema /2 (asserts every non-head tensor bit-identical; the MLP arrays need a model-side module, so ADOPT also needs an architecture amendment the MM lands, flagged here so it is not discovered late) and a registry row; **BELOW GOAL** printed if the point is below 94.55. **Next arm: L3 EP-GEOM inside the SH-scored safe set** (SAFE-HEAD 8.4) and the EP-HEAD card.
5. **SEPARATED-SMALL** = P1 ^ G, P2 or R fails. Not adopted. **Next arm: SH-MLP with 3 seeds**, then DATA-SCALE.
6. **NULL.** The CI of Delta0 contains 0. Not adopted. **Next arm: DATA-SCALE.**
- **INCOMPLETE** (at `confirm`): a family output is missing; a work item.

## 5. Launch gate (SAFE-HEAD 7 + this card)
Profiles `refe_shmlp_{s0,s1,red}`; checks G-HYG, G-DVB (new entries `mlp_hidden`, `select`; parameter count 66,052; last layer exactly 0), G-LIVE, G-EVAL (schema /2 loader == trainer forward <= 1e-4; step 0 bit-exact), G-CKPT (the head hash covers the MLP arrays), G-SUITE (`test_safehead.py --fast`, `test_safehead_next.py --fast`, `test_launch_gate.py --static --no-banked`), G-CLOCK NA. No training starts without a PASS token bound to the commit, the argv sha256 and the data-manifest sha256s. The deliberate-regression arms (the 11 SAFE-HEAD ones + `sh_mlp_off`, `sh_mlp_nonzero_init`) are each required to turn their check RED in the full mode of `test_safehead_next.py`; the five reachability mutations are MEASURED in its `--fast` mode (13/13 PASS, synthetic); the full-mode result is in `regression/REGRESSION_VERDICT.md` when the chain has run.

## 6. Read adaptation (mechanics only; the same note as EP-HEAD card 9.6)
`safehead_read.py` must read heads through `safehead_data.head_logits` (rows from the head's meta), use the profile names `refe_shmlp_*`, and report the trained-row pooled AUC; it is generalised and frozen under a dated CODE amendment BEFORE any head is trained. No bar, outcome rule, estimator or control threshold may change in that edit.

## 7. UNVERIFIED at freezing
- Whether 6 held-out logs (425 tokens, 18 C1 misses) can resolve any effect of this size at all; the rule resolves about +/-2 misses (diagnostic caveat). SH-MLP shares this resolution limit with SAFE-HEAD: it is the reason EP-HEAD changes the selection rule.
- Whether 66,052 parameters on 1,847 train tokens in 29 logs overfit within 250 steps (the 250 / 500 saved steps are the early-stopping grid).
- The torch GELU's float32 agreement between the trainer's batched forward and the read's per-row forward (G-EVAL tolerance 1e-4; MEASURED on synthetic data in `test_safehead_next.py`).

## 8. Integration and banking (MM)
Identical to PREREG_EPHEAD 11 (one patch carries both families): changed `safehead_data.py`, `safehead_train.py`, `safehead_gate.py`; new `test_safehead_next.py`; the 8 test pins 43 -> 49; `launch_gate.py` untouched.
