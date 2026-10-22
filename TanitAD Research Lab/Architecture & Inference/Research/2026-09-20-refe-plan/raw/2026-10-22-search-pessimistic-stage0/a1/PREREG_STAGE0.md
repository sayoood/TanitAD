# PRE-REGISTRATION (SEALED): SEARCH-PESSIMISTIC Stage 0, the head-level session-exposure gap gate

**Status: SEALED by the Master Mind, 2026-10-22 14:34 Berlin, before any real read.** The text below is the build agent's draft, unchanged except this line, the title and the seal block. Written 2026-10-22 by the build agent (the Master Mind seals). No real token has been scored: the real inputs were opened only for shapes, counts, sha256 and the integrity checks in §1.
**Evidence class of everything this read will write:** MEASURED (ours), **SELECTION tier, seen-scene, label-only**. It is not a T1 driving number, and nothing here is a search result.
**Code binding:** `constants_sha256: c71c7dd71c97bea7179322e2af7f5f59bf528f4dd16a6531f8fdf2c67ffd811c` (digest of every constant in §2–§7, from `python code/stage0.py constants`). `stage0.py` refuses to run unless the sealed prereg quotes this digest and its own sha256 equals the one given on the command line.

**Seal block:** sealed at 2026-10-22 14:34 Berlin by the MM. This file's sha256 is in `PREREG_STAGE0.sha256` beside it. Code manifest `code/MANIFEST.sha256` sha256 `e4fbaa136f9b4886b8daa7404550683605109700cbfc17fa8feee46fe3273013` (verified at seal: **7 of 8 entries OK**, including `stage0.py` and `test_stage0.py`. `geometry_real.json` does NOT match its manifest line: now sha256 `9cc51ab8…`, file mtime 12:56 before the manifest's 14:33. It feeds only `synth.py`, the synthetic tests and the power simulation, never `stage0.py run`, so it does not touch the real read. Recorded, not corrected.). Judgement calls §11.1–§11.7 are **CONFIRMED as drafted**: Δ = (0.01, 0.02, 0.35); R0 gating; GAP through any one certified measure; NULL-UNDERPOWERED does not license the NULL sentence; C-II-b on any planted κ; C-III gates on M1; C-II-a as non-reversal on real tokens. GAC row: `D-REFE-SESSION-EXPOSURE-1` is updated with the outcome in the same turn as the read.

## 0. Question, and what it can and cannot decide

**Question.** Does P2's scorer head score tokens from a drive **session** it never trained on worse than tokens from sessions it did train on?

This is the gate in front of SEARCH-PESSIMISTIC Stage 1 (FEASIBILITY.md §7). `D-REFE-SESSION-EXPOSURE-1` established that every pre-navtest REFe read was on sessions P2 trained on. Stage 0 asks whether the **head** (the five linear rows the search composite uses, on the cached pre-head features) shows any session-exposure gap that a pessimism rule could then recover.

- **In scope:** rows NC, DAC, EP, TTC, C of `score_head = Linear(256,6)` (row DDC is unused by the composite and stays frozen), refit on the cached `score_head` inputs of the 1,705 held-out tokens.
- **Out of scope:** the trunk and the decoder (`score_q_mlp`, `score_dec`) have seen all 21 sessions through the 16,633 sibling M-PDM tokens and the un-detached context (FEASIBILITY.md §6.2). That exposure is **equal across folds, cancels in IN − OOF, and caps the test at head-level exposure**. A decoder-level variant (about 1.5 GPU-h) is not part of this read.
- **No search is run.** No K1–K4 knob is evaluated.

## 1. Inputs (fixed; MEASURED sha256 2026-10-22, identical to the banked manifests)

Root `D:/Projects/TanitAD-artifacts` (override with `TANITAD_ARTIFACTS` if the drive letter flips). `stage0.py` aborts on any sha256 or size mismatch.

| key | path under root | bytes | sha256 |
|---|---|---|---|
| table | `p2_heldout_table_full_2026-10-18/table_p2_heldout.npz` | 13,304,426 | `8eacc19b35b12aee9b0ab047716c901399a711d22ddd3c25cd890ecf2f02004b` |
| prehead | `p2_heldout_table_full_2026-10-18/prehead_features_p2_heldout.npz` | 51,442,644 | `e37b9c554bf2c2546e958ff763dca23b71e0b77036a271d6a003c8ba2672f522` |
| feats | `l2_offprop/feats.npz` | 67,074,052 | `49f6a034464580ef6e14bae9dc397b898a360bc9d7051d6d0676acdba141e616` |
| head | `l2_offprop/p2_score_head.npz` | 7,142 | `4b64c3feb89ca571228ed8d4cfa795f3ec5143e71d06545e287e0b9b13ff55d0` |
| bundle | `l1_stepA_run/bundle/bundle.npz` | 8,309,154 | `77e8584d8a02ee7a93b9239e1e33a16edade75689f36e1f32959960d0e2eb5c4` |
| perts | `l1_stepA_run/perts.npz` | 18,972,106 | `40ecafc88d401463767d74af5d41809d73b36009083e19d62e197f4bfd6bb8d5` |
| labels | `l1_stepA_run/labels.npz` | 2,151,734 | `7c986dcc736a6dcaf49dfaccd33f3ded9e9c4f2ec49b88f28bb728a71ebf9b27` |

Shapes (MEASURED): features `[1705, 64, 256]`; Step A perturbation features `[400, 96, 256]`; 1,705 tokens, 24 logs, **21 sessions**. The loader also aborts unless: token/log order agrees across table and prehead; all 400 Step A tokens are in the table; the table's fan logits equal Step A's bundle logits; the table's and Step A's anchor labels agree; the provenance hashes stored inside `feats`, `perts`, `labels` equal the sha256 of the files used; all labels and features are finite; `pdms` is on `[0,1]`. Step A rows use the float32 Step A features (fp16-table drift 0.0035); every other token uses the fp16 table features.

What the build agent touched on the real files: sha256, shapes, token/log/session counts, the fold and half assignment below, the equality and finiteness checks above, and the `pdms` range. **No IN/OOF statistic, no concordance, no regret and no gap was computed on a real token.**

## 2. Design

- **Session** = log name minus its last two `_` parts (21 sessions). **Folds:** 5, greedy-balanced by tokens (sessions by descending token count, name as tie-break, each to the fold with the fewest tokens so far). Realised: fold tokens **338 / 340 / 344 / 337 / 346**, sessions per fold 3 / 3 / 5 / 5 / 5.
- **Halves:** inside every LOG, tokens are ordered by `sha256("20261022|token")`; even ranks are the **FIT half**, odd ranks the **SEARCH half**. Realised: **860 fit, 845 search** tokens; search tokens per fold 168 / 169 / 170 / 167 / 171, of which Step A tokens 35 / 43 / 42 / 40 / 41 (**201** total). Fit-half Step A tokens per fold 43 / 37 / 38 / 40 / 41.
- **Models.** Model *h* (h = 0..4) trains on the **fit halves of every fold except h** (about 690 tokens, about 44,000 fan points, about 160 Step A tokens of 96 pairs each). Nothing else: no search-half token is in any training set.
- **Roles on the search-half tokens of fold g.** OOF = model g (never saw g's sessions). IN = model (g+1) mod 5 (saw g's logs through their fit halves, never g's search tokens). Both trained on four folds' fit halves, so on equal amounts of data. Swap-null models: (g+2), (g+3), (g+4) mod 5, also IN.
- **Correction to the memo's counts** (for the MM's register fix): the memo's "about 850 search-half tokens of fold g" is the **pooled** search total (845); per fold it is about 169. The memo's "about 1,364 training tokens" is four **full** folds; each model trains on the fit halves of four folds, about 690.

## 3. Heads and recipe

Rows 0–4 (NC, DAC, EP, TTC, C) are initialised at P2's rows; row 5 (DDC) is frozen, bit-identical to P2's, and its gradient is asserted exactly 0.0 at every step. Adam (lr 1e-3, betas 0.9/0.999, eps 1e-8, no weight decay), batch 256 fan points and 256 Step A pairs per step, **600 steps**, per row:

`loss_r = lam_bce · BCE(σ(z_r), label_r)  [64 in-fan proposals of the fit tokens, soft targets]  +  pairwise logistic over Step A (anchor, perturbation) pairs of fit-half Step A tokens whose TRUE sub-score r differs by ≥ 0.05  +  prox · ‖row_r − P2 row_r‖²`, lam_bce = 10.

| recipe | prox | role |
|---|---|---|
| **R0** | 0.0 | **PRIMARY, gating.** P2's own head was trained by BCE on the six sub-scores with no pull-back; R0 keeps that and adds L2-OFFPROP's pairwise term. No proximal pull, so the head can absorb a session. |
| **R1** | 0.3 | **descriptive only, never gating.** L2-OFFPROP's chosen cell. |

Provenance of the budget (INHERITED, `l2/out/sweep_semi.json`, EP row only, REAL tokens with SYNTHETIC labels): L2's cell `lam10_lr0.001_ep15_prox0` (465 steps) read C-REACH 0.999 and in-fan concordance −0.012 / −0.006 against its BCE-only control; the chosen `prox0.3` cell read −0.004. **The recipe for the other four rows (NC, DAC, TTC, C) is UNVERIFIED.** Training is numpy with analytic gradients, pinned to finite differences and to `torch.optim.Adam` by `test_stage0.py`.

## 4. Measures (each on the SEARCH-half tokens; quality direction, so a positive gap means OOF is worse)

Composite = `σ(NC)·σ(DAC)·(5σ(EP)+5σ(TTC)+2σ(C))/12` (equal to `stepA.aggregate`, pinned by a test).

| | definition | gap | unit | Δ (smallest gap of interest) |
|---|---|---|---|---|
| **M1** | in-fan concordance of the composite with true PDMS: all ordered pairs of the 64 proposals whose true PDMS differs by ≥ 0.05, ties 0.5 | IN − OOF | concordance | **0.01** |
| **M2** | A3 aggregate concordance on Step A (anchor, perturbation) pairs (`stepA.concordance_counts` semantics, threshold 0.05), Step A search tokens only (201) | IN − OOF | concordance | **0.02** |
| **M3** | regret of the argmax of the composite over the 64-fan, 100 × (best true PDMS − PDMS at the pick) | regret(OOF) − regret(IN) | PDMS points | **0.35** |

Δ is a **judgement**, flagged for the MM (§11). M3's 0.35 is half of the C1 − C2 contrast, 0.713 points (+0.558 against −0.155, `D-REFE-SESSION-EXPOSURE-1`). M1 and M2 have no derived mapping to search points; the simulation in §8 shows a M1 gap of 0.010 goes with a M3 gap of roughly 0.8–1.4 points on the synthetic generator, which is UNVERIFIED for REFe.

## 5. Estimator and noise floors

- **Session-cluster bootstrap over the 21 sessions**, B = **10,000**, seed **20261022**, the same resample draws for every quantity. Each gap is a ratio of per-session sums. It answers *"would another draw of SESSIONS say this?"* and nothing else: not another training run, not another split (H-ESTIM-SEED-1).
- **Family-wise level:** one-sided α = 0.025, Bonferroni over the three measures, so each measure uses the **0.8333 %** bootstrap quantile (`lo`) and the 99.1667 % quantile (`hi`).
- **Model-swap null.** The gap between two IN models on the same tokens, for the chained pairs (g+1 vs g+2), (g+2 vs g+3), (g+3 vs g+4). **ν_m = the largest absolute value of the three point estimates.**
- **Training-seed replicate.** The identical pipeline with minibatch seed 1 (primary is seed 0). The seed floor is the lever of H-ESTIM-SEED-1.
- **Leave-one-session-out.** `loso_m` = the smallest of the 21 gaps recomputed with one session deleted (no refit).

## 6. Controls (every one has a deliberate-regression arm in `test_stage0.py`, and 38 of 38 textual mutants of `stage0.py` are killed by the suite: `raw/MUTATION_CENSUS.json`)

| control | known value, pass rule (numbers fixed here) |
|---|---|
| **C-STRUCT** | The exclusion line `train_folds = [f for f in range(N_FOLDS) if f != h]` and a structural check that **fails closed**: no token of fold h, no search-half token, no session of fold h in model h's training set; and the tokens judged as "search" are search-half tokens. Deleting the line makes the pipeline raise `C-LEAK`. |
| **C-I-eval** | Labels shuffled within token ⇒ chance. R = **100** shuffles (seeds 9001–9100) of the labels the trained R0 models are judged against. A token without Step A pairs has its 64 labels permuted; a **Step A token has its 64 fan labels and 96 perturbation labels pooled and re-dealt** (a perturbation is usually worse than its anchor, so shuffling the two sets separately would not put A3 at 0.5). Six statistics: M1, M2 concordance − 0.5; M1, M2, M3 gaps; M3 regret − 100(max − mean). Each residual has mean exactly 0 under the shuffle. **Pass iff |z| ≤ 4 for all six and, for M1 and M2, 4·SE ≤ Δ** (the control could have seen a bias that matters). If M3's 4·SE > 0.35, M3 is **uncertified**: reported, but it cannot trigger GAP and cannot make a NULL POWERED. It never voids the read by itself. |
| **C-I-trained** | The same, with the heads **trained** on shuffled labels (10 worlds, seeds 9101–9110, pooled). Pass iff |z| ≤ 4 on all six and M1, M2 informative. |
| **C-II-a** | A model scored on tokens it trained on is not WORSE than the model that did not: SEEN = model (g+1) mod 5 and OOF = model g, on the fit-half tokens of fold g. **Real tokens: pass iff the token-level paired z of the M1 difference exceeds −4 (non-reversal).** The strict rule *SEEN beats OOF* is gated where a gap exists by construction, in the planted world (C-II-b). Reason: the token-level overfit of a 257-parameter row on about 44,000 points is of the order of the swap-null floor (synthetic null worlds: −0.001 to +0.0125; 1 of 36 read ≤ 0), so a strict real-token rule would void about one valid read in thirty. This deviates from the literal brief and is flagged in §11. |
| **C-II-b** | A PLANTED session effect on the REAL features with SYNTHETIC labels (no real label read). True logit of row r in session s = P2's own logit + κ·std(P2 logit_r)·(u_{s,r}·x), κ ∈ **{0.5, 1, 2, 4}**, seed 7771. **Pass iff for at least one κ:** M1 gap ≥ **0.02**, its `lo` > the planted run's ν_M1, **SEEN − OOF > 0 strictly** in that planted world (the model scored on its own fit tokens beats OOF), and C-III collapses in that planted world. The table of which κ are read calibrates the instrument. |
| **C-III-real** | Delete the exclusion line (a textual mutation of the real source, structural check stubbed) and rerun: the M1 gap must collapse to the null. **Pass iff \|gap_mut\| ≤ max(ν, ν_mut, 2·SE_mut, 0.2·\|gap\|)** (ν from the unmutated run, ν_mut and SE_mut from the mutated one). M2 and M3 are reported with the same bound, descriptively. |

## 7. Decision rule (a pure function of the numbers above: `decide()`)

For each measure m, with G_m the primary gap (R0, seed 0), G′_m the seed-1 gap:

`GAP_m = [lo_m > ν_m]  ∧  [G_m ≥ Δ_m]  ∧  [G′_m > 0 and G′_m ≥ 0.5·G_m]  ∧  [loso_m > 0]  ∧  [m certified]`

1. **INVALID** if any control in §6 fails (or any ABORT check in §1 fires, which stops the run before an outcome). *The gap is then not read.*
2. **GAP** if INVALID is not met and some GAP_m is true.
3. **NULL** otherwise, labelled **POWERED** iff `hi_1 < Δ_1` and `hi_3 < Δ_3` and M3 is certified, else **UNDERPOWERED**.

**Both outcomes are committed in advance:**

| outcome | what happens next |
|---|---|
| **GAP** | **Stage 1** (FEASIBILITY.md §7): search plus knobs K1–K4 (FEASIBILITY.md §5) with the four arms P2HEAD on Step A's 400, OOF, IN and a seed replicate; about 30 CPU-h, 0 GPU-h, about 8 h wall in parallel, K1–K4 offline after about 2 CPU-h of labels. A GAP means a head-level gap exists **under this exposure**, which shares segments (§9); Stage 1 gets its own prereg. |
| **NULL, POWERED** | *"Pessimism cannot be tested on P2; the lever is D-SESSION-RESERVE (PI)."* No session-unseen stratum exists outside navtest, and P6 trains on the same 103,037-token bank. |
| **NULL, UNDERPOWERED** | No gap was detected, but a gap of size Δ is **not excluded**. The sentence above is **not licensed**. The MM decides between the decoder-level variant (about 1.5 GPU-h) and D-SESSION-RESERVE. Stage 1 does not run. |
| **INVALID** | Which control failed, and the fix: |

| failed control | meaning | fix (to be re-sealed; the failed run's gap is never read) |
|---|---|---|
| C-STRUCT | a model saw a session it is scored on | code defect: restore the exclusion; nothing is read |
| C-I-eval / C-I-trained, a z-test | the evaluator or the training path carries label information into the scores, or the shuffle is not exchangeable | audit `measure` / `shuffled_labels` / the training path against the failing statistic |
| C-I-eval / C-I-trained, informativeness | the control is too noisy to have seen a bias of size Δ | more shuffles (the seeds list is a constant: amend, re-digest, re-seal) |
| C-II-a | SEEN is significantly WORSE than OOF on tokens SEEN trained on: a fit / search mix-up or a leak, not a weak head | audit the halves, `build_cache` and the fold indices; nothing is read |
| C-II-b | a planted session effect up to κ = 4 is not read, or SEEN does not beat OOF where a gap exists: the recipe is too constrained to absorb a session, or a linear head cannot see token-level exposure at all | raise steps / lr, chosen on SYNTHETIC or planted worlds only; failing that, the decoder-level variant (4.30 M parameters, about 1.5 GPU-h) |
| C-III-real | the gap is not produced by the exclusion line: it comes from another path | audit the halves and the folds for a path that carries session identity |

## 8. Power: the smallest gap detectable at 80 %

**Simulation (ESTIMATED, `raw/power_sim.json`).** 8 planted strengths × 24 worlds, the REAL geometry (21 sessions, 1,705 tokens, 400 Step A tokens, D = 256), heterogeneous sessions, labels from the same generator as C-II-b. The synthetic regret noise, session heterogeneity and effect are the generator's, **not REFe's**.

| planted κ | 0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|---|---|
| realised M1 gap (mean) | +0.0010 | +0.0018 | +0.0027 | +0.0044 | +0.0097 | +0.0146 | +0.0181 | +0.0269 |
| **P(GAP through M1)** | **0.00** | 0.04 | 0.00 | 0.08 | 0.29 | 0.67 | **1.00** | 1.00 |
| realised M2 gap | +0.0015 | +0.0014 | +0.0028 | +0.0044 | +0.0078 | +0.0097 | +0.0108 | +0.0159 |
| P(GAP through M2) | 0 | 0 | 0 | 0 | 0 | 0.04 | 0.08 | 0.08 |
| realised M3 gap (points) | +0.14 | +0.36 | +0.38 | +0.45 | +0.81 | +1.41 | +2.74 | +4.56 |
| P(GAP through M3) | 0 | 0 | 0 | 0 | 0 | 0 | 0.25 | 0.42 |

- **False positives:** 0 of 24 GAP verdicts at κ = 0; 1 of 72 over κ ≤ 0.2 (a world whose realised gap reached Δ). SEEN − OOF on synthetic null worlds ranged from −0.001 to +0.0125 (1 of 36 worlds read ≤ 0), which is why control (ii) is gated strictly only in the planted world.
- **Smallest M1 gap detected with ≥ 80 % by the full rule: 0.016** (linear interpolation on the κ grid). Mean achieved SE of the M1 gap 0.0041–0.0050, so `MDE80 = (z_0.99167 + z_0.80)·SE = 3.236·SE` = 0.013–0.016, in agreement.
- **M2:** SE 0.0042–0.0053, so the statistical MDE80 is 0.014–0.017, **below Δ_2 = 0.02, so Δ_2 binds**: the smallest M2 gap the rule can call is 0.02, and it was not reached in the simulated range. M2 rests on 201 tokens.
- **M3:** simulated SE 0.47–1.25 points, not transferable. **INHERITED real scale:** the banked paired half-widths of FEASIBILITY.md §4 (0.10 and 0.37 points) imply SE 0.05–0.19, so **MDE80 for M3 is 0.17–0.61 points**, against Δ_3 = 0.35. The achieved SE is printed by the read, together with `achieved_mde80` for each measure.
- **What a NULL therefore excludes:** with the simulated SEs, a head-level M1 gap of about 0.016 or more (80 % power). Whether that is a regret gap of 0.35 points or more is **not established** by this read: it depends on M3's real SE, hence the POWERED / UNDERPOWERED label.

## 9. Interpretation limits (declared before the read)

1. **The IN arm is stronger exposure than P2's.** IN shares the same *segments* (interleaved tokens of one log) with the search token; P2's real exposure was *segment-disjoint, session-shared* (24 held-out segments against 16,633 M-PDM sibling tokens in the same 21 sessions). The table carries no time order inside a segment, so adjacency cannot be guarded. **A NULL is therefore conservative evidence** (even the stronger exposure shows no gap). **A GAP is an upper bound on the session effect:** it cannot separate session familiarity from within-segment adjacency. Three of the 21 sessions have two segments, too few to separate them.
2. **Linear head only.** Capacity 257 parameters per row on frozen features. A NULL here says nothing about the 4.30 M-parameter decoder.
3. **Forking paths.** Step A, Step B, L2, L2-FRESH, C1 and the dose read used these 24 logs. Stage 0 is a further read of the same logs; cross-fitting protects nothing about the choice of question.
4. **21 sessions, two of them 28 % of the tokens** (236 and 241). The cluster bootstrap with 21 clusters under-covers in its tails; this is why GAP needs `lo` above the swap-null, a seed replicate and leave-one-session-out stability on top of the interval.
5. **One split.** The fit/search split is one hash draw (seed 20261022). A split replicate is not gating.
6. **Not T1.** Concordance and regret over a fixed fan are label-only SELECTION-tier quantities.

## 10. What the read writes

`raw/RESULT_STAGE0.json`: per arm (R0 seed 0, R0 seed 1, R1 descriptive, R0 exclusion-deleted, planted κ ∈ {0.5,1,2,4} with and without the exclusion) the three gaps with point, SE, `lo`/`hi` (Bonferroni), 95 % interval and leave-one-session-out range; the swap-null gaps; SEEN − OOF; absolute levels of OOF and IN; the fit audit of all 25 fold models; all six controls with every statistic; the decision with each conjunct; P2's own levels on the same tokens for context (never a gap). The register update for `D-REFE-SESSION-EXPOSURE-1` is the MM's.

## 11. Judgement calls the MM should confirm before sealing

1. **Δ = (0.01, 0.02, 0.35)** (§4). M3's is derived; M1 and M2 are not.
2. **R0 = prox 0 as the gating recipe** (§3). Using prox 0.3 as primary would make a NULL cheaper to obtain.
3. **GAP through any one certified measure** (Bonferroni over three), rather than requiring two.
4. **NULL-UNDERPOWERED does not license** "pessimism cannot be tested on P2".
5. **C-II-b passes on any planted κ** (a larger κ is not always easier: labels saturate and the swap-null grows).
6. **C-III gates on M1 only**; M2 and M3 are descriptive because their run-to-run noise alone exceeds a tight bound about once in ten reads on synthetic null worlds.
7. **Control (ii) is gated strictly in the PLANTED world and as non-reversal on the real tokens** (§6, C-II-a). The literal brief reads *an IN model scored on its own fit tokens must beat OOF*; on the real heads the true effect may be within the noise floor. The MM can restore the strict real-token rule by changing one comparison in `control_seen`, then re-digesting.
