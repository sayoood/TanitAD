# PREREG L2-FRESH-STEPA (amendment to `PREREG_L2_OFFPROP.md`): does the FINAL L2 EP head re-pass L1 Step A on tokens it was not fitted on?  READ 1 = universe B, stamped "token-fresh, LOG-SHARED"

**Status: SEALED** (2026-10-19; the exact time is in `SEALED_AT.txt`).
Sealed by the L2-FRESH-STEPA agent under the Master Mind's pre-seal ruling `MM_RULING_L2_FRESH.md`, which is carried VERBATIM in section 1 as "Revision 1 (MM ruling, pre-seal)".
**No fresh-carve label or score existed when this file was sealed**: at the seal the pre-launched B chain was still waiting for the dev-box GPU lock (`chain_B.log`: `ZZGPUWAIT`), and `B_token_disjoint/` held no `bundle.npz`, `perts.npz`, `labels.npz`, `scores.npz`, `feats/` or `fresh_ep_scores.*` (recorded in `SEALED_AT.txt`).
It is the committed next arm of `D-REFE-L2-OFFPROP-1` ("L2-ADMIT -> Step A re-run on a FRESH carve, then Step B"). It supersedes the draft file `PREREG_L2_FRESH_DRAFT.md`, which stays in the package as the history of the edit.
The read command is in `READY.md`. The read refuses unless this file's sha256 is the one given on its command line and every pin in section 8 is current (`analyse_fresh.py --seal-check-only` proves it without reading any label or score).

Parents (all unchanged): `PREREG_L2_OFFPROP.md` sha256 `3592a1039693dcb5cac1df465fa4dd597ee190d80a934d89fc765d223607a983` (sealed 2026-10-19T15:43+02:00) with its recipe `lam10_lr0.001_ep15_prox0.3`
(`recipe_chosen.json` sha256 `c7f5b75b0a8f25f96c00dcc379ac583fa9c80622ce5600ad4af8387159119263`; sealed code manifest `bdb6a95dba4a6a69557a2c7ff158bf529f5ab226ac17f90271f00988f05f9939`);
`PREREG_L1_SEARCH_ADMISSION.md` sha256 `a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801`; L1 Amendment 1 sha256 `45f5c7917feef407b3ce18951e35f7abdcef5b7a9fb65e41e9cebd3d50fd72ff` (adopted `MM_ADOPTION_L1_A1.md`).

## 1. Revision 1 (MM ruling, pre-seal) -- carried verbatim, and binding on every section below

````text
{{MM_RULING}}
````

How each element of the ruling is folded into this text (the ruling above wins over any paraphrase here):

| ruling element | where it is applied |
|---|---|
| D0: B ratified as READ 1, stamped "token-fresh, LOG-SHARED", every report says it is not a fresh-log test | title; section 2 (universe); the `STAMP` string printed into `FRESH_RESULT.json` / `.md` by `analyse_fresh.py` |
| outcome rule = L1 section 4 with Amendment 1's `interval_aware` rule, VERBATIM; no point rule | section 5 (rule quoted verbatim); the literal reading is printed for the record and DECIDES NOTHING |
| expectation: O3 | section 5.3 |
| O3 next arm replaced: READ 2 on universe D | section 5.4 |
| O1 -> Step B under its own amendment; O2 -> L2-BRANCH (ratified) | section 5.4 |
| seeds: the most severe of seeds 0 and 1 (ratified) | section 5.2 (`combine_seeds`, tested) |
| A3 reported (paired, log-cluster), not gated; pre-stated consequence | section 6.2 (`a3_consequence`, tested) |
| log-OOF companion, reported-only, never gated | section 6.3 (`logoof_companion.py`; fold heads refit and EXACT reproduction proven before the seal) |
| reachability unchanged: the read refuses without the sealed amendment naming the carve and head hashes; the chain stops at `CHAIN_DONE_NO_READ` | section 8 |

## 2. The universe: the brief's carve does not exist; B is READ 1 (MEASURED)

The brief: *"drop every token whose log is one of Step A's 24 logs; draw 400 tokens ... with a NEW seed"*, from P2's held-out plan. **MEASURED (`raw/fresh_universe_probe.json`, `code/probe_fresh_universe.py`; test `test_REAL_inputs_literal_universe_is_empty`):**

| fact | value |
|---|---|
| plan tokens / logs (`heldout_tokens.json`) | 1,705 / 24; the 24 logs are exactly `sft_heldout_logs.txt` and exactly Step A's 24 logs (`stepA_logs_equal_plan_logs: true`) |
| **tokens left after dropping Step A's logs** | **0** (`raw/fresh_sample.json`: status EMPTY, written as EMPTY, never as an n=0 sample; `fresh_carve.py` exits 3) |
| why | Step A's sampler gives every log >= 1 token, so its 400 tokens touched all 24 logs. L1 Amendment A1 (adopted) already says it: *"a 'second carve of navtrain held-out logs' does not exist - the only logs outside M-PDM are these 24"* |
| the two requirements are mutually exclusive on navtrain | A1.2 (G0 PASS = fit logs disjoint from M-PDM) needs a log OUTSIDE P2's PDM-label bank; the 24 logs are exactly the complement of that bank among NAMED logs (token table: 103,288 tokens = 1,191 named logs + 251 tokens with no recorded log; M-PDM = 1,167 named logs, held out = 24; the 251 are candidate D below). So **any carve disjoint from Step A's 24 logs lies inside M-PDM and FAILS A1.2 by construction** (measured on both log-disjoint candidates below: G0 FAIL, 396 / 136 logs overlap) |

Candidates measured (all carved with Step A's own sampler `stepA.sample_stratified`, seed **20261019**, recorded; the bootstrap seed stays 20260927; no label or score read):

| id | universe | n / logs | G0 A1.2 (vs M-PDM) | overlap with Step A | G2b (bundle = banked fan) | exposure to P2 / L2 |
|---|---|---|---|---|---|---|
| U0 literal | plan minus Step A's logs | **0** / 0 | not evaluable | - | - | - |
| **B token-fresh, LOG-SHARED (READ 1, this read)** | plan tokens not in Step A's 400 (1,305 available) | 400 / 24 | **PASS** (0 tokens, 0 logs); M-SCENE 400 / 24 = declared exposure (A1.3) | 0 tokens, **24 of 24 logs** | **literal** (all 400 are in the banked table) | P2: SEEN-SCENE as Step A. **L2 head: token-fresh, LOG-SHARED** (fitted on 400 other tokens of the same 24 logs) |
| C log-disjoint, all | navtrain tokens of logs outside the 24 with a DB and 4 resolvable frames on this box: 11,024 / 396 logs; 98 % of them `served_train_sets > 0` | 400 / 396 | **FAIL** (396 logs, 393 tokens overlap) | 0 / 0 | **none**: no banked P2 fan; a bridge gate is needed | P2: IN-SAMPLE on 393 of 400 tokens (its PDM-label loss trained on these proposals); L2 head: log-fresh |
| C' log-disjoint, unserved | the 222 of those 11,024 tokens P2's PDM-label bank never served (136 logs) | 222 (SHORT, all taken) / 136 | **FAIL** (136 logs, 0 tokens overlap) | 0 / 0 | none: bridge needed | P2: sees these LOGS through other tokens, not these tokens' labels; L2: log-fresh |
| **D scene-unseen (A1.5(i)) = READ 2, prepared under its own amendment** | the 251 navtrain tokens whose log the token table does not record: never served to P2's PDM-label loss AND absent from `train_grow` (`raw/logless251_probe.json`) | <= 103 (SHORT) / 87 logs: 103 of the 251 have a nuPlan DB on this box (4 of them in the 24 held-out logs); **148 have no DB** | token level: no M-PDM and no M-SCENE token overlap; log level: unmeasured at the time of this seal (READ 2's amendment measures it) | unknown, mostly 0 | none | P2: the SCENES are unseen (neither bank holds the token); L2: log-fresh for most. **0 of the 103 have frames on this box** (a targeted fetch of 412 jpgs is READ 2's job, and needs PI approval) |
| A1.5(ii) unseen stratum | the 1,167 trainval DBs on this box that are NOT navtrain logs (+ 1,349 test DBs) | - | would PASS | - | none | UNSEEN by both; **no frames on this box**; navtest (136 logs) is the reserved single look |

**B is NOT a fresh-log test, and a pass on it is not evidence about unseen logs.** On the log axis the banked 5-fold-by-LOG OOF of `L2_RESULT.md` (A1 0.827 [0.783, 0.873] / 0.825 [0.782, 0.871] against P2 0.685) is already stricter than B. What B adds: the first read of the FINAL heads (fitted on all 400 Step A tokens) on 400 tokens none of which the head has seen, with the L1 gates literal. B is the replicate Amendment 1 C7 already accepts for the same shortage. Every report of this read carries the stamp **"token-fresh, LOG-SHARED"** and says it is not a fresh-log test.

## 3. The read (the brief's; nothing added to L1's gates)
On the sealed carve, for each of {P2 (reference), L2 final head seed 0, L2 final head seed 1}: **A1-A5 (and A5b) pooled lon + lat, plus lon and lat separately**, exactly `stepA.analyse`'s definitions (pair rules, thresholds 0.05 / 0.02 / 0.10, perturbation grid, anchors = P2's top-8 + 8 random, 96 pairs per token), with the head's EP logit in the V1_W = (5, 5, 2) aggregate for A3-A5 (`l2lib.evaluate_arm`; DAC / NC / TTC / C / DDC rows frozen = P2's, asserted bit for bit).
Estimator: paired log-cluster bootstrap over the carve's logs (24 for B), 10,000 resamples, seed 20260927, the same draws for every quantity and head. It answers *"another draw of LOGS"* only; on B the 24 logs are the SAME logs as Step A's and as the L2 head's fit, so it cannot answer "an unseen log". Not another training seed (H-ESTIM-SEED-1: the two final heads are near-replicates, EP-row cosine 0.9962, `heads/final_heads.json`). P2's scorer is deterministic (no inference-seed variance; no search is run).

## 4. The heads (fixed, banked before any fresh label or score exists)
Sealed recipe, ALL 400 Step A tokens, no folds, seeds 0 and 1: `code/fit_final_heads.py` (drives the sealed `l2lib.make_trainset` / `fit_head` / `g_live` unchanged; refuses unless prereg, recipe and four code files match their pins).

| head | sha256 | steps | EP-row L2 distance to P2 (P2 norm 0.5539) |
|---|---|---|---|
| final head s0 | `{{S0}}` | 585 | 0.4530 |
| final head s1 | `{{S1}}` | 585 | 0.4462 |

P2 head (`p2_score_head.npz`, the L2 reference): file sha256 `{{P2_FILE}}`, content sha256 of its arrays (`l2lib.sha256_arrays(weight, bias)`) `{{P2_CONTENT}}`. 9,888 training pairs (the A1 denominator), 25,600 in-fan rows.
Gates on this fit (MEASURED, `heads/final_heads.json`): **G-LIVE PASS** on the all-400 train set (finite terms; EP-row gradient non-zero, min |sum| 22.0; frozen rows exactly 0.0; the mutated-loss control reads non-zero; H0 and prox levers reach); **C-ID PASS** (zero steps = P2's six rows bit for bit; EP scores on the exported tokens equal the exported P2 logits to 2.1e-6). The five rows other than EP equal P2's bit for bit in both heads.

## 5. Outcome

### 5.1 The rule, VERBATIM (L1 prereg section 4; Amendment 1 C7 `interval_aware`); no point rule

L1 `PREREG_L1_SEARCH_ADMISSION.md` section 4 (sha256 `a76f62d8...`), the three rows that apply to a Step A read:

| outcome | rule (fixed now) |
|---|---|
| **O1 ADMIT** | A1 point >= 0.80 AND A4 point <= 0.10 AND A5 point <= 0.05 |
| **O2 NOT ADMITTED** | A1 < 0.80 OR A4 > 0.10 OR A5 > 0.05 |
| **O3 INCONCLUSIVE** | the interval of A1 or A4 contains its gate |

L1 Amendment 1 (sha256 `45f5c791...`, adopted `MM_ADOPTION_L1_A1.md`), **C7 Outcome precedence**, verbatim:

> The prereg writes O2 by point estimates and O3 by intervals, so a straddling gate can satisfy both. Proposed `interval_aware` rule: **O2** if a gate fails decisively (A1 interval wholly < 0.80, or A4 interval wholly > 0.10, or A5 point > 0.05); else **O3** if the A1 or A4 interval contains its gate; else **O1**. The literal reading (point rules first) is computed and printed beside it as `outcome_if_literal`; they differ only when an interval straddles a gate that the point estimate already fails.

C7's remaining sentence (O3's next arm: "re-run Step A once on ALL 1,705 tokens (log count stays 24)") is REPLACED by section 5.4, per the ruling.

Applied by `stepA.classify(A1, A4, A5, "interval_aware")` on the POOLED lon + lat A1, A4, A5 (MM ruling E2 of the L2 build). **The `interval_aware` outcome is the outcome.** The literal reading is printed beside it for the record, as C7 requires, and DECIDES NOTHING; the MM ruling above: "Switching to a point rule after seeing that would move a goalpost post-data."

### 5.2 Two seeds
The rule is applied to each final head. **The L2 head's verdict is the most severe of its two seeds (O2 > O3 > O1), so O1 needs both seeds at O1** (`combine_seeds`, tested). The seeds are near-replicates (cosine 0.9962); `H-ESTIM-SEED-1` stays open.

### 5.3 The stated expectation: O3 (written before any fresh label or score; it is not a bar and not a goalpost)
`interval_aware` O1 needs the A1 interval's LOWER end above 0.80. With 24 log clusters the A1 half-width is about 0.045 (Step A: 0.685 [0.614, 0.761]; banked L2 OOF seed 0: 0.827 [0.783, 0.873], seed 1: 0.825 [0.782, 0.871]), so an A1 point of 0.83 classifies **O3**, and O1 needs an A1 point of about 0.85 or more at 24 clusters. The banked OOF numbers already read O3 under this rule. **O3 is the EXPECTED outcome of this read.** It is not a surprise if it happens, and under C7 an O3 on B cannot narrow in the log dimension: the binding term is the number of LOG CLUSTERS, so the lever is more logs, not more tokens.

### 5.4 Next arms, committed now

| verdict on the L2 head | next arm |
|---|---|
| **O1** | **L1 Step B with the L2 EP head** (K=5, M=64, elites M/8, warm start at the pick, 3 search seeds; endpoint dPDMS >= +0.30 with lower end > 0 and dNC, dDAC, dTTC, dC each >= -0.25), under its own amendment (`PREREG_L2_STEPB_DRAFT.md`, built in parallel; it must first fix C8's placeholders), SEEN-SCENE, before the navtest confirm. The frozen EP-ADMISSION gate still precedes any navtest look. **Any integration of an L2 head into REFe's scorer goes through a `launch_gate.py` profile with G-HYG / G-DVB / G-LIVE entries for the new flags (PI rule 2026-09-26; MM ruling E7): a launch blocker for Step B, not for this read.** READ 2 (below) is then the log-fresh confirmation B cannot give. |
| **O2** | **L2-BRANCH**, as committed in the L2 prereg section 4 (last scorer-branch block trainable on cached ctx; own amendment before any run). RATIFIED by the MM. |
| **O3** | **READ 2 on universe D**, under its own amendment, sealed BEFORE any D label or score: the scene-unseen tokens with a local nuPlan DB (103 tokens over 87 logs), minus the 4 that fall in the 24 held-out logs; (i) a targeted pod fetch of the 412 jpgs (about 85 MB), which needs PI approval for the job, under the P6 overlap rule; (ii) a G2b BRIDGE gate: P2's fan comes from the same bundle code, which must first reproduce the banked fan bit for bit on B tokens (identity), with a RED arm; (iii) log-level M-PDM and M-SCENE exposure MEASURED for those logs; (iv) the same `interval_aware` rule, with clusters = logs. The drafted alternative (re-run on the 905 unused B tokens) is WITHDRAWN: it cannot narrow the interval (the log count stays 24). **READ 2 is prepared NOW, whatever READ 1 shows**: `PREREG_L2_FRESH_D_DRAFT.md` (a draft; it has no force until sealed). |

## 6. Reported beside it, ungated

### 6.1 The standing reports
* **P2** as the reference, on the same pairs.
* **Paired A1 difference (L2 - P2)**, log-cluster, same resamples; plus the in-fan top-8 EP concordance (SELRANK instrument) of each head and its paired difference.
* **Stamps:** SEEN-SCENE, SELECTION tier (A1.3); the G0 record of the carve (`fresh_sample*.json` -> `g0`); overlap with Step A (tokens / logs); **"token-fresh, LOG-SHARED" and "not a fresh-log test"**; one training seed per head.
* **Four families, per family never pooled:** LONGITUDINAL = A1, A2, speed-scale-wise A1 (0.85 / 0.92 / 1.08 / 1.15) and the in-fan EP concordance; LATERAL = the +-0.3 m pairs' EP (A1 lat) and DAC concordance; TACTICAL = the command-class change share (unchanged by a head retrain, reported); **STRATEGIC = PAUSED** (PI ruling 2026-09-19, ITEM 25), reported as PAUSED, never as a pass. Each with its CI and estimator. Tier: a scorer-admission diagnostic on perturbed plans labelled by the PDM instrument; not a driving claim (T0/T1 do not apply).

### 6.2 A3: reported (paired, log-cluster), NOT gated, with a pre-stated consequence
The paired A3 difference (L2 - P2) is reported for each final head. In the banked OOF read the aggregate concordance FELL for the retrained head (A3 0.736 [0.679, 0.787] vs P2 0.766, `L2_RESULT.md`) although A1 rose by 0.14; Step B climbs the AGGREGATE. **Pre-stated consequence (MM ruling; no new gate is added after the banked dip was seen):** if the paired dA3's UPPER CI end is below 0 on BOTH final-head seeds, the read records an **EP-RELIAB-class risk**, and the Step B amendment must then report the aggregate concordance of the search outputs beside dPDMS. Step B's endpoint (the true PDMS of the search output vs P2's pick, plus dNC / dDAC / dTTC / dC) measures directly what A3 stands in for. Computed by `a3_consequence` on the FINAL heads only (tested); it changes no outcome.

### 6.3 The log-OOF companion: reported-only, NEVER gated (MM ruling)
L2's 5 x 2 fold heads were NOT banked (`arms/H_s*.npz` holds the OOF scores only; searched at `D:/Projects/TanitAD-artifacts/l2_offprop/` and the landed package `.../raw/2026-10-19-l2-offprop/` at `origin/agent/arch-inf-20260803` 5944fc6: no fold-head file at either). They were therefore REFIT with the sealed recipe through the sealed code path (`logoof_companion.py`, which calls `l2lib.fold_split` + `l2lib.fit_head` exactly as `l2lib.oof_scores` does). **The ruling's condition holds, MEASURED before this seal** (`raw/logoof/logoof_heads.json`):
* IDENTITY: the OOF scores assembled from the refit fold heads equal the banked `arms/H_s0.npz` and `arms/H_s1.npz` BIT FOR BIT (`ep_in` 25,600 values and `ep_p` 38,400 values per seed, max abs difference 0.0, torch threads 2);
* RED arm 1: a refit with seed + 1 differs (max abs 0.515 / 0.735 on `ep_in`); RED arm 2: fold heads applied to the wrong folds (rolled by one) differ (max abs 1.66 / 1.73);
* the five frozen rows of every fold head equal P2's bit for bit.

In the read, each B token is ALSO scored by the fold head whose held-out fold contains its LOG (`LOGOOF_s0`, `LOGOOF_s1`); A1 / A3 (and the other quantities) are reported beside the final heads, with the paired differences against P2. They carry **no outcome** and never enter the L2 verdict or the A3 consequence. The B carve shares Step A's 24 logs, so the fold of a B token is the fold of its log in the same 24-log ranking; `compatible_logs` refuses (the companion is then omitted, and the report says so) if the carve's log set differs. The companion is a log-held-out score for the HEAD only: it does not make B a fresh-log test for P2 or for the scene.

## 7. What this licenses, and what it does not
Licenses (on O1): Step B on the same SEEN-SCENE footing, then the frozen EP-ADMISSION gate, then the shared navtest look. Does **not** license a capability claim, a navtest number, an unseen-log claim (B) or an A4 / A5 safety claim beyond the 96 perturbation shapes of the grid. An O3 or O2 licenses only the next arm in section 5.4.

## 8. Pipeline, gates and the pins the read refuses without

### 8.1 Pipeline and gates for the carve (the L1 code, byte-identical; `code/fresh_chain.cmd` runs it as one detached chain and STOPS before the read)
Step 3 (one GPU forward, dev-box lock, compute-type guard) -> 4 perturb -> 5 label (G1 first) -> 6 P2 scores on CPU -> 6b export of the EP head's INPUT token for every (token, anchor, perturbation) and every in-fan proposal (L2's `export_feats.py`, 2 workers) -> merge -> 6c EP scores of P2 and both final heads (`score_fresh.py`). Any failure = no read:
* **G0:** A1.1-A1.4 as adopted: PASS on M-PDM (B: 0 / 0 measured), M-SCENE overlap DECLARED (B: 400 tokens / 24 logs). Control (deliberate overlap) reads FAIL.
* **G1 (label identity), G2a, G2c, G3:** as in L1 (G1 on 20 tokens, <= 1e-6, shifted-trajectory control must change the label; G2c premise must read true).
* **G2b (hard):** the bundle forward reproduces the banked fan, logits and pick **bit for bit on every token** (`ZZBUNDLE_OK` + `STEP3_RC` 0). For B this is the literal L1 gate (all 400 tokens have banked P2 rows). Cross-device (C9, reported): CPU fp32 pick agreement vs the GPU bundle; if < 0.99 the pre-committed contingency re-scores on the GPU before step 6b (a new decision, not automated).
* **C-FEAT (L2 sec 2, on the fresh run):** P2's EP head on the exported tokens reproduces step 6's EP scores, max abs diff <= 1e-4 on >= 99 % of rows, in-fan and perturbed, and the shards come from THIS bundle / perts (`score_fresh.py`; RED-tested). Fail = exit 3, no score file.

### 8.2 The seal guard on the read (`analyse_fresh.py`; RED-tested, including on THIS file)
The read refuses (exit 5) unless (1) this file's sha256 (CRLF folded to LF) equals `--prereg-sha256`, (2) this file carries the marker "Status: SEALED" and not the draft marker, (3) every pin below is named in this text, (4) each code file's current LF sha256 equals the one named below on a line that also names the file, (5) the run's final heads and P2-head arrays are the ones named here, and C-FEAT passed on the run. Editing any pinned code file after this seal makes the read a refusal; a change goes into a sealed Amendment 2 before the read.

### 8.3 Pins
* carve token-list sha256 (B, seed 20261019, 400 tokens, `raw/fresh_sample_token_disjoint.json`): `{{CARVE_SHA}}`; Step A's: `{{STEPA_SHA}}`
* final head s0 sha256: `{{S0}}`
* final head s1 sha256: `{{S1}}`
* P2 head file sha256 (L2 reference): `{{P2_FILE}}`
* P2 head content sha256 (arrays): `{{P2_CONTENT}}`; P2 checkpoint sha256 `{{P2_CKPT}}`
* log-OOF fold-heads file sha256 (`logoof_heads.npz`): `{{LOGOOF_NPZ}}`
* log-OOF fold-heads manifest sha256 (`logoof_heads.json`): `{{LOGOOF_JSON}}`
* code (LF sha256; the files the read runs):
{{CODE_PIN_LINES}}
