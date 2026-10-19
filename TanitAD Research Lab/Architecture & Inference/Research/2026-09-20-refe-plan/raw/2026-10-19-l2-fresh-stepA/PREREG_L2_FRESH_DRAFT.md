# PREREG L2-FRESH-STEPA (amendment to `PREREG_L2_OFFPROP.md`): does the FINAL L2 EP head re-pass L1 Step A on a carve it was not fitted on?

**Status: DRAFT** (written 2026-10-19 by the L2-FRESH-STEPA agent for the Master Mind; no A-quantity exists on any fresh carve: steps 0-6 produce scores, not outcomes, and `analyse_fresh.py` refuses to run on this file while this line is present).
It is the committed next arm of `D-REFE-L2-OFFPROP-1` ("L2-ADMIT -> Step A re-run on a FRESH carve, then Step B"). Sealing = the Master Mind changes the status line above, fills section 8, and takes the sha256 (LF bytes). The read command is in `READY.md`.

Parents (all unchanged): `PREREG_L2_OFFPROP.md` sha256 `3592a1039693dcb5cac1df465fa4dd597ee190d80a934d89fc765d223607a983` (sealed 2026-10-19T15:43+02:00) with its recipe `lam10_lr0.001_ep15_prox0.3`
(`recipe_chosen.json` sha256 `c7f5b75b0a8f25f96c00dcc379ac583fa9c80622ce5600ad4af8387159119263`; sealed code manifest `bdb6a95dba4a6a69557a2c7ff158bf529f5ab226ac17f90271f00988f05f9939`);
`PREREG_L1_SEARCH_ADMISSION.md` sha256 `a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801`; L1 Amendment 1 sha256 `45f5c7917feef407b3ce18951e35f7abdcef5b7a9fb65e41e9cebd3d50fd72ff` (adopted `MM_ADOPTION_L1_A1.md`).

## 0. READ THIS FIRST: the brief's carve does not exist, and no navtrain carve can satisfy both of its requirements

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
| **B token-disjoint** | plan tokens not in Step A's 400 (1,305 available) | 400 / 24 | **PASS** (0 tokens, 0 logs); M-SCENE 400 / 24 = declared exposure (A1.3) | 0 tokens, **24 of 24 logs** | **literal** (all 400 are in the banked table) | P2: SEEN-SCENE as Step A. **L2 head: token-fresh, LOG-SHARED** (fitted on 400 other tokens of the same 24 logs) |
| C log-disjoint, all | navtrain tokens of logs outside the 24 with a DB and 4 resolvable frames on this box: 11,024 / 396 logs; 98 % of them `served_train_sets > 0` | 400 / 396 | **FAIL** (396 logs, 393 tokens overlap) | 0 / 0 | **none**: no banked P2 fan; a bridge gate is needed | P2: IN-SAMPLE on 393 of 400 tokens (its PDM-label loss trained on these proposals); L2 head: log-fresh |
| C' log-disjoint, unserved | the 222 of those 11,024 tokens P2's PDM-label bank never served (136 logs) | 222 (SHORT, all taken) / 136 | **FAIL** (136 logs, 0 tokens overlap) | 0 / 0 | none: bridge needed | P2: sees these LOGS through other tokens, not these tokens' labels; L2: log-fresh |
| D scene-unseen (A1.5(i)) | the 251 navtrain tokens whose log the token table does not record: never served to P2's PDM-label loss AND absent from `train_grow` (`raw/logless251_probe.json`) | <= 103 (SHORT) / 87 logs: 103 of the 251 have a nuPlan DB on this box (4 of them in the 24 held-out logs); **148 have no DB** | token level: no M-PDM and no M-SCENE token overlap; log level: unmeasured for the 87 logs | unknown, mostly 0 | none | P2: the SCENES are unseen (neither bank holds the token); L2: log-fresh for most. **0 of the 103 have frames on this box**; `raw/logless251_probe_needed_frames.json` lists the 412 exact jpgs (4 cameras) for a targeted fetch, as `p2heldout/plan/needed_frames.json` did |
| A1.5(ii) unseen stratum | the 1,167 trainval DBs on this box that are NOT navtrain logs (+ 1,349 test DBs) | - | would PASS | - | none | UNSEEN by both; **no frames on this box** (probed the three frame roots only: proxy zips and M5 jpgs are navtrain logs, the held-out root is the 24 logs; absence at THREE locations, not a settled absence) = a data job; navtest (136 logs) is the reserved single look |

**Decision for the Master Mind (D0): which universe is this read run on?** The draft is written for **B** because it is the only candidate that can run with every L1 gate literal and no new plumbing, and it is the remedy Amendment 1 C7 already adopted for the same shortage ("re-run on ALL 1,705 tokens (log count stays 24)"). **B is NOT a fresh-log test, and a pass on it is not evidence about unseen logs**: the 5-fold-by-LOG OOF of `L2_RESULT.md` was already stricter than B on that axis (A1 0.827 [0.783, 0.873] / 0.825 [0.782, 0.871] against P2 0.685). What B adds: the first read of the FINAL heads (fitted on all 400 Step A tokens), on 400 tokens none of which the head has seen, with the L1 gates literal. For log-freshness the options are C' (a build, see `READY.md` section E) or A1.5 / navtest (the Step B confirm). Each is a separate amendment; none is licensed by this one.
A chain for B was **pre-launched** (waiting for the GPU lock; kill by PID in `HANDOFF.md`). It computes scores only. If the MM picks another universe, B's bundle is still a valid replicate and costs one ~11 min GPU slot.

## 1. The read (the brief's; nothing added to L1's gates)
On the sealed carve, for each of {P2 (reference), L2 final head seed 0, L2 final head seed 1}: **A1-A5 (and A5b) pooled lon + lat, plus lon and lat separately**, exactly `stepA.analyse`'s definitions (pair rules, thresholds 0.05 / 0.02 / 0.10, perturbation grid, anchors = P2's top-8 + 8 random, 96 pairs per token), with the head's EP logit in the V1_W = (5, 5, 2) aggregate for A3-A5 (`l2lib.evaluate_arm`; DAC / NC / TTC / C / DDC rows frozen = P2's, asserted bit for bit).
Estimator: paired log-cluster bootstrap over the carve's logs (24 for B), 10,000 resamples, seed 20260927, the same draws for every quantity and head. It answers *"another draw of LOGS"* only; on B the 24 logs are the SAME logs as Step A's and as the L2 head's fit, so it cannot answer "an unseen log". Not another training seed (H-ESTIM-SEED-1: the two final heads are near-replicates, EP-row cosine 0.9962, `heads/final_heads.json`). P2's scorer is deterministic (no inference-seed variance; no search is run).

## 2. The heads (fixed, banked before any fresh label or score exists)
Sealed recipe, ALL 400 Step A tokens, no folds, seeds 0 and 1: `code/fit_final_heads.py` (drives the sealed `l2lib.make_trainset` / `fit_head` / `g_live` unchanged; refuses unless prereg, recipe and four code files match their pins).
| head | sha256 | steps | EP-row L2 distance to P2 (P2 norm 0.5539) |
|---|---|---|---|
| final head s0 | `bded124e1322b56d9bfd749f73b22c86430ef41f6829856927e5a08831320df4` | 585 | 0.4530 |
| final head s1 | `86f0b20f6ab0d2c46af6b3ae9874b73cc9d94a8065cf7fc0344acbeadcfd4654` | 585 | 0.4462 |
P2 head (`p2_score_head.npz`) sha256 `4b64c3feb89ca571228ed8d4cfa795f3ec5143e71d06545e287e0b9b13ff55d0`. 9,888 training pairs (the A1 denominator), 25,600 in-fan rows.
Gates on this fit (MEASURED, `heads/final_heads.json`): **G-LIVE PASS** on the all-400 train set (finite terms; EP-row gradient non-zero, min |sum| 22.0; frozen rows exactly 0.0; the mutated-loss control reads non-zero; H0 and prox levers reach); **C-ID PASS** (zero steps = P2's six rows bit for bit; EP scores on the exported tokens equal the exported P2 logits to 2.1e-6). The five rows other than EP equal P2's bit for bit in both heads.

## 3. Pipeline and gates for the carve (the L1 code, byte-identical; `code/fresh_chain.cmd` runs it as one detached chain and STOPS before the read)
Step 3 (one GPU forward, dev-box lock, compute-type guard) -> 4 perturb -> 5 label (G1 first) -> 6 P2 scores on CPU -> 6b export of the EP head's INPUT token for every (token, anchor, perturbation) and every in-fan proposal (L2's `export_feats.py`, 2 workers) -> merge -> 6c EP scores of P2 and both final heads (`score_fresh.py`).
Gates, all as in L1 / L2, any failure = no read:
* **G0:** A1.1-A1.4 as adopted: PASS on M-PDM (B: 0 / 0 measured), M-SCENE overlap DECLARED (B: 400 tokens / 24 logs). Control (deliberate overlap) reads FAIL.
* **G1 (label identity), G2a, G2c, G3:** as in L1 (G1 on 20 tokens, <= 1e-6, shifted-trajectory control must change the label; G2c premise must read true).
* **G2b (hard):** the bundle forward reproduces the banked fan, logits and pick **bit for bit on every token** (`ZZBUNDLE_OK` + `STEP3_RC` 0). **For B this is the literal L1 gate** (all 400 tokens have banked P2 rows). For C / C' no banked fan exists: the gate would be (i) a bridge of >= 20 Step A tokens in the same forward reproducing THEIR banked fan bit for bit and (ii) two forwards of 10 fresh tokens bit-equal; this needs patched copies of `make_stepA_bundle.py` and `label_perturbations.py` (also: the 64 in-fan proposals of every fresh token must be PDM-labelled, because the anchors' true values come from the banked table today) and is NOT built.
* **G2b cross-device (reported, C9):** CPU fp32 pick agreement vs the GPU bundle; if < 0.99 the pre-committed contingency re-scores on the GPU before step 6b.
* **C-FEAT (L2 sec 2, on the fresh run):** P2's EP head on the exported tokens reproduces step 6's EP scores, max abs diff <= 1e-4 on >= 99 % of rows, in-fan and perturbed, and the shards come from THIS bundle / perts (`score_fresh.py`; RED-tested). Fail = exit 3, no score file.
* **Seal guard on the read:** `analyse_fresh.py` refuses (exit 5) unless this file's sha256 matches, the draft marker is gone, and the text names the carve's token-list sha256, both final heads' sha256 and P2's head sha256 (section 8), and C-FEAT passed on this run. RED-tested.

## 4. Outcome (L1 section 4 with Amendment 1's `interval_aware` rule, applied to the L2 head on BOTH seeds)
Per seed, `stepA.classify` on the pooled A1, A4, A5: **O1** A1 point >= 0.80 AND A4 point <= 0.10 AND A5 point <= 0.05 and no interval straddling the A1 or A4 gate; **O2** a gate fails decisively (A1 interval wholly < 0.80, A4 interval wholly > 0.10, or A5 point > 0.05); **O3** otherwise (an A1 or A4 interval contains its gate). The literal reading (point rules first) is printed beside it. **The L2 head's verdict is the most severe of its two seeds (O2 > O3 > O1), so O1 needs both seeds at O1** (`combine_seeds`, tested).

| verdict on the L2 head | next arm |
|---|---|
| **O1** | **L1 Step B with the L2 EP head** (K=5, M=64, elites M/8, warm start at the pick, 3 search seeds; endpoint dPDMS >= +0.30 with lower end > 0 and dNC, dDAC, dTTC, dC each >= -0.25), SEEN-SCENE, before the navtest confirm. A Step-B amendment must first fix C8's placeholders. **Any integration of an L2 head into REFe's scorer goes through a `launch_gate.py` profile with G-HYG / G-DVB / G-LIVE entries for the new flags (PI rule 2026-09-26; MM ruling E7): a launch blocker for Step B, not for this read.** |
| **O2** | PROPOSED (not in the brief): **L2-BRANCH** as committed in the L2 prereg sec 4 (last scorer-branch block trainable on cached ctx; own amendment before any run) |
| **O3** | PROPOSED: re-run Step A once on the remaining B tokens (905 unused; log count stays 24, so the interval does not narrow in the log dimension, per Amendment C7); still straddling = treat as O2 |

## 5. Reported beside it, ungated (the brief)
* **P2** as the reference, on the same pairs.
* **Paired A1 difference (L2 - P2)** and **paired A3 difference (L2 - P2)**, log-cluster, same resamples; plus the in-fan top-8 EP concordance (SELRANK instrument) of each head and its paired difference.
* **Stamps:** SEEN-SCENE, SELECTION tier (A1.3); the G0 record of the carve (`fresh_sample*.json` -> `g0`); overlap with Step A (tokens / logs); "token-fresh, log-shared" for B; one training seed per head.
* **Four families, per family never pooled:** LONGITUDINAL = A1, A2, speed-scale-wise A1 (0.85 / 0.92 / 1.08 / 1.15) and the in-fan EP concordance; LATERAL = the +-0.3 m pairs' EP (A1 lat) and DAC concordance; TACTICAL = the command-class change share (unchanged by a head retrain, reported); **STRATEGIC = PAUSED** (PI ruling 2026-09-19, ITEM 25), reported as PAUSED, never as a pass. Each with its CI and estimator. Tier: a scorer-admission diagnostic on perturbed plans labelled by the PDM instrument; not a driving claim (T0/T1 do not apply).

## 6. Choices I had to make (the brief left them open) - ratify or overrule at the seal
1. **The universe (section 0).** Brief's recipe = EMPTY; B drafted and pre-launched, C / C' / A1.5 measured and left to the MM.
2. **Seed 20261019** for the carve (the date, Step A's convention: 20260927), fixed before any fresh label or score. Same sampler, same stratification (every log 1, rest proportional to size - 1).
3. **Combination of the two seeds = most severe.** Alternatives (pooled, or "either") were not chosen because the seeds are near-replicates (cosine 0.9962); the rule only prevents a one-seed O1.
4. **O2 / O3 next arms** (section 4) are PROPOSED; the brief commits only O1 -> Step B.
5. **No gates added.** Two things the MM may want gated before the seal, left as reported-only: (a) **A3**: in the banked OOF read the aggregate concordance FELL for the retrained head (A3 0.736 [0.679, 0.787] vs P2 0.766, `L2_RESULT.md`) although A1 rose by 0.14; Step B climbs the AGGREGATE, so A3 is the objective a search sees, and L1 section 4 does not gate it; (b) the L2 in-fan clause (H >= H0 - 0.02) has no counterpart here (no H0 arm in the final fit): the in-fan top-8 concordance is reported, not gated.
6. **Rule geometry, stated BEFORE the read, not a bar.** `interval_aware` O1 needs the A1 interval's LOWER end above 0.80. With 24 log clusters the half-width is about 0.045 (Step A: 0.685 [0.614, 0.761]; banked L2 OOF: [0.783, 0.873]), so an A1 point of 0.83 classifies **O3**, and O1 needs a point of about 0.85 or more. L2-ADMIT itself used a point rule (A1 >= 0.80 and lower CI above the null maximum). The OOF numbers already banked would read O3 under this rule. Expect O3; it is not a surprise if it happens, and under C7 an O3 cannot narrow in the log dimension on B. If the MM wants the point rule here, that is a sealed edit, not mine.
7. **Export reuse.** The EP-head input tokens come from L2's own exporter on the fresh bundle (unchanged code); step 6's CPU scores are kept as the C-FEAT reference, which duplicates ~70 min of CPU on purpose.

## 7. What this licenses, and what it does not
Licenses (on O1): Step B on the same SEEN-SCENE footing, then the frozen EP-ADMISSION gate, then the shared navtest look. Does **not** license a capability claim, a navtest number, an unseen-log claim (B) or an A4 / A5 safety claim beyond the 96 perturbation shapes of the grid.

## 8. Pins the seal must name (the read refuses without them)
* carve token-list sha256 (B, seed 20261019, 400 tokens): `0629811ff5d05879eb62abc5726005ccc4924f03539423003503a099ba29577c` (`raw/fresh_sample_token_disjoint.json`); Step A's: `211e26406f5b4eb083ae38b1c73839826d560042cd75ccf9ff6656262e7c0ffd`
* final head s0 sha256: `bded124e1322b56d9bfd749f73b22c86430ef41f6829856927e5a08831320df4`
* final head s1 sha256: `86f0b20f6ab0d2c46af6b3ae9874b73cc9d94a8065cf7fc0344acbeadcfd4654`
* P2 head sha256: `4b64c3feb89ca571228ed8d4cfa795f3ec5143e71d06545e287e0b9b13ff55d0`; P2 checkpoint sha256 `ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51`
* code: `code/MANIFEST.sha256` of this package (staged copy)
If the MM chooses another universe, replace the first line with that carve's `tokens_sha256` and re-run `fresh_carve.py` for it; nothing else in the read changes.
