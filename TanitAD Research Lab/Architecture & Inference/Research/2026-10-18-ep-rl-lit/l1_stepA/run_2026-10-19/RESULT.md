# L1 Step A RESULT: is test-time search on P2-s0's own scorer admissible for EP?  (2026-10-19)

**Outcome: O2 NOT ADMITTED, by BOTH rules (`interval_aware` and literal). Do not search. The committed next arm is L2.**
Run package: `D:/Projects/TanitAD-artifacts/l1_stepA_run/` (W). Pre-registration: parent `PREREG_L1_SEARCH_ADMISSION.md` sha256 `a76f62d8...` + Amendment 1 sha256 `45f5c791...` (ADOPTED, `MM_ADOPTION_L1_A1.md` sha256 `1f60a939...`), all three re-hashed today and equal to `STEP0_SHA256.txt`. Scorer under test: P2-s0 only (`model_final.pt` sha256 `ea31db22...`, ONE training seed).
Evidence class of every number below: **MEASURED (ours)**, artifact `RESULT.json` unless a line names another file. Rows marked **DERIVED** were computed today from the banked arrays by `derived_tables_stepA.py` (control: it reproduces `RESULT.json` to max abs diff 0.0 before it prints anything).
Repo dir: this package follows `out/paths.json` (`.../2026-10-18-ep-rl-lit/l1_stepA/run_2026-10-19/`), NOT the `2026-09-20-refe-plan/raw/2026-10-19-l1-stepA/` path named in the brief, so it sits beside the landed `l1_stepA/` (commit `d8b77c1`) whose own `RESULT.md` is the launch-ready package and is not overwritten.

## Headline (needs the Master Mind)
1. **O2.** A1 (EP concordance, longitudinal + lateral pairs pooled as the prereg defines it) is **0.685 [0.614, 0.761]** against a gate of >= 0.80. The whole interval is below the gate. A4 (0.015) and A5 (0.008) pass; the O2 verdict rests on A1 alone.
2. **Decision needed before L2 is designed.** Prereg section 4 says search is retried "only after L2's scorer re-passes this Step A", and the O2 row says the Step A pairs are L2's training data. If L2 trains on pairs from these 400 tokens / 24 logs and is then re-scored on the same 400 tokens, the re-pass is IN-SAMPLE by construction. The Master Mind has to say which tokens or logs the re-pass uses before L2 trains. (I do not design L2; fact only: the 1,705-token universe holds 1,305 tokens that are not in this sample and carry no perturbation labels.)
3. **One-place artifacts.** `bundle/bundle.npz` (8.3 MB) and `bundle/ctx.npy` (1.57 GB) exist ONLY on `D:` (an external drive). They were made by a 0.2 GPU-h forward, so they are the expensive part. `perts.npz` (19 MB) is regenerable on CPU from the bundle in seconds (checked today: `stepA.py perturb` rebuilt it array-for-array equal). Over 5 MB, so not staged; sha256 in `MANIFEST.md`.

## 1. Outcome by both rules
| | value |
|---|---|
| `interval_aware` (C7, fixed before any number existed) | **O2** |
| literal (point rules first) | **O2** |
| the two rules differ? | **No.** They differ only when an interval straddles a gate the point already fails; here no interval straddles (below). |
| prereg section 4, O2 row (verbatim rule) | "A1 < 0.80 OR A4 > 0.10 OR A5 > 0.05" |
| the three clauses on these data | **A1 0.685 < 0.80: TRUE** ; A4 0.015 > 0.10: false ; A5 0.008 > 0.05: false |
| Amendment 1 section 4 decision-table row that applies | **"hi < 0.80, any, any: O2 / O2"** (A1 interval hi = 0.761 < 0.80) |
| O1 (needs A1 >= 0.80 AND A4 <= 0.10 AND A5 <= 0.05) | not met: A1 fails |
| O3 (an A1 or A4 interval contains its gate) | not met: A1 [0.614, 0.761] excludes 0.80; A4 [0.0093, 0.0231] excludes 0.10 |
| prereg next arm for O2 | "Do not search. ... use it as the training data of **L2** ... Search is retried only after L2's scorer re-passes this Step A." |

The gap: the A1 point is 0.115 below the gate and the interval's upper end is 0.039 below it. The in-fan references are EP 0.887 (the prereg's own reference: navtest, UNSEEN, D-REFE-SELRANK-1) and 0.892 [0.849, 0.939] (the Amendment 1 diagnostic on this same SEEN universe, `raw/exposure_diagnostic.json` of the landed package). Those are pairwise concordances over the scorer's top-8 fan, a different pair population from the perturbation pairs here, so 0.685 against 0.887 is a contrast, not a paired difference.

## 2. A1-A5 per family (paired log-cluster bootstrap: 24 logs, 10,000 resamples, seed 20260927, the same draws for every cell)
Pairs: 38,400 = 400 tokens x 16 anchors (scorer top-8 + 8 random) x 6 perturbations, none dropped. **Longitudinal** = the 25,600 speed-scale pairs (0.85, 0.92, 1.08, 1.15). **Lateral** = the 12,800 pairs at -0.3 / +0.3 m (ramped, C3).
Denominators: A1 = pairs whose true EP differs by >= 0.05; A2 = same for TTC; A3 = same for PDMS; A4 and A5 = pairs where the scorer aggregate rises by >= 0.02 (6,177 of 38,400); A5b = pairs where the true NC or DAC goes 1 to 0 (descriptive, C6).
The prereg gates read the `pooled` column (C5: A1 to A5 use ALL pairs). It is shown first because it is the gate, but it is dominated by longitudinal pairs (9,629 of the 9,888 A1 pairs), so the two family columns are the per-family reading and are never averaged into a composite.

| quantity (gate) | pooled, the gate quantity | LONGITUDINAL (speed-scale pairs) | LATERAL (+-0.3 m pairs) |
|---|---|---|---|
| **A1 EP concordance** (>= 0.80) | **0.6851 [0.6144, 0.7610]** den 9,888 | 0.6803 [0.6068, 0.7592] den 9,629 | 0.8610 [0.7309, 0.9471] den 259 |
| A2 TTC concordance (no gate) | 0.8431 [0.7842, 0.9031] den 937 | 0.8466 [0.7819, 0.9135] den 841 | 0.8125 [0.6857, 0.9206] den 96 |
| A3 aggregate vs true PDMS (no gate) | 0.7663 [0.7059, 0.8201] den 2,311 | 0.7586 [0.6984, 0.8174] den 1,984 | 0.8135 [0.6777, 0.9172] den 327 |
| **A4 exploit rate** (<= 0.10) | **0.0151 [0.0093, 0.0231]** den 6,177 | 0.0188 [0.0114, 0.0289] den 4,202 | 0.0071 [0.0015, 0.0162] den 1,975 |
| **A5 NC/DAC exploit rate** (<= 0.05) | **0.0081 [0.0031, 0.0156]** den 6,177 | 0.0105 [0.0044, 0.0191] den 4,202 | 0.0030 [0.0000, 0.0112] den 1,975 |
| A5b converse (descriptive) | 0.0790 [0.0327, 0.1289] den 633 | 0.0934 [0.0397, 0.1589] den 471 | 0.0370 [0.0000, 0.1165] den 162 |

Counts behind the rates (DERIVED, `raw/derived_tables_stepA.json`, `I_pair_inventory`): A1 pairs 9,888 = 6,774 strictly concordant + 3,114 strictly discordant + 0 scorer ties; A4: 93 of 6,177 endorsed moves lose >= 0.10 true PDMS; A5: 50 of 6,177 endorsed moves take a true NC or DAC from 1 to 0; A5b: of 633 such breaches the scorer endorsed 50.

Reading, scoped:
* **A1 fails on the longitudinal pairs, decisively.** The lateral A1 (0.861) rests on 259 qualifying pairs; its interval [0.731, 0.947] contains the 0.80 gate, so Step A makes **no lateral claim either way**. Lateral EP rarely moves by >= 0.05 under +-0.3 m (259 of 12,800 pairs).
* A4 and A5 are small in both families and their intervals sit far under their gates. The scorer is weak at ordering EP off-proposal and rarely endorses a move that wrecks the true PDMS or NC/DAC. This is the measured pattern; a gate was not moved to make use of it (see the last section).
* A2 (TTC) reads 0.843, above the in-fan 0.722. The pair populations differ (speed-scaled neighbours of one anchor vs top-8 pairs of a fan), so this is not an improvement claim.

**TACTICAL** (pre-registered quantity: the share of perturbations that change the command class; proxy = nav_command bearing rule on the plan end point, threshold 0.245 rad, **HYPOTHESIS** that it stands for the manoeuvre class). `RESULT.json` carries a point only; the interval is DERIVED with the same draws:
all **0.0273 [0.0172, 0.0370]** (1,049 of 38,400); longitudinal 0.0246 [0.0148, 0.0348] (631 of 25,600); lateral 0.0327 [0.0190, 0.0447] (418 of 12,800).
This is a measure of how much the grid exercises the tactical decision (about 3 %), NOT a measure of the scorer's tactical-decision quality. **Step A says nothing about tactical quality.**

**STRATEGIC: PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` ITEM 25, "temporarily paused ... concerns only the strategic part"). Reported as PAUSED, not as a pass, not omitted. Longitudinal, lateral and tactical keep full force.

Family coverage, stated per rule: Step A is an ordering test of a scorer, so each family enters through its pre-registered quantity. Not produced here, because this study does not roll a driver: target-speed accuracy and headway, heading / curvature / yaw-rate / cross-track error, manoeuvre-decision confusion. None of this is a driving-performance number.

## 3. Gates (values from `scores.npz.gates.json`, `labels.npz.g1.json`, `g0_rerun.json`, `step3.log`)
| gate | prereg / amendment rule | value | control | verdict |
|---|---|---|---|---|
| **G0 vs M-PDM** (A1.2, the PDM-label manifest) | fit logs disjoint from the scorer's PDM fine-tune | universe 1,705 tokens / 24 logs: 0 logs, 0 tokens overlap; the 400 sample: **0 logs, 0 tokens** | a manifest with one injected shared log/token: `passed` False, logs_overlap 1, tokens_overlap 1 | **PASS** |
| **G0 vs M-SCENE** (A1.3) | DECLARED EXPOSURE, not a gate | the 400 sample: **24 of 24 logs, 400 of 400 tokens** are in the final-run bank P2 warm-starts from (103,037 tokens / 1,191 logs) | n/a | **EXPOSURE DECLARED**: stamp SEEN-SCENE (section 4) |
| **G1** label identity | CPU labeller reproduces banked PDM sub-scores of the 64 in-fan proposals on 20 sets to <= 1e-6 | 20 tokens, max abs diff **0.0** | a plan shifted 0.5 m forward changed the label on **17 of 20** tokens (`control_ok`: it must change SOME; the 3 unchanged tokens were not examined) | **PASS** |
| **G2a** set-scoring identity | unperturbed fan through the perturbation path equals the plain score, bit for bit | 20 of 20 tokens, max abs diff **0.0**, input identical 20 of 20 | n/a (identity) | **PASS** |
| **G2b** bundle forward vs banked fan (HARD gate, step 3, attempt 3) | fan, logits and pick reproduced bit for bit | `ZZBUNDLE_OK 400 664 s`, 0 bad, `STEP3_RC` 0; **400 of 400** tokens. Re-checked today from the SAVED bundle against the banked `proposals.npz` / `table_p2_heldout.npz` with `np.array_equal`: fan (A7-repaired) 400/400, logits 400/400, pick 400/400 (DERIVED, `C2_g2b_independent`) | none run in this run; the failure path is `ZZBUNDLE_FAIL` + exit 1 on any fan / logit / pick mismatch (attempts 1 and 2 died before reaching the comparison) | **PASS** |
| **G2c** set-dependence premise | dropping 1 of 64 changes the other 63 scores, else the section 5 caveat is withdrawn | premise TRUE **20 of 20**; max abs change of the other 63 logits min 0.0150 / median 0.0545 / max 1.816 | n/a (it is the control) | premise holds; prereg section 5 caveat **stands** |
| **G3** reachability from the caller | the perturbation reaches `score_trajectories` | 20 of 20 passed; the changed slot equals the expected slot in 20 of 20 | the "delete the perturbation line goes RED" mutation lives in the landed `test_stepA.py` (24 red-under-mutation); **not re-run in this run** | **PASS** (mutation arm inherited from the prep package's 79 tests) |
| **G2b cross-device** (C9) | CPU fp32 scoring vs the banked GPU logits; pick agreement < 0.99 triggers a GPU re-score | max abs logit diff **0.004425**, pick agreement **1.0000** (400 of 400) | n/a | **CONTINGENCY DID NOT FIRE; no GPU re-score was run** |

Two caveats on the cross-device line. (1) The 0.0044 is the COMBINED effect of CPU-vs-GPU fp32 and of the ctx being rounded to bf16 by the attempt-3 fix (section 6); this run cannot separate them (**UNVERIFIED** which dominates). (2) DERIVED bound (`N_numerics_bound`): only 177 of the 9,888 A1 pairs have an EP-logit gap inside +-2 x 0.004425, so a bit-exact re-score could move the A1 point by at most 177 / 9,888 = 0.0179, to at most 0.703, far from 0.80. It bounds the point, not the interval.

## 4. Stamps (carried on every number above)
* **SEEN-SCENE.** All 24 logs, all 400 tokens are inside the final-run bank that P2 warm-starts from and re-trains its trajectory loss on (A1.3; `g0_rerun.json`: 24 logs / 400 tokens). "Held-out" here means held out from P2's PDM-label fine-tune ONLY (A1.4). Consequence per A1.3 (HYPOTHESIS, monotone direction, no counter-mechanism known): **O2 is robust to the exposure**, since a scorer that is wrong off-proposal on scenes it has seen is not expected to be better on scenes it has not. An O1 would have been conditional; none arose. In-fan, the exposure showed no detectable familiarity gain at 24-log resolution (EP 0.892 vs 0.887, the intervals overlap), which does not bound the off-proposal effect.
* **Tier: SELECTION tier, label-only; T0-class labels** (true PDM sub-scores of open-loop plans against the recorded future; the scorer reads frozen `visual_ctx`). Not T1 (no self-action roll), not T2 (not provisioned). **No driving claim, no closed-loop claim, nothing on navtest** (navtest stays unopened by Step A).
* **P2 in-sample** at the scene level (above); P2-s0 is ONE training seed (H-ESTIM-SEED-1).
* **Estimator: paired log-cluster bootstrap, 24 logs, 10,000 resamples, seed 20260927. It answers "would another draw of LOGS say this" and nothing else.** It does not answer "would another TRAINING run say this" (one seed; the A1 shortfall is a statement about P2-s0's scorer, not about the scorer family), and the third variance (inference) does not arise: the scorer is deterministic given ctx and no search ran.

## 5. DESCRIPTIVE table, POST-HOC (not pre-registered; not an outcome input; no multiplicity correction; 24 logs; one seed)
A1 (EP concordance) by speed-scale value and by anchor type, same draws as the headline. Source: `raw/derived_tables_stepA.json`, key `T_A1_descriptive` (23 rows; the table shows 20 of them, omitting only the lateral-only anchor split, which is quoted below, and the headline check row). "true up / scorer up" = the share of the A1-qualifying pairs in which the true EP rises / the scorer's EP logit rises (raw shares, no estimator).

| stratum | A1-qualifying pairs (den) | A1 [95% CI] | true EP up | scorer EP up |
|---|---|---|---|---|
| speed s = 0.85 | 3,928 | 0.789 [0.724, 0.841] | 0.053 | 0.233 |
| speed s = 0.92 | 1,642 | 0.720 [0.647, 0.798] | 0.073 | 0.303 |
| speed s = 1.08 | 1,174 | 0.551 [0.462, 0.675] | 0.884 | 0.512 |
| speed s = 1.15 | 2,885 | 0.563 [0.449, 0.680] | 0.913 | 0.538 |
| all slow-downs (s < 1) | 5,570 | 0.769 [0.702, 0.827] | 0.059 | 0.254 |
| all speed-ups (s > 1) | 4,059 | 0.559 [0.457, 0.677] | 0.905 | 0.530 |
| anchor: scorer top-8 (lon + lat) | 5,106 | 0.712 [0.625, 0.812] | 0.423 | 0.296 |
| anchor: random-8 (lon + lat) | 4,782 | 0.657 [0.587, 0.721] | 0.406 | 0.448 |
| anchor: scorer top-8 (lon only) | 5,039 | 0.712 [0.624, 0.814] | 0.424 | 0.297 |
| anchor: random-8 (lon only) | 4,590 | 0.645 [0.572, 0.713] | 0.407 | 0.451 |
| s = 0.85 / top-8 | 2,088 | 0.876 [0.820, 0.924] | 0.017 | 0.117 |
| s = 0.85 / random-8 | 1,840 | 0.689 [0.605, 0.762] | 0.093 | 0.365 |
| s = 0.92 / top-8 | 791 | 0.815 [0.744, 0.895] | 0.020 | 0.172 |
| s = 0.92 / random-8 | 851 | 0.632 [0.547, 0.718] | 0.122 | 0.424 |
| s = 1.08 / top-8 | 564 | 0.495 [0.373, 0.688] | 0.957 | 0.495 |
| s = 1.08 / random-8 | 610 | 0.603 [0.537, 0.679] | 0.816 | 0.528 |
| s = 1.15 / top-8 | 1,596 | 0.523 [0.353, 0.698] | 0.967 | 0.524 |
| s = 1.15 / random-8 | 1,289 | 0.611 [0.537, 0.679] | 0.847 | 0.555 |
| lateral d = -0.3 m / +0.3 m | 136 / 123 | 0.853 [0.724, 0.944] / 0.870 [0.721, 0.972] | 0.419 / 0.341 | 0.301 / 0.407 |

**Where EP ordering fails off-proposal (post-hoc reading).**
* **On speed-ups it is a coin flip.** For s > 1 the true EP rises in about 90 % of the qualifying pairs, while the scorer's EP logit rises in only about 53 %: A1 0.559, an interval that contains 0.5. The scorer's EP response to going faster is close to flat.
* **On slow-downs it is better but not at the gate:** 0.769 [0.702, 0.827]; the true EP falls in about 94 % of pairs and the scorer's falls in about 75 %. Among the longitudinal cells only the two slow-downs from scorer-top-8 anchors have a point estimate >= 0.80 (s = 0.85: 0.876 [0.820, 0.924]; s = 0.92: 0.815 [0.744, 0.895]).
* **Anchor type changes sign with direction.** Scorer-top-8 anchors are better than random-8 on slow-downs (0.876 vs 0.689 at s = 0.85) and no better, or worse, on speed-ups (0.495 vs 0.603 at s = 1.08, 0.523 vs 0.611 at s = 1.15, intervals overlap). A search warm-started at the scorer's pick (prereg section 4, O1) would start from a top-ranked anchor, and the speed-up cells from top-8 anchors are the weakest (0.495 and 0.523).
* The lateral-only anchor split (top-8 0.657, den 67, 9 logs; random-8 0.932, den 192, 18 logs) is too thin to read (`raw/derived_tables_stepA.json`).
* These are 20 cells read after the fact. They localise where to look; they are **HYPOTHESIS-grade as lever statements** and none moves the outcome. Anything built on them needs its own pre-registration.

## 6. Plumbing deviations from the landed package (`d8b77c1`), none changes a pre-registered quantity
The prereg, Amendment 1, `stepA.py`, `label_perturbations.py`, `real_scorer.py` and their tests were used UNCHANGED: checked today by LF sha256 against the `d8b77c1` blobs, all identical. The landed `launch_stepA.py` stays frozen. Deviations, in the order they happened, quoting `C:/Users/Admin/par18/l1run/STATE.md`:

1. **Launcher copy `launch_stepA_ctype.py`** (PI decision 2026-10-19 "Allow compute-type check"): refuse while a type-`C` process is in the `nvidia-smi` table plus a 6,000 MiB backstop, and a RAM >= 8 GB gate before the lock. Reason: the landed `MAX_USED_MIB = 1500` "can NEVER pass on this box" because desktop C+G apps hold about 2.3 GB. 41 tests, and the original suite against the copy: 30 pass (`raw/pytest_*.log`, `raw/launch_stepA_ctype.diff`). STATE.md: "Patched launcher `code/launch_stepA_ctype.py` (compute-type guard + 6000 MiB backstop + RAM >= 8 GB gate before the lock; 41 tests, RED mutations)."
2. **`tree_hybrid`.** STATE.md: "the p2heldout tree has NO ctx_cache.py; hybrid = its exact code + tip `ctx_cache.py`" (`raw/TREE_HYBRID_PROVENANCE.txt`, `raw/tree_hybrid_MANIFEST.sha256`, 650 files).
3. **Attempt 1 FAILED on plumbing (06:30 to 06:36:25, rc 1).** STATE.md: "Attempt 1 took the lock at ~06:30 and died at the first goal: `planner._goal_for` imports `augment_routes` from `<tree>/code/`, and `tree_hybrid` had NO `code/` dir (`ModuleNotFoundError`, rc=1, lock released). No forward output was produced; nothing was read." Fix: "`tree_hybrid/code/` = the p2heldout tree's `code/` (59 files, byte-identical, sha256 list compared; `augment_routes.py` = tip blob `b4a7b78a...`)". Relaunched 06:39:54 per `step3.log` (STATE.md writes 06:44); it then waited behind other sessions' GPU lock and the RAM gate until about 11:25.
4. **Attempt 2 FAILED on plumbing (about 11:25 to 11:30:25, rc 1).** STATE.md: "`ctx_cache.to_u16` requires a bf16 tensor ('the trainer's autocast dtype'), but `planner.infer` runs fp32 (the planner has NO autocast; the banked P2 fan was made the same way). So `visual_ctx` arrived fp32. Nothing was read." (`step3.log`: `ctx_cache.CaptureError: expected a bf16 tensor ... got torch.float32`.)
5. **The bf16 rounding fix, in the run's COPY `make_stepA_bundle.py`** (`raw/make_stepA_bundle_bf16.diff`; the pre-patch copy `.pre_bf16` is byte-identical to the landed blob, LF sha256 `a7ae5559...`; the patched copy is `8e544b86...`). STATE.md: "`visual_ctx` is rounded fp32 to bf16 explicitly before `to_u16`, so the bundle format (bf16 bits) and `real_scorer.py` are unchanged. G2b (fan / logits / pick bit for bit) does not touch ctx and is unaffected. The CPU scoring phase is already declared NOT bit-comparable (amendment C9) ... Anchor and perturbations share the same ctx, so every comparison is like for like." Consequence worth knowing: the scorer read bf16-rounded ctx (the banked fan was scored on fp32 ctx), so the cross-device number in section 3 includes this rounding.
6. **Attempt 3** launched 11:32:13, took the lock at once, `ZZBUNDLE_OK 400 664 s` (24 logs, 0 bad), `STEP3_END rc=0` 11:44:22, lock released. About 0.20 h wall against 0.25 to 0.4 GPU-h ESTIMATED. Steps 4 to 7 (CPU, one at a time): perturbations 4 s; G1 + labelling 11:45:46 to 11:56:38 (`ZZLABEL_OK 400/400 591 s`); scoring 11:56:38 to 13:08:28 (1 h 12 min); analysis 1 s. `ZZANALYSE_OK O2`. No other session's lock was touched.

## 7. The committed next arm: L2
Prereg section 4, O2 row: "Do not search ... use it as the training data of **L2** (the SR-TTC extension: EP-output contender pairs and designed negatives, `raw/lever_ranking.csv` row L2). Search is retried only after L2's scorer re-passes this Step A."
* `lever_ranking.csv` row L2 (rank 2): "SR-TTC extension (SR-TTC itself is registered, not re-proposed): EP-output contender pairs + designed negatives + EMA soft targets on SR-TTC's cached features"; reach 0.1 to 0.8 PDMS (mid 0.45), cost 1.0 A40-h-equiv, evidence ESTIMATED (policy-level DPO PUBLISHED; scorer-only analogue +0.2). L1, rank 1, is therefore parked until an L2 scorer re-passes.
* **The Step A pairs ARE L2's data:** 38,400 labelled off-proposal pairs (25,600 longitudinal, 12,800 lateral), each with the scorer's six logits for anchor and perturbation and the true PDM sub-scores of both; 9,888 have a true EP gap >= 0.05, of which 3,114 are ordered wrongly by the scorer (`I_pair_inventory`). Arrays: `raw/labels.npz`, `raw/scores.npz` (staged); plans in `perts.npz` and the bundle (D: only).
* **L2 is NOT designed here.** The Master Mind designs and pre-registers it, including the held-out partition question in the Headline.

## 8. What this does not say, and the register
* It does not say search is refuted or that TOAD-style search cannot work on REFe. It says P2-s0's EP ordering off-proposal, on 24 SEEN logs under +-15 % speed / +-0.3 m grids, misses the pre-registered admission bar, so Step B is not earned. No search was run; O4 and Step B do not apply.
* ⛔ Not re-litigated: A1 fails while A4 and A5 pass. The gate is the one written before any number existed; changing it now is a goalpost move and would need its own pre-registration.
* **Proposed register rows (NOT applied; edit from the tip, not from the lagging E: tree):**
  * `GOALS_AND_CLAIMS.md`, `H-L1-TOAD-ADMIT`: "Step A RUN 2026-10-19: O2 NOT ADMITTED (both rules). A1 0.685 [0.614, 0.761] vs gate >= 0.80; A4 0.015, A5 0.008 pass; SEEN-SCENE, P2-s0 one seed, SELECTION tier, T0-class labels. Next arm L2. Artifact: `.../l1_stepA/run_2026-10-19/RESULT.json`."
  * `D-L1-G0-1`: append "Step A ran with this exposure declared; O2 is robust to it (A1.3)."
  * `BACKLOG.md`: `L1-STEPA-RUN` DONE; `L1-UNSEEN-STRATUM` stays 0-GPU backlog and only matters if a later scorer reaches O1.

## 9. Rule Zero
The refutation is not the deliverable. The next lever is committed and unblocked on the evidence side (L2, the Step A pairs are its data); what blocks starting it is the Master Mind's L2 design and the in-sample question in the Headline, named above. The cheapest discriminating work that did not need a decision was run: the post-hoc stratification in section 5, which localises the failure to the speed-up direction.

## Deliverable manifest (full table with sha256 and sources: `MANIFEST.md`)
| artifact | where it lives |
|---|---|
| this `RESULT.md`, `RESULT.json`, `derived_tables_stepA.py`, `raw/derived_tables_stepA.json`, `raw/scores.npz.gates.json`, `raw/labels.npz.g1.json`, `raw/sample.json`, `raw/scores.npz`, `raw/labels.npz`, `raw/step3.log`, `raw/steps4to7.log`, patched `make_stepA_bundle.py` + `raw/make_stepA_bundle_bf16.diff`, launcher copy, `STATE.md`, `MANIFEST.md` | `out:` `C:/Users/Admin/par18/l1run/out/` (staging; `paths.json` maps each to its repo path). **NOT in git yet: the Master Mind lands from `paths.json`.** RESULT.json, scores.npz, labels.npz also on `D:` (W). |
| `bundle/bundle.npz` (8.3 MB), `bundle/ctx.npy` (1.57 GB), `perts.npz` (19.0 MB) | **`D:` only** (W). Over 5 MB, so not staged. bundle and ctx are NOT regenerable without a GPU forward. |
