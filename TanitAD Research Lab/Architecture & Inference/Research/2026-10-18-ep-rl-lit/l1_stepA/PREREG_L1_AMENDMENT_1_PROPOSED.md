# PREREG L1 AMENDMENT 1 (PROPOSED, not adopted) - Step A made launch-ready: G0 ruling, cache ruling, nine clarifications

Parent: `../PREREG_L1_SEARCH_ADMISSION.md`, sha256 `a76f62d8043e4cc9d6c73b35cf2884f27027a1e01bcd3deec658380cc58bd801` (LF blob; verified against `git show FETCH_HEAD:...` on 2026-10-18; a Windows
`git archive` export of the same file hashes differently because it is CRLF). The parent is NOT edited. Written 2026-10-18 by the L1-PREP agent.
**No Step A outcome number exists.** Every number below is a provenance or cost measurement, not an A1-A5 value, so this amendment does not breach "pre-register before any outcome".
Evidence classes: MEASURED (ours, artifact named), INHERITED (another package, not re-verified), ESTIMATED, HYPOTHESIS.

## 1. G0 - what the prereg asked, what was measured, and the ruling this amendment proposes

The prereg (section 2): *"G0 provenance: the 400 sets share no log with P2-s0's training manifest."* It does not say WHICH manifest, and there are two.
`raw/g0_summary.json`, `raw/g0_overlap.json` (code `g0_overlap.py`, `g0_scan_traingrow.py`), fit universe = P2's own held-out table, 1,705 tokens / 24 logs (`table_p2_heldout.npz` sha256 `8eacc19b35b1...`).

| manifest | what it is | tokens / logs | overlap with the universe | with the 400 sample | class |
|---|---|---|---|---|---|
| **M-PDM** | the 174,912 PDM-labelled sets P2's SCORER loss trains on (carved bank, `p2r_carve.py`; `served_keys.tsv.gz` sha `a9cbdf29...` + `navtrain_token_table.tsv.gz` sha `c7456c88...`) | 99,495 / 1,167 | **0 tokens, 0 logs** | **0 / 0** | MEASURED |
| **M-SCENE** | the final-run target bank `train_grow` (`refe_train_grow.tar.zst`, md5 `e07c7d6a...`, streamed). P2 warm-starts from the model trained on it AND its trajectory loss runs on it (`PREREG_P2.md` Design: "Trajectory loss: unchanged ... `train_grow` bank") | 103,037 / 1,191 of 1,192 yaml logs | **1,705 tokens, 24 logs (all)** | **400 / 24 (all)** | MEASURED |

Inside M-SCENE, for the 1,705 universe tokens (per-member counts in `raw/g0_summary.json`): `targets_rank0` 1,705 rows; `targets_rank1` 1,432; the base's fixed-bank **scorer targets** `scorer_targets.jsonl`
18,755 rows on 1,705 tokens and `scorer_targets_rank1.jsonl` 15,752 rows on 1,432 (teacher-calculator labels, not PDM; the base scorer trained on them). INHERITED (`rl_pool_by_base.json`): the bank the main run read also held
the 2,924 on-policy sets of these 24 logs, 2,412 of them labelled before the main run's last bank reload (2026-09-30T18:57:53Z); whether the main scorer loss consumed them is UNVERIFIED.
Control (prereg G0): a manifest with one injected shared log/token makes `g0_check` FAIL (`raw/g0_overlap.json`, `control_deliberate_overlap_on_A`: logs_overlap 1, tokens_overlap 1); a mutation that makes `g0_check` always pass turns
`test_stepA.py::test_red_under_mutation[g0_overlap_control]` green->red as required.

**Does the exposure show up in-fan? (MEASURED, descriptive, `exposure_diagnostic.py`, `raw/exposure_diagnostic.json`; not a Step A quantity: banked labels of P2's own fan, no perturbed pair.)** Top-8 pairwise concordance, the prereg's own reference definition, log-cluster bootstrap 10,000 / seed 20260927:
EP **0.892 [0.849, 0.939]** on the SEEN universe (24 logs) vs **0.887 [0.876, 0.899]** on navtest (UNSEEN, 136 logs); TTC **0.757 [0.655, 0.863]** vs **0.722 [0.686, 0.759]**. The instrument reproduces the registered navtest values to three decimals (0.8871 / 0.7218 against 0.887 / 0.722).
Point estimates sit slightly above navtest, the intervals contain it: no familiarity gain is detectable in-fan at the resolution of 24 logs (+-0.045 EP, +-0.1 TTC). It does NOT bound the off-proposal effect, and it cannot exclude a gain of a few hundredths.

**Reading.** The programme calls this universe "navtrain held-out". It is held out from P2's PDM-label fine-tune ONLY. Its images and teacher trajectories were in the training of the trunk, the trajectory head and the
base scorer. **No navtrain log is disjoint from M-SCENE** (1,191 of 1,192), so the prereg's sentence cannot be satisfied on navtrain under the broader reading, and a literal G0 would fail the whole study before it starts.

**Proposed text (replaces prereg section 2, G0).**
* **A1.1** G0 is evaluated on BOTH manifests and both results are printed in the verdict line.
* **A1.2** G0 PASSES iff the fit logs are disjoint from **M-PDM** (the PDM-label supervision of the scorer under test). Measured: PASS.
* **A1.3** The overlap with **M-SCENE** is a DECLARED EXPOSURE, not a pass/fail gate. Its consequence is on what each outcome licenses, not on whether Step A runs:
  * **O2 (not admitted)** is robust to it: a scorer that is wrong off-proposal on scenes it has seen is not better on scenes it has not (HYPOTHESIS, monotone direction; no counter-mechanism is known).
  * **O1 (admit)** is conditional: it licenses **Step B only**, never a performance claim. The unseen read is the navtest confirm that prereg section 4 already reserves for a Step-B pass. State "SEEN-SCENE" in every O1 verdict line.
  * **O3's remedy changes** (see C7): a "second carve of navtrain held-out logs" does not exist - the only logs outside M-PDM are these 24. The log-cluster interval has 24 clusters whatever the token count, so widening tokens narrows only within-log noise.
* **A1.4** Wherever a register row calls this universe "held-out" it means "held out from P2's PDM fine-tune". (`D-REFE-SELRANK-1`, `D-REFE-EP-1`, `D-REFE-EP-2`, `D-REFE-HN-CENSUS-1` and the SR-TTC dev gate use it; none is invalidated, each is a SELECTION-tier read on seen scenes. Proposed footnote in RESULT.md.)
* **A1.5 (option, not launch-ready)** A scene-disjoint unseen stratum, if the PI wants Step A's O1 to be unconditional: candidates are (i) the 251 navtrain yaml tokens absent from `train_grow` (their logs are unknown), (ii) nuPlan trainval-split DBs present on this box that are not in the bank (1,167 DB files by name; whether any is outside `navtrain.yaml` is UNVERIFIED because the yaml pairs no token with a log, and no local camera frames were found - absence at ONE location: `D:/Projects/TanitAD/data/nuplan-camera` holds `_cdcache`, `mini`, `probe`, `scenarios`, `scenarios4` only). Needs a token-to-DB join and a frame copy; it is a work item, not part of this amendment.

## 2. `visual_ctx` cache ruling

Not cached. `refe/ctx_cache.py` names units `units/<id>.ctx.u16` + `.fan.f32` + `.json` and a `manifest.json`; a cache of another trunk is a `StaleCache` refusal. Probed by name (`*.ctx.u16`, `*.fan.f32`, `ctx_cache*`, `*visual_ctx*`) on `D:/Projects/TanitAD-artifacts` and `D:/Projects/TanitAD` (recursive), `C:/Users/Admin/{qland,p2heldout,p2heldout_work}` (recursive), `C:/Users/Admin` and `E:/Projects/TanitAD/data` (depth 7): the only `*.ctx.u16` files are a pytest temp directory (synthetic). The only real `visual_ctx` cache is the M5 trunk cache
`D:/Projects/TanitAD/data/refe_trunk_cache/d7c59f4f/` (different layout: `visual_ctx.npy` shards; checkpoint `snap_epoch015.pt`, md5 `d7c59f4f...`, NOT P2-s0): 1,600 + 300 navtrain and 200 navtest scenes, **47 of the 1,705** universe tokens
(`raw/real_scorer_check.json`). It is also invalid by construction: `trunk_fingerprint(P2-s0) != trunk_fingerprint(base model_final)` (MEASURED, ctx_cache.trunk_fingerprint, `46a494ff...` vs `9c548285...`), because P2 un-detaches the scorer
context so the LoRA moved.
The banked P2 table cannot stand in either: its `proposals20` stores the A7-REPAIRED plan (last two headings equal on all 1,705 x 64 rows, MEASURED) while the scorer reads the RAW trajectory, and `prehead_features` is the decoder's OUTPUT.

**The forward (one, GPU):** `make_stepA_bundle.py` (derived from `p2ho_dump.py`; one planner forward per token with `ctx_cache.VisualCtxTap`), checkpoint `D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt` sha256 `ea31db2296210b6a35db...`.
* GPU time: **0.25-0.4 GPU-h, ESTIMATED** (basis: `p2ho` dump 1.2-1.8 s/token + ~20 s/log unit, `HANDOFF_2.md`; 400 tokens, 24 logs). Under the dev-box lock only (`launch_stepA.py`).
* Disk: **1.46 GiB** ctx (400 x 7,680 x 256 x 2 B = 3.75 MiB/token, from ctx_cache's own layout) + 6 MB bundle.
* Inputs present (MEASURED for the 400 sampled tokens, plan `C:/Users/Admin/p2heldout/plan/heldout_tokens.json`): all 24 nuPlan DB paths exist (11 in `p2heldout_work/dbs`, 13 under `nuplan-v1.1/splits[/trainval]`); frame roots `p2heldout_work/frames_heldout` (312 tokens), `refe_proxy/pixels` (81), `refe_m5/pixels` (7) all exist. Whether every individual JPEG resolves is checked by the forward itself (`FrameResolver`; a missing frame is a `bad` row and exit 1).
* Everything else is CPU: scoring **MEASURED 0.061 s per 64-set** on the real P2-s0 scorer (`raw/smoke_real_scorer.json`, 4 threads) = ~40 min for 400 x 97 sets; labelling ESTIMATED ~16 min per worker (38,400 plans at ~25 ms, basis `p2ho_score` 1.6 s/token).

## 3. Clarifications - degrees of freedom the prereg left open, fixed BEFORE any outcome (each also recorded in RESULT.json)

* **C1 Sample.** Unit = TOKEN (P2's own fan, 64 proposals). Universe = the 1,705 tokens of P2's table, not the "3,137 sets" (those are the older final-model fan's labelled sets, 3,137 label sets over the same 1,705 tokens; the scorer under test picks from ITS fan). 400 tokens: every log gets 1, the remaining 376 allocated proportionally to (log size - 1) by largest remainder, draw `default_rng(20260927)` over logs in sorted order. Reads no label. Sample sha256 of the sorted token list: `211e26406f5b4eb083ae38b1c73839826d560042cd75ccf9ff6656262e7c0ffd` (`raw/g0_overlap.json`).
* **C2 Speed perturbation.** The same path travelled s times as fast: pose i moves to arclength s x S_i on the original polyline (extrapolated along the last segment for s > 1); yaw interpolated along the path and held past its end. NOT a scaling of coordinates (the mutation arm proves they differ).
* **C3 Lateral offset.** +-0.3 m along each pose's left normal, **ramped** with weight sin^2(pi/2 x i/T) (0 at the ego, 1 at the horizon); yaw untouched. Alternative recorded, not used: constant offset from the first pose (`--lat-profile const`).
* **C4 Raw vs executed space.** The scorer reads the model's RAW trajectory (corrupted last heading, A7); the labeller scores the EXECUTED plan. A perturbation is built on the executed plan, labelled as such, and handed to the scorer with the anchor's raw last heading restored, so identity (s = 1, d = 0) returns the raw anchor bit for bit.
* **C5 Pairs.** pair = (anchor in its own unperturbed set, perturbation in the set with the anchor's slot replaced). Anchors: the scorer's top-8 by aggregate + 8 random of the other 56 (draw a pure function of (seed, token)); 6 perturbations; 96 pairs/token. A1/A2 use ALL pairs (lon and lat), also reported split; A3 uses the same 0.05 true-difference rule as A1/A2; a scorer tie counts 0.5; ordering uses logits (monotone with the head's probability). A4 is reported pooled (the gate) AND per kind.
* **C6 A5.** Primary reading: of the pairs the scorer endorses (aggregate rises >= 0.02), the share where true NC or DAC goes exactly 1 -> 0 (the gate, <= 0.05). The converse (of the true NC/DAC breaches, the share the scorer endorses) is reported as A5b, descriptive. NC takes values {0, 0.5, 1}; 1 -> 0.5 is not counted.
* **C7 Outcome precedence.** The prereg writes O2 by point estimates and O3 by intervals, so a straddling gate can satisfy both. Proposed `interval_aware` rule: **O2** if a gate fails decisively (A1 interval wholly < 0.80, or A4 interval wholly > 0.10, or A5 point > 0.05); else **O3** if the A1 or A4 interval contains its gate; else **O1**. The literal reading (point rules first) is computed and printed beside it as `outcome_if_literal`; they differ only when an interval straddles a gate that the point estimate already fails. **O3's next arm becomes**: re-run Step A once on ALL 1,705 tokens (log count stays 24); if still straddling, treat as O2. (The prereg's "second carve of navtrain held-out logs" is impossible, section 1.)
* **C8 Step B placeholders.** The prereg fixes K=5, M=64, elites M/8, warm start, 3 seeds, but not the sampling space or widths. `tts_cem.py` searches the ADMITTED space theta = (speed scale exp(u), lateral offset d) with initial widths (0.10, 0.20 m) and clips (x0.7-1.3, +-0.6 m), seeds 20260927/8/9. These are placeholders to be fixed in a Step B amendment before it runs; O4's "widen the covariance once" is `widen=2`.
* **C9 Numerics (G2).** G2a (always, bit for bit): the unperturbed fan scored through the perturbation path equals the plain set score. G2b: the GPU bundle forward must reproduce the banked fan, logits and pick bit for bit (hard gate in `make_stepA_bundle.py`; the P2 table's own determinism check read max abs 0.0). The CPU scoring phase is fp32 and NOT bit-comparable with GPU logits; it reports max |difference| and pick agreement (`G2b_cross_device`, not gating). **Pre-committed contingency:** if pick agreement < 0.99, the scoring phase is re-run on the GPU with `--autocast bf16` under the launcher before analysis. Anchor and perturbation are both scored by the same path, so every pair is like for like. G2c (set-dependence premise) is reported; if it reads 0 the prereg section 5 caveat is withdrawn (on real weights with a random ctx it read 0.020 and 0.482 logits, `raw/smoke_real_scorer.json`: premise true).

## 4. Decision table as implemented (`stepA.classify`, `test_stepA.py::chk_classify`)

| A1 interval | A4 interval | A5 point | interval_aware | literal |
|---|---|---|---|---|
| lo > 0.80 | hi < 0.10 | <= 0.05 | O1 | O1 |
| hi < 0.80 | any | any | O2 | O2 |
| contains 0.80, point < 0.80 | not decisive | <= 0.05 | **O3** | **O2** (the C7 difference) |
| any | lo > 0.10 | any | O2 | O2 |
| any | contains 0.10, point <= 0.10 | <= 0.05 | O3 | O3 |
| any | any | > 0.05 | O2 | O2 |

Strategic: PAUSED (PI ruling 2026-09-19, ITEM 25), printed as PAUSED. Tactical: share of perturbations that change the nav_command-rule class of the plan's end point (HYPOTHESIS: a proxy; threshold 0.245 rad from `nav_command.py`).
Estimator for every interval: log-cluster bootstrap over the 24 logs, 10,000 resamples, seed 20260927, the same draws for every quantity; it answers "another draw of LOGS" only - not another training seed (H-ESTIM-SEED-1), not another search seed.

## 5. Not changed

Thresholds (A1 >= 0.80, A4 <= 0.10, A5 <= 0.05), the perturbation grid (0.85, 0.92, 1.08, 1.15; -0.3, +0.3 m), anchor counts, the 0.05 / 0.02 / 0.10 pair thresholds, O2's next arm (L2), O4's rule, the navtest policy (opened once, only after a Step-B pass), the rule that Step A runs P2-s0 only.

## 6. Adoption checklist (Master Mind)
1. Accept or amend C1-C9 and A1.1-A1.4; write this file's sha256 beside it. 2. Copy the parent prereg and this amendment unchanged into the run package. 3. Run the plan printed by `python launch_stepA.py plan --work-dir <D>` (one GPU step, ~0.3 GPU-h, behind the lock; S0 currently holds it).
