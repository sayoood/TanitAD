# DRAFT: NOT REGISTERED, NOT RUN
# PREREG_MEASURE5 AMENDMENT 1: re-test V4 vs V3 against the REPAIRED harness truth, on fresh tokens

Drafted 2026-09-28 ~02:00 Berlin, at the coordinator's request, after the measure-5 readout
(`eval/RESULT_M5_EFFECTIVENESS.md`). The PI decides whether to register it.

⛔ **This file is not a registration.** Registration means adding this file's blob to `eval/raw/SPEC_PREREG_HASH.txt`,
BEFORE any input it names exists. `eval/PREREG_MEASURE5.md` (blob `c668b5b6`) is not edited, and its verdict (ADOPT,
2026-09-28 01:5x) stands as written.

## Why

The registered readout judged ranking against the E-6 table of snapshot 015. That table was scored BEFORE SPEC
Amendment 7 was adopted. Amendment 7 makes the executed plan take heading[18] at the last native pose.

A POST HOC re-read of the same 200 tokens against the repaired truth:

| metric | registered truth | repaired truth |
|---|---|---|
| E1 V4 - V3 | +0.1534 [+0.1037, +0.2104] | -0.0388 [-0.0791, -0.0009] (seed floor 0.0517) |
| 0.75x copy minus its own source | +21.01 PDMS | +1.43 PDMS (copy worse on 60.8 % of 1,600 pairs) |

The v4 comfort label rewards the copy only where the source carries the heading defect: +0.686 there vs +0.006 where
it does not, over 15,200 pairs. The post-hoc re-read cannot overturn the registered verdict. The case for deployment now
rests on a question that needs its own test.

## Models (no retraining)

A0, which is `snap_epoch015.pt` unchanged, plus the NINE scorers the M5 readout used:
* location: `D:/Projects/TanitAD/data/refe_m5/ft/{V3,V4,V4all}_s{0,1,2}.pt`;
* sha256 of each file to be listed here at registration.

## Tokens

Amendment 5's 923 confirmation tokens:
* 43 logs, none of them in W3's 200;
* file: `eval/raw/a5_confirm/a5_confirm_tokens.json`, md5 `3098d178...`.

⛔ W3's 200 are SPENT for this question: the post-hoc re-read has seen them.

## Truth (the reference)

The NAVSIM v1 harness, unchanged (`eval/score_navtest_refe.py`), on the plans the DEPLOYED planner would execute.
`refe/planner.py repair_last_heading` is applied to every candidate:
* the 64 originals;
* the 0.75x copies (the `refe/slow_copies.py` construction) of snapshot 015's own top-8;
* STOP, zeros[20, 3], on which the repair is the identity (asserted).

## Inputs to build (none exists yet for these tokens)

| input | how | cost |
|---|---|---|
| scorer context, proposals and logits for the 923 tokens | `eval/m5_forward.py navtest` extended to a token-list argument (the planner's path, fp32). ALTERNATIVE: M6's `cache_eval_ep015` (1,123 tokens, `planner.REFePlanner.infer`, fp32), admissible only if a gate shows that its context reproduces the forward's logits to 0.0 on these tokens | GPU ~51 min at the MEASURED 3.3 s/token, or 0 GPU via the alternative |
| per-slot repaired harness table, 64 slots | 64 seams (one per slot, repair applied) through the unchanged harness | ~64 runs x ~9 min (MEASURED 115 s per 200 tokens) ≈ 9.6 CPU-h; ~2.4 h at 4 workers |
| repaired copies and STOP | 8 copy seams + the stopzeros seam | ~9 runs ≈ 1.4 CPU-h |
| cross-check | `proptable/a7confirm_ep015/proposals.npz` (the pipeline's own ep015 proposals on these tokens) must equal the new forward's NAVSIM-grid proposals bit for bit | CPU, minutes |

## Gates (all must pass before the statistic is read)

1. The M5 eval gates, unchanged:
   * copies scored by the model == the harness-scored copies (max abs 0.0);
   * harness complete (every run PASS, every token valid);
   * A0 reproduces the forward's logits (<= 1e-3) and its picks on every token;
   * the masked route leaves the 64 unchanged (<= 1e-4).
2. The repair gates:
   * each repaired seam differs from its unrepaired rebuild ONLY in the 4.0 s heading;
   * the unrepaired rebuild equals the pipeline's proposals bit for bit;
   * the repair is the identity on STOP.
3. The independent A0 reference: an explicit-loop concordance on the same inputs reproduces A0's E1 and E3a to rounding.

## Metrics and estimator (identical definitions to PREREG_MEASURE5 §4; only the truth and the tokens change)

* **E1 (PRIMARY):** the masked slow-plan pair concordance.
* **E3a:** concordance among the 64.
* **E3b:** PDMS of the pick among the 64.
* **E2 (reported):** PDMS of the pick among the 64 + extras.
* **Estimator:** per token V4 - V3 of seed means; paired log-cluster bootstrap over the 43 logs, 10,000 resamples,
  95 % percentile, seed 20260927.
* **Seed floor:** the largest |seed_i - seed_j| of the token-mean E1 within V3 and within V4, on these tokens.
* Inference is deterministic.

## Decision (both outcomes committed now)

* **CONFIRMED** iff ALL of these hold:
  * E1 V4 - V3 >= +0.03, with CI lower bound > 0, and larger than the seed floor;
  * E3a CI lower bound > -0.01;
  * E3b CI lower bound > -2.0 PDMS.

  Then PREREG_MEASURE5 §8's live deployment becomes eligible for the PI's go.
* **REFUTED FOR DEPLOYMENT** iff E1's CI upper bound < 0. label_version 4 as built is NOT deployed. The next lever,
  pre-registered as Amendment 2, is **v4r**: the same design with the v4 labeller applying `repair_last_heading` to
  every original and copy before the teacher and NAVSIM-comfort labelling. Its validity is first proven as DAC and
  comfort were: 100 % agreement with the harness on repaired seams.
* **Otherwise NOT PROVEN.** Not deployed. The same next lever applies.
* **Reported, never gating:** EP and the four metric families (`families6.py`) of every arm's E3b and E2 picks; V4all;
  E4; the selection mix.
