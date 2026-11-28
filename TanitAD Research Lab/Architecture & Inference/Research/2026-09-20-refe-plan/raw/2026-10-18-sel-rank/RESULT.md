# RESULT: SEL-RANK step 1 (0 GPU). P2's 5.88-point non-zero selection pool is 77 % EP in the swap but TTC-bound in the counterfactual: a perfect TTC head recovers +1.83 [1.57, 2.11] of it, a perfect EP head +1.57 [1.35, 1.80] (+0.80 once its violation signal is removed). No CPU recalibration of the six logits recovers any of it. A navtrain held-out table exists, but for SFT-4 sibling scorers, not P2.

Agent for the Master Mind, 2026-10-18, BACKLOG row `SEL-RANK`. Everything here is CPU, descriptive, and nothing is fitted on navtest.

**Tier / estimator / evidence class.**
- Tier T1 (NAVSIM v1.1 pseudo-simulation of one open-loop plan), P2 `model_final` (sha256 `ea31db22...`), full navtest 12,146 tokens / 136 logs.
- Estimator: paired log-cluster bootstrap (`zero_tail.Boot`, vendored byte-identical, sha256 `149c3e97...`), 10,000 resamples, seed 20260927, 95 %. **It answers "another draw of LOGS" only.** P2 is ONE training seed (H-ESTIM-SEED-1) and its inference is deterministic, so no interval below speaks to another training run.
- Evidence class: MEASURED (descriptive) on navtest. The dev-set sections (6, 7) are MEASURED on a different fan and different scorers and are labelled DEV. Anything about an arm is ESTIMATED or HYPOTHESIS and says so.
- Navtest is the CONFIRMATION set: the table was read, never fitted. The one thing fitted anywhere in this package (section 7) is fitted on the navtrain held-out table by grouped-by-log cross-validation.
- Inputs and hashes: `inputs.json`. Table `D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz`, sha256 `33441082b7044c95...` (verified).

## 0. The selection score (file:line, mirror tip 309953b)

P2 is a **6-output** scorer. Its deployed score is the `navsim_v1` branch of `REFePlanner.aggregate`, argmax in `_pick`:
- `refe/planner.py:520` `V1_W = (5.0, 5.0, 2.0)`;
- `refe/planner.py:558-561` `if self.rule == "navsim_v1": ... return p[..., 0] * p[..., 1] * (p[..., 2]*w1[0] + p[..., 3]*w1[1] + p[..., 4]*w1[2]) / w1.sum()`, with `p = score.sigmoid()` (`:537` def, DDC not in the product);
- `refe/planner.py:576-583` `_pick` ends in `int(self.aggregate(score)[0].argmax())`.

The trainer's `sel_score` is `refe/scorer_finetune.py:998-1000`: `agg_v1(p[..., :6]) * (p[..., 6] if (a.lane_head and p.shape[-1] == 7) else 1.0)`, with `agg_v1` at `refe/scorer_finetune.py:85-87` (`NC*DAC*(5EP+5TTC+2C)/12`). For a 6-output scorer it is the same function. **Gate: recomputing the planner rule in float32 torch from the table's `logits` reproduces `pick` on 12,146 / 12,146 tokens** (`raw/attribution.json` G).

**NAVSIM v1 PDMS formula, file:line.** `NC * DAC * (5 EP + 5 TTC + 2 C) / 12`, DDC at weight 0: `navsim @ 3e8291b pdm_scorer.py:38-42`. That file:line is **INHERITED by citation** (`refe/planner.py:517`, `eval/rule_mismatch_diag.py:7`); the NAVSIM source is not on this box. The formula itself is **MEASURED**: it reproduces the table's `pdms` on all 12,146 x 64 entries with **max |diff| = 0.0**, and the control with DDC in the product reads **1.0** (it does not reproduce).

## 1. Gates and controls (all PASS)

| gate | reading |
|---|---|
| table sha256 | `33441082b7044c95...` |
| PDMS = NC*DAC*(5EP+5TTC+2C)/12 on all entries | max abs diff **0.0**; DDC-in-product control **1.0** |
| planner rule reproduces `pick` | **12,146 / 12,146** (torch f32) and 12,146 / 12,146 (numpy f64) |
| `pdms[pick]` equals the banked csv | max abs diff 0.0; mean **88.8184** (matches `anatomy.json`) |
| pool reproduces the register | non-zero-pick gap **5.875 [5.535, 6.213]** (register 5.88 [5.54, 6.21]); zero-pick gap **3.538 [2.899, 4.231]** (register SEL ceiling 3.54 [2.90, 4.23]) |
| independent cross-check | my oracle-DAC bound **+3.184 [2.498, 3.955]** equals the O-DAC oracle in `D-REFE-SELRULE-DAC-CONFIRM` (+3.184) |
| decomposition exact | five terms sum to oracle - pick token by token, max abs error **2.8e-16** |
| identity control (no head replaced) | gain exactly **0.000**; picks identical on 12,146 / 12,146 |
| all five heads replaced by truth | **98.231** = best-of-64 oracle, max per-token difference 0.0 |
| negative control (realised EP, shuffled across proposals within token) | gain **-3.756 [-4.243, -3.266]** (must be < 0: PASS) |
| families instrument | on the P2 pick, `families6.py` reproduces RESULT_P2's P2 row: speed MAE 0.806 [0.778, 0.836] (banked [0.778, 0.836]); along MAE 1.136 [1.093, 1.180] (banked [1.093, 1.180]); tactical accuracies 0.9237 / 0.6748 / 0.7794 exactly |

## 2. Deliverable 1(a): per-term attribution of the non-zero pool

Pool S = pick non-zero AND oracle strictly better: **7,847 tokens** (11,637 non-zero picks; **3,790** of them already have zero gap). Method: Shapley value over the three multiplicative factors (NC, DAC, W) of the move pick -> oracle proposal (6 orderings), W split over EP/TTC/C by its own linear weights 5/5/2 over 12; averaged over proposals tied at the oracle PDMS (mean 16.1 tied, median 6, max 64 per token). Exact.

| term | contribution to the 5.875 (x100, over all 12,146 tokens) | share | oracle strictly better in | mean at pick -> at oracle |
|---|---|---|---|---|
| **EP** | **4.531 [4.258, 4.816]** | **77.1 %** | **97.3 % [96.6, 97.9]** of pool tokens | 0.796 -> 0.965 |
| **TTC** | **1.308 [1.073, 1.555]** | **22.3 %** | **4.9 % [3.9, 5.9]** (384 tokens) | 0.950 -> 0.999 |
| NC | 0.032 [0.006, 0.066] | 0.5 % | 0.1 % | 0.999 -> 1.000 |
| DAC | 0.000 | 0 % | 0 % | 1.000 -> 1.000 |
| C | 0.004 [0.000, 0.009] | 0.1 % | 3 tokens | 1.000 -> 1.000 |

Reading: the oracle's proposal wins on **ego progress in almost every pool token** (95.1 % of tokens have EP as the dominant term), and on **TTC in a small set of tokens that each carry a lot** (373 tokens dominated by TTC carry 1.39 points). Comfort and the multiplicative terms are not the pool. Dev replicate (nine sibling scorers): EP 81-83 %, TTC 17-19 %.

## 3. Deliverable 1(b): scorer calibration per head, and which head mis-ranks

Rank correlation of the head's probability against the realised sub-score across the 64 proposals of a token (average ranks), then pooled. `n` is tokens where both vectors vary.

| head | per-token Spearman, all 64 [95 %] (n) | pooled within-token rank corr | inside the scorer's own **top-8**: Spearman (n) | top-8 **pairwise concordance** (pairs, tokens) | mean prob - mean realised |
|---|---|---|---|---|---|
| **EP** | 0.442 [0.426, 0.458] (11,937) | 0.433 [0.416, 0.449] | **0.765 [0.747, 0.784]** (9,154) | **0.887 [0.876, 0.899]** (100,168; 8,412) | +0.002 [-0.006, +0.009] |
| **TTC** | 0.502 [0.482, 0.522] (6,393) | 0.390 [0.363, 0.416] | **0.317 [0.266, 0.369]** (1,018) | **0.722 [0.686, 0.759]** (11,443; 1,018) | -0.010 [-0.018, -0.002] (pool: -0.030) |
| C | 0.315 (39 tokens) | 0.020 | not estimable | not estimable | -0.000 |
| NC | 0.506 [0.487, 0.525] (4,728) | 0.338 | 0.324 (347) | 0.722 [0.663, 0.782] (4,004; 347) | -0.002 |
| DAC | 0.482 [0.460, 0.503] (4,962) | 0.320 | 0.329 (711) | 0.723 [0.690, 0.756] (8,114; 711) | +0.008 |

- **C carries no ranking information in this table**: its realised value varies within a token on 0.32 % of tokens.
- Three lenses, and they do not agree on EP:
  1. **Raw rank agreement over all 64**: EP is lowest (0.442), but the spread is small (0.44-0.51).
  2. **Where the scorer actually chooses (its own top-8 contenders)**: EP ranks them well (concordance 0.887); **TTC, NC and DAC rank them at ~0.72**. TTC is the contender mis-ranker with pool value, because the pool has TTC contrast in 384 tokens worth 1.31 points; NC and DAC have almost none (11 and 0 tokens).
  3. **Counterfactual value** (section 5): a perfect TTC head is worth at least a perfect EP head on the pool.
- **Decisive pair (pick vs oracle) on pool tokens.** In 68.2 % [65.9, 70.6] of pool tokens the EP head's predicted term favours the pick while the realised EP favours the oracle; for TTC it is 3.2 % [2.5, 4.0]. This is conditioned on the pick being the argmax, so it is a description of where the inversion sits, not a head ranking. The error-budget table in `raw/attribution.json` (`decisive_pair_error_budget_pool_S`) is **not quoted**: its predicted terms are winner's-curse inflated for every head.
- **Answer: the TTC head mis-ranks most where it counts** (contender concordance 0.72, largest counterfactual single-head bound); the EP head is the dominant term in the swap (77 %) and has the lowest raw correlation, but it ranks the contenders well. Dev replicate: TTC pool-share 0.239-0.263 > EP 0.210-0.228 in **all nine** sibling scorers (point estimates; 24-log intervals overlap).

## 4. Deliverable 1(c): concentration, and by driving command

Pool = **5.875 [5.535, 6.213]** x100. Top-k share is of the pool among the 11,637 non-zero-pick tokens; the threshold is re-determined in every resample.

| top fraction of tokens by gap | tokens | share of the pool [95 %] |
|---|---|---|
| 1 % | 117 | 9.1 % [8.6, 9.7] |
| 5 % | 582 | 31.7 % [29.7, 33.6] |
| **10 %** | **1,164** | **44.9 % [42.8, 47.0]** (top 10 % of ALL 12,146 tokens: 45.9 % [43.8, 48.0]) |
| 20 % | 2,328 | 63.6 % [61.5, 65.7] |
| 50 % | 5,819 | 94.3 % [92.8, 95.7] |

- **The pool is diffuse, not a tail.** Median per-token gap 0.034 PDMS points, p90 0.138, p99 0.480, max 0.815. 1,440 tokens cover half of it, 3,761 cover 80 %. The 10 % threshold is 0.138 points per token.
- **By log**: 135 of 136 logs carry some. The 10 largest logs hold 28.9 % of the pool on 25.6 % of the tokens; a permutation null (same gaps spread at random over logs) gives 26.4 % [25.3, 27.6]. Mild log clustering, nothing a per-log fix would catch.
- **By driving command** (`navtest_inputs.json.gz`, one-hot LEFT / STRAIGHT / RIGHT / UNKNOWN; 0 UNKNOWN):

| command | tokens (share) | P2 pick | oracle | non-zero-pick gap per token (x100) | share of pool [95 %] | zero-pick rate |
|---|---|---|---|---|---|---|
| LEFT | 2,501 (20.6 %) | 86.57 [84.00, 88.97] | 97.52 | 5.93 [5.04, 6.80] | 20.8 % [16.7, 24.7] | 6.2 % |
| STRAIGHT | 8,070 (66.4 %) | 90.38 [89.67, 91.05] | 98.74 | 5.78 [5.40, 6.19] | 65.4 % [60.1, 70.8] | 2.9 % |
| RIGHT | 1,575 (13.0 %) | 84.38 [81.91, 87.10] | 96.76 | 6.27 [5.40, 7.10] | 13.8 % [10.6, 17.3] | 7.8 % |

  The pool follows the token mix. Turns are lower in level and have more zero picks (the P6/DAC domain), but the non-zero pool per token is the same across commands, and EP/TTC split alike.

## 5. Deliverable 1(d): analytic upper bounds, nothing fitted

At selection, replace the named head(s)' predicted probability by its **realised** sub-score (same token, same 64 proposals), keep the other heads as P2 predicted them, apply the planner's rule, take the argmax. Gain = PDMS at that pick - PDMS at P2's pick, x100. **An upper bound for a perfect head, not an estimate of any trainable arm.**

| replaced by truth | total gain [95 %] | from non-zero-pick tokens (the pool) | share of the 5.875 pool | from zero-pick tokens | tokens better / worse |
|---|---|---|---|---|---|
| **TTC** | **+3.043 [2.629, 3.481]** | **+1.833 [1.569, 2.112]** | **31.2 % [27.3, 35.1]** | +1.210 | 4,181 / 153 |
| TTC, realised only where NC*DAC != 0 (sensitivity) | +2.938 [2.598, 3.295] | +1.723 [1.518, 1.951] | 29.3 % [26.4, 32.4] | +1.215 | 4,158 / 167 |
| **EP** | **+4.617 [3.949, 5.316]** | **+1.573 [1.349, 1.796]** | **26.8 % [23.2, 30.3]** | +3.045 | 6,017 / 336 |
| EP, realised only where NC*DAC != 0 (sensitivity) | +2.665 [2.104, 3.254] | +0.801 [0.500, 1.103] | 13.6 % [8.5, 18.8] | +1.864 | 5,782 / 444 |
| C | -0.004 [-0.022, +0.015] | -0.011 | -0.2 % | +0.007 | 9 / 8 |
| **EP + TTC** | **+6.959 [6.325, 7.609]** | **+3.647 [3.378, 3.921]** | **62.1 % [59.4, 64.8]** | +3.312 | 7,379 / 65 |
| NC | +0.989 [0.776, 1.203] | +0.154 | 2.6 % | +0.835 | 2,261 / 203 |
| DAC | +3.184 [2.498, 3.955] | +0.684 | 11.6 % | +2.501 | 3,330 / 400 |
| NC + DAC | +4.043 [3.336, 4.812] | +0.933 | 15.9 % | +3.110 | 5,122 / 200 |

- **Realised EP is not a clean "progress" signal.** MEASURED: EP is exactly 0 on **100 %** of the 154,428 proposals with NC = 0 or DAC = 0 (and on 0.015 % of the rest), and it is normalised by the best non-violating proposal. So "oracle EP" also hands the scorer a violation oracle, which is why its **total** (+4.62) is larger than TTC's while its **pool** share is not. The sensitivity row removes that signal. Realised TTC leaks less: TTC = 0 on 97 % of NC = 0 proposals and 49 % of TTC = 0 proposals are also NC = 0, yet restricting TTC to non-violating proposals moves the pool bound only from 1.833 to 1.723.
- **Paired difference of the two bounds on the pool (same tokens, same draws): TTC - EP = +0.260 [-0.066, +0.594]** (not separated); **TTC - EP(non-violating only) = +1.031 [+0.672, +1.395]** (separated). On the total (zero-pick tokens included) EP's bound is larger: -1.575 [-2.373, -0.822].
- **Where the realised gain of a perfect TTC head comes from** (Shapley, pool): TTC term 1.303, **EP term 0.652**, DAC term -0.152, NC 0.028. A perfect TTC head works partly by letting the scorer take a higher-progress proposal safely. A perfect EP head: EP term 1.496, TTC 0.060.
- **The pool is very sensitive to TTC**, because the scorer's selection margins are tiny (median top1-top2 log-score margin 0.0024, top1-top8 0.025 on the dev set). Moving the TTC probability a fraction lam of the way to the truth (probability space, an ASSUMED dose model, not a head-quality measure): lam = 0.1 -> pool +1.139 (19.4 %), 0.25 -> +1.348 (22.9 %), 0.5 -> +1.506 (25.6 %), 1.0 -> +1.833 (31.2 %). EP (non-violating only): 0.1 -> +0.076, 0.25 -> +0.156, 0.5 -> +0.426, 1.0 -> +0.801. The same sensitivity means any head retrain will move many picks, so the seed floor (section 7) matters.

## 6. The four metric families for the selectors (as far as the table allows)

`families6.py` unchanged (tanitad venv), one executed 8-pose plan per token from the table. Intervals are families6's own per-arm episode-cluster bootstrap (136 logs, 2,000 resamples) and are **unpaired**: reading a difference off two of them is conservative. Full table: `raw/families_table.md`.

| family: component | P2 pick (PDMS 88.82) | oracle-TTC (91.86) | oracle-EP (93.44) | oracle-EP+TTC (95.78) | best-of-64 (98.23) |
|---|---|---|---|---|---|
| LONG: speed MAE m/s | 0.806 [0.778, 0.836] | 0.769 [0.741, 0.800] | 0.870 [0.831, 0.911] | 0.841 [0.800, 0.886] | **0.983 [0.919, 1.052]** |
| LONG: along-track MAE m | 1.136 [1.093, 1.180] | 1.093 [1.047, 1.138] | 1.255 [1.201, 1.313] | 1.229 [1.172, 1.292] | **1.435 [1.350, 1.525]** |
| LONG: speed within 1 m/s | 0.721 [0.708, 0.735] | 0.738 [0.722, 0.752] | 0.692 [0.673, 0.709] | 0.703 [0.683, 0.722] | **0.644 [0.616, 0.671]** |
| LONG: progress ratio (human = 1) | 1.024 [1.006, 1.042] | 1.059 [1.043, 1.076] | 1.011 [0.988, 1.035] | 1.044 [1.024, 1.066] | **1.177 [1.158, 1.196]** |
| LAT: heading MAE deg | 2.484 [2.283, 2.671] | 2.452 [2.256, 2.635] | 2.532 [2.341, 2.713] | 2.535 [2.345, 2.718] | 2.652 [2.431, 2.857] |
| LAT: yaw-rate MAE deg/s | 3.022 [2.834, 3.211] | 2.956 [2.769, 3.144] | 3.023 [2.843, 3.203] | 2.976 [2.794, 3.159] | 2.935 [2.743, 3.121] |
| LAT: curvature MAE 1/m | 0.021 [0.019, 0.022] | 0.020 [0.018, 0.021] | 0.020 [0.018, 0.021] | 0.019 [0.017, 0.020] | 0.016 [0.015, 0.017] |
| LAT: cross-track MAE m | 0.392 [0.363, 0.418] | 0.382 [0.355, 0.408] | 0.406 [0.376, 0.434] | 0.405 [0.375, 0.434] | 0.445 [0.405, 0.486] |
| TACT: lateral decision acc | 0.924 [0.914, 0.933] | 0.924 [0.914, 0.933] | 0.928 [0.918, 0.938] | 0.927 [0.916, 0.936] | 0.928 [0.917, 0.939] |
| TACT: longitudinal decision acc | 0.675 [0.642, 0.705] | 0.677 [0.643, 0.710] | 0.668 [0.630, 0.706] | 0.671 [0.632, 0.709] | 0.628 [0.588, 0.667] |
| TACT: 5-way manoeuvre acc | 0.779 [0.758, 0.798] | 0.781 [0.760, 0.801] | 0.786 [0.765, 0.805] | 0.786 [0.766, 0.805] | 0.749 [0.729, 0.768] |
| TACT: goal-point error m | 3.147 [3.034, 3.264] | 3.006 [2.886, 3.134] | 3.403 [3.246, 3.573] | 3.297 [3.127, 3.483] | 3.885 [3.628, 4.158] |
| STRATEGIC | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25) and UNAVAILABLE in NAVSIM. Never a pass. | | | | |
| LONG: distance-keeping (headway, time gap, TTC to lead) | **UNAVAILABLE**: `families6.py` is not given a lead-agent track (`lead=`), n = 0. A work item, not a pass. | | | | |

**Reading (descriptive).** The 98.23 oracle is reached by driving **farther and faster than the human** (progress ratio 1.177, speed MAE +0.18 m/s, along-track MAE +0.30 m, all separated from the pick). The EP-oracle selection moves the longitudinal family **away** from the human (overlapping with the pick); the TTC-oracle selection moves it **toward** the human (overlapping). So the pool is bought with progress, and a selector that captures it through EP will show longitudinal-family movement that the binding four-family rule requires us to report.

## 7. A navtrain held-out sub-score table EXISTS, but not for P2

**The ZT5 dev set** (`PK/raw/2026-10-17-tracking-execution/data/dev/dev_{0,1}of2.npz`, repo): 2,466 sets (1,225 + 1,241) on 13 logs, carrying `banked [n, 64, 6]` (nc, dac, ep, ttc, c, pdms) for all 64 proposals and the picks of three scorers, but **no scorer logits**.

**The full dev table** (3,137 sets on 24 navtrain logs, **0 shared with the 136 navtest logs**, checked):
- labels: `D:/Projects/TanitAD-artifacts/s1_read_cache/labels/x/heldout_pdm/pdm_onpolicy_r{0,1}_hw{0..6}.jsonl`. These hold nc, dac, ep, ttc, c, ddc, pdms for 64 proposals per set, `label_version` 5, plans "executed (A7 repair)", **fan = `model_final` ckpt_step 10075 (u400's generator, NOT P2's fan)**. They reproduce `NC*DAC*(5EP+5TTC+2C)/12` to 6.7e-6 (5-decimal rounding), and EP is 0 on 100 % of violating proposals as on navtest;
- scorer logits `[3137, 64, 7]`: `.../s1_read_cache/arms/<arm>/dump/scores_update400_B.npz` for **nine SFT-4 sibling scorers** (u400 + 400 updates; arms N-s0/1/2, H1-s0/1/2, H2-s0/1/2), 7 outputs (6 + lane), native rule `navsim_v1_lane`. **None of them is P2's scorer.** I found no P2 held-out dump: `PREREG_P2.md` lists the held-out selection read as reported-not-gating, `RESULT_P2.md` does not contain it, `D:/Projects/TanitAD/data/refe_runs_eval/p2/` holds only `model_final.pt`, `summary.json` and `hf/`, and a search of `D:/Projects/TanitAD-artifacts` for score dumps found only the SFT-4 sibling arms and RL temp dirs.

**Descriptive replicate on the dev table** (`raw/dev_replicate.json`; 24-log intervals, so wide):
- Level: 6-head `navsim_v1` pick 90.49-90.94 (nine scorers), oracle 98.78, non-zero pool **5.06-5.11** (navtest P2: 5.88).
- Swap attribution: EP 81-83 %, TTC 17-19 %.
- Bounds, share of pool, range over nine scorers: **TTC 24-26 %**, EP 21-23 %, EP non-violating 14-16 %, EP+TTC 47-51 %. The order TTC > EP > EP-non-violating holds in all nine.
- Top-8 Spearman (N-s0/1/2): EP 0.66-0.68, TTC 0.36-0.40 (on 213-229 tokens).
- **Training-seed floor on the pick (MEASURED, dev, 24 logs):** the nine scorers' pick PDMS spans **0.44** (SD 0.16); within a recipe, three seeds span 0.35 (N), 0.40 (H1), 0.33 (H2). The bootstrap interval cannot see this.

**What building P2's own held-out table costs** (ESTIMATED; arithmetic shown):
- **GPU forward of P2 on the held-out sets**: 3,137 sets x 1.44 s/set = 4,520 s = **1.26 h** (1.44 s/set = 907 s for the final 629-token navtest chunk, `STATUS.json` `last_dump_seconds`, `chunks.json`); 2,466 sets = 0.99 h. The dev-box GPU is contended: row `P2F-1` records a > 19 h wait for the lock.
- **Labels for P2's own fan** (needs the DBs, on the dev box for the 13 logs / 2,466 sets only): about 3 s metric-cache parts + 2-3 s scoring per set (`PREREG_ZT5.md` section 0) -> 2,466 x 5.5 s = 13,563 s = **3.8 h** on one CPU worker, ~1.3 h on three.
- **Or skip the relabel**: score the already-labelled u400 fan with P2's scorer. The labels exist for all 3,137 sets; this needs a proposal-injection path into `REFe.score_trajectories` that I did not find (UNVERIFIED), plus the 1.26 h forward. The scorer would then see a fan it was not trained on.
- No pod is needed for either.

## 8. A CPU post-hoc recalibration of the six logits has no measurable reach (a waypoint, not the arm)

`raw/cal_proxy.json`, `code/selrank_cal_proxy.py`. DEV only, SIBLING scorers (N-s0/1/2), 4-fold cross-validation grouped by log (24 logs), objective fixed in advance: expected PDMS under a softmax relaxation of the selection, hard argmax evaluated on the held-out fold. Two families: **W5** (5 parameters: exponents on NC/DAC and the three weights) and **CAL14** (14: plus a per-head affine on every logit).

| scorer | pool (x100) | W5 out-of-fold gain | CAL14 out-of-fold gain | CAL14 in-sample | shuffled-label control (CAL14) |
|---|---|---|---|---|---|
| N-s0 | 5.07 | -0.014 [-0.255, 0.177] | +0.003 [-0.258, 0.228] | 0.202 | -0.244 |
| N-s1 | 5.08 | -0.008 [-0.227, 0.171] | +0.045 [-0.260, 0.281] | 0.311 | -0.560 |
| N-s2 | 5.08 | +0.079 [-0.056, 0.231] | +0.022 [-0.223, 0.253] | 0.213 | -0.416 |

- Out-of-fold share of the pool: **-1.6 % to +1.5 %** (upper CI end +4.9 %). In-sample, at best 0.31 points (6 %).
- Controls: identity parameters reproduce the planner pick on 3,137 / 3,137; shuffled-label fits give out-of-fold gain <= 0 on all six cells (a fit that manufactured gain from noise would be a broken probe).
- **Temperature disclosure.** The first run used tau 0.1 and was 40x wider than the median top1-top8 log-score margin (0.025), so it was near-uniform. I scanned tau in {0.003, 0.01, 0.03} on N-s0 only (`raw/scan_tau_*.json`: out-of-fold -0.02 to +0.015) and fixed 0.01 for all arms. That scan tuned on dev, so these numbers are if anything optimistic for dev; navtest was not touched.
- Consistent with `D-REFE-SELRULE-FIT` and `D-REFE-SELRULE-DAC-CONFIRM` (re-weighting the scorer's own confident heads reached 0.7 % of the DAC oracle). **The logits as given do not hold a recoverable ranking that a smooth re-weighting can reach. The arm must change what the head knows.**

## 9. Reading, and the next lever (Rule Zero)

1. **The 5.88 pool is mostly ego progress given up** (EP 77 % of the swap; the oracle proposal has EP 0.965 vs 0.796), **but the scorer gives it up because of the TTC head** (a perfect TTC head recovers 31 % of the pool, a perfect EP head 27 % with its violation signal and 14 % without; jointly 62 %). Both are the dev replicate's order in nine of nine scorers.
2. **It is diffuse** (top 10 % of tokens = 45 %; 3,790 non-zero picks have no gap), **not command-specific, and barely log-clustered.** No tail fix reaches it.
3. **A refutation, and its next arm.** A CPU recalibration of the logits is refuted as a lever (section 8). The next cheapest arm is a **head-level retrain of P2's TTC output on the scorer's own contenders**, trunk and fan frozen. Written as `PREREG_SELRANK_ARM.draft.md`. It needs a P2 forward on navtrain sets (about 7 GPU-hours in total, ESTIMATED), so it is the Master Mind's or the PI's call. Nothing was run.
4. **Blocked items, named.** (i) P2's own held-out table: needs 1.0-1.3 GPU-hours on the contended dev box (section 7). (ii) Longitudinal distance-keeping: needs lead-agent tracks passed to `families6.py`. (iii) STRATEGIC: PI pause.

## 10. Limits

- One training seed, deterministic inference: no interval here addresses another training run or inference run. The dev seed floor (0.33-0.44 points on the pick) is the only measured handle on training noise, and it is on different scorers.
- The oracle proposal is ambiguous when PDMS ties (mean 16 tied); every per-term number averages over the tied set. The bounds do not depend on this.
- Oracle-head bounds replace one head and keep the rest as predicted; a different joint replacement gives different numbers (EP+TTC is reported).
- The dev table is the u400 fan with sibling scorers; the transfer of any dev result to P2 is an ASSUMPTION.
- NAVSIM's `pdm_scorer.py` is not on this box: the file:line for the formula is INHERITED, the formula is MEASURED.
- The "cross_scorer_transfer" rows inside `cal_proxy.json` fit and score on the same sets (only the scorer changes); they are not out-of-sample and are not quoted.

## Deliverable manifest

Everything lives in `C:/Users/Admin/selrank/` (NOT in the repo, NOT staged). `paths.json` lists `[repo path, abs src, "raw"]` for each file; the suggested repo location is `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-sel-rank/`. **Integration needed: the Master Mind must land this folder.** Every file exists in ONE place (the dev box) until it is landed.

| artifact | where | reproduces |
|---|---|---|
| `RESULT.md`, `PREREG_SELRANK_ARM.draft.md`, `paths.json`, `inputs.json` | `C:/Users/Admin/selrank/` | |
| `code/selrank_attr.py` | same | `raw/attribution.json` (sections 0-5; ~45 s) |
| `code/selrank_dev.py` | same | `raw/dev_replicate.json` (section 7) |
| `code/selrank_cal_proxy.py` | same | `raw/cal_proxy.json`, `raw/scan_tau_*.json` (section 8; ~7 min, ran detached) |
| `code/selrank_families.py`, `code/selrank_family_table.py` | same | `raw/families/*.json`, `raw/families_table.{json,md}` (section 6) |
| `code/zero_tail.py` | same | vendored byte-identical (sha256 `149c3e97...`) |

## Draft GAC row (do not edit the register; the Master Mind lands it)

| **D-REFE-SELRANK-1** (2026-10-18) | **P2's 5.88-point non-zero selection pool (T1 navtest, 12,146 tokens / 136 logs) is EP in the swap (77.1 %) but TTC-bound in the counterfactual: replacing the TTC head by truth recovers +1.833 [1.569, 2.112] of it (31.2 % [27.3, 35.1]), EP +1.573 [1.349, 1.796] (26.8 %; +0.801 [0.500, 1.103] = 13.6 % without its violation signal), both +3.647 [3.378, 3.921] (62.1 %); TTC - EP on the pool +0.260 [-0.066, +0.594] (not separated).** DESCRIPTIVE, nothing fitted; navtest read, not fitted. MEASURED on P2's own full table (`table_p2_full.npz` sha256 `33441082...`), paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (another draw of LOGS only; P2 one training seed; inference deterministic). Per-term pool: EP 4.531 [4.258, 4.816], TTC 1.308 [1.073, 1.555], NC 0.032, DAC 0.000, C 0.004. Head calibration: inside the scorer's top-8, EP pairwise concordance 0.887 [0.876, 0.899] vs TTC 0.722 [0.686, 0.759]. Concentration: top 10 % of tokens = 44.9 % [42.8, 47.0] of the pool (diffuse), by command proportional to token mix. Realised EP is exactly 0 on 100 % of NC/DAC-violating proposals, so the EP bound includes a violation oracle. Dev table (3,137 navtrain held-out sets / 24 logs, 0 shared with navtest; u400 fan; nine SFT-4 sibling scorers, NOT P2) replicates the order TTC 24-26 % > EP 21-23 % in 9 of 9; training-seed floor on the dev pick 0.33-0.44. A CPU recalibration of the logits (5 and 14 parameters, grouped-by-log CV) recovers -1.6 % to +1.5 % of the dev pool: refuted as a lever. Oracle selection moves the longitudinal family away from the human (progress ratio 1.177, speed MAE 0.983 vs 0.806). STRATEGIC PAUSED. Next arm drafted, not run: `PREREG_SELRANK_ARM.draft.md` (SR-TTC). `raw/2026-10-18-sel-rank/` |

## Draft BACKLOG status line

`SEL-RANK | STEP 1 DONE 2026-10-18, 0 GPU (raw/2026-10-18-sel-rank/RESULT.md, D-REFE-SELRANK-1): pool = EP 77 % in the swap, TTC-bound in the counterfactual (perfect TTC head +1.83 [1.57, 2.11] of 5.88; EP +1.57; both +3.65); CPU logit recalibration refuted on the dev table. NEXT: SR-TTC prereg draft (head-only TTC retrain on contenders; ~7 GPU-h ESTIMATED incl. a P2 forward on navtrain train + held-out sets) | the PI's / Master Mind's GPU decision; P2's own held-out table (1.0-1.3 GPU-h) is its first gate`
