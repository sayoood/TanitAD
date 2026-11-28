# PREREAD_V8F_STOPDEC: on DAG-OACC, does T's own tactical decision see the stop while its speed cells miss it? (FROZEN 2026-11-10 by the Master Mind; no full-read number exists)

Status: FROZEN (drafted by a Sonnet agent; reviewed by the Master Mind against `stopdec_read.py`: thresholds, the NOT-EVALUATED rule and the both-seeds verdict match this card; no text changed at freeze). Zero training, CPU. Instrument `theory/stopdec/stopdec_read.py` (imports `theory/speedsrc/speedsrc_read.py`: same loop, window set, strata and controls). Only the 64-window smoke has been run. The sha256 of this card and script go in `PREREAD_V8F_STOPDEC.sha256`; every threshold is fixed in both.

## Why
SPEEDSRC (committed outcome DIFFUSE; `theory/speedsrc/out/SPEEDSRC.json`, MEASURED) found, beside its bar and NOT promoted, that windows whose GT goal is STOP_POINT are ~5 % of windows but ~20 % of T's squared speed-error mass (ratio ~4), and GT-speed-only tactical ADE gain there is -1.8 / -1.55 m versus ~-0.3 elsewhere; GT regimes STOP and DECEL point the same way. That pattern is post-hoc. This card asks the new, pre-registered question: in those windows, does T's OWN decision already say "stop", so the lever is coupling (decision to speed cells), or does it miss, so the lever is perception of a stop cue?

## Arms
`PARITY_DAGOACC_s0`, `PARITY_DAGOACC_s1` (checkpoints as banked, one environment, no training). Both seeds must agree.

## Definitions fixed from source (read tree `F:/TanitAD-scratch/otband_read/tree`)
* `forward_topk` returns `code = base["code"]` of `forward_infer` (`stack/tanitad/models/v8f_mini.py:1144-1150`), built by `_code_pred` (`:890-892`): `[goal_prob(22), softmax(lat_logits)(5), softmax(lon_logits)(7), cons_std(26)]`.
* **Goal part, code[:, 0:22]: a per-goal SIGMOID, not a one-hot or softmax** (`out["goal_prob"] = torch.sigmoid(out["goal_logits"])`, `:968`; trained by masked BCE, `:1169-1170`). Goals are independent. **"T predicts STOP_POINT" = `code[:, 6] >= 0.5`** (`STOP_I = 6`, `v8f_mini.py:46` on the tip; index into `GOAL22`, `stack/tanitad/data/v8f_mini_data.py:52-55`). 0.5 is the threshold of the GT mask (`goal_y > 0.5`). Decision: `goal_stop`.
* **Longitudinal part, code[:, 27:34]: 7-way SOFTMAX** over `LON9 = ("HOLD","CREEP","STOP","FOLLOW","DECELERATE","ACCELERATE","KEEP")` (`v8f_mini_data.py:57`; `N_LON = 7` at `v8f_mini.py:42`). It HAS stop and decelerate classes. Decisions on its argmax: `lon_stop` = STOP; `lon_slow` = in {HOLD, STOP, DECELERATE} (CREEP is not slow).
* Groups (SPEEDSRC strata, thresholds exactly as its card): `STOP_POINT` = tlstop stratum STOP_POINT (GT goal STOP_POINT, no traffic-light goal); `REG_STOP` = `v_end < 0.5 m/s`; `REG_DECEL` = not STOP and `dv < -1 m/s` (`v2` = mean of knots 8-11, `v_end` = mean of knots 26-29). Windows: `lab_t.has` and >= 1 present speed cell. A smoke note: a GT STOP_POINT goal that also has a TL goal is excluded from `STOP_POINT` by the SPEEDSRC rule.
* Per window speed error `mse_w` = mean over present speed cells of `(own - gt)^2` in (m/s)^2, `mass_w` their sum (as SPEEDSRC).
* Per (group, decision): HIT RATE (hit windows / group windows); FALSE-POSITIVE RATE (decision true in the complement / complement windows); `mse` of hit windows and of miss windows; RATIO = mean `mse_w` of hit windows / mean `mse_w` of the complement; MISS MASS SHARE = `mass_w` in group-miss windows / `mass_w` in the group.

## Estimator
`taniteval.ci.stat_cluster_bootstrap`, B = 2,000, seed 0, resampling the eval-141 episodes. **It answers "another draw of episodes" only**: not "another training run" (two seeds are the only look at that), not "another inference run" (the planner is deterministic). 9 (group, decision) cells are reported; only the primary cell below decides; the other eight are beside, no bar, no multiplicity correction.

## Committed outcomes (primary cell: group STOP_POINT, decision `goal_stop`; both seeds)
A seed needs >= 50 STOP_POINT windows, else NOT-EVALUATED.
* **DECISION-SEES:** hit rate (point) >= 0.6 AND ratio (point) >= 2.0 with its 95 % interval lower bound > 1. T knows it will stop but its speed cells do not follow. Trigger: a coupling card, tying T's speed cells to its own stop decision in training.
* **DECISION-MISSES:** hit rate (point) < 0.4 AND miss mass share (point) > 0.5. The speed error sits where the decision also misses. Trigger: a stop-cue perception card.
* **MIXED:** otherwise (both cards drafted, the larger error mass first).
* Both seeds the same class = that outcome; different classes = SEEDS-DISAGREE (reported, no trigger).
* **Beside, no bar:** the same table on STOP_POINT x `lon_stop`/`lon_slow`, and on REG_STOP and REG_DECEL x all three decisions; the lon argmax census per group; mean STOP_POINT probability per group.
* Tier stamp: T1-kin first tick, oracle diagnostic, never driving performance. STRATEGIC: PAUSED (PI item 25), reported as paused, never as a pass.

## Controls (the read is INVALID if one fails)
1. Checkpoint sha256 equals the bank manifest where one exists.
2. TOP1 equals the dump `ol` (max abs diff <= 1e-4 m); window set, GT future `g` and `v0` equal the dump's.
3. IDENT equals TOP1 (<= 1e-5 m); OWN-ALL equals IDENT (<= 1e-6 m).
4. Goal probabilities in [0, 1]; lon probability rows sum to 1 (max deviation <= 1e-5).
5. Memory: 4 GB free before an arm, 3 GB before a chunk; `OMP_NUM_THREADS=2`, CPU only, one python read at a time.

## Log
* 2026-11-10 DRAFT written; smoke only (64 windows; dump-side controls need the full read).
* 2026-11-10 FROZEN by the Master Mind before the full read; sha256 in `PREREAD_V8F_STOPDEC.sha256`.
