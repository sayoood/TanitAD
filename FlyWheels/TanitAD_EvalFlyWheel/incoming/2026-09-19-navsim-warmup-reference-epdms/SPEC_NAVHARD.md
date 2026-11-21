# SPEC addendum — navhard_two_stage (priority 2, orchestrator + PI authorisation) — PRE-REGISTERED

**Written 2026-09-19 13:12 local, before any navhard cache or scoring run.** Same harness, runtime,
wrapper and controls as SPEC.md; only the split changes. Protocol tag `EPDMS_v2_navhard_two_stage`
(the registry's OFFICIAL column).

## Expected counts (MEASURED from the local yaml, two derivations agree)
76 logs; 225 mapping entries → **450** stage-1 tokens; 2,731 pairs → **5,462** stage-2 tokens.
Cache must hold exactly 5,912 entries whose token sets equal the yaml sets (C2, C3 as in SPEC.md).

## Prediction N-CV (the external cross-check that validates the harness)
PUBLISHED, re-read from the banked primary: [N2] **arXiv 2506.04218v3** (library key `2506.04218`,
sha256 `a8431697…`), **Table 2**, "navhard leaderboard. Snapshot from 03/2026" — Constant Velocity:
**EPDMS 11.4**; S1/S2: NC 88.8/83.2 · DAC 42.8/59.1 · DDC 70.6/76.5 · TLC 99.3/98.0 · EP 77.5/71.3 ·
TTC 87.3/81.1 · LK 78.6/47.9 · HC 97.1/97.1 · EC 60.4/61.9. Leaderboard value (INHERITED,
`NAVSIM_PROTOCOL.md` §6.2, PUB-LB 2026-08-23): **11.4816**.
Our devkit `0a380a9` is on the **post-#151** side, as is [N2]v3 (whose sub-scores equal v1's while the
combined moved — `NAVSIM_PROTOCOL.md` §6.2). Readings, committed now:
* **|100·EPDMS − 11.4| ≤ 0.05** (the paper's printed precision) ⇒ the harness reproduces the published
  baseline; |100·EPDMS − 11.4816| ≤ 0.0001 would additionally match the leaderboard exactly.
* each sub-score within ±0.05 of Table 2 (1 dp) ⇒ per-term reproduction; a term outside it is named.
* **a miss > 0.5 points on the combined EPDMS is a HARNESS DEFECT** to diagnose before anything else
  (orchestrator instruction); 0.05–0.5 is reported as a partial reproduction with the delta.

## Prediction N-HUM
MEASURED before running: 40/40 randomly sampled navhard synthetic pickles have 4 frames and
`num_future_frames = 0` ⇒ the human agent is **undefined on stage 2** here too (as on warmup, where the
official runner crashed). ⇒ the human is scored on **stage 1 only** (one-stage runner, IDM-reactive,
filter ON), n = 450, and its combined EPDMS is reported UNDEFINED. C1 (DAC = TLC = TTC = LK = 1.0,
NC/DDC ∈ {0.5, 1}) must hold on every token. [N2]v3 Table 2 has no Human row, so there is no published
comparison for it.

## Controls re-run on navhard
C3 counts, C4 per-token formula (all rows), C5 aggregate recompute, C1 on the human stage-1 run, and
the external comparison above. No CI (7-log rule is warmup-specific; navhard's estimator is still the
unsettled cluster-unit question — `{UNAVAILABLE, reason, n}`).

## Addendum, 2026-09-20 09:35 — the estimator (written while N1 was still scoring, before any navhard number existed)
W2 landed the settled estimator (`taniteval/adapters/navsim_ci.py`): a **log-cluster bootstrap**
(clusters = OpenScene `log_name`, B = 2000, n ≥ 8), pre-registered before any interval existed and
reported to reproduce the devkit's own `extended_pdm_score_combined` on E1's warmup CV run exactly.
⇒ **navhard (76 logs) CARRIES an interval; warmup (7 logs) still does not** (D-BENCH-PORT).
E1 will call that adapter (never edit it) on the pre-CSV frame `N1_final_scores_frame.csv` — which my
wrapper banks after `compute_final_scores` and before `run_pdm_score.py:422` drops `weight`/`log_name`
— with `protocol = EPDMS_v2_navhard_two_stage` and W2's `mapping_key_orig_token_to_log_name` cluster map.
Committed now: the interval is reported **whatever it says**; it is an EVALUATION-SET interval only
(blind to training and inference variance, H-ESTIM-SEED-1), and the published 11.4 comparison is
judged on the point estimate as pre-registered above, not on interval overlap.

## Budget
ESTIMATED from warmup (raw/navhard_price.json): cache 3.80 CPU-h + CV 3.63 CPU-h + human ≈ 0.35 CPU-h;
run with the devkit's process pool, 2 workers (RAM-limited) ⇒ ≈ 3.9 h wall. Stop and report if the
cache's measured rate projects beyond 6 h wall.
