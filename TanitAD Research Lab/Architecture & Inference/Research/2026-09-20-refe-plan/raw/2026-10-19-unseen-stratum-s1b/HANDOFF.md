# HANDOFF S1b (written 2026-10-19 18:25). A fresh agent starts here. Package `C:/Users/Admin/par18/unseen_s1b/`.

**Goal:** navtrain's OWN token filter (CV PDMS > 0.8 out; human PDMS < 0.8 out) with OUR PDM scorer; identity vs the known navtrain tokens, then apply to U. MM ruling: `par18/unseen_s1/MM_RULING_S1.md`. S1 result: `par18/unseen_s1/RESULT.md` (frozen module `unseen_s1/code/unseen_token_rule.py`, imported not edited).

## State: a CHAIN is running detached (nothing for you to launch)
`code/s1b_chain.py` (python pid in `chain.pid`, started 18:18:54; kill by explicit PID only). Stages, each a marker file:
1. IDENTITY: 20 seeded logs (seed 20261019, `raw/identity_logs.json`) of the 214 navtrain-in-val logs, ALL candidates (about 8,990 tokens), 3 workers -> `raw/identity/<log>.jsonl` + `.done`; then `analyze_identity.py` -> `raw/identity_s1b.json`; marker `STAGE1_DONE`. Rate seen at start: about 4.8 tokens/s over 3 workers => about 30 min.
2. SEQ SAMPLE: the 123 S1-provisional S3 logs (`raw/s3_provisional_logs.json`), per log S1's own seeded permutation (seed 20261020), scored until 4 tokens pass the filter and resolve 4 cameras (cap 60, then cap 120 for unfilled logs if < 1.5 CPU-h used) -> `raw/seq/`; then `make_filtered_fetch_list.py` -> `raw/fetch_list_s2_s4_filtered.{json,txt}`; markers `STAGE2_DONE`, `CHAIN_DONE`. Failure: file `CHAIN_FAIL`, `state.json` stage FAILED, `logs/chain.log`.
Check: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe code/headline.py` prints every headline number. Do not wait with model turns: if `CHAIN_DONE` is absent, END again after ensuring the pid is alive (`state.json`, `logs/identity_w*_a*.log`, `logs/seq*_w*_a*.log`). Worker exit 75 = RSS limit; the chain relaunches by itself.

## Decisions already taken (do not redo)
* Full filtering of S1's candidates is NOT within the bars: MEASURED about 0.6-1.2 CPU-s/token (cold map first token 7-36 s) x 66,402 candidates (the 123 logs) = 8-20 CPU-h (bar 2) and x 548,812 (all U) = 69-175 CPU-h (bar 4). So step 4(a) "all candidates" is NOT run. Instead stage 2 = sequential acceptance: the first 4 filter-passing tokens of S1's seeded permutation per log == a uniform draw of 4 from the filtered set, at about 30 scored tokens per log. The per-log filtered COUNT is therefore an ESTIMATE ((r-1)/(n-1) x candidates), not a count. State this plainly.
* Thresholds stay 0.8 / 0.8 (paper). Inclusivity literal: keep CV <= 0.8 and human >= 0.8; the other 3 variants are in `identity_s1b.json`. The threshold sweep is DIAGNOSTIC only, never a tuned threshold.
* The ego-progress sub-score is not stored (inside PDMS by formula); no A1/EP/endpoint/head/model/frame quantity exists anywhere in the package.
* Admissibility (MM ruling): population label only; logged future and DB map used at the population stage.

## Sources (verified, quote in RESULT.md)
* Scorer path = Step A's labeller: `l1run/code/label_perturbations.py:31-34` (label_plans) and `:79-105` -> tip `origin/agent/arch-inf-20260803` @ 5ba3f3f `refe/navsim_pdm_targets.py:87-96` (metric_cache_parts: NAVSIM MetricCacheProcessor = PDM-Closed + GT observation), `:99-129` (score_plans: PDMSimulator + PDMScorer, EP pairwise vs PDM-Closed), `refe/navtrain_scenarios.py:189-276` (one NuPlanScenario per token); env from `l1run/steps4to7.cmd` (NAVSIM v1.1 @ 3e8291b `D:/Archive/devbox-C/navsim/navsim-3e8291bf...`, REFE_SIM_HZ=10, nuplan-maps-v1.0, venv `C:/Users/Admin/navsim-crun/venv`). Copies in `tree/refe/` are byte-identical to the tip (sha256 checked).
* CV agent: pinned devkit `navsim/agents/constant_velocity_agent.py:35-36` (speed = |ego_statuses[-1].ego_velocity|), `:42-45` (poses [(i+1)*dt*v, 0, 0]), `:15` (TrajectorySampling(time_horizon=4, interval_length=0.5) = 8 poses); crun devkit same logic at `:34-35, :41-44`.
* Human: `navsim/agents/human_agent.py:35-41` -> `navsim/common/dataclasses.py:288-313` (Scene.get_future_trajectory); our DB route `sc.get_ego_future_trajectory(0, 4.0, 8)` in the rear-axle frame, MEASURED 0.0000 m vs NAVSIM's `human_future_poses` (`eval/refe_navtest_seam.py:44-47, :296-301`).
* Filter: Dauner 2024 sec 3.1 via `products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md:386-391` ("exceeds a PDMS of 0.8" / "less than 0.8").
* Scorer fidelity (PRIOR measurement, tip): `raw/2026-10-04-lane-discipline/validate_pdm_targets.json`: 300 navtest tokens, 19,500 trajectories, PDMS exact (|d|<1e-4) 94.733 %, mean |d| 0.0008, argmax same 100 %.
* S1 baselines: window rule alone recall 1.0000 / precision 0.1948 (214 logs); label-free shares navtrain 6.3 % stationary, 36.3 % turning (pooled), design-weighted 0.077 / 0.321.

## Your steps (CPU-cheap; ~15 min)
1. `headline.py`. Sanity: kept count vs known; any `n_scorer_errors`, `n_guard_skipped`, `n_candidates_not_scored` must be 0 or explained.
2. If identity precision is far below 1: read the residual blocks (`residual_false_keeps`, `clauses`, `threshold_sweep_diagnostic`, `red_arms.CVDEF_*`) and state the reasons with numbers (our scorer vs NAVSIM's: 94.7 % exact on navtest; CV definition; inclusivity; whether navtrain's list is a subsample of the filtered set). Do NOT retune thresholds.
3. Write `RESULT.md` (headline table first; evidence class per number; the deviation: sequential acceptance instead of all-candidate filtering; U counts are ESTIMATED), run `code/make_stage.py` (-> `out/repo/...` + `out/paths.json`; repo dir `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-unseen-stratum-s1b/`), stage nothing else, no commit/push, nothing on D:.
4. Hand back <= 300 words; first line: identity recall + precision (and the shuffled control), RED arm, cost per token, filtered U counts, fetch-list size in jpgs and MB.

## PRELIMINARY (18:21, 626 tokens scored, prefix of each log, NOT the result): recall 93/93 = 1.0000, precision 93/94 = 0.9894 (window-rule-only baseline on the same tokens 0.1486). `code/prelim.py` reproduces it from the partial jsonl. `out/paths.json` is INTERIM (code + partial raw): re-run `code/make_stage.py` after CHAIN_DONE.
