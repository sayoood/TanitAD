# PARSE8 — refcv8 NavSim summary parser (EvalFlyWheel, 2026-10-20, STAGED, nothing committed)

`code/parse8.py` = a port of refcv7's `parse7.py`. **Statistics unchanged** (official `score` column, E2's `s2_group_uniform`, `navsim_ci`
log-cluster bootstrap B = 2,000 / seed 0 / 95 % percentile on the mapping key's `log_name`, paired variant, stop fraction, seed-floor rule).
Changed: arm names, paths, fail-loud handling of a missing arm. It is the producer for `navsim_bars8.navhard_bar` (BAR-R8-N2) and `warmup_bar` (BAR-R8-N3).

**Reads** (per split): `<run>/scores_<split>/score_<arm>[__navhard_two_stage].csv` + `.counts.json` (must read `PASS`) +
`score_<tag>_wrapper/<tag>_final_scores_frame.csv` (navhard only), `<run>/bridge_<split>/seam_<arm>.npz|.manifest.json`, the split's
`navsim_agent_inputs.json`, and the model-free floors (`CV_official`, `STOP_zero`, `ECHO_ha0_ext`) from `--floors`
(default: first existing `…/2026-09-28-refcv7-standard-tests/navsim/raw/floors/<split>` — INHERITED refcv7 floors, same token set, model-free).
**Writes** `summary_navhard.json` / `summary_warmup.json` (LF endings). Interval/estimator/seed-floor semantics: see the parse8 docstring.

**Arm map R7 → R8:** `R7_A1→R8_LEGAL` · `R7_A1_s1→R8_LEGAL_s1` · `R7_BLIND→R8_BLIND` · `R7_A1NT→R8_NT` (optional) · `R7_NAVOFF→R8_NAVOFF` (optional) ·
new optional `R8_PRIV`, `R8_RT` · unchanged `PRIOR_ha0p`, `CV_official`, `STOP_zero`, `ECHO_ha0_ext` · dropped (R7-recipe only) `R7_VMAXOFF`, `R7_FILTOFF`, `R7_CEILDECL_d`.
Expected per split — navhard: LEGAL, LEGAL_s1, PRIOR, 3 floors; warmup: + R8_BLIND. An expected arm with no CSV, or beside a failed count guard, reads
`{"status":"UNAVAILABLE","reason":…}` and so does every pair / interval needing it; `seed_floor_S2_EPDMS_u` reads `null` (+ `…_status`). parse7 skipped silently.

**Seed-1 mirror rows (A-R8-NS1 AMENDMENT 3, bars hold on both inference seeds):** `pairs` and `paired_intervals` also carry
`R8_LEGAL_s1__minus__{STOP_zero,CV_official,ECHO_ha0_ext}` — same function and record shape as the R8_LEGAL rows (`delta`, `lo`, `hi`, `separated`,
`separated_log_name`, `sensitivity_nuplan_drive`); R8_LEGAL_s1 absent → UNAVAILABLE with a reason. **Warmup:** parse7 emits the R8_LEGAL rows there as `pairs` only
(point deltas, W/T/L; 7 logs < 8 ⇒ no intervals), so the mirror rows on warmup are `pairs` only too. The seed-0 rows are unchanged.

**Control (MEASURED, 40 + 8 tests green, `raw/pytest_parse8/test_parse8.green.txt`, `test_mutation_parse8.green.txt`):** parse8 over refcv7's banked step-30000 inputs
(`D:/…/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step30000/`, R7 stems read as the mapped R8 arms via `--control-r7-names`) reproduces the banked
`summary_navhard.json` / `summary_warmup.json` **exactly** — arms, pairs, intervals, paired intervals, seed floors; the only diffs are free-text `what` labels and
parse8's additive provenance. E.g. navhard official two-stage EPDMS LEGAL 0.2269259… vs STOP 0.2985321…, interval 0.2269 [0.1922, 0.261] (76 logs, 225 keys), paired vs STOP
−0.0716 [−0.1112, −0.0346]; warmup S2-EPDMS-u LEGAL 0.52237 vs STOP 0.52125, seed floor 0.006928, BLIND 0.21933. Through `navsim_bars8` that control reads
N2 = FAILED, N3 = NOT PROVEN with the BLIND panel OK (non-UNAVAILABLE; refcv7-30k numbers, **not** refcv8's score). The control exercises the PARSER, not the model.
**IDENTITY control for the mirror** (real data): R7_A1 files read as BOTH R8_LEGAL and R8_LEGAL_s1 → `R8_LEGAL_s1__minus__{STOP,CV,ECHO}` equals the banked `R7_A1__minus__…` record
(e.g. vs STOP: −0.0716 [−0.1112, −0.0346], drive level [−0.1247, −0.035], 34 drives; `raw/pytest_parse8/control/identity_control_s1_equals_s0.txt`). It cannot see a mirror that reuses seed-0 rows
(identical by construction), so that defect is covered by the synthetic s0 ≠ s1 tests.
Mutations (`PARSE8_MUTATION`: `swap_seed_arms`, `drop_blind_arm`, `skip_count_guard`, `silent_skip_missing`, `ignore_cv_standin`, `s1_pair_reuses_s0`) each turn named literal tests RED (`raw/pytest_parse8/mutation_*.red.txt`).

**Run after the 48,852 navhard/warmup rolls exist** (Git Bash; `O` = the run's `--out`; use `--label RESULT` only if `$O/INDEX.json` says `RESULT`):
```
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; P=<repo>/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim; O=<…>/step48852_ema
F=D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/floors
$PY -X utf8 $P/code/parse8.py --split warmup_two_stage --scores $O/scores_warmup --bridge $O/bridge_warmup --floors $F/warmup_two_stage \
  --inputs D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/raw/navsim_agent_inputs.json --label RESULT --out $O/summary_warmup.json
$PY -X utf8 $P/code/parse8.py --split navhard_two_stage --scores $O/scores_navhard --bridge $O/bridge_navhard --floors $F/navhard_two_stage \
  --inputs C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json --csv-suffix __navhard_two_stage --label RESULT --out $O/summary_navhard.json
$PY -X utf8 $P/code/navsim_bars8.py --run-dir $O --tag step48852_ema --out-dir <dir> --navhard-summary $O/summary_navhard.json --warmup-summary $O/summary_warmup.json
```
Then **assert on the artifacts, not the exit code**: both summaries exist; `arms_refused == {}`; no arm has `"status": "UNAVAILABLE"`; `seed_floor_S2_EPDMS_u` is non-null; `intervals.R8_LEGAL.status == "OK"` (navhard).
Tests: `pytest -q -p no:cacheprovider tests/test_parse8.py tests/test_mutation_parse8.py` (CPU, ~4 min; `PARSE8_BARS8_CODE` points at `navsim_bars8.py` if not beside parse8).

**Notes for the Master Mind:** (1) N2's `separated` rule (record `separated` AND `lo > 0` AND `sensitivity_nuplan_drive.lo > 0`) and the both-seeds evaluation are ruled (SPEC_REFCV8 l.651-652, §6.2, AMENDMENT 3); parse8 now supplies every field that rule reads, for both seeds. (2) The bars package is landed in git (5310735); its local copy `landed_hold_r8nsbars` (the earlier `hold_r8nsbars` is gone); the tests search both, and import is read-only (parse8 never edits bars8). (3) Floors are the refcv7-era model-free arms (INHERITED, `FLOORS_PROVENANCE.json` sha256 recorded in `_parse8`).
