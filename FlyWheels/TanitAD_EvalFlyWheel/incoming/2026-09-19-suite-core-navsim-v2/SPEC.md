# SPEC — W1: the one-command suite core + NAVSIM v2 end to end (PRE-REGISTERED)

**Written 2026-09-19 15:40 local, BEFORE any product code and before any scoring run.**
Commissioned by the PI in chat 2026-09-19 (*"build what is missing for a one-command suite …
start now and take the risk"*); contract = `../2026-09-19-eval-suite-build/BUILD_PLAN.md` §1,
row W1 of §2, constraints §3. Owner: EvalFlyWheel W1. Exclusive write:
`taniteval/taniteval/bench/**`, `taniteval/tests/test_bench_*.py` (⛔ except the PRE-EXISTING
`taniteval/tests/test_bench_diagnostic.py`, which is not mine and is not touched — my tests are
`test_bench_suite_*.py`).

## 0. Facts measured before writing this (they shape the design)

| fact | evidence (MEASURED 2026-09-19) |
|---|---|
| ⛔ `taniteval/taniteval/bench.py` ALREADY EXISTS (the diagnostic panel, 42 KB, tracked) and is imported as `taniteval.bench` by `runner.py:39`, `refc_rerank.py:78`, `generalization.py:901`, 6 test files and 5 stack scripts; `taniteval/rerun_all.sh:7` runs `python3 -m taniteval.bench --model …` | `grep` over the repo, this session |
| A package `taniteval/taniteval/bench/` **shadows** that module (the import system prefers a package directory over a same-named `.py`) | CPython `FileFinder.find_spec` order |
| ⇒ the package MUST keep every legacy name and the legacy CLI working, or it silently breaks the diagnostic panel, the runner and six test files | design §3.1 |
| same collision for W5: `taniteval/taniteval/report.py` exists (39 KB) and is imported by `test_estimator_closeout.py:30` | `grep` |
| E1's CV CSV and E2's CV CSV are BYTE-IDENTICAL (md5 `b4b33ac29c39c37832371b8b68c9f898`, 224 lines, two separate processes, `PYTHONHASHSEED=1`) | `md5sum` of `…warmup-reference-epdms/raw/A1/devkit_2026.09.19.12.24.20.csv` and `…refcv4b-bridge/raw/score_CV_official.csv` |
| E2's STOP CSV md5 `4a15c3a1d59234e518291a1a9e1d5bff`; its `extended_pdm_score_combined` `score` = `0.3009023137456225` | `…refcv4b-bridge/raw/score_STOP_zero.csv` line 224 |
| CV `extended_pdm_score_combined` `score` = `0.1853562745165113` | E1 CSV line 224 |
| `C:/Users/Admin/navsim` is a JUNCTION to `D:\Archive\devbox-C\navsim` (so the warmup metric cache is ON D:); `C:/Users/Admin/navsim-crun` is on C: | `dir /AL C:\Users\Admin`, `fsutil reparsepoint` |
| the C: devkit copy's `dataloader.py` was already patched in place by E1 (13:04) for pooled navhard scoring; the in-process `--patch-loader` is applied on top with identical semantics | E1 `raw/navhard/dataloader_copy_patch.json`; E2 STOP manifest `original_source_sha16 f064e9ab9dcb841d` vs E1 A1 `9dd27c9e1cfdee0a` |
| the TanitAD venv has NO `jsonschema`; it has numpy 2.5.1, pandas 3.0.3, psutil 7.2.2 | import probe |
| Windows `nvidia-smi --query-compute-apps` reports `used_memory` = `[N/A]` for every process (WDDM) — per-process GPU memory is NOT observable; only the device total is | `nvidia-smi`, this session |
| A8 (`refc_v3_train.py`, PIDs 16996/21724) at step ≈2,910/5,000, ≈6.8 s/step ⇒ done-marker ≈19:30 local | `run.log` tail 15:32 |

## 1. What is built (the §1 contract, W1's share)

1. `python -m taniteval.bench <benchmark> --ckpt <path|none> --split <split> [--arms …] [--device auto|cpu|cuda]`
   with `benchmark ∈ {navsim_v2, navsim_v1, nuscenes_ol, internal_t1}`; `navsim_v2` and
   `internal_t1` implemented here; `navsim_v1` / `nuscenes_ol` are PLUGIN slots (W3 / W6) that
   refuse loudly until their module lands.
2. The run directory `taniteval/results/bench/<benchmark>/<split>/<run_id>/` with
   `bench_run.json`, `scores/<arm>.csv` (devkit output, unmodified), `artifacts/<arm>.json`,
   `criteria/<arm>.txt`, `summary.json`, `report/` (W5) — and a JSON SCHEMA for `bench_run.json`
   and `summary.json` in `taniteval/taniteval/bench/schema/`, enforced by the suite on write.
3. NAVSIM v2 end to end by PROMOTING E1/E2 code (E1: `navsim_win.py` wrapper + controls; E2:
   export, seam agent, scoring driver + guards, score parser, artifact builder + gate self-check,
   floor seam makers, bridge). Promoted files carry their origin + the origin blob; a fidelity
   test pins promoted functions to their E1/E2 originals (AST-equal) so "promoted" cannot quietly
   become "rewritten". E1/E2 package files are NOT edited.
4. Floors: **STOP and CV mandatory on every NavSim run** (added automatically, recorded
   `added_by_rule`); **ECHO** added whenever a checkpoint is given.
5. The **GPU-gap launcher**: our-model inference may use the GPU only when no training process is
   alive AND device memory used < 1,024 MiB; re-checked every 60 s; the moment either changes it
   backs off to CPU. `--device auto` ⇒ CPU unless a gap exists. Scoring is always CPU.
6. **Submission refusal**: `python -m taniteval.bench submit …` is REFUSED unless
   `--pi-approval <decision-id>` names a recorded PI decision that approves submission; none
   exists (PI 2026-09-19 *"dont submit now until I approve"*), so today it can only refuse.
7. `internal_t1` wraps `taniteval/tools/refcv3_arm.py` (the REF-C T1 harness) into the same
   run-dir contract.
8. `python -m taniteval.report <run_dir>` is W5's: bench calls it at the end only if W5's package
   form exists, else records `REPORT_PENDING` (the legacy `report.py` must never be invoked on a
   run dir — it is a different tool).

## 2. Hypotheses / claims this package makes (IDs)

* **H-W1-REPRO** — the suite, driven only through its one command from a clean shell, reproduces
  the banked official numbers of E1/E2 bit-for-bit (the harness moved into a product did not change
  what it computes).
* **H-W1-LEGACY** — the `bench` package keeps every pre-existing importer of `taniteval.bench`
  working (same objects, same CLI).

## 3. ACCEPTANCE — committed now, both outcomes

**Command (the only one that counts), from a CLEAN shell** — an environment reduced to
`SYSTEMROOT`, `PATH` (venv Scripts + System32), `PYTHONPATH="D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval"`,
`PYTHONIOENCODING=utf-8`, nothing else inherited (no `NUPLAN_*`, `NAVSIM_*`, `OPENSCENE_*`):

```
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m taniteval.bench navsim_v2 --ckpt none --split warmup_two_stage --arms CV,STOP
```

| id | criterion (literal) | PASS | FAIL means |
|---|---|---|---|
| ACC-1 | exit code 0; `bench_run.json` + `summary.json` validate against the schema; `scores/CV.csv`, `scores/STOP.csv`, `artifacts/{CV,STOP}.json`, `criteria/{CV,STOP}.txt` exist and are non-empty | all | the contract is not met — the suite is not done |
| ACC-2 | `summary.json` → `arms.CV.headline.value` **==** `0.1853562745165113` (float equality) and it was read from column `score`, row `extended_pdm_score_combined`; ×100 rounded to 4 dp **== 18.5356** (HF warmup LB `baseline_constant_velocity`, INHERITED via `NAVSIM_PROTOCOL.md` §6.3) | exact | harness drift — diagnose before anything else |
| ACC-3 | `arms.STOP.headline.value` **==** `0.3009023137456225` | exact | same |
| ACC-4 | cell level: all 224 rows × all columns of `scores/CV.csv` equal E1's banked CSV and of `scores/STOP.csv` equal E2's banked CSV, max \|Δ\| **= 0.0**, NaN pattern identical | 0.0 | a difference is NAMED (row, column, Δ), never relabelled |
| ACC-5 | file level: md5 of `scores/CV.csv` == `b4b33ac29c39c37832371b8b68c9f898`, of `scores/STOP.csv` == `4a15c3a1d59234e518291a1a9e1d5bff` | both | if ACC-4 passes and ACC-5 fails, the difference is ROW ORDER / formatting only — reported as such (partial), not as a pass |
| ACC-6 | the run refuses to start without STOP or CV if one is removed from its arm list by a code mutation (the floors are added by rule, and the validator refuses a NavSim summary missing either) | refusal observed | the floor rule is prose, not machinery |

**Unit tests (literal expectations, RED mutations)** — `taniteval/tests/test_bench_suite_*.py`:
* the summarizer's headline on E1's banked CV CSV reads `0.1853562745165113` and on E2's banked
  STOP CSV `0.3009023137456225` (literals);
* ⛔ **MUTATION M-COL**: the summarizer pointed at `pdm_score` instead of `score` must go RED — on a
  row carrying BOTH columns with different values the headline test fails, and a CSV carrying only
  `pdm_score` is refused by name;
* **M-FLOOR**: a NavSim summary without STOP (or CV) fails schema/contract validation;
* **M-GAP**: the gap predicate with its comparison inverted (or the training check removed) must
  fail the literal gap tests (training alive ⇒ CPU; 1,024 MiB used ⇒ CPU; 1,023 MiB and no
  training ⇒ CUDA);
* **M-SUBMIT**: a submission gate that returns "approved" without reading the decision register
  must fail the refusal tests;
* **M-LEGACY**: the legacy attribute delegation removed ⇒ `from taniteval.bench import run, _agg,
  _suite, diagnostic` fails; the pre-existing importer tests (`test_bench_diagnostic.py`,
  `test_runner_gate_print.py`, `test_estimator_closeout.py`, `test_ci.py`,
  `test_efficiency_levers.py`) are run before and after the package lands and their pass/fail
  counts must be identical.

## 4. What is explicitly NOT claimed

* No new NavSim number: the acceptance REPRODUCES banked ones. A warmup number carries NO interval
  (7 log groups < RG-14's floor of 8) — `{status: UNAVAILABLE, reason, n}` by construction.
* The GPU path of the gap launcher cannot be exercised on hardware today (A8/A7 own the card, and
  this package is CPU-only by brief): it is **mock-tested only** and reported UNVERIFIED-ON-HARDWARE.
* Model arms (`--ckpt <path>`) reuse E2's bridge; their end-to-end live run is heavy compute and is
  scheduled only after A8's done-marker, if at all in this package.
* Nothing is submitted anywhere.

## 5. Compute

CPU only. Unit tests and < 5-min smokes anytime; the acceptance run (≈ 15 min: E2 measured STOP
373 s, CV 533 s) is HEAVY and starts only after `C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/summary.json`
exists, ONE worker (`worker=sequential`), RAM floor 3,000 MB enforced by the promoted wrapper's
guard. NavSim experiment outputs go to C: (`C:/Users/Admin/navsim-crun/exp/tanitad_bench/`), never
to the D: junction; the navhard metric cache is the agent-free waiter's
(`C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage`), never rebuilt here.
