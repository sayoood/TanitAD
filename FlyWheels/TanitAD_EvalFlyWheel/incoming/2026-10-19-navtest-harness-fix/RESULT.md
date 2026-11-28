# NAVTEST-HARNESS-FIX -- W3 navtest driver: cross-drive relpath, stale partial csv, partial-average leak

Eval FlyWheel agent, 2026-10-19, 0 GPU, CPU tests only. Worked from `git fetch origin agent/arch-inf-20260803`
(FETCH_HEAD `d568e539f03f72de0fabe2893eeb6774743f0dd8`, `run_v1.py` blob byte-exact via `git cat-file blob`;
note `git archive` re-encodes this file to CRLF and changes its sha256 -- do not hash an archive copy).

## Headline

* **All three fixes are DONE and proven, but they live in a NEW file `code/run_v1_fix1.py`, not in `run_v1.py`**,
  because `run_v1.py`'s sha256 is on record in three places (below). `run_v1.py` is untouched
  (sha256 `125670597d99b243...`, re-verified after every run).
* **INTEGRATION NEEDED (escalation): the fixes do nothing until a caller loads `run_v1_fix1` instead of `run_v1`.**
  Nobody has been edited. Sites (3-4 lines each): `score_navtest_refe.py:64` (the R0e-pick path),
  `score_navtest6.py:42`, `score_navtest7.py:43`, `colab/navtest/linux_run_v1.py`. Deliberately NOT
  `score_navtest_thor.py:65` (Thor's byte-identical copy is the admitted one). Snippet in section 5.

## 1. Is `run_v1.py` bound? -- CHECKED, YES (attestation / manifest), so: wrapper

MEASURED at tip `d568e539`, `git grep` of the full sha256 over every tracked text file, plus the 8-char prefix:

| where | what it is | enforced at run time? |
|---|---|---|
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-04-navsim-thor-backend/raw/THOR_BACKEND_VALIDATED.json:534`, `THOR_BACKEND_VALIDATION.json:934` | Thor admission record (`ADMITTED_UNDER_POLICY`), `byte_identical_code_on_thor["W3 run_v1.py"] = 125670597d99b243...` (truncated) | no -- an attestation of what was validated |
| `colab/raw/2026-09-27-navtest-p0/code_manifest.json:117` | Colab code-bundle MANIFEST written by `colab/navtest/pack_code.py` (`w3code/run_v1.py`, 28,630 B, full sha256) | no -- evidence of the bundle that ran |
| 24 files `*.thor.json` (8 under `2026-09-28-refcv7-standard-tests`, 16 under `2026-10-04-navsim-thor-backend`) | `w3_run_v1_sha256`, stamped by `score_navtest_thor.py:119` at score time | no -- per-score provenance |

* **No launch-gate token binds it**: `stack/scripts/launch_gate.py` and `launch_gate_arms.py` contain no `navsim` / `run_v1` / `navtest` (git grep: no match).
  Nothing compares the live file to a pin. So an in-place edit would not break a gate -- it would leave one admission
  record, one manifest and 24 provenance stamps naming bytes that are no longer at that path, and would desynchronise
  the Thor copy from the repo. Your rule says that case goes in a NEW file that imports and wraps the original.
* `run_v1_fix1.load()` imports the original, **verifies the sha256 pin** (CRLF-checkout tolerated, any other drift raises
  `Fix1DriftError`), and patches the imported module object. Everything that decides a score is still W3's.

## 2. The three fixes (what, where, why)

| | defect (MEASURED RL-1 V1, 2026-10-18; INHERITED here from TRAPS.md, reproduced by the CONTROL tests below) | fix |
|---|---|---|
| **A** | `run_v1.py:459` `os.path.relpath(dst, PKG)` raises `ValueError` across drives, after the devkit finished and before C2/C3 ran | module's `os` replaced by a shim whose only override is `os.path.relpath`; on `ValueError` it returns the absolute posix path. `counts.json` gets `csv_path_kind: "abs"` (`"rel"` otherwise; same-drive value byte-identical to the original's) |
| **B** | the glob fallback (`run_v1.py:435-437`, `sorted(glob(run_out/*.csv))[-1]`) re-copies a partial left by an earlier attempt | at the start of `cmd_score`, every top-level `*.csv` in `<EXP>/runs/<label>/` **and** a previous `<RAW>/<label>/<label>.csv` is MOVED (never deleted) to `quarantine_<YYYYmmddTHHMMSS>[_n]/` beside it. Each record carries from/to, bytes, sha256, token_rows; written to `counts.json` (`stale_csv_quarantine`) and also printed on stdout so the driver log has it if the run dies |
| **C** | see audit below | when csv token rows / valid rows / average rows != expected, or the token set differs: `summary` and `summary_x100_4dp` are removed from `counts.json` AND from the printed line; `summary_suppressed {reason, csv_token_rows, n_expected}` says why; status FAIL, rc != 0 |

**C2 audit (what `run_v1.py` ALREADY does, lines 469-478):** a CSV with `len(tok_rows) != n_expected`, `len(valid) != n_expected`
or `avg_rows != 1` appends a `C2 CSV rows ...` failure; a token-set difference appends `C3 ...`; any failure makes
`status = "FAIL"` and the return code 1. **So the PASS/OK path was already guarded -- I added no second PASS gate.**
What was MISSING (lines 479-487, 513-516): the summary block is computed over the partial CSV *before* the status is
decided and is both stored and printed, so a FAIL run still printed a plausible average (the control test prints
`PDMS 72.9167` for a 4-of-6 run; the real incident printed 88.44 / 88.47). That is the only thing C adds.

## 3. Tests -- `tests/test_run_v1_fix1.py` (synthetic tmp dirs; fake devkit = a stand-in for `run_wrapped`; W3's own `cmd_score` runs end to end)

MEASURED, logs in `raw/`. Interpreters: py3.13 (`C:/Users/Admin/venvs/tanitad`) and py3.9 (NAVSIM venv, `PYTHONPATH` = v1.1 tree),
`CUDA_VISIBLE_DEVICES=-1`, one python process at a time, full suite never run.

| run | result | log |
|---|---|---|
| new file, fixed module | **20 passed** (py3.13), **20 passed** (py3.9) | `fix1_GREEN_py13.log`, `fix1_GREEN_py39.log` |
| **RED mutant A** (fallback line -> bare `raise`) | **2 failed, 18 passed**: `scn_a_sim`, `scn_a_real` | `fix1_RED_mutantA_py{13,39}.log` |
| **RED mutant B** (`quarantined = []`) | **2 failed, 18 passed**: `scn_b_stale_never_scored`, `scn_b_stale_beside_a_good_run` | `fix1_RED_mutantB_py{13,39}.log` |
| **RED mutant C** (`rep.pop(k, None)` -> `pass`) | **2 failed, 18 passed**: `scn_c_partial_prints_no_average`, `scn_c_wrong_token_set` | `fix1_RED_mutantC_py{13,39}.log` |
| existing importers of `run_v1` (`test_reconstruct_rows.py`, `test_w3_agents.py`; NAVSIM venv, documented invocation) | **BEFORE 11 passed -> AFTER 11 passed, 0 new failures** | `existing_tests_BEFORE_py39.log`, `existing_tests_AFTER_py39.log` |
| all three files together (py3.9) | 31 passed | `combined_3files_AFTER_py39.log` |

* `FIX1_MUTANT=A|B|C pytest ...` runs the *same* test file against a mutant: only that fix's two scenarios go red, the other 18
  stay green (independence). The mutation builder asserts the fix line occurs exactly once, so a mutant cannot silently no-op.
  `test_mutant_goes_red_only_where_it_should[A|B|C]` keeps the RED arm permanently in the suite.
* **CONTROLS** (unfixed `run_v1.py`, same scenarios): crashes with `ValueError ... mount` before any report is written,
  on a monkeypatched relpath AND on a real second drive letter; re-scores the planted 3-row partial (`csv_src` == the stale file,
  prints PDMS 70.8333 for it); prints the 4-of-6 average 72.9167. So each fix test is a discriminator, not a tautology.
* Non-regression: a good run through original vs fixed gives identical `status/failures/summary/csv/C1` and identical stdout line.
* Also pinned: `run_v1.py` is still the validated bytes; drift is refused; CRLF checkout accepted; quarantine never overwrites
  (`_2` suffix), leaves `quarantine_old/`, other files and bracketed paths alone.
* `test_w3_agents.py` cannot be collected under `tanitad`/`driverl-eval` (no `pyquaternion`); that is environmental and identical
  before and after -- it was run in the NAVSIM venv as its docstring says. First baseline (without `raw/CV_smoke20`) read
  10 passed / 1 skipped; re-exported the two smoke files so the skip became a real test (11 passed).

## 4. Limits and UNVERIFIED (read before relying on it)

* **UNVERIFIED against the real devkit.** No navtest was run (0 GPU, CPU tests only; a real score is long and RAM-heavy -- `nt_run2.py` gates at 6,500 MB). The fake
  reproduces the log lines and CSV format the harness greps; the real end-to-end check is the next navtest look.
* **Not upstreamed: the memory gate** from `nt_run2.py` (psutil >= 6,500 MB for 3 min before each attempt) and its independent
  `N_TOK = 12146` check. They belong to a launcher, not to `cmd_score`; keep `nt_run2.py` for them. After caller adoption
  `score_e2.py` (relpath patch) and `nt_run2.py`'s csv-aside step are redundant. `PASS` now implies `csv_token_rows == n_expected`
  (12,146 for the full split; `len(tokens)` for a subset).
* **Stale `<label>_rows.jsonl`** is not quarantined (the real wrapper truncates it at start, `navsim_v1_win.py:138`). A new attempt that
  dies before the wrapper opens it could reconstruct a CSV from the old rows; that CSV is then count-checked and, being partial,
  fails C2 and now prints no average. Residual, covered by C, not removed.
* **A deliberate re-run of a banked PASS label moves its CSV aside** (kept, recorded). Thor's idempotence guard
  (`score_navtest_thor.py:101-115`) runs *before* `cmd_score`, so a skipped banked PASS is unaffected.
* **Repointed copies fail the pin**: `score_e.py` uses a sed-repointed copy (`w3e/code`, `D:/`->`E:/`), a different byte sequence.
  `load()` refuses it unless `allow_drift=True` / `RUN_V1_FIX1_ALLOW_DRIFT=1`; `counts.json` then records
  `run_v1_matches_pinned: false` and the sha that ran. Intentional: drift must be loud.
* Python 3.9-compatible (the NAVSIM venv). `run_v1_fix1.py` must sit beside `run_v1.py` (pod/Colab bundles need both;
  `colab/navtest/pack_code.py`'s file list does not yet include it).

## 5. Integration (caller adoption, NOT applied)

```python
# was:  spec = importlib.util.spec_from_file_location("w3_run_v1", os.path.join(W3_CODE, "run_v1.py"))
#       w3 = importlib.util.module_from_spec(spec); sys.modules["w3_run_v1"] = w3; spec.loader.exec_module(w3)
spec = importlib.util.spec_from_file_location("w3_fix1", os.path.join(W3_CODE, "run_v1_fix1.py"))
fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)
w3 = fx.load()              # then w3.RAW / w3.PKG / w3.EXP and w3.cmd_score(ns) exactly as before
```

## 6. Proposed register text (NOT applied -- register files are the Master Mind's)

**TRAPS.md, footnote under the W3 entry (after the "Fix pattern" list):**
> **Fixed upstream 2026-10-19** in `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code/run_v1_fix1.py`
> (a pinned wrapper, because `run_v1.py`'s sha256 is recorded in the Thor admission record, the Colab code manifest and 24
> `*.thor.json` stamps): (1) cross-drive relpath falls back to the absolute path, `csv_path_kind`; (3) stale `*.csv` are moved
> to `quarantine_*` at run start and recorded; (4) a count/set-mismatched CSV emits no `summary`/average, printed or stored.
> Item (2), the memory gate, stays in the launcher. **Callers must load `run_v1_fix1`** (see
> `incoming/2026-10-19-navtest-harness-fix/RESULT.md` s.5); until a caller does, its runs still have the old behaviour.

**BACKLOG.md row (section A, 0 GPU):**
> | NT-FIX1 | **Adopt `run_v1_fix1.load()` in the navtest callers** -- `score_navtest_refe.py:64` first (R0e-pick), then `score_navtest6/7.py`, `colab/navtest/linux_run_v1.py` + `pack_code.py` file list; leave `score_navtest_thor.py` on the admitted copy until Thor is re-validated with `run_v1_fix1.py` shipped beside it | Without it the three fixes (landed 2026-10-19, `incoming/2026-10-19-navtest-harness-fix/`, 20 tests, RED arm per fix) are inert, and R0e-pick hits the same relpath crash and stale-partial pickup. 0 GPU; unblocked; needs only the caller edits. |

## 7. Deliverable manifest

| artifact | where | other copy? |
|---|---|---|
| `run_v1_fix1.py` (252 lines, sha256 `6e019f8c902b1ee50a2cb734cf7760bac91b2d6c16a414d627a6585ad0755f83`) | staged `C:/Users/Admin/par18/ntfix/out/code/run_v1_fix1.py` -> repo `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code/run_v1_fix1.py` | work copy `par18/ntfix/work/...` |
| `test_run_v1_fix1.py` (406 lines, sha256 `f5275db5f198e7b032614b22b9d37ed10e25bfa1014ab697b47ea271ad19b3fc`) | staged `.../out/tests/test_run_v1_fix1.py` -> repo `.../2026-09-19-navsim-v1-navtest/tests/test_run_v1_fix1.py` | work copy |
| `RESULT.md` | staged `.../out/RESULT.md` -> repo `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-navtest-harness-fix/RESULT.md` | -- |
| 11 pytest logs | staged `.../out/raw/*.log` -> repo `.../2026-10-19-navtest-harness-fix/raw/` | -- |
| `paths.json` | `.../out/paths.json` | -- |

Nothing was committed, pushed, `git add`ed, or run on a GPU/pod; `par18/rl/` and all locks untouched. Anything under `par18/ntfix/work`
is a scratch export and exists only on this disk until the Master Mind lands `out/`.
