# NT-FIX1 integration (Master Mind, 2026-10-19)

The agent's `run_v1_fix1.py` is inert until a caller loads it (its own escalation, `RESULT.md` §5). The Master Mind adopted it in the R0e-pick / head-confirm navtest path in the same landing.

## Verification of the agent's package (MEASURED, on the staged bytes over a byte-exact tip export of `2026-09-19-navsim-v1-navtest/`, `cat-file --batch`)
* `tests/test_run_v1_fix1.py`: **20 passed** (py3.13, `venvs/tanitad`, `CUDA_VISIBLE_DEVICES=-1`).
* RED arms `FIX1_MUTANT=A|B|C`: **2 failed, 18 passed** each, matching the agent's logs.

## Caller adoption: `TanitAD Research Lab/.../2026-09-20-refe-plan/eval/score_navtest_refe.py`
* Its sha256 is pinned nowhere at the tip (full-sha `git grep`: 0 hits). It is the scorer called by `refe_head_confirm.py` and the REFe navtest arms.
* Edit: lines 64-67 now load `W3_CODE/run_v1_fix1.py` and call `fx.load()`, which byte-verifies `run_v1.py` against `125670597d99b243...` and applies fixes A/B/C. A MISSING `run_v1_fix1.py` exits loudly; it NEVER silently falls back to the unfixed module. The docstring was updated.
* ⚠️ `W3_CODE` is `D:/Projects/TanitAD/FlyWheels/.../code`, i.e. the D: working tree. That tree must be synced to include `run_v1_fix1.py` before the next REFe navtest; until then the caller refuses, by design.
* New test `eval/test_score_navtest_refe_fix1.py` (3 tests: the caller reaches `FIX1` and not `UNFIXED` through a fake W3 dir; a missing fix1 file is loud; the source names fix1 exactly once and no longer execs `run_v1.py` directly):
  * fixed caller: **3 passed** (`raw/integ_fixed.log`);
  * RED arm (`SCORE_NAVTEST_REFE=<pre-NT-FIX1 caller>`): **3 failed** (`raw/integ_red.log`).
* The only other test naming the caller (`test_refe_head_confirm.py:961`) checks the `refe` label rule, which is unchanged.

## Not adopted (BACKLOG NT-FIX1)
* `score_navtest6.py:42`, `score_navtest7.py:43`, `colab/navtest/linux_run_v1.py` and the `pack_code.py` file list are still on the unfixed module.
* `score_navtest_thor.py` deliberately stays on the admitted copy.
* `score_e.py`'s sed-repointed `w3e/code` copy fails the pin by design (`RUN_V1_FIX1_ALLOW_DRIFT=1` to override, recorded in counts.json).
