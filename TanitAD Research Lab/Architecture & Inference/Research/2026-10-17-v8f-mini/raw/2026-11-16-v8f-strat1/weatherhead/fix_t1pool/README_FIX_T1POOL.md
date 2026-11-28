# Weather head form (a): the T1 pooled-head dump moves into a wrapper (2026-11-17)

## What failed
* **The pod full G-SUITE of commit 2b61be9d read 28 regressions, all in `tests/test_v8f_t1_speedsub.py`** (MEASURED, pod `/root/strat1/runs/gsuite_wh.json`).
* `v8f_t1_speedsub.py` pins the sha256 of `v8f_t1_eval.py` at 180044eb. The build had edited that file (+14 lines for `--ctx-pool`).
* The same class of failure hit the STRAT1 build (its 28 speedsub regressions). The rule now in memory: a study flag goes into a wrapper, never into a pinned shared evaluator.

## The fix: commit c02bf3a6 on `agent/v8f-whead-20261116` (parent 2b61be9d)
* `stack/scripts/v8f_t1_eval.py` equals its a4357e79 blob again (sha256 180044eb).
* New `stack/scripts/v8f_t1_eval_pool.py` imports the tool unchanged and adds three things:
  * `run --ctx-pool off|on`, refused when it differs from the checkpoint cfg;
  * `ctx_pool_logits` among the planner's head keys (from the same `forward_infer` call);
  * one `ctx_pool_logits` row per tick with `--dump-heads`.
* With `--ctx-pool off` nothing of the tool is patched. The wrapper refuses to run when the tool sha256 differs from the pin, and it undoes every patch when the call returns or raises.
* `stack/tests/test_v8f_weatherhead.py` gains these tests:
  * the tool bytes equal the speedsub pin;
  * the refusal holds both ways;
  * the patches apply only inside an `on` call, and are restored after a raise too;
  * the recorder writes the rows.
* Four mutations of the wrapper each turn a test RED: the recorder append dropped, the refusal dropped, the restore dropped, the head key not added.

## Gate records of the c02bf3a6 tree (tree manifest ac1bcd9e)
* **G-SUITE, full scope, on the pod.**
  * Tree new8 = new4 plus the 5 files of `wh5b.sha256`, with `v8f_t1_eval.py` checked at its pin. 8 shards in the base composition, compared with junit/new6 (a4357e79).
  * **14,941 tests, 0 regressions, 31 new tests all passing.** `gsuite_wh2.json` is the pod comparison; `gsuite_result.json` is wh_check's own (`wh_gsuite_from_pod2.py`), and the two agree.
* **G-EVAL, both nav families** (`geval_text.json`, `geval_v9.json`):
  * T0 on the 2,399-window grid: max abs diff 1.31e-6 (bar 1e-5).
  * T1 tick 0 through the wrapper: 0.0.
  * Both evaluators refuse a flag that disagrees with the checkpoint, with the refusal message checked (the T1 refusal and the message check are new in `wh_check.py`).
  * Regression arm: a one-frame-shifted window reads 0.140 / 0.138, so FAIL as required.
* **Tokens:** all 13 arms minted into the run folder `run2` (`mint_run2.log`).

## Files here
* `wh_run.py`, `wh_check.py`, `make_manifest.py`: the build scripts as they minted the run2 tokens. They differ from the copies in `../build/` by the T1 path (the wrapper), the T1 refusal check in G-EVAL, and the manifest's file list.
* `job_wh2.cmd`: the launch job for run2.

## Launch state
* **Not running.** The first launch stopped before any fit on a missing `run2/logs` folder, which now exists.
* The relaunch was refused by the auto-mode classifier as training without an explicit PI go, so it waits for the PI.
