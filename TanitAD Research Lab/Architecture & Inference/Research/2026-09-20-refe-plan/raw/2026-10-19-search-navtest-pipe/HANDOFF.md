# HANDOFF, SEARCH-NAVTEST-PIPE (written 2026-10-20 ~01:00, Sonnet agent; READ THIS FIRST)

## State
Built and tested (CPU only; no navtest or S0 label/score/search output computed; nothing committed, pushed or `git add`ed). Package: `C:/Users/Admin/par18/navpipe/` (`DESIGN.md`, `code/`, `ref/`, `raw/`, `logs/`). Mirror tip pinned: `fb7ad9a2...` (`TIP_SHA.txt`).
* Suite: `test_navpipe.py` + `test_navpipe_s0.py` + `test_navpipe_guards.py` (119 tests at last count, all green): `cd code && CUDA_VISIBLE_DEVICES=-1 python -m pytest test_navpipe.py test_navpipe_s0.py test_navpipe_guards.py -q -p no:cacheprovider --basetemp=C:/Users/Admin/par18/navpipe/pt`.
* Modes of `code/navpipe_chain.py`: `--mode navtest|c2` (T1 route, sealed C2 amendment + admission record), `--mode s0bank --tokens <list>` (label_stepB route, sealed S0-bank amendment sha, no admission record), `--mode fixture` (carve B).

## Three detached jobs are running (no model polls them)
1. **Fixture chain** (carve B, 5 chunks, work dir `run/fixture` on C:): log `logs/fixture_chain.log` (only the LAST `ZZNAVPIPE_START fixture` block counts). Done marker: `ZZNAVPIPE_FIXTURE_IDENTICAL` (good) or `ZZNAVPIPE_FIXTURE_DIFFERS` / `ZZNAVPIPE_STOP`.
2. **Mutation check** (50 guard lines, scratch copies only): `logs/mutation_check.log`, result `raw/mutations.json`. Done marker `ZZMUT_SUMMARY`.
3. **Watcher** `watch_finish.ps1` -> when 1 and 2 have ended it runs `code/finish_navpipe.py` once (log `logs/finish.log`): verdict -> `raw/fixture_result.json`; re-runs fixture chunk 0 with the FINAL code; full suite -> `logs/pytest_final.log`; `COST.md`; EP-ADMISSION adapter layout check -> `raw/adapter_compat.json`; stages `out/<repo path>/...`, `out/paths.json`, `out/RESULT.md`; appends a status block below.

## Exact next step (a fresh agent, after `logs/finish.log` shows `ZZFINISH_DONE`)
1. Read `raw/finish_status.json` and `RESULT.md`. If `fixture` is not `FIXTURE_IDENTICAL`: diagnose from `raw/fixture_result.json` (`payload.arrays_differing` names the array; the first suspect is the thread count, `--threads 4` is Step B's setting, MM_RULING_SPC DQ8) before anything else.
2. If `raw/mutations.json` lists survivors: add a test that plants that one defect in `code/test_navpipe_guards.py`, then `python code/mutation_check.py --only <GUARD> --out raw/mut_<GUARD>.json`; merge the result into `raw/mutations.json` by hand-free means (re-run `finish_navpipe.py`, it is idempotent).
3. Report to the Master Mind in <= 300 words with the manifest = `out/paths.json`. The Master Mind lands it.

## Not done / named blockers
* The GPU stage (`make_navtest_bundle.py`, `navpipe_bridge.py`, `navpipe_gpu_stage.py`) has never run; the T1 route has never run on a navtest token. First real run = chunk 0's bridge.
* **D: is nearly full** (24 GB -> ~0 during this build); C2 and the S0 bank each need >= 9.7 GiB free on the work volume (13.4 with `--prefetch 1`). C: also ~5 GB free.
* S0 bank inputs I could not find: a plan json for the 117 logs (dev box holds DB + frames for 22/117), P2's labelled fan on the 8,922 S0 tokens (`--fan-labels`), the S0 amendment (pins listed in `NC.REQUIRED_PINS_S0`).
* The brief's "no banked P2 navtest fan" is wrong: `D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz` (use `--table-ref`; read token/proposals/logits/pick only, never pdms/sub/valid).


## FINISHED 2026-10-20 00:35
```json
{
 "fixture": "STOPPED",
 "fixture_final_code_chunk0": "SKIPPED (the main fixture did not end IDENTICAL)",
 "pytest_rc": 0,
 "pytest_tail": "119 passed in 43.29s",
 "adapter_compat": "NOT_RUN",
 "mutations": {
  "caught": 48,
  "judged": 49,
  "survived": [
   "G-DELETE-STATUS"
  ],
  "untestable": [
   "MB-IDENT-REF"
  ]
 },
 "seconds": 44
}
```
