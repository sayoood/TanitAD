# HANDOFF - DAG-ORC (DAG-OACC base) r2 chain build, rewritten 2026-10-28 (after the Master Mind's stream specification)

State: BUILT, GENERATED, SELF-TESTED. Nothing is in flight. Nothing installed into box2_ship, box 2 never contacted, no git write. Read `NOTES.md` (every difference from r1 with file:line; section 6 = the result), `LAUNCH.txt` (install + launch), `DRY_RUN.out` (the offline runs).

## Result (MEASURED, `logs/selftest_chain_r2.out`)
`SELFTEST: 66/66 tests pass; MUTATION: 223/223 mutants killed, 0 survivors`, `ZZSELFTEST-R2-END PASS`, `ZZWRAPPER-END rc=0`. Re-run (detached, ~10 min): `Invoke-CimMethod Win32_Process Create 'cmd /c E:\Projects\TanitAD-artifacts\dagorc_r2\gen\run_selftest.cmd < NUL'`, then `grep -n "^FAIL\|^SURVIVOR\|Traceback\|ZZSELFTEST\|ZZWRAPPER" logs/selftest_chain_r2.out`.
Debug slices: `gen/mk_dbg_x26.py` + `dbg_x26.py` (X2, X4, X6: ~2 min), `mk_dbg_x3.py`, `mk_dbg_x45.py`.
Regenerate after any change: the generators in `gen/` (never hand-edit `box2_ship/*`): `make_chain_r2.py`, `build_r2_lib.py`, `build_r2_checks.py`, `build_r2_asserter.py`, `build_r2_small.py`, then `python make_chain_r2.py --check`, `python mk_selftest_r2.py`, the self-test, `python make_dry_run.py`, `python make_manifest.py`.
(The shell tool double-unescapes `\\n` in heredocs: write generator edits with the Edit / Write tools.)

## The registered stream specification (Master Mind, 2026-10-28)
* `STREAM_PATTERN='stack/tanitad/data/v8f_mini_data.py|^EGO_PERTURB_COMPOSED_STREAM = 1[[:space:]]*$'` (fails on e7ea20d1 and on today's tip c7963d7, on the constant at 0 and on a commented-out constant; passes on the staged fix).
* `STREAM_RECORD='levers.ego_perturb_stream=1;0'`: the int 1 on A, the int 0 on REP, absent fails, in summary.json AND selfcheck.json (run record) and in the gate selfcheck (new asserter `gate_dagoaccorc_r2_assert.py`, which is r1's + one clause).

## Exact next steps (the Master Mind; none is blocked on a model decision)
1. The trainer fix (staged at `E:/Projects/TanitAD-artifacts/dagland/compose_fix/stage`) must be LANDED on `agent/arch-inf-20260803` (remote tip is still `c7963d7`). It must not change the registered argv or `:dagoaccorcarm` (the staged one adds no flag and no import: NOTES 3).
2. `LAUNCH.txt` 0e: `--check-tree` on the landed tip must print `every one of 43 patterns`; `--dry-run` -> 4 SAME, 4 ONE-LEVER, no `!!`; `--wait-only` -> `ZZOAF-WAIT-GO`.
3. Install the 6 files of `LAUNCH.txt` section 1 (note the NEW `gate_dagoaccorc_r2_assert.py`), launch, run `wake_v8fdagoaccorc_r2.sh` as a script poller.
4. ORDER HAZARD (NOTES 4.1): launching BEFORE the fix is landed renames r1's artifacts aside (idempotent) and then stops at the lever grep. Do not launch until step 2 passes.

## Open items (not decidable here)
* Whether the fix that LANDS is the staged file (only `--check-tree` on the landed tip decides).
* No fake-box end-to-end of `gate_one` / `runone` (as in r1); the aside, predecessor wait, lever grep, refusals, dry-run and every added check ARE exercised for real or against real records.
* The real fixture records are test-scale (6345 eligible); K-REACH at DAG-OACC scale decides the 0.30 +- 0.02 share on the real runs.
