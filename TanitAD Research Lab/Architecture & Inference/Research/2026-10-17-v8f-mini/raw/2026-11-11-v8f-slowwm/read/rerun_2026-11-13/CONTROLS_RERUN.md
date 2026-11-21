# SLOWWM read controls: four fresh re-runs on 2026-11-13, and the merge defect they exposed

**Tier T0** (planted arms and the R35H identity; no driving claim). **MEASURED**, from `controls_retry2.log` (this folder).

## What ran

The read controls (`controls_read_slowwm.py`, blob `e9940e45`) were re-run fresh four times on the dev box, each one gated at 13.5 GB free RAM or more:

| start | free RAM | a adapter identity | b basis route | c mismatch | d outcome logic | `all_pass` written |
|---|---|---|---|---|---|---|
| 01:18:17 | 13.67 GB | True | True | True | True | False |
| 01:24:51 | 14.08 GB | True | True | True | True | False |
| 01:54:56 | 13.62 GB | True | True | True | True | False |
| 02:18:06 | 14.29 GB | True | True | True | True | False |

All four controls passed on every run. They agree with the banked 2026-11-11 record `CONTROLS_READ_SLOWWM.json` (blob `be6b4f0d`, `all_pass` True), which came from the same code on the same planted and R35H inputs.

## Why `all_pass` read False

`main()` loads the existing JSON and merges this run's keys into it. Earlier that night (00:xx), a run was refused by the controls' own RAM guard (`MemoryError: free RAM … < 8.0 GB`), and that refusal wrote an `error` key. Nothing ever removed the key. Because `all_pass` requires `"error" not in res`, every later run read False even with a, b, c and d all True.

Root-cause class: **a stale key carried by a merge-write**. A refusal from an earlier run outlived the run that produced it.

## Fix (this land)

- `controls_read_slowwm.py` → blob `f7b3ec70`. `main()` moves a stale `error` to `error_previous_run` before the run starts, and a control that raises in this run sets `error` again. The change is 5 inserted lines and nothing else.
- Two new tests in `tests/test_read_slowwm.py` (31 inserted lines):
  - `test_controls_main_a_stale_error_from_an_earlier_run_does_not_survive_a_passing_run`
  - `test_controls_main_red_arm_an_error_in_this_run_still_fails`
- **Mutation:** with the pop replaced by a no-op, the first test goes **RED** and the red arm stays green. The file was restored afterwards and verified by blob hash.
- The package's light tests: **22 passed**, with the 2 heavy RAM-gated tests deselected because the planner was using the RAM. On 2026-11-11 the full file read 22 passed, before these 2 were added.

## The record the read uses

At 09:53:57 the retry script's deadline fallback restored the banked record byte-for-byte (`git hash-object` = `be6b4f0d`). `report_slowwm.py` reads that record (`all_pass` True). The four fresh runs above are independent confirmation of each of its four controls. No control result changed.
