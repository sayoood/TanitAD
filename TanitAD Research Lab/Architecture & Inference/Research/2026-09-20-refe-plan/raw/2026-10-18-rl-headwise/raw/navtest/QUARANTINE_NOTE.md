# QUARANTINE NOTE: RL-1 V1 navtest, the RAM-abort runs (2026-10-18)

Status: the files named below are quarantined, NOT read as results, and NOT staged. Only this note and the `nt_run2` trail (`score_logs/`: attempt logs, trail log, status JSON) are staged.

## What happened

Rows 1-2 are INHERITED from the Master Mind's diagnosis (HANDOFF UPDATE 23:25); the quarantined first-run logs and files were NOT opened by the continuation agent (that is what quarantine means). Rows 3-4 are MEASURED from the staged trail logs listed in the last column.

| run | window (Berlin) | tokens streamed | end | evidence |
|---|---|---|---|---|
| nt_run.py attempt 1 (INHERITED) | 22:24-22:38 | 4,433 / 12,146 | W3's RAM guard aborted the devkit (available memory < 3,000 MB; two other sessions' full stack test suites held 7.3 GB) | `quarantine_ram_abort_20261018/refe_rl1_V1.attempt1.log` (not staged) |
| nt_run.py attempts 2 and 3 (INHERITED) | 22:55, 23:02 | 0 new | aborted at once on RAM; the harness's glob fallback re-copied attempt 1's partial CSV | `...attempt2.log`, `...attempt3.log` (not staged) |
| nt_run2.py attempt 1 | 23:30-23:51 | 8,756 / 12,146 | wrapper rc 3 (RAM abort, likely a sibling agent's stream scan); the C2/C3 row checks RAN and REJECTED the partial: status FAIL, missing 3,390 tokens | `score_logs/refe_rl1_V1.attempt1.log` |
| **nt_run2.py attempt 2** | 23:55-00:23 | **12,146 / 12,146, 0 failed** | harness status PASS, C1 max abs delta 0.0, `DONE_V1` = OK | `score_logs/refe_rl1_V1.attempt2.log`, `refe_rl1_V1.status.json`, `refe_rl1_V1.nt_run2.log` |

The only scoring that is read is attempt 2 of `nt_run2.py`: CSV `score/refe_rl1_V1/refe_rl1_V1.csv`, sha256 `67905a3152c84d2160e19e82af36ea55fe22851e801399889fa6686fd1eb2a1e`.

## The two defects, and the guards that fixed them

1. A RAM-guard abort leaves a PARTIAL CSV reconstructed from the streamed rows; the harness's glob fallback re-copies that file on later attempts, so a later "success" can be the stale partial.
2. With `--out` on C: and the package on E:, `os.path.relpath` crashed (`run_v1.py:459`) before the C2/C3 row checks ran, so nothing flagged the partial.

Fix pattern, in `nt_run2.py` + `score_e2.py`: a cross-drive relpath fallback; a memory gate (psutil available >= 6,500 MB for 3 consecutive minutes) before each attempt; every stale `*.csv` in the devkit run dir moved aside before each attempt; OK additionally requires 12,146 token rows.

## What is quarantined (kept on the dev box, deliberately not staged)

* `nav/quarantine_ram_abort_20261018/runs_RECONSTRUCTED_attempt1_partial_4433rows.csv`, sha256 `d2539c5f001690703f9758524efe1943078d7b9da250f07449e3e69513baf0a0` (identical bytes were re-copied by attempts 2 and 3), and `score_refe_rl1_V1/*`.
* The partial from `nt_run2.py` attempt 1 (8,756 rows): moved aside by the wrapper as `*.moved` in the devkit run dir `E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs/refe_rl1_V1/`.

## Disclosure: two partial views of the average row were seen before the full read

| view | tokens | PDMS x100 | who / when |
|---|---|---|---|
| reconstructed CSV, attempt 1 | 4,433 | 88.44 | Master Mind, while diagnosing the first failure (HANDOFF UPDATE 23:25) |
| `nt_run2` attempt-1 status JSON | 8,756 | 88.47 | Master Mind (HANDOFF UPDATE 23:58); the RL-1 continuation agent read the same line in `score_logs/refe_rl1_V1.attempt1.log` before the full read |

V1 and its bar were frozen in `PREREG_RL1_B_COMPOSITES.md` before either view; nothing was selected, tuned or re-run on them; the bar compares to BASE with a paired CI over all 12,146 tokens. `rl2/out/raw/MM_DECISION_NAVTEST2.md` was written after the first view and before the second; its rule is keyed on the outcome category and selects nothing.
