# RJ5 navtrain bulk re-roll: the other 877 logs (launched 2026-11-13, not finished)

Master Mind, 2026-11-13. Status: two detached chains launched ~13:15Z on the dev box. Both start rolling when the local labelling job (`label_local/`) has ended.

## Why
- The RJ5 teacher (jerk 5.0) was re-rolled on the 198 navtrain logs held on the dev box (16,133 rows, `../local_roll/RJ5_LOCAL_ROLL.md`). The other **877** of the bank's 1,075 navtrain logs have no local DB.
- **PI decisions, 2026-11-13 (verbatim):**
  - "you can download": the nuPlan DBs from the public bucket.
  - "dont use box2 it is reserved for flagship v8": box 2 is dropped, so `../box2/` is not used.
- **Uses:**
  - the full RJ5 student label table, for P5-C and the next student;
  - a navtrain development set for the REPAIR-D successor. TF-N5 showed the navtest residual cannot be used for that, because those tokens have been read twice.

## Fetch smoke (MEASURED 2026-11-13)
- **Result:** 1 log, 0.08 GiB compressed → one 145,805,312 B DB in ~80 s (`NAVTRAIN_DBS_OK`). The smoke DB was then deleted.
- **First attempts failed** on two dev-box path defaults:
  - the navtrain split YAML (`REFE_NAVTRAIN_YAML`, taken from `run_local.sh`);
  - the zip central-directory cache on a non-existent `D:/` (`REFE_CD_CACHE`, now `F:/tanitad_scratch/rj5nt/cdcache`).
- **Plan for the 877 logs:** found 877 / 877 in the train zips; 80.94 GiB compressed → **135.55 GiB inflated**.
  - Average per log: 0.155 GiB.
  - The `train_vegas_6` logs average 0.49 GiB each (67 logs).

## Defects fixed before launch (in `../rj5_chain.py`, tests in `../test_rj5.py`, 12 pass: `test_rj5.log`)
1. **`--min-free-gb` was declared and never read.**
   - Same class as "built, tested and unreachable": the docstring promised a disk guard that did not exist.
   - **Now:**
     - before a fetch, the batch is refused (rc 6, DISK-LOW) if free space minus `--est-gb-per-log` × (logs to fetch) is below `--min-free-gb`;
     - after a fetch, if free space is below the floor, the batch's DBs are deleted and the chain stops (rc 6, resumable).
   - **Tests:**
     - `test_disk_guard_reachable_from_main` drives `main()` through a stub fetch script and requires rc 6 with no fetch;
     - its mutation (guard bypassed) must run the fetch, and the post-fetch check still stops it at low space.
2. **The control pick used the index within the batch.**
   - With batches of 2, that would have rolled R0 on half the logs.
   - It now uses the global log index (one control per 20 logs). Its test fails under the old rule.

## Launch (both detached via WMI `cmd /c ... < NUL`)
| chain | logs | DB staging | floor (`--min-free-gb`) | work dir | launcher |
|---|---|---|---|---|---|
| A | `logs_fetch_A.txt` (439, first half sorted) | `C:/Users/Admin/rj5db` | 4.8 GiB | `F:/tanitad_scratch/rj5nt/run_fetch_A` | `launch_fetch_A.cmd` |
| B | `logs_fetch_B.txt` (438) | `F:/tanitad_scratch/rj5nt/dbB` | 20.6 GiB (F: carries the 20 GB pinned-run floor) | `F:/tanitad_scratch/rj5nt/run_fetch_B` | `launch_fetch_B.cmd` |

- **Shared settings:** `--batch-logs 2 --procs 2 --arms RJ5 --control-every 20 --est-gb-per-log 0.35`.
- **The halves are disjoint:** their union equals `logs_fetch_877.txt` (checked).
- **B waits for F: to reach 21.5 GiB free.** F: read 19.7 GiB at launch, held mostly by other sessions' data. The HB-fix agent's ~2 GB scratch goes when that agent ends.
- **Why staging is spread over two drives:** C: 6.6 GiB, E: 5.2 GiB and F: 19.7 GiB free. No drive can hold more than a couple of DBs above its floor.
- **Runtime (ESTIMATED from the local roll, 12 logs per ~16 min with 2 rollers):**
  - about 2.5 to 3 min of rolling per 2 logs, plus about 1 min of fetch;
  - chain A alone: about 13 h; with both chains running, about 13 to 15 h in total.
  - The dev-box CPU read 100 % load at launch (other sessions), so it may be longer.

## Next (MM)
On `ZZRJ5_CHAIN_DONE` in both chain logs:
1. `rj5_merge.py --rows-dir run_local/rows --rows-dir run_fetch_A/rows --rows-dir run_fetch_B/rows`.
2. Label (`rj5_label.py` → `p5c_dual_labels.py`).
3. Bank to HF.
4. Delete the work dirs.

The R0 control is read on `teacher_device == cpu` bank rows only, as for the local roll.
