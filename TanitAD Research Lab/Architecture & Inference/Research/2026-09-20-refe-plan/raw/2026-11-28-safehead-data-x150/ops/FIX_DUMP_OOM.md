# x150 dump out-of-memory + the cascade it caused (MM, 2026-11-28)

## What happened (MEASURED)
- **10:57:43Z, `sh_dump`:** `cv::OutOfMemoryError`, `Failed to allocate 6220800 bytes`, on log `2021.08.31.16.37.21_veh-40_01247_01379`.
  - It was in a 43-log batch, after 31 dumps had succeeded.
  - `sh_run` stopped the whole chain on that one failure and wrote `chain.FAILED`.
- **The cascade:** the downstream chains each stopped as designed, one after another:
  - `seal_185.sh` wrote SEAL.FAILED (10:57:54Z);
  - `ds_step2.sh` wrote STEP2.FAILED (10:59:27Z);
  - `ds_read_chain.sh` wrote DSREAD.FAILED (11:03:34Z).
- **Commit memory at 11:38Z:** 3.9 GB free of 62.3 GB.
  - None of the large consumers is ours: chrome 13.1 GB, other sessions' python (a reconstruction server 1.6 GB, three `wua_a1.py` at 0.8 GB each) and the Claude apps 6.8 GB.
  - Commit exhaustion is therefore the **likely** cause of the ENS seam's 0xC0000005 at 08:49Z as well. That is a HYPOTHESIS; `ens_memtrace.tsv` will test it.

## Fixes (procedure only; no dump, label or merge arithmetic touched)
1. **`sh_run.py`:** dump batches are capped at `SH_DUMP_BATCH` logs (default 8), so the memory gate is re-checked often.
   - A failed batch is retried after 120 s plus the memory gate.
   - The chain stops only after `SH_DUMP_RETRIES` (default 4) consecutive batches that produce no new dump. Dumps are resumable per log.
2. **Markers set aside, never deleted:** x150 `chain.FAILED` and run1 `{SEAL,STEP2,DSREAD}.FAILED` became `*.attempt_1`.
   - Relaunched, all waiting: x150 (pid 69148), seal_185, ds_step2 and ds_read_chain.
3. **`ens_relaunch2.sh`:** the ENS seam runs **alone**.
   - Why: it is 8.4 h and cannot resume, while x150 can resume per log.
   - Steps:
     1. Wait for x150 to finish the 49 logs of the archives already extracted (cap 3 h).
     2. Stop x150 top-down.
     3. Relaunch ENS.
     4. Relaunch x150 once `ENS1.seam.DONE` exists (ctrl and score are CPU).
   - It traces commit every 60 s, and retries once only if commit never fell below 1,500 MB.
