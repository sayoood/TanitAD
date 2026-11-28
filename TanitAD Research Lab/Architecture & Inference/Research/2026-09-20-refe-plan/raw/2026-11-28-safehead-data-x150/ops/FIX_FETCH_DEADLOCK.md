# x150 fetcher deadlock fix (MM, 2026-11-28)

## Symptom (MEASURED)
- x150 was relaunched at 07:59:09Z, after it had yielded to ENS. It then wrote **nothing for 85 min**.
- F: had 4.4 GB free.
- 51 logs from 15 archives were already extracted on F:, each with `fetch_<name>.DONE`. None of them was dumped or labelled.

## Cause
In `sh_run.fetcher`, the disk-floor wait came **before** the `fetch_<name>.DONE` check:
`while free < F_FLOOR_GB + est + 1: sleep(30)`, with up to 6 h per archive.

After a restart, `ready` is empty. So an archive that was already extracted waited for disk space that only its own logs could free, because a finished log deletes its frames. The first such archive (camera_44) needed 9.46 GB free against 4.4 GB, so the whole chain stalled.

## Fix (procedure only; no label, dump or merge arithmetic touched)
1. **Check the marker first.** An archive with `fetch_<name>.DONE` marks its logs ready and is never gated on disk.
   - Verified live: at 09:28:59Z there were 15 "FETCHED ... (already extracted)" lines, then "DUMP batch 7 logs".
2. **`F_FLOOR_GB` is set from `SH_F_FLOOR_GB`** (default 8.0, unchanged). `run_x150_chain.cmd` sets it to **5**.
   - Why 5: F: sits near 4.4 GB, and the space is held by data that is not this chain's.
   - 5 GB is the programme's sweep line (CLAUDE.md, the auto-delete rule).
   - A new archive still needs `5 + est + 1` GB free. `est` is 0.02 to 0.46 GB per archive, and its frames are deleted as each log finishes.

## ENS relaunch (same turn)
- ENS-SCORE `seam_ENS1` died with **rc 3221225477** (0xC0000005, a native access violation) after 3,482 s at about 1,300 of 12,146 tokens.
- There is **no WER record**, so the cause is UNKNOWN. Commit was not traced at the time.
- `ens_relaunch.sh` runs the following:
  - It waits until x150 has dumped every extracted log, so that the 8.4 h seam does not strand them.
  - It relaunches the marker-resumable chain.
  - It traces commit free and the seam's private MB every 60 s (`mm/ens_memtrace.tsv`).
  - It retries **once** only if commit never fell below 1,500 MB.
