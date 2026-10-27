# HANDOFF: dag2t1, the dev-box CPU T1-kin read of PREREG_V8F_DAGGER2. BUILT AND SELF-TESTED, NOT LAUNCHED. 2026-10-27

> **UPDATE 2026-10-27 (`AMEND1.md`)**: decision 1 below is RESOLVED by PREREG_V8F_DAGGER2 Amendment 1 (`A1-2026-10-27`, `1446333d`); the reader, its pin (`PREREG2_SHA256` / `PREREG2_SHA` already updated, the PREREG staged at `inputs/PREREG_V8F_DAGGER2.md`) and the self-test are re-coded and re-proved (196/196, 68/68). Step 1 of the next steps is done except `bash dag2_t1_chain.sh --dry-run` on the REAL configuration and `python dag2t1_selftest.py --real-pulls`, which hash the pulled DAG2 eval files and were not run (blinding).

## State
Everything is in `E:/Projects/TanitAD-artifacts/dag2t1/` (detail: `NOTES.md`, inventory with sha256: `MANIFEST.txt`). `selftest_output.txt` holds the counts and every mutant's result. Nothing was run on rollouts, nothing is committed or pushed, box 2 was not contacted.

## Decisions waiting on the PI (the build cannot make them; no DAG2 row has been read, so there is still time)
1. PREREG_V8F_DAGGER2's outcome table leaves 4 of the 16 (R-CRAWL, R-MOVE) x 2 seeds combinations unplaced (R-MOVE at exactly one seed with R-CRAWL failing at that seed). The reader prints UNPLACED (rc 3) and names no outcome. An amendment must place them BEFORE the first row is read.
2. The RF bars have no consumer in the table (guards are "not separated-worse"). Reported beside the outcome only.
3. C: has ~0.45 GB free; the chain's start gate needs 2048 MB (E: 1500 MB; E: has ~1.5 GB free for ~1.3 GB of outputs, so E: needs freeing too). PI-only deletes.

## Next steps (a fresh, cheap agent)
1. After the PI frees C: and (if wanted) amends the PREREG, re-run `bash dag2_t1_chain.sh --dry-run` (needs the amended PREREG pin: update `PREREG2_SHA` in `mk_dag2_helpers.py` and `PREREG2_SHA256` in `dag2_read.py` together, then `python mk_dag2_helpers.py` and re-run `dag2t1_selftest.py`).
2. Launch detached with the box-2 stop watcher (command in `NOTES.md` section 2). Markers: `logs/chain.out`.
3. The chain waits for box 2 to pull s1 (it polls; waiting costs no tokens). Reader rc 0 writes `v8f-dagger2-t1/read/DAG2_READ.json`; rc 2 REFUSED lists the failed control in `logs/dag2_read.out`; rc 3 UNPLACED; rc 4 BLOCKED.
4. A fresh agent then writes the result and updates the claims register (`H-V8F-DAGGER2-1`). Do NOT edit the reader after seeing an outcome.

## Single-copy items (INTEGRATE)
* `dagreread/code/{rr_lib,readA,build_report}.py` (Read A's driver) is not in the repo at `1d4f4fbb`; second copies are in `dag2t1/readA_code/`.
* This whole directory is on E: only (exFAT, external). Land `dag2_read.py`, `dag2t1_manifest.py`, `dag2t1_pullcheck.py`, `dag2_t1_chain.sh`, `mk_dag2_helpers.py`, `dag2t1_selftest.py` once the PI has ruled on items 1 to 2.
