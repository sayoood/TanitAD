# HANDOFF Q5C0 (2026-11-15, build agent). Built and dry-run tested; NOT launched. Nothing was written to box 2.

## State
* Eval npz DONE (MEASURED): `s0p_extract_eval.py` re-run on the dev box (CPU, 2 threads, 91 s) -> `F:/TanitAD-scratch/q5_416/run/c0/s0p_eval.npz`, sha256 `0c25fe722f3359ee49ec6c31affa6761a4cd6f3ecb1e63d4ccc475ae2935fb39` = the banked one (the S0-P copy had been deleted as re-creatable).
* Chain `C:/Users/Admin/q5c0/q5c0_chain.sh` (+ `q5c0_tool.py`, `launch.cmd`), tests in `tests/run_tests.sh` (fake box 2 in `tests/fake_*`; results in `tests/last.out`).
* Waits for `actsens_state/DONE.txt`; a `STOP.txt` there = exit rc 12, last log line `ZZ Q5C0 FAIL ZZ actsens-stop ...`.
* Then: dev busy marker `box2_ship/BOX2_BUSY.txt` absent (noclobber claim, released by mv to busy_aside), box 2 `D:\tanitad\BOX2_BUSY.txt` absent, GPU < 3500 MiB, no `python.exe` with `tanitad` in its command line.
* Box 2 only under `D:\tanitad\stopperc\q5c0\{s0p,jobs,out}`. Ships `s0p_lib.py` + `s0p_extract_train.py` (sha256 checked), `--dry-run` S1 + S2, then 4 shards x 2 samples one at a time (~160 s each MEASURED in S0-P), each pulled with sha256 + MANIFEST + meta checks, box-2 copy deleted, then assembled to `c0/S1`, `c0/S2` (concatenation checked against the dry-run's sample.json), `c0/C0_RESULT.json`.
* Resume: `Q5C0_RESUME=1` + same launch.

## Important finding (C0 as frozen cannot hold literally)
The banked `shard_*.verify.json` `manifest_sha256` hashes a MANIFEST that lists `meta.json`, and `meta.json` holds `elapsed_s` (161.4 / 156.6 / 160.7 / 159.4 s in S0-P). A rerun cannot repeat it. `c0check` therefore: (a) records the literal comparison (`literal_manifest_equal_all_shards`, expected false), (b) PASSES a shard only if its meta equals the banked meta except `elapsed_s` AND the manifest rebuilt with the banked job's own meta.json (parsed from box 2's S0-P stdout, copied read-only to `ref/sh{0..3}.log`) equals the banked hash. That proves every array and sample.json byte. The MM should note this reading when reporting C0.

## Not done / open
* The S0-P banked train shards no longer exist (deleted by S0-P), so no byte-level diff is possible beyond the hash rebuild; the refit check (E|linear, V40+E|linear eval scores bitwise) of C0 in Amendment 7 is NOT in this chain (needs the 4 pulled shards fed to `s0p_fit.py`; the chain deletes them after PASS). Say so if it is needed, and keep `_pull`.
* Box-2 `D:\tanitad\repo` may have drifted since 11-12; `state/repo_probe.txt` records the sha256 of `v8f_t1_eval.py` and `v8f_mini_data.py`. A C0 FAIL would point there first.
* Deletions: see `deletions.log`. One refused: `F:/TanitAD-scratch/q5_416/run/c0/_tmp_q5c0` (own 8-window smoke npz, < 1 MB) - the tool refused the rm; MM may delete it.
