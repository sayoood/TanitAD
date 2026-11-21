# HANDOFF RAPR-P01 (addendum 2 row E): instruments built, NOTHING launched, no ssh, no commit (2026-11-04)

Arm = RAPR's frozen argv with `--recover-p 0.1`, runs `RAPRP01_0`, `RAPRP01_1` (argv `argv/RAPRP01_{0,1}.train.txt`, 60 tokens, tokens 7 and 51 differ from RAPR's, MEASURED by test). Old copies of every changed file: `_pre_raprp01/<name>.pre_raprp01`; full old/new sha256: `_pre_raprp01/SHA_CHANGES.txt`.

## What exists (created)
* `box2_files/rnext_chain_raprp01.cmd` (CRLF): differs from `rnext_chain_rapr.cmd` ONLY at lines 2, 3, 6 (comments / launch line) and 23 (`for %%T in (RAPRP01_0 RAPRP01_1)`), pinned by a test.
* `launch_raprp01.sh` + `mk_launch_raprp01.py` (generated from `launch_rapr2.sh`, asserted replacements). Lock text names RAPR-P01 and takes `PI go ${RAPRP01_PI_GO}` (the script refuses to run without that env var). Old marker is moved aside as `CHAIN_RNEXT.rapr_<ts>.done`. Prints `ZZRAPRP01-LAUNCHED`.
* `box2_files/argv/RAPRP01_{0,1}.train.txt` (byte-equal to `argv/`), `read/rnext_read_chain_RAPRP01.cmd` (LF, like the RAPR one), `read/logs/dryrun_RAPRP01.out`.

## Changed (old to new, 16 hex; full values in SHA_CHANGES.txt)
* `box2_files/rnext_check.py` bb7f0fac46a60140 to 833c16bb6192d1ce. NEEDED BEYOND THE BRIEF: it held a six-tag table, so the chain would have refused `RAPRP01_*` at `pin`. Added kind `RAPRP01` (p 0.1, apriori, no table) and `EXTRA` tags that are in scope only when their argv is shipped (`shipped_runs()`), so six-run behaviour is unchanged.
* `tests/test_rnext_chain.py` 27668f9360785787 to 182ec3829a178435 (+4 RAPRP01 tests; pull test stages 8 tags; pull-order check expects 8).
* `pull_rnext.sh` c4e66c6d5ea54dec to 60156928cbe47c19 (two tags appended; the loop body is untouched). `wake_rnext.sh`: NO change needed.
* `read/code/recover_read.py` aeee6b1b1351c811 to a99104733397a53a (ARM_SPEC["RAPRP01"] + docstring; 5 lines; old arms identical: ARM_SPEC, run_table, argv_audit compared old vs new).
* `read/tests/test_recover_read_arms.py` 098244389a3b79c9 to f41ea020c58bc06a; `read/rnext_read_chain.sh` 915cf7048a3eca11 to fe5e9b7dc973ea85 (accepts RAPRP01); `read/PINS.txt` 9366f81fa1e9 to 85811f07478b; three `read/logs/pytest_*.log`.
* `read/raw/RECOVER_READ_R.json` 0637a0e2b6ab to 3c6876823c45, `read/raw/KA_RARM_R.json` 265a811f9ee5 to be2e24c81c22, 30 R array files in `raw/arrays` (old ones in `_pre_raprp01/raw_R_ka/`). Why: gate 0 requires `RECOVER_READ_R.json.reader_sha256` == the pinned reader, so the known answer had to be recomputed (CPU, ~2 min).
* `precheck_pulled.py`, `recover_next_combine.py`, `ka_rarm_r.py`: unchanged (precheck is arm-generic via ARM_SPEC, proven on a RAPRP01 stand-in).

## Tests (MEASURED)
reader `test_recover_read.py` 27 passed; `test_recover_read_arms.py` 45 passed (was 30); `test_recover_next_combine.py` 18 passed. `tests/test_rnext_chain.py`: 19 passed, 1 failed; the failure (`t_package_consistency_and_hygiene`, `read/rnext_read_chain_RAPR.cmd: not CRLF`) is the SAME at baseline (before any edit: 15 passed, 1 failed). Mutations: 5 checker mutations and 6 reader mutations each go RED. Known answer: PASS, 0 differences over 2,599 leaves, 15 of 15 arrays bit-identical. `bash read/rnext_read_chain.sh RAPRP01 --dry-run`: RAPRP01 runs ABSENT, all else present.

## Launch sequence (after the PI go)
1. `RAPRP01_PI_GO="<HH:MM>" bash E:/Projects/TanitAD-artifacts/dagland/recover_next/launch_raprp01.sh` (waits for the box-2 lock; on abort it moves our lock aside).
2. Waker: `RNEXT_SEEN=6 bash .../recover_next/wake_rnext.sh` (6 = RUN_*.done on box 2 now: RAPR, RTL, RP01 x2 seeds; confirm with the same listing the waker uses, `Get-ChildItem D:/tanitad/runs/v8f_mini_full/rnext -Name` piped to `grep -c '^RUN_.*\.done$'`). It exits on the first new done ("7 of 6", cosmetic): re-arm with `RNEXT_SEEN=7`; the chain end exits 0. If an old `RUN_*.FAILED` exists set `RNEXT_SEEN_FAILED`.
3. `bash .../recover_next/pull_rnext.sh` (re-pulls all 8 checkpoints, ~4 GB; expect `bad=0`), then move `BOX2_BUSY.txt` aside.
4. `python -I read/code/precheck_pulled.py --r-arm RAPRP01` (rc 0).
5. `bash read/rnext_read_chain.sh RAPRP01 --dry-run`, then `powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='E:\Projects\TanitAD-artifacts\dagland\recover_next\read\rnext_read_chain_RAPRP01.cmd'}"`; result `read/raw/RECOVER_READ_RAPRP01.json`, marker `raw/RNEXT_RAPRP01.done`.

## Open
* The combiner's `ARMS` tuple has no RAPRP01: its JSON is not summarised in `RECOVER_NEXT.json` and the next-arm table has no row for it (not mine to change; the Master Mind reads the RAPRP01 JSON directly).
* `MANIFEST.txt` files were not regenerated. The pre-existing LF read-chain `.cmd` files fail the package hygiene test (CRLF rule); I kept the RAPR style.
* The box-2 gate (`--selfcheck-only`) per RAPRP01 tag still runs on box 2 at launch; nothing here replaces it.
* `python -I` ignores PYTHONDONTWRITEBYTECODE, so test runs refreshed `__pycache__` folders (one new: `_pre_raprp01/__pycache__`); not shipped, harmless, left in place (no deletions).
