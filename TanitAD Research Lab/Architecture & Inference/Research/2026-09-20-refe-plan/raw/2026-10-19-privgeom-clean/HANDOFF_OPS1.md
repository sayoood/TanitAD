# HANDOFF OPS1: PRIVGEOM-CLEAN worker-completion fix (prepared 2026-10-20 ~01:30; CPU only; NOTHING launched, NOTHING sealed)

**State.** The sealed read stopped at 01:09:44 (`extraction worker crashed 4 times`). Extraction is COMPLETE (76/76 `.npz`, no lock, no `.failed`); no stage after it has run, no label opened. **Cause VERIFIED:** `Popen.pid` of the venv `python.exe` is a launcher; `pg_extract.py` writes `WORKER_DONE_<interpreter pid>.txt`; the controller looked up the launcher pid (table of the 5 pid pairs: amendment section 2). A same-root second defect: the dead-worker lock was matched on the launcher pid too.

**Right now the package FAILS CLOSED**: `python code/pgc_seal.py check` refuses (the draft amendment has no `.sha256`; three pinned files differ from the seal). No stage can run until the Master Mind seals.

## What the Master Mind must do (2 steps)
1. Review `ops1/pgc_read_chain.py.diff`, `ops1/pgc_seal.py.diff`, `ops1/PINS.json.diff` (about 235 lines; the `pgc_seal.py` one is the judgment call: a SEALED `_AMENDMENT_OPS<k>` may re-pin ONLY `REPINNABLE` operational files, never analysis code; no other way exists to ship ANY change to a pinned file) and `PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md`.
2. `cd C:\Users\Admin\par18\pgclean ; C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe -B ops1\seal_amendment_ops1.py` -> prints `SEALED {...}` (`amend_ops1` in pins, 3 files re-pinned) or `ROLLED BACK`. It flips DRAFT to SEALED, writes the `.sha256`, runs `check_seal(inputs=True)`. Rehearsed on a temp mirror: sealed, and the real `pgc_run.py preflight` then passed there with `subset_clean.json` byte-identical to the real one.

## Launch (after step 2; `launch.ps1` is blocked by execution policy)
```powershell
cd C:\Users\Admin\par18\pgclean
Copy-Item run_clean\CHAIN_STOP.txt run_clean\CHAIN_STOP_attempt1.txt   # the chain deletes CHAIN_STOP.txt at start
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}
$line = 'cmd /c "C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe -B -W ignore C:\Users\Admin\par18\pgclean\code\pgc_read_chain.py < NUL >> C:\Users\Admin\par18\pgclean\run_clean\read.log 2>&1"'
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$line; CurrentDirectory='C:\Users\Admin\par18\pgclean\code'; ProcessStartupInformation=$si}
```
Expect in `run_clean/read.log` (appended): `ZZPGC_START`, `ZZPGC_PREFLIGHT_OK`, `extraction already complete (76/76): no worker spawned`, `ZZPGC_EXTRACT_DONE 76/76`, then `ZZPGC_JOIN_OK`, `ZZPGC_GATE_PASS`, `ZZPGC_FIT_DONE`, `ZZPGC_ANALYZE_OK`, last line `ZZPGC_DONE <call>` or `ZZPGC_STOP <reason>`. No worker is started (harmless-work question moot). Wall: join + gate ~10 min + fits ~5 min (> 4 min: poll by script, not turns): `until [ -f run_clean/CHAIN_DONE.txt ] || [ -f run_clean/CHAIN_STOP.txt ]; do sleep 120; done`. Never `pkill -f`; kill only by explicit PID. After it ends, the NEXT step is the one in `HANDOFF.md` ("Exact next step", items 1 to 4); any further change to a pinned file needs another sealed amendment (`_AMENDMENT_OPS2`, chain `from` = the shas in this one).

## Evidence (all under `ops1/`)
`test_ops1_result.txt` (34 tests OK: RED on current bytes, GREEN on the fix, RED arms, 9 mutation arms, seal RED arms, real-package mirror), `test_pgc_seal_result.txt` (24 OK against the changed seal tool), `chain_scenarios_output.txt` (old vs fixed controller logs on planted launcher+child processes), manifests `scalars_npz_sha256.txt` (76 outputs kept; `pg_extract.py` sha `64a14890...` unchanged), `orig/` (old bytes), `pgc_pins_output.txt` (`pgc_pins.py --check` OK).

## Not fixed / disclosures
Amendment section 8: `p.terminate()` on a `.failed` stop hits the launcher only; `run_clean/seal_proofs/` not regenerated (old checker); sealed prereg section-13 table still lists the OLD shas by design; `build_package.py` not re-run (files staged by hand in `out/`).
