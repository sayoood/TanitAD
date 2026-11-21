# HANDOFF: the dev-box CPU chain of PREREG_V8F_ORECOVER (orct1). BUILT AND SMOKED, NOT LAUNCHED. 2026-10-26 12:25

## State
`C:/Users/Admin/orct1/orc_t1_chain.sh` (+ `orc_t1_chain.cmd`, CRLF) waits for each of the four O-RECOVER pulls (ORC s0, ORC-0 s0, ORC s1, ORC-0 s1) to be COMPLETE by content, runs the two T1-kin rollouts per run
(`--limit causal`, `--limit logged`, plain, `--dump-ticks --dump-heads`, the T1-HEADS tool tree / inputs / threads; 2 lanes = at most 2 processes), then writes `runs.json` and runs `orc_read.py`.
Dry-run, 39-case self-test, 6 guard stubs, SMOKE chain (24 windows, resume, half-pull, changed checkpoint, pin tamper) and a full-size equivalence run (32/32 arrays bit-identical to the banked T1-HEADS rollout) all pass.
Detail and evidence: `NOTES.md`. At 12:20 the box-2 chain's gate stage had already created `E:/Projects/TanitAD-artifacts/v8f-orecover/M2b_s0_orc/` (empty); the dev-box chain has NOT been started.

## K4: decided by PREREG Amendment 4 (landed 853f2fc3, before any run)
K4's comparator is the arm's own `--ages fresh` T0 eval, under its own manifest key `t0_fresh`. R-T0, T0-CKPT and the families stay on box 2's
standard `--ages cycle` eval (`t0`). The chain always runs the fresh eval. `ORCT_T0_FRESH=0` and any `ORCT_T0` are refused (NOTES §11).
The reader tree is `E:/Projects/TanitAD-artifacts/rt4/tree`, a tip export carrying the A4 reader.

## Exact launch (detached; it only waits and costs no tokens)
Dry run first (touches nothing except `logs/orc_t1_chain.launch.out`):
```
cmd.exe /c C:\Users\Admin\orct1\orc_t1_chain.cmd --dry-run
```
Real, with the box-2 stop watcher (read-only on that log):
```
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c set ORCT_B2_OUT=C:/Users/Admin/box2_ship/logs/chain_v8f_orc.out&& C:\Users\Admin\orct1\orc_t1_chain.cmd'; ProcessStartupInformation=$si}
```

Markers in `C:/Users/Admin/orct1/logs/chain.out` (use the LAST one; the file is appended across resumes): `ZZORCT1-DONE 8/8 rollouts, reader rc <0|2|3> (...)` or `ZZORCT1-STOP <why>`.
Resume after any STOP or kill: the same command. The STOP file `logs/STOP` is removed at start. Outputs on E:: `v8f-orecover-t1/` (`t1v8f_*.json`, `ticks/`, `dump_*`, `done_*.txt`, `PINS.txt`, `runs.json`, `read/ORC_READ.json`).
Prerequisites: C: >= 2 GB free (the chain waits for it), the session scratchpad `sfresh` dir present (cache, tracks, v9, p10), `E:/Projects/TanitAD-artifacts/oc/tree` and `v8f-t1heads-run/tool_81aa3e36` unchanged (pinned; a change STOPs the chain).

## Next step
1. Launch (above) once `rt4/tree` exists (a tip export with the A4 reader).
2. When `logs/chain.out` ends with `ZZORCT1-DONE`: rc 0 -> `E:/Projects/TanitAD-artifacts/v8f-orecover-t1/read/ORC_READ.json` is the read (a fresh agent writes the result; the reader is not edited after an outcome is seen);
   rc 2 -> `logs/orc_read.out` lists every failed control (REFUSED: fix and re-run, per the PREREG); rc 3 -> UNPLACED (the table needs an amendment).
3. If a pull never completes: the chain waits up to 40 h per pull, or stops on the box-2 chain's `ZZV8FMINI-STOP` if `ORCT_B2_OUT` was set.
4. Landing (the MM's): `orc_t1_chain.sh`, `.cmd`, `orct1_pullcheck.py`, `orct1_manifest.py`, `orct1_selftest.py`, `NOTES.md` into the repo with the reader.
