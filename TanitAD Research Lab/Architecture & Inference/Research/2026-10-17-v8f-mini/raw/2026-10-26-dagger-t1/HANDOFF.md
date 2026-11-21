# HANDOFF: dagt1, the dev-box CPU T1 chain of PREREG_V8F_DAGGER. BUILT AND SELF-TESTED, NOT LAUNCHED, SMOKE CHAIN NOT RUN. 2026-10-26

## State
`dag_t1_chain.sh` (+ `.cmd`), `dagt1_pullcheck.py`, `dagt1_manifest.py`, `dagt1_selftest.py` (counts in `selftest_output.txt`: 219/219, incl. 45/45 mutants), `dagt1_smoke.py`, `loader_test.sh` are in `E:/Projects/TanitAD-artifacts/dagt1/`. Detail: `NOTES.md`.
Revised 2026-10-26 22:xx: the bank-file hole is closed (this chain verifies the 11 bank files per seed the box-2 chain now publishes, and the manifest carries `bank_file`), and decision O1 is enforced (a SHA256SUMS.txt carrying a box-2 fresh eval is refused; `DAGT1_B2_FRESH` / `--b2-fresh` removed). `bankpub.diff` in this directory is the change (code and docs, against the bytes before this revision).
The reader (`C:/Users/Admin/dagread/dag_read.py`) reads this chain's manifest cleanly on a synthetic world (176 controls PASS) and REFUSES each deliberate regression.
Not run: the SMOKE chain. Reason: C: fell to about 0.28 GB free (2.3 GB at 20:08); two torch processes on a nearly full system drive was not worth the risk, and the chain's own start gate (C: >= 2 GB, PREREG section 8) refuses a real start in that state (`dry_run_C_gate_refused.txt`).

## Next steps (a fresh, cheap agent)
1. Free C: to >= 2 GB (PI-only deletes; see memory `devbox-drive-letter-and-detached-jobs`: runaway task outputs, orphan `lg_full_*`, `box2_ship` tip tars).
2. SMOKE (about 3 min, CPU): `python dagt1_smoke.py setup` (copies an 87 MB stand-in checkpoint), then `DAGT1_SMOKE=1 DAGT1_RUNS=dag:0 bash dag_t1_chain.sh`, then `python dagt1_smoke.py check`. Expected end: `ZZDAGT1-SMOKE-DONE 2/2 rollouts, reader rc 2 (REFUSED, nothing written)`. Re-run the same command to see the resume (`skip ...`).
3. Inputs: the chain reads the T1 inputs from `E:/Projects/TanitAD-artifacts/sfresh_inputs` by default (`DAGT1_INPUTS` overrides); nothing of it is read from C:. Before the real launch: (a) land the DAgger code, export the landed tip to a directory, set `DAGT1_TREE` to it (not `rt4/tree`; the chain refuses that); (b) set `DAGT1_READER` to the landed `dag_read.py`; (c) the box-2 chain (dagq) now publishes the 11 bank files per seed with a SHA256SUMS.txt (the shard npz were the former open defect 2, CLOSED: NOTES.md section 6, `dagq/NOTES.md` section 10, ESTIMATED pull ~0.66 GB per seed) and this chain's bank gate verifies them against the trainer gate's declared `--dagger-bank-sha256`; the generator must still write K-B4/K-B5 on a real bank; box 2 pulls NO fresh-age eval (decision O1: a SHA256SUMS.txt carrying one is refused; `DAGT1_B2_FRESH` is removed and STOPs if set); (d) `bash dag_t1_chain.sh --dry-run` must end `ZZDAGT1-DRY-OK`.
4. Launch, detached, with the box-2 stop watcher (edit the `.cmd` environment line, a WMI process does not inherit the shell):
```
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c set DAGT1_B2_OUT=C:/Users/Admin/box2_ship/logs/chain_v8f_dag.out&& set DAGT1_TREE=<landed tip export>&& E:\Projects\TanitAD-artifacts\dagt1\dag_t1_chain.cmd'; ProcessStartupInformation=$si}
```
Markers: `E:/Projects/TanitAD-artifacts/dagt1/logs/chain.out` (last one wins). Reader rc 0 -> `v8f-dagger-t1/read/DAG_READ.json` (a fresh agent writes the result; do not edit the reader after seeing an outcome); rc 2 REFUSED -> `logs/dag_read.out` lists the failed controls; rc 4 BLOCKED.
