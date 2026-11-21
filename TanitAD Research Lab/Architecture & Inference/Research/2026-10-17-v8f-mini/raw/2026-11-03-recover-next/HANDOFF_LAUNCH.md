# HANDOFF_LAUNCH: RECOVER-NEXT six-run chain for box 2 (BUILT, NOT LAUNCHED; every box-2 start needs the PI's go in chat)

**Card:** `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `01046fe6...`, frozen before any of these runs exist). **Runs, in order:** RAPR_0 RAPR_1 RTL_0 RTL_1 RP01_0 RP01_1 (`M2b_s{0,1}_recft_{RAPR,RTL,RP01}`).
**Code:** the trainer on box 2 is unchanged (`D:\tanitad\repo`, landed tip `ad0bcb72...`); **no new tar and no refresh.py.** `rnext_check.py` binds to it exactly as the RECOVER-FT checker did: `preflight` re-hashes `v8f_mini_train.py` (`1576a578...`) and `v8f_recover.py` (`eff087da...`) in `D:\tanitad\repo`, `shipped_line` reads the shipped argv file (one line), and a gate token (`gate-cached`) is reused only if commit, argv sha256, selfcheck.json sha256, init sha256 and both shipped-byte hashes are identical.
**New in the checker:** a frozen-argv audit (a changed lever is REFUSED at `preflight` and `pin`, with the flag named), per-kind BUILT-lever checks at the gate (RAPR: p 0.3, apriori, rejoin, 10, seed s, no table and no bank built; RTL: p 0.3, empirical, logged, rejoin_steps = the trainer default 10 (`v8f_recover.py:23`), seed s, table from bank s; RP01: p 0.1, empirical, rejoin, 10, seed s, table from bank s), and a `RECOVER_DEFAULTS` re-read from the shipped bytes. C0 / C0rep branches are gone (comparators are the existing C0_s, C0rep_s, P10).

## Files (dev box `E:/Projects/TanitAD-artifacts/dagland/recover_next/box2_files/` -> box 2 `D:\tanitad\rnext\`)
`rnext_check.py rnext_chain.cmd rnext_run.cmd rnext_gate_only.cmd rnext_guard.ps1 rnext_run_guard.cmd rnext_run_telemetry.cmd argv\{RAPR,RTL,RP01}_{0,1}.train.txt`.
Reused by path, unchanged: `D:\tanitad\env.cmd`, `D:\tanitad\launch.ps1`, `D:\tanitad\telemetry_long.ps1`. Changed copies (the RECOVER-FT ones name their own marker folder and chain-end files): guard (`rnext_guard.ps1` + `rnext_run_guard.cmd`; the stop rule is the same, plus a logged warning when the telemetry csv is stale) and the telemetry wrapper (moves a leftover `telemetry.stop` aside, never deletes it, then runs `telemetry_long.ps1`).
Markers `D:\tanitad\runs\v8f_mini_full\rnext\` (`RUN_<tag>.done|FAILED`, `gate_token_<tag>.json`, `pin\`, `logs\`, `CHAIN_RNEXT.done|STOPPED`, `GUARD_STOP.txt`); gates `...\gate_rnext\`; chain log `D:\tanitad\logs\rnext_chain.log`; guard `D:\tanitad\logs\rnext_guard.log`.

## Master-Mind commands (after the PI's go). Nothing here has been executed.
```bash
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"; SCP="scp -q -o BatchMode=yes -o ConnectTimeout=20"
SRC=E:/Projects/TanitAD-artifacts/dagland/recover_next/box2_files
# 0. read-only preconditions: lock free, no python on box 2, the RECOVER-FT chain ended, D: free GB (the chain refuses to start with a python process present or < 8 GB free)
test ! -e /c/Users/Admin/box2_ship/BOX2_BUSY.txt && echo "lock free"
$SSH "powershell -NoProfile -Command \"@(Get-Process python -ErrorAction SilentlyContinue).Count; Test-Path D:/tanitad/runs/v8f_mini_full/recft/CHAIN_RECOVER.done; [int]((Get-PSDrive D).Free/1GB)\""
# 1. take the lock (atomic)
( set -o noclobber; echo "RECOVER-NEXT six-run chain (D:\\tanitad\\rnext\\rnext_chain.cmd), owner: Master Mind, since $(date '+%m-%d %H:%M'), ETA ~9.5 h. Done when D:\\tanitad\\runs\\v8f_mini_full\\rnext\\CHAIN_RNEXT.done exists. Do NOT launch box 2 jobs while this file exists; remove it after the pull." > /c/Users/Admin/box2_ship/BOX2_BUSY.txt ) || echo "BUSY: stop"
# 2. ship (new folder; no .cmd is overwritten while a queue runs: step 0 showed no python process)
$SSH "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force D:/tanitad/rnext/argv | Out-Null\""
for f in rnext_check.py rnext_chain.cmd rnext_run.cmd rnext_gate_only.cmd rnext_guard.ps1 rnext_run_guard.cmd rnext_run_telemetry.cmd; do $SCP $SRC/$f tanitad-box2:D:/tanitad/rnext/$f; done
for t in RAPR_0 RAPR_1 RTL_0 RTL_1 RP01_0 RP01_1; do $SCP $SRC/argv/$t.train.txt tanitad-box2:D:/tanitad/rnext/argv/$t.train.txt; done
# 3. verify the shipped bytes (hashes must equal MANIFEST.txt's box2_files rows; the box-2 half goes in on stdin, so bash does not expand $_)
( cd $SRC && sha256sum rnext_* argv/*.train.txt | sed 's/ \*/  /; s# argv/# #' ) | sort > /tmp/rnext_dev_sha.txt
$SSH "powershell -NoProfile -Command -" <<'PS' | tr -d '\r' | sort > /tmp/rnext_box2_sha.txt
Get-ChildItem D:\tanitad\rnext -Recurse -File | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() + '  ' + $_.Name }
PS
diff /tmp/rnext_dev_sha.txt /tmp/rnext_box2_sha.txt && echo "shipped bytes identical"
# 4. telemetry first, check it is FRESH (two rows 35 s apart differ), then the guard, then the chain
L() { $SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\tanitad\\launch.ps1 -Job $1 -Log $2"; }
L 'D:\tanitad\rnext\rnext_run_telemetry.cmd' 'D:\tanitad\logs\rnext_telemetry.log'; sleep 60
tl() { $SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1\"" | tr -d '\r'; }
t1=$(tl); sleep 35; t2=$(tl); [[ $t2 == 20* && $t2 != "$t1" ]] || echo "STOP: no fresh telemetry, the guard would be blind"
L 'D:\tanitad\rnext\rnext_run_guard.cmd' 'D:\tanitad\logs\rnext_guard.out'
L 'D:\tanitad\rnext\rnext_chain.cmd' 'D:\tanitad\logs\rnext_chain.log'
```
**Poll** (short ssh, or the waker as a background task / Monitor: `bash E:/Projects/TanitAD-artifacts/dagland/recover_next/wake_rnext.sh`; it exits 0 on a new `.done` or the chain end, 3 on a FAILED run, 4 on a safety stop, 5 on timeout; re-arm with `RNEXT_SEEN=<n done>` / `RNEXT_SEEN_FAILED=<n failed>`):
```bash
$SSH "powershell -NoProfile -Command \"Get-Content D:\tanitad\logs\rnext_chain.log -Tail 6\""
$SSH "powershell -NoProfile -Command \"Get-ChildItem D:\tanitad\runs\v8f_mini_full\rnext -Name\""
$SSH "powershell -NoProfile -Command \"Get-Content D:\tanitad\runs\v8f_mini_full\M2b_s0_recft_RAPR\train_log.jsonl -Tail 1\""
```
**Stop between runs:** write `D:\tanitad\runs\v8f_mini_full\rnext\GUARD_STOP.txt` (the chain halts at the next run boundary and writes `CHAIN_RNEXT.STOPPED`). A stopped trainer resumes from `ckpt_last.pt`: relaunch step 4's chain line (done runs are skipped; a cached gate token is reused only if every binding is identical).
**Pull** (after `CHAIN_RNEXT.done`): `bash E:/Projects/TanitAD-artifacts/dagland/recover_next/pull_rnext.sh` -> `E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{0,1}_recft_{RAPR,RTL,RP01}/` + `rnext_markers/`, ckpt sha256 checked against each marker. Then **remove `C:\Users\Admin\box2_ship\BOX2_BUSY.txt`** and confirm `D:\tanitad\logs\telemetry.stop` exists and the guard log says "chain ended".

## Expected wall (card section 5, ESTIMATED; the 42 min table build is MEASURED on the RECOVER-FT R gate, 45 m 20 s)
RAPR ~0.7 h per run (no table build) x 2; RTL and RP01 ~2 h per run (gate table build ~42 min + the training start's build ~42 min + training) x 4; whole chain ~9.5 h. Dev-box read ~2 h afterwards (card section 3).

## Tests (dev box, CPU, network blocked in-process): `tests/test_rnext_chain.py`, log `logs/test_rnext_chain.log`
Argv audit and pin for all six; 15 changed-argv cases REFUSED with the flag named; gate PASS for a stub of each kind (built from the real banked R_0 selfcheck with `levers.recover` taken from the shipped library's own `RecoverPerturb.built_levers()`) and 70 wrong-built stubs (per kind and seed, plus 4 stubs of another kind) FAIL; post, finish, markers; the shipped trainer's own parser accepts all six argv and its `bad_recover` refuses a bank with the a-priori law; waker and pull scripts against local stand-ins; guard stop rule on a copy with the kill pattern neutralised. Mutation check of the checker itself (`logs/mutation_meta.log`): 4 deliberate breaks of `rnext_check.py` each turned the matching test RED (run with `RNEXT_TEST_CHECK=<mutant>`).

## UNVERIFIED until box 2 runs it
1. The `.cmd` files are renamed copies of the proven RECOVER-FT chain; they have never been executed. Only the checker is exercised (CLI, dev box).
2. **RAPR_0's gate is the first read of an a-priori-law selfcheck.json.** The expected shape (no `info.recover`, `levers.recover_table` None, no bank in `levers.recover`) is read from the shipped source (`v8f_mini_train.py` make_recover / the levers block, `v8f_recover.py` built_levers), not from a real RAPR selfcheck. A different real shape makes the gate refuse (fail closed), and the chain moves on.
3. RTL's built `recover_rejoin_steps` = 10 is the trainer default (argv drops the flag); the real value is first seen at the RTL_0 gate.
4. The real banks are not on the dev box: the positive `preflight` test stubs the bank.json hash; box 2 hashes the real files against `90c43c66...` / `a06c8289...`.
5. `rnext_guard.ps1`: its kill path (Stop-Process on our trainer) has not been exercised anywhere; the stale-telemetry warning is new code. `telemetry_long.ps1` is not in this package; how it treats an old csv is unknown (the wrapper only moves a stale `telemetry.stop` aside).
6. The ssh / scp lines above are unexecuted; `launch.ps1` taking a job path and a log (no arguments) is read from `chain_safe.sh`.
7. Disk: six run dirs plus four tables; the floor is 8 GB free before each run (`RNEXT_FLOOR_BYTES`).
