# HANDOFF_TSWAP (2026-11-07 ~13:27 dev-box clock). Folder `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/tswap/` (`$PK`). Scratch `F:/TanitAD-scratch/motion_c/tswap/` (`out/`, `work/`).
Card `../PREREG_V8F_TSWAP.md`: frozen prefix 3,642 bytes sha256 fc775ca1... (re-verified after both log lines; the chain re-checks it). `IMPL_CHOICES_TSWAP.md` sha256 32734a5ffb919c4d1ff8c533be9897bea95457b72e38d2a028493a4f61077bf7 (frozen 13:16:02, log line (a); `.sha256` re-checked by the chain). Launch line (b) 13:24:40. Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, CUDA hidden, 4 threads. Nothing staged, committed or pushed.

## DONE
* `tswap_run.py` (`--arm own|tswap|mutant`): thin wrapper over the pinned tool (sha 0417c838...). TSWAP = TOWN's twin batch (verbatim) -> `forward_infer(twin)` -> O reads the twin's 60-dim code via `forward_diag(batch, code_override)`. O reads exactly four T outputs (`goal_prob`, `lat_logits`, `lon_logits`, `cons_std`; v8f_mini.py:686-688, 737). Mutant = same plumbing, twin := own.
* Subset (10 episodes / 170 windows, 4 threads): **G-ID PASS** (S_OWN == TOWN's bare S_BARE, tick file 27+ keys and 10 dumps bitwise, max |delta| 0.0), **G-TICK0 PASS** (170 rows, 27 tick-file keys bitwise, code read == T's own code), **G-MUT PASS** (mutant == OWN bitwise, nothing swapped), **G-LIVE PASS** (340/340 later rows: code O reads differs from T's own, plan differs from OWN; logged p10f mismatch 0). `raw/SUB_GATES.json`, `logs/sub_gates.out`, rollouts `F:/.../tswap/out/S_*`. First subset run had a bug (the wrap was installed once per chunk, nested); fixed to wrap once, subset rerun, all PASS.
* `tswap_gates.py`, `tswap_result.py` (outcome rule `outcome()` pure; T share bootstrap ratio), `tests/test_tswap.py`: **5 passed** (`logs/pytest_tswap.out`): flipping ONE M1 interval changes the outcome (T-CARRIES -> T-NOT / T-PARTIAL), T share known answer 0.5 / 1.0 / 0, G-LIVE FAILS on the mutant's artifacts, G-ID FAILS on one flipped bit, G-TICK0 FAILS on a tick-0 plan changed by 1e-9.
* Writer rehearsal: `SUBSET_RESULT_TSWAP.md` (banner SUBSET, not a result). ⚠️ The pinned extractor asserts 141 episodes, so the subset rollouts cannot be extracted; the rehearsal used the first 170 windows of the BANKED RIC0_s0 arrays (M1 shifted by -0.02 as a known answer, `make_rehearsal_arrays.py`). The extraction of the real TSWAP rollouts is therefore first exercised in stage C of the chain.

## RUNNING: detached chain, cmd pid 22756 (WMI, 13:24:13), resumable
Stages: A rollouts OWN_s / TSWAP_s / MUT_s for s = 0, 1 (6 rollouts) | B gates per seed -> `raw/GATES_s{0,1}.json` (G-ID vs TOWN's `BARE_s{s}`, G-TICK0, G-MUT, G-LIVE) | C `tswap_result.py` -> `RESULT_TSWAP.md`, `raw/READ_TSWAP.json` (INVALID, nothing read, if a gate fails). Log `logs/chain_tswap.out` (grep ZZTSWAP). **End marker `$PK/CHAIN_TSWAP.done` = `rc=<n> ...`** (0 ok; 2 preflight; 3 a rollout or the writer failed: read the log it names). Stage markers `raw/CHAIN_STAGE_{A,B,C}.done`.
ESTIMATED runtime ~40 min (TOWN's rollouts took 162-280 s each at 4 threads; OWN_s0 progress 3 s per 32 windows, about 225 s; 6 rollouts ~25 min, gates ~1 min, 4 extractions + bootstrap ~10 min) -> done about 14:05.
Relaunch after a failure (finished rollouts and stages are skipped): `powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c \"E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\tswap\chain_tswap.cmd\" < NUL'}"`.
If only stage C fails: `python -I tswap_result.py --out-root F:/TanitAD-scratch/motion_c/tswap/out --work-dir F:/TanitAD-scratch/motion_c/tswap/work --raw $PK/raw --out-dir $PK --gates-dir $PK/raw` (CUDA_VISIBLE_DEVICES=-1).

## NEXT (fresh agent, Sonnet, <= 100k context)
1. Wait for `CHAIN_TSWAP.done` with ONE background watcher; read `RESULT_TSWAP.md`. A gate FAIL gives INVALID: report which gate from `raw/GATES_s*.json`; do not tune.
2. Append ONE dated entry to the card's Log (time from `date`): outcome word, gates, M1 and ADE6 rows with CIs at both seeds, T share, rows where TSWAP is separated-worse, the next step the card commits. Re-check the 3,642-byte prefix.
3. Hand back <= 300 words with the manifest.

## Disclose
* The identity references are TOWN's `BARE_s0/1` and `S_BARE` (pinned tool, 4 threads, same flags), accepted by the chain only if their done markers carry the pinned tool sha and `threads 4`.
* The twin's internal S runs on the logged state and conditions only the twin's T (TOWN's twin as it was); O's S input (`ck_for_o`) is the own call's.
* M1 "better" = interval below 0 (lower is better, gapdecomp.sep_worse); T share on the windows where OWN, TSWAP and P10 are all finite.
* Deleted (own scratch, literal paths): `F:/TanitAD-scratch/motion_c/tswap/pt` 4,796 KiB (pytest basetemp), `F:/TanitAD-scratch/motion_c/tswap/work_subset` 296 KiB (rehearsal arrays, re-creatable). Kept: subset rollouts `out/S_*` (the tests read them).

## Manifest (all under `$PK` unless a path is given)
IMPL_CHOICES_TSWAP.md(+.sha256), tswap_run.py, tswap_gates.py, tswap_result.py, make_rehearsal_arrays.py, run_subset.sh, chain_tswap.sh/.cmd, tests/test_tswap.py, raw/grid_first10.json, raw/SUB_GATES.json, SUBSET_RESULT_TSWAP.md, raw/SUBSET_READ_TSWAP.json, logs/*; card Log lines (a) 13:16:02 and (b) 13:24:40. Pending from the chain: raw/GATES_s{0,1}.json, raw/READ_TSWAP.json, RESULT_TSWAP.md, CHAIN_TSWAP.done. Rollouts, receipts and arrays: `F:/TanitAD-scratch/motion_c/tswap/{out,work}`.
