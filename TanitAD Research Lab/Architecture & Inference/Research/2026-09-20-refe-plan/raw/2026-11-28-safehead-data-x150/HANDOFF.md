# HANDOFF - L2 SAFE-HEAD cache x150 (+150 logs), launched 2026-11-27 ~19:49 local (18:49Z)

Approval: PI "+150 logs" 2026-11-27. Root `F:/safehead_cache/x150/` (own out/markers/logs/plan/dump/lab/frames); code `F:/safehead_cache/code_x150/`
(copy of sh_*; changes: SH_WORK root guard, `--only`, frame throttle F: >= 8 GB, frames of a DONE log always deleted, commit gate in MB = 6000, plan script `sh_plan_x150.py`).
Nothing under `F:/safehead_cache/{out,markers,logs,plan,navtest,ckpt,code}` was written. GPU lock: same per-log protocol as sh_run (shared with navtest/TRK).

## Plan (PLAN.json, DOWNLOAD_FILES.md)
150 logs = 143 train + 7 held-out (all remaining held-out with a local DB; 11 of the 18 not yet cached have no local DB / are not navtrain-eligible: listed in the plan log),
9,541 tokens, 40 HF archives, transfer UPPER BOUND 177.97 GB (<= 200; the stream stops once the needed frames are found), 38,164 frames (~8 GB if all kept; deleted per log).

## Running
* chain: `cmd.exe /c F:\safehead_cache\code_x150\run_x150_chain.cmd` PID 38644 -> python 27324 -> sh_run.py PID 44136 (children: curl/tar fetch, sh_dump, sh_label). Started via WMI Create ReturnValue 0, flags 0x08000200.
* log: `F:/safehead_cache/x150/CHAIN.log` (+ `logs/chain_x150_stdout.log`, `logs/dump.log`, `logs/label_<log>.log`, `logs/fetch.log`)
* markers: `x150/markers/log_<log>.DONE` per log; `fetch_<archive>.DONE`; `chain.DONE` (success) / `chain.FAILED`. (A stale smoke chain.DONE was removed.)
* Smoke PASSED: log 2021.06.14.13.27.42_veh-35_01342_01461 (8 tokens): `check_contract.py` PASS, CTRL-LOGIT PASS (5.7e-6); the dump/label/merge code is byte-identical to the original (CTRL-PICK/LABEL passed on it earlier). Its DONE marker is kept, so the full run skips it.
* Kill only your own PIDs: 38644, 27324, 44136 and their children. Resume = rerun `run_x150_chain.cmd` (per-log DONE + sha re-check).
* Check: `tail F:/safehead_cache/x150/CHAIN.log`; `ls x150/markers | grep -c log_`; F: free must stay >= 8 GB (chain waits before fetching).
* Expected end: ~12-18 h (label ~3 s/token x 9.5k tokens ~8 h CPU, dump ~2 s/token, plus GPU-lock waits behind navtest/TRK), i.e. 2026-11-28 morning to midday local. Memgate: RAM >= 2000 MB, commit >= 6000 MB on 5 samples; it can wait long when navtest holds ~5.5 GB.
* Failure: `markers/chain.FAILED` -> read the last lines of CHAIN.log, fix, rerun the cmd (resumable).

## After chain.DONE: build the 185-log index for sealing
`C:/Users/Admin/venvs/tanitad/Scripts/python.exe F:/safehead_cache/code_x150/build_index_185.py --out F:/safehead_cache/x150/index_185.csv`
= old `F:/safehead_cache/index.csv` (35 logs, 2,272 tokens) + `x150/index.csv`; re-hashes every npz against its sha, fails on duplicate log/token; `file` column becomes absolute (old rows -> F:/safehead_cache/out/, new -> x150/out/).
Expect 185 logs, 2,272 + 9,541 minus dropped-no-label tokens (see `out/*.merge.json` dropped_no_label), 13 held-out logs (6 + 7).
Then `check_contract.py x150/out/*.npz` and `sh_ctrl.py logit` (SH_WORK=F:/safehead_cache/x150) over a sample. Held-out logs stay out of fitting. Cleanup after sealing: x150/dump + x150/lab (dump holds ego/cmd; keep until sealed), then `x150/frames` (empty).

## Deliverable manifest
| artifact | location |
|---|---|
| code_x150 (sh_* copy + sh_plan_x150.py, check_contract.py, build_index_185.py, run_x150_{chain,smoke}.cmd) | F:/safehead_cache/code_x150/ ; staged copy E:/Projects/TanitAD-artifacts/x150/stage/code/ |
| PLAN.json, DOWNLOAD_FILES.md, smoke ctrl | F:/safehead_cache/x150/plan/ ; staged E:/Projects/TanitAD-artifacts/x150/stage/plan/ |
| the cache (growing) | F:/safehead_cache/x150/out/ + index.csv - ONLY copy, F: only |
Repo mirror path: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-28-safehead-data-x150/`.
