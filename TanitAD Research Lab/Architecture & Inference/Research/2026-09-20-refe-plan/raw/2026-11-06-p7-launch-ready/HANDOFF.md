# HANDOFF: P7-LAUNCH-READY (2) + (3) + PKG-SHIP-COMPLETE (0 GPU, 2026-11-06)

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Files here mirror `PK/`: `refe/{p7_pod.sh,p7_prep.py,p7_ship_list.py,p7_ship_list.json,test_p7_chain.py,test_p7_ship_list.py}`. Nothing staged, committed or pushed; mirror and E: untouched. Tests ran on a raw-blob export of `origin/agent/arch-inf-20260803` (`14a58248`) + these files, venv `driverl-eval`.

**Tests (MEASURED, CPU, `driverl-eval`, overlay tree = tip export + these files):** `test_p7_chain.py` ZZP7CHAIN_PASS 81 (16 static mutants, 22 sandbox refusal arms, 5 behavioural mutants, all RED as required); `test_p7_ship_list.py` ZZP7SHIP_TEST_PASS 20; `test_p7_integration.py` ZZP7INT_PASS 221; `test_launch_gate.py --static --no-banked` ZZLG_STATIC_PASS 1169; `test_p2r_chain.py` ZZTEST_PASS 139.
**Job still running when this was written (detached via WMI, ~45-60 min):** the traced derive of the whole refe_p7 suite (+ the two new tests) on `C:/Users/Admin/px2` -> `par18/p7pod/_trace_run.out` (one `ZZP7SHIP_TRACE <script> rc= result=` line per script = every suite count, last line `rc=0`), then `refe/p7_ship_list.json` and `p7_trace.json`. If a script fails the derive stops ("a failing suite is not a list"): read the line, then run `derive` without `--trace`. The static-only list (seconds) already finds the four files P6 lost plus three more (`eval/P2REP_POD_RUNBOOK.md`, `eval/S1_POD_RUNBOOK.md`, `raw/2026-10-13-aux-head/PREREG_BLANKS_PROPOSED.md`; over-approximation by design).

## Dev box, before shipping (the order matters)
1. Land the six files (scoped lander). Then **re-derive the ship list at the landed bytes** (it binds the suite scripts' sha256, so any landed change makes it STALE and the pod refuses): `python refe/p7_ship_list.py derive --pkg <PK> --trace-file <trace.json> --extra-script test_p7_chain.py:ZZP7CHAIN_PASS --out refe/p7_ship_list.json` (add `--trace` to re-take the ~45 min traced run). THEN `launch_gate.py ship-manifest --commit <landed> --pkg <PK> --out <PK>/SHIP_MANIFEST.json` (it hashes the list).
2. Ship to `/workspace/refe-plan-p7` (NOT P6's dir): `code/ refe/ splits/ SHIP_MANIFEST.json` + every file the list names outside them (`eval/rls1{b,d,e}_arms.json`, `raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md`, ... : `p7_ship_list.py check` prints the first missing one).
3. Ship the **full aux label store** (103,139 keys, 351 MB) to `$DATA/refe_navtrain/aux_label_store_p7`. It is **not on HF**; its only confirmed copy is `D:/Projects/TanitAD-artifacts/aux_drivable/label_store`.

## Pod (L40, local disk): start it while P6 still trains; it waits at no load
```
export HF_TOKEN=$(grep -oE 'hf_[A-Za-z0-9]+' <TOKEN_FILE> | head -1)
P7_LABEL_WORKERS=6 P7_WAIT_P6=1 nohup setsid bash /workspace/refe-plan-p7/refe/p7_pod.sh > /workspace/data/refe_p7/p7_stdout.log 2>&1 < /dev/null &
```
Optional knobs: `P7_RUN`, `P7_REPO` (defaults `refe_runs/p7_reserve_head`, `Sayood/tanitad-refe-p7-reserve-head`; either equal to P6's REFUSES), `P7_CBANK` (default the 3-C bank; the reserve stage writes `<it>_p7`), `P7_TWIN_SRC`, `P7_STOP_AFTER`. Watch `refe_p7/chain.log`: `ZZP7_WAITING_FOR_P6`, then `ship list ok`, `ZZP7LOCAL_OK` x2, `P7 reserve ok`, `D7 twins`, `D7 rank 1..3`, `launch gate PASS`, `ZZP7_PINS_OK`, `ZZP7_NOPARTS_OK`. Any `REFUSED:` ends the chain; fix the cause, rerun the same command (stages are idempotent).

**Which twins the link places, and why.** Only the `lane_rank` rows of `train_grow_cmd_3c_p7/targets_rank*.jsonl` (the reserve-free bank, 57,707 rank-1 rows): `<log>/<token>__lane_rank_<k>.pkl.xz` from `metric_cache_twins_3c` into `metric_cache`. That is P6's kept set (21,821 distinct) minus the 116 reserve logs' twins; the count prints in `twins_plan.out` (UNVERIFIED until then). Placing a reserve log's twin would recreate `<log>/` in the cache after the reserve move and fail the G-RESERVE cache leg; the 22,588 dropped twins are not rows of the bank. Each row's logged part must already be in the cache, so the link follows the cache stage. After P6 the cache already holds them, so the plan should read `to_place=0`.

## Open risks / UNVERIFIED without a pod
- **GPU idle gap, ESTIMATED 2.5-4 h**, not zero: the reserve stage MOVES dirs out of the cache and pixel root that P6's trainer reads, so P7 cannot prep beside P6 (the chain refuses while P6's lock is held). Levers not built: `launch_gate.py suite-prerun` for G-SUITE, a pre-staged reserve-free copy.
- `test_p7_chain.py` is **not in refe_p7's G-SUITE** (adding it is a `launch_gate.py` edit, then a new tree hash: MM call). Its ~5 min and the 7 suite scripts' ~40 min are CPU.
- Step F (the pod's own reserve filter reproduces the pins `92af8d66`, 150,603, `df2ddd1b`) and the full five-input G-RESERVE are first exercised on the pod. Hard link vs copy on the pod's disk: `P7_TWIN_HOW=auto` falls back to copy. The pins kill was tested under Git-bash `kill`, not on Linux. `no_parts` threshold 0.001 and 20 events: ESTIMATED (R67 read 0.106, P6 0 of 944,101).
- L40 (48 GB) vs A40: G-LIVE-MEM's 44 GB bound, the head's step time, and G-LIVE-LAB's 6.735 sets/s at 6 workers (measured on an 8.5-CPU pod) are all re-measured by the gate there; `--label-workers` is pinned, never derived from the 15.3-CPU quota.
- SPEC_P7 section 3's HF-bank receipt (G-EGR bank leg, restore test) is **not built**; the bank is verified by G-CLOCK's content pin instead.
- The reserve move is reversible (`p6_prep.py reserve-move --restore`); later P6 reads of reserve logs use `${MC}_reserve` / `${PIX}_reserve`.

## ADDENDUM 2026-11-06 ~15:00Z (MM, session c216c849; supersedes the items it names above)
- **The traced derive finished** (MEASURED, `_trace_run.out` here, now complete): 9/9 suite scripts rc 0, 293 files, 9 outside roots, closure `302fe5870a30`; `p7_trace.json` is the trace. The 3-line `_trace_run.out` landed in ef78c0c5 was a partial copy.
- **`test_p7_chain.py` is now IN refe_p7's G-SUITE** (`launch_gate.py` `P7_SUITE_EXTRA`, as refe_p6 carries `test_p6_chain.py`); `test_p7_integration.py`'s IV suite pin follows. The `--extra-script test_p7_chain.py:ZZP7CHAIN_PASS` in step 1 is no longer needed.
  - MEASURED on the dev box: `launch_gate.py suite-prerun --profile refe_p7` over the tip export + these files. Its first run FAILED on test_p7_chain (rc 1), which caught a false positive of the FD200 rule on a refusal text containing the word "curl"; the text was reworded. The re-run PASSED: 8/8 scripts rc 0, 0 skips, 735 s (`suite_prerun_PASS1.*`; the failed first run is kept as `suite_prerun_FAIL1.*`). After the ship-list fix below, a third run on exactly the landed bytes is `suite_prerun_FINAL.*`.
  - The trace binds the suite closure, so the ship list was re-derived with a fresh `--trace` at these bytes (`_trace_run.out`, `p7_trace.json`), not with `--allow-stale-trace`.
- **The aux label store IS on HF** (corrects step 3 and the old refusal text). MEASURED: `refe_aux_drivable_label_store_2026-10-23.tar` (353,802,240 B) in `Sayood/tanitad-refe-navtrain-banks` equals `E:/Projects/TanitAD-artifacts/aux_drivable/label_store` (the drive letter moved D: → E:): 1,196/1,196 files sha256-equal, `store_sha256` 4a17db67…, 103,139 keys. On the pod, before the chain:
  ```
  mkdir -p /workspace/data/refe_navtrain && cd /workspace/data/refe_navtrain
  printf 'header = "Authorization: Bearer %s"\n' "$(cat <TOKEN_FILE>)" | curl -sS -f -L -K - -o /root/aux_store.tar https://huggingface.co/datasets/Sayood/tanitad-refe-navtrain-banks/resolve/main/refe_aux_drivable_label_store_2026-10-23.tar
  [ "$(stat -c %s /root/aux_store.tar)" = 353802240 ] && tar -xf /root/aux_store.tar && mv label_store aux_label_store_p7
  python3 -c "import json;m=json.load(open('aux_label_store_p7/manifest.json'));print(m['store_sha256'][:8], m['coverage_vs_navtrain_yaml']['stored_in_yaml'])"   # expect 4a17db67 103139
  ```
- **`P7_LABEL_WORKERS=13` on the L40, not 6.** MEASURED 2026-11-06: P6 at 6 workers REFUSED G-LIVE-LAB on this pod (stall 5.4 % > 5 %; PREREG_P6 Amendment (e), `raw/2026-11-06-p6-l40-gate/`). At 13 it trains at stall ~1 %. SPEC_P7 pins no worker count, and P7 is a fresh run with no argv sha to keep. The value stays PINNED by the env (never derived), as the chain requires.
- **Schedule and cost (ESTIMATED from P6's MEASURED L40 rate of 50.5 s/step):**
  - 9,075 steps × ~50.5 s ≈ 127 h, plus a gap of 2.5–4 h, ≈ 5.5 days;
  - at $0.90/h that is about $118.
- **Canonical files after this landing:** `PK/refe/{p7_pod.sh,p7_prep.py,p7_ship_list.py,p7_ship_list.json,test_p7_chain.py,test_p7_ship_list.py}` and `PK/refe/launch_gate.py`.
  - The `refe/` copies in THIS directory are the morning versions (ef78c0c5). `p7_pod.sh` and `test_p7_chain.py` there predate the text fixes above. Do not ship them.
  - `p7_ship_list.json` was derived by a fresh traced run at exactly the landed bytes: 292 files, 8 outside roots, closure `fccec3323a4b`, and `check` reads `ZZP7SHIP_OK` on the package. So step 1's "re-derive at the landed bytes" is DONE for this commit; `SHIP_MANIFEST.json` is still made at ship time.
- **DEFECT FOUND AND FIXED before any pod time: the ship list could never pass `check` on a real package.**
  - MEASURED 2026-11-06: `derive` pinned the sha256 of files made AFTER it ran. These were the list itself (hashed before being overwritten), its default trace sidecar `refe/p7_ship_list.json.trace.json`, and `SHIP_MANIFEST.json`, which is regenerated at ship time for the landed commit.
  - The suite's tree readers glob `refe/` and read the manifest, so all three were traced. `check` then read `differs=1..2` on every real package.
  - The consequence: `p7_pod.sh` line 123 refuses on that, so the P7 chain would have stopped at its first stage on the pod.
  - The synthetic test missed it because its list sat outside the package and its suite never read the manifest.
  - **Fix:** those files are presence-only; `REQUIRED_TOP` still requires the manifest.
  - **Test:** `test_p7_ship_list.py` gains 5 checks (25 total), including two controls: a traced suite that really opens the manifest, and the manifest still required.
  - **Mutation proof:** the unfixed tool goes RED on exactly the 3 new non-control checks (`differs` on `SHIP_MANIFEST.json` and the list itself).
- **The P6 read is unaffected by this landing** (`eval/` is untouched here; its layout fix is a7ec6e84, PREREG_P6 Amendment (f)).
