# HANDOFF -- stream YSEL2 (fresh-token confirmation of H-SEL-1, gBG(0.20, 0.05)), written 2026-11-07

`$P` = `E:/Projects/TanitAD-artifacts/refe_selection_confirm_20261107/`. Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (chain, scoring, analysis);
`C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe` (dump, harness). Nothing staged, committed or pushed; the folder is outside the repo (MM lands it).
**The MM lands `PREREG_SEL_CONFIRM.md` + `.sha256` + `CODE_PIN.json` + `tokens_confirm.json` BEFORE reading any result.**

## Frozen: `PREREG_SEL_CONFIRM.md` sha256 `ad12c2d9204eb7be74c9b500e885d2cbf0394dc027e2a5d0459b8634f2781bab`; `CODE_PIN.json` sha256 `86e7b9fd...` (pin guard RED-tested on a scratch copy: one appended byte -> REFUSED).

## State at hand-off (19:25): launched 19:23:47 (cmd PID 74236 -> python, PID in `markers/selc_full.STARTED`). Preflight GREEN (pins, G-RULE 1058/1058 + RED 175, G-ANA replay max diff 0.0).
`dump_smoke` is QUEUED for the GPU: no lock file is held, but VRAM read 4,277 MiB (> the 3,500 bar) -- desktop apps plus a FOREIGN compute process
`C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u r0_capladder.py` (PID 82636, started 19:18, type C, running WITHOUT the shared lock). Not ours; not touched.
The dump polls every 60 s and gives up after 10 h (-> `selc_full.FAILED`, rc 5). If it is still queued for hours, that process / desktop VRAM is the blocker (PI/MM call).

## Chain: ONE detached `selc_chain.py` (WMI, CreateFlags 0x08000200)
Stages (idempotent, `stages/<name>.json` rc 0 = done): preflight -> dump_smoke (GPU) -> score_smoke -> dump (7 GPU segments, lock released + 180 s
yield between them, so YAXN can take the GPU) -> consolidate -> verdict -> analyze -> table (448 harness calls) -> table_read.
ETA (ESTIMATED, `POWER.md` §4): verdict ~4 h + GPU lock waits; full table a further 4.6-6.2 h.

## How to check (one command, no model turns)
`ls $P/markers; tail -5 $P/CHAIN.log` -- then, by stage: `dump/dump.log`, `score/score.log`, `logs/*.log`, `logs/retries.jsonl`, `score/retries.jsonl`.

## Outcomes and what to do
* **`markers/selc_verdict.DONE`** -> read `CONFIRM_RESULT.md` / `confirm_result.json`. First check the gates are green (they gate the stages, but quote them):
  `gate_preflight.json`, `gate_ana_replay.json`, `dump/smoke_identity.json` (GATE), `score/smoke_score_gate.json` (GATE, 384/384 cells), `score/verdict_gates.json` (PASS).
  Report the verdict by `PREREG_SEL_CONFIRM.md` §4 (ADOPT-for-sim / REFUTED / NOT PROVEN), the four families (TACTICAL n/a, STRATEGIC PAUSED), tier T0,
  and the by-log-class line. MM updates `H-SEL-1` in `GOALS_AND_CLAIMS.md` from the tip, and acts on the §4 consequence line (Rule Zero).
* **`markers/selc_full.DONE`** -> also `CONFIRM_TABLE_APPENDIX.md`: the SECONDARY pre-registered `H-SEL-LANE-1` (L(0.25)) verdict (§7) + descriptive rules.
* **`markers/selc_full.FAILED`** -> read `stage` and `why`. Typical: `dump_smoke` rc 2 (GPU identity not bit-for-bit: compare `dump/smoke_identity.json` `ok`
  fields; do NOT loosen the gate), `score_smoke` (harness: `score/logs_smoke/*.log`), GPU busy 10 h (rc 5), disk < 3 GiB, RAM (MemoryError retries exhausted).
  To resume: fix the cause; if any pinned file changes, write `AMENDMENT_1.md` (what, why, before any result was read) and re-freeze (`make_pin.py`,
  update the CODE_PIN sha in the prereg, new `PREREG_SEL_CONFIRM.sha256`); delete the FAILED marker (literal path); relaunch exactly as below.
  Done stages are skipped; dump chunks and harness columns already PASSed are not redone.

## Relaunch command (inline PowerShell; `.ps1` is blocked)
```
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'cmd /c "cd /d E:\Projects\TanitAD-artifacts\refe_selection_confirm_20261107 && C:\Users\Admin\venvs\tanitad\Scripts\python.exe selc_chain.py < NUL >> E:\Projects\TanitAD-artifacts\refe_selection_confirm_20261107\markers\chain_stdout.log 2>&1"'; ProcessStartupInformation = $si }
```

## Known gaps (named)
* GPU bit-identity of the fresh dump is asserted only on the 6 smoke C3 tokens (+ G0-fresh pick reproduction + fidelity vs the banked full-navtest seam on all
  5,877). The CPU plumbing dump is NOT bit-identical (max 5.7e-4 m, `PREREG` §0), so the smoke MUST run on the GPU -- it does (dump_smoke).
* The slow-down clause (ii) is the at-risk condition: C3's TRAIN-random CRUISE drop (-6.7 pts) would give it power 0.25; held-out/pooled give 1.00/0.99.
* Harness speed under today's contention was ~2x C3's (2-token calls 34-56 s); the table may pass the 12 h envelope; the verdict does not depend on it.
* Seams + harness run dirs on `F:/TanitAD_selc_20261107/` (data dir, not for the repo). Plumbing leftovers: `F:/TanitAD_selc_20261107/_plumb/` and the
  session scratchpad (`plumb/`, `plumb2/`); harmless, never read by the chain (distinct labels `refe_selcf_*`).

## ADDENDUM 2026-11-08 09:08 (stream YSEL2-FIX): AMENDMENT 1 implemented, chain relaunched
* `selc_dump.py`: `frame_ctrl_meta` (export vs DB 4.5 s/9 rows at the export's own meta timestamps, tol 1 mm; nan = FAIL, also an all-nan log now fails) replaces the nominal-grid reference in `frame_ctrl_m`; old reference kept as `frame_ctrl_old` for the proof only.
* `selc_analyze.py`: human-ADE secondary masks the 17 tokens of `fc_masked_tokens.json` (new, pinned); reports `n_human_masked_cruise` / `_all`. `selc_score.py`: `check_pins` also accepts the re-pin via `AMENDMENT_1_CODE_PIN.sha256` (valid only while the amendment text matches its sha). `make_pin.py` pins `fc_masked_tokens.json`.
* Proof `fc_diag/fc_fix_result.json` (CPU, 5,877 tokens): new 5877/5877 pass (max 7.3e-15 m), old reproduces exactly the 17, RED 0.5 s shift fails 5877/5877, RED 0.05 rad rotation fails 5877/5877. GATE true.
* CODE_PIN.json re-pinned (sha a4b84011...; old copy `CODE_PIN.before_am1.json`); pins OK, G-RULE PASS, G-ANA replay max diff 0.0. Preflight stage re-run by the chain (old record `stages/preflight.attempt1.json.bak`).
* Relaunched 09:07:53 (chain PID 10192; dump queued on the GPU lock). Old markers kept as `selc_full.FAILED.attempt1_framectrl`, `selc_full.STARTED.attempt1`.
* Other consumers of `human_future_poses`: meta-only count over ALL 12,146 navtest tokens (`fc_diag/fc_all_navtest_meta_count.json`): 36 tokens in 4 logs have a missing/shifted OpenScene frame in their 8-step future (17 are ours; 19 elsewhere).
