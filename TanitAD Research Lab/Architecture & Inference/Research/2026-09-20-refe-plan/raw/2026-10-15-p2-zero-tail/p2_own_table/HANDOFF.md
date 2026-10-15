# HANDOFF: P2's own 64-proposal table on its 509 zero tokens (written 2026-10-15 22:30 Berlin)

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. PACKAGE DIR = `C:/Users/Admin/AppData/Local/Temp/p2t/<PK>/raw/2026-10-15-p2-zero-tail/p2_own_table/` (code = archive of `ddc9ce9`; E:/D: trees untouched).

## State
ONE detached chain, launched 22:23 via WMI with `CreateFlags 0x08000200` (NEW_PROCESS_GROUP | NO_WINDOW). Two earlier launches WITHOUT the flags died with a bare `^C` in the log (cause not found; a Ctrl-C probe of powershell/tasklist/nvidia-smi did not reproduce it). Relaunch with the flags.
PIDs: cmd 5220 -> python 13708 -> **driver 17596** (the pid in `STATUS.json` and in the lock body) -> seam 20296 / 20908.
Stages: dump (GPU, `devbox_gpu.lock`, takes it only when `memory.used <= 3500 MiB`; ~2 s/token, **~17 min**, ends ~22:41) -> fidelity gate (P2's pick vs the banked seam, 1e-3 m; **STOPS before any CPU scoring**) -> 64 harness columns x 509 tokens, 4 workers (~25 min) -> table (G2) -> strict same-pick proof -> `P2_MEASURED.json` -> copy to D:. **ETA `P2TABLE_DONE` ~23:15 Berlin.**

## Check (no model turns)
```
cd "<package dir>"; ls P2TABLE_DONE P2TABLE_FAILED 2>/dev/null; tail -3 driver_main.log | cut -c1-300; grep -E "^\s+\[[0-9]+/86\]" logs/1_dump.log | tail -1; ls C:/Users/Admin/AppData/Local/Temp/p2t/score_p2tab | wc -l
```
Result: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I -c "import json;r=json.load(open('P2_MEASURED.json'));print(json.dumps(r['reading'],indent=1));print(json.dumps(r['Q2_cause_split'],indent=0)[:1500])"`, plus `same_pick_proof.json`.
Marker `P2TABLE_FAILED` holds the reason (rc 1 dump/preflight, **2 = fidelity gate = P2's pick not reproduced: report it, do not re-run**, 3 score, 4 table G2/G3, or "SAME-PICK PROOF FAILED").
Resume after a crash: same WMI command (`p2tab_driver.py --tag main --workers 4`); done stages and done score columns are skipped. CPU-only redo of proof + analysis: `--from-stage proof`.

## Same-pick proof
* **CPU smoke, 8 of the 509: PASS** (`_smoke8/SMOKE8_PROOF.json`). 8/8 dumped picks are the SAME PROPOSAL as P2's banked pick (nearest of 64 by 484x to 3111x margin); table PDMS at the pick == banked csv, 8/8 exact (0.0 = 0.0); six sub-scores equal.
  **Deviation, flagged:** on CPU the pose diff was 1.2e-4 .. 1.36e-3 m, so 1 of 8 exceeded C3's 1e-3 m bar (`_smoke8/attempt1_gate_fidelity_cpu.json`). That bar was registered GPU-vs-GPU. The smoke criterion (same proposal, margin >= 100x, d <= 5 mm) was fixed AFTER seeing those 8 diffs. Zero tokens carry EP = 0 and discrete other terms, so score equality is a weak test; the pose test carries the precision.
* **GPU, all 509: strict** (pose <= 1e-3 m, expected bit-identical as in C3; PDMS at pick <= 5e-5 of the csv and exactly 0.0; sub-scores). Not run yet. Written to `same_pick_proof.json`.

## Not verified
The GPU fidelity gate, all 64 x 509 scores, the strict proof and every SEL/PROP number are NOT yet produced. `measured()` was dry-run on a SYNTHETIC table only (it reproduced RESULT.json's V1 439/70 bridge exactly); the dry output was discarded.

## Landed-code caveats found
* W3's `cmd_score` ends with `os.path.relpath(csv, PKG)`; a C: archive and a D: score root crashed it AFTER scoring (`_smoke8/attempt2_relpath_failure_p01.log`). Fixed in `c3_run_p2.py` by `SCORE_ROOT = C:/Users/Admin/AppData/Local/Temp/p2t/score_p2tab` (in `c3_run_vs_p2.diff`).
* P2's scorer has **6** logits (u400: 7).

## Manifest (all in the package dir unless noted)
`make_tokens_p2.py`, `tokens_p2zero.json` (sha256 c6b5cae6...), `tokens_p2zero_smoke8.json`, `c3_run_p2.py` + `c3_run_vs_p2.diff` (original blob sha256 94a5a6ed...), `p2tab_driver.py`, `test_p2tab.py` + `test_p2tab.log` (9 OK, 4 deliberate-regression arms RED), `_smoke8/`, `driver_*.log`; AFTER the job: `table.npz`, `proposals.npz`, `same_pick_proof.json`, `P2_MEASURED.json`, `p2_zero_tokens_sel_prop.csv`, `gate_*.json`, `STATUS.json`, markers. Second copy of these at `D:/Projects/TanitAD-artifacts/p2_zero_tail_table/` (+ `MANIFEST.sha256.json`). Not in git: the 64 seams `D:/Projects/TanitAD/data/refe_navtest/seams/p2tab/main/` and the 64 score dirs on C:.
