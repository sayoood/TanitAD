# HANDOFF -- PREREG_REFCV9_INMODEL_PA Phase A (EvalFlyWheel, 2026-10-26)

**State: A1 PASS, A2 frozen + REGISTERED (MM 14:08:29), A3 under AMENDMENT1 (sha256 1c25a064...).**

* A1: `result/R9PF_A1_PREREQ.json` ALL_PASS.
* A2: `heads/*.pt` (12), `R9PF_FIT_INFO.json`, `FROZEN_R9PF_PA.sha256`; registered in `REGISTERED_R9PF_PA.sha256` (17 lines, MM).
* AMENDMENT1 change 1: `code/FIX_R9_RNG.DIFF.txt` + `FIX_R9_RNG.sha256` (fixed refc_v3.py in `F:/refcv9_r9pf_scratch/launch_tree_fix`, CRLF kept,
  only line 2125 differs). Every a3 script re-hashes the amendment seal and every pin line and refuses otherwise (`r9pf_common.check_amendment1 / check_fix_pin`).
* A3 compose (CPU): `result/R9PF_COMPOSE.json` ALL_PASS (124 controls). Packages -> `F:/refcv9_r9pf_scratch/arms/<head>/R9_POS_step48852ema_pf_s{0,1}.npz`
  (PRIMARY = the banked fb47a58 battery plan; `*_own` = eval-bank traj_own, beside) + `arms/EVAL_INPUTS_step48852ema_pf_s{0,1}.npz`.
* A3 K-OFF / K-EQ (GPU, dev box): `code/run_keq.sh` -> rolls `F:/refcv9_r9pf_scratch/keq_s{0,1}_{on,off}/` (ROLL_RECORD.json, keq_records.npz,
  OUTPUT_DIGESTS.json), then `code/a3_keq_compare.py` -> `result/R9PF_KEQ.json`. The runner polls `code/gpu_free.py` before EACH roll and never
  runs beside another agent's roll (LONHEAD). Log: `logs/run_keq.log` (ZZ...ZZ markers).
* A3 bars: `code/a3_bars.py` -> `result/R9PF_PA_BARS.json` (written to scratch as PROVISIONAL while R9PF_KEQ.json is absent).
* Verdict: `code/a3_verdict.py` -> `result/R9PF_PA_VERDICT.{json,md}`; manifest `MANIFEST_R9PF_PA.sha256`.

## Rerun order
```
cd C:/Users/Admin/refcv8_pf/refcv9_r9pf; source code/env_r9pf.sh
$PY -X utf8 code/a3_compose.py > logs/a3_compose.log 2>&1          # CPU
bash code/run_keq.sh                                                # GPU, polite; writes result/R9PF_KEQ.json
$PY -X utf8 code/a3_bars.py > logs/a3_bars.log 2>&1                # CPU
$PY -X utf8 code/a3_verdict.py > logs/a3_verdict.log 2>&1
$PY -X utf8 code/write_manifest.py
```
Judge every step on its ARTIFACT, never on an exit code through a pipe. Never feed a heredoc AND `</dev/null` to python on this box (NUL is a
console device -> an interactive REPL that loops; deviation D18).
