# HANDOFF_WSPD (WSP-GAIN-D agent, 2026-11-07 ~04:25 dev-box clock). Folder `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_edit_d/`
Card `PREREG_WSP_GAIN_D.md`: frozen prefix sha256 d6e691a3... over its first 5,211 bytes, verified again after every append (`head -c 5211 PREREG_WSP_GAIN_D.md | sha256sum`). Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, CUDA hidden. Nothing staged, committed or pushed. The banked `wsp_edit/` was not edited.

## DONE (all pinned in card section 6, entries 04:08 PIN, erratum, 04:09 fix, 04:1x second PIN)
* g_data (20,000 TRAIN fit windows): ds/dv0 0.9303 [0.9126, 0.9483], ds/da0 1.3637 [1.3374, 1.3901]; P5 beside 0.9355 / 1.3841. Delta-g (+0.160, +0.156) s0, (+0.177, +0.233) s1; delta-m +0.469 / +0.478 m/s. Ckpts `E:/Projects/TanitAD-artifacts/v8f-wspedit-d/M2b_s{0,1}_wspedit_d/ckpt_last.pt` (sha256 30bdd83c... / 8d89c610...).
* Code with the BASE targets reproduces the banked E-GAIN alpha 1.0 ckpts byte for byte (`raw/REPRO_BANKED.json`). 23 tests pass, 14 of 14 mutations RED, the card's null test (g_data := g_model gives C0 byte for byte) passes on the real shards.
* **READ 1 on the TRAIN fit windows = MECH-OK** (`raw/READ1_D.json`): s0 dS/dv0 0.8533 (8.3 % off 0.9303), dS/da0 1.3168 (3.4 %); s1 0.8427 (9.4 %), 1.2980 (4.8 %). Eval-141 gains beside: `raw/READ1_D_EVAL.json` (s0 0.8747 / 1.3531; s1 0.8443 / 1.2731).

## RUNNING: detached chain `chain_d.cmd` -> `chain_d.sh` (started 04:08; resumable, finished stages are skipped)
Stages E (two T1-kin rollouts RIWSPD_s0, RIWSPD_s1, ~4-5 min each) and F (reader `wspd_read.py`) were running at 04:25; then G = `dk_beside.sh` (distance keeping beside; non-fatal, ~30 min). Markers in `raw/`: `CHAIN_STAGE_<A..F>.done`, **`CHAIN_D.done`** (READ 2 written), `CHAIN_D.fail`, `CHAIN_D.mechfail`, `DK_BESIDE.finished`. Log: `logs/chain_d.out` (grep `ZZWSPD`).

## NEXT (fresh agent, Sonnet)
1. Wait for `raw/CHAIN_D.done` (or read `raw/CHAIN_D.fail` + the named log; fix; relaunch: `powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\wsp_edit_d\chain_d.cmd'}"`). If the reader alone failed, rerun it: `python -I wspd_read.py --out-root F:/TanitAD-scratch/motion_c/wsp_edit_d/out --work-dir F:/TanitAD-scratch/motion_c/wsp_edit_d/work --raw <dir>/raw --workers 4` (rollouts stay banked).
2. `python -I result_d.py` -> `RESULT_WSPD.md/.json` (verdict word by card section 4; `result_d.py` was smoke-tested on READ 1 and on the banked WSP-GAIN READ 2 in `logs/_rtest_result_writer_sandbox/`, which reproduced the banked HARMS rows). If DK finished, mention `raw/DK141_TABLE_D.md`; if `dk_beside.sh` failed, rerun `bash dk_beside.sh` (C0_s0 eval is already banked in F:/.../wsp_edit_d/dk).
3. Append a dated result entry to the card's section 6 (append-only, `date` for the time), name the outcome by section 4, give the next step it commits. Any change to a pinned script needs its own dated entry with the new sha256.
4. Report: M1 and every guard row for both seeds vs C0 and vs P10, four families (STRATEGIC PAUSED), distance keeping beside, manifest. No T2 stage unless MOVES-M1 with guards holding.

## Machine state to know
`C:` has ~40 MB free (Claude temp lives there); `E:` ~4 GB, `F:` ~12 GB free. Nothing was deleted. This agent's scratch: `F:/TanitAD-scratch/motion_c/wsp_edit_d/` (rollouts ~60 MB, dk evals); `E:` ckpts ~350 MB incl. `_repro_banked/`. The CPU is shared with the yawrate / navsim agents (load 100 %).
