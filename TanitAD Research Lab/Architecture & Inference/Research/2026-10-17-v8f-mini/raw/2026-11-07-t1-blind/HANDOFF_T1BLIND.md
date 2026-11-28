# HANDOFF_T1BLIND (2026-11-07 ~05:05 dev-box clock). Folder `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/t1_blind/` (`$PK`)
Card `../PREREG_T1_BLIND.md`: frozen prefix 4,574 bytes, sha256 9619414a... (`head -c 4574 PREREG_T1_BLIND.md | sha256sum`), re-check after the section-7 append. Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`, CUDA hidden. Nothing staged, committed, pushed, deleted.

## DONE
* `t1blind_run.py` = thin wrapper over the pinned RIC0 tool (sha 0417c838...): patches `build_item` (only `feats` := pinned mean fp16 [40,1024], sha256 cd6a7ab5...) and `V8FMini.forward_infer` (receipt: bitwise compare of every RECEIVED frame to the mean; per-row digest of every other received input). Modes off | blind | mutant | mutant-ego. **Do not edit it**: the done markers pin its sha.
* `t1blind_gates.py` (G-ID / G-MEAN / G-EGO as pure functions), `t1blind_read.py` (card section 5 applied; refuses unless `raw/GATES.json` PASS), `t1blind_result.py`, `t1blind_dk.py` (distance keeping beside from the TICK-0 `plan_raw` of the tick dumps, same tool for RIC0 and BLIND1), `chain_t1blind.sh/.cmd`.
* Tests: `tests/test_t1blind_gates.py` 8 passed (synthetic + 5 real 64-window smoke rollouts: blind PASSES G-MEAN on 3,072 frames and G-EGO; off / token-unchanged mutant / SOURCE mutant with the replacement line deleted all FAIL G-MEAN; ego mutant FAILS G-EGO; one flipped bit FAILS G-ID); `tests/test_t1blind_rules.py` 7 passed. Log `logs/pytest_gates.out`.
* Self-tests on banked data (in `F:/TanitAD-scratch/motion_c/t1_blind/selftest/`): G-ID on RIC0_s0 re-extracted = PASS (arrays byte-equal, 19 deltas exactly 0); reader with BLIND1 := RIC0 = B exactly 0, VISION-UNUSED (known answer); result writer and dk script ran.

## RUNNING: detached chain (WMI pid 56676, started 04:58:50), resumable
Progress at 05:03: stage A done. **G-ID seed 0 = PASS** (ID_s0 wall 213 s; the off-mode rollout's array file is sha256-identical to the pinned RIC0_s0, 31e2ad8d...; tick dump and all 141 episode dumps byte-equal; the 19 deltas read exactly 0; `raw/GATE_ID_s0.json`). ID_s1 was running.
Stages: A ID_s0 (off) + G-ID s0 | B ID_s1 + G-ID s1 | C BLIND1_s0 | D BLIND1_s1 | E `gates all` -> `raw/GATES.json` | F reader -> `raw/READ_T1BLIND.json` | G `RESULT_T1BLIND.md/.json`. ~5 min per rollout, ~25 min total. Log `logs/chain_t1blind.out` (grep ZZT1B). Markers in `raw/`: `CHAIN_STAGE_<A..G>.done`, **`CHAIN_T1BLIND.done`**, `.gidfail`, `.gatefail`, `.fail`. Rollouts/receipts: `F:/TanitAD-scratch/motion_c/t1_blind/out/`.

## NEXT (fresh agent, Sonnet, <=100k context)
1. Wait for a marker (`until [ -f raw/CHAIN_T1BLIND.done ] || ls raw/CHAIN_T1BLIND.*fail* ...`). `.gidfail` / `.gatefail`: NO row is read (card section 3); report which gate and why from `raw/GATE_ID_s*.json` / `raw/GATES.json`; stop. `.fail`: read the log it names, fix, relaunch: `powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\t1_blind\chain_t1blind.cmd'}"`.
2. On `.done`: read `RESULT_T1BLIND.md`. Then `python -I t1blind_dk.py` (~1 min) -> `raw/DK141_TABLE_T1BLIND.md` (beside; tick-0 plan only, not the executed rollout).
3. Append ONE dated entry to the card section 7 (`date` for the time; append-only): outcome word, gates, ADE6 and M1 B with CIs at both seeds, the next step the card commits for that outcome. Re-check the 4,574-byte prefix hash.
4. Hand-back <=300 words: outcome word, gate results, M1 and ADE6 B with CIs at both seeds, other separated rows, manifest, unverified items.

## Disclose in the report
* The card does not say how "At both seeds" binds its "or"; `t1blind_read.py` pins it per seed (ADE6 clause or M1 clause at EACH seed) and prints the stricter and at-least-one-seed readings beside (`outcome_reading_beside`).
* G-EGO compares tick-0 digests (logged history) only; at ticks 1-2 the ego history is the executed motion and differs when the plan differs. The wrapper also counts items whose non-feats keys changed identity (must be 0).
* `frame_ok` is left unchanged (only the tokens change). Floor (P10) items also carry the mean but the floor reads no tokens.
* `logs/extract_RIC0_s0.log` comes from the G-ID self-test, not from a BLIND1 run.
* Machine: C: ~12 GB, E: ~3.7 GB, F: ~6.9 GB free at start; rollouts use ~60 MB each on F:.
