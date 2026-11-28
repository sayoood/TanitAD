# HANDOFF NATIVE40 (agent, 2026-11-19)

## State
* Built, not run: free commit memory was 1.8-2.0 GB (gate 5.5 GB). No model was loaded; no head number exists.
* Checkpoints verified by sha256 against SHA256SUMS.txt and the LLR read's bank record (llr_read/readout/LLR_READOUT.json):
  * LLR_s0 `E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_recft_LLR/ckpt_last.pt` 5a52b28b...6817bc
  * LLR_s1 `.../M2b_s1_recft_LLR/ckpt_last.pt` 71ee9fea...835f1e9dd... (full: 71ee9feab2683a66a80e6a1fcbd4bc23e8a961ee7ef702ed160b00835cbf40c6)
* Code of record: the LLR read tree `F:/TanitAD-scratch/motion_c/llr_read/tree` (model v8f_mini.py sha 5fd4baea, tool v8f_t1_eval_ri.py sha 0417c838, both match the LLR pins). Mirror source commit for the LLR read files: 5f5f8e26 (branch agent/arch-inf-20260803; 1d13be96 is an ancestor).
* Head read: `model._encode` then `model._lead` (the same call `_forward` makes, v8f_mini.py:870-875); gap_hat = sigmoid(2nd output) = gap/40, so the gap in metres = lead_gap * 40 (LEAD_RANGE_M 40, v8f_mini.py:148). The smoke also checks `forward_infer` returns `lead_logit` and `lead_gap` and that a head-off model does not.
* Labels: `lead_eval139.npz` (row sha12(clip), raw frame k = 2 * cache step), same file and rule as the training labels (`lead_label_arrays`, v8f_mini_data.py:260-300). Evaluation set: the LLR read's own eval cache, `eval_grid_windows.json` (141 episodes, 2,399 windows), `join_eval141.npz`.

## MAP CONTROL RESULT (labels only, no head number): THE CARD'S MAP FAILS
* Coordinator list `LS_REF40_WINDOWS.json` sha256 166198fb4cbd9da116689cc47380726af053a7dced5e91ca3c1db8448f41c0be. All 54 episodes are in our 141; all 534 (eid, t) are eval-grid windows.
* Source-derived map: our window = (episode, cache step t), label raw frame k = 2t. Our label gap equals `gap_t` (1e-3 m) for only **12 of 534 (2.2 %)**.
* Their `gap_t` equals our label at raw frame **k = 2 * g0** for **533 of 534 (99.8 %)** (t - g0 = 11 on every window). So the coordinator's gap is the label at the window START (cache step g0), not at the tick t the head reads. k = t: 22, k = g0: 20 of 534.
* Under the MM's rule this is NOT-COMPUTABLE (share 0.022 < 0.95). The scorer says so. The chain also reads the head at cache step g0 for the 534 windows (`--extra-windows`), so one MM decision (an amendment of the map: head read at the frame the coordinator's label belongs to, or ask the coordinator which frame their heads read) needs no second dump.

## Exact next step (MM starts it; it gates itself on >= 5.5 GB free commit, polls 60 s, cap 6 h per stage)
Launch (WMI, CreateFlags 0x08000200): `cmd /c C:\Users\Admin\actsens_launch\native40_chain.cmd`
Log `F:/TanitAD-scratch/native40/chain.log`; sentinel `ZZ NATIVE40 DONE ZZ` or `ZZ NATIVE40 FAIL <why> ZZ`. Waiter: `bash /c/Users/Admin/actsens_launch/wait_native40.sh`.
Chain: smoke (2 episodes, key check, no metric) -> dump_s0 -> dump_s1 -> scorer; outputs in `F:/TanitAD-scratch/native40/out/` (dump_s0.npz, dump_s1.npz, NATIVE40_RESULT.json/.md). CPU only, 3 threads, one process at a time.
Manual scorer: `venvs/tanitad/Scripts/python.exe -I code/native40_score.py --d0 out/dump_s0.npz --d1 out/dump_s1.npz --windows F:/TanitAD-scratch/v8proof/leversheet/LS_REF40_WINDOWS.json --out-dir out` (Windows drive paths for python).
Unknown: wall time (ESTIMATED under 2 h by the card). Not yet exercised on the real model: the dump script (scorer tested on synthetic dumps only, then deleted).

## Files
`F:/TanitAD-scratch/native40/code/` native40_dump.py, native40_score.py, labels_only_synth.py, map_diag.py, map_diag2.py; `src/` exported LLR read files + ds_run.py (reference); `DELETIONS.log`.
