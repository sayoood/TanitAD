# HANDOFF latvis2 (2026-11-15)

State: A to G built and tested (see LATVIS2_DEFS.md). No real token-level number was read. Departures from the card are in the DEFS (ridge penalty fixed 0.01, k6 reference = the 1,948-window R1 list, lead windows exclude panel windows, L1 empty).
`_R3g` = context-3 stack rollout (MEASURED, 1.9e-5 on G_s0); `_R1` = context 1.

Pod side (Linux CPU; ship lv2_pod_tokens.py and imaghead/imag_dump.py side by side or pass --imag-dir; about 4 GB, tokens memmap ~0.8 GB in --workdir):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=<tree>/stack python lv2_pod_tokens.py --ckpt <ARM_sK>/ckpt_last.pt --tree <tree> --windows LATENTS_CTX_G_s0_R1.npz --pack eval_pack.npz --states <ARM_sK_eval_states.npz> --cache <eval cache root> --banked-roll LATENTS_CTX_<ARM>_sK_R3g.npz --panel-windows WINDOWS_LV2.json --out DUMP_LV2_<ARM>_sK.npz --workdir <dir> --device cpu
(G and RR, seeds 0 and 1; WINDOWS_LV2.json holds `window_indices`.)

Dev side:
  python lv2_read.py read --dump DUMP_LV2_G_s0.npz DUMP_LV2_G_s1.npz --arm G --seeds 0,1 --out-dir OUT/G --ref-windows LATENTS_CTX_G_s0_R1.npz   (same for RR)
  python lv2_read.py pair --a-dir OUT/RR --b-dir OUT/G --out-dir OUT/pair
  python lv2_sheet.py --windows WINDOWS_LV2.json --out-dir . --tok G_s0=DUMP_LV2_G_s0.npz --tok RR_s0=DUMP_LV2_RR_s0.npz --imag LS_s0=../imaghead/dump/DUMP_LS_s0.npz --frames-dir private/frames
Already made: WINDOWS_LV2.json (lv2_select.py), private/frames (lv2_frames.py, 18 windows), first_L4L5/ (public L4/L5 sheets of the 6 lead windows), private/sheets + sheets_public (2 windows, layout check).

Next step: the lead's note, then the pod run; then read, pair, sheets. Tests: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_latvis2.py --basetemp F:/TanitAD-scratch/v8proof/latvis2/work/pt/<new name> (about 3 min).
Temp data: latvis2/work/ (pt/ test folders run1..run11, probe_k5.py, test_part2.py, pytest_final.log). Private: latvis2/private/.
