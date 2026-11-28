# HANDOFF — ADD155 World Understanding Audit (Phase A), 2026-11-28 ~12:20 local

State
* Card `F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD155.md` sha256 dcdefe34… (sidecar + TIMES_NOTE). AM1 = `wua/a5/IMPL_A5_AM1.md` (A5 history 16 frames). AM2 = `PREREG_V8F_PROOF_ADD155_AM2.md` c7ffb661… (A5 centre-matched control C-CENTRE, post hoc in origin).
* Tools `F:/TanitAD-scratch/v8proof/wua/`: wua_extract.py, wua_labels.py, wua_a1.py, wua_imag.py, wua_card.py, run_extract.sh, run_reads.sh; `wua/a5/` wua_a5.py (subagent), wua_a5_centre.py (AM2), IMPL_A5*.md.
* Work data `C:/Users/Admin/wua_work/` (re-creatable: ext/ ~1.4 GB extraction, labels/, smoke/, smoke12/; results a1/, imag/, a5/, WUA_CARD.json).
* Done: extraction s0-s2 (bit-identical repeat on 12 clips = inference variance none); A2/A3/A4 (RESULT_IMAG_s*.json); A5 primary USED at 3 seeds (RESULT_A5.json).
* Running: run_reads.sh (A1 LIN 3 seeds, then A1 MLP, then wua_card.py -> reads.done); wua_a5_centre.py 3 seeds + combine -> RESULT_A5_CENTRE.json.

Next
1. When reads.done: `python -I wua_card.py`, read `wua/WORLD_UNDERSTANDING_CARD.md`; check A1 controls (c1-c4) in card controls.
2. Add the A5 + C-CENTRE line to the card md (wua_card.py reads RESULT_A5.json; add C-CENTRE by hand or extend).
3. Rows H-V8F-PROOF-221+ in GAC_ROWS_158.md (6 pipe fields = 4 columns: headline | card/tools | controls+numbers | meaning), hold 158 after 156/157 land; procedure in the coordinator scratchpad (mk_mk_list.py, mk_hold_v8proof3.py, hold_refresh.py, scan_hold.py, land_v8proof.sh); strict scan 0 hits; verify ls-remote; FYI to local_7718c5d9-da1d-4e01-bfe3-fb70778185a6 per row.
4. Ranked Phase B levers per card section 6 -> report to the coordinator.
5. Delete C:/Users/Admin/wua_work/{ext,smoke,smoke12} after rows land (re-creatable by wua_extract.py, ~25 min/seed CPU).
