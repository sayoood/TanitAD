# HANDOFF_P4 (2026-11-07, Sonnet agent). State: priorities 1-3 done, 4 written; nothing waits on this agent.
Card FROZEN 2026-11-07T02:40:08+0100, sha256 6717a0cfe5067f26b7d5175af56b88baf567b457c13810783edc19af207f6731 (`PREREG_P4_CUECLASS.sha256`; `python -I p4_cueclass.py verify-card`).
Done: card; `p4/p4_counts.py` + `P4_COUNTS_PREFREEZE.json`; `p4/p4_cueclass.py` (verify-card, fit, compare, controls); `p4/test_p4_cueclass.py` (8 passed); `P4_CONTROLS.json` (a-d PASS, both pops);
`P4_V+E.json` baseline (descriptive); `p4/extract_v_lora.py` (pod-side, written, not run on a real frame).
Open: (1) `p4/selftest_zero_b.log` - CPU zero-B bit-for-bit test of the merged encoder vs the frozen one on a synthetic frame was started at 02:44 (`extract_v_lora.py --selftest-zero-b --stack p4/scratch/tipx/stack`);
read the last line (`bit_for_bit`). If the process died, rerun the same command (several minutes on CPU). (2) W-B (eval-141, no refit) subcommand not written; at current counts it cannot carry a verdict
(P-STOP LIGHT 8, LEAD 4 not-GO, SELF 15 GO; P-MOVE LIGHT 15 GO). (3) When L exists: pod `extract_v_lora.py --ckpt <L>/ckpt_last.pt --frames-root <v2ep dir> --probe-raw <trigger_probe/raw copy> --out <dir>`,
copy `V_L_{pstop,pmove}.npy` (2 x 983 MB) to the dev box, `p4_cueclass.py fit --featureset L_<tag> --source raw:<dir>` (PCA from raw: ~12 min per population, run in background), then `compare --a L_<tag> --b V+E`.
Do not edit the card body; append to its LOG only. Scratch: p4/scratch/tipx (tip export, small).
