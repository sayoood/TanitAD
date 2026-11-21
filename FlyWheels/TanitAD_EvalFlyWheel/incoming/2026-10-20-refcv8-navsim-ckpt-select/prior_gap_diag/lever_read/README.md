# Lever attribution on navtrain: 18 arms scored on Thor (POST-HOC / DESCRIPTIVE, decides nothing)

Built by `../build_lever_seams.py` (refcv8 EMA-48,852 R8_LEGAL rows + its emitted prior), scored 2026-10-21 10:14-10:36 Berlin
on the Thor CPU backend (thor_pool, nice 19), read by `read_levers.py` with the reading rule fixed in `../PRIOR_GAP_DIAG.md`
before any lever arm was scored. Result `LEVER_READ.json`: the rule reads NEITHER clause at both seeds (path and speed profile
both cost); the progress cap R8PATH_MINPROG recovers the full gap to the prior (73.78 vs 73.84) without beating it.
Per-token `*_rows.jsonl`, `*_hooks.json`, `*.calls.jsonl` are OFF-BANK (sha256 in `OFFBANK.sha256`; copies on the dev box
`C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag/thor_scores/scores/` and Thor `/home/nvidia/r8ns/navtrain_sel/lever/scores/`).
NavSim v1 open-loop PDMS (T1-family); 2,089 tokens / 30 Las Vegas logs; one training seed.
