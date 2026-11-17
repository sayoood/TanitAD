# HANDOFF adh_decomp (2026-11-17) — all 5 questions done; nothing running
State: Q1-Q5 banked in ADH_DECOMP_P7.md / adh_decomp_p7.json. No job in flight.
Next step (MM, needs the PDMS scorer): PDMS-score `rerank_p7_seam.npz` (sha256 ced64bce...1ee, 1,296 changed rows of 12,146) vs `stage2_p7_full.npz`; read the delta on the 1,296 changed tokens and the 268 where the human did not follow the command (idx in rerank_selection_idx.npz; human strata in adh_q4.json).
Optional 0-GPU follow-up: margin-gated re-rank sweep over the 439 turn-active selection misses.
Rerun: scripts need instrument code exported from mirror FETCH_HEAD (PK/eval/command_adherence.py, proxy_eval.py, instruments/strategic_family.py, refe/planner.py) into F:/tanitad_scratch/adhdecomp/code (scripts/common.py SCR); run s1, s2, s3 in order with the tanitad venv, CUDA_VISIBLE_DEVICES=-1.
Deletions logged: F:/tanitad_scratch/adhdecomp/ (code exports + cls64.npy + s1_res.json, ~15 MB; all re-creatable from mirror + scripts; results banked in hold).
Nothing staged/committed/pushed.
