# BRIEF: SEARCH-NAVTEST-PIPE, the chunked navtest pipeline for SEARCH-P2-CONFIRM READ C2 (Master Mind, 2026-10-19 ~23:10)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/navpipe/`.
- CPU only, at most 2 Python processes, each under 2 GB (other CPU jobs are running).
- Never commit or push.

## Why
The navtest T1 look (C2) is the decisive read of the search on P2's own scorer, and the only scene-unseen one. It is a named BLOCKER (ruling: `C:/Users/Admin/par18/spc/MM_RULING_SPC.md`, last section).
- Navtest is 12,146 tokens / 136 logs (`navtest_full_tokens.json`).
- Its ctx is about 44.5 GiB against about 24 GB free on D: and 7 GB on C:.
- The cost estimate (`C:/Users/Admin/par18/spc/raw/cost.json`, §10 of the sealed `spc/PREREG_SEARCH_P2_CONFIRM.md`): bundle 5.4-5.6 GPU-h (UNVERIFIED), search 12.8 CPU-h, labels 3.7-6.1 h via `label_stepB` or about 0.9 h via the NAVSIM harness seam (UNVERIFIED).
- The C1 machinery is in `C:/Users/Admin/par18/spc/code/` (`spc_gpu_stage.py`, `spc_launch.py`, `spc_cid.py`, `spc_analyse.py`), plus Step B's `C:/Users/Admin/par18/l2stepb/code/` and L1's `make_stepA_bundle.py` / `launch_stepA_ctype.py`.
- P2's navtest T1 result is 88.82 [87.86, 89.77] (MEASURED). Find its harness and seam (`eval/refe_navtest_seam.py`) from the registry row and the REFe package on the mirror tip (fetch, then pin the SHA ONCE).

## Deliverables, in PRIORITY ORDER
1. **The design, `DESIGN.md`:**
   - the data path for navtest tokens (frames, ego state, the P2 forward that produces fan, logits, pick and ctx);
   - whether an existing builder serves navtest, or a new one or a bridge is needed;
   - the T1 scoring route for a search OUTPUT. The endpoint is T1 PDMS of the search output vs P2's own pick, scored by the SAME NAVSIM v1.1 harness that produced 88.82 (`file:line` for each hop);
   - the chunking: per log group, bundle → search (3 seeds) → T1 score → delete ctx;
   - the disk peak per chunk, and a resumable state file.
2. **The pipeline code**, with these gates:
   - (a) a seal refusal: it refuses to run without a sealed C2 amendment sha and an `ADMIT` record from EP-ADMISSION. Build a placeholder check that both exist and are bound by sha.
   - (b) a disk guard: stop if D: free < 6 GB.
   - (c) a per-chunk identity: P2's pick in the fresh bundle equals P2's banked navtest pick for every token, bit for bit. P2's own navtest outputs were already read when 88.82 was measured, so this reads no new label.
   - (d) chunk deletion only after that chunk's outputs and scores are banked and hashed.
   - (e) RED arms for each.
3. **The fixture test (CPU):** run the pipeline's search + label + analysis stages on carve B's ALREADY-READ inputs (`C:/Users/Admin/par18/l2stepb/run/`, the carve-B bundle) as a stand-in chunk. It must reproduce Step B's banked P2HEAD read EXACTLY: +0.3273 [+0.0565, +0.7216], the per-seed points and the families.
4. A measured cost table:
   - the per-token CPU search time at `--threads 4`;
   - the label or score time per token on the T1 route (measure it on carve B, or on 0 navtest tokens if that is impossible);
   - the GPU time stays ESTIMATED;
   - the total wall time at 1 and at 2 processes.
5. Stage `out/paths.json` (items `[repo path, abs src, "raw"]`), repo dir `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-search-navtest-pipe/`. Write `HANDOFF.md` (2k tokens or fewer).

## ⛔ Hard limits
- NO GPU: build the GPU stage, never launch it.
- NO navtest LABEL, score or search output may be computed. The pipeline's first navtest run is C2 itself, after its amendment and admission.
- Do not open anything under `D:/Projects/TanitAD-artifacts/spc/` or `C:/Users/Admin/par18/spc/run*` (C1 is running there).
- Do not touch the GPU lock.

## Hand-back (300 words or fewer)
Design summary (new builder needed or not, T1 route), fixture identity result, cost table, test count, manifest, blockers.
