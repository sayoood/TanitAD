# BRIEF: read SH5 grid 5 against its pre-registration and bank it (Master Mind, 2026-10-20 ~04:00)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- The work dir is `C:/Users/Admin/par18/sh5diag/`.
- CPU only, NO new training, NO relaunch of anything. C: has about 0.1 GB free: write small files only, and if C: drops below 40 MB, stop and write the HANDOFF.
- Never commit or push.

## State (MEASURED by the MM at ~04:00)
- `sha256sum -c PREREG_GRID5.md.sha256`: the code pins are OK.
- `work/diag5/diag_summary.json` holds G5_s0 and G5_s1.
- **`D.passing_cells({G5_s0, G5_s1}) = []`, so G1 (PASS_REG) is NOT met.**
- G5_s2 crashed at 03:30: a numpy MemoryError allocating 2.25 MiB, because the host commit limit was hit (C: full, so the pagefile cannot grow). Partial `diag_G5_s2.json`. ROBUST is moot, since PASS_REG failed.
- `work/diag5f` (F12_s0 / F12_s1, the fp32 control) crashed: CUDA OOM on a 20 MiB allocation with 3.58 GiB "free", the same host-commit cause.

## Do
1. Follow `HANDOFF.md`'s "Exact next steps" 2-5 as far as the banked data allows:
   - Name the outcome in `PREREG_GRID5.md` §6's order: G1 (already failed), then G3, G2, G4, otherwise.
   - Compute each from the s0 / s1 curves with the prereg's literal thresholds. Quote the numbers per cell and per seed (beta_eff, top-1, F at c24 and c30, ALLQ, R²).
   - Replicate controls (§4): report what is computable. The fp32 control did NOT run: name it UNAVAILABLE (crash, cause) and do not substitute.
   - Run nothing that trains.
2. **Committed next arm:** state what §6 commits for the outcome you name. The MM has DEPRIORITISED SH5 LOCAL / FULL behind the search line, because PRIVGEOM-CLEAN measured the held-out proximity complement at about +0.010 AUC, below the +0.02 bar (`D-REFE-PRIVGEOM-CLEAN-1`). So write the next arm as PARKED with its exact unblock condition. Do NOT run it.
3. Write `RESULT_GRID5.md`: the outcome, the per-cell table, the crashes and their cause, and the stamps (SIMULATED? no: MEASURED on the synthetic SH5 rig, a diagnosis tier; one run per seed; no CI claimed, as the prereg says).
4. Add proposed register rows (GAC + BACKLOG) as TEXT in `RESULT_GRID5.md`; do not edit registers.
5. Build `out/paths_grid5.json` (`[repo path, abs src, "raw"]`) for:
   - grids 3-5's prereg, pins, code, summaries and logs. Truncate logs over 2 MB to their last 2 MB and say so.
   - NO checkpoints (.pt).
   - Repo dir: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-20-sh5-grid5/`.
   - Check whether grids 3-4 were already landed (`a8eeef6`, via `git -C C:/Users/Admin/tanitad-push show --stat a8eeef6`); do not duplicate files that are already on the tip.

## Hand-back (250 words or fewer)
The outcome named with its numbers, the committed (parked) next arm, the manifest path and its item count.
