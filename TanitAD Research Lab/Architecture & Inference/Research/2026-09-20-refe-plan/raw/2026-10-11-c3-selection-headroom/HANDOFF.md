# HANDOFF -- C3 selection-headroom test (written 2026-10-11 ~18:53 Europe/Berlin)

**Deliverable:** run `eval/P6_GO_SIGNAL.md` section 3 "C3" as written; report the verdict.  **Pre-registration is followed as written;**
the choices it leaves open are pinned in `OPERATIONALISATION.md` (written before any C3 proposal was scored).

## State (MEASURED / done)
- `tokens.json` written first (529 zeroed + 529 random non-zeroed, seed 20261011; sha256 `03072566...e2254d`).
- `c3_run.py` (dump -> fidelity gate -> 64 x harness scoring -> table -> analysis) and `c3_analyze.py` built.
  `c3_analyze.py --selftest` GREEN (known-answer worlds, analytic identities, deliberate-regression arm).
  Plumbing smoke on 2 tokens GREEN: CPU dump (57 s), gate (max |d| 4.1e-4 m CPU-vs-banked-GPU), 4+2 harness columns PASS,
  table, and the subset harness reproduced the banked per-token score at the pick EXACTLY (diff 0.0, all six sub-scores).
- DRY RUN GREEN (`dry_check.py` -> `dry_logs/dry_check.json`): the BANKED u400 picks on the 1,058 tokens, scored through the subset harness,
  reproduce the banked full-navtest per-token score and all six sub-scores EXACTLY (max diff 0.0, 0 of 1,058 differ; 529 zero scores). The Hydra
  --tokens command line fits at 1,058 tokens. One column took 205.7 s wall (CPU contention) -> expect ~55-90 min for 64 columns at 4 workers.
  The neighbour video job processes ~10 logs one seam process at a time (about half done at 18:54, ~3 min per log); the lock is re-taken within
  seconds, and a resident neighbour keeps `memory.used` ~6 GB, so expect to win the GPU only when that job ends (est. 15-30 min).
- **The main pipeline is RUNNING, detached:** `python c3_run.py all --workers 4`, log `run.log`, status `STATUS.json`
  (both in this directory). It is first waiting for the shared GPU lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`,
  held in turns by the u400 real-frame VIDEO agent (do not touch `raw/2026-10-11-u400-real-frame-video/`). It polls every 3 s and
  takes the lock only when `nvidia-smi memory.used <= 3500 MiB`.

## How to check (one command, no model turns)
`cat "<this dir>/STATUS.json"` -- fields `stage` (dump | gate | score | table | analyze), `state`, `result`.
Expected timeline after the lock is won: dump ~25-40 min (1,058 tokens x ~1.4 s) -> gate (seconds) -> score 64 columns
(~140 s each, 4 workers, ~40-60 min) -> table -> analysis (seconds).

## Outcomes and what to do
- `STATUS.result` starts with **"DONE"**: `readout.json` and `readout_tables.md` exist. Write
  `eval/RESULT_C3_SELECTION_HEADROOM.md` from them (verdict by the committed rule; (a), (b), (c) with CIs; every number
  MEASURED + artifact path; the consequence line from the prereg outcome table; DESCRIPTIVE blocks labelled). Check first:
  `gate_fidelity.json` PASS, `gate_table.json` G2_PASS, `readout.json` `b_pdms_x100.sanity_mix_actual_vs_banked_full_navtest`.
- `result` starts with **"INSTRUMENT FAILURE"**: that is the verdict-less outcome; report it plainly with the gate JSON, then diagnose
  (do not re-run until a cause is named). Fidelity failure -> compare `gate_fidelity.json` worst_tokens.
- `result` starts with **"FAILED"**: read `logs/1_dump.log` / `logs/pNN.log`; stages are resumable (`python c3_run.py score|table|analyze`,
  same flags; done columns are skipped).
- Never wait on it while alive (token rule 6a).

## Then (integration; do not skip)
1. Stage by EXACT path (never a directory): `eval/RESULT_C3_SELECTION_HEADROOM.md` and everything in this directory except
   `__pycache__/` and `dry_logs/` bulk (see manifest in the final report). Verify with `git diff --cached --name-only`
   (a `git add` on a file in a newly created directory can silently no-op).
2. Escalate to the Master Mind: update the C3 row of `eval/P6_GO_SIGNAL.md` section 3 (`⏳ not run`) and `GOALS_AND_CLAIMS.md`.
   An agent does not commit.
3. `python tools/kb_add.py` is NOT needed: no paper is cited.

## Where things live
- scripts / tokens / gates / logs / `readout*`: this directory (repo). Heavy intermediates (64 per-proposal seams, 64 score dirs):
  `D:/Projects/TanitAD/data/refe_navtest/seams/c3/main/`, `.../score/refe_c3_u400_main_p00..p63/` (data dir, not in git).
- `proposals.npz` (dump: 1058 x 64 x 8 x 3 poses + 7 raw scorer logits) and `table.npz` (PDMS + 6 sub-scores per token x proposal) land HERE.
- Leftovers from plumbing checks (harmless, data dir): `score/refe_c3_u400_dry_pick/`, `seams/c3/dry/`, `seams/c3/smoke/`,
  `score/refe_c3_u400_smoke_p0*`, `_p09`, `_p04`.
