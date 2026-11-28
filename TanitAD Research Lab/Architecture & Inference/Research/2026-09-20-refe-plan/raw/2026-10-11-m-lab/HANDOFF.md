# HANDOFF -- M-LAB (condition C4: labeller cost c -> pod vCPU, P6-C)   written 2026-10-11 ~19:27 Europe/Berlin

**Deliverable:** `eval/RESULT_M_LAB.md` = c (CPU-s per 64-proposal set) as median + spread over >= 3 independent runs, RSS per worker,
worker-scaling table, vCPU/RAM recommendation by the spec formula (`SPEC_P6_PAPER_FAITHFUL.md` 3.3: need = 4.49 x c, workers 1.5x, pod = workers + 4).

## State
- Instruments built + smoke-checked: `mlab_plan.py`/`mlab_plan_extra.py` (seeded disjoint samples -> `plan/`), `mlab_worker.py` (one process, per-phase
  process-CPU, per-set background load + processor id, optional `--pin`), `mlab_driver.py` (arms, 1 Hz sampler of the real worker RSS and of
  background load), `mlab_chunk.py` (cost of splitting a set into chunks), `mlab_summarize.py` (-> `results.json`), `mlab_render.py` (-> DRAFT result).
- **Everything is RUNNING detached, no model turns needed:** `mlab_driver.py` -> `chain.log`; then `post_chain.sh` (chunk probe + `results.json`) ->
  `post2.sh` (extra arms cached_r4..r6 PINNED to P-core threads 4/6/8, plus paired pinned replays `cached_r1p`, `build_r1p`; log `chain2.log`) ->
  `post3.sh` writes `RESULT_M_LAB.draft.md` here and `ZZMLAB_POST3_DONE` to `chain2.log`. Expected finish ~20:50 Europe/Berlin (started 19:08).
  Resumable: `python mlab_driver.py` (navsim venv) skips arms whose `runs/<tag>/summary.json` exists.
- First results (MEASURED, `runs/cached_r{1,2,3}_n1`): c(cached load + label) = 0.472 / 0.668 / 0.833 CPU-s/set at background load 10 / 17 / 21 logical CPUs
  of 24. **c is strongly LOAD-DEPENDENT on this hybrid P+E / SMT i9-12900F** (per-set CPU 0.50 at bg 10-14, 0.98 at bg >= 18; `results.json` `load_dependence`),
  so one number is not honest: report the bracket, the pinned (no E-core migration) runs, and the load slope. PDMS equals the NAVSIM census on 15,168/15,168 per run.
  Sizing is robust to it: even 2x the loaded figure (c ~ 2) gives ~18 vCPU, far from the spec's infeasible c ~ 18.

## How to check
`tail chain.log chain2.log` -- one `[done] <tag> ...` per arm. Finished when `ZZMLAB_POST3_DONE` is in `chain2.log`. If an arm's `rcs` is non-zero see `runs/<tag>/worker*.log`.

## Next step (when ZZMLAB_POST3_DONE is there)
1. Read `results.json` and `RESULT_M_LAB.draft.md`. Sanity first: every arm `check.n_fail == 0`; cached arms `pdms_exact_1e4 == pdms_total`; `rcs == [0]`;
   worker JSON `ctrl_cross_token_max_abs > 0`; pinned arms `affinity_end` is one CPU and `frac_sets_on_ecore` is 0; build arms agree ~89 % within 1e-4 (rebuilt cache is not NAVSIM's file).
2. Write `eval/RESULT_M_LAB.md` from the draft (fill section 7 Recommendation; fix any prose the numbers contradict). Keep the caveats: dev box = i9-12900F P-core, so the
   dev-box c is a LOWER bound for a pod vCPU (ratio NOT measured); navtest cache (no navtrain cache on the dev box); proposals = banked `navtest_final` of `model_final.pt`
   (C3's u400 dump was not ready); IPC not measured; my own stray `find / -maxdepth 6 ...` (killed ~19:20) ran during cached_r1..r3 and part of build_r1 (disk IO + ~1.4 CPU).
3. Stage by EXACT path (never a directory): `eval/RESULT_M_LAB.md`, and in `raw/2026-10-11-m-lab/`: `*.py *.sh *.md *.json *.log`, `plan/*`, `runs/*/*` (not `__pycache__/`, not `index.json` if >1 MB is a concern -- it is ~2 MB, stage it).
   Verify with `git diff --cached --name-only` (a `git add` on a file in a newly created directory can silently no-op). Do NOT commit.
4. Escalate to the Master Mind: update the C4 row of `eval/P6_GO_SIGNAL.md` and P6-C in `PI_DECISION_QUEUE.md`.

## Do not
Run CPU-heavy work on this box while the chain runs (it measures CPU time). Do not touch other agents' processes (C3 job, video render, pytest runs are expected neighbours).
