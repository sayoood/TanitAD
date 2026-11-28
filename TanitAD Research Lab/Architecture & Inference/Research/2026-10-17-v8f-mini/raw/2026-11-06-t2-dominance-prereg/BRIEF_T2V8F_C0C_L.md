# Brief: the composite arm C0C at T2, and the stage-4 package (AGENT L; dev-box CPU; no download; do NOT launch)

## Why (MEASURED by the Master Mind, 2026-11-06)
* The frozen card `motion_c/t2_prereg/PREREG_T2_CRAWL.md` section 6, outcome 1, commits the next read: *a T2 dominance read of C0's
  COMPOSITE against P10 over all 62 scenes, all four families, with Read A's rules*. Every later T2 dominance read needs the same arm.
* The T2 harness builds no composite. `ARMS` in `t2v8f/code/t2v8f_run.py:24` and `t2v8f_stage1.py:38` is
  `C0_s0, C0_s1, DAGOACC_s0, DAGOACC_s1, P10, ZERO, BLIND`, and `grep -i "sg72|kbox|composite|C0C"` over `t2v8f/code/*.py` returns zero.
* At T1-kin, the composite is the planner's plan passed through `smooth_plan(raw, p10, "sg72")` and then `feasible_plan(., "kbox")`.
  * Source: `motion_c/ri/code/v8f_t1_eval_ri.py:579-624`, wired at `:626-660` in `model_planner(..., smooth="sg72", feasible="kbox")`.
  * Pinned sha256: `0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7`.
  * `kbox` is `tanitad.refs.feasible_decode.project_feasible` with the curvature clamp only, `a_max = inf`, `clamp_entry = False`.
* The composite changes the executed motion. At T2 the next frame is rendered at the executed pose, so the composite is NOT a post hoc filter of
  C0's T2 rollouts. It needs its own T2 rollouts.

## Rules
`BRIEF_T2V8F_STAGE2_I.md` binds:
* the preamble (`Project Steering/AGENT_OPERATING_STANDARD.md`), never commit, stage or delete, LF, banned words;
* <= 150k context, `HANDOFF_L.md` at about 130k, bank incrementally;
* no ssh, no box 2, CPU only (`CUDA_VISIBLE_DEVICES=-1`), no download, no training;
* the frozen card is never edited.

Box 2 is RUNNING stage 2 (driver sha256 `33b65b19...`), and stage 3 (driver `0688bb47...`) is chained after it. ⛔ Neither driver's
behaviour changes. New arms and stage 4 go in new files or a new mode, and every existing test stays green.

## Items, in priority order (bank each before the next)
1. **The arm.** `C0C_s0` and `C0C_s1`: C0's checkpoint, then the T1-kin composite on every replan, using the SAME arithmetic. Load the two
   functions BY FILE PATH from `v8f_t1_eval_ri.py`, sha-checked, or ship a byte copy whose sha256 is pinned. The prior is `out["p10"]` of the
   same forward. Prove `tanitad.refs.feasible_decode` is importable on box 2 from the shipped stage-1/2 package (`D:\t2v8f\pkg`, whose
   manifest is in `stage1/` and `stage2/`); if it is not, ship it and pin it.
2. **Gate C0C-ID, with RED arms.**
   * The tick-0 plan of C0C from the banked cached features must equal the RI tool's composite tick-0 plan for the same window, bit for bit
     (or name the tolerance and why).
   * Source: a banked T1-kin `RIC0C` / `C0C` dump. Find it under `motion_c/ri*/` or `motion_c/cl_read/`, and name the path.
   * RED arms: smooth off, feasible off, and the prior swapped for zeros. Each must break the identity.
   * Also check the T2 harness's own `C0` arm against `C0C` with `--smooth off --feasible off`: it must reproduce `C0` exactly.
3. **The stage-4 package.**
   * Cells: `C0C_s0` and `C0C_s1` under T2, deciding, then under RR, beside. The same starts as stages 2 and 3 (stage 2's 81, stage 3's 40),
     the `base` executor, `objects` agents.
   * Scenes are already on box 2: 42 in `D:\t2v8f\scenes2` and 20 in `C:\t2v8f\scenes3` once stage 3 has run. NO scene copy. The preflight
     sha-checks them against MANIFEST2 / MANIFEST3.
   * P10(T2) is REUSED from stages 2 and 3. Prove the window keys pair exactly (`t2v8f_score.pair_check`) in a test.
   * The pull goes to `D:\t2v8f\pull4`. Floor rules as stage 3; state the footprint per drive.
4. **Rehearsal and mutations.** The full stub rehearsal, plus the REAL `read_ref_frame` / `falsify_actors` path, as in stage 3. Mutation arms
   RED: composite off, wrong prior, a scene byte changed, stage-3 not run (scenes3 absent), and a P10 key mismatch.
5. **`STAGE4_LAUNCH.md`.**
   * Overlay put + certutil, launch, canary.
   * The waiter uses EXACT marker names: `Test-Path <pull>/STAGE4.done -PathType Leaf`, then `grep -x`. ⛔ Never a listing pattern: a
     renamed `.done.prev` matches it (MEASURED today).
   * Pull, then the score: the 19 rows of C0C(T2) − P10(T2) per seed over 62 scenes, scene-cluster bootstrap, B 2000, seed 0.
   * ESTIMATED wall time at the MEASURED ~18 s per cell of stage 2's run 2.
6. **Close-out.** Full suite detached (`cd t2v8f && CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests -q -p no:cacheprovider`, about 12 min),
   then `PINS_L.txt`.

## Hand-back (<= 300 words)
The arm, the gate result with its RED arms and the dump path, the package sha256 and size, the cell count and footprint, the mutations,
open items, blockers, and a deliverable manifest (every artifact and where it lives).
