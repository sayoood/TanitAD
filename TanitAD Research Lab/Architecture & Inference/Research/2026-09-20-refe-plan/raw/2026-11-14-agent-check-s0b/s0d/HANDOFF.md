# HANDOFF S0-D (R-sim), 2026-11-14 (rev 2: R-sim-t0v + S0D-STOPFLAG added per MM): built, tested, smoked; FULL READ NOT RUN, no gain/capture/precision seen

State: `test_s0d.py` 13/13 OK, `--mutations` 12/12 RED (adds "t0v drops velocity", "STOPFLAG disabled"; cv ignores velocity, t0 hook not applied, none keeps agents, t0 keeps velocity, identity time-shifted,
statics dropped, t0 admits late agents, GO and MOTION thresholds broken, monotone check inverted). Smoke on 2 held-out tokens (`smoke/smoke_report.json`):
R-sim-true flags == truth 0 mismatches, raw nc/ttc max abs diff 0.0; hooked == unhooked NAVSIM observation 0 mismatches; my split scorer == PT.score_plans 0 mismatches;
C-SIM-NONE monotone 0 violations. 6-set code-path self-test through assembly (bootstrap, outcome row, npz, md) ran clean; output deleted unread.
Peak RSS 467-537 MB. ESTIMATED 0.7-1.7 CPU-h held-out (1.9 s/set warm, 4.5 s mean), ~2-5 h for --all. Run detached; it is resumable per log (`<run>/parts`).

Artifacts (all in `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/agent_check_s0b/s0d/`): s0d_rsim.py s0d_hook.py test_s0d.py FREEZE_S0D_DRAFT.md HANDOFF.md smoke/smoke_report.json.
Reuses (unchanged, parent hold): s0b_run.py s0b_readout.py s0b_raster.py ref/. Env override `S0D_S0B_DIR` exists only for the mutation runner.

## MM decisions (each is a definition I chose where the card is silent; none silently)
1. **Velocity is read by the scorer, in one place:** `get_collision_type` -> `is_track_stopped` (agent speed <= 0.05 m/s => STOPPED_TRACK => at-fault when ego moves, any geometry),
   on the track's first-appearance object. TTC does not read agent velocity. So R-sim-t0 (velocity 0) makes every agent a stopped track and turns rear/lateral contacts into
   at-fault: t0 tests pose AND that classification. MM ruled: that separating arm is now built as **R-sim-t0v** (secondary), with row **S0D-STOPFLAG** (c(t0) < 0.25 and c(t0v) >= 0.25), order INVALID, GO, STOPFLAG, MOTION, NULL. Re-smoked on the same 2 tokens: identity checks 0 mismatches; 5 arms, 2.1 s/set warm, 0.8-1.6 CPU-h held-out; peak RSS 468 MB (self-test 537 MB).
2. **Statics are never modified** (brief: "as NAVSIM already treats them"). S0-B's R-t0 took statics from t0 too; here a static first detected after t0 still appears.
3. **Agents absent at t0 do not exist** in t0 and cv; agents present at t0 persist all 5 s even if they truly vanish (single-frame perception).
4. **cv:** (vx, vy) = nuPlan Agent.velocity = DB lidar_box vx, vy, global frame (the S0-B R-cv rotated the same vector into the ego frame: equivalent).
5. **C-SIM-NONE check is the stricter one:** per-proposal subset (NONE's NC flags within true's); INVALID on either that or the per-set count. Removing agents cannot create an agent collision.
6. **Identity also requires** raw nc and ttc equal to the fit pack and per-set gain == oracle exactly, not only flag equality.
7. The hook is applied ONLY to `_interpolate_gt_observation`; PDM-Closed's planner input uses the real scenario (it affects EP, not nc/ttc).
8. A set whose scenario guard rejects or errors is dropped from every arm jointly and counted in `status_counts`; if any part is missing the assembly refuses (rc 3).

## Deletions (own scratch, literal paths)
Rev 2: recreated and deleted again C:/Users/Admin/s0d (selftest ~24 KB, tmp 0 KB), same literal paths as below.
- C:/Users/Admin/s0d/work/selftest (24 KB: parts, s0d_result.json, S0D_RESULT.md, s0d_flags.npz of the 6-set self-test, unread)
- C:/Users/Admin/s0d/work/tmp (0 KB, empty) and C:/Users/Admin/s0d/work/shas.txt (<1 KB); then C:/Users/Admin/s0d/work and C:/Users/Admin/s0d (empty).
Mutation temp dirs were removed by the runner itself. Nothing staged, committed or pushed. Hold size under 100 KB.
