# FACTS_KINLAB_3: the MM's construction fix to `--dagger-label kin_v1` (BRIEF_BUILD_KINLAB3, 2026-11-23)

Evidence class: MEASURED-FROM-SOURCE of `F:/TanitAD-scratch/kinlab/tree` (df250a00 + phase 2 + this fix). RV = `stack/tanitad/data/v8f_recovery.py`, T = `stack/tests/test_v8f_kinlab.py`. Phase-2 files are kept beside the new ones as `<name>.phase2`. Only `RV` and `T` changed in phase 3; `v8f_dagger.py` and `v8f_mini_train.py` are byte-identical to phase 2 (so no new trainer pin edit beyond phase 2's).

## What changed (RV, `kin_label`)
| piece | where |
|---|---|
| docstring items 3 and 4 rewritten (longitudinal = speed_v1 RULE on the integrated state; merge needs the speed condition; tail from the merged v) | RV:410-415 |
| new constant `MERGE_SPEED_MS = 0.05` | RV:426 |
| `a_ref` (speed_v1 per-step accelerations) removed; the `L` array (reference arc) removed (the tail no longer uses it); knot array `uh_k` (`u_h[0]` := `u_h[1]`) | RV:517-518 |
| longitudinal: `u_h(t)` linear between the 0.2 s knots, `a_rule = clip((u_h(t) - v)/DT, +-A_LABEL)`, `a = clip((1-w) a_o + w a_rule, +-A_LIM)` | RV:532-535 |
| merge: lateral < 0.10 m AND heading < 1 deg AND `abs(v - u_h[kk]) <= MERGE_SPEED_MS`, all at the same 0.2 s output point with t >= T_B; `v_m` kept | RV:561-562 |
| tail: `label_steps(v_m, uh_k[k_m:])` (speed_v1's own rule: step toward u_h by at most A_LABEL*DT, then u_h once equal), cumulative arc from `s*_m`, point = P at that arc (position-continuous, no step of v above A_LABEL*DT) | RV:571-574 |
Unchanged: `T_B`, `LD_*`, `K_LIM` / `F_LIM` clips, the refusal reasons and counters, the speed_v1 fallback, the identity rule, `kin_label_bank`, `kin_census`.

## Interpretations (this phase; none tuned on data)
1. Knots of the interpolated human step speed: knot k = `u_h[k]` (the chord speed of the step ENDING at 0.2 k s, the same indexing speed_v1 uses for `u_l[k]`), knot 0 = `u_h[1]` (`u_h[0]` is unused in speed_v1); `u_h(t)` for t in step k is linear between knots k and k+1, sampled at the sub-step start.
2. The speed condition's `u_h` at the output point kk is knot kk = `u_h[kk]`; `v` is the integrated speed at the end of the sub-step that produces output point kk.
3. Interpretations 2-6 of FACTS_KINLAB_2 are confirmed as built (the MM's ruling). Known limit (6): `kappa_o` understates the true curvature below 2 m/s; kept because the label should match what the model reads.
4. The bound "no step may change v by more than A_LABEL*DT + 1e-9" is built for the TAIL (after the merge) and by the rule; the 1 s blend keeps the phase-2 `|a| <= A_LIM` clip (3.96 m/s2 -> 0.792 m/s per step), so a large `a_o` can still change v by up to 0.792 in one blend step.

## Test changes (T)
* Constants test also pins `MERGE_SPEED_MS == 0.05` and `A_LABEL == 3.0`.
* ADJUSTED (the merge rule changed): the phase-2 test "a planted a_o above A_LIM is refused for the accel term at the merge step" no longer has its premise (no speed jump at the merge, so such a record is emitted with |a| clipped to A_LIM). It is now `test_planted_a_o_above_a_lim_is_clipped_in_the_integration_and_a_checker_accel_flag_refuses_and_keeps_the_speed_label`: clip asserted, and the accel refusal path of the checker is exercised with the checker's `a_lim` tightened to 1.0 by monkeypatch.
* NEW: `test_phase3_a_planted_a_o_different_from_a_ref0_merges_with_no_speed_step_above_a_label_dt` (4 cases; tail step change <= A_LABEL*DT + 1e-9); `test_phase3_a_record_that_cannot_meet_the_speed_condition_by_six_seconds_is_refused_not_merged` (a human accelerating at 3.6 m/s2, above the rule's 3.0 clip; the control with the speed band opened merges at t = 1.0, so the speed condition is what refused it).
