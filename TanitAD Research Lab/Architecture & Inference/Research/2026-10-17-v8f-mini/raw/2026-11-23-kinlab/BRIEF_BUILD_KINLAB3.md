# Brief: KINLAB phase 3, the MM's construction fix to `kin_v1` (merge without a speed jump), then the gold, tests, mutants and census again. Zero training

You are a TanitAD build agent working for the Master Mind (MM).

First read:
* `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip;
* `F:/TanitAD-scratch/kinlab/BRIEF_BUILD_KINLAB.md` (phase 2: the design, the rules, the dev-box limits; they all still bind);
* `F:/TanitAD-scratch/kinlab/FACTS_KINLAB_2.md`;
* `F:/TanitAD-scratch/kinlab/HANDOFF_KINLAB2.md`;
* `F:/TanitAD-scratch/kinlab/KINLAB_CENSUS_2.md`.

The phase-2 build sits in `F:/TanitAD-scratch/kinlab/tree`, an export of df250a00 with the changed files listed in `CHANGED_FILES.txt`. Work there.
* Keep a copy of each phase-2 changed file as `<name>.phase2` beside it.
* Keep the phase-2 census and mutant results as they are, under new names with a `_phase2` suffix.

## Why the fix (MM ruling on the phase-2 finding; this is a construction defect, not a tuned constant)
The phase-2 census relabelled 27.4 % of ok records. 61.5 % were refused as infeasible, and 98.6 % of the acceleration refusals sat exactly on the merge step.

The cause is in the MM's brief:
* After the 1 s blend, the longitudinal used the speed_v1 per-step accelerations. Those never pull the integrated speed back to the reference speed, so the speed stays offset by about 0.5·(a_o − a_ref0)·1 s.
* The merge tail then put the speed back on the speed_v1 profile in one step.

The card is not frozen, and no model number exists. RECLAB fixed a construction defect of the same kind before its freeze and recorded it, and this follows that precedent.

## The fix (binding)
1. **Longitudinal:** the target is the speed_v1 **rule** evaluated on the integrated state, not the speed_v1 per-step accelerations.
   * a_rule(t) = clip((u_h(t) − v(t)) / 0.2 s, ±A_LABEL), with A_LABEL = 3.0 m/s² (the speed_v1 constant). Here u_h(t) is the human step speed, linearly interpolated between the 0.2 s knots, from the same human polyline chords that speed_v1 uses.
   * a(t) = (1 − w)·a_o + w·a_rule(t), then clip |a| ≤ A_LIM as before.
2. **Merge:** both the lateral test of phase 2 (interpretation 2: |lateral| < 0.10 m and |heading error| < 1.0°, at the 0.2 s output points with t ≥ T_B) and **|v − u_h| ≤ 0.05 m/s** must hold at the same output point.
   * After the merge, the remaining points are the human polyline at the cumulative arc. The speed follows the speed_v1 rule from the merged v (step toward u_h[k] by at most A_LABEL·0.2 s, then u_h once equal), exactly as speed_v1's own tail after its merge.
   * Positions and speed must stay continuous: no step may change v by more than A_LABEL·0.2 s + 1e-9.
3. **Interpretations 2–6 of `FACTS_KINLAB_2.md` are confirmed as built.**
   * Interpretation 6 is recorded as a known limit: kappa_o understates the true curvature below 2 m/s. It is kept because the label should match what the model reads.
4. Everything else is unchanged: T_B, L_d, K_LIM / F_LIM clips, the refusal reasons and the fallback to speed_v1, the identity rule, and the counters.

## Then
1. **Tests and gold.**
   * Gold: human and speed_v1 bit for bit vs df250a00 (planted, fixture, and the 4,000 real records as before).
   * The phase-2 tests, adjusted only where the merge rule changed.
   * New tests: a planted record with a_o ≠ a_ref0 must merge with no step above A_LABEL·0.2 s, and a record that cannot meet the speed condition by 6.0 s must be refused as not_merged.
   * Whole-file runs. Also run the launch-gate test file(s) that fingerprint the trainer, whole file: find them under `stack/tests/` (`launch_gate`). Report which fingerprints move (memory: a pin edit moves the profile fingerprints). Do not edit `launch_gate.py`.
2. **Mutants:** the 17 of phase 2, re-run, plus two new ones: the merge ignores the speed condition; the longitudinal uses the per-step accelerations again. All must be RED, with the control green.
3. **Census:** the same script and shard as phase 2 (shard 0 of RH bank s0, pulled read-only, deleted afterwards), the same outputs, plus the share refused by reason, compared with phase 2.
   * **The bar is pre-committed by the MM now, before this census:** the card will require a relabelled share ≥ 0.5 of ok records at each seed's bank.
   * Report the share. Do not change any constant or rule on it.
   * If it reads below 0.5, also give the top three refusal reasons with their shares and a 30,000-record diagnostic of where they occur, as in phase 2.

## Dev box (as phase 2)
* CPU only, `CUDA_VISIBLE_DEVICES=-1`, one python process at a time, each behind `wait_commit.py 5.5 60` (4.0 for a run under 1.5 GB if the gate stays shut; say so).
* Nothing on C:. TMP, TEMP and `--basetemp` go under `F:/TanitAD-scratch/kinlab/tmp`.
* Your peak stays under 1.5 GB of RAM. If F: drops below 6 GB free, stop and hand back.
* Never kill a process you did not start.
* Never wait more than about 4 minutes in one call. If your context passes about 150k, write `HANDOFF_KINLAB3.md` and end.

## Deliverables
In `F:/TanitAD-scratch/kinlab/`:
* the updated files with `CHANGED_FILES.txt` (path, sha256, base blob at df250a00);
* `FACTS_KINLAB_3.md` (what changed, with file:line);
* test logs including the launch-gate run;
* `mutants/MUTANTS_RESULT.json`;
* `KINLAB_CENSUS_3.json` and `.md`;
* `HANDOFF_KINLAB3.md`, `MANIFEST_SHA256.txt`, `DELETIONS.log`.

Hand back in at most 250 words, saying plainly what passed and what did not run.
