# refcv7 Training Watch: pace judged against the PI-approved ceiling (2026-09-28)

**Defect (MEASURED on the first live build, step 1,750).** `build_watch_refcv7.py` judged pace only against
the pre-approval 8.0 s/step line (refcv6's 6.4 × 1.25). It showed the live run's 9.91 s/step as an AMBER chip and
an amber alarm tile. It also called the cost "the PI's reserved decision". The PI took that decision on
2026-09-27: ~9.9 s/step approved, ceiling 10.5 (SPEC_REFCV7 sec. 25; the gate's `pi_cost_approval.json`
`max_s_per_step` 10.5). So the page asked the PI a question that had already been answered. The statement was
true of the rule and wrong for the reader.

**Fix.** `PACE_PI_APPROVED_S = 10.5`. Amber now fires only above the approved ceiling. The 8.0 line stays on the
pace chart and in the text as the pre-approval reference. The summary JSON gains `pace_over_pi_approval`, and
`pace_above_pi_line` is unchanged so its readers are unaffected. The fix is 11 exact-once replacements
(`code/watch_pace_patch.py`, which refuses on any count other than 1).

**Evidence.**
- `taniteval/tests/test_build_watch_refcv7.py`: the pace test is rewritten with LITERALS for 8.5 (good,
  within approval), 11.0 (amber, above approval) and 6.5 (under both lines).
  45/45 pass (`raw/pytest_after.log`).
- Mutation: the new pace test against the landed builder (blob 86e47b33) goes RED
  (`raw/mutation_old_builder_pace_test.log`: 1 failed).
- The lander's guards pre-run on both files against the tip: ZZSUPERSET-OK and ZZDEMOTE-OK, no waivers.
- Live build at step 1,750 (`raw/watch_summary_step1750.json`): pace 9.914 s/step, `pace_over_pi_approval`
  false. Published as the "refcv7 Training Watch" artifact.
