# AMENDMENT 9 to BEHAVIOUR_SCORECARD.md: chunk stamps for drift-guard redraws (extends AMENDMENT 8)

**Date:** 2026-11-08 (file mtime is the time of writing). **Before** the tier 1 aggregate exists. Reporting only: no threshold changes, no scene excluded, the running suite is untouched.

**Why.** The sim session measured on the 3 scored scenes (saved frames, INHERITED, `~/data/nuplan_t2/analysis/blur_probe.json` on the sim host) that the scenery changes it had reported coincide with the sim's own drift-guard redraws: 8 chunks over 3 scenes, first at 4.3 to 5.4 s. A redraw has no image condition, only map and caption, so all 5 views get new scenery. The sim session has withdrawn "Cosmos follows the caption over the seed" as the main explanation. AMENDMENT 8's Why paragraph named the drift as a HYPOTHESIS and does not depend on that explanation.

**Consequence.** A redraw can fall inside AMENDMENT 8's `pre` window (0 to 6 s), so `pre` is not a window of unbroken continuity from the seed.

1. **Chunk stamp.** `render_fidelity` carries, per chunk, `chained` or `redrawn`, with the trigger of each redraw. Descriptive only.
2. **Added window `to_first_redraw`** for the rules a1, a2, c4, e1, e2: from 0 s to the scene's first redraw, limited by `human_ref_valid_until_s`. It sits beside `pre` and `post`. It is descriptive and decides no rule. Its end time differs per scene and per checkpoint, because a redraw depends on the rollout: report it, and do not compare it across checkpoints as if it were a fixed window.
3. `pre` and `post` stay as AMENDMENT 8 defined them. The 6 s is not re-chosen.
4. AMENDMENT 8 section 5 is read accordingly: a "drives well" statement is limited to what `to_first_redraw` supports until fidelity has been measured.
5. **Not ruled here, for the sim session after the suite ends:** the 2 of 8 redraws that fired on the marks check alone on frames over 2x sharper than the reference look like false alarms; the rolling-state reset on each of 31 requests; the sharpness loss per pass. Any change to the guard or rollout is a new baseline and needs its own u400 run, because scores across different renderer settings are not comparable.
