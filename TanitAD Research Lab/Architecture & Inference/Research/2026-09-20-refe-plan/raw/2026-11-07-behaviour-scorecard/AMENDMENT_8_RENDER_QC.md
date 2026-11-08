# AMENDMENT 8 to BEHAVIOUR_SCORECARD.md (sha256 fac6c7f1...): render fidelity is reported, never used to filter

**Date:** 2026-11-08 ~14:45Z. **PI, same day:** "go ahead with the render rule".
**Ruled before** the tier 1 aggregate exists. **Disclosed:** the Master Mind had seen the per-scene rows of `86cad912` and `59714859` (2 of 28) and three front frames of `86cad912`, as reported by the sim session. No value below was chosen from those rows.

**Scope.** Reporting only. No threshold of any scored rule changes. No scene is excluded, down-weighted or re-run because of this amendment. The running suite (`SUITE_TOKENS.json` f0c49923..., scorer 69d13d71...) is not touched: everything here is computed after the rollouts by a separate script. Gate H, the adoption rule (section 4) and the tier 1 power statement are unchanged.

**Why.** The sim session reports that, on `86cad912`, road geometry and markings stay consistent while the surroundings change within about 80 m of travel, and some facades smear by 14 s (three frames, one scene: HYPOTHESIS). A rule scored against the human trace compares an ego in a rendered world with a human in the real one. Once those worlds differ, the number mixes model error with simulator fidelity, and speed responds to scene context. Geometry-only lateral rules are less exposed.

1. **The full-window verdict stays primary and is reported exactly as defined.** Everything below sits beside it.

2. **`render_fidelity`, one stamped field per scene and checkpoint.** A fixed pretrained image embedding, chosen and pinned by sha256 by the sim session before it reads any value, is applied to the saved front-camera frames, at every saved frame.
   - Preferred reference: the real logged front frame at the nearest **arc length** along the route, not the nearest time, because a slower ego has legitimately seen less of the road. Report cosine similarity against arc length.
   - If real future frames are not available on thor3 for a scene, report `no_real_reference` with the reason, and give only the weaker within-render consistency (similarity between renders 1 s apart), labelled as such.
   - If frames were not saved for a scored scene, report `unavailable` with the reason. Do not re-run the scene for this.
   - **No threshold is set.** With 10 to 28 scenes, a cut cannot be calibrated, and setting one after seeing values would be a moved goalpost.

3. **Two windows for the rules measured against the human path (a1, a2, c4, e1, e2).** `pre` = 0 to 6 s, `post` = after 6 s, each further limited by `human_ref_valid_until_s` (AMENDMENT 5). The 6 s is the window the scorecard already requires the render to hold (section 3.1, QC 1). It was not chosen from any score. These windows are descriptive and do not decide a rule.

4. **Real-frame control, `G-R`, on every tier 1 scene.** Run the same checkpoint with the same settings on the real logged seed frames, and compare its first-call plan (plan_v0, plan_v4, end speed) with the first call in the sim. Tier stamp: T0-class for the real-frame forward. If a slowdown is present on the real frames, it belongs to the model and rendering cannot explain it. If it appears only on renders, the render is implicated even early in the run.

5. **How verdicts are read.**
   - Render drift is **not** a reason to discount a failed rule unless `G-R` or the `pre` window separates it. A failure in `pre` that also shows on real frames is a model failure.
   - A scene with poor fidelity is reported with its token. It is never skipped silently and never filled from a later rank.
   - Any "drives well" statement is limited to what `pre` supports until fidelity has been measured, and carries the stamp `T2-class, closed-loop sim`.
   - An adoption read compares checkpoints under the same scenario seed, so renderer behaviour is shared by construction, but only while the two egos travel similar paths. Where a candidate and the baseline diverge strongly, say so beside the read.

6. **Open and owned by the sim session:** the tier 1 report lists, per scene, the frames inspected, so that a render statement always carries its n.
