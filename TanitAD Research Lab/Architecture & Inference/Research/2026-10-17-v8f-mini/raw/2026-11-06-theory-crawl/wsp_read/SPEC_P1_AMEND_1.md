# SPEC_P1_AMEND_1 (2026-11-06, written after P1_RESULT.json was read; SPEC_P1.md is not edited)

Defect found in SPEC_P1.md, reported as a defect, not as a changed criterion:
* Leg C is defined as the SIGNED contrast (B_C0C - B_P10) < 0. The prediction says "P10's map lies nearer the identity", a statement about distance from the identity. P10 in T2 has a band mean of +7.27 m/s (median +2.29) because 28 of its 279 tick rows (all at tick 2, scenes in the launch stage) are runaways (plan speed at +2 s above 40 m/s). The signed rule therefore reads HOLDS (diff -6.84 / -6.21, CI uppers -4.91 / -4.34), and that reading is an artefact of the sign. Read as written (distance to the identity), P10 is not nearer: |B| 7.27 (mean), 2.29 (median) against C0C 0.43 / 1.06 (mean), 0.39 / 0.84 (median).
* The formal P1 verdict (A fails) does not depend on leg C and is unchanged. Leg C is reported as NOT HELD in the prediction's wording, with the spec value shown beside it.
* p1b_controls.py and p1c_t1kin_fit.py are DESCRIPTIVES added after the first read (medians, the human's own band map, a runaway count, the T1-kin dumps). No verdict leg uses them.
