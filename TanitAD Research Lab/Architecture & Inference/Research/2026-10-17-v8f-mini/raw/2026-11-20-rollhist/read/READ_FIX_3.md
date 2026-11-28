# READ FIX 3 (MM, 2026-11-21T06:41:12Z): the one-seed read's provenance check looked for seed-1 run records

* After READ FIX 2 the provisional seed-0 read ran to the end (`ZZ ROLLHIST-READ DONE ZZ`) and printed `PROVISIONAL ROLLHIST-BLOCKED`.
  * The block came from provenance alone: "run record missing B0_R0C10Hb20_s1; ...".
  * `rh_read.py` `provenance()` looped over `C.SEEDS` (0, 1) and not over the read's seeds, so a one-seed read always blocks.
* Fix: `provenance(w, shas, seeds=None)` loops over the read's seeds, and the caller passes `seeds`. The default stays (0, 1), so the selftest's mutation arm is unchanged.
  * rh_read.py sha256 17297d65 → 912928a2; the copy is kept as `rh_read.py.bfix3`.
* Checks (out/read/PROV_FIX3_CHECK.json, PASS):
  * selftest rc 0, ALL_PASS;
  * the real seed-0 tree is clean at seeds (0,);
  * a planted wrong RH_s0 sha gives 6 problems (RED);
  * the old default (0, 1) on this tree reproduces the false block.
* The blocked result was moved, not deleted, to `out/read/RESULT_ROLLHIST_blocked_provenance_s0.json`. The `read` step marker was moved to `out/steps_aside/`. Only the `read` step re-runs.
* The MM saw the blocked result's P1 family words before this fix. The fix is mechanical (seed scoping) and changes no cell, floor or rule.
