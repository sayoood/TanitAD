# v8 proof programme: properties of the EXISTING hard rule code, version 3 (frozen before its check script; true time = this file's modification time)

Object under test: `project_speed(traj, cap_ms, mu=0.7, dt=0.1, g=9.81)` in `stack/tanitad/models/v8f_mini.py` at commit 3e264bbf (line 494), unmodified, imported from an export of that commit. The code under test is never edited to make a property pass; a failing property is a finding. Judge: a randomised property check (20,000 paths per family, fixed seed, float64 inputs) for every property, plus a solver proof for C-CAP.

Path families: (F1) smooth arcs with random curvature and speed profile; (F2) polylines with one or two sharp vertices (5 to 60 degrees) at random speed; (F3) stop-and-go profiles with zero-length segments; (F4) straight lines above and below the cap.

## Properties claimed by the docstring, or needed for a "projection"
* **C-CAP.** For every step of the OUTPUT, chord length / dt <= cap + 1e-6 (when a cap is given).
  Solver part (VC-K1): on one straight segment, positions s1 <= s2 on the same line with s2 - s1 = v*dt and v <= cap give chord = v*dt <= cap*dt; (VC-K2): for any three points a, b, c with |a-b| <= x and |b-c| <= y, |a-c| <= x + y (the chord across a vertex is at most the arc), encoded on squared norms in the plane.
* **C-GEOM.** Every output point lies on the input polyline (including the origin): distance <= 1e-6 m.
* **C-MONO.** The arc-length positions of the output along the input polyline never decrease.
* **C-NOOP.** If the input already satisfies v <= min(v_kamm, cap) on every interval, the output equals the input (max abs difference <= 1e-6 m).
* **C-KAMM-IN.** On the INPUT segmentation, the capped interval speeds satisfy v_new^2 * kappa_in <= mu*g*(1 + 1e-9).
* **C-KAMM-OUT.** On the OUTPUT, the same quantity computed by the same formulas from the output path (v_out^2 * kappa_out) is <= mu*g*(1 + 1e-3).
* **C-IDEM.** project_speed(project_speed(x)) equals project_speed(x) (max abs difference <= 1e-6 m).
* **C-FIRST.** A plan whose FIRST segment points away from the ego heading (heading change at step 0) is slowed by the Kamm term as a later vertex of the same angle would be.

## Reporting
For each property and family: pass share, the worst violation, and one minimal counterexample when it fails. No property is dropped for failing. Mutation control: the same checks on a mutated copy of the function with the cap line removed must turn C-CAP red.
