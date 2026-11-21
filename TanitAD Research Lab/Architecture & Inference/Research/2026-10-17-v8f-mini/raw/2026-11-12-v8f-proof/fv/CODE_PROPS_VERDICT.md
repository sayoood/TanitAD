# What the existing hard rule code guarantees: `project_speed` at 3e264bbf (MEASURED, code property check; not a model eval, no tier)

Spec `FV_SPEC_V3_CODE.md` (sha256 36d2aacf...e649, frozen before the check script). Raw: `CODE_PROPS_RESULT.json`. Code under test imported unmodified from an export of commit 3e264bbf (`stack/tanitad/models/v8f_mini.py:494`). 20,000 random paths per family, fixed seed, float64. Solver part on Z3 5.4.1: VC-K1 PROVED, VC-K2 PROVED, mutation M-K2 REFUTED.

| property | result |
|---|---|
| C-CAP: output speed <= cap | **HOLDS**, 100 % of every family; proved for the construction (VC-K1, VC-K2: the chord across a vertex is at most the arc). Mutation control (cap removed) turns it red: pass share 0.35 |
| C-GEOM: output lies on the input path | **HOLDS**, 100 % |
| C-NOOP: a plan that already complies is returned unchanged | **HOLDS**, 100 % |
| C-KAMM-IN: capped speeds satisfy v^2*kappa <= mu*g on the INPUT segmentation | **HOLDS**, 100 % (by construction) |
| C-MONO: arc position never decreases | 99.97 % on smooth arcs, 100 % elsewhere. The misses are paths of about 2.3 full turns in 6 s (curvature 0.076 1/m at 30-34 m/s), where the nearest-point arc position of the checker is ambiguous; not classified further. The code builds the positions as a cumulative sum of non-negative terms |
| **C-KAMM-OUT: the OUTPUT satisfies v^2*kappa <= mu*g by the same formula** | **FAILS.** Pass share 0.49 (smooth arcs), 0.10 (polylines with vertices), 0.58 (stop-and-go), 0.0008 (vertices, no cap). Median ratio to mu*g among failing paths 1.75, 6.07, 1.17, 6.33; worst excess 33 times |
| **C-IDEM: applying the projection twice equals applying it once** | **FAILS** on the same paths (the second pass moves points by up to 9.7 m) |
| **C-FIRST: a turn at step 0 is slowed like the same turn later** | **FAILS** on 100 %: a turn of 10-60 degrees at step 5 is always slowed, the same turn at step 0 never (the heading change of the first segment is set to zero). Example: 32.3 degrees at 17.0 m/s |

## Reading
* The speed cap is a real guarantee for every network output. That is theorem T3 in the code that already exists.
* The grip term is not a projection. It caps each interval with the curvature measured on the input sampling, then re-samples the same polyline more finely. A vertex angle then falls into a shorter chord, so the output read by the same formula exceeds the bound by about v_in / v_out. Any consumer that measures curvature or yaw rate on the projected plan sees spikes: several steps with no heading change, then a jump.
* Whether this moves a reported number depends on how often the grip term is active in the evaluation; that share is NOT MEASURED here.
* A projection that holds on its own output needs the constraint stated on a path that does not change under re-timing (curvature as a function of arc length, the start heading included). That is a code change with its own card; nothing was edited here.
