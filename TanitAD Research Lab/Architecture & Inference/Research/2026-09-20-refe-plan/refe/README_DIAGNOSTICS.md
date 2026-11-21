# `diag_*.py` — the measurements that settled the scorer, kept so they can be re-run

These are not tests; they are **controls**. Each one exists because a plausible conclusion had
already been drawn without it, and each is re-runnable against any banked simulation log:

```
python diag_<name>.py <path-to-a-simulation_log.msgpack.xz>
```

| file | the question it answers | what it measured |
|---|---|---|
| `diag_keys.py` | does the READER see everything the calculators write? | `CenterLine.info` 0.0691 vs 0.3211 while `.reward` was 0.7500 for both — the discrimination was there from the first run and the reader was discarding it |
| `diag_curbdist.py` | does the deliberate-regression arm actually violate anything? | the "25 m sideways veer" moves the ego AWAY from the nearest curb: 10.66 m → 19.06 m at +14 m, minimum 0.87 m at +40 m, never touching |
| `diag_aimed.py` | endpoint scoring vs step-wise rollout | the SAME over-curb path reads `OffRoad.info` 0.0000 at its endpoint and 1.0000 step by step |
| `diag_determinism.py` | are the targets reproducible across processes? | `collision_reward_weight` drew −1.0415 instead of −1.0; stable 5/5 within a process, different between them |
| `diag_cost.py` | what does a target bank actually cost? | one pass 41.6 ms; rollout 842 ms/candidate at stride 1, 444 ms at stride 2. ⚠️ take MINIMA — a first run under load reported stride 4 slower than stride 2, which is arithmetically impossible |
| `diag_batch.py` | can the N dimension score all proposals at once? | 2.5× at N=8, and 3 of 6 calculators still untiled (`goal_reaching`, `center_line` index a tensor left at N=1). Not the lever; parallel processes are |

⭐ The gate itself is `scorer_gate.py`; it is the thing to run before trusting any target bank.
Its acceptance criterion is not "it runs" — both earlier failures ran perfectly.
| `diag_goal.py` | is `goal_reaching` broken, or correctly silent? | goals sit at 30 m and 60 m with a 3 m threshold while the teacher's 4 s path ends at 24.78 m — so 0 everywhere in that frame is CORRECT. Driven onto the goal point it reads 1.0. Over 664 banked rows: `lon x1.5` 45/74, `teacher` 13/74, `stopped` **0/73** |

⚠️ **Two of these exist because a single observation was generalised into a property of the
instrument** — `SCORER_INERT` from one readout, `goal_reaching is unvalidated` from one frame. Both
were retracted the same day by a control, and neither would have been caught by re-reading.
| `diag_enrich_cost.py` | the lane graph is a measured no-op — is it also free? | **28 ms**, 0.2 % of the ~12 s frame cost, 0.3 min over a 654-frame rank. It stays. ⚠️ I had assumed ~1 s and was about to remove it on that guess; pricing it took one probe |

## Running these, and one way the run itself can hide

⛔ **Launch with `python -u` whenever stdout is redirected to a file.** MEASURED 2026-09-20: a
400-step training run wrote **0 bytes** for over 20 minutes because Python block-buffers a
redirected stdout, so a healthy job and a hung one look identical the whole time. That is the
`wait-loops-must-match-failure` family in a different costume: the monitor cannot fail loudly
because it never receives anything to fail on.

## The conformance instruments (2026-09-20)

Written to prove the seven fixes that answered the paper-conformance review. Each is re-runnable and
each carries controls, because a fix passing a test its own author wrote is weak evidence.

| file | what it proves | key numbers |
|---|---|---|
| `diag_architecture.py` | registers compress, decoders read different tensors, rotary encoding replaced the random table, the WTA metric no longer prices a radian like a metre | trajectory decoder context 16 tokens vs scoring decoder 128, different storage pointers; rotation preserves the norm to 9.5e-07; the review's counterexample flips back |
| `diag_scorer_components.py` | progress is bounded and route-relative, the ego's kinematics follow the candidate, selection is the benchmark rule | half-speed EP exactly 0.500, overspeed clips at 1.000, stopped 0.000; derived speed 6.197 against displacement/time 6.196 |
| `diag_schedule.py` | the learning rate starts at 2e-4 and reaches EXACTLY zero | halfway 1.000e-04; the `--no-cosine` control stays flat |
| `diag_planner_holds.py` | the planner cannot silently drive a parked car | refuses after 20 consecutive holds; a single hold correctly does not raise |

⛔ **Three of these exist because the FIX was wrong, not the original code.** Writing the proof
caught a velocity that was exactly 2x too large, a "progress" measured along whichever lane happened
to be nearest, and a polygon cache I orphaned behind an early `return` while editing. All three
compiled, imported and ran cleanly.
⭐ **Make one arm an IDENTITY rather than a comparison.** The 2x error was invisible to every
comparison because both the wrong and the right value look like plausible speeds; it only shows up
against `displacement / elapsed time`, which must equal it.
