# LEAN_RESULT_RULE: T3 (rules by construction), T7 (imagination) and T9 (averaging members) checked by the Lean 4 kernel

Evidence class: PROVED (Lean 4 kernel with Mathlib; `lake build` of the project rc 0 on the pod, 2026-11-13 20:36:03Z). Theorems about model classes with explicit premises; not a measurement, not a driving-performance number, and not a claim that v8F meets the premises. STRATEGIC family: PAUSED (PI item 25); no metric family is measured here.

## What was checked
* Source `lean/src/Rule.lean` (sha256 19946d8f073888f5), 18 theorems, no `sorry`; axiom audit `lean/src/AuditRule.lean` (sha256 3de2cf6595e5e79e): 18 of 18 theorems depend on `propext`, `Classical.choice`, `Quot.sound` only, read from the build log of the project `lean/src/lake_build_rule.log` (sha256 6c1be5ebdc82f532). That log is the build of all three files: 18,194 jobs, 0 errors, 68 of 68 audited theorems on the standard axioms (31 pooling, 19 hierarchy, 18 here); 71 warnings, all style notes (23 on this file: line length, docstring delimiter, module system). The one line of the log that contains the word "error" is the audit line of the theorem `model_gap_from_step_error`.
* The project now holds 72 theorems: `Pool.lean` 35, `Hier.lean` 19, `Rule.lean` 18. Toolchain and Mathlib revision as in `lean/LEAN_RESULT_POOL.md`. The log of the earlier build (`lean/src/lake_build.log`, two files) is kept as banked.
* Check history, in full: the first check of the file ended with 0 errors and one deprecation note (`if_pos`, replaced by `ite_eq_left`; no statement and no proof step changed). The check was run after the three-arm speed window of the pod queues had ended.

## Statements against the solver version
| statement of `V8_THEORY.md` | Lean theorem(s) | generality in Lean |
|---|---|---|
| T3 stop-line shield (VC-C1 checked ONE step) | `fallback_feasible`, `shield_step`, `shield_forever` | every sequence of proposals, any number of steps, any braking law that obeys the recursion |
| T3 minimal intervention | `shield_keeps_safe_proposal` | a safe proposal passes unchanged |
| T3 masked selection (VC-C3 checked K = 1..6) | `masked_argmax` | any finite candidate set, any admissibility predicate |
| T3 penalty bound (VC-C4) | `penalty_bound` | as the solver version |
| T3 residual risk is perception (a paper sentence so far) | `violations_le_misses` | any finite set of situations: the true line is passed at most as often as the predicted line lies beyond it |
| T7 imagined gain above 2 G (VC-D1) | `imagined_gain_is_real` | any candidate set |
| T7 selection (VC-D2 checked two candidates) | `imagined_choice_regret`, `exists_choice_within_two_gaps` | any candidate set; existence on any finite non-empty set |
| T7 tightness and the stronger twin (M-D1) | `two_gaps_attained`, `one_gap_not_enough` | kernel-checked examples: 2 G is attained; a gain above G alone can be a real loss |
| T7 where G comes from (a paper step so far) | `return_gap_le`, `model_gap_from_step_error` | G is at most K times the summed composed bound of T1 (`Hier.compose_bound`) for a step cost that is K-Lipschitz in the state |
| T9 ensemble identity (VC-G1, VC-G2 on the real line) | `ensemble_identity`, `ensemble_le_member_sum` | members in any real inner-product space, any number of members |
| T9 weight average of a linear head | `linear_head_soup` | any linear map between real vector spaces, any weights |

## Main statements in words
1. `shield_forever`: let the braking distance B obey B(v) = v dt + B(max(v - q, 0)) for v > 0, with B(0) = 0 and B >= 0. Start in a state where the braking distance fits into the distance to the line. Then for EVERY sequence of proposed speeds, at every step: the braking distance still fits, the speed is not negative, and the remaining distance is not negative. The network that makes the proposals does not appear in the premises.
2. `violations_le_misses`: if the shield never lets the car pass the PREDICTED line, the TRUE line is passed at most as often as the predicted line lies beyond the true one. The rule mechanism adds no violation of its own; what remains is the miss rate of the cue.
3. `imagined_gain_is_real`, `exists_choice_within_two_gaps`, `two_gaps_attained`: with imagined and true values within G, a gain above 2 G is real, the best imagined candidate is within 2 G of every candidate, and the 2 G cannot be improved.
4. `model_gap_from_step_error`: with a one-step state error eps, an expansion factor L and a K-Lipschitz step cost, the imagined cost over H steps is within K times the sum over t below H of eps (1 + L + ... + L^(t-1)) of the true one. This joins T1 to T7: the admissible horizon of imagination follows from the measured step error.
5. `ensemble_identity`: the summed squared error of the members equals their scatter around the member average plus K times the squared error of the average. So the average is never worse than the member mean, in any output dimension.

## What this does and does not establish
* Established as mathematics: a shield of this form keeps its rule for every network and every proposal sequence; the only residual risk is the miss rate of the predicted constraint; imagination is admissible exactly above twice the model gap, and that gap is bounded by the composed step error.
* NOT established here: that v8F has a cue sharp enough for the shield to be useful (RULESHIELD and CUEPROBE: with the model's own stop position nothing is gained), or a model gap small enough for imagined gains on objects (LATVIS and the trained-state reads: the pooled state holds no object motion). Both are measured quantities with their own rows.
* The braking law is a premise. `Brake` asks for the recursion at every positive speed; the closed form of the code obeys it on its grid (VC-C2, solver version), which is not re-proved here.
