# LEAN_RESULT_HIER: T1 (multi-level prediction) and T2 (a decision before the trajectory) checked by the Lean 4 kernel
Evidence class: PROVED (Lean 4 kernel with Mathlib; `lake build` rc 0 on the pod, 2026-11-13 19:15:02Z). Theorems about model classes with explicit premises; not a measurement, not a driving-performance number, and not a claim that v8F meets the premises. STRATEGIC family: PAUSED (PI item 25); no metric family is measured here.
## What was checked
* Source `lean/src/Hier.lean` (sha256 00eda7ec845aef01), 19 theorems, no `sorry`; axiom audit `lean/src/AuditHier.lean` (sha256 a28c5db8fe945bae): 19 of 19 theorems depend on `propext`, `Classical.choice`, `Quot.sound` only, read from the build log `lean/src/lake_build.log` (sha256 e05125a62a652469).
* Toolchain and Mathlib revision as in `lean/LEAN_RESULT_POOL.md`. Build of the project: 18,190 jobs, 0 errors; 19 warnings on this file, all of them style notes (line length, docstring delimiter, module system, one flexible `simp`).
* Check history, in full: the first check stopped in ONE theorem (`risk_split_one`) on a notation of this Mathlib version (the real inner product is written without a subscript); the statement and the proof steps were not changed, the notation was. The other 18 theorems passed the first check.
## Statements against the solver version and the paper steps
| statement of `V8_THEORY.md` | Lean theorem(s) | generality in Lean |
|---|---|---|
| T1 composed bound (reduction: induction on the step count) | `compose_bound`, `geom_succ` | any step count H, any L >= 0 |
| T1 the bound is attained | `compose_bound_attained` | the bound satisfies the recursion with equality |
| T1 jump bound | `jump_bound` | any jump length k, any number of jumps m |
| T1 criterion (VC-A1..A5 were instances) | `geom_factor`, `jump_sum_pos`, `jump_beats_compose_iff`, `jump_beats_compose_iff_one` | every horizon m*k with m >= 1; the horizon drops out |
| T2 exact risk split (VC-B1, VC-B2; reduction: linearity of finite sums) | `risk_split_one`, `risk_split` | outcomes in any real inner-product space, any weights, any finite set of groups |
| T2 non-dominance under squared error | `mean_is_optimal` | no single plan beats the weighted mean |
| T2 channel (b), VC-B3 and VC-B4 | `mean_plan_misses_mode_a`, `mean_plan_misses_mode_b`, `decided_hits_mean_misses` | any real normed space (the solver version: the real line) |
| T2 channel (b), new: every single plan, not only the mean | `single_plan_cannot_hit_both`, `flat_hits_one_mode`, `flat_hit_count_le` | any normed group; any plan c |
| T2 channel (b), new: dominance in counts | `decided_hit_count_ge`, `hierarchy_dominates_flat` | any finite set of outcomes of one context, any selector |

Both reductions that `V8_THEORY.md` lists as paper arguments for T1 and T2 (induction on the step count; linearity of finite sums) are now machine-checked.
## Main statements in words
1. `compose_bound`: if the error grows by at most the factor L per step and gains at most eps per step, then after H steps it is at most eps * (1 + L + ... + L^(H-1)). `jump_bound`: a level that predicts k steps at once with error eps_k has at most eps_k * (1 + L^k + ... + (L^k)^(m-1)) after m jumps.
2. `jump_beats_compose_iff`: at every horizon m*k the jump bound is below the composed bound if and only if eps_k < eps * (1 + L + ... + L^(k-1)); for L = 1 if and only if eps_k < k * eps (`jump_beats_compose_iff_one`). This is a comparison of worst-case bounds: it says which model class has the better guarantee, and the premise (eps_k against the composed bound of one jump) is measured (WMCEIL part d, JUMPCURVE).
3. `risk_split`: for any plan that is constant on each mode, weighted squared error = scatter inside the modes + weighted squared distance of the plan from the mode means. `mean_is_optimal`: so under squared error no plan beats the mean, and a selector that reads only what the flat model reads cannot gain.
4. `flat_hits_one_mode`, `flat_hit_count_le`: under a hit-or-miss criterion with tolerance tol, if two modes are further apart than 2*tol + 2*r (r the spread inside a mode), then the outcomes that ANY single plan hits all lie in one mode, so it hits at most as many outcomes as the larger mode holds.
5. `hierarchy_dominates_flat`: under the same premises and r <= tol, a plan decided by a selector hits every outcome whose mode was selected correctly; if the selector is right on more outcomes than the larger mode holds, the decided plan hits strictly more outcomes than every single plan.
## What this does and does not establish about dominance
* Established as mathematics: the two exact conditions under which the hierarchy has the better guarantee (T1: the jump error against the composed bound of one jump; T2 channel (b): mode separation above 2*tol + 2*r and selection accuracy above the share of the larger mode), and the exact condition under which it has none (squared error with equal information: `mean_is_optimal`).
* NOT established here: that v8F meets the premises on driving data. The premises are measured quantities: the T1 premise holds in the linear class with margin (0.416 of ten one-step errors at 2 s, T0, `wm/VERDICT_WMCEIL.md`); the T2 premise is open (top-1 selection 57 to 59 %, T0; the share of the larger mode per context and the mode separation against a tolerance are not yet tabulated on the eval windows). The measured comparison of hierarchy against flat is the six-seed replicate HIERREP (addendum 11), running.
* The dominance theorem is conditional by construction: where the selector is right on fewer outcomes than the larger mode holds, the theorem is silent and the flat plan can be ahead. That is the reading of the bulk-window result of HIERWHERE (hierarchy behind on common-case windows in three of four cells).
