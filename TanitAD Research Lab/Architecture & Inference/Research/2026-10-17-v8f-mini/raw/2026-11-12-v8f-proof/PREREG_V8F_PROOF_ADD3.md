# PREREG_V8F_PROOF addendum 3: RULESHIELD, a rule held by construction on the v8F plan (frozen 2026-11-13 ~02:40, before its data)

Theorem T3 (VC-C1, VC-C2, VC-C5; `fv/FV_SPEC_V1.md` sha256 4e746f09...d3fe) says a braking-envelope shield keeps "never past the rest point" for ANY proposed speed profile. This card measures what that costs and gains on real v8F plans. Tier T1-kin on logged frames (own-motion open loop): a diagnostic, never driving performance. Zero training; banked checkpoints of the DAG-OACC recipe (`stopbal_BAL-NULL_s0`, `stopbal_BAL-NULL_s1`); eval-141 first-tick windows.

## The shield (longitudinal only; the path is not changed)
* From a plan (60 steps at 0.1 s) take the along-path speed v_t and the path. With a cue distance d_0 (along-path distance from the ego to the rest point) apply, step by step: vb = max(v - q, 0) with q = a*dt; uc = max(proposal, vb); d2 = d - v*dt; next speed = uc if B(uc) <= d2 else vb, where B is the closed-form braking distance of VC-C2. Then re-time the plan along its own path with the shielded speeds.
* a = 3.0 m/s2 (fixed before data). If the start state already violates the invariant (B(v_0) > d_0: the stop is not reachable at a), the window is counted INFEASIBLE-AT-START, braked at a, and reported apart.

## Cue variants
* ORACLE: d_0 from the logged rest point, on windows whose logged ego comes to rest inside the 6 s horizon (the existing STOP_POINT definition, quoted with file and line). Privileged: a ceiling for the mechanism, not a deployable arm.
* T-CUE: the shield is active only where the model itself flags the stop (T's STOP_POINT goal p >= 0.5), with d_0 from the model's own outputs if a stop-distance output exists; if the model emits no usable distance, this variant is reported NOT AVAILABLE with the reason, and the missing output is named as the work item.
* NULL: the unshielded plan.

## Controls that must read known values
* (s1) OVERRUN, the along-path distance travelled beyond the rest point by 6 s, is EXACTLY 0 on every cued, feasible window after the shield (tolerance 1e-6 m). Any other value is an implementation fault: SHIELD-BROKEN, stop and report.
* (s2) Deliberate regression: the shield with the fallback removed (mutation M-C1a) must show OVERRUN > 0 on at least one window where NULL overruns.
* (s3) The shielded path geometry equals the NULL path: cross-track against the NULL plan's own path is 0 (tolerance 1e-6 m).
* (s4) With no cue (non-stop windows under ORACLE) the plan is bit-identical to NULL.

## Metrics, four families
* LONGITUDINAL: OVERRUN (share of windows and metres), speed error, along-track error, 2-6 s ADE and 6 s ADE on stop windows. LATERAL: cross-track and heading error (expected unchanged up to re-timing). TACTICAL: lon decision agreement is unchanged by construction (the shield acts below T); say so. STRATEGIC: PAUSED (PI item 25).
* Estimator: paired episode-cluster bootstrap, B 2,000, shield minus NULL on the same windows; two seeds.

## Outcomes
* SHIELD-HOLDS-AND-GAINS: (s1)-(s4) pass and the ORACLE shield lowers 2-6 s ADE on stop windows with a CI excluding 0 at both seeds.
* SHIELD-HOLDS-NO-GAIN: controls pass, no separated ADE gain (the rule is then kept for free but buys no accuracy on this metric).
* SHIELD-COSTS: controls pass and a lateral or longitudinal metric gets worse with a CI excluding 0; report which.
* Reference beside (not a gate): STOPPERC S0-C forced the logged stop in the tactical code at coverage 0.7 and read -0.957 / -0.855 m tactical ADE.
* Next lever, committed: the cue. If T-CUE is NOT AVAILABLE or weak, the stop-distance output and its conformal envelope are the next card (theorem T3, residual risk = cue miss rate).
