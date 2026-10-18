# AMENDMENT A-R8-CLAMP-2: refcv8's bars are scored on CEILING-COMPLIANT windows, with a compliance row beside them

**PI decision, 2026-10-18 ~20:05 Berlin, in chat, verbatim:** *"(b ) relabling is eventuallysomething for refcv9"*.
This answers the `PI_DECISION_QUEUE.md` item of ~20:00 (option b).

**Registered by the Master Mind the same evening.** This comes BEFORE any data from the final checkpoint (step 48,852,
expected ~2026-10-21 05:00 Berlin).

⚠️ **Disclosure, kept with the rule:** the option was proposed and chosen AFTER the post-hoc capped read at step 30,000
(`D-R8-CLAMP-INSTRUMENT — first measurement`). It is the PI's decision, not an analyst's; the banked 5k–30k verdicts are
not rewritten by it.

Amends `AMENDMENT_A_R8_CLAMP_1_PLAN_VMAX_CLAMP.md` §2.2 and `SPEC_REFCV8_DRAFT.md` §5. Everything in A-R8-CLAMP-1 that is
not changed here stands: capped refcv8 primary, uncapped beside, controls K1–K4, and the NR reference side left to
`D-R8-NR-INSTRUMENT`.

## 1. Why

On T1 open loop the plan is scored against the logged human. That human exceeds the fed ceiling on **12.52 % [8.94, 16.26]**
of eval windows (30 km/h class: 23.2 %). The hard cap changes ONLY those windows: 1 of 2,283 changed windows has a
compliant human. On them a compliant plan is scored as an error, so on all windows the bar measures the label against the
human's speeding. On the compliant windows the bar keeps its meaning, and R1 is measured by its own row.

## 2. The rule (committed)

1. **Compliant window.** The logged ego's maximum realised speed over the window's valid 6-s future (`gt_max_speed_6s`,
   from `refcv6_extras.npz`) is **≤ the fed ceiling**. The fed ceiling is the model's own value:
   `MaxSpeedOneHotEncoder` → `limit_ms_of_bin`, with an all-zero row → +inf. Strict comparison, no tolerance.
   * One window set for every surface (S2 and S6) and every family.
   * The set depends only on the logged data and the fed label, never on a model, so every arm is scored on the SAME
     windows: capped refcv8, uncapped refcv8, refcv7-50,400 and the echo `ha0_ext`.
   * Its n (windows / episodes) is printed with every read.
2. **Bars.** BAR-R8-E1, E3 and NR are decided on the compliant set, with capped refcv8 primary. The all-window values are
   printed beside them and cannot override. Same estimator: paired episode-cluster bootstrap B 2,000 over the episodes
   the set contains.
3. **COMPLIANCE row** (LONGITUDINAL family), for every arm, on ALL windows and separately on the compliant set:
   * the rate of windows whose emitted plan exceeds the fed ceiling on the 0.5-s waypoint grid (`planned_max_speed`);
   * the residual excess in m/s (median, p90, max);
   * the rate of rows whose `v0` is already over the ceiling.
   On the compliant set, a plan above the ceiling is an R1 violation the human did not commit. **That number is the R1
   metric.**
4. **The braking allowance stays as landed:** `brake_ms2 = sel.accel_max` = 2.5 m/s², v7F's "brakes at −a_max until it is
   under". The PI was asked and did not change it. The strict cap (brake 0) is reported beside as a sensitivity read.
5. **Scope.** The final battery's verdict follows this rule. Banked milestones (5k–30k) may be re-read under it as EARLY
   READS. They are labelled that way and never replace a banked verdict.
6. **Relabelling** the ceiling, for example the 30 km/h class the human exceeds, is deferred to **refcv9** (PI).

## 3. Controls (before any compliant-set number is used)

* **(C1) All-True mask.** The compliant-set pipeline run with every window marked compliant reproduces every banked cell
  exactly: delta, CI, n.
* **(C2) Mispairing.** The mask rolled by one episode block (~35 windows) must change the window set and at least one
  headline cell.
* **(C3) Model-independence.** The mask is computed once per tree and is identical for every arm. Assert its sha.
