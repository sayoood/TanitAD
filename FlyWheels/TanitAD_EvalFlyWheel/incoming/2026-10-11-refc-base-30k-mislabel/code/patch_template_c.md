
## 3. Other files - grep of `Project Steering/` (at `@@REV@@` and in the live tree)

Patterns: `FANCEIL`, `0.1914`, `0.2813`, `41.09`, `0.2127`, `0.3543`, `0.4728`, `0.4109`, `oracle-in-fan`, `fan ceiling`, prose variants; both `41.09 %` and `41.09%` forms. **No further file attributes these figures to refcv7.**
* CORRECT (arm named base-30k / REF-C-base) - leave alone: `GOALS_AND_CLAIMS.md` 9131 (D-REFCV5-RANKS-THE-UNSAMPLED-FAN, "41.09 % (base) / 45.4 % (REF-C-XL)") and 10080-10083 (D-P14-RANK-1, "base-30k ADE 0.4728 -> 0.1914 m"),
  `MODEL_REGISTRY.md` 726-730, 2515-2517, 2595-2597, 3747, `REFCV5_MISSING_PIECES_PLAN.md` 68/237/460-461 (P14), `PREREG_D-SEL_REFC_SELECTION_SURFACE.md` 31/48/319/340/554 (line 319 even states "The arm travels with the number").
* ARM-UNNAMED, not a conflation (optional hygiene): `PI_VIDEO_REVIEW_2026-09-06.md` line 30 quotes 41.09 % and "0.4728 -> 0.1914 m" under P14 without naming REF-C-base; it predates any refcv7 fan by weeks, so it cannot be a refcv7 claim.
* Clean of the figures: `PLAN_REFCV8.md`, `SPEC_REFCV8_DRAFT_SOURCES.md` (its "+0.2127" is an unrelated D0 dose-response CI), `PI_DECISION_QUEUE.md`, `LOOP_STATE.md`, `PROGRAM_OVERVIEW.md`, `Benchmarks & Eval/LEADERBOARD*.md`.
* `Mission Plan.md`: not touched, no edit proposed.
* A hit for the bare digits in a raw CSV/JSON elsewhere in the repo is not a steering claim and was not counted.

## 4. RETRACTION_LOG entry (draft; append-only; class number assigned at landing - the highest class referenced at `@@REV@@` is C166)

| date | retracted claim | class | what it cost / would have cost |
|---|---|---|---|
| 10-11 | *"refcv7's proposal fan: oracle-in-fan 0.1914 m vs selected 0.4728 m, delta +0.2813 [+0.2127, +0.3543], a >2x-better candidate on 41.09 % of windows, MEASURED at 50.4k"* (GOALS_AND_CLAIMS `D-REFCV7-FANCEIL`, MODEL_REGISTRY section 15.3, and the message of commit `235156d`) | **C<NEXT> - A number's ARM travelled by adjacency, not by artifact: a MEASURED row for arm B was filled with arm A's banked figures, correct as numbers and wrong as a claim.** Recognition signal: a MEASURED row that names no raw artifact path and no `n_windows` / anchor count, whose figures equal another arm's registered figures at 4 dp. Nearest existing classes: C4 (inherited without re-verification) and C6 (the contrast varied more than the label admits - here arm, step 29,999 vs 50,400, window set 881/40 vs 4,754/139 and anchor count 128 vs 117 all changed under one name). | All five figures are `refc-base-30k`'s (`taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json`); refcv7's own (X8 raw `ctl_refcv7_50400_fan_s0.json`, 4,754 windows / 139 episodes / 117 candidates) are @@R70@@ / @@R7S@@ / +@@R7G@@ [+@@R7L@@, +@@R7H@@] / @@R7F@@ %, so the row overstated refcv7's selected ADE @@RSEL@@x and its gap @@RGAP@@x while the qualitative finding survived. **Why every figure-level check passed:** the digits were right, so only the arm label, the scope clause ("at 50.4k") and the tier label ("T1-family open loop"; the raw stamp says T0 diagnostic) were wrong, and a hybrid sentence (`sampler_ranks_the_fan` false IS refcv7's) made the row read as verified. **Evidence-class error:** stamped MEASURED although I found no refcv7 fan artifact on the dev box until the X8 roll of 2026-10-11 16:19-16:52, after the 15:39 landing (Thor not searched). Caught by the EvalFlyWheel X8 agent while producing the missing random-pick control, before the figure sized any experiment that I can find. **Durable fix proposed, not built:** a claims/registry lint that, for every row carrying figures, requires an artifact path and flags any quoted figure equal at 4 dp to a different arm's registered figure; its positive control is exactly this row. |

## 5. Mechanical guards (`raw/verify_mislabel.json`)

| guard | result |
|---|---|
@@CHK@@

A* tie the doc figures to the base-30k raw and rule out refcv7's; B* prove the OLD text is verbatim and unique in both trees; C* simulate the apply in memory (nothing written to the repo); M* are deliberate-regression controls - each guard must go RED on a knowingly broken input, and does.
Interval scope: the CI bounds quoted answer *another draw of EPISODES?* only (FULL-SET mean, episode-cluster bootstrap n_boot 2000, paired for deltas). The refcv7 panel is one training seed with an inference-seed replicate; the base-30k panel is one seed.

## 6. Unverified / not done
* `dfd97373` (the row's original commit) is not in the git dir I may read - the row's authoring time and source are UNVERIFIED beyond the landing commit `235156d`.
* Metric-horizon equality of the base-30k and refcv7 panels is UNVERIFIED (both use `_de_all` over 4 waypoints; refcv7's waypoint spacing was not checked), hence "do not pool or difference".
* Whether Thor holds an earlier refcv7 fan artifact was not checked (dev box only).
