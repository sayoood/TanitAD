=== LFS ===
The level floor LF (§7.40, addendum A3, zero training) reads P-FAILS-M1-MOVED at both doses. Read outside O as a floor on the plan's progress, the heads' level binds inside the crawl and moves M1 the right way. It closes 0.062 / 0.056 of the excess at κ 0.50 and 0.395 / 0.392 at κ 0.75, where the guards fail broadly (ADE6 +0.3683 / +0.2172 m), and the composite fails D2 on M1 (+0.0380 / +0.0401 against P10). The floor speed misses the logged speed by 3.894 / 3.979 m/s, so the committed branch names the level estimator as the bottleneck. Before any compute is asked for the estimator levers, addendum A4 measures the floor with the logged speed in place of the estimate, a ceiling diagnostic that is never a driving result (§7.40.6).
=== S81 ===
30. **The level floor LF reads P-FAILS-M1-MOVED at both doses** (§7.40). Read on T1-kin `reindexed` with the paired episode-cluster bootstrap (B = 2,000, seed 0, eval-141); zero training; the heads come from one LLR run per seed.
    * **Designed and frozen before any rollout (MEASURED).** Addendum A3 (`{{RES_A3P}}…`, landed `{{LAND_A3P}}`) puts LLR's level and lead heads outside O, as a floor on the plan's along-track progress along O's own path: C0's planner, κ 0.50 and κ 0.75. Two rulings fixed the reader before any rule was read.
    * **Result.** <<LFS>> Its limits are in §7.40.6.
=== R83 ===
**Update (v40).** One item is written up here, and the adopted base is unchanged (DAG-OACC):
* **The level floor LF (§7.40, T1-kin on `reindexed`, zero training):** <<LFS>>
* **What is blocked:** no counted DOMINANT exists, so no evidence video and no scale signal follow. The estimator levers (the A40, T2 re-perception, a stronger estimator on more data) are the PI's to provision, and A4's ceiling read is the measurement that says whether they are worth asking for.
=== ROW ===
| 1az | **LF** (zero training; `H-LEVELFLOOR-1`): C0's planner with LLR's level and lead heads read outside O as a floor on the plan's progress, κ 0.50 and κ 0.75 | card `PREREG_LEVELFLOOR_A3.md` `{{RES_A3P}}…` (before any LF rollout) with rulings `{{RES_AR31}}…` and `{{RES_AR32}}…`: the CL rules with CL := LF; committed outcomes: D holds ⇒ counted DOMINANT and a scale signal with the video evidence; P holds with G or D failing ⇒ a named dose or time-gap arm; P fails with M1 moved ⇒ the estimator is the bottleneck and its levers go to the PI | **P-FAILS-M1-MOVED** at both doses (§7.40, T1-kin): share closed 0.062 / 0.056 at κ 0.50 and 0.395 / 0.392 at κ 0.75, guards failing; next: addendum A4, the ceiling read with the logged speed in place of the estimate (`{{RES_A4P}}…`, §7.40.6) |
=== I18 ===
18. **v40 reads and their limits.**
    * **Training.** The LF read (§7.40) has zero training. O is C0's own checkpoint in every LF − C0 contrast, and the floor heads come from one LLR training run per seed (`H-ESTIM-SEED-1`). κ 0.75's M1 effect therefore sits inside the C0REP floor at s0 (0.0211 against 0.0362) and outside it at s1 (0.0234 against 0.0038).
    * **What the interval answers.** The paired episode-cluster bootstrap answers another draw of episodes only. The planner and F are deterministic, so no inference seed applies.
    * **The scene.** Two doses were read. `reindexed` is a longitudinal-only approximation: under the floor the ego runs ahead of the human on part of the ticks, while the dynamic agents stay at the human's time.
    * **The reader.** Its lift control was replaced after the rollouts and before any rule was read (A3_RULING_2), and the negative chord lift is reported as a result.
    * **A4.** Its level input is the human's logged speed and is inadmissible at inference, so its read is a ceiling and never a driving result.
    * No T2 number exists.
