# Rung β on v8F-mini M2b: the verdict (PREREG_V8F_BETA Amendment 1, 2026-10-20)

**Status:**
* Box 2's chain ended `ZZBETA-DONE 10-20 14:17`, tip `5741dd85`, boot unchanged. It was eval-only: no training, zero tuned parameters.
* Inputs: M2b seeds 0 / 1 from `PREREG_V8F_MINI` Amendment 4 (full corpus, the RA-1 holdout withheld).
* The verdict is computed by `taniteval/v8f_beta_eval.py`'s committed decision table: `raw/box2/eval_beta/verdict_M2b.json`.
* Tier: T0/T1 open loop on eval-141 (2,399 windows, 141 episodes). `beta_oracle` and `fan_oracle` read the ground truth: T0 diagnostics, never a result.
* Estimator: the paired episode-cluster bootstrap per seed (another draw of EPISODES). Training variance enters only through F = 0.1207 m (M2b).

## Verdict: **FAIL-GOAL** (table row 3, as committed)

| reading (seed 0 / seed 1) | value | bar | met |
|---|---|---|---|
| P1: β − M2, ADE@6s | **+2.084** [+1.648, +2.553] / **+2.225** [+1.767, +2.708] | < 0, separated, \|Δ\| > 2F = 0.2414 | ✗ (much WORSE) |
| P2: shuffled goal − β | +13.35 / +12.63, separated | the control loses | ✓ |
| P3: no family separated-worse | 10 of 13 separated-worse | none | ✗ |
| signal (T0): oracle-goal fan-only vs random | −11.15 / −11.20, separated; recovery **0.838 / 0.841** | separated and ≥ 0.5 | ✓ |
| inert? | β replaces M2's plan in **70.6 % / 63.6 %** of windows | replaced ≥ 0.05 | not inert |

**6 s ADE by arm (m):**

| arm | seed 0 | seed 1 |
|---|---|---|
| M2b's own plan (M2) | 2.1753 | 2.0546 |
| β (predicted goal) | 4.2593 | 4.2792 |
| β_oracle (GT goal latent; T0 diagnostic) | 3.3630 | 3.2366 |
| fan_random | 14.6810 | 14.6810 |
| fan_oracle (best fan member by GT; ceiling) | 1.3699 | 1.3699 |

* **Separated-worse (β − M2):**
  * speed MAE +0.32 / +0.42;
  * along-track +0.92 / +1.08 m;
  * cross-track +1.62 / +1.66 m;
  * heading +3.3 / +4.2 deg;
  * tactical lat / lon accuracy −0.10 / −0.06;
  * route error at 30 / 50 / 80 m +1.2 to +2.2 m;
  * consistency violations +0.10 to +0.14.
* **Separated-BETTER:** accel MAE −0.56 / −0.29, yaw-rate MAE −4.5 / −3.0 deg/s, curvature −0.07 / −0.05. The fan's members are kinematically clean controls.
* The committed next arm, verbatim from the table: *"z² ranks candidates (the oracle works) but S's route is not a good enough goal: build the plan's trained ĝ² head (vision + nav + turn distance only, target = the level-2 latent +6 s ahead) under its own PREREG and launch gate."*

## What the verdict does not say, and the cheaper arm it implies (author's reading, not a verdict)

**The oracle bound.**
* Even with the ground-truth goal latent, β reads **3.36 / 3.24 m**. That is **1.19 / 1.18 m worse than M2's own plan (point differences)**, against P1's 2F = 0.24.
* A trained ĝ² head can at best approach that oracle.
* ⇒ On this 117-member fan, **no goal head can make β clear P1**. The committed next arm is capped below its own bar before it is built.
* The table names it correctly. The bound is what decides whether it is worth building.

**What the WM does do.**
* Its level-2 distance ranks the fan far better than chance: recovery 0.84 of the random-to-ceiling gap.
* The fan contains far better plans than M2's: the ceiling is 1.37 m.
* So neither the WM's ranking signal nor the fan is the whole problem. The fan is wide (a random member scores 14.7 m), and the ranking resolution is not fine enough to pick from it a plan that beats a direct regressor.

**The cheapest discriminating arm (eval-only; needs the PI's go and a box-2 slot).**
* β_oracle on a LOCAL fan: candidates are perturbations of M2's own plan, not the wide control grid.
* If the oracle-goal ranking on a local fan beats M2 beyond 2F at both seeds, the WM resolves local plan differences. Then ĝ² is worth building, as a refiner around M2's plan.
* If it does not, the WM-as-planner route is closed for v8F-mini at this scale.
* Its PREREG (Amendment 2) and code come first. Both are 0-GPU.

**Where the WM route stands on v8F-mini:**
* As an auxiliary (WM-1): FAIL.
* As a planner over the control fan (β): FAIL-GOAL, with the oracle bound above.
* The open question is whether it can refine locally.
