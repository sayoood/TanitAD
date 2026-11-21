# RA-1 full, Amendment 2: IDM at γ = 200, the result (2026-10-21)

**Status:**
* Run: `PREREG_RA1_FULL.md` Amendment 2, A1 ×2 at level-1 γ 200 (γ_up 50), β 0.08, S = 3000. Pre-registered 2026-10-20 before any run; PI go *"Continue with the ra-1's committed IDM."*
* Box 2's chain ended `ZZRA1A2-DONE 10-21 00:34`. The code was sha-checked at tip `5b1abaa3`; boot unchanged since 2026-10-17 18:43:26.
* The temperature guard read a max USB temperature of 48 °C, against the 61 / 63 °C stop rule.
  * It switched itself off at 22:43, one minute into training: its unanchored end test matched the chain's own waiting line.
  * It was re-armed by hand within about a minute and ran to the chain's end. The fix is `142e169`.
* Built-γ check: both runs carry `cfg.gamma` 200.0 and `cfg.gamma_up` 50.0. Parameter audit: no untrained parameter; instrument_ok at both seeds.
* Reader: `code/ra1_full_amend2.py`. Output: `raw/ra1_full/amend2_final.json`.
* Tier: T1-class open loop (sensitivity is T0). Eval-141: 2,399 windows, 141 episodes.
* Estimator: the paired episode-cluster bootstrap, B = 2,000 (another draw of EPISODES); training-seed variance only through F = |s0 − s1|.

## IDM-200 (decides): **FAIL**

| | seed 0 | seed 1 | bar |
|---|---|---|---|
| A1γ200 − A0, eval-141 ADE@6s | −0.0654 [−0.1355, +0.0049] (**not separated**) | −0.1098 [−0.1864, −0.0322] (separated) | < 0, separated, \|Δ\| > 2·max(F_A1γ200 0.0121, F_A0 0.0565) = **0.1130**, at both seeds |
| met? | ✗ (not separated, below the floor) | ✗ (below the floor by 0.003) | **FAIL** |

**The committed FAIL branch, verbatim:**

> *"IDM's trajectory benefit is NOT established at gamma 50 or 200. IDM is kept for its SENS PASS (the action-sensitive latent) and dropped as a trajectory-error lever, with no further gamma arms; any later read is SEEDS (A0 / A1 seeds 2, 3) under its own amendment."*

## Beside it (decides nothing)

**Per-run 6 s ADE** (eval-141; holdout in brackets). P10-yaw-4512 reads 2.4870 on eval-141.

| arm | seed 0 | seed 1 | F (eval-141) |
|---|---|---|---|
| A0 | 2.2617 (2.7522) | 2.3182 (2.7853) | 0.0565 |
| A1 γ 50 | 2.2005 (2.6531) | 2.2111 (2.6463) | 0.0106 |
| **A1 γ 200** | **2.1963 (2.6466)** | **2.2084 (2.6488)** | 0.0121 |
| A2 (the K arm) | 2.2096 (2.6081) | 2.1923 (2.6315) | 0.0173 |

* **γ barely moves the error.** A1γ200 − A1γ50: −0.0042 [−0.0387, +0.0291] and −0.0027 [−0.0361, +0.0307], not separated at either seed. Quadrupling the IDM weight changes 6 s error by less than 5 mm.
* **γ does move the latent.**
  * Level-1 yaw-rate sensitivity (k = 10): **0.847 / 0.868**, against A0's 0.284 / 0.245. The paired difference is +0.562 [+0.459, +0.670] / +0.623 [+0.510, +0.738].
  * At γ 50 it was 0.759 / 0.772 (Amendment 1).
  * So a stronger IDM makes the latent more action-sensitive at no trajectory cost. That property matters for planning with the world model, not for direct regression.
* **Holdout (non-deciding):** A1γ200 − A0 = −0.1056 [−0.2693, +0.0371] (seed 0, not separated) and −0.1365 [−0.2856, −0.0049] (seed 1, separated).
* **K vs P10-yaw-4512:** −0.2907 [−0.3792, −0.2009] and −0.2786 [−0.3645, −0.1934]. Both separated, well beyond 2F.
* **The K arm stays A2.** Amendment 1's holdout re-pick does not include γ 200, and A2 is better on the holdout (2.6081 / 2.6315 vs 2.6466 / 2.6488). Re-picking now would move a goalpost after the data.

## The four families

**A1γ200 − A0, paired by component** (`*` = separated; every component is a mean ABSOLUTE error, so negative means closer to the human path):

| seed | speed MAE | along MAE | accel MAE | heading (deg) | yaw rate (deg/s) | curvature (1/m) | cross-track |
|---|---|---|---|---|---|---|---|
| 0 | −0.0283 | −0.0470 | −0.0119* | −0.246 | −0.060 | −0.0013 | −0.0256 |
| 1 | −0.0484* | −0.0737* | −0.0222* | −0.226 | −0.181* | −0.0020* | −0.0596* |

* Every component points the same way at both seeds. Seed 1 separates on 6 of 7, seed 0 only on acceleration.
* Same picture as the headline: IDM helps the direction of every component, but two seeds cannot separate it from A0's own seed spread.

**A1γ200 − P10-yaw-4512:**

| seed | speed MAE | along MAE | accel MAE | heading (deg) | yaw rate (deg/s) | curvature (1/m) | cross-track |
|---|---|---|---|---|---|---|---|
| 0 | −0.0596* | −0.1133* | −0.1161* | −0.814* | −1.802* | −0.0113* | −0.2432* |
| 1 | −0.0532* | −0.0826* | −0.1175* | −0.836* | −1.845* | −0.0117* | −0.2650* |

* All 7 LONGITUDINAL and LATERAL components are better than P10 at both seeds, separated.
* In every run the P10 − P10 control reads exactly +0 [0, 0].
* **Distance keeping:** NOT COMPUTABLE. The pooled cache carries no lead-agent track (n = 2,399 would have been scored). A work item, with a proposed session card.
* **TACTICAL** (trajectory-derived 5-way collapsed manoeuvre, per run, NOT paired):

  | | seed 0 | seed 1 |
  |---|---|---|
  | A1γ200 | 0.565 (κ 0.432) | 0.552 (κ 0.415) |
  | A0 | 0.555 | 0.544 |
  | A2 | 0.551 | 0.567 |
  | P10 | 0.491 (κ 0.342) | |

  * Per-window classes are not saved, so the arms cannot be paired. A work item, not a pass.
* **STRATEGIC:** PAUSED (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25).

## What this settles, and what is next

* **IDM's role in RA-1 is settled at two seeds.**
  * It makes the latent action-sensitive (SENS PASS at γ 50, more so at γ 200).
  * Its trajectory benefit over A0 is −0.06 to −0.11 m at both γ: consistent in sign and in every family component, but never beyond A0's seed floor at both seeds.
  * Two seeds with F_A0 = 0.0565 cannot resolve below about 0.11 m (`H-ESTIM-SEED-1`).
* **The committed next read is SEEDS** (A0 / A1 seeds 2 and 3), under its own amendment, on box 2, with a PI go.
  * It only refines a lever whose sign is already consistent.
  * The higher-value box-2 slot is `PREREG_V8F_TAC`, eval only, about 40 min: it decides whether the hierarchy's tactical layer is the next lever at all.
* **For the world model as a planner, γ 200 is the latent to plan in.** It carries the highest measured action sensitivity (0.85–0.87), at no trajectory cost. That is an input to the β local-fan diagnostic and to RA-2's design, not a result.
