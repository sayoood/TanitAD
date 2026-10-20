# RA-1 full, Amendment 1: the result (2026-10-20)

**Status:**
* `PREREG_RA1_FULL.md` Amendment 1 (A0 ×2, A2 ×2, A1x seed 1, at S = 3000, cell b0.08g50), plus its IDM reading (pre-registered 2026-10-20, before any run).
* Box 2's chain ended `ZZRA1A1-DONE 10-20 19:07`. The code was sha-checked at tip `890e12a6`; boot unchanged since 2026-10-17 18:43:26.
* The temperature guard never fired: max USB 48 °C, against the 61 / 63 °C stop rule.
* Reader: `code/ra1_full_amend1.py`, at the tip it ran from. Output: `raw/ra1_full/amend1_final.json`; verdict input: `raw/ra1_full/full4_verdict.json`.
* Tier, as the reader stamps it: *"T1-class open-loop trajectory (SENS is T0)"*. Eval-141: 2,399 windows, 141 episodes; holdout: 4,076 windows, 60 episodes.
* Estimator: the paired episode-cluster bootstrap, B = 2,000. It answers another draw of EPISODES; training-seed variance enters only through F = |s0 − s1|.

## The four readings, as committed

| reading | quantity (seed 0 / seed 1) | bar | verdict |
|---|---|---|---|
| 1. SENS | A1 level-1 yaw-rate sensitivity (k = 10): **0.759 / 0.772**; A1 − A0 +0.475 [+0.398, +0.556] / +0.527 [+0.438, +0.625] | ≥ 0.1 and separated above A0, at both seeds | **PASS** |
| 2. Gate replicate | A1x_s1 − A1_s1, eval-141 ADE@6s: **+0.0343 [+0.0043, +0.0658]**, separated; holdout +0.0354 [−0.0121, +0.0875] (non-deciding) | > 0 and separated | **REPLICATES** |
| 3. Holdout re-pick | holdout seed-mean ADE@6s: R0 2.7246, A1 2.6497, **A2 2.6198** (A2 better than A1 by 0.0299 > the 0.01 m tie) | R0 < A1 < A2, tie to the simpler | **K arm = A2**; K for A2: −0.2774 [−0.3784, −0.1717] / −0.2947 [−0.3882, −0.2030] vs P10, 2F = 0.0346 ⇒ **PASS** |
| 4. IDM | A1 − A0, eval-141 ADE@6s: −0.0612 [−0.1314, +0.0030] (not separated) / −0.1070 [−0.1772, −0.0407] (separated) | < 0, separated at both seeds, \|Δ\| > 2·max(F_A1, F_A0) = **0.1130** | **FAIL** |

**Per-run 6 s ADE (eval-141; holdout in brackets).** P10-yaw-4512 reads 2.4870 on eval-141.

| arm | seed 0 | seed 1 | F (eval-141) |
|---|---|---|---|
| R0 | 2.2843 (2.7301) | 2.3302 (2.7190) | 0.0459 |
| A0 | 2.2617 (2.7522) | 2.3182 (2.7853) | **0.0565** |
| A1 | 2.2005 (2.6531) | 2.2111 (2.6463) | 0.0106 |
| A1x | 2.2381 (2.6502) | 2.2454 (2.6817) | — |
| A2 | 2.2096 (2.6081) | 2.1923 (2.6315) | 0.0173 |

**What each reading says, within its bar:**
* **SENS PASS.** At full scale, A1's level-1 latent responds to the measured yaw rate (0.76–0.77, about 3× A0's). The IDM term makes the latent ACTION-SENSITIVE.
  * With K PASS, this is `PREREG_HJ_STAGE0`'s *"frozen + adapter suffices"* branch.
  * The committed next step is to design refav2 (RA-2): a Master Mind / PI step, not pre-registered here.
* **The gate replicates at a second seed.** Time-shuffled targets cost +0.034 m at seed 1, as they cost +0.038 at seed 0.
  * ⇒ At full scale the auxiliary's benefit needs TRUE time order.
  * At 540 clips it did not: there A1x matched A1.
* **The holdout picks A2, but A2 − A1 on eval-141 is not separated at either seed:** +0.0091 [−0.0218, +0.0404] and −0.0189 [−0.0492, +0.0114].
  * The level-3 latent buys 0.03 m on the holdout and nothing measurable on eval-141.
  * K for A2 passes on its own floor, as committed. A1 passed K too (Phase 2).
* **IDM FAIL.** It is not established that IDM lowers 6 s error beyond seed noise at full scale.
  * The point estimates are negative at both seeds (−0.061 / −0.107), and seed 1 is separated.
  * But A0's own seed spread (0.0565) puts the floor at 0.113, and seed 0 is not separated.
  * A1x − A0 (descriptive): −0.0236 (not separated) / −0.0728 (separated). The time-shuffled IDM gains less than the true one, consistent with reading 2.
  * **The committed next lever: A1 with γ = 200 ×2 at S = 3000**, read under this same bar. It is the stage-0 ladder's first rung; box 2, PI go.
  * ⚠️ Two seeds, with A0's spread at 0.0565, cannot resolve an IDM effect below about 0.11 m. More A0 / A1 seeds would answer the same question more cheaply than a new γ (`H-ESTIM-SEED-1`). Both need the PI's go.

## The four families (the new runs; paired arm − P10-yaw-4512 on the same windows)

`*` = separated. Every component is a mean ABSOLUTE error, so negative means closer to the human path. In every run the P10 − P10 control reads exactly +0 [0, 0].

| run | speed MAE | along MAE | accel MAE | heading (deg) | yaw rate (deg/s) | curvature (1/m) | cross-track |
|---|---|---|---|---|---|---|---|
| A0 s0 | −0.0313 | −0.0663 | −0.1042* | −0.568* | −1.741* | −0.01001* | −0.2176* |
| A0 s1 | −0.0048 | −0.0089 | −0.0953* | −0.610* | −1.663* | −0.00977* | −0.2053* |
| A2 s0 | −0.0486* | −0.0917* | −0.1111* | −0.832* | −1.764* | −0.01078* | −0.2549* |
| A2 s1 | −0.0552* | −0.0869* | −0.1141* | −0.882* | −1.797* | −0.01116* | −0.2818* |
| A1x s1 | −0.0310 | −0.0460 | −0.1106* | −0.796* | −1.813* | −0.01154* | −0.2561* |

* **LONGITUDINAL and LATERAL:** **A2, the K arm, beats P10 on all seven components at both seeds, separated.**
  * Distance keeping is **NOT COMPUTABLE** from the pooled cache: it carries no lead-agent track (n = 2,399 would have been scored). A work item.
* **TACTICAL:** NOT PAIRED. Per-window manoeuvre classes are not saved; a work item, not a pass.
* **STRATEGIC:** PAUSED (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25).
* **IDM by family:** cannot be paired. A1's Phase-1 runs predate the per-window family arrays (`6fbd5ba`). It needs an A1 ×2 re-eval with the current eval (minutes on box 2; PI go).

## Beside v8F-mini (DESCRIPTIVE, not pre-registered)

* Both lines are read on the same eval-141 windows against the same P10-yaw-4512.
* RA-1's K arm (A2) reads 2.2096 / 2.1923 m; v8F-mini's M2b reads 2.1753 / 2.0546 m (`…/2026-10-17-v8f-mini/RESULT_FULL.md`). The point ADE is similar.
* **Seed stability is not:** F is 0.017 for RA-1 A2 against 0.121 for M2b.
* **Neither are the dynamics:** RA-1 A2 beats P10 on acceleration (−0.11), while every v8F-mini run is worse than P10 there (+0.29 to +0.61).
* Different models, step budgets and training paths. This names a question, not an answer.
