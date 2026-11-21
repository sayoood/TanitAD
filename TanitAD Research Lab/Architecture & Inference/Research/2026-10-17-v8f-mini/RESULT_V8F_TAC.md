# PREREG_V8F_TAC: the result (2026-10-21) — the operative layer USES the tactical layer, and the tactical layer is the lever

**Status:**
* PI go 2026-10-21: *"Yes run it"*.
* Box 2's chain ended `ZZV8FTAC-DONE 10-21 06:56`. The code was sha-checked at tip `c56add43`; boot unchanged since 2026-10-17 18:43:26.
* The v8F guard (end test anchored) ran from 06:51 to the chain's end; max USB 45 °C.
* Eval only, zero training: 12 evals in about 5 min.
* Reader: `code/v8f_tac_read.py`, at the same tip. Output: `raw/2026-10-20-tac/tac_read.json`; the 12 evals are under `raw/2026-10-20-tac/eval/`.
* Tier: T0/T1 open loop. **The oracle arm is T0:** it reads GT, so it ranks levers and is never a result.
* Eval-141: 2,399 windows, 141 episodes; 34 windows have no v9 row and fall back to T's own code.
* Estimator: the paired episode-cluster bootstrap, B = 2,000 (EPISODES). Training-seed variance enters only through F: M1 0.2222, M2b 0.1207.

**Manipulation checks, all passed:**
* Each tacpred run reproduces the banked eval with a max per-window deviation of **0.0 m**.
* The shuffle changed T's code on **100 %** of windows at every checkpoint (mean |Δcode| 0.46–0.49).
* The oracle shifted it by 0.41–0.44.

## Verdicts, as committed

| checkpoint | ADE@6s: own code | donor code (shuffle) | GT code (oracle, T0) | shuf − pred | orc − pred (T0) |
|---|---|---|---|---|---|
| M1_s0 | 2.2466 | 8.6507 | 0.9506 | **+6.4041** [+5.2137, +7.7249] | **−1.2960** [−1.4511, −1.1545] |
| M1_s1 | 2.0244 | 7.3285 | 0.9614 | **+5.3041** [+4.1900, +6.4733] | **−1.0630** [−1.1794, −0.9490] |
| M2b_s0 | 2.1753 | 7.7118 | 0.9709 | **+5.5365** [+4.3396, +6.8563] | **−1.2045** [−1.3352, −1.0801] |
| M2b_s1 | 2.0546 | 6.4743 | 1.0037 | **+4.4198** [+3.3636, +5.5498] | **−1.0508** [−1.1656, −0.9386] |

* **TAC-USED: USED.** At all four checkpoints, O fed another episode's tactical code is 4.4–6.4 m worse at 6 s, separated.
  * The operative layer does not merely tolerate the tactical layer: it is steered by it.
  * The hierarchy functions end to end.
* **TAC-HEADROOM (T0): LARGE for both arms.**
  * M2b: −1.2045 / −1.0508 against 2F = 0.2415.
  * M1: −1.2960 / −1.0630 against 2F = 0.4444.
  * With the GT tactical code, O's 6 s error falls from about 2.1 m to about **0.95–1.00 m** at every checkpoint.
* **The committed next lever, verbatim:** *"USED + LARGE: … The tactical layer's QUALITY is the lever: next arm = v8F-HJ (the v9 tactical labels supervise the H-JEPA level-2 latent, T reads it …), under its own PREREG and launch gate; the weakest head first (longitudinal)."*

## The families (paired, per component; `*` = separated; MAE, so negative = closer to the human path)

**Oracle − own code (T0), M2b:**

| seed | speed MAE | along MAE | accel MAE | heading (deg) | yaw rate (deg/s) | curvature (1/m) | cross-track |
|---|---|---|---|---|---|---|---|
| 0 | −0.576* | **−1.201*** | −0.276* | −1.074* | −0.359* | −0.005 | −0.180* |
| 1 | −0.434* | **−1.022*** | −0.065* | −0.760* | +0.136 | +0.002 | −0.218* |

* **The headroom is mostly LONGITUDINAL.**
  * Along-track error drops by 1.0–1.2 m and speed error by 0.43–0.58 m/s.
  * Laterally, heading improves by 0.76–1.07° and cross-track by about 0.2 m, while yaw rate and curvature are mixed.
  * The same holds for M1: along −1.27 / −1.03, speed −0.65 / −0.45.
* It matches the tactical heads' quality: longitudinal macro-F1 0.47–0.53 against lateral 0.82–0.84 (`RESULT_FULL.md`).
* **Shuffle − own code:** every component of both families is worse at every checkpoint, separated. The largest effects are along-track (+4.5 to +6.4 m) and heading (+4.8° to +10.3°).
* **Distance keeping:** NOT COMPUTABLE in this eval. **TACTICAL:** read per run in `RESULT_FULL.md`, not paired. **STRATEGIC:** un-paused for v8F-mini; not intervened on here.

## What the oracle can and cannot say

⚠️ **The GT code includes continuous constraints computed from the FUTURE**: `lon_v_target_ms`, `v_end_ms`, `lon_d_reach_m`, `stop_x_m`, the lead gap and others.
* So "a perfect tactical layer" here includes perfectly predicting the target speed and the end speed. That is partly the operative longitudinal problem, restated as a label.
* The 1.0–1.3 m is therefore an UPPER bound on what any tactical head can buy, and it does not say WHICH part of the code carries it.
* ⇒ **The cheapest discriminating experiment is a per-group oracle**, again eval only on the same checkpoints. It replaces one part of the code at a time:
  * goals;
  * lateral decision;
  * longitudinal decision;
  * lateral constraints;
  * longitudinal speed constraints;
  * stop / lead constraints.
* If the categorical decisions carry the gain, the lever is a better tactical classifier: v8F-HJ.
* If the speed constraints carry it, the lever is a speed-target head inside T.
* It decides how v8F-HJ is built, so it runs first (`PREREG_V8F_TAC` Amendment 1).
