# PRE-REGISTRATION — E2-TAIL: read K2DC and K0 on their TAIL-EMA weights (MM 4a3d05, 2026-11-16 ~22:50Z)

**Written before any tail-weight number exists, and before the tail-weights tool is built.** Zero training.

## 0. Why: E2-RAW's split result (MEASURED, banked 7956bda6 + 68fe22a8)

**On raw `ckpt.pt`:**
* (a) map, (b) box, (c) S6c and (e) W4 all PASS.
* (d) FAILS at BOTH seeds on LATERAL heading / yaw-rate / cross-track / final cross-track.

**Lateral means on EMA → raw** (seed 0, `families_s0.json`, T1):

| arm | cross-track | heading |
|---|---|---|
| K2DC | 0.1040 → 0.1163 m | 2.129 → 2.227 deg |
| K0 | 0.1007 → 0.1051 m | 2.082 → 2.129 deg |

* **Reading:** the EMA read helps the PLANNER of both arms, while it under-reads K2's NEW map modules, which are 30 % step-0 weights in the EMA.
* **Two candidate explanations for the planner benefit:**
  * **(P1) Averaging.** The EMA smooths the weight trajectory.
  * **(P2) Init pull.** 30 % of the EMA is the step-0 weights, i.e. the pre-trained `ckpt_ema_48852` planner, and that pulls the planner back toward a smooth one.

## 1. The weights (exact, no new training, no init needed)

* **Formula.** `W_tail = (EMA_6000 - d^1000 * EMA_5000) / (1 - d^1000)`, with d = 0.9998 and d^1000 = 0.81872.
* **What it is.** The EMA recursion makes `W_tail` exactly the exponentially weighted mean of the raw weights at steps 5001 to 6000. The oldest weight is 0.82 of the newest, so it is close to a uniform average over the last 1,000 steps.
* **No init contamination.** It carries no step-0 weights at all. That is also true of any EMA at E3's length: 73,280 steps leave ≈ 4e-7 of the init.
* **Inputs:** each run dir's `ckpt_ema_5000.pt` and `model_ema.pt`.
* **Guards:**
  * the same decay;
  * `num_updates` 5000 and 6000;
  * the identical tensor set;
  * fp32 arithmetic;
  * a frozen tensor (identical in both files) is copied, not divided.
* **Output:** written beside them as `model_ema_tail5k6k.pt`, with a provenance block.

## 2. The read

* E2's rule unchanged (REFCV9_DESIGN 4.4), at seeds 0 and 1, on DEV.
* Clause (a) at each arm's own FIT thresholds, as in E2-RAW.
* The roll uses `R9E2_CKPT_NAME=model_ema_tail5k6k.pt` (landed 1c48ee8f).
* **Stage A** (map, ≈ 1.5 Thor-h), then **stage B** (planner, ≈ 4.4 Thor-h), as in E2-RAW. Stage B runs only if (a) passes.

## 3. The rule

* **GO** iff (a)–(e) all hold at both seeds, exactly as E2 and E2-RAW.
* **Caveat:** one training run per arm (`H-ESTIM-SEED-1`).

## 4. Mechanism predictions (decide the DIAGNOSIS, not the GO)

* **M-map:** K2DC's tail fitted map primary is ≥ raw − 0.02 (0.4712).
  * **Mechanism:** the new modules are mature in the tail.
  * **If it fails:** averaging, not maturity, hurts the map.
* **M-lat (P1 vs P2):** compare K2DC's tail LAT cross-track mean at seed 0 with its EMA (0.1040) and raw (0.1163) values.
  * **≤ 0.1080** (within 30 % of the EMA-to-raw gap, from the EMA side): **P1 averaging** carries the EMA's lateral benefit.
  * **≥ 0.1123** (within 30 % of the gap, from the raw side): **P2 init pull** carries it.
  * **In between:** MIXED.

## 5. Committed outcomes (Rule Zero: each names the next lever)

1. **GO.**
   * This is the read whose weights resemble what E3 will deliver, so E2-TAIL is the most E3-faithful screen.
   * Report to the PI, who decides whether E3 fires.
   * Code lever: short screens read tail-EMA (or a debiased EMA).
2. **(d) fails, M-lat = P2.**
   * K2's planner drifts further from the pre-trained init than K0's. That is trained in.
   * The next arm is the pre-registered E2-RAW outcome 3: K2DC at share target 0.02, ≈ 20.5 Thor-h, which needs the PI.
   * A cheaper alternative goes to the PI beside it: a planner-only weight anchor toward init (L2-SP). It is new code and needs its own pre-registration.
3. **(d) fails, M-lat = P1.**
   * Averaging helps both arms, but K2 still loses.
   * The next arm is the same 0.02 share target.
4. **(a) fails on the tail.**
   * Report it and diagnose before proposing anything.
   * Stage B is not run.

## 6. Cost and authority

* Zero training. Building the weights takes minutes of CPU on Thor. Stage A + B ≈ 6 Thor-h (ESTIMATED from E2-RAW's MEASURED stages).
* Thor is this stream's resource (PI 2026-11-02) and has been idle since the E2-RAW verdict (22:32Z).
* The pre-registered training arm (0.02 share target) is NOT started: it goes to the PI.
