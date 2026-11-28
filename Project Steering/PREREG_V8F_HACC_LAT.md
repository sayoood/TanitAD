# PREREG_V8F_HACC_LAT: remove HACC's T1-kin lateral deficit at zero training (2026-10-26)

**Written before any number of either arm exists.** Master Mind, 2026-10-26 ~01:00 Berlin. Claim row `H-V8F-HACC-LAT-1`.

## 0. Why (MEASURED, `RESULT_V8F_T1KIN_HACC.md`, outcome T1 BEATS THE FLOOR)

* **HACC − P10 at T1-kin m = 10:**
  * ADE6 −0.829 / −0.795 m, separated, at both seeds.
  * Curvature MAE is separated-WORSE at both seeds (+0.040 / +0.016 1/m), and yaw rate at s0 (+4.2 deg/s).
* **The committed next step:** LATERAL is the next lever's target, and the first lever is *"HACC's λ rule re-fitted on T1 executed tracks of TRAINING windows"*.
* **The diagnosis (descriptive):**
  * The excess sits in plans issued from the model's own executed state, at low speed. GT speed below 5.2 m/s carries +0.40 / +0.25 1/m, and speeds above 11.8 m/s carry ≈ 0.
  * It is not the replan seams and not the camera viewpoint.
  * HACC's objective is in metres (positions plus a second difference), so it barely sees a heading wobble at walking pace.
  * ⇒ A second arm aimed at the low-speed lateral residual is pre-registered beside the committed one.

## 1. Tier, estimator, data

* **Tier T1 (T1-kin)**, exactly `PREREG_V8F_T1KIN_HACC`'s settings: m = 10, `--nav deadreckon --scene logged --limit logged --schedule periodic`; eval-141 (2,399 windows, 141 episodes); dev-box CPU. ⛔ Not closed loop.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0. The rows are the four-family rows of `PREREG_V8F_T1KIN_HACC` Amendment 1, computed by the same reader.
* **F = 0.1207 m**, F/2 = 0.0604 m, 2F = 0.2415 m.
* **Fitting data: TRAINING windows only.** These are HBLEND's 141 training clips and 9,582 windows (`fit_clips.json`, cached at `scratchpad/hblend/cache/train`), disjoint from eval-141.
* **Fitting samples:** the plans issued at T1-kin ticks (m = 10) of the HACC model on those windows. Per tick the build records:
  * τ (O's plan before the post-map, `traj30_o`);
  * P (the P10 prior);
  * the measured ego speed v0 at the tick (`p10f`, an admissible input: measured ego motion);
  * the GT from the tick, in the tick's executed frame (the logged future, i.e. "rejoin the human's path").

  A `--dump-ticks` flag on `v8f_t1_eval.py` records these. Without the flag the tool's output must be unchanged bit for bit.

## 2. Arms (zero training; inference keeps vision + ego dynamics only; neither reads the situation classifier)

| arm | what | fitted on |
|---|---|---|
| **A, HACC-T1** (committed) | HACC's [30, 2] weights re-fitted: HACC's objective (position + λ · D2, origin prepended, w ∈ [0, 1], no monotone constraint) and its train-only λ rule (grid m ∈ {0, 1, 4, 16, 64}, λ = m · λ0; the smallest m with train accel and curvature ≤ 0.98 × P10 and ADE2, ADE6 < P10), applied to the tick samples | train tick samples |
| **B, HACC-V** (diagnosis) | HACC unchanged, except that the CROSS axis falls back to the prior below a speed gate: w_cross,k := 0 when the measured tick speed v0 < v\* (the along axis untouched). v\* ∈ {2, 4, 6, 8} m/s: the smallest grid value whose train tick-sample curvature MAE (first 10 knots, HACC-V − P10) is ≤ 0, AND whose train ADE6 is within F/2 of HACC's; if none qualifies, v\* = 8 | train tick samples |

Arm B needs one model change: an optional gate scalar beside `hacc_w` in `V8FMini`. It is absent by default, which leaves `forward_infer` bit-identical, and it is never read by `_forward`.

## 3. Controls (read before any outcome; a failure ⇒ REFUSED)

* **C-A1:** arm A's pipeline, run with HACC's own weights, reproduces the T1KIN HACC m = 10 dumps to ≤ 1e-9 m.
* **C-A2:** arm A's fit with λ = 0 equals the clipped closed form to 1e-12.
* **C-B1:** arm B with v\* = 0 (the gate never fires) reproduces the T1KIN HACC m = 10 dumps to ≤ 1e-9 m.
* **C-B2:** with v\* = ∞, every tick's cross-axis plan equals P10's cross plan to 1e-6 m.
* **C-T:** `--dump-ticks` absent leaves the tool's output unchanged bit for bit.
* **C-T2:** the tick samples at tick 0 equal the open-loop batch's τ and P at tick-0 ages.
* **Reachability:** a mutant that drops the gate (or the new weights) from `forward_infer` turns a test RED.
* **The reader's controls** R-1..R-3 and C-1..C-4, as in `PREREG_V8F_T1KIN_HACC` (with Amendments 1–3).

## 4. Read, and the outcome table (no bar moves after data)

**Per arm X and seed s, at T1-kin m = 10 on eval-141:**
* **LAT-DOM:** X − P10 meets `PREREG_V8F_T1KIN_HACC`'s DOMINANCE. That is: ADE6 point ≤ −2F with the interval excluding 0, AND no Amendment-1 row separated-worse.
* **KEEP:** the 95 % upper bound of X − HACC ADE6 is ≤ +F/2.

| outcome | condition | next (committed) |
|---|---|---|
| **T1 DOMINANCE REACHED** | some arm has LAT-DOM and KEEP at both seeds. If both arms do, the deployed one has the lower mean ADE6 over the two seeds; a tie goes to A | Bake the arm into the deployable checkpoints. Re-render the evidence video on its T1 executed tracks. Restate the A40 case (`D-V8F-SPEED-PERCEPTION`) with AR-1 at T1 dominance. Then the brackets (`--scene frozen`, `--limit off`) on the deployed arm |
| **LATERAL FIXED AT A COST** | some arm removes every separated-worse row at both seeds, but fails KEEP or the ADE6 bar | Report the trade per seed. Next: the same arm with a finer grid, under its own amendment, before any further read |
| **NOT FIXED** | no arm removes the separated-worse lateral rows at both seeds | The residual is O's lateral output itself. Next: a TRAINING lever, O's lateral head with a heading term at low speed (the lateral analogue of OACC), on box 2 under its own PREREG |
| **MIXED** | anything else | Reported per seed and per arm |

**Beside (decides nothing):** each arm − HACC on every row; the low-speed quintile table of `RESULT_V8F_T1KIN_HACC.md` §diagnosis for each arm; m = 30 per arm; the chosen λ / v\* per seed.

## 5. What this does NOT license

* ⛔ Closed-loop or "drives" wording.
* ⛔ Re-choosing v\*, the λ grid or the rules after an eval number is seen.
* ⛔ Any eval window in a fit.
