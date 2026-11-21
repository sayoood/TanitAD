# STOPPERC stage 0: verdict on S0-PRE, S0-C and S0-M (Master Mind, 2026-11-11)

Card `PREREG_V8F_STOPPERC.md` (frozen, sha256 `70c75176…8877`; PI approval in §8), amendment `AMENDMENT_1_S0.md` (sha256 `6c4ad83b…d123`, written before any S0-M or S0-P number). Numbers come from `RESULT_S0.json` (MEASURED, this run), whose per-arm rows are banked with their sha256. Tier: **T1-kin first tick, eval-141, zero training: an oracle and ablation diagnostic, never driving performance.** Estimator: episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of episodes" only. The planner is deterministic, so there is no inference-seed floor. STRATEGIC: **PAUSED** (PI item 25).

## S0-PRE: `pool8x20` exists on both sides (MEASURED)
* Box 2's train cache has 4,572 files. The first 3 in sorted order are 20,904,597 bytes each with `meta.grids` = ['4x10', '8x20'].
* The local eval-141 cache has 141 files, and its first 3 hold `pool8x20` fp8 [101, 160, 1024].
* ⇒ Arm R needs no re-encode.

## S0-C: **CEILING-NEAR, c\* = 0.7** (both seeds)
Every control passed on both arms:
* IDENT = OWN 0.0 m;
* OWN plan vs dump 0.0 m;
* counts 117 / 2,234;
* c = 1.0 reproduces STOPORC's primary point with difference 0.0;
* k = 0 reads exactly 0.0 on every cell;
* the mutant code equals OWN bit for bit;
* own p matches the STOPDEC npz.

Tactical ADE (knots 11-30) on the 117 stop-point windows: the F-GT plan on the top-k windows by T's own p(STOP_POINT), minus OWN. The P floor is 0.7027 m.

| c (k of 117) | BAL-NULL_s0 | BAL-NULL_s1 | clears at both |
|---|---|---|---|
| 0.3 (35) | −0.342 [−0.553, −0.161] | −0.135 [−0.267, −0.038] | no (below floor) |
| 0.5 (59) | −0.605 [−0.932, −0.286] | −0.568 [−0.995, −0.244] | no (below floor) |
| **0.7 (82)** | **−0.957 [−1.222, −0.662]** | **−0.855 [−1.374, −0.410]** | **yes** |
| 1.0 (117) | −1.539 [−2.034, −1.036] | −1.477 [−2.019, −0.952] | yes (STOPORC) |

* **Reading.** The descriptive k-scan (point only, no CI) first passes −0.703 m at k = 68 (s0) and k = 73 (s1): 58-62 % of the stop-point windows. T itself says stop (p ≥ 0.5) on 7 / 14 of the 117 (6 % / 12 %).
* ⇒ A perception lever pays at the primary only if it lifts T's stop decision to roughly **60-70 % of stop-point windows**, in about the order T ranks them now. That is a 5-10× rise in coverage, with false stops held.
* This is an approximation: a better T may flag different windows.
* **Four families at c = 0.7 (s0 / s1; * = separated):**
  * PRIMARY: tactical FDE −2.565* / −2.104* m.
  * LONGITUDINAL: tactical speed −0.579* / −0.477* m/s; accel −0.034* / −0.036 m/s².
  * **LATERAL, a cost:** tactical heading +1.288* / +0.590* deg; yaw rate +2.166* / +2.178* deg/s; operative yaw rate +1.155* / +0.061 deg/s.
  * TACTICAL: n = 0, because the decision is the forced input.
  * STRATEGIC: PAUSED.
* The lateral cost comes with the forced stop. HYPOTHESIS: heading and yaw taken from near-stationary planned waypoints are ill-conditioned. This is named, not explained. STOPORC's full forced stop shows the same pattern: tactical yaw rate +2.930* / +2.917* deg/s; tactical heading +1.433* / +1.207 deg (`RESULT_STOPORC.json`).
* **What it licenses (card §4):** CEILING-FAR does not hold, so the ceiling does not go to the PI before box-2 time. Stage 1 still needs the PI's go with all of stage 0 (§8). S0-P is pending.

## S0-M (Q7): the causal `limit` carries little of T's stop ranking (not separated)
STOPCUE's rule on 2,365 windows, 131 positives, 139 episodes. T's p(STOP_POINT) with `limit` in the `off` format vs the dumps' causal bucket:

| arm | AUROC off | AUROC causal | off − causal | hit at 0.5 (off / causal) | FPR at 0.5 (off / causal) |
|---|---|---|---|---|---|
| BAL-NULL_s0 | 0.7892 | 0.8294 | −0.0403 [−0.1035, +0.0273] | 0.137 / 0.076 | 0.0067 / 0.0040 |
| BAL-NULL_s1 | 0.8159 | 0.8578 | −0.0419 [−0.0977, +0.0147] | 0.130 / 0.137 | 0.0081 / 0.0067 |

* **The finding (AMENDMENT_1_S0.md, MEASURED):** the STOPBAL-line T1-kin reads feed T a causal past-speed bucket, not the ego-future band. Q7's worry, that T's AUROC reads ego-future speed, does not apply to these reads.
* Removing even the causal bucket moves AUROC by −0.04 at both seeds, inside the interval. T's stop ranking does not come mainly from the speed channel.
* ⚠️ **Out of distribution:** T trained with `limit_ok` = 1 on all 4,572 training clips (box 2's `refcv6_speed_max_v8_train.jsonl`: 4,572 rows, 0 invalid, MEASURED 2026-11-11). So the `off` state was never seen in training, and this is an ablation, not a calibrated "vision-only T".
* It decides nothing in §4's table. It is the T-side comparator printed beside HEAD-UNDERUSES when S0-P lands.

## Next (committed by the card; nothing here needs a PI decision yet)
1. **S0-P:** the probes are being built. The eval side is local; the training side runs on box 2 after the dose-2 and SLOWWM chains (§8, Q2).
2. **Then the PI's go on stage 1,** with S0-PRE + S0-C + S0-M + S0-P. Under §8, A runs first and R runs only on RES-ADDS.
