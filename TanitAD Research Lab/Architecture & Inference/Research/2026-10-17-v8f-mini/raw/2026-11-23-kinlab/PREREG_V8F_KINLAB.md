# PREREG_V8F_KINLAB (FROZEN 2026-11-23, before any RK step, pod census or number): the own-state label starts with the OWN kinematics

## FREEZE RECORD (MM, 2026-11-23)
* **When frozen:** after the zero-training build and before any RK training step, pod census, rollout or read. No RK number exists.
  * Build passes: `FACTS_KINLAB.md`, `FACTS_KINLAB_2.md`, `FACTS_KINLAB_3.md`.
  * One-shard construction censuses: `KINLAB_CENSUS.md`, `_2`, `_3`.
* **Code:** commit **0ca6f49d** on `agent/v8f-rollhist-cdb-20261120`, parent df250a00 (files in `CHANGED_FILES.txt`). RH, RL, RA and B0 stay at their own commits. The gold tests prove that `--dagger-label human` and `speed_v1` reproduce df250a00 bit for bit, so RK − RL is a one-lever contrast.
* **Pre-freeze construction fix, recorded:**
  * The first form (phase 2) used the speed_v1 per-step accelerations after the blend, which never pulled the integrated speed back to the reference. Its merge then put the speed back in one step.
  * On RH bank s0, shard 0, it relabelled 27.4 % of ok records. 61.5 % were refused as infeasible, and 98.6 % of the acceleration refusals sat on the merge step.
  * The MM fixed the construction: the longitudinal follows the speed_v1 RULE on the integrated speed, and the merge also needs |v − u_h| ≤ 0.05 m/s. **The bar (≥ 0.5) was committed before the second census.** No constant was changed on any census.
  * The same kind of fix is recorded in PREREG_V8F_RECLAB.
* **The run needs a PI go with its cost** (section 6).

## 0. Why (the measured case, quoted)
* **Four levers have now left the T0 cost of own rolled history standing** (ROLLHIST line; tier T1-kin/hb0.2 with G-T0 at T0; paired episode-cluster bootstrap B 2,000 seed 0):
  * the dose (RHp3: HARMS);
  * the bank readout (RHK: HARMS);
  * the speed label (RECLAB: MECH-NO, RL − RH s0 −0.0024 [−0.0273, +0.0247], s1 −0.0156 [−0.0376, +0.0058]);
  * replacement (ROLLADD: MECH2-NO, RA − RH s0 −0.0071 [−0.0352, +0.0226], s1 −0.0011 [−0.0207, +0.0157]).
* **G-T0 against B0 fails at s0 and holds at s1** on RH, RL and RA alike: RA − B0 s0 +0.0432 [+0.0059, +0.0848], s1 +0.0205 [−0.0134, +0.0559].
* **The own-action gains at T1 stand** in every arm. RA − B0 under C\*: STRATEGIC −1.82 / −3.66 / −5.83 m at 30 / 50 / 80 m, yaw rate −5.12 deg/s, ade_tac −0.369 m (`2026-11-22-rolladd/read/MM_VERDICT_ROLLADD.md`).
* **What every arm shares (MEASURED, `FACTS_KINLAB.md`).** In an own-state sample the inputs carry the OWN kinematics. `kin3` = (v, a, kappa = clamp(omega / max(v, 2), ±0.3)), `p10f` is derived from it, and **`base` = `kdx_base(kin3)` is O's prior path**. The label `gt` (30 positions at 0.2 s) follows the human path with the human's heading and curvature from NOW; speed_v1 changes only the arc timing. `reanchor_rows` moves the pose history only.
* **The gap is large (MEASURED, RH bank s0 shard 0, n = 372,791 ok records):**

  | at NOW | p50 | p95 | human window-to-window p95 |
  |---|---|---|---|
  | yaw-rate gap | 1.84 deg/s | 23.5 deg/s | 1.78 deg/s |
  | acceleration gap | 1.55 m/s² | 6.42 m/s² | 0.58 m/s² |

  So the label contradicts the prior path the model is handed.

## 1. Hypothesis (HYPOTHESIS, committed before data)
**H-V8F-KINLAB-1 (kinematic conflict).**
* **The claim:** the T0 cost of own rolled history is the conflict between the own kinematics in the INPUTS (kin3, p10f and O's prior path `base`) and a label that ignores them. The own samples teach a large steer away from the prior path, and that steer carries over to logged history, where the prior path is right.
* **The prediction:** a label that starts from the own (v, a, kappa) and merges into the human path within the feasibility envelope removes that steer. At RL's bank and dose, RK − RL at T0 ADE6 < 0, G-T0 holds, and the T1 gains stay.

**Named risks, committed before data:**
* **R1, the target moves further.**
  * The kin label departs from the speed_v1 label by p50 0.97 m and p95 7.8 m at 6 s (point distance at knot 30), and by p50 0.73 m and p95 2.8 m at 1 s (census 3).
  * If T1 reads the human timing and path as the target, longitudinal cells (along_mae, speed_mae) and lateral cells can worsen while T0 improves.
* **R2, the conflict is not the cause.** If MECH3 reads no, the cost is not in the label.
* **R3, the low-speed curvature.** kappa_o understates the true curvature below 2 m/s (it is the model's input channel, kept on purpose).

## 2. Arms
| arm | what | argv | init |
|---|---|---|---|
| **RK_s0** (the lever) | RL's stage B with the kinematic label | **RL_s0's argv with exactly `--dagger-label speed_v1` → `kin_v1`**, and `--out` changed; RH bank `/root/rollhist/bank/s0` (sha256 e5e44c01…), swap mode | UE_s0.A, as RL_s0 |
| **RK_s1** (replicate) | as above at seed 1 | the same token change; RH bank `/root/rollhist/bank/s1` (bde6566e…) | UE_s1.A |
| comparators, no new cost | RL_s0/s1, RH_s0/s1 and B0_s0/s1 (banked runs, T0 evals and T1 dumps on the dev box) | | |

**Semantics** (`v8f_recovery.kin_label_bank`, commit 0ca6f49d), for matcher-ok records only:
* The own start is the record's kin3 at NOW.
* The reference is the record's speed_v1 label; a record speed_v1 refuses is refused.
* The unicycle runs at h = 0.05 s for 6 s.
* The blend weight is w = min(1, t / 1.0 s).
* Longitudinal: a = (1 − w)·a_o + w·a_rule, where a_rule is the speed_v1 rule, clip((u_h − v) / 0.2 s, ±3.0 m/s²), on the integrated speed.
* Lateral: kappa = (1 − w)·kappa_o + w·kappa_pp, where kappa_pp is pure pursuit to the human polyline with L_d = max(4 m, 1 s·v).
* Clips: A_LIM 3.96, K_LIM 0.198, F_LIM 6.80.
* Merge: lateral < 0.10 m, heading < 1°, and |v − u_h| ≤ 0.05 m/s at one output point.
* Refusal keeps the speed_v1 label. Off-route, end-edge and non-finite records are dropped as today, and the eligible index is equal under all three label values.

## 3. Reads and bars
T1-kin/hb0.2, with G-T0 and MECH3 at T0. ROLLHIST card section 3 binds unless changed here, with the same executor, windows, cells, verdict rules and two-seed F_c.
* **P1 (lever, decides):** RK − B0 under readout C\* (`R0C10XHb20`), all four families, STRATEGIC in full.
* **G-T0 (guard, decides):** RK − B0 at T0 ADE6. The upper interval bound must be at most +0.0705 m, per seed.
* **MECH3 (mechanism, decides the next lever only):** RK − RL at T0 ADE6, paired, per seed. MECH3-YES needs the interval below 0 at both seeds.
* **ATTR (beside, no bar):**
  * RK − RL under `R0C10XHb20`, four families;
  * RK − RH at T0 and T1;
  * P2, P3 and P4 as the ROLLHIST card defines them, on RK.
* **F_T0 at four seeds (beside, printed, never used here):** |UE_sI.A − UE_sJ.A| at T0 over the four stage-A seeds, once UE_s2.A and UE_s3.A exist (the REPS2 chain). A later card may adopt it, before its own data.
* **Lever-acts check (known values):**
  * The train log's `dagger_label_kin` is greater than 0 for RK, and the selfcheck refuses a kin_v1 batch with 0 kin labels.
  * Under `speed_v1` on 0ca6f49d it is absent / 0.
  * The share of swapped rows whose `gt` differs from the speed_v1 label is greater than 0.

## 4. Pre-launch census (the trainer's selfcheck on the pod, full banks, both seeds)
* Report per tick: the relabelled share of ok records, and refusals by reason.
* **KINLAB-BLOCKED** if the relabelled share of ok records is below **0.5** at either seed's bank. This is the bar committed before census 3. Census 3 read 0.639 on shard 0 of s0 only.

## 5. Outcomes, committed in advance
Two seeds. The outcomes are mutually exclusive; a one-seed read prints the same name prefixed PROVISIONAL.

| name | condition | reading | next lever |
|---|---|---|---|
| **KINLAB-REPAIRS** | G-T0 holds at both seeds; in P1, at least one family BETTER at both seeds and none WORSE at either | the kinematic conflict was the T0 cost; own rolled history is admissible with the kinematic label | the dose with the larger T1 gains: kin_v1 at p 0.3 (stage B only, same bank) |
| **KINLAB-HARMS** | G-T0 fails at both seeds, or a family WORSE at both | the kinematic label costs more than it repairs | MECH3-YES: T_B 2.0 s, a stronger own share, under a new card. MECH3-NO: the input side (K-INPUT), under a new card: O's prior path `base` of an own sample rebuilt from the logged kinematics at r', the own pose history kept |
| **KINLAB-PARTIAL** | G-T0 holds at one seed only, or holds at both while P1 shows no family BETTER at both seeds | part of the cost is the kinematic conflict | MECH3-YES: T_B 2.0 s. MECH3-NO: K-INPUT, as under HARMS |
| **KINLAB-SEED-DISAGREE** | a family BETTER at one seed and WORSE at the other, or the decisive cells straddle the floor | not decidable on two seeds | a third seed pair on UE_s2.A (B0_s2, RL_s2 and RK_s2 stage B): back to the PI with its cost |
| **KINLAB-BLOCKED** | a gate check fails, the lever-acts check fails, or the census bar of section 4 fails | the arm does not launch, or its number is not read | fix and re-gate; a census below the bar goes back to the MM for a new card, and is never tuned |

**Precedence:** BLOCKED, then HARMS, then SEED-DISAGREE, then the rest in table order.

## 6. Launch gate, reachability and cost
* **Gate:** the RA chain's gate pattern on the new commit.
  * G-SUITE at 0ca6f49d on the pod with 0 regressions.
  * The tree blob-checked against the commit.
  * The gate record mints only for RK argv that names `--dagger-label kin_v1` and the RH banks, with the selfcheck's `dagger_probe` reading `n_label_kin` > 0 and the census bar met.
  * zerograd and post gates after training.
* **Reachability (dev box, 0ca6f49d):** a non-zero gradient difference between kin_v1 and speed_v1 on a planted batch, and exactly 0.0 for speed_v1 against itself (`test_v8f_kinlab.py`). 19/19 mutants RED.
* **Cost: ESTIMATED 0.6–0.75 pod-h of GPU-lock time** on the LoRA pod.
  * The basis is the MEASURED RL chain (0.74 pod-h, including tree prep) and RA chain (0.52 pod-h).
  * The labeller runs once at load, on the CPU.
  * The two arms run serially after the ADD123 probe.
  * A projection above 0.9 pod-h goes back to the PI.
* **The read:** the dev box, CPU, about 2-3 h, reusing the RL, RH and B0 dumps.

## 7. Decision for the PI
Run RK_s0 and RK_s1 at about 0.6–0.75 pod-h of lock on the LoRA pod, after ADD123.
