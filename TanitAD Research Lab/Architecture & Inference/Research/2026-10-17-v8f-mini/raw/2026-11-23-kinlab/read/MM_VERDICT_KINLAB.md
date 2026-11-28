# MM verdict: KINLAB read (RK_s0, RK_s1), 2026-11-23

**Outcome: KINLAB-PARTIAL with MECH3-NO.**

* **Card:** `PREREG_V8F_KINLAB.md`, sha256 `7db5d7aa…`, with Amendment 1 (`948670d9…`, the label cache). Both were frozen before any RK step or number.
* **PI go:** "Yes, run both" (RK, then ADD125), in the MM chat on 2026-11-23, before 06:09Z.
* **The answer:** starting the label where the own state's kinematics point and blending it into the human's path over 1.0 s (`--dagger-label kin_v1`) does **not** lower the T0 cost of rolled history against the speed label. The T1 gains against B0 stand, as for RH, RL and RA.
* **Next lever** (the card's PARTIAL row, MECH3-NO branch): **K-INPUT**, the input side, under a new card. In a swapped own sample, O's prior path `base` is rebuilt from the logged kinematics at r', and the own pose history is kept.

Evidence class MEASURED (ours). The RK arms trained on the LoRA pod; the read ran on the dev box with zero training.

## Arms and gates
| item | value |
|---|---|
| arms | RK_s0 / RK_s1 = RL's argv with exactly the three changes of Amendment 1: `--dagger-label kin_v1`, `--out runs/RK_sK/train`, and `--dagger-label-cache /root/rollhist/kin_cache/s<K>_kin_v1.npz`. Code 1a8b11a1, RH banks, readout causal_g1x, p 0.1, 3,000 steps of stage B. 66 tokens; argv sha256 s0 `b33c3e9c…`, s1 `ef02f855…` |
| GPU lock | 06:33:55Z to 07:07:05Z = **0.55 pod-h**. The card ESTIMATED 0.6-0.75; its return line is 0.9 |
| census (section 4, bar 0.5) | relabelled share of ok records: s0 **0.6398** (953,301 of 1,490,036), s1 **0.6397** (945,608 of 1,478,090) |
| label cache | sha256 s0 `c882733d…`, s1 `5a99afdc…`. The selfcheck's built levers read the same path and sha256 as the file on the pod at launch |
| gates | gate record MINTED (both); zerograd PASS; post 252/252 tensors with Adam state and moved; G-SUITE at 1a8b11a1: 0 regressions, 0 new failures, 0 missing |
| the lever acted | train log `dagger_label_kin` sum **9,987** (s0) / **9,976** (s1) over 60 records (rows 191,664 / 191,687); selfcheck `dagger_probe`: 23 of 46 (s0) and 33 of 51 (s1) selected rows carry a kin_v1 label, `rl_label_rows_changed` 33 / 45, every control 0.0, pass |
| records | `pod_rk/RK_s{0,1}/` (argv, selfcheck, selfcheck.result, summary, summary_check, post, train.log, kclip, zerograd, lever_acts, mint / verify / pair logs), `pod_rk/rh_rk_chain.log`; all 27 files checked against the pod's sha256 (`pod_rk/SHA256_POD.txt`) |

⚠ The card's third lever-acts item asks for "the share of swapped rows whose `gt` differs from the speed_v1 label". The selfcheck has no field under that name. It is read here from the kin-label counts (a kin_emitted label differs from its speed_v1 reference at its start by construction), not from a direct comparison.

## Tiers and estimator
* P1 and ATTR are read at **T1-kin/hb0.2**, the executor of record: a kinematic rollout on the model's own actions, with a heading bound of 0.2 1/m.
* G-T0 and MECH3 are read at **T0**. T0 is a world-model diagnostic, never driving performance; the card makes it binding.
* Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 (`taniteval/ci.py`), over 141 episodes and 2,399 T0 windows. It answers "another draw of EPISODES"; the two seeds answer "another TRAINING run". Inference is deterministic.

## G-T0 (decides): RK − B0 at T0 ADE6
The bar is an upper bound ≤ +0.0705 m per seed. F_T0 is INHERITED from one seed pair, so it is a weak floor.

| seed | point | interval | verdict |
|---|---|---|---|
| s0 | +0.0364 m | [+0.0004, +0.0746] | **FAILS** (upper bound 0.0041 m above the bar) |
| s1 | +0.0070 m | [−0.0304, +0.0456] | **HOLDS** |

Beside, at the same windows: RL − B0 s0 +0.0479 [+0.0090, +0.0897], s1 +0.0061 [−0.0312, +0.0470]; RH − B0 s0 +0.0503 [+0.0060, +0.0929], s1 +0.0217 [−0.0122, +0.0596]. RK has the smallest s0 point of the three, but the split is the same.

## MECH3 (decides the next lever): RK − RL at T0 ADE6
| seed | point | interval |
|---|---|---|
| s0 | −0.0115 m | [−0.0321, +0.0074] |
| s1 | +0.0009 m | [−0.0157, +0.0180] |

Neither interval sits below 0, so this is **MECH3-NO**. Instrument: `mech_rk.py`, output `MECH3_KINLAB.json`. Beside: RK − RH s0 −0.0139 [−0.0385, +0.0123], s1 −0.0147 [−0.0395, +0.0073]. Neither is separated.

## P1: RK − B0 under C* (R0C10XHb20), four families
Each cell gives the two-seed D, its interval, the per-seed values, and the floor F (two-seed F_c).

**LONGITUDINAL: MIXED**
* speed −0.248 m/s [−0.374, −0.142] (−0.188 / −0.308), F 0.126: BETTER.
* along −0.209 m [−0.379, −0.062] (−0.124 / −0.294), F 0.122: BETTER.
* headway unsafe rate U +0.0178 [+0.0065, +0.0310] (+0.0178 / +0.0178), F 0.0231: NOT-SEPARATED. Its interval excludes 0 on the WORSE side, but |D| is below F; it straddles the floor.

**LATERAL: BETTER**
* heading −0.803° [−0.989, −0.629] (−0.861 / −0.746), F 0.699.
* yaw rate −4.10 °/s [−5.43, −3.03] (−4.72 / −3.49), F 3.37.

**TACTICAL: BETTER**
* ade_tac −0.357 m [−0.608, −0.138] (−0.228 / −0.485), F 0.164.

**STRATEGIC: BETTER** (in full; the pause was lifted on 2026-11-16)
* Position at the 30 / 50 / 80 m marks: −1.83 [−2.31, −1.42] / −3.65 [−4.59, −2.82] / −5.74 [−7.28, −4.29] m, F 0.71 / 1.77 / 2.47.

No family is WORSE at either seed, and the reader lists no seed-disagree cell.

## ATTR (beside, no bar)
* **RK − RL under C\***: all four families NOT-SEPARATED. Examples: speed +0.006 m/s [−0.005, +0.017], heading +0.109° [−0.035, +0.276], ade_tac +0.011 m [−0.021, +0.043], position at 80 m −0.45 m [−0.73, −0.17] with F 1.16. At T1 the kinematic label adds nothing measurable over the speed label.
* **RK − RH**: under C\*, LONGITUDINAL MIXED and the other three NOT-SEPARATED; under R0Hb20, STRATEGIC MIXED and the other three NOT-SEPARATED. At T0, see MECH3 beside.
* **P2 fails**: RK is not NOT-WORSE on every family under ACC / D1 / D2, and the B0 attribution control is worse on LATERAL and TACTICAL.
* **P3 fails**: it reads DEGRADES under R0Hb20 and R0C10XHb20, and the B0 control reads DEGRADES as it must.
* **P4 fails**: RK under C\* against B0 under K reads LONGITUDINAL MIXED, LATERAL WORSE, TACTICAL WORSE, STRATEGIC NOT-SEPARATED.
* **F_T0 at four seeds** (printed, never used here): see the section below.

## Name by the card's precedence
BLOCKED, then HARMS, then SEED-DISAGREE, then the rest in table order.

1. **Not BLOCKED:** every gate passed, the census is above 0.5 at both seeds, and the lever acted.
2. **Not HARMS:** G-T0 does not fail at both seeds, and no family is WORSE at both.
3. **Not SEED-DISAGREE:** no family is BETTER at one seed and WORSE at the other. The only floor-straddling cell in P1 is headway U. The reader's sign-resolved name (`name_straddles_by_sign`) is the same, PARTIAL, so that cell does not decide the name.
4. **PARTIAL:** G-T0 holds at one seed only. This is the name the reader prints independently (`RESULT_ROLLHIST_RK.json`, `next_lever`).

**The reading.** The card's PARTIAL reading, "part of the cost is the kinematic conflict", is **not supported by MECH3**: both points sit within 0.012 m of zero, and RK − RL is not separated on any family at T1. Three label-or-mode levers have now been read on the same banks: the speed label (RL), add mode (RA) and the kinematic label (RK). None moves the T0 cost against its parent, and all three keep the T1 gains against B0. The cost is in neither the label nor replacement.

## Next lever, as the card commits it: K-INPUT
In a swapped own sample, O's prior path `base` is rebuilt from the logged kinematics at the matched row r', and the own pose history is kept. This asks whether the T0 cost comes from O reading a prior path built from off-distribution kinematics.

* The build started at zero spend: facts first, then `--dagger-base own|logged` (default `own` = fde8b716 bit for bit), tests, mutants and census.
* A new card is frozen before any K-INPUT number.
* The GPU run waits on a PI go with its cost, ESTIMATED as RK's measured 0.55 pod-h for two seeds.
* Not used: the T_B 2.0 s lever (the MECH3-YES branch). Its code landed at fde8b716 before this read and stays at the default.

## F_T0 at four seeds (beside, printed, never used here)
This uses the read's own T0 instrument (`rh_t0.argv_t0`, CPU, read tree 43319227) on the four stage-A checkpoints. UE_s2.A and UE_s3.A were pulled from the LoRA pod and match their delivery records (sha256 `6d434525…` and `8f5eb364…`, 86,970,901 B each). Script `ft0_4seed.py`, output `ft0_4seed/FT0_4SEED.json`. MEASURED (ours).

* **Control:** the instrument reproduces the INHERITED FACTS E3 means exactly: UE_s0.A **1.9671** and UE_s1.A **2.0376** m.
* **Mean T0 ADE6 over the 2,399 windows:** UE_s0.A 1.9671, UE_s1.A 2.0376, UE_s2.A 1.9202, UE_s3.A 1.9806 m.
* **|UE_sI.A − UE_sJ.A|, the six pairs:** 0.0135 (s0-s3), 0.0469 (s0-s2), 0.0570 (s1-s3), 0.0604 (s2-s3), **0.0705 (s0-s1, the card's F_T0)**, **0.1174 (s1-s2, the largest)**.
* **Reading:** the stage-A seed-to-seed spread at T0 reaches 0.117 m. That is above the bar KINLAB used (0.0705) and above every G-T0 point on this line (largest: RH − B0 s0 +0.0503).
* KINLAB's G-T0 stays as frozen and its name does not change. A later card may adopt a four-seed F_T0, **before its own data**, as KINLAB section 3 allows.

## Limits
* Two training seeds.
* F_T0 used by G-T0 rests on one seed pair.
* T1-kin only, with no re-perception (T2 is not provisioned).
* T0 is a world-model diagnostic.
* ATTR has no bar.
* The read chain's pairs line had a quoting fault that stopped the first run at the headway reader. Fixed by the MM, with backup `rh_read_rk_chain.sh.bk_quote`; no reader code, cell or estimator changed. The rollouts, T0 evals and MECH3 were kept from the first run.

## Files
* `readrk/MM_VERDICT_KINLAB.md`
* `readrk/MECH3_KINLAB.json`
* `readrk/ft0_4seed.py`, `readrk/ft0_4seed/FT0_4SEED.json`
* `read_rk/` = `out/read_rk/`: RESULT_ROLLHIST_RK.json, GT0_ROLLHIST_RK.json, READ_STRAT_ROLLHIST.json, hw/, q3/
* `read_rk/logs/`: rh_read_rk_chain.log, rh_read_rk_retry.log, rh_read_rk_retry.sh/.cmd, rh_read_rk_chain.sh
* `pod_rk/`, as listed under Arms and gates
