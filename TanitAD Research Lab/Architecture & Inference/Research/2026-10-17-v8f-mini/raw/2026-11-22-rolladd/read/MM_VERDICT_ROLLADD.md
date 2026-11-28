# MM verdict: ROLLADD read (RA_s0, RA_s1), 2026-11-22

**Outcome: ROLLADD-PARTIAL with MECH2-NO.**

* **Card:** `PREREG_V8F_ROLLADD.md`, sha256 `30b61bdd…`, frozen 2026-11-22T14:22:32Z before any RA number.
* **The answer:** keeping the logged samples whole and adding the own rolled states does **not** remove the T0 cost of rolled history. Replacement is not the cause; the card's named alternative, interference through the shared trunk, is the reading left standing.
* **Next lever** (the card's PARTIAL row, MECH2-NO branch, "as HARMS with MECH2-NO"): the kinematic label, R2 of the RECLAB card.

Evidence class MEASURED (ours). The RA arms trained on the LoRA pod; the read ran on the dev box with zero training.

## Arms and gates
| item | value |
|---|---|
| arms | RA_s0 / RA_s1 = RH's argv + exactly `--dagger-mode add`, code df250a00, RH banks (s0 e5e44c01, s1 bde6566e), readout causal_g1x, p 0.1, 3,000 steps of stage B |
| GPU lock | 18:41:01Z to 19:12:21Z = **0.52 pod-h** (PI go "Yes, run at 0.67") |
| gates | gate record MINTED and VERIFY OK (both); zerograd PASS (untrained numel 0); post 252/252 tensors moved; summary check at ckpt step 3000 |
| the lever acted | appended 15,411 of 191,664 logged rows (s0) and 15,380 of 191,687 (s1); rows out 207,075 / 207,067; selfcheck `dagger_probe` mode add, rows_appended 46 / 51, `live_added` pass, `control_prior_grad_sum` 0.0 |
| records | `pod_ra/RA_s{0,1}/` (argv, selfcheck, summary, post, summary_check, train_log, kclip, zerograd, mint/verify logs), `pod_ra/rh_ra_chain.log` |

## Tiers and estimator
* P1, P4 and ATTR are read at **T1-kin/hb0.2**, the executor of record (kinematic rollout on the model's own actions, heading bound 0.2 1/m).
* G-T0 and MECH2 are read at **T0**. T0 is a world-model diagnostic, never driving performance; the card makes it binding.
* Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 (`taniteval/ci.py`), over 141 episodes and 2,399 T0 windows. It answers "another draw of EPISODES". The two seeds answer "another TRAINING run". Inference is deterministic.

## G-T0 (decides): RA − B0 at T0 ADE6
The bar is an upper bound ≤ +0.0705 m per seed. F_T0 is INHERITED from one seed pair, so it is a weak floor.

| seed | point | interval | verdict |
|---|---|---|---|
| s0 | +0.0432 m | [+0.0059, +0.0848] | **FAILS** |
| s1 | +0.0205 m | [−0.0134, +0.0559] | **HOLDS** |

Beside, RH − B0: s0 +0.0503 [+0.0060, +0.0929], s1 +0.0217 [−0.0122, +0.0596]. This is the same split as RH and RL.

## MECH2 (decides the next lever): RA − RH at T0 ADE6
| seed | point | interval |
|---|---|---|
| s0 | −0.0071 m | [−0.0352, +0.0226] |
| s1 | −0.0011 m | [−0.0207, +0.0157] |

Neither interval sits below 0, so this is **MECH2-NO**. Instrument: `mech_ra.py`, output `MECH2_ROLLADD.json`.

⚠ `mech_ra.py`'s console line prints "RL" where it means "RA": a label left over from mech_rl.py. The JSON names the arms correctly (A = RA_s*, B = RH_s* / B0_s*), and its means match `GT0_ROLLHIST_RA.json`.

## P1: RA − B0 under C* (R0C10XHb20), four families
Each cell gives the two-seed D, its interval, the per-seed values, and the floor F (two-seed F_c).

**LONGITUDINAL: MIXED**
* speed −0.257 m/s [−0.376, −0.153] (−0.204 / −0.311), F 0.126: BETTER.
* along −0.219 m [−0.385, −0.073] (−0.137 / −0.301), F 0.122: BETTER.
* headway unsafe rate U +0.0214 [+0.0078, +0.0373], F 0.0231: NOT-SEPARATED. It straddles the floor, with the WORSE sign.

**LATERAL: BETTER**
* heading −0.833° [−0.998, −0.677], F 0.699.
* yaw rate −5.12 °/s [−6.49, −3.87], F 3.06.

**TACTICAL: BETTER**
* ade_tac −0.369 m [−0.612, −0.155] (−0.249 / −0.489), F 0.164.

**STRATEGIC: BETTER**
* Position at the 30 / 50 / 80 m marks: −1.82 [−2.30, −1.40] / −3.66 [−4.65, −2.81] / −5.83 [−7.47, −4.31] m, F 0.71 / 1.77 / 2.48.

No family is WORSE at either seed, and the reader lists no seed-disagree cell.

## ATTR: RA − RH at T1 (beside, no bar)
* **Under C\***, LONGITUDINAL, LATERAL and TACTICAL are NOT-SEPARATED; for example ade_tac −0.006 [−0.061, +0.044] and yaw rate +0.31 [−0.17, +0.89].
* **STRATEGIC reads BETTER** under C\*: −0.21 [−0.32, −0.11] / −0.43 [−0.67, −0.22] / −0.81 [−1.27, −0.42] m at 30 / 50 / 80 m, F 0.15 / 0.31 / 0.57.
  * s0 alone is smaller (−0.08 / −0.20 / −0.46).
  * The second difference named before the data, SIGReg × n (the add-mode batch is about 1.08× B; `hjepa.py:131-134`, `:185-187`), confounds this cell.
  * It is reported as an observation, not a lever effect. It has no replicate floor of its own beyond the two seeds (H-ESTIM-SEED-1).
* **Under R0Hb20**, all four families are NOT-SEPARATED.

## P2, P3, P4
* **P2 fails.** The treated arm is not NOT-WORSE on every family under ACC / D1 / D2, and the B0 attribution control is worse on LATERAL and TACTICAL.
* **P3 fails.** It reads DEGRADES under R0Hb20 and R0C10XHb20, and the B0 control reads DEGRADES as it must.
* **P4 fails.** RA under C\* against B0 under K reads LONGITUDINAL MIXED, LATERAL WORSE, TACTICAL WORSE, STRATEGIC NOT-SEPARATED.
* The reader's line (`ra_answer`): "add mode keeps NEITHER P2 nor P3 and G-T0 FAILS at one seed". RH on the same bank in swap mode reads the same.

## Name by the card's precedence
BLOCKED, then HARMS, then SEED-DISAGREE, then the rest in table order.

1. **Not BLOCKED:** every gate passed and the lever acted.
2. **Not HARMS:** G-T0 does not fail at both seeds, and no family is WORSE at both.
3. **Not SEED-DISAGREE:** no family is BETTER at one seed and WORSE at the other. The only floor-straddling cell is headway U, which is not a decisive cell of a BETTER family.
4. **PARTIAL:** G-T0 holds at one seed only. This is the name the reader prints independently (`RESULT_ROLLHIST_RA.json`, `next_lever`).

**The reading.** The card's PARTIAL reading, "part of the cost is replacement", is **not supported by MECH2**. Both points sit within 0.01 m of zero. The card routes PARTIAL with MECH2-NO to the HARMS / MECH2-NO lever.

**What survives.** The own-action gains of rolled history stand in add mode as in swap mode, on all four families against B0.

## Next lever, as the card commits it: R2, the kinematic label
In the swapped (here: appended) own-state sample, the yaw-rate and acceleration targets blend from the own history into the human's, under the speed label's feasibility envelope.

* Build and card now, at zero spend.
* The GPU run waits on a PI go with its cost. The LoRA pod is held by UE_s2 / UE_s3 (about 7-8 pod-h from 22:00Z), then by the ADD45 union probe.
* Not triggered: SEED-DISAGREE's third seed pair, although UE_s2.A / UE_s3.A are now training for the fork's ADD121.

## Limits
* Two training seeds.
* F_T0 rests on one seed pair.
* T1-kin only, with no re-perception (T2 is not provisioned).
* T0 is a world-model diagnostic.
* ATTR has no bar.
* The SIGReg × n confound applies to every RA − RH cell.

## Files
* `readra/MECH2_ROLLADD.json`
* `readra/MM_VERDICT_ROLLADD.md`
* `read_ra/` = `out/read_ra/`: RESULT_ROLLHIST_RA.json, GT0_ROLLHIST_RA.json, READ_STRAT_ROLLHIST.json, hw/, q3/
* `read_ra/logs/`: rh_read_ra_chain.log, rh_read_ra_retry.log
* `pod_ra/`, as listed under Arms and gates
