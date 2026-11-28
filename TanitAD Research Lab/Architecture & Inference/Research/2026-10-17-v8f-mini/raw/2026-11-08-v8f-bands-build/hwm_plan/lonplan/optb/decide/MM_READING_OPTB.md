# MM reading: ACTRSP Option B deciding read (2026-11-18): **OPTB-HARM at both chassis**

**Evidence and scope.**
- Evidence class: MEASURED (ours). CPU only; zero training.
- Tier: **T0, HWM-ORACLE diagnostic**: the goal is the encoded true future. This is never driving performance.
- Rules: `optb_decide.py` applies AMEND_OPTB section 3 and the ACTRSP card's sections 4.2-4.4 mechanically. They were committed before any arm was read.
- Data: 994 SCORE windows / 71 episodes, 10 FT arms plus the R35 baselines, inference seeds 0 and 1.
- Result file: `DECIDE_OPTB.json`.

**Estimators, and the question each answers.**
| estimator | question |
|---|---|
| paired episode-cluster bootstrap (B 2,000, seed 0) | another draw of EPISODES |
| F_T,k = max - min of the three FT-NULL point means per chassis | another TRAINING run |
| F_I = max abs(seed 1 - seed 0) over 12 arms | another INFERENCE run |
| Floor_k | max(F_T,k, F_I) |

## Deciding cells, band 0-2 s (FTRSP - FTNULL; the mean of the two lever-seed deltas is one paired statistic; positive = worse)
| chassis | family | quantity | Delta-bar [95 % CI] | the two pair deltas | Floor | reading |
|---|---|---|---|---|---|---|
| 0 | LONGITUDINAL | acc_mae (m/s^2) | **+0.1928 [+0.1087, +0.2790]** | +0.1746, +0.2110 | 0.0341 | WORSE |
| 0 | LATERAL | yaw_mae | **-0.0164 [-0.0236, -0.0096]** | -0.0155, -0.0174 | 0.0067 | BETTER |
| 1 | LONGITUDINAL | acc_mae (m/s^2) | **+0.2545 [+0.1802, +0.3301]** | +0.2136, +0.2954 | 0.0621 | WORSE |
| 1 | LATERAL | yaw_mae | **-0.0293 [-0.0365, -0.0221]** | -0.0331, -0.0254 | 0.0120 | BETTER |

- **Label.** It is OPTB-HARM at chassis 0 and at chassis 1, so this is not SEEDS-DISAGREE. The card makes a WORSE deciding quantity at either chassis HARM, whatever the other quantity reads.
- **Controls hold for all 10 arms:**
  - C2, C3 and C5 hold;
  - no band level was lost at either chassis;
  - the (episode, time) windows of every arm and both baselines equal the manifest's windows file (994 / 994).

## Mechanism: NOT-MECH-MOVED (`mech.json`)
- The response term was meant to raise the world model's response gain to the action. Instead it **lowered** it on every pair: FTRSP_s0 0.749 against FTNULL_s0 0.910 (gap -0.161), and the other pairs read alike.
- So the term moved the mechanism in the wrong direction. The lateral gain above comes **without** the mechanism the term was built for.

## The four families
| family | status |
|---|---|
| LONGITUDINAL | `acc_mae` WORSE beyond the floor at both chassis |
| LATERAL | `yaw_mae` BETTER beyond the floor at both chassis. Cross-track against the lane is NOT COMPUTABLE (n = 0: no lane geometry; the offset to the logged trajectory end is reported in `rows`). |
| TACTICAL | Manoeuvre agreement (kappa) of the implied manoeuvre is reported in `tactical_kappa_first_pair`. Goal and anchor setting is NOT COMPUTABLE (n = 0: the planner exposes no goal or anchor selection). |
| STRATEGIC | Reported in full on the 6 s band in `strategic` (PI 2026-11-16: the pause ended). Its structural n/a items carry reasons. It is never a pass, and no outcome is decided on it. |

## Deviation, before the read
The read chain's manifest step looked for plan files under `readchain/out/plans/`, while the chain wrote them to `optb/plans/<tag>/`. Its manifest therefore listed no arms. `fix_manifest_plans.py` rebuilt it with the chain's own builder and the real plans folder, under the same rule (a DONE file and both seed files). No number was typed. The window identity was checked for every arm.

## Next (Rule Zero)
1. **Committed next arm (card):** the term stays off for planning; **lambda 0.3, four runs**, with the same pairs and floors. This changes the dose, so it needs the PI's go. Cost: about 1.36 box-2 h per run (MEASURED on this chain), so about 5.5 box-2 h for four runs (ESTIMATED).
2. **Zero-cost diagnostic first (CPU, no training):** why does the term lower the eval response gain? Check:
   - the sign and target of `_response_loss_fixed_k` against the eval gain definition;
   - the per-run training curve of the response term.

   A wiring or target defect would make lambda 0.3 a smaller dose of the same wrong push. So the diagnostic decides whether item 1 is the right next arm.
