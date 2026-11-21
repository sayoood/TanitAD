# MM reading: ACTRSP Option B Amendment 3 (target arm) deciding read, 2026-11-20: **TGT-HARM at both chassis**

**Evidence and scope.**
- Evidence class: MEASURED (ours). CPU only, zero training in the read.
- Tier: **T0, HWM-ORACLE diagnostic.** The goal that conditions the plan is the encoded true future, so no number here is driving performance.
- Rules: card `AMEND_OPTB3_TARGET.md` (sha256 cb4f943f) section 4, plus the MM ruling `MM_RULING_OPTB3_DECIDE.md` (frozen before any read data). `optb3_decide.py` applies them mechanically.
- Data: 994 SCORE windows / 71 episodes. Arms: FTT15_s0/s1 (g* = 1.5) and FTT125_s0/s1 (g* = 1.25), each against the FT-NULL of its seed. The FT-NULL, FT-RSP and R35 plans are the banked Amendment 2 outputs.
- Result file: `out/DECIDE_OPTB3.json`. Plans are banked on the private HF dataset (`BANK_OPTB3_PLANS.json`, sha256 58fe5da6, hub-equal).

| estimator | question it answers |
|---|---|
| paired episode-cluster bootstrap, B 2,000, seed 0 | another draw of EPISODES |
| F_T,k = max - min of the three FT-NULL point means per chassis | another TRAINING run |
| F_I = max abs(seed 1 - seed 0) over every arm with both inference seeds (16) | another INFERENCE run |
| Floor_k | max(F_T,k, F_I) |

## Deciding cells, band 0-2 s (arm - FT-NULL of its seed; positive = worse)
| chassis | family | quantity | FTT15 - FTNULL [95 % CI] | FTT125 - FTNULL [95 % CI] | Floor | reading |
|---|---|---|---|---|---|---|
| 0 | LONGITUDINAL | acc_mae (m/s^2) | **+0.1356 [+0.0418, +0.2289]** | +0.1534 [+0.0659, +0.2435] | 0.0341 | WORSE |
| 0 | LATERAL | yaw_mae | **-0.0157 [-0.0231, -0.0083]** | -0.0154 [-0.0227, -0.0080] | 0.0067 | BETTER |
| 1 | LONGITUDINAL | acc_mae (m/s^2) | **+0.2485 [+0.1716, +0.3300]** | +0.2284 [+0.1467, +0.3119] | 0.0621 | WORSE |
| 1 | LATERAL | yaw_mae | **-0.0348 [-0.0435, -0.0265]** | -0.0340 [-0.0427, -0.0256] | 0.0120 | BETTER |

- **Label.** TGT-HARM at chassis 0 and chassis 1: FTT15 is WORSE on a deciding quantity at either chassis (card section 4). Not CHASSIS-DISAGREE.
- **FTT125 line (beside, card section 4):** the same shape at both chassis, acc_mae WORSE and yaw_mae BETTER beyond the floor.
- **Attribution FTT15 - FTT125 on acc_mae:** -0.0178 [-0.0320, -0.0046] at chassis 0 and +0.0201 [+0.0080, +0.0332] at chassis 1. The two chassis disagree in sign, and both are below the floor. On yaw_mae neither is separated. ⇒ The raised target (1.5 against 1.25) carries no effect of its own on either quantity.

## Mechanism and target
- **TARGET-READ (beside, no bar).** The term reached its own target on the eval clips through the frozen readout: 1.493 / 1.488 for g* = 1.5, and 1.248 for g* = 1.25. That is within 0.1 at every arm. The response L1 fell to about 4e-4 by step 4,000 in all four runs.
- **MECH-RAISED does not hold at any pair** (`mech_rows`). The IA-3P-K gain of FTT15_s0 is 1.104 [0.997, 1.249] against FTNULL_s0's 0.910 [0.806, 1.039]. That gap of +0.194 is above F_T, but the CIs overlap. The other three pairs sit below their floor, and FTT125_s1 reads -0.095.
- Unlike Amendment 2's FTRSP, the response does not move the wrong way here. **The term does what it was trained to do and the 0-2 s acceleration error still rises**, by about the same amount as under FTRSP (+0.19 / +0.25 there).

## Controls
- C2, C3 and C5 hold for all four arms, and no BAND-MET level was lost.
- Every arm's (episode, time) windows equal the manifest's (16 / 16 identity flags true).
- **STRATEGIC identity control against Amendment 2 (`IDENTITY_FLOORS_VS_AMEND2.json`): PASS.** Row B (FTNULL_s0), HOLD, the logged path and the text rows are equal JSON in both reads. Row A is each read's own lever arm, so it differs as it must.
- **Floors against Amendment 2:** F_T,k is unchanged. F_I rises from 0.01397 to 0.01514 on acc_mae because it now spans 16 arms instead of 12, and yaw_mae's F_I is unchanged. F_T dominates at both chassis, so Floor_k is identical in both reads: 0.0341 / 0.0621 on acc_mae and 0.0067 / 0.0120 on yaw_mae.

## The four families (FTT15 against FT-NULL, all bands)
| family | status |
|---|---|
| LONGITUDINAL | 13 cells worse and 17 none over the bands, none better. The 0-2 s acc_mae is the deciding cell above. |
| LATERAL | 14 cells better, 9 none and 7 worse; yaw_mae 0-2 s is better at both chassis. Cross-track against the lane is NOT COMPUTABLE (n = 0, no lane geometry). |
| TACTICAL | 4 cells better, 8 none and 6 worse. At chassis 0, 0-2 s, the implied-manoeuvre disagreement is +0.1097 [+0.0563, +0.1630] (worse) and the longitudinal one +0.1429 [+0.0795, +0.2072] (worse). Goal and anchor setting is NOT COMPUTABLE (n = 0, the planner exposes no goal or anchor). |
| STRATEGIC | Reported in full on the 6 s band (PI 2026-11-16). At chassis 0, S_FALSE is +0.0246 [+0.0012, +0.0497] higher (more false turns, n 853 / 70 episodes); at chassis 1 it is not separated. The s_adh_deg "better" cells rest on 3 windows in 1 episode and 2 windows in 2 episodes, too few to read. Structural n/a items carry their reasons in the JSON. Never a pass, and no outcome is decided on it. |

## Beside the label (post-hoc, decides nothing)
**yaw_mae 0-2 s is BETTER beyond the floor at both chassis under three different response-term variants**: FTRSP, FTT15 and FTT125. The acceleration error rises in all three. This is a post-hoc pattern across reads, and it needs its own pre-registration before it is quoted as a lever effect.

## Next (Rule Zero)
- Card row TGT-HARM: **the response-gain lever is closed on this chassis family; next is LON-PLAN on the planner side.**
- With this read, the LON-PLAN line has eliminated these levers:
  - the readout (Amendment 6 R6-SAME, Amendment 7 R7-FIT-FAIL);
  - the WM response-training term (Amendment 2 OPTB-HARM, Amendment 3 TGT-HARM).
- The zero-training planner-side lever the card names is the **prior-regularised planner** (the OPTB-MECH-ONLY / TGT-MECH-ONLY row).
- The v8f coordinator owns the line's ordering under the PI's 2026-11-19 delegation, so this reading goes to the coordinator. The coordinator decides whether that planner card is the next arm, or whether the integrated candidate's longitudinal path takes the lever instead.
