# PREREG_V8F_ACTRSP, Amendment 3 (Option B target arm): raise the response gain instead of pulling it to 1.0. FROZEN by the MM before any code of this arm exists

**PI go:** 2026-11-18. To the question "Box 2's next Option B arm?", the PI answered **"Target arm 1.5"**. It replaces the committed OPTB-HARM next step (lambda 0.3, four runs) with four target runs, about 5.5 box-2 h.

**Why** (MEASURED, landed at `754e1d4d`, `optb/diag/`):
- The Option B term `_response_loss_fixed_k` minimises mean (G - 1)^2 through the frozen R35 readout. R35 already reads G of about 1.25 there (1.255 / 1.246; Amendment 1 section 1.5), so the term **shrank** the response by about 20 %. That is OPTB-HARM, read as NOT-MECH-MOVED.
- Lambda 0.3 is a smaller dose of the same shrink.
- Across the banked arms, `acc_mae` falls as the frozen-probe gain rises. This is HYPOTHESIS-grade: n = 3-5 per chassis, confounded with term on/off.

Everything not changed here is Amendment 2 (`AMEND_OPTB.md`) as written: the recipe, readout, pairs, floors, controls and read path.

## 1 The lever
- **New flag:** `--wm-rsp-target <float>`, default **1.0**, which is exactly the Amendment 2 term.
- **Loss:** `mean (G - g*)^2`. Nothing else in the term changes: contrasts, K = 10, the readout, s_a, and the gradient path (`predictors.0`, `act_embed`).
- **Naming:** every RSP arm of this amendment names `--wm-rsp-target` in its argv. NULL arms name none of the RSP flags (as Amendment 2).
- **Gate:** the flag needs a G-DVB entry: declared equals built, read back in the selfcheck. A trainer without it is refused.

## 2 Arms (box 2, Amendment 2's recipe: init `R35_s{k}` WM, `wm.proj` frozen, 4,000 steps, lambda 1.0, `fixed_k` readout of chassis k)
| priority | tag | chassis | seed | `--wm-rsp-target` | pairs with (banked) |
|---|---|---|---|---|---|
| 1 | FTT15_s0 | 0 | 0 | 1.5 | FTNULL_s0 |
| 2 | FTT15_s1 | 1 | 1 | 1.5 | FTNULL_s1 |
| 3 | FTT125_s0 | 0 | 0 | 1.25 | FTNULL_s0 |
| 4 | FTT125_s1 | 1 | 1 | 1.25 | FTNULL_s1 |

- **1.5** raises G by about 20 % above R35's level, the mirror of the shrink just measured.
- **1.25** is the near-neutral control. It holds about R35's own level through this readout. It separates the effect of raising the gain from the effect of having the term at all.
- The seeds equal the paired FT-NULL's seed. The pairing control C4 must hold: equal start-state sha256 and equal (ci, t) lists over the first 50 batches.
- **One lever seed per chassis**, by the PI-approved four-run budget. The two chassis are the replication, and the per-chassis statistic is the single paired delta (not Amendment 2's two-seed mean).

## 3 Read (as Amendment 2 section 3, with the arms above)

**Mechanism.**
- IA-3P-K: the calibrated L1 2.0 s longitudinal gain, cross-fitted on the 141 eval clips.
- **MECH-RAISED:** at both chassis, FTT15's gain lies above its FT-NULL pair's, CIs disjoint, and the gap is above F_T,k on that metric.
- TARGET-READ, beside with no bar: the arm's frozen-probe cycle gain on the eval clips, expected near g*, and its `rsp/L1` trajectory.
- FTT125 is reported on the same rows.

**Deciding.**
- HWM-PLAN phase-1 rows, band 0-2 s: `acc_mae` (LONGITUDINAL) and `yaw_mae` (LATERAL).
- Each arm is paired against the FT-NULL of its seed. Paired episode-cluster bootstrap, B 2,000, seed 0, over 994 windows / 71 episodes. It answers another draw of EPISODES only.
- BETTER at a chassis: CI wholly below 0 and |delta| > Floor_k. WORSE: CI wholly above 0 and delta > Floor_k.
- Floor_k = max(F_T,k, F_I):
  - F_T,k = max - min of the three FT-NULL point means of chassis k (unchanged, the banked arms);
  - F_I = max |seed 1 - seed 0| over every arm present, recomputed with the new arms.
- **Attribution row:** FTT15 - FTT125, paired, same seed and start, at each chassis. It is reported with its CI and enters the outcome table below.

**Families, per band, never pooled:**
- LONGITUDINAL, LATERAL and TACTICAL as Amendment 2. TACTICAL goal / anchor setting is NOT COMPUTABLE, with its reason.
- STRATEGIC is REPORTED IN FULL on the 6 s band, never a pass.

## 4 Committed outcomes (written before any arm exists; read per chassis, then combined)
| outcome | condition | next step |
|---|---|---|
| **TGT-GAIN** | At BOTH chassis, FTT15 is BETTER on at least one deciding quantity and not WORSE on the other. On that quantity, FTT15 - FTT125 has its CI wholly below 0 at both chassis. MECH-RAISED holds. | g* = 1.75 at both chassis plus FTT15 at seeds 2 / 3 (four runs, PI go); this WM goes to the HWM-PLAN read |
| **TGT-GAIN-UNATTRIBUTED** | FTT15 is BETTER as in TGT-GAIN, but FTT15 - FTT125 is not below 0 at a chassis | the term's presence, not the raised target, may carry it. Next: a target sweep 1.25 / 1.5 / 1.75 at a second seed (PI go) |
| **TGT-MECH-ONLY** | MECH-RAISED; no deciding quantity BETTER | the response is raised and the plan did not improve: LON-PLAN prior-regularised planner (zero training, as OPTB-MECH-ONLY) |
| **TGT-NULL** | not MECH-RAISED; nothing BETTER; controls hold | (i) The term reached g* on its own readout (TARGET-READ within 0.1 of g*) but IA-3P-K did not move: the lever is the readout mismatch (about 1.27x between the two readouts, `optb/diag/`); next, align the term's readout with the eval readout. (ii) Otherwise: lambda 3.0 at g* = 1.5. |
| **TGT-HARM** | FTT15 WORSE on a deciding quantity at either chassis, or BAND-MET lost at a level | the response-gain lever is closed on this chassis family; next: LON-PLAN (planner side) |
| **CHASSIS-DISAGREE** | a label at one chassis only | reported as such; the less favourable label's next step applies |

- An arm that misses its bar FAILED and is reported so. Then its next step is built (Rule Zero).
- Bars, arms, floors and outcomes do not change after any number is seen.
- FTT125's own reading against FT-NULL is reported beside every label. If FTT125 is BETTER or WORSE beyond the floor, the label carries that line.

## 5 Controls (Amendment 2 C1-C9' unchanged; added)
- **C10 (flag parity):**
  - `--wm-rsp-target` absent equals `--wm-rsp-target 1.0`, bit for bit: state dict, loss terms, outputs and RNG.
  - The absent case also equals the 86da79cd trainer.
- **C11 (analytic identity, float64; the C9' world `z' = z + B* a`, W = I, `s = K w.B*[:,0]` so G = 1):**
  - with g* = 1.0, loss and gradient are 0;
  - with g* = 1.5, loss is exactly 0.25, and the gradient on B* points to raise G;
  - with g* = 1.25, loss is exactly 0.0625.
- **Mutation arms (each must turn a named test RED):** hard-code 1.0 in the loss; drop the flag from `make_config`; flip the default; drop the G-DVB clause.
- **Post gate:** C5 (Adam on 141 of 145 `wm.*` tensors, `wm.proj` bit-equal) per arm, as Amendment 2.

## 6 Cost
| item | amount | class |
|---|---|---|
| training | 4 runs x about 1.36 box-2 h = about 5.5 box-2 h | MEASURED rate on the Option B chain (FTRSP_s3 1.36 h start to done); ESTIMATED total |
| read | 4 checkpoints x about 2.2 h dev-box CPU = about 9 h | ESTIMATED from Amendment 2 section 5 |

## 7 Code base
- The branch child of `86da79cd` (agent/v8f-optb-20261117, Option B's code of record). The new flag and its gate entry are its only change.
- It is not landed on agent/arch-inf-20260803 while the refcv9 closure freeze holds.
