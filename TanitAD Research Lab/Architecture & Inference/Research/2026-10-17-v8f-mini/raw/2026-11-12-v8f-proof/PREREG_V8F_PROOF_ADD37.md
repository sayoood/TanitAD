# PREREG_V8F_PROOF_ADD37 : IMAGACT, the effect of the ego's own action taught to the imagination by a kinematic consistency term

Card of the v8F proof programme, MOTION line; the lever that card `PREREG_V8F_PROOF_ADD36.md` commits for `IMAG-DEAF`. Frozen before the lever is built and before any arm of it exists. Tier of the reads H1 and H2: T0 (world-model diagnostics on logged frames, never driving performance). Tier of the family read F: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react). STRATEGIC: PAUSED (PI item 25).

## 1. Why (MEASURED, `imaghead/read/LS/RESULT_IMAG_LS.json` sha256 787aa0c6a03f1bbe826e0af7c8cf2a4c93f3584808e88f4a6cbd15899aa23624; card ADD36 with note N1, controls c1 to c5 PASS)

* The supervised lead head reads the lead gap 2.0 s ahead from imagined states of the base predictor (arm LS, both seeds): added to the present read +0.070 and +0.045, pooled +0.058 [+0.017, +0.108]; beyond the ego-only floor +0.047 and +0.032, pooled +0.040 [+0.0003, +0.085]: `IMAG-HEAD-ADDS` with `IMAG-HEAD-ADDS-LEAD`. Mean absolute error against the gap at t + 2.0 s: 6.14 m from the imagined states, 6.83 m from the present states, 5.23 m from the true future states.
* The imagined gap moves the wrong way when the ego's own acceleration is changed: slope against the kinematic expectation -0.444 and -0.493, pooled -0.468 [-0.530, -0.415] (expected +1; 356 windows per seed); at 1.0 s -0.704 and -0.823. Braking by 2 m/s2 moves the imagined gap by -1.81 and -2.01 m where the kinematics give +4.0 m; braking by 1 m/s2 by -0.94 and -1.01 m (expected +2.0); accelerating by 1 m/s2 by +0.78 and +0.87 m (expected -2.0). The response is monotone in the offset and has the opposite sign. By the rule of card ADD36 the name is `IMAG-DEAF`; the measured fact beside the name is a response of the wrong sign, not an absent one.
* Checked on the data, no model involved: the acceleration channel has the stated sign (the speed change over 2.0 s against the summed acceleration: slope 1.000, 1,948 windows). In the labels, the true change of the gap over 2.0 s against the kinematic term of the LOGGED ego acceleration has slope -1.70 (correlation -0.40, 318 windows): in logged driving the ego brakes while the gap closes.
* Reading, as a hypothesis this card tests. With g the gap, T the horizon and d an offset of the ego acceleration, an intervention gives g_T(do(a + d)) = g_T(do(a)) - 0.5 d T^2 for a lead that keeps its motion: the slope of the response on -0.5 d T^2 is exactly 1. A predictor fitted on logged driving learns the conditional E[g_T given state, a], and where the state does not hold the lead's motion in full, the logged action carries information about it (the driver brakes because the lead brakes). The conditional slope is then 1 plus a confounding term, measured in the labels as -1.70, so the confounding term is about -2.7 and the fitted predictor reproduces its sign. No amount of logged data identifies the interventional slope; the kinematics give it for free. A planner that ranks plans by this imagination would prefer accelerating toward the lead.

## 2. The lever (flag `--w-act-cons W`, default 0.0 = the base tree bit for bit; `--act-cons-max` default 1.5 m/s2)

For each training window and each start frame s of the start set of the rollout loss (frames from which 10 steps fit), with the lead head of the arm:
* one offset d per sample, uniform in [-max, +max], constant over the 10 steps, added to the acceleration channel of the level-1 action in physical units (yaw rate unchanged);
* two rollouts of the stack from the state of frame s (start state without gradient): under the logged actions (no gradient, the reference) and under the offset actions (gradient);
* the lead head (its parameters without gradient in this term) reads the gap from both rollouts at T = 1.0 s (steps 1 to 5) and T = 2.0 s (steps 6 to 10);
* term = mean over valid samples and the two times of the absolute value of [gap(offset) - gap(reference) + 0.5 d T^2], in metres. Valid: a label lead at frame s (present, gap between 0 and 60 m) and an ego speed at frame s of at least max times 2.0 s (no standstill inside the rollout). The total gains W times the term.
* Inference is unchanged and vision only; the label enters the validity mask at training only.
* Gradient: the predictor and its action input receive it; the lead head, the start state and everything before it receive exactly 0.0 from this term.

## 3. Arms

Fine-tunes of 6,000 steps at learning rate 1e-4 (warm-up 100) from LS_s0 and LS_s1, in the tree of card ADD35 plus the flag:
* LFT (control, nothing added) and LFTb (its rerun): trained under card ADD35 (chain `leadcond_a`); read here with the tool of ADD36.
* LA: the argv of LFT plus `--w-act-cons 1.0` (the named arm).
* LA01: `--w-act-cons 0.1` (dose arm, names nothing).
* When RL_s0 and RL_s1 of card ADD36 exist: RLFT (control) and RLA (weight 1.0) from them, same rule (names of section 4 given separately for this base).
* Budget (ESTIMATED from LFT at 21 min plus two extra rollouts per step): about 30 min per arm on one pod slot (L40); one arm with rollouts at a time, behind chain `motion_e`.

## 4. Reads and names (each name needs both bases)

* H2 with the tool and windows of card ADD36 (offsets -2, -1, +1 m/s2; the offset -2 lies outside the range of the training offsets). `ACT-CONS-RESPONDS`: LA reads `IMAG-RESPONDS` (slope at least 0.5 at both bases, two-level interval above 0) and LFT does not. `ACT-CONS-PARTLY`: not that, and the slope of LA minus the slope of LFT is above 0 at both bases with the paired two-level interval excluding 0 and larger than the difference LFTb minus LFT. Else `ACT-CONS-FLAT`. Beside: the mean response per offset in metres; the slope at 1.0 s and on gaps below 20 m; the response of the presence output.
* H1: `ACT-CONS-KEEPS-IMAGINATION` if LA reads `IMAG-HEAD-ADDS`; else `ACT-CONS-LOSES-IMAGINATION`. Beside: H1 of LA minus H1 of LFT (paired), the mean absolute errors of the three reads, and the error of the head on the present states against the label at t (the head on true states must not degrade: reported for LA and LFT).
* F, families at T1-kin: LA against LFT by the rule of ADD12 Q3 with the same-sign rule of ADD13 (`NO-COST`, `COST-<family>`, `GAIN-<family>`); the unsafe rate U and the position error over 6 s reported. The plan head does not use the imagination, so no gain is expected here; the read is a cost check.

## 5. Controls (all must pass; else nothing is named)

* a1: with the flag at its default the patched tree equals the base tree bit for bit on the argv shapes of LFT and of a from-scratch arm (max abs difference 0.0 after the same steps).
* a2: the term equals a value computed by hand on a small batch; samples without a label lead or below the speed bound give exactly 0.0 term and gradient; the gradient on the predictor and on its action input is non-zero; on the lead head, on the start state and on the encoder path it is exactly 0.0.
* a3 (known response, CPU): on a planted predictor that follows the kinematics exactly the term is 0 to within 1e-6; with the sign of the kinematic target flipped, or with the offset not applied to the rollout, the term is far from 0 and the check goes red.
* a4: pod gate per arm on real data: declared equals built, the term finite and logged with its count of valid samples, a deliberately wrong expectation refused; no training without the PASS token. The start checkpoint is applied (strict load, config equal to the base).
* a5 (floor, in the same read): LFTb minus LFT on the slope and on H1 is not named different.
* a6: the controls c1 to c5 of the read tool pass for every arm read (part (c) of c2 is not evaluated, note ADD36_N1).
* a7: the family read keeps its controls q1 to q5.

## 6. What follows (committed before the data)

* `ACT-CONS-RESPONDS` with `ACT-CONS-KEEPS-IMAGINATION` and `NO-COST`: planning by imagination at T1-kin on LA, the card that ADD36 section 6 describes (candidate plans rolled out with their own actions, the distance rule applied to the imagined gap), and the same term in a from-scratch arm.
* `ACT-CONS-PARTLY`: the dose that reads best between LA and LA01 is trained longer and from scratch with the lead head (the response learned together with the states), own card; planning by imagination waits for `IMAG-RESPONDS`.
* `ACT-CONS-FLAT`: the term with the gradient into the start state and the encoder path, and the lead head trained on rolled-out states; own card.
* `ACT-CONS-LOSES-IMAGINATION` or any `COST-<family>`: the arm is not promoted; the term is combined with the rollout loss (base RL) so that the reference rollout stays anchored to the true future; the response outcome is reported as written.

## 7. Limits

Two bases, one fine-tune per base and one rerun of the control. The term teaches the response for a lead that keeps its motion; a lead that reacts to the ego is outside it. Offsets are constant over 2.0 s; other profiles are not read. The read offsets -1 and +1 lie inside the training range; only -2 lies outside it. T0 reads on logged frames; a correct response in imagination is a property of the world model and says nothing yet about driving. One corpus.
