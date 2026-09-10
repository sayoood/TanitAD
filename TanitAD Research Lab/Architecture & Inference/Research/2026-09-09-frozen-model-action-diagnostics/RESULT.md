<title>Action-identifiability diagnostics that survive a frozen trunk (2026-09-09)</title>

# An action-identifiability instrument our frozen trunk does NOT block

`TanitAD Research Lab - Architecture & Inference - daily pass 2026-09-09 (LAB-RUN-010).`
`Serves gate problems P-2, P-3, P-4/L3; backlog GS-1, GS-2, GS-8, GS-9, MM-1; debt D-9.`

---

## Findings

### F1 - Three passes found the frozen trunk blocking this family. It does not block this one.

| pass | method | why it was blocked |
|---|---|---|
| 2026-09-02 | Delta-JEPA **LDAD** | encoder-shaping: `Dz` is built from two ENCODER outputs, so on a frozen trunk it has nothing to shape (GS-2 REVISED) |
| 2026-09-05 | ATM **AITS** | encoder-side; recorded as *"SECOND consecutive action-identifiability objective blocked by our frozen trunk"* |
| **2026-09-09** | **ACPC / IR / SR** (`2608.12939`) | ⭐ **NOT BLOCKED - it is a DIAGNOSTIC on an already-trained, frozen model, not a training objective** |

`MEASURED from the primary, full text.` Verbatim: *"Let F-theta denote the **frozen** action-conditioned
predictor"*; the encoder is frozen throughout the diagnostic computation; and the paper explicitly
contrasts ACPC with MWM, which *"enforces action-conditioned rollout consistency **during training**"*.

⭐ **This changes what is reachable.** Two consecutive passes concluded that our frozen-trunk decision
costs us the action-identifiability literature. **That conclusion was over-general: it applies to the
OBJECTIVES in that literature and not to its MEASUREMENTS.** `PUBLISHED`.

### F2 - The criterion is the formal version of what our actdiv probe gropes at

Verbatim: *"two observations should be treated as the same state only when their **action-conditioned
consequences agree**"* - bisimulation. Our hold-action control asks a binary version of this question.
ACPC gives it a metric, a quantile summary, and a proof.

| quantity | definition (ASCII transcription of the paper's formula) | reads |
|---|---|---|
| **ACPC** | `ACPC_H(h, h~, a) = || G_a(E(h)) - G_a(E(h~)) ||_2`, a weighted H-step rollout distance | divergence of a clean and a perturbed history **under the same action sequence** |
| **IR** | `IR_q(theta) = Q_q({R_i})` over anchors | clean-vs-perturbed rollout spread |
| **SR** | `SR_q,delta(theta) = mean 1[ D_diff > IR_q(theta) + delta ]` | whether **different states stay distinguishable after rollout** |

They **prove** ACPC divergence bounds the perturbation-induced change in multi-step prediction error
**and planner cost**. `PUBLISHED`.

### F3 - SR ships with a published collapse signature, which our instruments lack

Verbatim: *"we train LeWM on TwoRoom with four SIGReg weights... the representation collapses... its
**SR falls to 0.066**, compared with **0.967-0.984** for nonzero SIGReg."* `PUBLISHED`.

⭐ **We have no comparable operating point for any of our separation statistics.** H-RANK-16 is open
precisely because *"the gate compares arms to a number no current instrument reproduces (20.52 vs 40.77
vs 8.56 on three rigs)"*. **SR is defined as a RATE in [0,1] against a self-calibrated threshold
(`IR_q + delta`), so it is rig-relative by construction** - which is exactly the property the 8.56
participation floor lacks.

### F4 - SR reads the denominator that MM-1 says our ratio is hiding

MM-1 MEASURED that `o5_k` 8 to 60 moved the **action** spread 0.83x but the **scene** spread 1.71x, so
the halved h=1 ratio was *"mostly a denominator effect"*, and it asks for a criterion reading numerator
and denominator separately.

⭐ **SR is a denominator instrument.** It asks whether genuinely different states remain separated after
rollout - i.e. whether scene spread is *informative separation* or *undifferentiated drift*. A rollout
whose scene spread grows while SR falls is diffusing, not discriminating. **That is the distinction
MM-1 needs and could not name.** `HYPOTHESIS` that it settles MM-1; `PUBLISHED` that SR measures this.

### F5 - It is a THIRD critic of LeWM, and debt D-9 hardens rather than weakens

Delta-JEPA (2026-09-02), ATM (2026-09-05), and now ACPC all characterise LeWM; **we have still never
read `2603.19312`.** Three independent critics agreeing is not a substitute for the primary - it is
three restatements through channels we did not choose. `MEASURED` from our own register.

### F6 - What the port costs: they ship no controls

Verbatim from the retrieval: *"No random baselines or chance-level controls provided."* Under
`CLAUDE.md`'s probe rule - a constant-only control that must read the no-information value, plus a
raw-input floor, plus `n` and `d` printed - **their panel is incomplete for our purposes.** Porting IR/SR
without adding controls would import the exact shape that produced four false results in one afternoon
on our ridge probe.

---

## Five-dimension analysis - `2608.12939` (the deep-read item)

| dimension | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ direct lever. Touches gate problems **P-2** (predictor does not use its actions), **P-4/L3** (does the predictor add anything over `z_t`), **P-3** (the o11 probe), and backlog **GS-8/GS-9/MM-1**. It is an instrument for the programme's stated actual problem. |
| **CONSEQUENCE** | If IR/SR port, the v7 ladder gains a **rig-relative separation statistic with a published healthy band and a published collapse value** - which H-RANK-16 has been blocked on since 2026-08-29. It also **partially retracts our own conclusion** that the frozen trunk locks us out of this literature. |
| **COMBINATION** | Composes with **GS-8** (anchored actdiv, monotonicity) - ACPC is the same measurement generalised from a binary hold-action control to a metric under perturbation. Composes with **GS-9** (transition-level rung) - SR is a transition-level statistic. ⛔ **Contrasts sharply with today's B5 item (RLIR):** ACPC compares two ROLLOUTS under identical actions and never reads the real `z_{t+1}` endpoint, so it **does not carry the GS-1 endpoint-concatenation leak**; RLIR's inverse-dynamics reward does. **Two instruments, same day, opposite leak exposure.** |
| **CHANCES / RISKS** | **Upside:** a 0-GPU diagnostic on banked latents and checkpoints, with a proved link to planner cost, that survives our frozen trunk. **Risks:** (a) their perturbations are Gaussian noise, blur and resize on *visual control tasks* - our corpus is driving video and the natural perturbation set differs; (b) **no controls shipped**; (c) their H=8 is not our K in {8, 60, 300}, so IR/SR may be horizon-sensitive in a way they never tested; (d) it is a diagnostic, so it **explains nothing and fixes nothing on its own** - it ranks arms. |
| **EXPERIMENT** | **Pre-registerable, 0 GPU, on banked v7 latents and checkpoints.** Compute SR at `q=0.9`, H=8, over 100 anchors x 5 perturbation draws, for the three v7 arms plus **two mandatory controls: a constant-predictor arm (must read SR at the no-information value) and a shuffled-action arm (must read chance)**. Report `n` per cell. **Committed in advance:** if SR separates the three arms in an order that `o5_loss` does NOT reproduce, SR is adopted as a ladder rung and H-RANK-16's participation-floor blocker is routed around. If all three arms and the constant control read within noise of each other, **SR is uninformative on our rig and we say so and stop** - no arm is spent. |

---

## What this changes for TanitAD - at most three recommendations

1. ⭐⭐⭐ **Run the SR/IR probe on banked v7 arms before the v7f freeze** (the experiment above, 0 GPU).
   It is the cheapest instrument yet found that speaks to P-4/L3, and it sits underneath the freeze.
2. ⭐⭐ **Amend the standing conclusion that the frozen trunk blocks the action-identifiability
   literature.** The correct statement is *"it blocks the OBJECTIVES in that literature, not the
   DIAGNOSTICS"* - record it in `LEDGER_A2_jepa.md` and against GS-2.
3. ⛔ **Read LeWM `2603.19312`.** Three critics deep is past the point where the register's own failure
   class applies. It is banked and 0 GPU.

## Limits of this package

`Literature only - no measurement was taken on our own rig this pass.` The IR/SR numbers are the
paper's, on their tasks; **none of them is a TanitAD number and none may enter our registry.** The
paper's own numbers carry no chance baseline. The primary is **cited but NOT banked** - the G: mount
refused content reads throughout (debt D-12); banking is owed at the first healthy window.
