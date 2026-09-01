<title>RESULT - long-horizon BPTT stability (answers ASK-1)</title>

# ASK-1 ANSWERED: no published recipe in our reference class back-propagates a 60-step full chain — and the clip is not the lever

**Package** `E-ARCH-LHB-1` · 2026-09-01 · Architecture & Inference · 0 GPU · literature only
**Answers** `LAB_ASKS.md` **ASK-1** (TrainingFlyWheel) · **also addresses** `LAB_BACKLOG` PROPOSED **P-8**
**Tier stamp** n/a — no eval was run here. Every external number is `PUBLISHED (PRIMARY, banked)`;
every internal number is `MEASURED` with its artifact path.

---

## 1. The findings, first

### F1 — H-LHB-1 resolves to **outcome B: no such recipe exists in our reference class.** `PUBLISHED (PRIMARY, banked x4)`

Every relevant primary avoids the configuration our k=60 arm ran. Four independent mechanisms,
none of which is "raise the clip":

| source (banked) | horizon actually back-propagated | mechanism |
|---|---|---|
| **TD-MPC2** `2310.16828` Table 8 | **Planning Horizon (H) = 3**, "the same hyperparameters across all tasks" | bootstraps past H with a **learned terminal value function** |
| **DreamerV3** `2301.04104` hyperparameter table | **Imagination horizon H = 15**; critic prediction horizon **T = 16** | **AGC(0.3)** + bootstrapped lambda-returns (lambda = 0.95) past the horizon |
| **Looped World Models** `2606.18208` §3.5.4 | **truncated**: "Backpropagation through the loop iterations is truncated at mu_bwd = ceil(mu_rec/2) steps" | progressive **curriculum over K**, plus an explicit growth-budget penalty |
| **InfinityDrive** `2412.01522` §3.1.2 (driving WM) | window grown **16 -> 32 -> 64 -> 128** frames | curriculum + batch size reduced as the window grows |

⭐ The Looped-World-Models paper states our failure mode as a known result, verbatim:
> *"Training directly with a large K is unstable because gradients must back-propagate through
> K × T shared-parameter applications."* (`2606.18208` §3.5.4)

⇒ **Our k=60 arm did not hit a bug. It ran the one configuration the entire reference class
is designed to avoid.** `MM-E19`'s own prereg already reached the mechanism independently —
*"`rollout_transitions` is explicitly not truncated BPTT ... A 60-deep backprop chain is exactly
where exploding gradients live"* (`Project Steering/PREREG_MM_E19_K60_HORIZON.md` §3c) — the
literature confirms it and supplies the three fixes the prereg did not have.

### F2 — H-LHB-2 resolves to **outcome B, and this contradicts the mitigation already chosen.** ⛔ decision-relevant

`PREREG_MM_E19_K60_HORIZON.md` §3c commits to relaunching as `k60clip05p30k` with **`--clip 0.5`**,
"the smallest change that addresses the measured" instability. **No banked primary uses a smaller
scalar clip as its long-horizon stability mechanism.** Worse, the prereg's own text shows why it
cannot work (`MEASURED`, same file §3c):

> *"`--clip` is **1.0** and `clip_grad_norm_` returns the **PRE-clip** norm ... every applied update
> is bounded to norm 1.0. But that is not benign — scaling a 4.29e7 gradient to 1.0 preserves
> its direction, so the step becomes 'move 1.0 along whatever exploded'."*

⇒ **The update magnitude was never the problem; the direction was.** Halving 1.0 -> 0.5 makes the
step "move 0.5 along whatever exploded". It changes the length of a bad direction and nothing else.
⚠️ Note this is a **prediction, not a measurement**: the clip-0.5 arm may still survive by luck, but
if it does, that outcome is uninformative about the mechanism.

What the primaries use instead, in ascending order of intrusiveness:

1. **Per-tensor RELATIVE clipping.** DreamerV3: AGC "clips per-tensor gradients if they exceed
   **30 % of the L2 norm of the weight matrix** they correspond to", default eps = 1e-3 — and,
   quotably, *"AGC decouples the clipping threshold from the loss scales"* (`2301.04104`). A global
   scalar clip does neither.
2. **Latent normalisation.** TD-MPC2: *"SimNorm is essential to the training stability of TD-MPC2"*;
   its own ablation reads **No Norm 46.8 · SimNorm 51.0 · LN + SimNorm 54.2** normalised score
   (multitask-80). TD-MPC's predecessor "is prone to exploding gradients"; the fix was the
   representation, not the optimiser.
3. **Curriculum over the horizon.** Looped WM: `K(step) = min(K_max, 1 + floor(step/Delta))`,
   Delta = steps between increments, *"begins with K = 1 ... allows the latent dynamics to first
   learn accurate single-step transitions before being challenged with longer decode-free
   rollouts"*. InfinityDrive ramps 16->32->64->128 frames. RDR `2608.25017` ramps its rollout weight
   alpha_e **linearly from 0 to 1 between epochs 2 and 5**, opening the rollout branch only after
   two warm-up epochs.

### F3 — the same mechanism answers injected row **I-1**, which we did not expect `PUBLISHED (PRIMARY)`

Both TD-MPC2 and DreamerV3 refuse to lengthen the backprop chain and instead **bootstrap past the
horizon with a learned value**: TD-MPC2 *"addresses this shortcoming ... by bootstrapping return
estimates beyond horizon H with a learned terminal value function"*; DreamerV3 computes
lambda-returns "to consider rewards beyond the prediction horizon T = 16". This is precisely
`LAB_BACKLOG` **I-1** (learned terminal value over composed h=1 rollouts). ⇒ **ASK-1 and I-1 have
the same answer**, and the deployment-side pricing of it is this pass's Deploy package
(`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-01-terminal-value-vs-fan-scoring/RESULT.md`).

### F4 — a fourth mechanism exists and is cheap to state, but we cannot yet recommend it `PUBLISHED (PRIMARY)`

Looped WM constrains its state-transition spectrum by construction (rho(A-bar) < 1 via SSM
discretisation), and reports: *"This constraint holds by construction throughout training; **no
gradient clipping, post-hoc normalisation, or sensitive hyperparameter tuning is required.**"*
It also adds a growth-budget penalty `beta * max(0, sum_k ||h_{k+1} - h_k||^2 - K * C_max)`,
described as controlling *"gradient explosion over long deferred horizons while still permitting
meaningful state changes induced by actions"*. ⚠️ **This requires an SSM-class transition** — it is
not a drop-in for our transformer predictor, and it is the same lever as injected **I-2**. Flagged,
not recommended, because adopting it changes the architecture under test.

---

## 2. What this changes for TanitAD — 3 recommendations

1. ⛔ **Do not spend the `k60clip05p30k` arm as specified.** It tests a lever no primary uses,
   against a mechanism the prereg's own text says is directional rather than magnitudinal (F2).
   **Replace `--clip 0.5` with a horizon CURRICULUM starting at the already-stable k=8** —
   `K(step) = min(60, 8 + floor(step/Delta))` in the Looped-WM form — which is the one mitigation
   with three independent primaries behind it (F1, F2.3). Cost is identical: one arm.
2. **If a second knob is wanted in the same arm, make it AGC, not a smaller scalar clip** (F2.1) —
   per-tensor relative clipping is what the closest-relative recipe at H=15 actually ships, and it
   is a small optimiser-side change. ⚠️ Registering both curriculum AND AGC in one arm makes the
   result non-attributable; if both go in, say so and do not claim which one worked.
3. **Re-frame the 6 s horizon question itself.** No banked primary reaches a long horizon by
   deepening BPTT; two reach it by **bootstrapping a value past a short horizon** (F3). ⇒ The
   horizon target and the BPTT depth are separable, and I-1 is the cheaper route to the same
   capability.

## 3. Named empty searches (do not re-run without a new trigger)

Recorded in `raw/search_log.md`. Headline: **no primary was found that back-propagates >= 60
untruncated steps through a shared-parameter latent transition.** Absence probed at four
independent angles (reference-class latent WMs · driving long-horizon WMs · rollout-loss /
exposure-bias literature · RNN/TBPTT stability literature). Per the operating standard this clears
the two-probe bar with margin — but it remains an absence claim, and a single counterexample
refutes it (falsifier stated in `SPEC.md`).

## 4. Evidence-class ledger

| claim | class | source |
|---|---|---|
| TD-MPC2 H=3; terminal-value bootstrap; SimNorm essential; 46.8/51.0/54.2 | PUBLISHED (PRIMARY, banked) | `Library/papers/2310.16828_*.pdf` |
| DreamerV3 H=15, T=16, AGC(0.3) at 30 % of weight L2, LaProp, lambda=0.95 | PUBLISHED (PRIMARY, banked) | `Library/papers/2301.04104_*.pdf` |
| K(step)=min(Kmax,1+floor(step/Delta)); mu_bwd truncation; rho(A-bar)<1 | PUBLISHED (PRIMARY, banked) | `Library/papers/2606.18208_*.pdf` |
| 16->32->64->128 window curriculum | PUBLISHED (PRIMARY, banked) | `Library/papers/2412.01522_*.pdf` |
| alpha_e ramp epochs 2->5, rollout after 2 warm-up epochs | PUBLISHED (PRIMARY, banked) | `Library/papers/2608.25017_*.pdf` |
| gnorm median 5.71 -> 4.29e7 at step 7600; clip=1.0 is PRE-clip; o5_loss +91 % | MEASURED (ours) | `Project Steering/PREREG_MM_E19_K60_HORIZON.md` §3c |
| "clip 0.5 will not fix a direction problem" | **HYPOTHESIS** (reasoning from the two rows above) | not measured; stated as falsifiable |
