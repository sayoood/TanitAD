# Action conditioning — hypotheses, and what the literature already knows

**Written** 2026-08-25 · **Author** Master Mind · **Tier** every number is
T0-DIAGNOSTIC · **Status** hypothesis register, not a decision

---

## 1. The failure, stated precisely enough to attack

Nine objective terms have failed (O1, O2, O3, O7, O8, O9, O10, O11, PSG). What we
know about *why*, all held-out and lead-matched:

| measurement | value |
|---|---|
| Δz predictable from `z_t` alone (**drift**) | **64 %** (t 65.6); 25 of 2048 directions carry 90 % |
| drift fraction, **5 scratch arms** | **0.6138–0.6416** — a 4 % band across unrelated recipes |
| drift fraction, **3 distilled arms** | 0.1753 / 0.1952 / 0.3650 — no overlap |
| action recoverable from the transition | **0 of 8 arms** (t > 2), at k = 2…30 (0.2–3.0 s) |
| drift-removed residual carries the action | **0 of 8 arms** |
| `nrmse` change when every action is replaced by noise | **≤ 0.0919 %**, 0 of 32 arms above 0.1 % |
| **Q2X** — true action vs a hard brake and a hard accelerate (chance 0.333) | **0.068–0.339**, 7 of 8 arms **BELOW chance** |
| predicted-delta scale α\* (E-DEC-47) | **1.10–1.51** — under-shoot real, but rescaling gains ≈ 0 |

**The one-line diagnosis: the transition is the latent's own drift; the action
leaves no recoverable trace in it; and forcing the predictor to use the action
produces separation without accuracy.**

⚠️ **Q2X below chance is the sharpest single fact.** An *extreme* action produces a
prediction closer to the true future than the arm's own true action does. E-DEC-47
shows this is **not** simply an amplitude error — α\* > 1 confirms under-shoot, but
correcting it gains nothing, so **the predicted delta points the wrong way, not
merely too short.**

---

## 2. ⭐ The literature has named this, and independently reproduced our failure

**ActSWM** (arXiv **2607.26712**, Gan et al.; banked) calls it **CONTEXT
COLLAPSE**: *"autoregressive latent predictors maintain high similarity to future
states while producing nearly indistinguishable futures under different action
sequences."* Their diagnostic is a **recorded-vs-all-zero-action gap** — the same
quantity as our `zero_sa` (`rdw8p30k`: **0.0058**).

⭐⭐ **They reproduce our O11 degeneracy exactly.** *"A jointly trained readout
increases separation at the cost of prediction quality"* — fidelity **0.972 →
0.698** while the action gap rose. Ours: `pick_acc` 1.000, `o5` **+18.7 %**,
`sep_rel` **12–17**. **Same failure, same shape, independent lab.** That is the
strongest external validation this campaign has.

**Their fix:** a hinge loss on rollout separation **plus a randomly-initialised,
PARAMETER-FROZEN action readout** on `[z_t, z_{t+1}]` whose loss backpropagates
**through the latent inputs only**. Reported: fidelity 0.923 with action gap 0.760.

⚠️ **Transfer risk, stated before any spend:** their domains are Minecraft, CS2,
GTA, Apex — **no driving**. In games the action largely *determines* the next
frame. Ours barely does over 0.4–3.0 s (E-DEC-39). **Their method creates the
signal rather than mining it, which is the right direction — but it has never been
tested where the causal effect is this weak.**

---

## 3. The hypotheses, ordered by evidence behind them

### H1 ⭐⭐⭐ — O11 asked the PREDICTOR for information the ENCODER never encoded

**Evidence.** E-DEC-37: O11's gradient is ~99.997 % concentrated on the
predictor's FiLM and essentially zero elsewhere. E-DEC-39: the information it
demanded is not in the transition. ⇒ **The term mined a seam that had nothing
behind it, so it manufactured separation instead.**

**Prediction.** A loss that backpropagates into the **latents** (ActSWM's frozen
readout) shapes the *encoder* to make transitions action-discriminative, rather
than asking the predictor to exploit what is absent.

**Test.** `--w-o12-readout` on the v7-tiny rig, matched pair against `ok8p30k`.
**Read:** does the drift-removed residual acquire action content (E-DEC-40 panel,
currently 0 of 8 arms at t > 2)? **Degenerate guard:** the readout must be
FROZEN — a trainable one reproduces O11 and ActSWM's own Table 5.

### H2 ⭐⭐ — The action is CONFOUNDED with the scene, so it adds nothing marginally

**Evidence.** Drivers brake *because* the lead brakes. In observational data the
action is a function of the scene, so a model that reads the scene can predict the
future **without** using the action — and the action's *marginal* contribution is
near zero even though its *causal* effect is real. This is the mirror of
**Causal Confusion in Imitation Learning** (de Haan et al., arXiv **1905.11979**;
banked), which shows the same entanglement destroying policies.

**Why it fits our data.** It explains why nine objectives failed *and* why the
information is genuinely absent from Δz: there is nothing for a marginal predictor
to gain.

**Test, cheap and decisive.** Measure `I(action ; future | scene)` empirically —
how much does the action improve future prediction **given** the scene features we
already have? If ≈ 0, no objective can help and the fix must be
**interventional data** (counterfactual actions, simulation) rather than a loss.
⚠️ This is the hypothesis that would most change the programme's direction, and it
is the cheapest to test.

### H3 ⭐⭐ — The horizon is wrong: predict what the action DETERMINES

**Evidence.** Over 0.4–3.0 s the ego's command moves the *scene* very little
relative to scene autocorrelation, but it determines the **ego's own pose**
almost exactly. We predict the whole 2048-d latent, where the action-dependent
part is a low-variance residual (E-DEC-40: action ≈ 4 % of drift, confined to
directions 8–16).

**Prediction.** An objective on the **ego-pose subspace** would be strongly
action-conditioned by construction, and could then be composed with the
scene-prediction objective rather than competing with it.

**Test.** Probe how well the action predicts Δ(ego pose) versus Δz — if the former
is near-deterministic and the latter is noise, the target is the defect, not the
model.

### H4 ⚠️ — The action embedding is an information bottleneck

**Evidence.** 3 scalars → FiLM modulation of a 256-d hidden. E-DEC-34 shows the
FiLM *does* open with training (|FiLM|/|act_emb| 0.072 → 0.308) but slowly, and
E-DEC-37 shows the seam receives ~25× less gradient than the state path.

**Status.** ⚠️ **Weakly supported and probably not the binding constraint** — the
`scale100_sa` probe shows arms *can* respond when the input is large enough, so
the pathway has capacity. Listed for completeness, not recommended.

### H5 ⛔ — Under-shoot / gain error — **TESTED AND LARGELY REFUTED**

α\* is 1.10–1.51 across arms, so the predicted delta *is* too small — but
rescaling gains ≈ 0 (−0.002 to +0.010). **The delta is mis-directed, not merely
short.** Retained only as a correction to the Q2X reading.

---

## 4. What I would do, and in what order

1. **H2 first** — it is the cheapest and the most consequential. If the action's
   marginal information given the scene is ≈ 0, then H1 and H3 are both wasted
   GPU and the answer is **interventional data**, not a loss.
2. **H1 second** — implement ActSWM's **frozen** readout as `O12`, matched pair,
   with the degeneracy guard already built into `actchan` (computed conjunction,
   validated on a known-bad arm).
3. **H3 third** — reframe the target rather than the loss.
4. ⛔ **Not H4.**

⚠️ **And adopt ACT-Bench** (arXiv **2412.05337**; banked) as an external
action-controllability yardstick. Every number in this campaign is our own
instrument; we have no external calibration for action-conditioning, and this
programme has now had **five instrument defects in two days** that only controls
caught.
