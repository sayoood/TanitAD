# The hierarchical WM redesign — eval doctrine, on-policy post-training, layered predictors, horizon

**Draft v1, 2026-08-07.** Response to the PI's four remarks (verbatim intent preserved per
point). Design + mathematics + training strategy + discriminating experiments. Evidence
classes marked; every number cited is MEASURED in this repo unless tagged PUBLISHED.

---

## 1. The eval doctrine — the PI's critique is correct and already measured

**Remark:** *"if the model is consuming at eval the future gt data, then its not really an
eval, eval must be without gt."*

Correct, and §1.12 quantified exactly how much it matters: with the recorded future actions
removed, ADE degrades 0.34→0.47 (v1.6) / 0.28→0.46 (v1.7), net-yaw ×6.7, S-curve
reproduction 97.9 % → ~5 % (hold-action arm: **0.0 %**). The open-loop numbers measured
prediction quality *conditioned on the answer's own action transcript*.

**Doctrine change (binding for every future claim):**

| tier | what runs | what it measures | status |
|---|---|---|---|
| T0 — WM diagnostic | teacher-forced roll under GT actions | predictor fidelity, readout quality. **Never quoted as driving performance.** | exists (all §1.10 instruments) |
| T1 — action-closed loop | §1.12 pipeline: predictor consumes the decoder/planner's own actions; perception context fixed at t0 | driving competence *within the WM's imagination* — the primary offline eval from now on | exists (`closed_loop_dump.py`) |
| T2 — perception-closed loop | AlpaSim / NuRec: re-render what the ego would now see | true closed-loop driving incl. scene interaction | **missing — the resource ask of this document** |

Every registry row gets its tier stamped. The four families + S-curve + lag instruments all
run unchanged at T1; T0 keeps its role as the attribution tool (it is how we proved the
readout, not the roll, caused the decel ramp).

---

## 2. Post-training the predictor/encoder on planner actions — the mathematics, honestly

**Remark:** *post-train predictor+encoder to be actioned with the decoder/CEM/MPC planner
outputs instead of GT data — "sort of end2end training".*

### 2.1 The counterfactual-supervision problem (why naive end-to-end is not available)

The log supplies triples `(o_t, a^human_t, o_{t+1})`. The predictor learns
`ẑ_{t+1} = f(z_t, a_t)` supervised only **on the human action manifold**. Under a planner
action `a^plan ≠ a^human` there is **no logged `o_{t+1}`** — the counterfactual future was
never observed. Any loss of the form `‖f(z, a^plan) − z'_{logged}‖` is *wrong by
construction*: it teaches the predictor that the world ignores actions.

⭐ **And §1.10's probes show our predictor already half-learned exactly that pathology,
asymmetrically:** under held (constant) actions, the rolled latents still linearly encode
the *real* future speed at R² 0.92 @1 s (the scene predicts the human's speed profile — the
action channel is partially ignored longitudinally), while lateral content collapses
(yaw-rate R² 0.46 @1 s, S-reversals 0 %). So longitudinally the WM is **under-controllable**
(actions don't steer the imagination enough) and laterally it is **over-reliant** on the
action transcript. Both directions must be fixed before any MPC over this WM can work.

### 2.2 What IS supervisable off-policy — the controllability losses (stage A, cheap)

The ego-motion component of the future is **fully determined by the commanded actions** —
no logged future needed. That yields self-supervised losses valid for ANY sampled action
sequence `ã` (perturbed human, planner output, or random within the envelope):

- **L_ctrl (action-controllability):** roll `ẑ_{1..K} = f(z_0, ã)`, decode with the frozen
  readout, and require the decoded ego trajectory to equal the unicycle integral of `ã`:
  `L_ctrl = ‖decode(f_roll(z_0, ã)) − Unicycle(ã, v_0)‖₁`.
  This trains the *predictor* (and optionally encoder) to carry action effects — the
  readout is frozen so the gradient cannot cheat by re-learning the decode.
- **L_scene (non-ego stability):** the same perturbation must NOT rewrite the scene:
  penalise divergence of the non-ego content between `f_roll(z_0, ã)` and
  `f_roll(z_0, a^human)` beyond what the ego-motion difference implies. Cheap proxy: distance
  in the latent subspace orthogonal to the ego-motion probe directions (we already have the
  ridge probes; their weight vectors define the ego subspace).
- **L_anchor (on-manifold fidelity):** the ordinary teacher-forced loss on human actions,
  keeping T0 fidelity from degrading (the catastrophic-forgetting guard).

Total: `L = L_anchor + λ_c·L_ctrl + λ_s·L_scene`, each λ with a pre-registered zero
ablation. **This is stage A: no simulator, no new data, runs on the existing corpus + the
existing frozen readout, and directly attacks the two measured §1.10/§1.12 defects.**
Gate for stage A (pre-registered): T1 closed-loop ADE and S-curve reproduction must
improve; T0 fidelity must not degrade beyond CI.

### 2.3 What needs an environment — stage B (the resource ask)

Scene *interaction* (a cut-in that responds to our braking; occlusions revealed by our lane
change) is not identifiable from logs under counterfactual actions — period. Two paths:

- **B1 — sim-in-the-loop (AlpaSim/NuRec):** Dreamer-style iteration: plan with MPC over the
  WM → execute in sim → the sim's rendered futures become NEW (o, a, o') triples on the
  *planner's* distribution → retrain. This is the only path to true on-policy correction,
  and it is exactly what NVIDIA's own closed-loop numbers (AlpaSim 910 scenarios, PUBLISHED)
  are built on.
- **B2 — model-ensemble pessimism (interim):** train 2–3 predictor heads, use their
  disagreement as an epistemic penalty in the MPC cost (candidates leaving the data
  manifold get penalised where the ensemble disagrees). Standard MBRL practice; no new data.

### 2.4 Verdict on "end2end"

End-to-end *gradient* coupling of planner→predictor→encoder on logged data alone cannot be
made sound (2.1). What IS sound: stage-A controllability post-training (self-supervised,
now), + B2 pessimism (now), + B1 sim-loop (with provisioning). The λ-seam machinery from v4
(`grad_scale`) is the right tool when we do couple layers: forward-identical, backward
scaled — attribution preserved.

---

## 3. The hierarchical architecture — design, math, training, data, the claim as an experiment

**Remark (the pillar):** separate tactical and strategic predictors in their own abstraction
spaces, emitting candidate goals, each with a selector; goals condition the level below;
horizons tactical 2–6 s, strategic >6 s; claims: data efficiency, inference efficiency,
edge-case generalisation (fast/slow thinking), navigation.

### 3.1 State spaces and clocks

```
level      state          clock   built by                          horizon
operative  z_op ∈ R^2048  10 Hz   existing encoder (frozen at first) 0–2 s dense
tactical   z_tac ∈ R^d_t   1 Hz   φ_tac: TCN/attn pool over z_op(t−3..t) 2–6 s
strategic  z_str ∈ R^d_s   0.2 Hz φ_str: pool over z_tac window          6–15 s
```
Down-sampling in TIME is the abstraction mechanism (slow features); d_t ≈ 512, d_s ≈ 256
(information decreases upward by design — the level must be *forced* to abstract).

### 3.2 Goal spaces (the admissibility rules bind here)

- **g_tac** = predicted ego-frame **goal point + heading + target speed at τ ∈ [2,6] s**,
  PLUS a manoeuvre class on **three independent axes with severity** (the Alpamayo
  meta-action lesson — our 5-way softmax's lat/lon mixing is a measured defect).
- **g_str** = goal point + corridor (left/straight/right at route scale) at τ ∈ [6,15] s.
- ⛔ **Neither goal may carry the situation classifier's output** (binding rule
  2026-08-03). Both are geometric/kinematic, predicted from vision. Information-disjointness
  stated per arm; any shared trunk justified explicitly.

### 3.3 Level dynamics and selectors

Each level owns a **predictor in its own space**:
`ẑ_tac(t+1s) = f_tac(z_tac, g_tac)` — the tactical "action" IS the goal; likewise f_str.
Each level proposes N candidates (goal fan), rolls ITS OWN predictor (cheap: small state,
slow clock — strategic roll of 15 s = 3 steps of a 256-d model), scores with a selector:

`score_ℓ(g) = w·[goal-feasibility, predicted-progress, comfort-at-level, consistency-with-g_{ℓ+1}]`

⛔ **The selector is the known failure point** — v5f measures a good fan and a selector
that never closes (`sel_gap` ~0.3–0.5, no trend). Two consequences baked in: (1) selectors
are trained with **ranking losses against hindsight-oracle outcomes** (which candidate's
rolled future was closest to what a good driver achieved), not classification; (2)
`sel_gap` per level is a first-class metric in the eval (oracle-vs-selected at every level).

**Conditioning downward:** operative decode consumes `g_tac*` (selected) as an additional
input token; tactical consumes `g_str*`. Fast/slow: operative re-plans at 10 Hz always;
tactical at 1 Hz + event triggers (TTC drop, goal invalidation); strategic at 0.2 Hz.

### 3.4 Labels (all hindsight, all admissible)

- Achieved-goal labels: `g_tac^label(t) = ego pose/speed at t+τ` (labels-may-use-ego rule).
- Manoeuvre axes: refb_labels machinery, extended with the severity split.
- Strategic corridor: net-heading over 25 s (`nav_command` exists in refb_labels).
- **Hindsight relabeling doubles as data augmentation:** every window supervises every τ.

### 3.5 Training strategy — staged, attribution-preserving

- **Stage 0:** freeze z_op trunk (v1arch). Train φ_tac + f_tac + tactical goal head on
  hindsight goals. Then φ_str + f_str likewise. Each with its own four-family-relevant
  gates. *(Zero risk to the operative stack; pure addition.)*
- **Stage 1:** train operative decode WITH g_tac* conditioning (readout-scale, ~2 M params,
  1 h/run — the v1.6/v1.7 loop). Pre-registered: goal-conditioned vs unconditioned at
  matched params; the S-curve rate and T1 closed-loop are the discriminators (a correct
  tactical goal should restore the counter-steer WITHOUT the GT action transcript —
  this is the direct attack on the §1.12 lateral collapse).
- **Stage 2:** joint fine-tune through λ seams (v4 `grad_scale`), λ swept from 0; every
  loss with a zero-ablation. Stage-2 proceeds only if stage-1 gates pass.
- **Stage A post-training (§2.2) composes with any stage** — it conditions the substrate
  all levels roll on.

### 3.6 The efficiency claim, made falsifiable (pre-registration sketch)

*Claim:* layered learning reaches equal T1 performance with substantially less data than a
monolithic block at matched total params.
*Experiment:* corpus-scaling ladder {150, 300, 600} episodes × two arms — (H) hierarchy
stages 0–1, (M) monolithic operative-only at SAME added parameter count. Outcomes
committed in advance: H beats M at low data on T1 ADE + S-rate + tactical goal accuracy
⇒ claim supported at this scale; H ≤ M everywhere ⇒ claim refuted at this scale, and the
next lever is the goal-space (not more layers). Cost: 6 head-scale runs ≈ 6 GPU-hours on
the frozen trunk. **This is the cheapest discriminating experiment for the programme's
central thesis, and it is runnable THIS WEEK.**

### 3.7 Data & resources

- Operative/tactical: existing v2bal corpus suffices for stage 0–1 (hindsight labels).
- Strategic: needs route diversity; PhysicalAI has **no maps** (settled 5-probe fact) —
  corridor labels come from ego hindsight; intersection-rich coverage now *measured* by the
  road classifier (aug stream) so the strategic train/val split can be stratified by it.
- The Alpamayo augmentation dataset (running) supplies **auxiliary distillation targets**:
  its meta-actions (3-axis, severity) and CoC traces are exactly tactical-level supervision
  — a second, independent teacher for stage 0. (Contamination caveat travels.)
- Sim (T2/B1): AlpaSim/NuRec provisioning — PI decision; everything else above runs without it.

---

## 4. Horizon — extend, with anchors

**Remark:** confirm or extend the 2 s trajectory horizon against the best-known work.

PUBLISHED anchors: Alpamayo 2 Super emits **6.4 s @ 10 Hz** (64 wp). nuPlan planning
evaluates **8 s**; Waymo motion/planning benchmarks use **8 s**; UniAD plans **6 s**;
nuScenes prediction convention is 6 s (3 s for some detection-adjacent tasks). **2 s is
short of every planning-grade reference.**

**Recommendation:** operative emitted plan → **6 s (60 wp)**, aligning the operative/
tactical boundary with the remark's 2–6 s band; control-grade density (0.1 s) kept
throughout; the 0–2 s prefix remains the dense-control segment the temporal-stability
instruments run on.

**Feasibility evidence already in hand:** the predictor's 2 s latents decode future speed
at R² 0.95 (§1.10 probes) — but 6 s is 3× the rollout length; error compounding is the
risk. **Experiment E-H1 (cheap, decisive):** extend the v1.7 readout to k=60 on the frozen
trunk (head-only retrain, ~1.5 h), measure ADE(τ) growth, reliance at long horizon, and T1
closed-loop at 6 s. If the curve breaks badly past ~3 s, the trunk needs longer-horizon
training (a v5f-class decision, not a head decision) — and we will know exactly where.

---

## 5. What this document commits us to (execution order)

1. **Eval doctrine flip** — T1 becomes primary (instrument exists; a doc + registry-note
   change, this week).
2. **E-H1 horizon probe** (1.5 GPU-h) and **stage-A controllability post-training**
   (pre-registered; first run ≈ 3 h on the free pod after the Alpamayo batch) — both
   head/predictor-scale, both falsifiable, both feeding every later stage.
3. **Stage-0 tactical layer** (φ_tac, f_tac, goal head, selector w/ sel_gap metric) —
   the first brick of the pillar; then the §3.6 efficiency ladder.
4. **Stage-1 goal-conditioned operative decode** — the S-curve-restoration test.
5. Strategic layer after tactical gates pass; B1 sim loop when provisioned.

*Supersedes nothing: MPC_WM_DESIGN.md remains the operative-selector option (its E0 probe
is unchanged); v5f continues as the co-trained monolithic reference arm the §3.6 ladder
needs for comparison.*
