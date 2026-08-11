# V6 FLAGSHIP — the driving-adapted measure catalog for teaching the 4B hierarchical WM
# the real physical world (PI directive 2026-08-11)

**Scope:** the full set of diverse, driving-adapted measures for a v6 redesign+retrain,
organised per abstraction layer of the 4B hierarchy — **each layer with its own predictor
(and possibly own encoder — E-ENC below), each focused on its slice of the driving task,
emitting ACTIONS that condition its own predictor and GOALS that condition the layers
below.** All trunk-side measures are LABEL-FREE (the JEPA thesis, PI-binding since the
aux-label retraction); labels appear only as frozen probes, eval strata, and goal-head
supervision (planner-side, never WM-trunk losses). Every measure names its gate; gates are
the frozen WM-physics battery (P1–P9, I4) so v5f→v6 progress is measured on ONE yardstick.

## 0. The two cross-cutting design questions, with their deciding experiments

**Q1 — separate encoders per layer vs one common encoder?** Pre-registered arm **E-ENC**:
(a) common encoder + per-layer adapters (parameter-cheap, one visual substrate;
DINO-WM-style layers-on-frozen-features); (b) per-layer encoders (each layer sees the
input at its own resolution/rate: operative full-rate near-field, strategic low-rate
wide-context). Decision metric: per-layer P-battery pass rate at MATCHED total params
(sub-300M invariant); tie → common encoder wins on the parameter budget. Prior from the
field: every frontier system (V-JEPA2, DINO-WM, Drive-JEPA) uses ONE encoder with
downstream consumers — separate encoders must EARN their params.

**Q2 — train layers alone-then-together, or together at once?** The evidence (Drive-JEPA,
V-JEPA2→2-AC, our stage-A/W4r gate history, H-COTRAIN pending) points to **bottom-up
staged**: the operative WM is the physical substrate and trains FIRST (S-W); each higher
layer then trains ON the frozen(or EMA-slow) layer below, consuming its latents and
emitting goals downward; a final OPTIONAL brief joint polish (S-J) with gradient isolation
(stop-grad from every planner/goal head into any encoder; inter-layer gradients gated).
Bound decision rule: if H-COTRAIN's milestone curve shows scene-state erosion under joint
training → staged is MANDATORY; if not → staged still preferred on the field evidence, and
S-J may lengthen. The `--lambda-plan 0` mode is the per-stage instrument.

## 1. LAYER O — operative (0–2 s; ego dynamics + near-field scene)

Actions: continuous controls (a, κ) — unicycle-constrained AT TRAINING TIME (W4's lesson
promoted from retrofit to design: the emission space is born feasible, a=a_max·tanh,
κ=κ_max·tanh). Conditions its own predictor; receives tactical goals (anchors/corridor).

| id | measure | driving adaptation | gate |
|---|---|---|---|
| O1 | action-conditioned latent prediction (JEPA core) with **L_ctrl in response form from step 0** | counterfactual action arms every batch; response gain trained INTO [0.5, 2]× the unicycle analytic — the stage-A repair as a native loss, not a patch | W3/P3: sign ≥95 % both channels, gain in band, WITHOUT post-training |
| O2 | **near-field distance-weighted latent loss** (V-JEPA 2.1 pattern) | prediction error on readout-grid cells weighted by inverse ego-frame distance — the 0–40 m band where physics bites (lead vehicles, cut-ins) outweighs sky/far field | P8 IoU retention ≥0.8× at k=10 AND the LF0 lead read-off becomes positive |
| O3 | **masked spatial-latent prediction** over the readout grid (I-JEPA adapted to BEV-ish cell tokens) | mask contiguous SPATIAL blocks (an occluded-vehicle surrogate) and near-field bands; predict them from context+action | P4 occlusion permanence; P8 occluded-recall <2× visible |
| O4 | **interaction-weighted sampling** | oversample windows by ego-kinematic saliency (|jerk|, |decel|, steering reversals — from actions ONLY, label-free); free-flow windows down-weighted | P1 lead battery (via LF0 route) improves; P1 speed/curv/yaw retained |
| O5 | short-horizon **rollout-consistency loss** (multi-step latent rollout ≤2 s, error at every step, not endpoint-only) | compounding-error shaping — the T1 lesson (P5) trained in | P5: T1 log-slope ≤ baselines through 2 s |
| O6 | SIGReg (full_relaxed) per layer — KEPT (PI 2026-08-11) | anti-collapse validated by spectrum monitoring across training (participation ratio, top-k share — the H-COTRAIN instrument becomes a standing training-time monitor) | effective rank retention ≥0.8× across any curriculum phase |

## 2. LAYER T — tactical (2–8 s; manoeuvre + interaction)

Actions: manoeuvre-scale geometric goals (anchor selection / corridor targets) — factored
LAT×LON (the 5-way-mixed-softmax defect retired by design). Conditions its own predictor;
receives strategic corridor goals; emits goal anchors down to O.

| id | measure | driving adaptation | gate |
|---|---|---|---|
| T1 | goal-conditioned latent prediction at 2–8 s (own predictor, coarser Δt) | the predictor rolls under a GOAL, not raw controls — tactical dynamics are "which corridor", not "which steering angle" | E4.4-family: fan-generates + selector-executes on ORACLE goals first (E5 lesson), then produced |
| T2 | **manoeuvre-contrastive windows** (label-free): time-reversal and lane-mirror augmentations as hard negatives for the tactical predictor | a lane change mirrored is the OPPOSITE manoeuvre — the predictor must not be invariant to it; teaches manoeuvre identity without manoeuvre labels | TACTICAL family: confusion improves on the E4.1-derived (offline, privileged-OK) eval strata |
| T3 | interaction curriculum: windows ranked by MULTI-AGENT kinematic entropy measured from the O-layer's own predicted occupancy (self-supervised, after O2/O3 make it non-degenerate) | curriculum from free-flow → dense interaction | P7 calibration ρ ≥0.3 held on interaction-rich strata, not just pooled |
| T4 | imagination-closed goal scoring (W7's roll-cost as a TRAINING signal for the tactical selector — distill roll-cost rankings into the selector, L4/W7-distill) | selection learns from the WM's own imagined consequences, not from labels | sel_gap ≤0.5× fan oracle at T1 tier; P7 ρ retained by the distilled selector |
| T5 | temporal-consistency selection loss (momentum-aware, Drive-JEPA pattern) | penalise plan flip-flop across consecutive windows (cross-frame comfort) | LATERAL family: yaw-rate/curvature MAE at selection level; plan-switch rate reported |

## 3. LAYER S — strategic (8–30 s+; route/corridor intent)

Actions: strategic goals (corridor keep/exit, target arc) conditioning T. This is the ONE
layer where external supervision is admissible BY DESIGN — into the GOAL HEAD, never the
trunk: goals are planner-side outputs, and the PI's goal rules (2026-08-03) apply
(derivation-source tags; never a function of the situation classifier's output).

| id | measure | driving adaptation | gate |
|---|---|---|---|
| S1 | long-horizon latent prediction (own predictor, Δt ≈ 1 s ticks) on the T-layer's latent sequence | strategic dynamics = evolution of manoeuvre context, not pixels | ADE(8–30 s) vs CV/corridor baselines at T1 |
| S2 | **g_str supervision from the VLM/geometric pipeline** (PH0→PH1→PH2, see §6) | hindsight-geometry + signage-derived goals; `route_to` only with OCR evidence; abstain honest | STRATEGIC family becomes computable for the first time — measured vs n/a today |
| S3 | domain-stratified training mix (geographic/domain diversity beats volume — arXiv 2607.04500) | the S1 scaling-ladder data-mix arm folds in here | cross-domain P-battery deltas reported per stratum |

## 4. CONTEXT/FALLBACK layer

| id | measure | adaptation | gate |
|---|---|---|---|
| C1 | nuisance NON-retention pressure: P2's clip-ID/appearance probes as a standing monitor (not a loss — measure-only; if P2 fails, the fix is masking/augmentation in O3, never a reverse-gradient hack) | driving abstraction must drop weather/texture identity with k | P2: clip-ID decodability falls with k |
| C2 | uncertainty/fallback: fan-spread calibration (P7) trained-in via T4's distillation target including cost VARIANCE | self-knowledge = knowing where imagination disagrees | P7 ρ ≥0.3 with CI excluding 0, per stratum |

## 5. Cross-layer measures (the hierarchy itself)

| id | measure | gate |
|---|---|---|
| X1 | **goal/action disjointness audits** at every seam (the 2026-08-03 rules as CI checks: no layer's goal input derivable from the situation-classification path) | audit script green pre-launch |
| X2 | **seam metrics** with paired episode-cluster bootstrap (the corrected ctx→tactical seam lesson: +0.0148 true, not +0.0439) — every seam quoted with the paired estimator only | per-seam paired CI in the registry row |
| X3 | **gradient isolation matrix** (who may backprop into whom): planner/goal heads NEVER into encoders; higher layers into lower layers' latents only through stop-grad or EMA-slow copies (H-COTRAIN's confirmed-branch lever, applied preventively) | H-COTRAIN battery flat across joint phases |
| X4 | per-layer SIGReg + per-layer spectrum monitors (O6 pattern at T/S scale) | rank retention per layer |
| X5 | staging protocol S-W → S-T → S-S → (optional) S-J, each stage gated by the frozen battery BEFORE the next begins — a failed stage never propagates upward | battery pass per stage |

## 6. The VLM/algorithmic pipeline in the v6 plan (restored explicitly — PI flag 2026-08-11)

The pipeline is NOT displaced by v6 — it is v6's strategic-layer supervision source and
was always queue position: **after the v5.8f release row closes** (§1.14 + T1 + P8/P9):
1. **PH0** (prereg BANKED, `PREREG_PH0_VLM.md`): 3 arms (Qwen3.5-9B / 27B-FP8 /
   Gemma-4-31B-QAT, all prefetched on pod4), 50 stratified clips, video-template smoke
   first, gates G1 sign-OCR ≥0.9 / G2 schema ≥0.9, PH1-model selection rule bound.
2. **PH1**: full ~4.8k-clip labeling run (spend approved from PH0's measured s/clip).
3. **PH2**: g_str supervision stream into v6's S-layer goal head (S2 above), P2 nuisance
   strata from weather/illumination fields, domain-stratified four-family evals,
   sign-conditioned goal experiments.
   Engine A (geometric hindsight) ships regardless of VLM choice — corridor/lane-level
   goals derive from integrated ego trajectories alone.

## 7. What decides GO on v6 (inputs the PI signs off on)

1. Tonight: W7-FULL selection verdict + H-COTRAIN curve + SIGReg spectrum + p8c gates.
2. v5.8f release row (T1 + four families) — the baseline v6 must beat.
3. E-ENC + staging preregs (this doc §0) formalised as PREREG_V6.md with costs.
4. Scaling-ladder S1 arm (data volume/mix) — the PI spend decision already queued.

*Every measure above traces to a measured defect or a PUBLISHED mechanism; none introduces
labels into any trunk. The battery (P1–P9, I4) is the single yardstick from v5f to v6 —
that continuity is what makes "v6 learned the physical world better" a MEASURED claim
rather than a hope.*
