# How JEPA-family world models learn the PHYSICAL world without labels — survey + the label-free lever program

**Written 2026-08-11 after the PI's course correction (verbatim intent: "the whole idea of
the jepa-like wm architecture is the unsupervised learning of the physical world … if we
add a lot of labels, the whole idea looses its charme"). This RETRACTS the auxiliary
lead-readout-loss lever (labels into the trunk) and replaces it with a pre-registered
LABEL-FREE program. The P1 H-absent measurement stands — frozen probes that never train
the trunk are LeCun-orthodox instrumentation; it is the RESPONSE that must be
self-supervised.**

## 1. What the successful systems actually do (all PUBLISHED, cited)

**V-JEPA 2 / V-JEPA 2-AC (Meta, arXiv 2506.09985).** SSL on >1M hours of video (masked
latent prediction); then an action-conditioned world model post-trained on **62 h of
UNLABELED robot video** (Droid). Result: **zero-shot MPC pick-and-place on Franka arms in
unseen labs, planning with image goals, no rewards, no task labels** — and ~16 s/action vs
Cosmos's ~4 min. ⇒ Geometric/physical state demonstrably enters latents from prediction
alone, given (a) scale, (b) masked LATENT targets, (c) action-conditioning learned from
unlabeled interaction data, (d) planning cost = latent distance to goal. *Their planning
loop is structurally OUR W7 roll-cost re-rank — the field's proven recipe matches the
selection mechanism we just calibrated (ρ 0.716).*

**DINO-WM (NYU/LeCun group, arXiv 2411.04983, ICML 2025).** Dynamics learned on frozen
DINOv2 **PATCH features** (never pooled to a global vector), predicting future patch
embeddings; zero-shot planning via MPC with latent-MSE goal cost. ⇒ The physical detail
(where things are, how far) lives in **spatial tokens; pooling is where geometry goes to
die**. No pixels, no labels, no rewards.

**AD-L-JEPA (arXiv 2501.04969) + JEPA LiDAR occupancy WMs (arXiv 2602.12540).** JEPA-style
BEV-embedding prediction for driving; **occupancy completion and forecasting emerge from
self-supervised latent prediction** — read out by light decoders, exactly our P8 pattern.
⇒ Occupancy-style environment decoding is the field-standard way to SHOW physical content
without supervising the trunk.

**V-JEPA 2.1 (arXiv 2603.14482).** Finding: *dense spatio-temporal features do NOT emerge
reliably from standard SSL*; fixes are **loss-shaping, not labels** — masked-token L1 plus
a **distance-weighted loss on nearby context tokens**. ⇒ Direct precedent that "the latent
drops a physical variable" is treated with objective design (weighting, masking), keeping
the pipeline label-free.

**Theory note (JEPA generalization, 2026 survey line).** JEPA risk behaves like a low-rank
factorization of the action-conditioned co-occurrence operator — **rare-event variables are
exactly what a fixed-rank latent sacrifices first**. A free-flow-dominated corpus makes
lead-distance a rare-event variable: the objective CAN be minimized while ignoring it.

## 2. Root-cause hypotheses for OUR miss (v5f latent, lead distance), each label-free-testable

- **RC1 — pooling bottleneck (DINO-WM lesson).** The P1 probes read the readout-grid
  state the planner consumes. If lead geometry lives in SPATIAL cell tokens and dies in
  aggregation, the defect is routing, not learning.
- **RC2 — the objective never needs it (theory + corpus statistics).** Windows where the
  lead CAUSALLY constrains ego (closing, braking) are a small corpus fraction; prediction
  loss is minimizable without lead state.
- **RC3 — no spatial masking pressure.** We predict full-context futures; I-JEPA/V-JEPA
  masking forces local scene structure into latents. We never applied it to the readout grid.
- **RC4 — horizon.** At 2 s, lead distance barely changes in free flow — little predictive
  value; V-JEPA-2-class models bind object dynamics over longer spans.

## 3. The pre-registered LABEL-FREE lever program (replaces the retracted aux-label lever)

| id | lever | labels? | cost | gate (frozen P1 lead battery, before/after) |
|---|---|---|---|---|
| **LF0** | **Locate first**: run the P1 lead probes on PRE-POOL spatial cell tokens (and P8-attempt-2's decoded BEV read-off) | probe-only (admissible) | ~0.5 GPU-h | if spatial tokens read gap (R² ≥ 0.3) ⇒ RC1 confirmed: the fix is exposing spatial tokens to the planner/probe path, ZERO new training |
| LF1 | interaction-weighted sampling: oversample windows by EGO-KINEMATIC saliency (decel/jerk events, from actions alone — no obstacle data) in a stage-B continuation | none | ~6 GPU-h | lead probes improve while P1's other targets + no-harm hold |
| LF2 | masked-latent-prediction auxiliary: I-JEPA-style masking over the readout grid (predict masked cell tokens from context) | none | v6-scale arm | same + P8 IoU retention improves |
| LF3 | dense distance-weighted latent loss on near-field tokens (V-JEPA 2.1 pattern) | none | stage-B/v6 | same |
| LF4 | longer-horizon latent rollout targets (E-H1's 6 s, as a REPRESENTATION lever, not just an eval) | none | rides W5 | lead probes at k=20+ |

**Standing rule restated:** labels (obstacle.offline, VLM outputs) are admissible as
FROZEN PROBE TARGETS and EVAL strata only — never as trunk losses, never as inputs. The
P8 occupancy readout is a measurement instrument (trunk md5-frozen), which is exactly how
the AD-L-JEPA line uses occupancy too.

**Order:** LF0 decides everything and is nearly free — it runs before any training lever
is spent. RC1-true collapses the problem to architecture routing; RC1-false makes LF1
(cheapest training lever, stage-B scale) the first spend, gated on the frozen probes.

- [ ] LF0 run · [ ] RC verdict banked · [ ] first training lever chosen by LF0's outcome

## Sources

- V-JEPA 2: https://arxiv.org/abs/2506.09985 · https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/
- DINO-WM: https://arxiv.org/abs/2411.04983 · https://dino-wm.github.io/
- AD-L-JEPA: https://huggingface.co/papers/2501.04969
- JEPA LiDAR occupancy WM: https://arxiv.org/abs/2602.12540
- V-JEPA 2.1 dense features: https://arxiv.org/html/2603.14482v2
- Driving-WM surveys: https://arxiv.org/html/2501.11260v2 · https://arxiv.org/pdf/2411.02914
