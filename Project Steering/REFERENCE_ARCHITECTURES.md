# Reference Architectures — build plans (directed by Sayed, 2026-07-09)

Two full 4B-structured reference stacks, built to be *beaten measurably* (or to beat us and
redirect Phase 1). Both share our data pipeline, episode contract, gates, and instrument doctrine —
differences are isolated to the architecture axis being tested. Both are SEPARATE packages under
`stack/tanitad/refs/` (never entangled with the main model).

---

## REF-A: TanitAD-4B-DINO — frozen-DINOv3 world model (H4 arm-B, full stack)

**Question it answers:** does two decades of open-web visual pretraining beat our task-specific
from-scratch SSL encoder at matched predictor capacity — on gates, sample efficiency, and OOD?

**Design:**
- **Encoder: DINOv3-B/16, FROZEN** (86 M params, not counted as trainable). 256 px → 16×16 tokens
  — drops into our existing `SpatialGridReadout` geometry unchanged.
- **Adapter (trainable, small):** LayerNorm + linear projection d_dino→768 (+ optional 2-layer
  bottleneck). This is where stability lives (below).
- **Predictor + tactical + readout: IDENTICAL to ours** (107.7 M + 26.5 M + readout) — the
  comparison isolates the encoder.
- Inverse dynamics + SigReg on predictions (NOT on frozen features — nothing to regularize there).
- **H15 imagination:** attaches to DINO tokens identically (sector masking on the token grid).

**Stability assurance (Sayed's explicit requirement):**
1. **Feature standardization:** per-channel running mean/var normalization of frozen features
   computed ONCE over the training corpus (frozen statistics — no train/eval drift possible).
2. **No gradient can touch the encoder** — enforced by construction (`requires_grad=False` +
   a CI test asserting zero encoder grads after a training step).
3. **SigReg only on predictor outputs** with the same ≥256-samples/step floor (F-2 rule).
4. **Adapter LR warmup** separate from predictor (adapters on frozen features are the known
   instability point — 10× longer warmup, gradient-norm monitor row).
5. **I2 batch-consistency** applies unchanged (DINO in eval mode, no dropout/BN issues — ViT+LN).
6. **Feature-cache training:** precompute DINO features once (embarrassingly parallel), train from
   the feature cache — removes the frozen encoder from the training loop entirely; steps are
   cheap and CANNOT be destabilized by encoder-side numerics.

**Cost & schedule:** feature precompute over comma caches ≈ hours (pod2/Colab); predictor training
~110 M trainable at feature-level batches — a fraction of main-run cost. First probe comparison vs
our 15k ckpt within a day of starting; gate-matched comparison at our 30k.

**Decision rule (pre-registered):** REF-A wins an edge only on gate deltas outside seed noise at
matched steps AND matched val routes (I3/I7). If REF-A wins D1/D3 but loses D2/OOD (or vice versa),
Phase 1 considers a hybrid (frozen DINO + our SSL fine-tune head) — evidence first.

---

## REF-B: TanitAD-4B-E2E — vision-action end-to-end with abstraction layers (NO world model)

**Question it answers:** does the world model earn its parameters — or would the same budget spent
on a direct hierarchical vision→action stack drive equally well? (The strongest fair "standard
E2E" opponent, and gate D4's learned baseline.)

**Design (matched ~260 M budget, hierarchical like ours but predictive of ACTIONS, not latents):**
- **Encoder: identical trunk to ours** (99.5 M, trained from scratch — same data, same aug).
- **Abstraction layer 1 — operative action head (10–20 Hz):** causal transformer over the state
  window → direct (steer, accel) regression + short action sequence (0.5 s) — the reactive path.
- **Abstraction layer 2 — tactical intent head (1–2 Hz):** predicts maneuver-class distribution +
  2 s waypoints (the "abstraction" — intent tokens condition the operative head, FiLM).
- **NO predictor, NO imagination, NO SigReg-on-predictions** — the parameter budget freed
  (~130 M) goes into deeper encoder + wider heads (budget-matched, enforced by test like D-008).
- Training: behavior cloning on the same real mix + the same inverse-dynamics auxiliary
  (fair aux), same schedule/steps.
- **What it structurally CANNOT do** (and where we expect to beat it, pre-registered): OOD
  self-knowledge (no imagination error signal → D8), hidden-actor anticipation (no latent rollout
  → LOPS/SC-02), counterfactual maneuver evaluation (no imagine-and-select → D4 tactical lift),
  closure reasoning (SC-01). If it MATCHES us there, the world-model premise is wounded — that is
  the point of an honest reference.

**Gates it runs:** D1 (waypoints from its tactical head), D4 closed-loop vs our tactical layer,
the scenario suite (SC-01/SC-04 CARLA), D8 (expected to fail — that failure is a *measurement*,
not an assumption), latency ledger (it should be FASTER per tick — the efficiency trade goes in
the table too).

**Cost & schedule:** one training run at main-run scale (pod-class, ~2 days) — schedule AFTER the
30k main run frees pod1, or as the first Phase-1 allocation. Cheap preview: BC heads on top of our
FROZEN 30k encoder (days earlier, weaker claim, labeled).

---

## Sequencing & ownership
1. REF-A feature precompute + predictor training: behind the K-step arms on pod2 (this week).
2. REF-B: full run post-30k on pod1 (next week); frozen-trunk preview optionally earlier.
3. Both feed the **architecture-design workflow panel** (D-026) — the panel judges OUR stack vs
   REF-A vs REF-B vs the bake-off lever variants on measured gate tables only.
4. Owner: MVP loop builds; Architecture agent reviews; results → paper §7 comparison table (the
   three-way comparison IS the H1/H4 evidence section).
