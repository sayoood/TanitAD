# LAB RUN 001 — Architecture & Inference — literature pass + ideation

`TanitAD Research Lab, daily run 001. Run label 2026-08-22 (Master Mind trigger);
executed 2026-08-23 wall-clock. Literature only, no GPU. Agenda targets:
RESEARCH_AGENDA.md Field 2, item 1 (⭐ anti-collapse, the live thread) and item 5
(encoder question). PRIORITY CONTEXT searched: 2026 work on JEPA/world-model rank
dynamics, prediction-horizon effects, input-frame-stacking effects.`

## Where the programme stands (the thing the literature is read against)

INHERITED from the register (`GOALS_AND_CLAIMS.md` H-RANK-1…15, read from
disk 2026-08-23 — it supersedes the trigger brief): the **two-term objective —
next-latent L1 at k=1 + subspace-SIGReg (K=32) — PREVENTS the collapse that
killed the six-term v6 objective**: all-six rose to 8.96 @16k then fell to
5.55 @20k (−38 %); two-term+k1 (`champ30k`) rises and then **PLATEAUS** (train
windows 6.748 peak → 6.623 at 24–30k, H-RANK-11). Val-side participation
**4.052 @6k → 6.499 @30k** (+60 %, n=1440, H-RANK-12) — yet **still below the
8.56 frozen-DINOv3 floor: the O6 gate FAILS** (H-RANK-13), and the σ-vs-σ²
inversion reproduced (lowest effective_rank, highest participation, H-RANK-15).
⚠️ The 8.56 floor's own provenance is now contested in the register (H-RANK-16:
the code floor 8.56 and E-TRUNK-3's DINOv3 reference 40.77 are different
instruments) — a gap statement here is conditional on which instrument rules.
⚠️ The trigger brief quoted "participation ≈ 11"; the register's val number is
6.499 — **registry wins**, and this file quotes the register. Elimination tree:
SIGReg weight REFUTED (H-RANK-3), term count SUPPORTED (H-RANK-4), estimator
conditioning REFUTED (H-RANK-5), capacity ratio REFUTED (H-RANK-6), k=1
SUPPORTED (H-RANK-7), train-pooled gate inflation SUPPORTED but NOT a fixed
offset (H-RANK-9/14), frame stacking OPEN/unrun (H-RANK-8). Not re-verified
here. **The open problem is therefore the remaining 6.5 → 8.56 gap**, which is
exactly what F2.2/F2.3 and H-RANK-17 below address.

## Findings (2026 literature vs the live thread)

### F2.1 — "Global non-collapse is insufficient" — independent convergence with C131 ⭐ banked
- **PhyLatent: Learning Dynamics-Relevant Representations for JEPA World
  Models** — arXiv **2608.05720** (6 Aug 2026). **PUBLISHED, BANKED (library key
  `2608.05720`, tag `anti-collapse`)**.
- **Finding:** "preventing global latent collapse does not ensure that a
  representation preserves physical states and action consequences"; names
  THREE dynamics-collapse modes invisible to any rank statistic — *physical
  invariance collapse*, *physical identifiability collapse*, *counterfactual
  dynamics collapse* — and fixes them with state grounding, future-representation
  alignment, static-visual invariance, counterfactual branch separation and
  latent denoising. OGBench-Cube failure rates 15.60/6.71/8.41 % → 7.53/0.95/
  4.62 %; MPC success 70.0 → 78.1 %; TwoRooms 81 → 98 %.
- **Impact:** an Aug-2026 paper arrives, independently, at our
  **rank-AND-decodability** doctrine (C131: v1-era rank 24.93(σ)/7.59(σ²) with
  zero environment interpretation). Our ladder's second gate is externally
  validated — and PhyLatent hands us a finer decomposition of "decodability".
- **What it would change:** the v7-tiny ladder gains a third probe —
  **counterfactual separation** (do different actions from the same state
  separate in latent space?) — the one collapse mode our current probes
  (speed/yaw/d_ego, detection AP) cannot see. Candidate v7 o-term; must pass the
  deliberate-regression arm before it is a gate.

### F2.2 — Prediction horizon: SCHEDULE it, don't fix it
- **Beyond the Next Step: Variable-Length Latent World Models for Long-Horizon
  Planning (VLWM)** — arXiv **2606.21775** (Jun 2026). **PUBLISHED (abstract
  verified); NOT banked — secondary.**
- **Finding:** a single predictor forecasts over variable action-sequence
  lengths, trained with a **curriculum that progressively expands the horizon**
  ("stabilising optimisation from short-range dynamics to long-range
  prediction"); **+13 % average over LeWM**, largest on long-planning tasks.
- **Impact:** explains the H-RANK-7 mechanism from the other side — v6's fixed
  o5-k=6 asked for long-horizon prediction before short-horizon dynamics
  existed, and a static latent is the cheapest solution to that ask.
- **What it would change:** the answer to "what after k=1?" is not k=6 — it is
  a **k-curriculum stage** in the v7 plan (S-W/S-T ladder), gated at each step
  by val participation + decodability. H-RANK-10's 30k trajectory tells us
  *when* the short-horizon base is learned.

### F2.3 — Temporal structure needs its OWN channel (the frame-stacking mechanism)
- **You Don't Need Strong Assumptions: Visual Representation Learning via
  Temporal Differences (TDV)** — arXiv **2606.15956** (Jun 2026). **PUBLISHED
  (abstract verified); NOT banked — secondary.**
- **Finding:** jointly train an image encoder and a **motion encoder** so that
  `z_t + m_t = z_{t+1}`; relies only on "the past causes the future", no
  augmentation/masking; **matches SOTA dense-task recipes**; removing the motion
  encoder and relying on temporal invariance alone **fails**.
- **Impact:** a mechanism-level account of H-RANK-8 and our measured lag-1 Δz
  autocorrelation of −0.075: when three frames are stacked into channels the
  temporal difference is folded into a spatial encoder that has no reason to
  keep it, and Δz becomes noise-like. TDV gives the difference an explicit
  additive latent instead.
- **What it would change:** (a) H-RANK-8 (n_stack=1) gets a mechanism and should
  run; (b) PREREG_E_ENC_3WAY gains a candidate 4th arm — *frozen spatial encoder
  + trained motion encoder* — which is cheaper than fine-tuning and attacks the
  exact deficit (speed R² +0.147 frozen vs +0.0025 trained, H-ENC-1).

### F2.4 — Theory frame (already banked, now cited)
- **A Generalization Theory for JEPA-Based World Models** — arXiv **2606.27014**
  — **BANKED earlier (key `2606.27014`, title placeholder in the index; no
  cited-by until now)**. JEPA pretraining risk ≡ matrix factorisation of the
  conditioned co-occurrence matrix, tying pretrain risk to downstream planning.
  Impact: the theoretical reason a spectral-mass statistic (participation)
  should track planning utility. Change: related-work anchor for the
  elimination-tree paper; fix the index title on next `--reindex`.

**Zeitgeist read:** LeWM (banked) → Sub-JEPA (banked) → VLWM → PhyLatent form a
coherent 2026 anti-collapse literature, and our elimination tree (weight ✗,
term-count ✓, conditioning ✗, capacity ✗, k=1 ✓) is a publishable contribution
to it **as it stands** — agenda item 1's "publish our elimination tree" is ripe.

## Ideation — our own hypothesis

**H-RANK-17 (OPEN, proposed).** *An explicit additive motion latent removes the
static-latent shortcut by construction.* Adding a TDV-style decomposition to
the two-term champion — predictor outputs `m_t = f(z_t, a_t)` and the L1 target
is `z_t + m_t ≈ z_{t+1}` with `m_t` regularised toward non-zero spectral mass —
will raise val participation and EM-vs-HOLD **beyond** the k=1 two-term arm,
because HOLD (`m_t = 0`) stops being a low-loss solution once the motion
channel is the only thing the loss can reward. Corollary (H-RANK-8 coupling):
the gain is larger at n_stack=1 than at n_stack=3.

- **Cheapest discriminating experiment:** ONE v7-tiny arm (≈29 min on Thor
  after the production run, or the dev-box 4060 at reduced batch) vs the
  two-term champion on the parity corpus; readouts val participation, EM vs
  HOLD, decodability (speed/yaw/d_ego), with the constant control, the pixel
  floor, and printed n/d. Run through `/TanitAD_ValidateAIDesign`.
- **Outcome A (participation and EM both up, decodability not down):** the
  motion latent enters the v7 design; H-RANK-8 is re-run with it.
- **Outcome B (no gain or decodability drops):** the two-term objective is
  already shortcut-free and Δz noise is an encoder property — the encoder
  question (F2.3's 4th arm) takes priority over the objective.

## Transfer note (charter §7)

→ **TanitAD_TrainingFlyWheel / Master Mind (design owner):** (1) k-curriculum
stage for v7 instead of a fixed k (F2.2); (2) counterfactual-separation probe as
a ladder gate candidate (F2.1); (3) H-RANK-17 as the next v7-tiny arm after
champ30k reads out. Accept/reject with reason requested.

## Evidence discipline

Programme numbers above are INHERITED (register/brief) and not re-verified;
paper numbers are PUBLISHED from verified abstracts; only the banked key is
registry-admissible. No number here enters MODEL_REGISTRY.md.
