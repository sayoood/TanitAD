# JEPA-like driving architectures with proven results — and what combines with ours

**Written** 2026-08-27 · **Author** Master Mind · **Requested by the PI** (*"search
if there are reference works with JEPA-like driving architecture proven with very
good results which can be combined with our achievements, ideas, findings and
particularly our multi-hierarchy concept"*).
**Evidence class:** PUBLISHED (all five primaries **banked**, sha256'd, Library
keys 74–78; per-paper numbers quoted from each paper's own reporting — cross-paper
tables are NOT reconciled here and must not be merged into one ranking without
reading the eval splits).

---

## 1. The direct answer: yes — three works, one per axis of our design

### ⭐ Drive-JEPA (arXiv 2601.22032, Jan 2026) — *the existence proof for our recipe class*

V-JEPA-pretrained ViT-L (**307 M** — at our sub-300M budget edge) on **208 h** of
driving video; masked **latent** prediction (no pixels), collapse handled by
**stop-grad + EMA target encoder**; then a *lightweight* query-based planner
distilled against a rule-scored trajectory vocabulary (8,192 k-means clusters,
EPDMS-filtered pseudo-trajectories alongside human ones).
**Results (its own reporting):** NAVSIM v1 **93.3 PDMS**, NAVSIM v2 **87.8
EPDMS**, Bench2Drive **64.52**; perception-free setting 89.0 PDMS.

**Why it matters to us:** it is the published proof that **a JEPA latent WM + tiny
planner drives at SOTA level with data and compute in our class** — no BEV stack,
no perception labels. Its collapse answer (EMA teacher) is the industrial-strength
version of what our two-term core approximates.

### ⭐ Latent-WAM (arXiv 2603.24581) — *closest published relative of our v7 line*

Encoder **distilled from a foundation model** (their SCWE), then a **causal
transformer predicting future latent states from visual+motion history** —
structurally our `distill_init` + predictor, matured. **89.3 EPDMS on NAVSIM v2,
perception-free** (their table).

### ⭐⭐ Hierarchical Planning with Latent World Models (arXiv 2604.03208, 2026) — *the empirical vindication of the multi-hierarchy thesis*

Two temporal levels, **each a next-latent predictor in a SHARED latent space**;
the long-horizon model's predictions become **subgoals for the short-horizon model
via latent matching — no task rewards**. Robotics/navigation domain, not driving.
**Result: 70 % real-robot success vs 0 % for single-level planning, at up to 3×
less planning compute.**

**Why it matters to us most of all:** it is the measured, published form of the
PI's thesis — *hierarchy is not a luxury; flat latent planning fails outright at
long horizon* — and its coupling mechanism (subgoal-by-latent-matching) is
information-disjoint by construction, i.e. compatible with our 2026-08-03
goal/situation-disjointness rule.

## 2. The supporting cast

| work | what it proves for us | banked |
|---|---|---|
| **World4Drive** (2507.00603, ICCV 2025) | an **intention level above a physical latent WM** — generate/evaluate/select multi-modal trajectories — a 2-level hierarchy already competitive on NAVSIM | ✅ |
| **JEPA LiDAR occupancy WM** (2602.12540) | **occupancy as the world-model target** works under JEPA training — exactly the "coarser, structured target" E-DEC-63 points v7 toward | ✅ |
| **SGDrive** (2601.05640) / ReCogDrive | scene→goal→action **three-level hierarchies at the top of NAVSIM** (87.4 PDMS, VLM-based) — the hierarchy premise wins even in the VLM lane; architecturally not our lane | PDF cached |
| **V-JEPA 2 / 2.1** (2025 / Mar 2026) | frozen-video-encoder + action-conditioned head trained on little interaction data; 2.1 sharpens **dense temporally-consistent features** — the strongest candidate teacher besides DINOv3 for our distill/fallback | known line |
| FF-JEPA (2606.09311) | latent planners for long horizon inside JEPA WMs | not yet |

## 3. The combination map — what we adopt, keep, and uniquely add

**Adopt (each fixes a measured defect of ours):**
1. **EMA-teacher latent target (Drive-JEPA's collapse core)** — an *external-ish*,
   non-collapsing target; candidate to test **against/with** our two-term core on
   the tiny ladder. ⚠️ Caveat: an EMA teacher is still model-generated — E-DEC-7's
   class boundary — so it enters ONLY as a gated tiny-ladder cell, judged by the
   E-DEC-63 absorption metric like every representational candidate.
2. **Trajectory-vocabulary distillation with simulator scoring (Drive-JEPA)** — a
   planner-side supervision source that needs no new labels; directly compatible
   with our Stage-C planner and our AlpaSim asset.
3. **Latent-subgoal coupling (2604.03208)** — the concrete mechanism for our
   strategic→tactical→operative interfaces: upper level emits a *latent* subgoal;
   lower level plans to match it. Information-disjoint, reward-free, and proven to
   beat flat planning 70-vs-0.
4. **Occupancy/free-space as a WM target (2602.12540)** — the E-DEC-63-recommended
   coarser target has published precedent under JEPA training.

**Keep (ours, already aligned with the field):** distilled-init trainable encoder
(Latent-WAM's shape), vision-only inference, the four-family + hold-action echo
instrumentation — **no published work we found runs an anti-echo control; our T1
discipline is ahead of the field, not behind it.**

**Uniquely ours (the gap in the literature = the claim):** none of the driving
works has a **>2-level hierarchy over a self-supervised latent WM with
information-disjoint levels**; 2604.03208 has the hierarchy but not driving;
Drive-JEPA/Latent-WAM have driving but are flat (one latent, one planner);
SGDrive's hierarchy is VLM-prompted, not latent-predictive. ⇒ **the PI's thesis —
multi-hierarchy reasoning/planning over a JEPA-style WM, demonstrated on driving
against REF-C/REF-D — remains unoccupied ground, and now has published
load-bearing precedent on both flanks.**

## 4. Consequences for the current plan (no changes executed — for PI review)

- The **representational arm note** gains a third candidate: **R3′ = EMA-teacher
  latent target** (Drive-JEPA core). It must beat R2 on the absorption metric to
  earn scale, same gates. *(Distinct from the refuted O7 weight sweep: EMA teacher
  ≠ fixed DINOv3 anchor.)*
- **Stage C/D design**: adopt latent-subgoal matching as the default inter-level
  interface, and trajectory-vocabulary distillation as the planner's supervision.
- **Fallback teacher bake-off**: DINOv3 vs **V-JEPA 2.1** dense features for the
  distill/fallback arm (A8) — one extra probe column, cheap.
- **Benchmark positioning**: NAVSIM PDMS/EPDMS is where this literature proves
  itself; our parity corpus proves *internal* claims. A NAVSIM(-style) eval is the
  external yardstick Stage D will eventually need — flagged, not scheduled (PI
  provisioning call).
