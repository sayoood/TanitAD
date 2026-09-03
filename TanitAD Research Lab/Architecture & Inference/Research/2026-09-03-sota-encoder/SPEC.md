<title>SPEC - SOTA pass: encoder quality - frozen vs fine-tuned vs shaped (theme 2 of 4)</title>

# SPEC - E-ARCH-SOTA-ENC-1: frozen, fine-tuned, or shaped encoders for control-relevant latent prediction — what is MEASURED?

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-sota-encoder/` · Research Lab overnight pass 2026-09-03 · literature only · 0 GPU
**Written BEFORE any primary of this pass was opened.** Staged in the same turn it was written.
**Trigger.** v7f design freeze — the trunk-freeze decision is escalated design constraint #2 (`LAB_BACKLOG` GS block, 2026-09-02); `LEDGER_A2_jepa.md` entry 2026-09-02-02 (LDAD is encoder-shaping; nothing to shape on a frozen trunk); `LAB_BACKLOG` rows GS-2 (REVISED), GS-10, 10, 13.

## 1. Our MEASURED state

| fact | value | source (class · tier) |
|---|---|---|
| trained tiny trunk at parity (`rdw8p30k`) | participation 25.58 / 26.96; predictor nrmse **0.7903** (beats constant); `n_agents` **−0.0180** | `MODEL_REGISTRY.md` §13.0 / §13.0d · MEASURED · T0 |
| frozen distilled trunk at parity (`splitp30k`) | participation 6.38 / 7.63 (4× lower); nrmse **0.8416** (beats constant, worse than trainable); `n_agents` **+0.3881** (t 28.21) vs frozen DINOv3 **+0.2754**; `lead_range_m` **−0.1611** (worst of any arm); `d_ego` −0.0399 below constant | §13.0d · MEASURED · T0 |
| the frozen predictor collapse at 130 clips was a small-data artefact | `splitfrz10k` nrmse 4.1757 → `splitp30k` 0.8416 at parity | §13.0d · MEASURED · T0 |
| one-variable freeze on the postrain recipe | drift 0.3905 but nrmse +14.6 % ⇒ DEGENERATE; **v7 encoder stays TRAINABLE** | E-DEC-64 · MEASURED · T0 |
| no arm is action-sensitive, frozen or trained | 0/32 arms move nrmse > 0.1 % under action shuffle; largest response 11.6 % < a 10 % latent nudge (17.7 %) | §13.0c census (E-DEC-33) · MEASURED · T0 |
| the encoder DISCARDS pixel information that predicts its own latent's future change | raw-pixel marginal over `z_t` +0.0096 (t 5.11); tokens alone NEGATIVE (t −3.95) | E-DEC-63 · MEASURED · T0, arm `rdw8p30k` |
| fallback teacher | DINOv3 beats V-JEPA2 9/9 on the static-decodability rig | E-DEC-68 · MEASURED · T0 |
| refav1 = frozen DINOv3 trunk, fp8 features, targets frozen (collapse-proof) | no registry row exists (0 hits at 10 patterns, two probes) | INHERITED — readiness report §D |
| REF-A frozen-encoder ceiling: 2.14 m plateau, speed R² 0.61 | INHERITED — `MODEL_REGISTRY.md` §2 not re-read in this pass |
| LDAD (Delta-JEPA) shapes ENCODER geometry: `Δz` is built from two encoder outputs | INHERITED — `LEDGER_A2_jepa.md` 2026-09-02-02, from the paper's figure; to be re-verified from the banked PDF here |

## 2. Questions

- **Q-E1.** Which primaries publish a **frozen vs. fine-tuned vs. shaped** encoder ABLATION for a latent world model, and on which metric (planning success, rollout error, decodability)?
- **Q-E2.** Does any primary show a frozen DINOv2/v3 trunk SUFFICES for control-relevant prediction — and under what readout / predictor (patch tokens vs. CLS, attention pooling, trained adapter)?
- **Q-E3.** Is there MEASURED evidence that fine-tuning the encoder for prediction CORRUPTS what it carries ("observer effect", over-specialisation) — i.e. an argument FOR freezing that is not merely stability?
- **Q-E4.** Do encoder-shaping objectives (inverse-dynamics regularisation, LDAD) have a frozen-compatible form (a trainable adapter / bisimulation encoder on top of frozen features) with published numbers?
- **Q-E5.** What do the field's numbers PREDICT for ours: should a frozen DINOv3 trunk with a trained predictor (refav1) reach action-sensitive prediction, and should a trained tiny trunk (v7) lose decodability?

## 3. Hypotheses — both outcomes committed

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| **H-SOTA-E1** | ≥ 1 banked primary reports a frozen-vs-fine-tuned ablation with a stated effect on an action-relevant metric | its direction and size become the PRIOR for the v7f trunk decision; state whether it agrees with E-DEC-64 (freeze = degenerate) | the field asserts the trade-off without an ablation ⇒ E-DEC-64 is the only measured datapoint we have, and the v7f decision rests on it alone (say so) |
| **H-SOTA-E2** | Frozen pretrained trunks are shown sufficient for planning-quality prediction WHEN the readout/predictor is trained on patch tokens | refav1's frozen-DINOv3 design is field-supported; its action-insensitivity (if it appears) is a PREDICTOR/TARGET problem, not a trunk problem | the field shows frozen trunks fail on fine geometry / ego kinematics ⇒ refav1 inherits a known ceiling and the paper must say so |
| **H-SOTA-E3** | Fine-tuning for latent prediction measurably corrupts physical decodability (an "observer effect") | our E-DEC-63 "encoder discards pixel-predictive content" and the frozen arm's higher `n_agents` have a published mechanism; a PARTIAL/late unfreeze arm is the field-supported compromise | no such evidence ⇒ the trainable choice stands on E-DEC-64 alone |
| **H-SOTA-E4** | A frozen-compatible shaping route exists (adapter on frozen features trained with an IDM/LDAD-style loss) with published numbers | GS-2's "inapplicable as published" is narrowed: the LDAD line is reachable on refav1 through an adapter, and an arm is priced | none exists ⇒ GS-2 (REVISED) stands: LDAD is inapplicable on a frozen trunk, and the `m_t` slot is the only route |

## 4. What would change our plan

1. Evidence for H-SOTA-E3 A ⇒ the v7f prereg records a partial-unfreeze arm (FS-2 asymmetry: cheap before the geometry freezes).
2. Evidence for H-SOTA-E4 A ⇒ an adapter-LDAD arm on refav1's banked fp8 features is priced (0 trunk compute).
3. Evidence for H-SOTA-E2 B ⇒ refav1's baseline role is re-scoped in the paper before any number is quoted.

## 5. Method and admissibility

As theme 1: banked PDFs only (tag `sota-2026-09-03-encoder`), ≥ 3 primaries read in full, every number stamped, agreement/disagreement with our measured state stated explicitly, cheapest discriminating experiment with both outcomes committed.

## 6. Falsifiers

- H-SOTA-E1 B is refuted by any banked ablation table with a frozen and a fine-tuned row on the same metric.
- H-SOTA-E4 B is refuted by any primary training an inverse-dynamics/LDAD-style loss through an adapter on frozen features with a reported effect.
