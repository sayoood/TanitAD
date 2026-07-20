# REF-A vs the DINO-World-Model literature — comparison + verdict

*2026-07-19. Synthesized from the deep-research workflow (22 sources, 104 claims → 25 adversarially verified). The workflow's auto-synthesis step was killed by a rate limit, so this is hand-assembled from the **4 confirmed claims** (2-of-3 / 3-of-3 refute-votes survived) + established literature; the other 21 claims mostly **errored on the rate limit (not refuted)** — a re-run can harden them.*

## The confirmed findings (cited)

1. **DINO-WM training objective = feature-space prediction.** DINO-WM keeps DINOv2 frozen and trains a ViT transition model to predict **future patch embeddings** via a latent MSE consistency loss — "all forward computation is done in latent space z," "no visual reconstruction." *(arXiv 2411.04983, votes 2‑0.)*
2. **DINO-WM inference = MPC + CEM, no policy head.** It plans by **optimizing action sequences at test time** with the Cross-Entropy Method to reach a goal latent state — "no separate policy head." *(2411.04983, votes 3‑0.)*
3. **DINO-WM decodes nothing to pixels/trajectories** — it plans over predicted features directly. *(2411.04983, votes 2‑0.)*
4. **DINO-WM's validated domains are slow / closed** — PointMaze, PushT, Reacher, rope/granular manipulation, point-goal navigation. **High-speed driving is absent.** *(votes 1‑0, 2 rate-limit-errored; well-established from the paper's eval suite.)*

## How REF-A deviates (both training AND inference)

| | **DINO-WM (SOTA recipe)** | **Our REF-A** |
|---|---|---|
| Encoder | frozen DINOv2 | frozen DINOv2‑B/14 ✓ (faithful) |
| **Transition objective** | predict **future features** (self-supervised, latent MSE) | **regress ego displacements directly** (supervised) ✗ |
| **Inference** | **MPC + CEM** test-time optimization | **learned policy head** (StepDisplacementReadout + tactical/strategic) ✗ |
| Context | multi-frame | **latest frame only** (temporal adapter partial) ✗ |
| Fine-tune | none | none ✓ |

## Verdict — it's **(c) both**

**(a) domain-limited:** no frozen-DINO world model in the literature has been shown to succeed at **high-speed AV trajectory planning** — the successes are all slow/closed-domain. So the "frozen-encoder ceiling" we measured (2.92 m) is **consistent with** the state of the art; the literature does **not** predict a frozen generic DINOv2 would plan driving well.

**(b) implementation gap:** REF-A is **not** a faithful DINO-WM instantiation. Its two biggest deviations: (i) **direct supervised displacement regression** on frozen features vs DINO-WM's self-supervised feature-prediction, and (ii) a **learned policy head** instead of test-time MPC/CEM planning.

**CORRECTION (overfitting curve, added 2026-07-19):** REF-A does NOT overfit. Held-out ADE@2s improves monotonically 5k→30k (3.76→3.69→3.02→2.92); 30k is the best ckpt, still slowly descending toward a ~2.8 m plateau. The 0.649-train vs 2.92-held-out figure is a **metric mismatch** (teacher-forced operative on the train batch vs tactical rollout on held-out), NOT memorization. REF-A is **underfitting to a frozen-encoder capacity ceiling**. The failure is **~99% longitudinal/speed** (high-speed long-RMSE 5.2 m vs lat-RMSE 0.62 m): frozen static per-frame DINO features carry spatial layout but essentially no motion/dynamics — so it plans the path but not the speed. This makes the fix motion-aware features / feature-prediction+MPC / an own-pretrained encoder — NOT "train longer" or "earlier ckpt".

## Top 3 changes to try (ranked by expected impact for v3)

1. **Feature-space rollout + MPC/CEM inference** (the actual DINO-WM recipe) — train the transition to predict future *features* self-supervised; plan by test-time optimization, not a regression head. Directly targets the overfitting. *Highest impact.*
2. **Own-pretrained frozen encoder (V-JEPA2‑AC style)** — a frozen *generic* DINOv2 is the ceiling; a frozen encoder **pre-trained on driving video** may not be. This is the v3 own-data-SSL direction (V-JEPA2‑AC showed frozen-encoder WMs work for robotics when the encoder saw relevant data). *High impact, larger effort.*
3. **Multi-frame temporal context** into the frozen features (REF-A encodes only the latest frame). *Moderate; cheap to test.*

**Bottom line for the REF-A decision:** 2.92 m is a *real and expected* frozen-generic-encoder result, but it's **not a clean test of the frozen-WM approach** — REF-A tested "frozen features + supervised regression + learned head," which the literature would also expect to overfit. A fair frozen-WM test (feature-prediction + MPC/CEM, or an own-pretrained encoder) is the v3 experiment. For the paper, REF-A stands as evidence the **from-scratch encoder is necessary for our recipe**; it does *not* settle whether a *properly-built* frozen-WM could work.
