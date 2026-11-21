# SFT-1 — scorer-only fine-tune of REFe's final model (pre-registration)

> Written before any SFT-1 training step. It is landed in git before the run's first held-out read is looked at.
> PI request 2026-10-04: *"How can we improve the scorer? start the scorer fine-tune on the pod."*

## Why

On the full navtest (`eval/RESULT_NAVTEST_FINAL.md`), 983 of 12,146 picks score exactly 0, about 6.7 points of the
mean. On W3's 200 tokens, where all 64 hypotheses are harness-scored, **14 of 15 zero picks had a hypothesis scoring
≥ 80**. These are selection failures. The scorer is trained per component (BCE), which calibrates; the planner needs
a within-scene ranking. Drivable area is the dominant failure (721 of 983).

## Design (fixed now)

- **Base:** `model_final.pt` (md5 `b54773f8…`), built and loaded exactly as the planner does.
- **Trainable:** ONLY `score_q_mlp`, `score_dec`, `score_head`. The scorer reads the trajectories and the visual context
  detached, so the 64 proposals stay bit-identical and only the choice among them moves.
- **Data:** train.py's own `TargetBank` + `OnPolicyBank` over the run's on-policy sets (the labels training used). Only
  samples with a complete labelled set are used, and none from the 24 held-out logs.
- **Arms, trained in ONE pass over the same batches (paired by construction):**
  - **A (control):** the training loss, per-component BCE.
  - **B:** BCE + 0.3 × ListNet over each 64-set. p_i ∝ the planner's navsim_v1 aggregate of the sigmoid scores;
    q_i ∝ exp(true navsim_v1_i / 0.1).
- **Settings:** AdamW, lr 3e-5 (50-update warm-up, cosine to 0), weight decay 0.01, batch 8 × accumulation 8, 1 epoch,
  seed 0, bf16.
- **Launch gate (`--preflight`, must PASS):**
  - G1 the trainable set is exactly the scorer;
  - G2 every trainable tensor gets a finite non-zero gradient, every frozen one none;
  - G3 the base, A and B copies equal the model's own scorer at step 0;
  - G4 B's loss at lam 0 equals A's, and differs at lam 0.3;
  - G5 a perturbed scorer, saved, reloads through the planner's loader bit-identically and differs from base;
  - G7 the held-out sets are disjoint from training;
  - plus the deliberate regression: unfreezing `traj_head` must turn G1 RED.

## Stage 1 — held-out (pod, label-only)

- **Data:** the 3,137 held-out sets: the final model's own proposals on 24 navtrain logs (sha1 % 40 == 0), labelled v5.
- **Primary:** pick(arm) − pick(base) of the true navsim_v1 score of the selected hypothesis, ×100. Paired log-cluster
  bootstrap over the 24 logs (10,000 resamples, seed 20260927).
- **Reported:** random, best, skill, picks scoring 0, within-set AUC of NC and DAC.
- **An arm PASSES stage 1** iff its lower bound > 0 at the final update. If both pass, B goes forward only if
  pick(B) − pick(A) > 0 on the same read; otherwise A goes forward.

## Stage 2 — navtest (dev box; never on the pod)

- The stage-1 winner's full checkpoint replaces `model_final.pt` in the unchanged seam (route fix ON, A7 ON, navsim_v1).
- **Confirmation tokens:** Amendment 5's 923 (43 logs, disjoint from W3's 200).
- **ADOPT** iff the paired PDMS lower bound > 0 (log-cluster bootstrap) AND no longitudinal or lateral family component
  separates adversely.
- **REFUTED** iff the upper bound < 0.
- Otherwise **NOT PROVEN**.
- The full 12,146-token navtest is then reported for the adopted scorer.

The interval answers *another draw of logs* only: one fine-tune seed, deterministic inference.
