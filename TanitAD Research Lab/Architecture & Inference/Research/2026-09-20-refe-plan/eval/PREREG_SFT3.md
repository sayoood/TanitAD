# SFT-3: train the selector on its EXPECTED true score (the exact form of RL for a one-step choice)

> Written 2026-10-04 before any SFT-3 step; landed in git before its first held-out read. Launch only on the PI's OK.
> PI 2026-10-04: *"Do we have other ideas to improve the selection process since you said the fan includes better
> trajectories? Should we go with RL post-training for the selector?"* and *"let's do the paper pure way"*.

## Why (MEASURED)

- **Headroom:** on navtest the best of the 64 hypotheses averages 98.33 PDMS, the pick 84.46 (LANE-1, `lane_result.json`).
- **Held-out sets:** best 97.60, pick 84.84 (SFT-1, 3,137 sets).
- **SFT-1, at updates 400, 800 and 1,200, moves nothing or hurts:**
  - A (BCE): +0.02, −0.10, −0.22 [−0.62, +0.14];
  - B (BCE + ListNet): −0.73, −0.92, −0.58 [−1.51, +0.45].
- **Why ListNet is the wrong objective:** ListNet matches the predicted distribution to a FLAT target (q ∝ exp(true / 0.1), where dozens of hypotheses score close to the best), so most of its gradient lands far from the choice that matters.
- **Why not standard RL:** the selection is a one-step decision whose outcome is KNOWN for all 64 actions (every set is labelled by the teacher's simulator). Classic RL (PPO/GRPO) would sample one action and learn from its reward: the same objective with sampling variance. The closed form needs no sampling.

## Design (fixed now)

- **Base and scope:** `model_final.pt` (md5 `b54773f8…`). Scorer modules only (`score_q_mlp`, `score_dec`, `score_head`); the 64 proposals stay bit-identical.
- **Labels: paper-pure.** The existing on-policy labels from the teacher's simulator (train v3, held-out v5; drivable area and comfort as decided by the PI on 2026-09-26). No NAVSIM direction side files.
- **Arms, one pass over the same batches:**
  - **A (control):** per-component BCE.
  - **B:** BCE + 1.0 × L_ER.
  - L_ER = −Σᵢ πᵢ Rᵢ, with π = softmax(log agg_v1(σ(s)) / 0.1) over each set and Rᵢ = agg_v1(labelsᵢ), the true score of hypothesis i (`expected_reward_loss`).
  - As τ → 0, π becomes the planner's own argmax.
- **Pinned by an analytic test:**
  - a uniform policy reads E[R] = 1/3 on R = [1, 0, 0];
  - the gradient raises the good hypothesis and lowers each bad one on the component it fails;
  - the direction output, outside navsim_v1, gets exactly 0;
  - a sharp policy on the good hypothesis reads E[R] → 1;
  - with equal outcomes the gradient is exactly 0.
- **Settings:** AdamW, lr 1e-4 (50-update warm-up, cosine to 0), weight decay 0.01, batch 8 × accumulation 8, 1 epoch (2,733 updates), seed 0, bf16, held-out read every 400 updates. Selection rule navsim_v1, unchanged.
- **Launch gate:** G1–G7 as SFT-1 (G4: B at λ 0 equals A bit-for-bit, and differs at λ 1). The deliberate regression is unfreezing `traj_head`, which must turn G1 RED.

## Stage 1: held-out (pod, label-only; 3,137 sets, 24 logs)

- **Primary:** pick(arm) − pick(base), on the true navsim_v1 score × 100.
- **Estimator:** paired log-cluster bootstrap (10,000 resamples, seed 20260927).
- **An arm PASSES** iff its lower bound > 0 at the final update. If both pass, B goes forward only if pick(B) − pick(A) > 0.
- **Reported:** random, best, skill, picks scoring 0, within-set AUC of NC and DAC, and the oncoming-pick share on the NAVSIM direction labels (yardstick only; not trained on).

## Stage 2: navtest (dev box; never on the pod)

- **Seam:** the winner's checkpoint in the unchanged seam (route fix ON, A7 ON, navsim_v1).
- **Tokens:** Amendment 5's 923 confirmation tokens.
- **ADOPT** iff the paired PDMS lower bound > 0 AND no longitudinal or lateral family component separates adversely.
- **REFUTED** iff the upper bound < 0.
- Otherwise **NOT PROVEN**.
- The full navtest and the four families follow for an adopted scorer.

## If it fails

The binding constraint is then the scorer's INPUT, not its objective (SFT-1 and SFT-3 together). The next levers:
1. a small trainable adapter on the visual context inside the scorer path, with proposals unchanged;
2. reward-weighted fine-tuning of the proposal head toward its own high-scoring hypotheses. That needs the PI's go-ahead: it trains the plan generator.

The interval answers *another draw of logs* only: one fine-tune seed, deterministic inference.
