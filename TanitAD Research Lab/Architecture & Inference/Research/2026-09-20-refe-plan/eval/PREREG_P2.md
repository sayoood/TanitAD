# P2: joint fine-tune, with the scorer's visual context un-detached, on the paper's PDM targets (pre-registration)

> Written 2026-10-05 before any P2 training step. It is landed in git before the first P2 read is looked at.
> PI 2026-10-05: *"I approve all your proposed levers, let use optimally the pod."*
> Lever P2 of `REVIEW_7_GAP_TO_PAPER.md` §4.

## Why (MEASURED / PUBLISHED)

**Selection is the bottleneck** (REVIEW_7 §2.4, full navtest). Of the same 64 proposals:
- best of 64 reads 97.07;
- the deployed pick reads 84.46 with the route fix (83.24 without);
- an oracle on the NC + DAC + TTC heads alone reads 95.16.

**Two defects in how the scorer is trained:**
1. **Its labels.**
   - The paper supervises its scorer with PDM targets (PUBLISHED, DriveZero Sec. 3; RETRACTION_LOG R30); ours used the
     teacher's calculators.
   - SFT-3 measured the consequence end to end: +0.56 on held-out against the teacher's labels became −0.20
     [−1.30, +0.92] on navtest. Re-read against PDM targets, the same arm would have failed (`eval/RESULT_SFT3.md`).
2. **Its gradient.**
   - The paper detaches only the candidate trajectories; REFe also detaches the visual context (`model.py:751`). So
     the score loss never shapes the visual features.
   - A re-ranker given NAVSIM's own labels but no images recovers only +1.12 [+0.50, +1.76] (REVIEW_7 §0.5), so the
     missing information is visual.

SFT-4 fixes defect 1 with the scorer alone. P2 fixes both.

## Design (fixed now)

- **Base:** `model_final.pt` (md5 `b54773f8…`), warm-started strictly (`train.py --init-from`). The optimiser and
  schedule start fresh.
- **Trainable:** exactly the final run's set: LoRA (rank 32, Q/V) and all heads, 18.99 M of 322.07 M parameters.
- **What changes:** the scorer's visual context is UN-detached (`--scorer-sees-trunk`), so the PDM loss reaches the
  LoRA. The candidate trajectories stay detached (the paper's detach).
- **Trajectory loss:** unchanged. WTA on the final run's teacher targets (`train_grow` bank, goal-augmented twins
  included; 189,888 tuples over 103,037 scenes).
- **Scorer loss:** BCE on the six NAVSIM PDM targets of the EXECUTED plans (`refe/onpolicy_relabel_pdm.py`, validated
  against NAVSIM's own cache with the argmax identical on 300 / 300 tokens).
  - The relabel must be complete: 174,912 served sets.
  - `--pdm-only`: the bank serves ONLY PDM-labelled sets, so no set mixes label sources. A sample without one keeps
    its trajectory loss and gets no scorer loss.
  - Scorer loss weight 0.1, as the final run.
- **Settings:** batch 4 × accumulation 64 (effective 256, as the final run); bf16 autocast, TF32, compiled trunk,
  2 workers; AdamW, lr **5e-5** (a quarter of the paper's peak, since it warm-starts a converged model), cosine to 0,
  no warm-up; weight decay 0.01; **3 epochs over scenes** (≈ 1,209 optimiser steps, ≈ 20 h at the final run's
  ~60 s / step); seed 0. Checkpoints every 20 min, resumable.
- **No lane head in P2.** The six PDM components are the paper's. The lane output is tested by SFT-4.

## Launch gate (must PASS; `refe/p2_gate.py`, `/workspace/data/refe_p2/gate/gate.json`)

- **G1 warm start = what the planner loads:** bit-identical trajectories and scores on real rows.
- **G2 the mechanism, as an analytic pair on one real batch, score loss only:**
  - un-detached → a finite NON-ZERO LoRA gradient;
  - detached → EXACTLY 0.0 on the LoRA (the deliberate-regression arm);
  - `traj_head` gradient exactly 0.0 in both.
- **G3 PDM sets only:** no served set lacks PDM targets, a row without one is masked out, and the same bank without
  `--pdm-only` MUST show sets without PDM targets (the mutation).
- **G4 the real trainer:** 2 optimiser steps with these flags; finite losses; `model_final.pt` reloads through the
  planner's loader; LoRA tensors moved; the frozen trunk is bit-identical.
- **Then** `train.py --preflight` on the real configuration must print PREFLIGHT_OK.

## Evaluation

**Stage 2 decides** (dev box, never on the pod), with `eval/stage2_sft.py`:
- the P2 final checkpoint in the unchanged seam (route fix ON, A7 ON, navsim_v1);
- Amendment 5's 923 confirmation tokens over 43 logs;
- against the deployed system's own scores on those tokens (85.61 PDMS);
- paired log-cluster bootstrap, 10,000 resamples, seed 20260927.

**Rule (committed now):**
- **ADOPT** iff the lower bound > 0 AND no longitudinal or lateral family component separates adversely.
- **REFUTED** iff the upper bound < 0.
- Otherwise **NOT PROVEN**.

On ADOPT, the full 12,146-token navtest and the four families follow. A training-seed replicate (H-ESTIM-SEED-1) is
then the next arm, before the lever is claimed.

**Reported, not gating:**
- the held-out selection read on the 3,137 held-out sets against PDM targets (the scorer ranking those sets' FIXED
  proposals; P2 changes the proposals, so only navtest is end to end);
- the per-epoch snapshots on the same 923 tokens (exploratory).

The interval answers *another draw of logs* only: one training seed, deterministic inference.

## Next arm, by outcome (Rule Zero)

- **ADOPT:** the seed replicate, then P6 (a paper-faithful full rerun, aimed above 94.55) builds on it.
- **NOT PROVEN or REFUTED:** P6 still runs, the paper's full chain from scratch. P2's training log (score loss, LoRA
  movement, held-out ranking) shows which half of the fix did not carry.
