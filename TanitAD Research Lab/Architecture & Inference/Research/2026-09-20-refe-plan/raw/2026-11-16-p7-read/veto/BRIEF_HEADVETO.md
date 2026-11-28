# BRIEF: the 0-GPU head-veto read on P7's 64 proposals (does the drivable-area head, used as a veto or score term, improve the pick?). Agent: Sonnet. MM, 2026-11-17

Read `Project Steering/AGENT_OPERATING_STANDARD.md` first.
- **Binding:** stage, never commit or push; end with a manifest.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/veto/`. **Scratch:** `F:/tanitad_scratch/p7veto/`.
- **Read first:** `p7_next/headiou/P7_HEAD_IOU.md`, `p7_next/headiou/HANDOFF.md` and `p7_next/P7_MECH_READ.md`.
- **Code:** read it from the mirror `C:/Users/Admin/tanitad-push` (`git fetch -q origin agent/arch-inf-20260803`; FETCH_HEAD). `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## Why
P7's head IoU is 0.8225 (≥ 0.80), but the head does not feed the scorer, and P7 = P6 on PDMS. Before any training arm (H2 feed-on), measure offline whether the head's drivable-area prediction would have improved P7's pick among its own 64 proposals.

## Inputs
- **P7 proposals and scorer logits, all 12,146 navtest tokens:** `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full.proposals.npz`.
- **Head predictions, 1,058 C3 tokens, 3 seeds:** `p7_next/headiou/main/head_pred_H1-s*.npz`, also on HF `Sayood/tanitad-refe-scorer1p6/evidence/2026-11-17-p7-headiou`.
- **Ground truth:** the per-proposal harness scores (DAC and the PDMS sub-scores) for P7's proposals. FIND an existing table or instrument first: `PK/eval/instruments/transp_score.py`, `selection_quality.py` and the P6 G2 dump reads. If none exists, build the smallest one that scores the 64 proposals per C3 token with the harness (`TANITAD_NAVSIM_ROOT` metric cache, as `p7_read_chain.py` families uses). Estimate the CPU cost first, and if it exceeds ~4 CPU-h, STOP and report a plan.
- **Dev box:** CPU only, `CUDA_VISIBLE_DEVICES=-1`, TMP in scratch, serial. Free-commit guard: wait (in a script) while it reads below 4 GB.

## The read (pre-register it in `VETO_PREREG.md` BEFORE computing; both outcomes committed)
- **Arms, on the same C3 tokens:**
  - (A) P7's actual pick;
  - (B) a head veto: among proposals whose path stays inside the head's predicted drivable area (define the inside-fraction threshold in advance), take the scorer's best, else fall back to A;
  - (C) the scorer score plus λ · head drivable fraction, with λ fixed in advance (no tuning on the scored tokens);
  - (N) a control: the same veto with a SHUFFLED head map (tokens permuted), which must not help.
- **Endpoint:** the harness PDMS and DAC of the picked proposal. Paired log-cluster bootstrap over logs (`proxy_eval.boot`, 10k, seed 20260927). Report B−A, C−A and N−A.
- **Rule:** the H2 feed-on arm is worth a card iff B−A or C−A has a lower bound > 0 AND N−A does not.

Write `P7_HEADVETO_READ.md` (each number with its evidence class, estimator, n and artifact path) and `HANDOFF.md` (≤ 2k tokens).

## ⛔ Do not
- Tune a threshold or λ on the tokens you score.
- Commit, land or push. Train.
- Use the GPU, the pod or box 2.
- Edit existing instruments or frozen files.
- Print or copy secrets.
- Delete outside your scratch.
- Exceed ~150k context.

**Report:** at most 300 words, plus a manifest.
