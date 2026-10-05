# SFT-3 result, stage 1: arm A PASSES, arm B does not; arm A goes to stage 2 (navtest)

> Pre-registration: `eval/PREREG_SFT3.md`.
> Raw: `raw/2026-10-04-sft3/sft.log` (md5 `9116231f…`, the pod's full event log) and
> `raw/2026-10-04-sft3/pod_chain_gate_md5.txt` (chain log, PREFLIGHT_PASS token, checkpoint and code md5s).
>
> **Tier:** stage 1 is a label-only held-out read, not navtest and not closed loop. The truth is the teacher's labels
> (v5: teacher calculators with the NAVSIM drivable area) on the 3,137 held-out sets.
>
> **Interval:** paired log-cluster bootstrap over the 24 held-out logs, 10,000 resamples, seed 20260927. It answers
> *another draw of logs* only. There is one fine-tune seed, so training variance is unmeasured (H-ESTIM-SEED-1).

## Verdict (the registered rule: an arm passes iff its lower bound > 0 at the final update)

Final read, update 2,733 of 2,733 (`ZZEXIT 0`, 8.84 h). Base pick 84.84; random 73.12; best of 64 97.60.

| arm | pick | vs base ×100 [95 % CI] | picks = 0 | AUC NC / DAC |
|---|---|---|---|---|
| base (deployed scorer) | 84.84 | – | 169 | 0.884 / 0.745 |
| **A: per-component BCE (control), lr 1e-4** | 85.40 | **+0.56 [+0.08, +1.14] → PASSES** | 161 | 0.882 / 0.746 |
| B: BCE + expected-score objective (τ 0.1) | 85.14 | +0.30 [−0.38, +1.21] → does not pass | 146 | 0.869 / 0.743 |

MEASURED, `raw/2026-10-04-sft3/sft.log`.

Only A passes, so **A goes to stage 2**. Checkpoint: `model_sft_A.pt`, md5 `0ab46127…`, copied to the dev box and verified.

### Reported, not gating
- **Selection × the direction head** (`x_ddc`):
  - A: +0.30 [−0.21, +0.86];
  - oncoming picks: −0.35 pp [−0.88, −0.04] vs base.
- **Trajectory of A** across the 7 reads after step 0 (updates 400 to 2,733): +0.25, +0.08, −0.01, +0.10, +0.38, +0.50,
  +0.56. It rose over the last third of the cosine schedule.
- **B:** it had the fewest zero picks at every late read (146 vs 169), but its mean never separated.

## What this does and does not show

- **The objective did not help.** The passing arm is the CONTROL: the same per-component BCE the model was trained with.
  The expected-score objective, the hypothesis SFT-3 was registered to test, did not pass.
- **What A shows (ESTIMATED reading, not yet a claim):** one more epoch of scorer-only training at lr 1e-4, on the
  model's own on-policy bank with the trunk frozen, lifts held-out selection. SFT-1's identical arm at lr 3e-5 read
  +0.01 [−0.20, +0.25] at its last read. The lever is the scorer's training budget or learning rate, not the loss.
- **Not a navtest result.** Stage 2 (`eval/stage2_sft.py`) decides:
  - checkpoint in the unchanged seam (route fix ON, A7 ON, navsim_v1);
  - Amendment 5's 923 confirmation tokens, against the deployed system's own scores (85.61 PDMS on those tokens);
  - ADOPT iff the paired lower bound > 0 and no longitudinal or lateral component separates adversely.
- **A single seed.** A claim that this LEVER moves the metric needs a training-seed replicate (CLAUDE.md,
  H-ESTIM-SEED-1). If stage 2 adopts, that replicate is the next arm.

## Stage 2 (navtest, dev box), 2026-10-05: arm A is NOT PROVEN, so it is not adopted

Setup: `eval/stage2_sft.py`, unchanged seam (route fix ON, A7 ON, navsim_v1), Amendment 5's 923 confirmation tokens over
43 logs. The arm's scores are compared with the DEPLOYED system's own scores on the same tokens (the adopted route-fix
run). Raw: `raw/2026-10-05-sft-stage2/sft3_A/` (`result.json`, `score_status.json`, families of both).

**Gates: all pass.**
- Harness PASS: 923 / 923 tokens, 0 failed, C1 0.0.
- Seam: 923 rows, 0 misses, frame control 7e-15 m.
- Consistency: the 517 tokens whose executed poses equal the deployed poses score identically (0 differ).

| | deployed | arm A |
|---|---|---|
| PDMS on the 923 tokens | 85.61 | 85.40 |
| paired Δ ×100 [95 % CI] | – | **−0.20 [−1.30, +0.92]** |
| better / worse / tied | – | 148 / 126 / 649 |
| picks scoring 0 | 56 | 55 |

**Four families** (families6, episode-cluster bootstrap over the 43 logs):

| family | deployed | arm A |
|---|---|---|
| longitudinal: speed MAE | 0.857 m/s | 0.886 m/s |
| longitudinal: along-track MAE | 1.226 m | 1.269 m |
| lateral: heading MAE | 2.08° | 2.11° |
| lateral: cross-track MAE | 0.325 m | 0.331 m |
| lateral: curvature MAE | 0.0207 1/m | 0.0212 1/m |
| tactical: lateral-decision κ | 0.848 | 0.854 |
| tactical: longitudinal-decision κ | 0.513 | 0.503 |
| strategic | UNAVAILABLE in NAVSIM (no strategic decision is scored) | same |

None of the 21 longitudinal / lateral components separates adversely. Every point estimate is slightly worse.

**Verdict by the registered rule: NOT PROVEN** (lower bound ≤ 0, upper bound ≥ 0). Arm A is not adopted, and the
deployed scorer stays.

### What it means, and the next lever (Rule Zero)

- **The held-out gain did not transfer.** Arm A's +0.56 on held-out came from a truth built from the TEACHER'S labels.
  On NAVSIM's own scoring, the same checkpoint reads −0.20 [−1.30, +0.92].
  - This is the label mismatch `REVIEW_7_GAP_TO_PAPER.md` diagnosed, now measured end to end.
  - Optimising selection against the teacher's labels does not improve selection against PDMS.
- **The seed replicate is moot.** It would only have been needed for an adopted arm.
- **Next lever: SFT-4** (`eval/PREREG_SFT4.md`), already chained on the pod.
  - It trains and reads on the paper's own labels: NAVSIM PDM targets of the executed plans. The relabeller was
    validated against NAVSIM's real cache (argmax identical on 300 / 300 tokens).
  - Its stage-1 truth is therefore the metric navtest scores.
  - It starts after the PDM training labels finish, around 06:45 UTC, plus the lane window.
