# P7 head-veto read: VH-NULL. The drivable-area head, used as a veto or as a score term, does not improve P7's pick; DAC headroom is real (oracle +1.64) (MM, 2026-11-17)

- **Pre-registration:** `VETO_PREREG.md`, written before any P7 proposal score was read (agent, 2026-11-17 ~00:39 local). The chain ran unchanged (`run_chain.cmd`): `score_p7_cols.py score`, then `table`, then `veto_read.py`, finishing with `chain.DONE` at 02:09.
- **Population:** the 1,058 C3 navtest tokens over 123 logs. All 64 P7 proposals per token were scored with the unchanged harness `score_e.py`, pins verified (21 files), giving `table_p7.npz`, which is banked on HF.
- **Estimator:**
  - The decision values are on the stratified navtest-population estimator (`strat_delta_x100`).
  - Intervals come from a paired log-cluster bootstrap, 10,000 draws, seed 20260927. It answers "another draw of logs" only, with one head seed and one P7 seed.
  - Tier: T1 (the plans are P7's own); the head is camera-derived, so it is admissible at inference.

## Gates (all PASS, `gate_table.json`, `veto_read.json` gates)
- **G1:** the table is all valid and finite.
- **G2:** the table at P7's pick equals the banked P7 navtest CSV, max abs diff 0.0 on PDMS and on every sub-score.
- **G3:** arm A reproduces the dump's pick on 1,058 / 1,058 tokens.
- **G4:** the zero control gives B = A and C = A.
- **G5:** the head's tokens equal the table's tokens.
- **G6:** the mutation arm's worst survivor reads -0.488 PDMS, so the instrument can see harm.

## Result (MEASURED; PDMS x 100 deltas vs A, stratified)
| arm | delta [95 %] | reading |
|---|---|---|
| B: head veto (footprint SDF < -0.25 m, fallback A) | **-0.117 [-0.602, +0.253]** | not separated |
| C: score + 0.5 x inside-share | **-0.103 [-0.557, +0.383]** | not separated |
| N_B: B with shuffled maps (no fixed points, no same log) | -1.825 [-2.761, -0.938] | the content matters: a wrong map hurts |
| N_C: C with shuffled maps | -0.523 [-1.013, +0.059] | |
| O: oracle DAC veto (reads harness DAC, never deployable) | **+1.640 [+0.902, +2.531]** | DAC headroom is real |

**Outcome row (pre-registered): VH-NULL.** Neither B nor C is separated above 0, and neither is harmful.
- H2 feed-on is NOT justified by this read.
- O - A has a lower bound of +0.90, above +0.3, so the DAC headroom is not the limit. The head's quality is.
- **Pre-registered next lever:** the head-quality arm, the 1 m grid / `--w-field` arm (D-AUX-4). That is consistent with D-REFE-P7-MECH-1: the head's IoU is 0.766 on the DAC-zero tokens vs 0.858 elsewhere.

**Reported only (post hoc; never a decision; tuning on the scored tokens is inadmissible):**
- B margin -1.0 m: +0.133 [+0.002, +0.380].
- B margin 0.0 m: -0.794 [-1.599, -0.088].
- C with lambda 0.25 / 1.0 / 2.0: -0.08 / -0.12 / -0.10, none separated.
- B on the head's SDF channel: +0.112 [-0.195, +0.518].

⚠️ The one positive margin is a post-hoc pick out of four margins. It is not quotable as a result, and it would need its own pre-registration on held-out tokens.

## MM descriptive add-on (0-GPU, same table; MEASURED, unweighted C3 means, no interval): does P7 pick too-long plans?
The question comes from D-REFE-STRATFAM-1 and D-REFE-P7-S3FAM-1, which found that P7's plans run 1-10 % further than the logged human.

**No.** Against the harness-best of its own 64 proposals, P7's pick is **shorter**:
- mean arc length 22.76 m vs 24.95 m (ratio of sums 0.912);
- on the 659 tokens where the pick is not the best, the pick is longer in only 12.4 % (mean -3.21 m).

Sub-scores, pick vs best of 64:

| sub-score | pick | best of 64 |
|---|---|---|
| EP | 0.738 | 0.917 |
| DAC | 0.878 | 0.966 |
| TTC | 0.896 | 0.986 |
| NC | 0.955 | 0.995 |
| DDC | 0.951 | 0.943 |
| comfort | 1.000 | 1.000 |

⇒ The overshoot relative to the human is NOT a PDMS lever (shrinking plans would cost EP). The pick-vs-best gap is a **selection** gap, led by EP, then DAC and TTC. That is the gap P5-C (C1's optimised-driver labels) targets.

## Files (hold `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/veto/`)
- `VETO_PREREG.md`, `veto_read.py`, `score_p7_cols.py`, `run_chain.cmd`;
- `chain_score.log`, `chain_table.log`, `chain_read.log`;
- `gate_table.json`, `veto_read.json`, `SCORE_STATUS.json`, `HANDOFF.md`;
- `table_p7.npz` (8,165,128 B), banked on HF `Sayood/tanitad-refe-scorer1p6` `evidence/2026-11-17-p7-headveto/`.
