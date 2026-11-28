# RESULT C3: u400's zero tail is SELECTION-bound, not generation-bound — 86 % of zeroed tokens had a safe plan in the model's own 64

> **Pre-registration.** `eval/P6_GO_SIGNAL.md` §3 "C3", landed in `8cd1182` before any proposal was scored.
> **Operationalisation.** `raw/2026-10-11-c3-selection-headroom/OPERATIONALISATION.md`, together with the population, landed in `842eacf` before the read.
>
> **Tier.** T1: NAVSIM v1.1 PDMS, non-reactive pseudo-simulation.
> **Estimator.** Paired log-cluster bootstrap over 123 logs: 10,000 resamples, seed 20260927, percentile 95 %, both strata on the same log draw. It answers *another draw of logs* only. The model is one fine-tune seed and inference is deterministic, so training variance is not measured (H-ESTIM-SEED-1).
> **Evidence class.** Every number is MEASURED: `raw/2026-10-11-c3-selection-headroom/readout.json`, `readout_tables.md`, `gate_*.json`, `table.npz`, `proposals.npz`.
> ⚠️ **These are navtest tokens.** This is a diagnostic. Nothing was fitted on them, and no selection rule derived from them may be quoted as a result.

## Verdict (committed rule): **SCORER-BOUND**

The rule:
- INSTRUMENT FAILURE first: the fidelity gate fails, or random ≥ actual;
- else SCORER-BOUND if rescuable ≥ 50 %;
- else GENERATION-BOUND.

**Instrument gates: all PASS.**
- **Fidelity.** u400 was re-run on all 1,058 tokens and reproduces the banked full-navtest plans bit-identically: 1,058 / 1,058, max |Δ| = 0.0 m (`gate_fidelity.json`).
- **Table.** Each pick's harness score equals the banked full-navtest per-token score on 1,058 / 1,058, max |Δ| = 0.0 (`gate_table.json`).
- **Mix sanity.** The mix-weighted actual is 88.28, and the banked 88.25 lies inside its CI.

## The reads

| read | value [95 % CI] |
|---|---|
| **(a) rescuable share of the 529 zeroed tokens** (≥ 1 proposal with NC = DAC = 1) | **86.0 % [80.2, 91.7]** (455 / 529); bar 50 % |
| (b) actual PDMS ×100, re-weighted to the full-navtest mix | 88.28 [87.35, 89.11] |
| (b) random pick | 73.42 [71.42, 75.42] |
| (b) **oracle pick** (max true PDMS of the 64) | **98.16 [97.79, 98.54]** |
| (b) actual − random (paired) | +14.86 [+12.96, +16.63]: the scorer is far better than chance |
| (b) **oracle − actual (paired)** | **+9.88 [+9.09, +10.78]**: zero stratum 3.62, non-zero stratum 6.26 |
| **(c) EP headroom** on non-zeroed tokens: the highest-EP safe proposal minus the pick | **+12.34 EP points [+11.05, +13.78]** (529 / 529 have a safe proposal) |

**What it means.**
- The model's own proposal fan already contains a safe plan on 86 % of the tokens it currently zeroes.
- On the rest, the fan contains a safe plan that progresses about 12 EP points further than the one it picks.
- The **10-point gap between pick and oracle is a ranking problem.** That is exactly what P2 and P6's levers act on: a scorer with un-detached visual context, trained on its own proposals' PDM labels.
- It **confirms D-REFE-SLOWPICK-SFT4-1** (the scorer picks short plans) on scored proposals.

## Descriptive (no decision rests on these)

- **By cause of the zero:**
  - NC = 0 only: 113 tokens, **111 rescuable**;
  - DAC = 0 only: 413 tokens, **343 rescuable**;
  - both: 3 tokens, 1 rescuable.
- **The scorer's near-misses.** On the 455 rescuable tokens, the scorer's best-ranked safe proposal sits at **median rank 4**: 37 % within its top 3, 67 % within its top 8. The scorer is close, not lost.
- **By ego speed v0 (m/s), and by command:** rescuable share is 81–100 % in every bin with n ≥ 10. Left 126 / 160, straight 231 / 252, right 98 / 117.

## What this does NOT license

- ⛔ **Not a P2 or P6 result.** C3 shows the headroom exists where the levers act; it does not show the levers capture it. P2 is the first measured evidence.
- ⛔ **The oracle is a ceiling, not a target.** It picks with the true PDMS, which no deployable scorer has. The random control is reported beside it, as the programme requires.
- ⛔ **Not a ruling on PI decision (a).** No tracker compensation is involved.

## Next lever (RULE ZERO)

**C3 is GREEN, and it was the condition that decided whether P6 is aimed at the right component.** It is.
- **S1:** a pod for P2, the scorer lever, measured on full navtest.
- **In parallel, 0 GPU:** implement P6's new trainer features (the navigation command, the yaw-loss option, the online labeller, the P6 loss terms), so that `refe_p6` stops refusing in the launch gate before P2 reports.
