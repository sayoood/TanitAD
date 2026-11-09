# H-SCORER1-1, Stage 0 result (Master Mind, 2026-11-09)

**Committed reading: Z1. The gap is present on navtrain held-out, so Stage 1 runs.** `PREREG_SCORER1` section 1 bar: mean delta-rho over TTC and NC <= -0.20 with CI-hi < 0. Measured: **-0.476 [-0.556, -0.396]** (MEASURED, `stage0.json`).

Tier: scorer diagnostic on held-out navtrain sets (T0-class). No driving claim. Families: LONGITUDINAL only through the within-token speed correlation below; LATERAL only through the DAC row; TACTICAL not applicable to this statistic; STRATEGIC PAUSED (PI 2026-09-19), not a pass.

## The statistic
On cruise tokens, the within-token Spearman of each head against the proposals' v4/v0: predicted score against true label. Gap = predicted minus true. Estimator: average-rank Spearman, paired log-cluster bootstrap, 10,000 draws, seed 20260927. A token enters a head only if its true labels vary. 3,137 sets scored, 193 cruise tokens in 12 logs.

| head | n tokens | predicted | true | gap | 95 % CI |
|---|---|---|---|---|---|
| NC | 22 | -0.703 | -0.217 | -0.486 | [-0.597, -0.381] |
| TTC | 51 | -0.688 | -0.222 | -0.466 | [-0.540, -0.381] |
| DAC | 80 | -0.234 | -0.076 | -0.158 | [-0.282, -0.030] |
| EP | 191 | 0.154 | 0.630 | -0.476 | [-0.561, -0.341] |

The gap on navtest was TTC -0.451, NC -0.396 (P1 Q4, an earlier estimator). A gap of the same size on held-out navtrain is what Z1 means: the defect is in the objective or the representation, not a navtrain-to-navtest shift.

## Inputs, as the Stage 0 rulings require
Checkpoint u400 (`model_sft4_B_u400.pt`, sha `8b4dba48...`). Sets from `refe_heldout_v5.tar.zst`, not the on-policy sets. PDM and lane labels from `Sayood/tanitad-refe-s1aux` `labels/`, not `pdm_labels_sft4r` (`LABEL_SOURCE.md`; join 3,137 of 3,137, 0 misses). Scores: `dump/scores_step0_base.npz`.

**Differences from the pod path:** dev-box RTX 4060, **batch 8 instead of 16**, step-0 scores only, through the scratch dump-only patch. Scores can differ at the size of a bf16 rounding step between batch compositions. The trainer's own preflight was not run here; G3 on the L40 is still required before any SCORER1 arm.

## Reading beyond the bar (post hoc, decides nothing)
The EP row was not part of the committed reading. On cruise the scorer believes progress rises with speed far less than it does (0.154 against 0.630), while it believes safety falls with speed far more than it does. Both errors push the same way: against faster proposals. That is the direction of the free-road crawl, measured here on the scorer only. HYPOTHESIS for driving.

SCORER1's pairwise loss covers TTC, NC and DAC, **not EP**. A pairwise term on EP would be a new lever and needs its own pre-registration, an arm and compute approval. The lane arm's head is unaffected by this reading.
