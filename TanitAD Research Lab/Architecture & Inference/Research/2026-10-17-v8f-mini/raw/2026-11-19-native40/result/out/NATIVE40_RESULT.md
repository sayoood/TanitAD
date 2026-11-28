# NATIVE40 result: T0 read of the native lead head (LLR_s0, LLR_s1)

**Label: NATIVE-OK**   (evidence MEASURED, tier T0; the interval is the episode-cluster bootstrap B=2000 seed 0 and answers another draw of EPISODES only)

Bars (card): mean gap MAE <= 5.80 m AND presence AUROC >= 0.85 at both seeds. Reference 5.04 m is INHERITED (LS_REF40.json), used only as the bar's base.

Map control (before any head number): 533 of 534 coordinator windows match our label within 1e-3 m (share 0.9981); rule: AMENDMENT_1: coordinator window read at (eid, cache step g0); our label raw frame k = 2*g0; head read at cache step g0 (x_* arrays); |our gap - gap_t| <= 1e-3 m.

| seed | gap MAE m [95% CI] | R2 [95% CI] | n gap | AUROC [95% CI] | n pos / n neg |
|---|---|---|---|---|---|
| s0 | 5.712 [4.783, 6.793] | 0.474 [0.253, 0.615] | 533 | 0.8727 [0.8277, 0.9149] | 630 / 1614 |
| s1 | 5.534 [4.618, 6.652] | 0.480 [0.257, 0.623] | 533 | 0.9016 [0.8606, 0.9367] | 630 / 1614 |

Mean over seeds gap MAE: 5.623 m [4.709, 6.658] (bar 5.80): holds. AUROC at both seeds >= 0.85: holds.

## Controls

| seed | shuffle MAE / R2 / AUROC | units x100 MAE (clipped / unclipped) | label arm MAE / R2 | all pass |
|---|---|---|---|---|
| s0 | 10.585 / -0.525 / 0.511 | 14.225 / 24.435 | 0.000 / 1.000 | True |
| s1 | 10.572 / -0.533 / 0.490 | 14.140 / 23.356 | 0.000 / 1.000 | True |

Windows file sha256 166198fb4cbd9da116689cc47380726af053a7dced5e91ca3c1db8448f41c0be. Dumps: s0 4000d4106df6af70 ckpt 5a52b28bcf3e2f15; s1 14f8209bbf4dced7 ckpt 71ee9feab2683a66.
Scope: a head choice only; no LONGITUDINAL, LATERAL, TACTICAL or STRATEGIC family is claimed from this read (card section 6).
