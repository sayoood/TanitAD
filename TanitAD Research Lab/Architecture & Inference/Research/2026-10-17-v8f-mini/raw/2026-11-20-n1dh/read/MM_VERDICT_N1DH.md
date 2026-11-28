# N1DH verdict of record: N1DH-COST (MM, 2026-11-21)

Card `PREREG_V8F_N1DH.md` (sha256 f1df05bc) with Amendments 1 to 5 (25d6b42a, 0f78a055, cd6a326c, 96a435df, cb2f0ce0). Code of record 1f217bf0. Arms N1DH_s0 and N1DH_s1 (an auxiliary route-decision head on O's plan feature, CE 0.5) against N1D_s0 and N1D_s1. Evidence class: MEASURED (ours). Estimator: paired episode-cluster bootstrap, B 2,000, seed 0; it answers "another draw of EPISODES"; the two seeds answer "another TRAINING run".

## The three parts the card joins
| part | artifact | reading |
|---|---|---|
| deciding cell, D1-HO TAC-LAT, N1DH - N1D, tier T1-kin/hb0.2 | `out_real/RESULT_N1DH_D1HO.json` (also `out_real_zpost/`) | s0 -0.0085 [-0.0186, +0.0010]; s1 +0.0080 [-0.0022, +0.0174]; neither separated: **no gain at either seed** |
| Z-POST, the route head's own T0 accuracy on D1-HO | `RESULT_N1DH_ZPOST.json` (pod sha256 dbc839d7; controls K-PERM, K-CONST, K-OL-REAL pass) | s0 **0.6876** [0.6678, 0.7070]; s1 **0.7101** [0.6914, 0.7280]; the decoded plan's T0 accuracy is 0.6177 and 0.6440. At or above 0.685 at both seeds (by point, the reader's rule), so POST-WEAK does not fire |
| eval-141 guard, four families, tier T1-kin/hb0.2 | `eval141/out/RESULT_N1DH_EVAL141.json` | **GUARD-COST-BOTH-SEEDS** |

## The guard, per family (N1DH - N1D on eval-141; floor = F2 from the banked old-executor C0 replicate)
* **LONGITUDINAL:** cost at both seeds, in one cell. `operative_0_2s:along_mae_m` is +0.0016 m [+0.0006, +0.0028] at s0 and +0.0027 m [+0.0011, +0.0045] at s1, against a floor of 0.0005 m. At s0, `operative_0_2s:ade_m` (+0.0018 m) and `tactical_2_6s:dk_tg_abs_err_s` (+0.048 s [0.0081, 0.1154], floor 0.0219) also cost. The costs are millimetres; the card's rule counts any separated cell above its floor.
* **LATERAL:** no cost. At s1, `operative_0_2s:heading_mae_deg` is -0.0205 deg [-0.0395, -0.0011] and cross-track is also a gain cell. s0 has no lateral cell separated.
* **TACTICAL:** TAC-LAT +0.0046 (s0) and -0.0083 (s1); TAC-LON -0.0138 (s0) and +0.0038 (s1). None separated, no cost.
* **STRATEGIC (T0):** S1-slot view S-PARTIAL for both N1DH and N1D (real minus donor separated at both seeds, not above the best constant at one seed or more); no route cost in the N1DH - N1D contrast.

## Label
The card's table, section 4: GAIN, ONE-SEED and GAIN-COST need a gain, and there is none. **N1DH-COST fires** (guard cost at both seeds). It is reported as written; the cost cells are small (millimetres along track at 0-2 s), and the card counts them.

## What it says, and the next lever
* The route head can read the decision: its own T0 accuracy is 0.69 to 0.71, at or above the card's logistic 0.685 at both seeds. The decoded plan does not follow it (0.62 to 0.64). That is the card's NO-GAIN diagnosis: the loss sits in the decoder's geometry, not in access to the decision.
* Card row N1DH-COST: "not adopted; the cost family gets a card". The lever named for the diagnosis is a mode-routed plan head: K = 3 residual branches of `head_o`, selected by the posterior. It is an architecture change, so it needs a PI go.
* The cost family (LONGITUDINAL, `along_mae_m` at 0-2 s) is small and has no hb0.2 replicate floor (only F2 from the old executor). The cheapest check of whether it is real is an hb0.2 same-seed replicate of N1D (about 0.3 pod-h, ESTIMATED), and only under its own card.

## Instrument repairs before the values counted (Amendments 3 to 5)
The eval grid in the pod tree (3), a space in the zero-grad gate line (4), and the K-HB0-141 ticks path plus Z-POST's reader modules on the pod (5). No training, argv, cell, bar or outcome rule changed. Amendment 5 was frozen after the deciding cell was read; it touches neither the deciding cell nor any outcome rule.
