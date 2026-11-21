# AMENDMENT_2 to PREREG_LANE_SPEED (written 2026-11-08 ~17:10Z, before any lane-truth correlation was computed)

PREREG sha256 `e99b2576...a166f`. AMENDMENT_1 sha256 `afce2519...`.

## 1. A defect in the control's comparison, and its correction

`analyze_lane.py` and the AMENDMENT_1 comparison joined rebuilt labels to the banked file on (log, token). The banked file holds two rows for most tokens: 3,137 rows, 1,705 distinct (log, token), 1,432 tokens with two rows (ranks 0 and 1). The last row read won, so our rank-0 labels were compared with rank-1 labels.

MEASURED on the same 24 sets (1,536 proposals, `out_ctrl_ph`, `code/gl1_key.py`):

| join | within 1 mm | max abs d |
|---|---|---|
| (log, token), as run | 21.9 % | 1.673 m |
| full key + ckpt_step, the identity of a row | 89.5 % | 0.0736 m |

The second row reproduces `raw/rl_tlane/identity_labels_smoke24_hz10.json` (max 0.0736 m). AMENDMENT_1's "reason not yet known" is therefore this join. Its 21.9 % is RETRACTED as a statement about the labeller. Root-cause class: a join key narrower than the row's identity.

The fix changes the join only. G-L1's bar is not changed.

## 2. G-L1 is reported as written

G-L1's bar stays ≥ 99 % within 1 mm and ≥ 99 % of soft targets within 0.01, on the 40-set control. At 89.5 % on the 24 sets it is expected to FAIL. If it fails on the 40 sets, it is reported as FAILED, and AMENDMENT_1's consequence stands: every outcome of PREREG section 6 is reported as INVALID, and the cause is the next lever.

The PREREG's premise that cross-machine noise is about 1e-7 was wrong. The 2026-10-12 identity file already recorded unequal values up to 0.0736 m. That premise was INHERITED and not checked.

## 3. The next lever, committed before the correlation: is the disagreement material to the read? (G-L1b)

G-L1 tests that the labels match. The read only needs the within-token ranking of the label against speed to survive the disagreement. This can be measured with no model output.

On the 40-set control, per set, with the G-L1-correct join:
- `rho_rebuilt` = Spearman(soft(d rebuilt), v4), `rho_banked` = Spearman(soft(d banked), v4), average ranks, valid proposals, v4 the 0.2 s chord at 4 s on the raw proposal of the queue row. A set enters if both rankings vary.

**G-L1b PASS** = mean over entered sets of |rho_rebuilt − rho_banked| ≤ 0.05 **and** |mean of (rho_rebuilt − rho_banked)| ≤ 0.03 **and** at least 20 sets entered.

Both tolerances are well below the 0.15 decision bar of PREREG section 6. They were fixed before G-L1b was computed.

## 4. How a read is labelled

- G-L1 passes: section 6 applies as written.
- G-L1 fails, G-L1b passes: section 6 is computed and printed as **CONDITIONAL: G-L1 FAILED as written, G-L1b passed**. If G lies within 0.03 of either decision threshold (−0.15, +0.15), the reading is O4.
- G-L1 and G-L1b both fail: INVALID, the correlation is printed as unlabelled diagnostic only.

A CONDITIONAL reading does not by itself unlock arm S1L. `PREREG_SCORER1_AMENDMENT_2.md` blocks S1L when G-L1 fails, so S1L would need G-L1 as written, or the PI's explicit acceptance with this disclosure.

The two CHAIN3 variants stay a test for the cause of the remaining 10.5 %, compared with the corrected join. If a variant passes G-L1 as written, the CRUISE tokens are relabelled with it and section 6 applies as written.
