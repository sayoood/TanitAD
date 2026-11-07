# SUBSET REHEARSAL OF THE TSWAP RESULT WRITER: NOT A RESULT

**SUBSET: 10 episodes / 170 windows, arm S_OWN / S_TSWAP, seed 0 only. This file rehearses the writer; the outcome word below is NOT the card's outcome.**

**Outcome by the card: T-CARRIES (subset rehearsal, not a result).**  T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2). TSWAP feeds the logged state (privileged): a diagnostic.
Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose) over the windows' episodes; answers another draw of EPISODES; inference is deterministic at a fixed thread count; the training-run question is answered by the two seeds.

Controls (card): all PASS


## s0: the named rows

| row | TSWAP - OWN [95 % CI] | status | TSWAP - P10 [CI] | status | OWN - P10 [CI] | status |
|---|---|---|---|---|---|---|
| M1 crawl share | -0.0200 [-0.0200, -0.0200] | better | +0.0169 [-0.0275, +0.0786] | n.s. | +0.0369 [-0.0075, +0.0986] | n.s. |
| ADE6 | +0.0000 [+0.0000, +0.0000] | n.s. | -1.0519 [-1.4704, -0.6856] | better | -1.0519 [-1.4704, -0.6856] | better |

T share: not reported (OWN - P10 on M1 is not separated above 0 at this seed); would be +0.542

### s0: all 19 rows

| family | row | TSWAP - OWN [CI] | status | TSWAP - P10 [CI] | status | OWN - P10 status |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | +0.0000 [+0.0000, +0.0000] | n.s. | -1.0519 [-1.4704, -0.6856] | better | better |
| crawl | M1 crawl share | -0.0200 [-0.0200, -0.0200] | better | +0.0169 [-0.0275, +0.0786] | n.s. | n.s. |
| LONGITUDINAL | speed MAE | +0.0000 [+0.0000, +0.0000] | n.s. | -0.6883 [-1.3108, -0.1975] | better | better |
| LONGITUDINAL | along-track MAE | +0.0000 [+0.0000, +0.0000] | n.s. | -0.5975 [-1.0634, -0.1564] | better | better |
| LONGITUDINAL | accel MAE | +0.0000 [+0.0000, +0.0000] | n.s. | -0.5682 [-1.5433, +0.0227] | n.s. | n.s. |
| LONGITUDINAL | DK-VIOL | +0.0000 [+0.0000, +0.0000] | n.s. | +0.0000 [-0.0667, +0.1667] | n.s. | n.s. |
| LONGITUDINAL | DK-HW | +0.0000 [+0.0000, +0.0000] | n.s. | -0.5832 [-1.2922, -0.0671] | better | better |
| LONGITUDINAL | DK-TG | +0.0000 [+0.0000, +0.0000] | n.s. | -0.1178 [-0.3201, -0.0050] | better | better |
| LATERAL | heading MAE | +0.0000 [+0.0000, +0.0000] | n.s. | -1.8055 [-2.8088, -0.7446] | better | better |
| LATERAL | yaw-rate MAE | +0.0000 [+0.0000, +0.0000] | n.s. | +2.2288 [-0.5674, +6.1556] | n.s. | n.s. |
| LATERAL | curvature MAE | +0.0000 [+0.0000, +0.0000] | n.s. | +0.0401 [+0.0005, +0.1026] | worse | worse |
| LATERAL | cross-track MAE | +0.0000 [+0.0000, +0.0000] | n.s. | -0.7957 [-0.9830, -0.6131] | better | better |
| LATERAL | L1 lateral-accel MAE | +0.0000 [+0.0000, +0.0000] | n.s. | -0.3068 [-0.6497, -0.0024] | better | better |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0000 [+0.0000, +0.0000] | n.s. | +0.0092 [+0.0002, +0.0268] | worse | worse |
| TACTICAL | TAC-LAT | +0.0000 [+0.0000, +0.0000] | n.s. | +0.2824 [+0.0882, +0.5001] | better | better |
| TACTICAL | TAC-LON | +0.0000 [+0.0000, +0.0000] | n.s. | +0.2118 [-0.0118, +0.4529] | n.s. | n.s. |
| STRATEGIC (PAUSED, decides nothing) | S-POS@30 | +0.0000 [+0.0000, +0.0000] | n.s. | -0.4892 [-1.7097, +0.6774] | n.s. | n.s. |
| STRATEGIC (PAUSED, decides nothing) | S-POS@50 | +0.0000 [+0.0000, +0.0000] | n.s. | -2.4602 [-5.4221, -0.0855] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@80 | +0.0000 [+0.0000, +0.0000] | n.s. | -5.5015 [-12.0142, -0.0445] | better | better |

TSWAP separated-better than OWN on: M1 crawl share.  TSWAP separated-worse than OWN on: none.

## Next, committed by the card for this outcome

T trained on its own states, the DT2 / RECOVER recipe on T's labels, under its own card.

STRATEGIC family: PAUSED (PI item 25), listed, decides nothing. Mechanism read at T1-kin, not a driving claim; TSWAP feeds a privileged signal.
