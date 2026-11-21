# RESULT_TSWAP (T1-kin, eval-141, scene `reindexed`)

**Outcome by the card: T-NOT.**  T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2). TSWAP feeds the logged state (privileged): a diagnostic.
Estimator: paired episode-cluster bootstrap B=2000 seed 0 (gapdecomp.decompose) over the windows' episodes; answers another draw of EPISODES; inference is deterministic at a fixed thread count; the training-run question is answered by the two seeds.

Controls (card): all PASS

* s0: G-ID PASS, G-TICK0 PASS, G-LIVE PASS, G-MUT PASS
* s1: G-ID PASS, G-TICK0 PASS, G-LIVE PASS, G-MUT PASS

## s0: the named rows

| row | TSWAP - OWN [95 % CI] | status | TSWAP - P10 [CI] | status | OWN - P10 [CI] | status |
|---|---|---|---|---|---|---|
| M1 crawl share | +0.0031 [-0.0003, +0.0073] | n.s. | +0.0569 [+0.0380, +0.0790] | worse | +0.0533 [+0.0356, +0.0744] | worse |
| ADE6 | -0.0466 [-0.0742, -0.0203] | better | -1.1594 [-1.3489, -0.9793] | better | -1.1128 [-1.3043, -0.9322] | better |

T share: -0.059 [-0.150, +0.006] (numerator -0.0031, denominator +0.0533, 2,105 windows, draws with non-positive denominator 0)

### s0: all 19 rows

| family | row | TSWAP - OWN [CI] | status | TSWAP - P10 [CI] | status | OWN - P10 status |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.0466 [-0.0742, -0.0203] | better | -1.1594 [-1.3489, -0.9793] | better | better |
| crawl | M1 crawl share | +0.0031 [-0.0003, +0.0073] | n.s. | +0.0569 [+0.0380, +0.0790] | worse | worse |
| LONGITUDINAL | speed MAE | -0.0511 [-0.0806, -0.0218] | better | -0.7852 [-0.9740, -0.6249] | better | better |
| LONGITUDINAL | along-track MAE | -0.0484 [-0.0814, -0.0159] | better | -0.6971 [-0.8659, -0.5381] | better | better |
| LONGITUDINAL | accel MAE | -0.0398 [-0.0749, -0.0036] | better | -0.8374 [-1.1737, -0.5620] | better | better |
| LONGITUDINAL | DK-VIOL | -0.0017 [-0.0107, +0.0065] | n.s. | -0.0118 [-0.0354, +0.0082] | n.s. | n.s. |
| LONGITUDINAL | DK-HW | +0.0549 [-0.0444, +0.1886] | n.s. | -0.3380 [-0.6986, -0.0201] | better | better |
| LONGITUDINAL | DK-TG | -0.0046 [-0.0246, +0.0150] | n.s. | -0.1238 [-0.1947, -0.0661] | better | better |
| LATERAL | heading MAE | +0.1727 [-0.0310, +0.3770] | n.s. | -1.6639 [-2.4387, -0.9178] | better | better |
| LATERAL | yaw-rate MAE | +0.9220 [-0.3256, +2.2073] | n.s. | +3.4240 [+1.1084, +5.8774] | worse | worse |
| LATERAL | curvature MAE | +0.0126 [-0.0017, +0.0262] | n.s. | +0.0967 [+0.0654, +0.1325] | worse | worse |
| LATERAL | cross-track MAE | -0.0062 [-0.0165, +0.0049] | n.s. | -0.8665 [-1.0017, -0.7457] | better | better |
| LATERAL | L1 lateral-accel MAE | +0.1501 [+0.0084, +0.3084] | worse | -2.0803 [-3.2073, -1.1711] | better | better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0075 [-0.0338, +0.0109] | n.s. | +0.0329 [+0.0174, +0.0509] | worse | worse |
| TACTICAL | TAC-LAT | +0.0058 [-0.0059, +0.0192] | n.s. | +0.1884 [+0.1517, +0.2272] | better | better |
| TACTICAL | TAC-LON | +0.0684 [+0.0383, +0.0987] | better | +0.2163 [+0.1610, +0.2733] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@30 | +0.1293 [+0.0304, +0.2369] | worse | -1.0894 [-1.4968, -0.7409] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@50 | +0.1991 [+0.0353, +0.3857] | worse | -3.7193 [-4.7835, -2.7889] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@80 | +0.2331 [-0.0321, +0.5376] | n.s. | -8.7508 [-10.8336, -6.7939] | better | better |

TSWAP separated-better than OWN on: ADE6, TAC-LON, accel MAE, along-track MAE, speed MAE.  TSWAP separated-worse than OWN on: L1 lateral-accel MAE, S-POS@30, S-POS@50.

## s1: the named rows

| row | TSWAP - OWN [95 % CI] | status | TSWAP - P10 [CI] | status | OWN - P10 [CI] | status |
|---|---|---|---|---|---|---|
| M1 crawl share | -0.0037 [-0.0086, +0.0007] | n.s. | +0.0559 [+0.0348, +0.0801] | worse | +0.0596 [+0.0368, +0.0860] | worse |
| ADE6 | -0.0563 [-0.0849, -0.0298] | better | -1.1175 [-1.3124, -0.9298] | better | -1.0612 [-1.2541, -0.8709] | better |

T share: +0.061 [-0.014, +0.133] (numerator +0.0037, denominator +0.0596, 2,106 windows, draws with non-positive denominator 0)

### s1: all 19 rows

| family | row | TSWAP - OWN [CI] | status | TSWAP - P10 [CI] | status | OWN - P10 status |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -0.0563 [-0.0849, -0.0298] | better | -1.1175 [-1.3124, -0.9298] | better | better |
| crawl | M1 crawl share | -0.0037 [-0.0086, +0.0007] | n.s. | +0.0559 [+0.0348, +0.0801] | worse | worse |
| LONGITUDINAL | speed MAE | -0.0699 [-0.0956, -0.0459] | better | -0.8277 [-1.0233, -0.6472] | better | better |
| LONGITUDINAL | along-track MAE | -0.0748 [-0.1044, -0.0477] | better | -0.6574 [-0.8338, -0.4869] | better | better |
| LONGITUDINAL | accel MAE | -0.0121 [-0.0521, +0.0425] | n.s. | -0.9564 [-1.3020, -0.6653] | better | better |
| LONGITUDINAL | DK-VIOL | +0.0034 [-0.0030, +0.0105] | n.s. | -0.0151 [-0.0344, +0.0019] | n.s. | better |
| LONGITUDINAL | DK-HW | +0.0406 [-0.0338, +0.1314] | n.s. | -0.3501 [-0.7312, -0.0172] | better | better |
| LONGITUDINAL | DK-TG | -0.0032 [-0.0240, +0.0127] | n.s. | -0.0841 [-0.1492, -0.0241] | better | better |
| LATERAL | heading MAE | +0.2215 [-0.0659, +0.4884] | n.s. | -1.6194 [-2.3907, -0.8466] | better | better |
| LATERAL | yaw-rate MAE | +0.2895 [-0.8417, +1.4739] | n.s. | +3.1665 [+0.3271, +6.2820] | worse | worse |
| LATERAL | curvature MAE | +0.0016 [-0.0101, +0.0131] | n.s. | +0.1003 [+0.0622, +0.1466] | worse | worse |
| LATERAL | cross-track MAE | +0.0200 [+0.0068, +0.0368] | worse | -0.8452 [-0.9741, -0.7290] | better | better |
| LATERAL | L1 lateral-accel MAE | +0.0447 [-0.0445, +0.1304] | n.s. | -2.5074 [-3.7106, -1.4986] | better | better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0107 [-0.0269, +0.0029] | n.s. | +0.0351 [+0.0145, +0.0625] | worse | worse |
| TACTICAL | TAC-LAT | -0.0167 [-0.0267, -0.0067] | worse | +0.1209 [+0.0787, +0.1664] | better | better |
| TACTICAL | TAC-LON | +0.0709 [+0.0387, +0.1022] | better | +0.2109 [+0.1543, +0.2651] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@30 | +0.1929 [+0.0668, +0.3464] | worse | -0.9816 [-1.3296, -0.6533] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@50 | +0.3265 [+0.1283, +0.5556] | worse | -3.5317 [-4.4589, -2.6461] | better | better |
| STRATEGIC (PAUSED, decides nothing) | S-POS@80 | +0.4633 [+0.1686, +0.8071] | worse | -8.8802 [-10.8914, -7.0187] | better | better |

TSWAP separated-better than OWN on: ADE6, TAC-LON, along-track MAE, speed MAE.  TSWAP separated-worse than OWN on: S-POS@30, S-POS@50, S-POS@80, TAC-LAT, cross-track MAE.

## Next, committed by the card for this outcome

The crawl does not sit in T's own-state decision. Next: the twin read on O (O fed the logged ego history), under its own card.

STRATEGIC family: PAUSED (PI item 25), listed, decides nothing. Mechanism read at T1-kin, not a driving claim; TSWAP feeds a privileged signal.
