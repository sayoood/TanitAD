<!-- TABLES:SUBSTRATE -->
### The universe and the label, per situation

| situation | TRAIN events / pos-frames / pos-clips | HELD-OUT events / pos-frames / pos-clips | held-out base rate | **C-POW** |
|---|---|---|---|---|
| **LANE CHANGE** | 87 / 2,127 / 77 | 185 / 4,599 / 158 | 0.01742 | ✅ OK (158 clusters) |
| **ROUNDABOUT** | 17 / 440 / 15 | 34 / 747 / 27 | 0.00277 | ⛔ **UNDERPOWERED** (27 clusters) |
| **INTERSECTION** | 120 / 3,032 / 105 | 300 / 7,813 / 269 | 0.02998 | ✅ OK (269 clusters) |

*Universe: 2,376 cached parity episodes, 197 chunks (66 TRAIN / 131 HELD-OUT); TRAIN 766 clips / 152,384 frames, HELD-OUT 1,610 clips / 320,243 frames. Anticipation lead 3.0 s; minimum useful lead 1.0 s (registered before measurement).*
<!-- /TABLES:SUBSTRATE -->

<!-- TABLES:VALIDATION -->
### V1 — the lane-change heading gate, swept (the ×89-collapse test)

| net-heading gate | events (with S-shape) | clip rate | median lateral | events (NO S-shape) | clip rate | median lateral |
|---|---|---|---|---|---|---|
| 1° | 43 | 0.05483 | 3.099 m | 51 | 0.06527 | 2.94 m |
| 2° | 58 | 0.06919 | 3.477 m | 128 | 0.14491 | 3.288 m |
| 3° | 71 | 0.08616 | 3.526 m | 230 | 0.25196 | 4.083 m |
| 4° | 77 | 0.09138 | 3.757 m | 294 | 0.31593 | 4.589 m |
| 6° | 85 | 0.10052 | 4.014 m | 368 | 0.38903 | 5.417 m |
| 8° ⭐ | 87 | 0.10183 | 4.581 m | 400 | 0.41253 | 5.848 m |
| 10° | 91 | 0.10313 | 4.617 m | 425 | 0.42950 | 6.118 m |
| 12° | 92 | 0.10444 | 4.622 m | 432 | 0.44386 | 6.357 m |

**Collapse ratio 10° → 1°: 1.88× with the S-shape clause, 6.58× without.** The S-shape clause rejects **78.2 %** of candidates at the operating gate. Lateral offset at the operating gate: median 4.581 m [p10 2.707, p90 5.618].

### V2 — the roundabout direction test (the corpus has **0** left-hand-traffic clips)

| variant | TRAIN (in-sample: the selection target) | HELD-OUT (**out-of-sample**) |
|---|---|---|
| `ROUND` (pre-registered) | 0.8824 (n=17) | **0.8235** (n=34) |
| `ROUND_core` | 0.6757 (n=37) | **0.68** (n=100) |

Maximum same-sign sweep anywhere in this universe: **282.1°** (282.1° at radius ≤ 50 m).

### V3 — turn left/right balance (a junction population must be ~50/50)

| side | n | left fraction |
|---|---|---|
| TRAIN | 125 | 0.544 |
| HELDOUT | 312 | 0.5385 |

### ⭐ V4 — is the TURN half a junction detector, or just a curve detector?

*Pre-registered in `PRE_REGISTRATION.md` §6.2 with both outcomes committed. If the ratio's CI included 1.0, the turn half would NOT be a junction detector and the intersection label would fall back to `CROSS`-only.*

| quantity | value |
|---|---|
| P(perpendicular cross traffic \| tight TURN, R ≤ 25 m) | **0.0803** (2,939 frames) |
| P(… \| matched-heading LARGE-RADIUS curve, R > 40 m) | 0.03326 (3,428 frames) |
| **ratio** | **2.415×** [1.057, 7.931] |
| separated from 1.0? | ✅ **YES** |

*paired episode-cluster bootstrap (taniteval.ci._draws, B=2000).*

### C-ALIGN — the obstacle clock map, in METRES

| quantity | value |
|---|---|
| clips attempted / admitted | 480 / 450 |
| median position residual | **0.0018 m** |
| p90 / p99 residual | 0.02845 m / 1.73231 m |
| failure fraction (> 0.5 m) | 6.25 % |
| **obstacle-window overlap (median)** | **0.9958** |
| cross-traffic frame rate (path-crossing) | 0.01619 |
| perpendicular-agent-present frame rate | 0.23833 |

<!-- /TABLES:VALIDATION -->

<!-- TABLES:RESULTS -->
### LANE CHANGE — held-out discrimination

*252,826 scored windows · 4,361 positives · 153 positive clip clusters of 1610 · base rate 0.01725 · **C-POW OK**.*

| arm | AP (episode-cluster bootstrap CI95) | AP / base | AUROC | ΔAP vs CHANCE | above chance? |
|---|---|---|---|---|---|
| `head_img_ego` ⭐ | 0.04347 [0.03106, 0.06221] | 2.5203× | 0.677 | 0.02703 [0.01377, 0.04466] | **YES** |
| `ridge_img_ego` | 0.03580 [0.02756, 0.04689] | 2.0753× | 0.6691 | 0.01849 [0.00922, 0.02906] | **YES** |
| `head_img` | 0.03741 [0.02959, 0.04745] | 2.1691× | 0.703 | 0.01987 [0.01141, 0.02901] | **YES** |
| `ridge_img` | 0.03405 [0.02578, 0.04653] | 1.9738× | 0.6659 | 0.01688 [0.00769, 0.02790] | **YES** |
| `head_ego` | 0.08699 [0.06506, 0.11805] | 5.043× | 0.814 | 0.07065 [0.04798, 0.09904] | **YES** |
| `ridge_ego` | 0.03057 [0.02294, 0.04058] | 1.7722× | 0.5631 | 0.01324 [0.00481, 0.02264] | **YES** |
| `head_img_ego_concat` | 0.03254 [0.02546, 0.04156] | 1.8864× | 0.6627 | 0.01498 [0.00749, 0.02321] | **YES** |
| `heur_kin` | 0.06665 [0.05070, 0.08714] | 3.8642× | 0.7747 | 0.04963 [0.03396, 0.06871] | **YES** |
| `head_priv` ⭐C-POS | 0.12008 [0.09322, 0.15790] | 6.9615× | 0.8801 | 0.10404 [0.07591, 0.13909] | **YES** |
| `head_img_shuf` ⭐C-NEG | 0.01715 [0.01380, 0.02112] | 0.9942× | 0.4997 | -0.00057 [-0.00624, 0.00338] | no |
| `ridge_img_shuf` ⭐C-NEG | 0.01921 [0.01542, 0.02364] | 1.1136× | 0.5373 | 0.00143 [-0.00415, 0.00592] | no |

| contrast | Δ | CI95 | separated? |
|---|---|---|---|
| `head_img_ego` − `head_ego` | -0.04361 | [-0.07252, -0.01914] | **YES** |
| `head_img` − `head_ego` | -0.05077 | [-0.07786, -0.02971] | **YES** |
| `head_img_ego_concat` − `head_ego` | -0.05567 | [-0.08280, -0.03417] | **YES** |
| `ridge_img_ego` − `head_ego` | -0.05216 | [-0.07820, -0.03038] | **YES** |
| `ridge_img` − `head_ego` | -0.05377 | [-0.07968, -0.03295] | **YES** |
| `ridge_ego` − `head_ego` | -0.05741 | [-0.08486, -0.03471] | **YES** |
| `heur_kin` − `head_ego` | -0.02102 | [-0.04501, -0.00043] | **YES** |
| `heur_lane_change` − `head_ego` | -0.02102 | [-0.04501, -0.00043] | **YES** |
| `heur_roundabout` − `head_ego` | -0.07433 | [-0.10274, -0.05193] | **YES** |
| `heur_intersection` − `head_ego` | -0.07413 | [-0.10205, -0.05184] | **YES** |
| recall: head_img_ego - random_at_rate | +0.0497 | [+0.0217, +0.0825] | **YES** |
| recall: head_img_ego - head_ego | -0.0918 | [-0.1421, -0.0415] | **YES** |
| recall: ridge_img_ego - random_at_rate | +0.0607 | [+0.0289, +0.0946] | **YES** |
| recall: ridge_img_ego - head_ego | -0.0808 | [-0.1326, -0.0299] | **YES** |

#### LANE CHANGE — operating point at B\* = 0.05 extra cams/frame

| arm | firing rate | recall | precision | precision lift | caught / total | θ\* |
|---|---|---|---|---|---|---|
| `head_img_ego` | 0.0200 [0.0167, 0.0234] | 0.0667 [0.0393, 0.0996] | 0.05757 [0.03279, 0.08562] | 3.34× | 291/4361 | 0.7990 |
| `ridge_img_ego` | 0.0229 [0.0200, 0.0259] | 0.0777 [0.0461, 0.1122] | 0.05848 [0.03452, 0.08377] | 3.39× | 339/4361 | -0.8818 |
| `head_img` | 0.0276 [0.0236, 0.0320] | 0.0821 [0.0534, 0.1158] | 0.05123 [0.03263, 0.07364] | 2.97× | 358/4361 | 0.6562 |
| `ridge_img` | 0.0236 [0.0202, 0.0268] | 0.0718 [0.0384, 0.1096] | 0.05250 [0.02820, 0.07936] | 3.04× | 313/4361 | -0.8897 |
| `head_ego` | 0.0192 [0.0158, 0.0226] | 0.1594 [0.1162, 0.2050] | 0.14342 [0.10156, 0.18945] | 8.31× | 695/4361 | 0.7263 |
| `ridge_ego` | 0.0253 [0.0222, 0.0283] | 0.0780 [0.0474, 0.1113] | 0.05308 [0.03158, 0.07659] | 3.08× | 340/4361 | -0.9303 |
| `head_img_ego_concat` | 0.0395 [0.0345, 0.0448] | 0.1053 [0.0709, 0.1443] | 0.04595 [0.03052, 0.06349] | 2.66× | 459/4361 | 0.7409 |
| `heur_kin` | 0.0279 [0.0238, 0.0321] | 0.1685 [0.1215, 0.2177] | 0.10406 [0.07332, 0.13680] | 6.03× | 735/4361 | 0.4285 |
| `head_priv` | 0.0217 [0.0180, 0.0254] | 0.2211 [0.1701, 0.2764] | 0.17611 [0.13239, 0.22882] | 10.21× | 964/4361 | 0.7624 |
| `head_img_shuf` | 0.0126 [0.0100, 0.0155] | 0.0128 [0.0020, 0.0274] | 0.01758 [0.00267, 0.03650] | 1.02× | 56/4361 | 0.8422 |
| `ridge_img_shuf` | 0.0145 [0.0111, 0.0186] | 0.0177 [0.0049, 0.0344] | 0.02096 [0.00571, 0.04134] | 1.22× | 77/4361 | -0.9276 |
| **(a) always-escalate** | 1.0 | 1.0 | 0.01725 | 1.00× | 4361/4361 | — |
| **(b) never-escalate** | 0.0 | **0.0** | — | — | 0/4361 | — |
| **(c) random @ matched rate** | 0.0200 | 0.0201 [0.0154, 0.0250] | 0.01734 | 1.00× | — | 200 seeds |
| **oracle** | 0.01725 | 1.0 | 1.0 | 58× | 4361/4361 | — |

#### ⭐ LANE CHANGE — LEAD TIME (registered minimum **1.0 s**; a high-AP zero-lead trigger FAILS)

| arm | events detected | event recall | **median lead** | p25 / p75 | frac of events at ≥ min lead | PASS? |
|---|---|---|---|---|---|---|
| `head_img_ego` | 29/177 | 0.1638 | **1.4 s** | 0.7 / 2.4 s | 0.1017 | ✅ |
| `ridge_img_ego` | 31/177 | 0.1751 | **2.1 s** | 1.15 / 2.95 s | 0.1412 | ✅ |
| `head_img` | 36/177 | 0.2034 | **2.15 s** | 0.775 / 2.9 s | 0.1469 | ✅ |
| `ridge_img` | 21/177 | 0.1186 | **2.6 s** | 2.0 / 3.0 s | 0.1017 | ✅ |
| `head_ego` | 55/177 | 0.3107 | **1.5 s** | 0.7 / 2.5 s | 0.2034 | ✅ |
| `ridge_ego` | 26/177 | 0.1469 | **2.75 s** | 1.075 / 3.0 s | 0.113 | ✅ |
| `head_img_ego_concat` | 44/177 | 0.2486 | **1.9 s** | 0.8 / 3.0 s | 0.1751 | ✅ |
| `heur_kin` | 48/177 | 0.2712 | **2.3 s** | 1.25 / 3.0 s | 0.2203 | ✅ |
| `head_priv` | 56/177 | 0.3164 | **2.2 s** | 1.175 / 3.0 s | 0.2599 | ✅ |
| `head_img_shuf` | 6/177 | 0.0339 | **1.95 s** | 1.825 / 2.525 s | 0.0282 | ✅ |
| `ridge_img_shuf` | 10/177 | 0.0565 | **2.55 s** | 1.725 / 2.95 s | 0.0508 | ✅ |

#### LANE CHANGE — the controls, each with its MDE

| control | what it catches | result | can it fail? |
|---|---|---|---|
| **C-POS** `head_priv` | instrument insensitivity | ΔAP vs chance 0.10404 [0.07591, 0.13909] → **YES** | it is a probe on the label's own defining quantity; a null here would mean UNPOWERED |
| **C-NEG** `head_img_shuf` | a pipeline leak | -0.00057 [-0.00624, 0.00338] → no | features permuted ACROSS clips — it must NOT separate |
| **C-NEG** `ridge_img_shuf` | a pipeline leak (closed form) | 0.00143 [-0.00415, 0.00592] → no | as above |
| **MDE** (upper 95 % bound of the C-NEG ΔAP) | the smallest effect this run can distinguish from nothing | **0.00592** | — |
| **C-POW** | reading a small-n null as a refutation | 153 positive clusters vs a 40 bar → **OK** | measured before any score |
| **C-BLIND** (packaged firewall, imported) | the target being a function of the conditioning | verdict **CIRCULAR**; `context_leaks` = **0.0**, `blind_skill_over_majority` = **-7.6e-05** | ⛔ **NO** — the *deterministic* clause did **not** fire (positive rate 0.02123 > eps 0.02); the verdict comes from the `vision_buys_nothing` clause, which compares **accuracies**. A gate that fires at ~3 % to buy recall is *designed* to lose accuracy against 'always predict negative', so on this target that clause fires for any useful classifier. |

> ⚠️ **The C-BLIND verdict is DEGENERATE on this target and must not be quoted alone — its own companion numbers refute it.** `context_leaks = 0` and `blind_skill_over_majority` ≈ 0: **the ego context carries no information about the target beyond the base rate.** The `CIRCULAR` label is produced by an *accuracy* comparison on a 2–3 %-positive target, where the majority-class predictor scores 0.9788 and any recall-seeking classifier necessarily scores lower — the head is *supposed* to fire more often than the base rate. The informative form of the same question — *does vision buy anything the ego state did not already give?* — is the AP-based `− head_ego` contrast above, which is the pre-registered primary comparison.

#### LANE CHANGE — the efficiency curve (recall at a fixed camera budget)

| budget B (extra cams/frame) | realised firing rate | **head recall** | head precision | ego-head recall | kinematic-rule recall | saving vs always-on-7 |
|---|---|---|---|---|---|---|
| 0.005 | 0.0002 | **0.0055** | 0.39344 | 0.0284 | 0.0165 | 85.7 % |
| 0.01 | 0.0011 | **0.0110** | 0.16667 | 0.0410 | 0.0307 | 85.7 % |
| 0.02 | 0.0054 | **0.0259** | 0.08224 | 0.0830 | 0.0754 | 85.6 % |
| 0.05 | 0.0200 | **0.0667** | 0.05757 | 0.1594 | 0.1685 | 85.1 % |
| 0.1 | 0.0428 | **0.1429** | 0.05757 | 0.2726 | 0.3020 | 84.5 % |
| 0.2 | 0.0937 | **0.2901** | 0.05341 | 0.4020 | 0.4682 | 83.0 % |

#### LANE CHANGE — the efficiency ledger (⚠️ read the RECALL column, not the saving)

| policy | extra cams/frame | cams/frame | saving vs always-on-7 | **recall** |
|---|---|---|---|---|
| **never escalate** (free and useless) | 0 | 1.000 | 85.7 % | **0.000** |
| **ORACLE** — fires exactly on the label | 0.0345 | 1.0345 | 85.2 % | **1.000** |
| **`head_img_ego` @ B\*** | 0.0400 | 1.0400 | 85.1 % | **0.0667** [0.0393, 0.0996] |
| **always escalate** | 6.000 | 7.000 | 0.0 % | **1.000** |

> The span of *saving* between a useless gate and a perfect oracle is **0.49 percentage points** — so **no compute-saving number here can distinguish a good gate from a useless one** (BOOST_PROGRAM §7.3). The axis that carries information is **recall at the budget**, and the lead time.

### ROUNDABOUT — held-out discrimination

*258,540 scored windows · 721 positives · 26 positive clip clusters of 1610 · base rate 0.00279 · **C-POW UNDERPOWERED**.*

| arm | AP (episode-cluster bootstrap CI95) | AP / base | AUROC | ΔAP vs CHANCE | above chance? |
|---|---|---|---|---|---|
| `head_img_ego` ⭐ | 0.01237 [0.00503, 0.02777] | 4.4361× | 0.7301 | 0.01014 [0.00176, 0.02436] | **YES** |
| `ridge_img_ego` | 0.01473 [0.00691, 0.02906] | 5.2803× | 0.7743 | 0.01234 [0.00340, 0.02575] | **YES** |
| `head_img` | 0.00721 [0.00382, 0.01316] | 2.5846× | 0.7494 | 0.00425 [-0.00011, 0.00982] | no |
| `ridge_img` | 0.01056 [0.00536, 0.01926] | 3.7853× | 0.7788 | 0.00773 [0.00194, 0.01557] | **YES** |
| `head_ego` | 0.02328 [0.01181, 0.03951] | 8.3494× | 0.8711 | 0.02066 [0.00836, 0.03596] | **YES** |
| `ridge_ego` | 0.01186 [0.00644, 0.01901] | 4.2543× | 0.777 | 0.00883 [0.00310, 0.01544] | **YES** |
| `head_img_ego_concat` | 0.00991 [0.00469, 0.01900] | 3.5535× | 0.7469 | 0.00713 [0.00123, 0.01559] | **YES** |
| `heur_kin` | 0.00482 [0.00309, 0.00795] | 1.7292× | 0.7373 | 0.00176 [-0.00162, 0.00422] | no |
| `head_priv` ⭐C-POS | 0.09091 [0.04742, 0.16079] | 32.5992× | 0.956 | 0.09262 [0.04372, 0.15720] | **YES** |
| `head_img_shuf` ⭐C-NEG | 0.00328 [0.00176, 0.00604] | 1.1763× | 0.5186 | 0.00011 [-0.00337, 0.00300] | no |
| `ridge_img_shuf` ⭐C-NEG | 0.00278 [0.00143, 0.00664] | 0.9958× | 0.4276 | -0.00022 [-0.00375, 0.00349] | no |

| contrast | Δ | CI95 | separated? |
|---|---|---|---|
| `head_img_ego` − `head_ego` | -0.01053 | [-0.02304, +0.00212] | no |
| `head_img` − `head_ego` | -0.01642 | [-0.02878, -0.00692] | **YES** |
| `head_img_ego_concat` − `head_ego` | -0.01353 | [-0.02457, -0.00518] | **YES** |
| `ridge_img_ego` − `head_ego` | -0.00832 | [-0.01920, +0.00060] | no |
| `ridge_img` − `head_ego` | -0.01293 | [-0.02438, -0.00426] | **YES** |
| `ridge_ego` − `head_ego` | -0.01183 | [-0.02318, -0.00342] | **YES** |
| `heur_kin` − `head_ego` | -0.01890 | [-0.03379, -0.00704] | **YES** |
| `heur_lane_change` − `head_ego` | -0.01987 | [-0.03426, -0.00881] | **YES** |
| `heur_roundabout` − `head_ego` | -0.01890 | [-0.03379, -0.00704] | **YES** |
| `heur_intersection` − `head_ego` | -0.01554 | [-0.03247, +0.00151] | no |
| recall: head_img_ego - random_at_rate | +0.1171 | [+0.0267, +0.2158] | **YES** |
| recall: head_img_ego - head_ego | -0.1599 | [-0.2696, -0.0592] | **YES** |
| recall: ridge_img_ego - random_at_rate | +0.1584 | [+0.0549, +0.2779] | **YES** |
| recall: ridge_img_ego - head_ego | -0.1185 | [-0.2446, -0.0065] | **YES** |

#### ROUNDABOUT — operating point at B\* = 0.05 extra cams/frame

| arm | firing rate | recall | precision | precision lift | caught / total | θ\* |
|---|---|---|---|---|---|---|
| `head_img_ego` | 0.0251 [0.0218, 0.0285] | 0.1415 [0.0557, 0.2346] | 0.01570 [0.00542, 0.02778] | 5.63× | 102/721 | 0.2642 |
| `ridge_img_ego` | 0.0246 [0.0218, 0.0275] | 0.1803 [0.0826, 0.2969] | 0.02047 [0.00795, 0.03600] | 7.34× | 130/721 | -0.9446 |
| `head_img` | 0.0166 [0.0139, 0.0195] | 0.0860 [0.0118, 0.1796] | 0.01446 [0.00185, 0.03050] | 5.18× | 62/721 | 0.4808 |
| `ridge_img` | 0.0254 [0.0221, 0.0288] | 0.1318 [0.0464, 0.2371] | 0.01445 [0.00425, 0.02797] | 5.18× | 95/721 | -0.9451 |
| `head_ego` | 0.0252 [0.0226, 0.0279] | 0.3010 [0.1801, 0.4286] | 0.03332 [0.01694, 0.05159] | 11.95× | 217/721 | 0.5037 |
| `ridge_ego` | 0.0235 [0.0208, 0.0264] | 0.1359 [0.0601, 0.2251] | 0.01610 [0.00623, 0.02867] | 5.77× | 98/721 | -0.9693 |
| `head_img_ego_concat` | 0.0152 [0.0127, 0.0177] | 0.1054 [0.0340, 0.1942] | 0.01940 [0.00560, 0.03701] | 6.96× | 76/721 | 0.4085 |
| `heur_kin` | 0.0419 [0.0361, 0.0477] | 0.0347 [0.0000, 0.1146] | 0.00231 [0.00000, 0.00752] | 0.83× | 25/721 | -0.0000 |
| `head_priv` | 0.0137 [0.0118, 0.0155] | 0.4674 [0.3533, 0.5825] | 0.09544 [0.05780, 0.13684] | 34.22× | 337/721 | 0.3957 |
| `head_img_shuf` | 0.0177 [0.0148, 0.0206] | 0.0541 [0.0042, 0.1204] | 0.00854 [0.00064, 0.02073] | 3.06× | 39/721 | 0.4008 |
| `ridge_img_shuf` | 0.0095 [0.0075, 0.0119] | 0.0264 [0.0000, 0.0841] | 0.00771 [0.00000, 0.02469] | 2.76× | 19/721 | -0.9741 |
| **(a) always-escalate** | 1.0 | 1.0 | 0.00279 | 1.00× | 721/721 | — |
| **(b) never-escalate** | 0.0 | **0.0** | — | — | 0/721 | — |
| **(c) random @ matched rate** | 0.0251 | 0.0252 [0.0139, 0.0361] | 0.00280 | 1.00× | — | 200 seeds |
| **oracle** | 0.00279 | 1.0 | 1.0 | 359× | 721/721 | — |

#### ⭐ ROUNDABOUT — LEAD TIME (registered minimum **1.0 s**; a high-AP zero-lead trigger FAILS)

| arm | events detected | event recall | **median lead** | p25 / p75 | frac of events at ≥ min lead | PASS? |
|---|---|---|---|---|---|---|
| `head_img_ego` | 9/27 | 0.3333 | **1.7 s** | 1.2 / 2.3 s | 0.2593 | ✅ |
| `ridge_img_ego` | 12/27 | 0.4444 | **1.85 s** | 1.05 / 2.25 s | 0.3333 | ✅ |
| `head_img` | 4/27 | 0.1481 | **2.55 s** | 2.1 / 2.775 s | 0.1481 | ✅ |
| `ridge_img` | 7/27 | 0.2593 | **2.0 s** | 1.45 / 2.2 s | 0.2593 | ✅ |
| `head_ego` | 18/27 | 0.6667 | **1.75 s** | 0.85 / 2.5 s | 0.4815 | ✅ |
| `ridge_ego` | 12/27 | 0.4444 | **1.8 s** | 1.1 / 2.25 s | 0.3704 | ✅ |
| `head_img_ego_concat` | 8/27 | 0.2963 | **1.9 s** | 1.125 / 2.425 s | 0.2222 | ✅ |
| `heur_kin` | 1/27 | 0.037 | **3.0 s** | 3.0 / 3.0 s | 0.037 | ✅ |
| `head_priv` | 21/27 | 0.7778 | **2.0 s** | 1.6 / 2.2 s | 0.7407 | ✅ |
| `head_img_shuf` | 4/27 | 0.1481 | **1.65 s** | 1.3 / 2.0 s | 0.1111 | ✅ |
| `ridge_img_shuf` | 1/27 | 0.037 | **2.3 s** | 2.3 / 2.3 s | 0.037 | ✅ |

#### ROUNDABOUT — the controls, each with its MDE

| control | what it catches | result | can it fail? |
|---|---|---|---|
| **C-POS** `head_priv` | instrument insensitivity | ΔAP vs chance 0.09262 [0.04372, 0.15720] → **YES** | it is a probe on the label's own defining quantity; a null here would mean UNPOWERED |
| **C-NEG** `head_img_shuf` | a pipeline leak | 0.00011 [-0.00337, 0.00300] → no | features permuted ACROSS clips — it must NOT separate |
| **C-NEG** `ridge_img_shuf` | a pipeline leak (closed form) | -0.00022 [-0.00375, 0.00349] → no | as above |
| **MDE** (upper 95 % bound of the C-NEG ΔAP) | the smallest effect this run can distinguish from nothing | **0.00349** | — |
| **C-POW** | reading a small-n null as a refutation | 26 positive clusters vs a 40 bar → **UNDERPOWERED** | measured before any score |
| **C-BLIND** (packaged firewall, imported) | the target being a function of the conditioning | verdict **CIRCULAR**; `context_leaks` = **0.0**, `blind_skill_over_majority` = **0.0** | ⛔ **NO** — the *deterministic* clause fired and it could not have failed: the positive rate is **0.00301** < `deterministic_eps` **0.02**, so the majority-class predictor alone clears `1 − eps`, and the largest accuracy gain ANY context could add is 0.00301. |

> ⚠️ **The C-BLIND verdict is DEGENERATE on this target and must not be quoted alone — its own companion numbers refute it.** `context_leaks = 0` and `blind_skill_over_majority` ≈ 0: **the ego context carries no information about the target beyond the base rate.** The `CIRCULAR` label is produced by an *accuracy* comparison on a 2–3 %-positive target, where the majority-class predictor scores 0.997 and any recall-seeking classifier necessarily scores lower — the head is *supposed* to fire more often than the base rate. The informative form of the same question — *does vision buy anything the ego state did not already give?* — is the AP-based `− head_ego` contrast above, which is the pre-registered primary comparison.

#### ROUNDABOUT — the efficiency curve (recall at a fixed camera budget)

| budget B (extra cams/frame) | realised firing rate | **head recall** | head precision | ego-head recall | kinematic-rule recall | saving vs always-on-7 |
|---|---|---|---|---|---|---|
| 0.005 | 0.0016 | **0.0402** | 0.06921 | 0.0000 | 0.0347 | 85.7 % |
| 0.01 | 0.0038 | **0.0638** | 0.04694 | 0.0208 | 0.0347 | 85.6 % |
| 0.02 | 0.0094 | **0.0860** | 0.02543 | 0.0999 | 0.0347 | 85.4 % |
| 0.05 | 0.0251 | **0.1415** | 0.01570 | 0.3010 | 0.0347 | 85.0 % |
| 0.1 | 0.0516 | **0.2427** | 0.01312 | 0.4993 | 0.0347 | 84.2 % |
| 0.2 | 0.1058 | **0.3870** | 0.01020 | 0.6519 | 0.0513 | 82.7 % |

#### ROUNDABOUT — the efficiency ledger (⚠️ read the RECALL column, not the saving)

| policy | extra cams/frame | cams/frame | saving vs always-on-7 | **recall** |
|---|---|---|---|---|
| **never escalate** (free and useless) | 0 | 1.000 | 85.7 % | **0.000** |
| **ORACLE** — fires exactly on the label | 0.0056 | 1.0056 | 85.6 % | **1.000** |
| **`head_img_ego` @ B\*** | 0.0503 | 1.0503 | 85.0 % | **0.1415** [0.0557, 0.2346] |
| **always escalate** | 6.000 | 7.000 | 0.0 % | **1.000** |

> The span of *saving* between a useless gate and a perfect oracle is **0.08 percentage points** — so **no compute-saving number here can distinguish a good gate from a useless one** (BOOST_PROGRAM §7.3). The axis that carries information is **recall at the budget**, and the lead time.

### INTERSECTION — held-out discrimination

*249,480 scored windows · 7,620 positives · 264 positive clip clusters of 1610 · base rate 0.03054 · **C-POW OK**.*

| arm | AP (episode-cluster bootstrap CI95) | AP / base | AUROC | ΔAP vs CHANCE | above chance? |
|---|---|---|---|---|---|
| `head_img_ego` ⭐ | 0.10757 [0.09068, 0.12764] | 3.5219× | 0.8221 | 0.07702 [0.06084, 0.09613] | **YES** |
| `ridge_img_ego` | 0.10809 [0.09114, 0.12805] | 3.5388× | 0.7931 | 0.07758 [0.06092, 0.09632] | **YES** |
| `head_img` | 0.07955 [0.06730, 0.09417] | 2.6043× | 0.7692 | 0.04894 [0.03735, 0.06277] | **YES** |
| `ridge_img` | 0.07767 [0.06518, 0.09245] | 2.5428× | 0.7625 | 0.04725 [0.03548, 0.06129] | **YES** |
| `head_ego` | 0.13494 [0.11502, 0.15835] | 4.4181× | 0.8652 | 0.10444 [0.08613, 0.12579] | **YES** |
| `ridge_ego` | 0.09521 [0.08076, 0.11250] | 3.1171× | 0.7844 | 0.06474 [0.05133, 0.08066] | **YES** |
| `head_img_ego_concat` | 0.08883 [0.07365, 0.10657] | 2.9083× | 0.774 | 0.05833 [0.04351, 0.07506] | **YES** |
| `heur_kin` | 0.07797 [0.06338, 0.09627] | 2.5528× | 0.6399 | 0.04776 [0.03380, 0.06400] | **YES** |
| `head_priv` ⭐C-POS | 0.49233 [0.44084, 0.54503] | 16.119× | 0.9684 | 0.46223 [0.41132, 0.51155] | **YES** |
| `head_img_shuf` ⭐C-NEG | 0.03304 [0.02823, 0.03872] | 1.0817× | 0.5231 | 0.00238 [-0.00308, 0.00793] | no |
| `ridge_img_shuf` ⭐C-NEG | 0.03143 [0.02691, 0.03687] | 1.0289× | 0.5051 | 0.00062 [-0.00446, 0.00599] | no |

| contrast | Δ | CI95 | separated? |
|---|---|---|---|
| `head_img_ego` − `head_ego` | -0.02742 | [-0.04895, -0.00620] | **YES** |
| `head_img` − `head_ego` | -0.05550 | [-0.07871, -0.03346] | **YES** |
| `head_img_ego_concat` − `head_ego` | -0.04611 | [-0.07164, -0.02197] | **YES** |
| `ridge_img_ego` − `head_ego` | -0.02686 | [-0.04215, -0.01100] | **YES** |
| `ridge_img` − `head_ego` | -0.05718 | [-0.08180, -0.03405] | **YES** |
| `ridge_ego` − `head_ego` | -0.03969 | [-0.05131, -0.02808] | **YES** |
| `heur_kin` − `head_ego` | -0.05667 | [-0.06710, -0.04618] | **YES** |
| `heur_lane_change` − `head_ego` | -0.11084 | [-0.13291, -0.09198] | **YES** |
| `heur_roundabout` − `head_ego` | -0.07077 | [-0.09323, -0.05118] | **YES** |
| `heur_intersection` − `head_ego` | -0.05667 | [-0.06710, -0.04618] | **YES** |
| recall: head_img_ego - random_at_rate | +0.1243 | [+0.0930, +0.1584] | **YES** |
| recall: head_img_ego - head_ego | +0.0297 | [-0.0085, +0.0702] | no |
| recall: ridge_img_ego - random_at_rate | +0.0916 | [+0.0672, +0.1174] | **YES** |
| recall: ridge_img_ego - head_ego | -0.0030 | [-0.0299, +0.0237] | no |

#### INTERSECTION — operating point at B\* = 0.05 extra cams/frame

| arm | firing rate | recall | precision | precision lift | caught / total | θ\* |
|---|---|---|---|---|---|---|
| `head_img_ego` | 0.0339 [0.0292, 0.0387] | 0.1585 [0.1285, 0.1921] | 0.14277 [0.11326, 0.17507] | 4.67× | 1208/7620 | 0.8942 |
| `ridge_img_ego` | 0.0233 [0.0206, 0.0262] | 0.1259 [0.1009, 0.1510] | 0.16492 [0.13272, 0.20030] | 5.40× | 959/7620 | -0.8114 |
| `head_img` | 0.0329 [0.0281, 0.0379] | 0.1147 [0.0837, 0.1438] | 0.10649 [0.07881, 0.13350] | 3.49× | 874/7620 | 0.7565 |
| `ridge_img` | 0.0218 [0.0177, 0.0266] | 0.0873 [0.0588, 0.1150] | 0.12215 [0.08489, 0.16093] | 4.00× | 665/7620 | -0.8535 |
| `head_ego` | 0.0214 [0.0192, 0.0235] | 0.1286 [0.1039, 0.1559] | 0.18369 [0.14773, 0.22435] | 6.01× | 980/7620 | 0.8020 |
| `ridge_ego` | 0.0257 [0.0229, 0.0283] | 0.1030 [0.0804, 0.1273] | 0.12245 [0.09464, 0.15288] | 4.01× | 785/7620 | -0.8298 |
| `head_img_ego_concat` | 0.0162 [0.0138, 0.0188] | 0.0769 [0.0536, 0.1016] | 0.14473 [0.10243, 0.19013] | 4.74× | 586/7620 | 0.8720 |
| `heur_kin` | 0.0260 [0.0231, 0.0288] | 0.1077 [0.0834, 0.1328] | 0.12637 [0.09688, 0.15890] | 4.14× | 821/7620 | 0.8698 |
| `head_priv` | 0.0213 [0.0190, 0.0238] | 0.4089 [0.3775, 0.4417] | 0.58582 [0.52655, 0.64335] | 19.18× | 3116/7620 | 0.8940 |
| `head_img_shuf` | 0.0300 [0.0259, 0.0343] | 0.0392 [0.0237, 0.0571] | 0.03990 [0.02375, 0.05897] | 1.31× | 299/7620 | 0.8712 |
| `ridge_img_shuf` | 0.0157 [0.0136, 0.0180] | 0.0144 [0.0058, 0.0242] | 0.02803 [0.01089, 0.04821] | 0.92× | 110/7620 | -0.9041 |
| **(a) always-escalate** | 1.0 | 1.0 | 0.03054 | 1.00× | 7620/7620 | — |
| **(b) never-escalate** | 0.0 | **0.0** | — | — | 0/7620 | — |
| **(c) random @ matched rate** | 0.0339 | 0.0340 [0.0303, 0.0381] | 0.03059 | 1.00× | — | 200 seeds |
| **oracle** | 0.03054 | 1.0 | 1.0 | 33× | 7620/7620 | — |

#### ⭐ INTERSECTION — LEAD TIME (registered minimum **1.0 s**; a high-AP zero-lead trigger FAILS)

| arm | events detected | event recall | **median lead** | p25 / p75 | frac of events at ≥ min lead | PASS? |
|---|---|---|---|---|---|---|
| `head_img_ego` | 100/277 | 0.361 | **2.0 s** | 0.7 / 3.0 s | 0.2491 | ✅ |
| `ridge_img_ego` | 104/277 | 0.3755 | **1.65 s** | 0.975 / 2.725 s | 0.2816 | ✅ |
| `head_img` | 67/277 | 0.2419 | **1.9 s** | 0.8 / 3.0 s | 0.1769 | ✅ |
| `ridge_img` | 51/277 | 0.1841 | **2.0 s** | 1.0 / 3.0 s | 0.1408 | ✅ |
| `head_ego` | 101/277 | 0.3646 | **1.7 s** | 1.0 / 2.4 s | 0.278 | ✅ |
| `ridge_ego` | 88/277 | 0.3177 | **1.5 s** | 1.0 / 2.475 s | 0.2527 | ✅ |
| `head_img_ego_concat` | 60/277 | 0.2166 | **1.9 s** | 0.5 / 2.725 s | 0.1372 | ✅ |
| `heur_kin` | 79/277 | 0.2852 | **1.9 s** | 1.0 / 2.7 s | 0.2202 | ✅ |
| `head_priv` | 216/277 | 0.7798 | **1.8 s** | 1.6 / 2.0 s | 0.7437 | ✅ |
| `head_img_shuf` | 35/277 | 0.1264 | **2.4 s** | 1.0 / 2.9 s | 0.0939 | ✅ |
| `ridge_img_shuf` | 16/277 | 0.0578 | **2.35 s** | 0.6 / 2.925 s | 0.0361 | ✅ |

#### INTERSECTION — the controls, each with its MDE

| control | what it catches | result | can it fail? |
|---|---|---|---|
| **C-POS** `head_priv` | instrument insensitivity | ΔAP vs chance 0.46223 [0.41132, 0.51155] → **YES** | it is a probe on the label's own defining quantity; a null here would mean UNPOWERED |
| **C-NEG** `head_img_shuf` | a pipeline leak | 0.00238 [-0.00308, 0.00793] → no | features permuted ACROSS clips — it must NOT separate |
| **C-NEG** `ridge_img_shuf` | a pipeline leak (closed form) | 0.00062 [-0.00446, 0.00599] → no | as above |
| **MDE** (upper 95 % bound of the C-NEG ΔAP) | the smallest effect this run can distinguish from nothing | **0.00793** | — |
| **C-POW** | reading a small-n null as a refutation | 264 positive clusters vs a 40 bar → **OK** | measured before any score |
| **C-BLIND** (packaged firewall, imported) | the target being a function of the conditioning | verdict **CIRCULAR**; `context_leaks` = **0.0**, `blind_skill_over_majority` = **0.0** | ⛔ **NO** — the *deterministic* clause did **not** fire (positive rate 0.02573 > eps 0.02); the verdict comes from the `vision_buys_nothing` clause, which compares **accuracies**. A gate that fires at ~3 % to buy recall is *designed* to lose accuracy against 'always predict negative', so on this target that clause fires for any useful classifier. |

> ⚠️ **The C-BLIND verdict is DEGENERATE on this target and must not be quoted alone — its own companion numbers refute it.** `context_leaks = 0` and `blind_skill_over_majority` ≈ 0: **the ego context carries no information about the target beyond the base rate.** The `CIRCULAR` label is produced by an *accuracy* comparison on a 2–3 %-positive target, where the majority-class predictor scores 0.9743 and any recall-seeking classifier necessarily scores lower — the head is *supposed* to fire more often than the base rate. The informative form of the same question — *does vision buy anything the ego state did not already give?* — is the AP-based `− head_ego` contrast above, which is the pre-registered primary comparison.

#### INTERSECTION — the efficiency curve (recall at a fixed camera budget)

| budget B (extra cams/frame) | realised firing rate | **head recall** | head precision | ego-head recall | kinematic-rule recall | saving vs always-on-7 |
|---|---|---|---|---|---|---|
| 0.005 | 0.0030 | **0.0146** | 0.14605 | 0.0005 | 0.0068 | 85.6 % |
| 0.01 | 0.0054 | **0.0285** | 0.16170 | 0.0055 | 0.0192 | 85.6 % |
| 0.02 | 0.0112 | **0.0626** | 0.17121 | 0.0239 | 0.0453 | 85.4 % |
| 0.05 | 0.0339 | **0.1585** | 0.14277 | 0.1286 | 0.1077 | 84.7 % |
| 0.1 | 0.0646 | **0.2806** | 0.13264 | 0.2690 | 0.2278 | 83.9 % |
| 0.2 | 0.1174 | **0.4369** | 0.11362 | 0.4516 | 0.3739 | 82.4 % |

#### INTERSECTION — the efficiency ledger (⚠️ read the RECALL column, not the saving)

| policy | extra cams/frame | cams/frame | saving vs always-on-7 | **recall** |
|---|---|---|---|---|
| **never escalate** (free and useless) | 0 | 1.000 | 85.7 % | **0.000** |
| **ORACLE** — fires exactly on the label | 0.0611 | 1.0611 | 84.8 % | **1.000** |
| **`head_img_ego` @ B\*** | 0.0678 | 1.0678 | 84.7 % | **0.1585** [0.1285, 0.1921] |
| **always escalate** | 6.000 | 7.000 | 0.0 % | **1.000** |

> The span of *saving* between a useless gate and a perfect oracle is **0.87 percentage points** — so **no compute-saving number here can distinguish a good gate from a useless one** (BOOST_PROGRAM §7.3). The axis that carries information is **recall at the budget**, and the lead time.

<!-- /TABLES:RESULTS -->

<!-- TABLES:CAMERA -->
### The multi-camera need — MEASURED, per situation

*An agent that projects into camera X but **not** into the canonical 51.4° encoder crop (H2's `T_off` machinery, per-clip `(cx,cy)` + per-clip extrinsics). Reported against the matched NOT-in-situation baseline on the same clips, because a camera that always sees something extra proves nothing.*

| situation | camera | in situation | not in situation | **lift [CI95]** | separated? |
|---|---|---|---|---|---|
| LANE CHANGE | `cross_right` | 0.44221 | 0.36595 | **1.208×** [0.964, 1.522] | no |
| LANE CHANGE | `rear_right` | 0.63916 | 0.56843 | **1.124×** [0.97, 1.274] | no |
| LANE CHANGE | `any_off_front` | 0.81635 | 0.72857 | **1.12×** [1.023, 1.234] | ✅ **YES** |
| LANE CHANGE | `cross_left` | 0.45741 | 0.41162 | **1.111×** [0.899, 1.361] | no |
| LANE CHANGE | `rear_left` | 0.69734 | 0.62957 | **1.108×** [0.949, 1.251] | no |
| LANE CHANGE | `rear_tele` | 0.66008 | 0.60413 | **1.093×** [0.948, 1.231] | no |
| LANE CHANGE | `front_tele` | 0.0 | 0.0 | **0.0×**  | no |
| ROUNDABOUT | `cross_right` | 0.81162 | 0.59185 | **1.371×** [1.037, 2.024] | ✅ **YES** |
| ROUNDABOUT | `cross_left` | 0.87127 | 0.67591 | **1.289×** [1.034, 1.674] | ✅ **YES** |
| ROUNDABOUT | `any_off_front` | 0.97017 | 0.92461 | **1.049×** [1.0, 1.125] | no |
| ROUNDABOUT | `rear_left` | 0.82889 | 0.83449 | **0.993×** [0.83, 1.152] | no |
| ROUNDABOUT | `rear_right` | 0.8022 | 0.87695 | **0.915×** [0.777, 1.037] | no |
| ROUNDABOUT | `rear_tele` | 0.6562 | 0.84835 | **0.773×** [0.62, 0.911] | no |
| ROUNDABOUT | `front_tele` | 0.0 | 0.0 | **0.0×**  | no |
| INTERSECTION | `cross_right` | 0.72304 | 0.5635 | **1.283×** [1.157, 1.448] | ✅ **YES** |
| INTERSECTION | `cross_left` | 0.69309 | 0.63025 | **1.1×** [1.002, 1.208] | ✅ **YES** |
| INTERSECTION | `any_off_front` | 0.84042 | 0.83288 | **1.009×** [0.97, 1.045] | no |
| INTERSECTION | `rear_right` | 0.69275 | 0.72835 | **0.951×** [0.862, 1.026] | no |
| INTERSECTION | `rear_left` | 0.6982 | 0.75253 | **0.928×** [0.843, 0.999] | no |
| INTERSECTION | `rear_tele` | 0.52841 | 0.70277 | **0.752×** [0.664, 0.821] | no |
| INTERSECTION | `front_tele` | 0.0 | 0.0 | **0.0×**  | no |

*Estimator: paired episode-cluster bootstrap over the clips carrying each situation, B = 400. ⚠️ **Read the LIFT, not the raw rate.** The not-in-situation rates are 0.4–0.9, i.e. an agent outside the front crop is visible in some other camera almost all the time — a raw 'need' rate is close to information-free (BOOST_PROGRAM §7.3). Only the lift over the matched baseline carries evidence.*

<!-- /TABLES:CAMERA -->

<!-- TABLES:VERDICT -->
| situation | C-POW | image arm above chance? | vision over ego? | median lead | **PRE-REGISTERED VERDICT** |
|---|---|---|---|---|---|
| **LANE CHANGE** | OK (153 clusters) | `head_img_ego`, `ridge_img_ego`, `head_img`, `ridge_img` | **none** | 1.4 s ✅ | **A− — predictable, but not *from the camera* beyond ego state** |
| **ROUNDABOUT** | UNDERPOWERED (26 clusters) | `head_img_ego`, `ridge_img_ego`, `ridge_img` | **none** | 1.7 s ✅ | **UNPOWERED** |
| **INTERSECTION** | OK (264 clusters) | `head_img_ego`, `ridge_img_ego`, `head_img`, `ridge_img` | **none** | 2.0 s ✅ | **A− — predictable, but not *from the camera* beyond ego state** |

*Evaluated in code by the rule fixed in `PRE_REGISTRATION.md` §7 — C-POS must separate, C-NEG must not, the median lead time must reach 1.0 s, and a situation with fewer than 40 held-out positive clusters is `UNPOWERED` and gets no verdict at all.*
<!-- /TABLES:VERDICT -->

<!-- TABLES:CV -->
### ⭐ The rank ladder, replicated on THESE targets (training-side CV, out-of-fold)

*A sibling stream MEASURED a monotone swamping dose-response on this same frozen v1 state (ego 3.659× → +k16 3.685× → +k64 3.000× → +k256 2.116× → +k2048 1.59×; INHERITED). This is the independent replication on the PI's three situations — **CV-AP is out-of-fold on TRAIN, grouped by chunk, and is NOT a result** (only held-out output is quotable); it is shown because it is where the ordering first appears, before the held-out side was touched.*

| arm \| config | mean CV-AP | selected? |
|---|---|---|
| `head_ego|pw20|d128|r16` | 0.07122 | ⭐ |
| `head_ego|pw50|d128|r16` | 0.07120 |  |
| `head_img_ego_concat|pw20|d128|r16` | 0.04518 | ⭐ |
| `head_img_ego_concat|pw20|d128|r64` | 0.04518 |  |
| `head_img_ego_concat|pw50|d128|r16` | 0.04107 |  |
| `head_img_ego|pw20|d128|r16` | 0.05545 | ⭐ |
| `head_img_ego|pw20|d128|r64` | 0.05420 |  |
| `head_img_ego|pw50|d128|r16` | 0.05473 |  |
| `head_img_shuf|pw20|d128|r16` | 0.01736 | ⭐ |
| `head_img_shuf|pw20|d128|r64` | 0.01624 |  |
| `head_img_shuf|pw50|d128|r16` | 0.01697 |  |
| `head_img|pw20|d128|r16` | 0.03811 | ⭐ |
| `head_img|pw20|d128|r64` | 0.03686 |  |
| `head_img|pw50|d128|r16` | 0.03816 |  |
| `head_priv|pw20|d128|r16` | 0.18568 | ⭐ |
| `head_priv|pw50|d128|r16` | 0.17696 |  |
| `ridge_ego|lam10000|r16` | 0.03966 | ⭐ |
| `ridge_ego|lam1000|r16` | 0.03993 | ⭐ |
| `ridge_ego|lam100|r16` | 0.04013 | ⭐ |
| `ridge_ego|lam10|r16` | 0.04027 | ⭐ |
| `ridge_ego|lam1|r16` | 0.04041 | ⭐ |
| `ridge_img_ego|lam10000|r16` | 0.04688 | ⭐ |
| `ridge_img_ego|lam1000|r16` | 0.04711 | ⭐ |
| `ridge_img_ego|lam100|r16` | 0.04719 | ⭐ |
| `ridge_img_ego|lam10|r16` | 0.04732 | ⭐ |
| `ridge_img_ego|lam1|r16` | 0.04749 | ⭐ |
| `ridge_img_shuf|lam10000|r16` | 0.01417 | ⭐ |
| `ridge_img_shuf|lam1000|r16` | 0.01426 | ⭐ |
| `ridge_img_shuf|lam100|r16` | 0.01439 | ⭐ |
| `ridge_img_shuf|lam10|r16` | 0.01442 | ⭐ |
| `ridge_img_shuf|lam1|r16` | 0.01443 | ⭐ |
| `ridge_img|lam10000|r16` | 0.03523 | ⭐ |
| `ridge_img|lam1000|r16` | 0.03467 |  |
| `ridge_img|lam100|r16` | 0.03443 |  |
| `ridge_img|lam10|r16` | 0.03439 |  |
| `ridge_img|lam1|r16` | 0.03439 |  |

**Every one of the ten arms selected rank 16 over rank 64.** The raw-2048 concatenation arm (`head_img_ego_concat`) is the far end of the same ladder and it is where the degradation is largest — see the held-out tables in §5.

| fold | chunks |
|---|---|
| 0 | 14 chunks |
| 1 | 13 chunks |
| 2 | 13 chunks |
| 3 | 13 chunks |
| 4 | 13 chunks |

<!-- /TABLES:CV -->