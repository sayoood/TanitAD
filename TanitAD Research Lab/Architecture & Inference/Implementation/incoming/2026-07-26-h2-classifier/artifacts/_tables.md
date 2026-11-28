### Discrimination — held-out, (camera, frame) pairs

*Base rate (chance AP) = **0.00305** (306 positives in 100238 pairs, 322 clips, 35 of them positive).*

| arm | AP (episode-cluster bootstrap CI95) | AP / base rate | AUROC |
|---|---|---|---|
| `head_img_ego` | 0.00795 [0.00285, 0.03071] | **2.60x** | 0.6832 |
| `head_img` | 0.00551 [0.00249, 0.01285] | **1.80x** | 0.6611 |
| `head_ego` | 0.01276 [0.00397, 0.03615] | **4.18x** | 0.6782 |
| `heur_speed` | 0.00199 [0.00108, 0.00320] | **0.65x** | 0.2965 |
| `heur_decel` | 0.00767 [0.00286, 0.01914] | **2.51x** | 0.5796 |

### Is any arm above CHANCE? (paired ΔAP against a constant score)

*A constant score has AP equal to the base rate **inside every bootstrap draw**, so this — not "does the AP interval clear the full-sample base rate" — is the correct above-chance test.*

| arm | ΔAP vs chance | CI95 | above chance? |
|---|---|---|---|
| `head_img_ego` | +0.00268 | [-0.00278, +0.02639] | no |
| `head_img` | +0.00024 | [-0.00421, +0.00918] | no |
| `head_ego` | +0.00749 | [-0.00113, +0.03170] | no |
| `heur_speed` | -0.00328 | [-0.00835, -0.00015] | below chance |
| `heur_decel` | +0.00241 | [-0.00314, +0.01535] | no |

### Paired AP deltas vs the primary (`head_img_ego`)

| contrast | ΔAP | CI95 | separated? |
|---|---|---|---|
| head_img_ego - head_img | +0.00244 | [-0.00003, +0.01804] | no |
| head_img_ego - head_ego | -0.00481 | [-0.01864, +0.00560] | no |
| head_img_ego - heur_speed | +0.00596 | [+0.00144, +0.02833] | **YES** |
| head_img_ego - heur_decel | +0.00027 | [-0.00633, +0.01462] | no |

### The operating point (pre-registered `theta*` fixed on TRAIN out-of-fold)

Budget `B* = 0.05` extra camera activations/frame ⇒ target camera-frame rate 0.0250; **realised 0.0315** (ratio 1.26x) — the calibration-transfer test.

| arm | firing rate | extra cams/frame | recall | precision | precision lift | recall on behavioural slice | missed |
|---|---|---|---|---|---|---|---|
| `head_img_ego` | 0.0315 [0.0221, 0.0424] | 0.0630 | 0.1373 [0.0188, 0.3199] | 0.01331 [0.00161, 0.03530] | 4.36 [0.62, 10.59] | 0.2778 | 264/306 |
| `head_img` | 0.0154 [0.0102, 0.0210] | 0.0308 | 0.0654 [0.0000, 0.1732] | 0.01295 [0.00000, 0.03819] | 4.24 [0.00, 11.80] | 0.0000 | 286/306 |
| `head_ego` | 0.0178 [0.0145, 0.0214] | 0.0356 | 0.1699 [0.0274, 0.3648] | 0.02915 [0.00392, 0.06388] | 9.55 [1.60, 20.58] | 0.2778 | 254/306 |
| `heur_ego_both` | 0.0274 [0.0209, 0.0342] | 0.0547 | 0.1144 [0.0277, 0.2420] | 0.01276 [0.00266, 0.02692] | 4.18 [1.07, 8.86] | 0.5833 | 271/306 |
| `heur_ego_both_rate_matched` | 0.0274 [0.0209, 0.0342] | 0.0547 | 0.1144 [0.0277, 0.2420] | 0.01276 [0.00266, 0.02692] | 4.18 [1.07, 8.86] | 0.5833 | 271/306 |
| `random_at_rate` | 0.0315 [0.0304, 0.0325] | 0.0630 | 0.0359 [0.0198, 0.0564] | 0.00349 [0.00157, 0.00587] | 1.14 [0.63, 1.80] | 0.0278 | 295/306 |

| paired recall delta | Δ | CI95 | separated? |
|---|---|---|---|
| head_img_ego - heur_ego_both | +0.0229 | [-0.1106, +0.1455] | no |
| head_img_ego - heur_ego_both_rate_matched | +0.0229 | [-0.1106, +0.1455] | no |
| head_img_ego - random_at_rate | +0.1013 | [-0.0158, +0.2727] | no |
| head_img - heur_ego_both_rate_matched | -0.0490 | [-0.1591, +0.0248] | no |
| head_img - random_at_rate | +0.0294 | [-0.0417, +0.1329] | no |
| head_ego - heur_ego_both_rate_matched | +0.0556 | [-0.0230, +0.1441] | no |
| head_ego - random_at_rate | +0.1340 | [-0.0110, +0.3247] | no |

### The efficiency trade-off CURVE (not a point)

| B (extra cams/frame) | head realised rate | **head recall** | head precision | head recall (behavioural) | heuristic recall | random recall | saving vs always-on-3 | saving vs always-on-7 |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.0083 | **0.0392** | 0.02885 | 0.0000 | 0.0000 | 0.0027 | 0.664 | 0.856 |
| 0.010 | 0.0165 | **0.0686** | 0.02536 | 0.0833 | 0.0033 | 0.0054 | 0.661 | 0.855 |
| 0.020 | 0.0313 | **0.0752** | 0.01466 | 0.1389 | 0.0196 | 0.0103 | 0.656 | 0.853 |
| 0.050 | 0.0630 | **0.1373** | 0.01331 | 0.2778 | 0.1144 | 0.0247 | 0.645 | 0.848 |
| 0.100 | 0.1337 | **0.2026** | 0.00925 | 0.3611 | 0.1144 | 0.0500 | 0.622 | 0.838 |
| 0.200 | 0.2448 | **0.2843** | 0.00709 | 0.3611 | 0.3856 | 0.1001 | 0.585 | 0.822 |

### The efficiency ledger — where the saving actually comes from

| policy | extra cams/frame | cams/frame | saving vs always-on-3 | saving vs always-on-7 | recall of `L2_trigger` |
|---|---|---|---|---|---|
| **never escalate** (free and useless) | 0 | 1.000 | 66.7 % | 85.7 % | **0.000** |
| **L2 ORACLE** — fires exactly on the label | 0.0061 | 1.0061 | 66.5 % | 85.6 % | **1.000** |
| **`head_img_ego` @ B\*** | 0.0630 [0.0442, 0.0847] | 1.0630 | 64.5 % [63.8, 65.2] | 84.8 % [84.5, 85.1] | **0.1373** [0.0188, 0.3199] |
| **always escalate** | 2.000 | 3.000 | 0.0 % | 57.1 % | **1.000** |

> The saving is set by the POLICY SHAPE, not by classifier quality: between the oracle and our operating point the saving vs always-on-7 moves by well under a percentage point. **What the classifier has to earn is RECALL at that budget** — which is why recall, not saving, is the axis the verdict is decided on.

### C12 — the LABEL's own structure, before any model

| quantity | value |
|---|---|
| `T_off` (`a_req_off ≥ τ*`) rate | **0.634 %** (318 frames) |
| `T_seen` (`a_req_seen < τ*`) rate | **96.724 %** (48477 frames) |
| composite `L2_trigger` rate (frame level) | 0.611 % (306 frames) |
| **P(trigger \| `T_off`)** | **0.9623** |
| P(trigger \| `T_seen`) | 0.00631 |

### C12 — the composite decomposed into its two conjuncts

| conjunct | base rate | AP (CI95) | AP / base | AUROC |
|---|---|---|---|---|
| `T_off` | 0.0063 | 0.00846 [0.00404, 0.02125] | **1.33x** | 0.5818 |
| `T_seen` ⛔ **NOT READ — mis-posed instrument, amendment A1** | 0.9672 | 0.97673 [0.96868, 0.98368] | **1.01x** | 0.5748 |
| `NOT_T_seen` read off the MIS-POSED head ⛔ **NOT READ** | 0.0328 | 0.03779 [0.02781, 0.04958] | **1.15x** | 0.4252 |

> ⛔ The `T_seen` row and its complement are printed for completeness and **are not read**: `T_seen` is a 96.7 %-positive target and the pre-registered BCE + `pos_weight` recipe up-weights its MAJORITY class, so that head carries no information about the rare side. The corrected diagnostic is the next table.

### C12 — the CORRECTED conjunct diagnostic (`NOT_T_seen`), amendment A1

*Target: NOT_T_seen = (a_req_seen >= tau*) — an agent INSIDE the encoder crop requires braking >= 0.5 m/s^2. **1642 positives** in 50119 frames (3.28 %), **101 of 322 clips positive** — 5.4x the composite's positives and far better powered.*

| arm | AP (CI95) | AP / base | ΔAP vs chance | CI95 | above chance? | AUROC |
|---|---|---|---|---|---|---|
| `head_img_ego` | 0.05205 [0.03679, 0.07550] | **1.59x** | +0.00601 | [-0.00040, +0.03947] | no | 0.6544 |
| `head_img` | 0.04914 [0.03452, 0.07926] | **1.50x** | +0.00310 | [-0.00291, +0.04284] | no | 0.6349 |
| `head_ego` | 0.12263 [0.08052, 0.17431] | **3.74x** | +0.07659 | [+0.05055, +0.13529] | **YES** | 0.7776 |

### Sensitivities

| stratum | n positives | base rate | AP (CI95) | AP / base |
|---|---|---|---|---|
| `encoder_unseen_clips` | 34 | 0.00162 | 0.00169 [0.00044, 0.00443] | **1.04x** |
| `junction_in` | 73 | 0.00864 | 0.01364 [0.00215, 0.03535] | **1.58x** |
| `junction_out` | 233 | 0.00254 | 0.00784 [0.00210, 0.03834] | **3.09x** |
| `residual_scope_target` | 102 | 0.00102 | 0.00183 [0.00068, 0.00366] | **1.80x** |

### Measured compute (A40, pod2)

- one frozen encoder+readout pass, batch 32: **3.244 ms** / camera-frame · analytic **23.40 GMAC**
- one head forward, batch 32: **18.9 us** / frame · analytic **16.9 MMAC**
- **head / encoder = 0.072 % of one camera pass (MACs), 0.58 % (wall-clock)** — the gate is nearly free, so the saving is set by the firing rate, not by the gate.

### Training-side cross-validation (out-of-fold on TRAIN, grouped by chunk)

*TRAIN: 31,032 windows · CV base rate for reference is the mean of the two per-camera base rates.*

| arm \| target | selected config | selected epoch | **CV-AP** | full CV-AP-vs-epoch curve (first 6) | params |
|---|---|---|---|---|---|
| `head_img_ego|trigger` | pos_weight 20, d 256 | 1 | **0.0097** | 0.0097, 0.0040, 0.0035, 0.0030, 0.0037, 0.0037 … | 2,173,699 |
| `head_img|trigger` | pos_weight 20, d 256 | 1 | **0.0095** | 0.0095, 0.0047, 0.0031, 0.0033, 0.0030, 0.0030 … | 2,173,187 |
| `head_ego|trigger` | pos_weight 20, d 128 | 14 | **0.0198** | 0.0056, 0.0058, 0.0061, 0.0097, 0.0056, 0.0056 … | 415,107 |
| `head_img_ego|T_off` | pos_weight 20, d 128 | 2 | **0.0078** | 0.0058, 0.0078, 0.0056, 0.0068, 0.0063, 0.0055 … | 677,122 |
| `head_img_ego|T_seen` | pos_weight 100, d 128 | 8 | **0.9807** | 0.9777, 0.9731, 0.9723, 0.9737, 0.9754, 0.9736 … | 677,122 |
| `head_img_ego|NOT_T_seen` | pos_weight 100, d 256 | 1 | **0.0296** | 0.0296, 0.0281, 0.0246, 0.0271, 0.0249, 0.0235 … | 2,173,442 |
| `head_img|NOT_T_seen` | pos_weight 100, d 128 | 2 | **0.0327** | 0.0280, 0.0327, 0.0308, 0.0251, 0.0244, 0.0224 … | 676,866 |
| `head_ego|NOT_T_seen` | pos_weight 100, d 256 | 18 | **0.0967** | 0.0548, 0.0523, 0.0604, 0.0629, 0.0623, 0.0570 … | 1,649,154 |

| fold | chunks |
|---|---|
| 0 | 0036, 0928 |
| 1 | 0170, 1852 |
| 2 | 0174, 1870 |
| 3 | 0834, 2433 |
| 4 | 0868, 2503 |
