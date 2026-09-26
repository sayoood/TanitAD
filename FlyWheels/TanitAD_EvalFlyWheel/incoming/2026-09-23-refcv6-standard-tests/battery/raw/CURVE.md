### refcv6 across checkpoints — a TABLE, not a fit

⛔ **Two experiments (SPEC A5).** Checkpoints at step ≤ 34,500 are pre-switch: *F3 detach-only, F4 on the last layer only; tactical labels ~0.37 s early*. A later checkpoint (the FINAL) is the post-switch *hybrid: F3 cascade loss + true label clock from step 34,500* (resumed on 82c2331). A difference across the switch MIXES training time with the fix and is never attributed to the fix alone; no learning-curve fit spans step 34,500. TACTICAL is not in this table: each tag's `TACTICAL_CLOCKS.md` carries it under both label clocks.

Steps: step5000 = 5000, step30000 = 30000, final = None


#### ADE paired cells (m, b − a, T1), inference seed 0

| cell | step5000 | step30000 | final |
|---|---|---|---|
| os − ha0_ext (echo) | +0.0885 [+0.0695, +0.1094] **sep** | +0.0225 [+0.0068, +0.0395] **sep** | — |
| os − ha (hold) | +0.0761 [+0.0569, +0.0974] **sep** | +0.0101 [-0.0066, +0.0280] | — |
| os − ha0 (CV) | -0.2993 [-0.3582, -0.2419] **sep** | -0.3653 [-0.4294, -0.3049] **sep** | — |
| os − refcv4b | +0.0797 [+0.0574, +0.1017] **sep** | +0.0137 [-0.0029, +0.0317] | — |
| os − refcv5-v2 s0 | +0.0681 [+0.0456, +0.0909] **sep** | +0.0021 [-0.0160, +0.0212] | — |
| nav withheld − os | +0.0246 [+0.0157, +0.0340] **sep** | +0.0266 [+0.0167, +0.0361] **sep** | — |
| max-speed withheld − os | +0.0020 [-0.0010, +0.0049] | +0.0000 [-0.0028, +0.0030] | — |
| S6: os − ha0_ext | -0.4729 [-0.7531, -0.1780] **sep** | -0.7475 [-1.0603, -0.4409] **sep** | — |
| S6: os − refcv4b | +0.3524 [+0.1955, +0.5140] **sep** | +0.0779 [-0.0651, +0.2413] | — |
| A4 L2 blend(os, ha) − ha0_ext | -0.0122 [-0.0200, -0.0042] **sep** | -0.0364 [-0.0464, -0.0265] **sep** | — |
| A4 L2e blend(os, echo) − ha0_ext (diag.) | -0.0202 [-0.0277, -0.0125] **sep** | -0.0419 [-0.0519, -0.0321] **sep** | — |
| A4 L1 seed-average − ha0_ext | +0.0831 [+0.0640, +0.1043] **sep** | +0.0172 [+0.0016, +0.0336] **sep** | — |
| A4 L3 eps0 − ha0_ext | — | — | — |

#### ADE paired cells (m, b − a, T1), inference seed 1

| cell | step5000 | step30000 | final |
|---|---|---|---|
| os − ha0_ext (echo) | +0.0885 [+0.0695, +0.1092] **sep** | +0.0224 [+0.0070, +0.0387] **sep** | — |
| os − ha (hold) | +0.0761 [+0.0569, +0.0971] **sep** | +0.0100 [-0.0066, +0.0272] | — |
| os − ha0 (CV) | -0.2993 [-0.3590, -0.2417] **sep** | -0.3654 [-0.4300, -0.3045] **sep** | — |
| os − refcv4b | +0.0797 [+0.0577, +0.1023] **sep** | +0.0136 [-0.0028, +0.0313] | — |
| os − refcv5-v2 s0 | +0.0681 [+0.0455, +0.0909] **sep** | +0.0020 [-0.0160, +0.0209] | — |
| nav withheld − os | +0.0232 [+0.0144, +0.0324] **sep** | +0.0258 [+0.0158, +0.0363] **sep** | — |
| max-speed withheld − os | +0.0014 [-0.0015, +0.0041] | -0.0018 [-0.0049, +0.0010] | — |
| S6: os − ha0_ext | -0.4601 [-0.7453, -0.1658] **sep** | -0.7472 [-1.0537, -0.4377] **sep** | — |
| S6: os − refcv4b | +0.3652 [+0.2097, +0.5275] **sep** | +0.0782 [-0.0706, +0.2431] | — |
| A4 L2 blend(os, ha) − ha0_ext | -0.0123 [-0.0200, -0.0045] **sep** | -0.0361 [-0.0461, -0.0266] **sep** | — |
| A4 L2e blend(os, echo) − ha0_ext (diag.) | -0.0203 [-0.0277, -0.0125] **sep** | -0.0418 [-0.0517, -0.0322] **sep** | — |

#### Levels and gates

| tag | step | G0-A2 | os ADE 0–2 s [CI] | os ADE 1–6 s | bars (BAR-R6-1..5) | T-FLIP | OBEDIENCE |
|---|---|---|---|---|---|---|---|
| step5000 | 5000 | PASS | 0.3771 [0.3458, 0.4138] | 3.0749 | FAIL/FAIL/FAIL/FAIL/PASS | FAIL | FAIL |
| step30000 | 30000 | PASS | 0.3111 [0.2847, 0.3409] | 2.8004 | FAIL/FAIL/FAIL/FAIL/PASS | FAIL | FAIL |
| final | — | — | — | — | — | — | — |
