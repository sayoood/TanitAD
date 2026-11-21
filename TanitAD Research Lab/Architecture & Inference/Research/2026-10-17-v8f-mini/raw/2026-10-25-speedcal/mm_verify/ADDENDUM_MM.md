

## MM independent re-read (2026-10-26, Master Mind; descriptive; it changes no row of the read)

A build agent produced the read above. Before it was quoted anywhere, the Master Mind re-read the two decision rows with code that shares nothing with the agent's reader (`raw/2026-10-25-speedcal/mm_verify/verify_sc.{py,log,json}`):
* its own ADE6, computed from the banked plans against HSPEC's eval GT, with the window order checked;
* its own episode-cluster bootstrap (numpy, B = 2,000, rng 20261026).

The 12 eval npz it reads are byte-identical to the banked copies (`cmp`, log header).

| check | s0 | s1 |
|---|---|---|
| my ADE6 from the plans vs the banked per-window ADE6 (own, cal, ceiling, identity), max abs | 0.0 | 0.0 |
| identity arm vs own, ADE6 max abs (control 1) | 0.0 | 0.0 |
| the lever ran: calibrated plans vs own, max abs (m) | 6.80 | 1.56 |
| the lever ran: windows whose T code moved | 100 % | 100 % |
| PRIMARY ADE6, SPEEDCAL − X_s | +0.0374 [−0.0120, +0.0895] | −0.0058 [−0.0154, +0.0033] |
| CEILING (T0), eval-GT map − X_s | +0.0296 [−0.0262, +0.0893] | −0.0075 [−0.0171, +0.0023] |

* **Every point estimate agrees exactly.** The intervals agree to the bootstrap draw.
* **The outcome row stands: INFORMATION-LIMITED.** PRIMARY is not separated at either seed, and CEILING is above −F = −0.1207 at both.

**What the result is, and what it is not:**
* **It is a clean elimination.** No affine map of T's own five speed cells, not even one fitted on the eval GT, moves ADE6 by F. Yet the GT speed cells (T0) cut ADE6 by 0.72 / 0.67 m (`READ.json` `ade6_m`). So speed is still the largest lever, and the error O responds to is per-window, not a global head bias.
* ⚠ **The tier is T1-class open loop** (CEILING and the `gt` arm are T0). One estimator question is answered: another draw of episodes. The second seed is the replicate.
* ⚠ **Disclosed by the build agent:** the fit-set GT is the trainer's `join_train.npz` (the same v9 md5 as all four trainings), not `v9_labels_train.npz`, which sat only in directories the task could not read.
* **Committed next, quoted from the PREREG:** the speed lever moves to perception (temporal context, encoder adaptation), and that needs compute beyond box 2. The arm for it already exists and is gated: `PREREG_V8F_LORA.md`, whose AR-2 smoke PASSED in run 3 and projects 470 h per seed on box 2, so its venue rule names the A40. The go is the PI's: `D-V8F-SPEED-PERCEPTION` in `PI_DECISION_QUEUE.md`.
