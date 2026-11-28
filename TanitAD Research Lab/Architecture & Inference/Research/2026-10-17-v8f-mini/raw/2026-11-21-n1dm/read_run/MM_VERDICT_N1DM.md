# N1DM (a mode-routed plan head: K = 3 residual branches of `head_o`, weighted by the detached route posterior): verdict of record N1DM-COST (MM, 2026-11-22)

## Card, arms and method
* **Card:** `PREREG_V8F_N1DM.md` (958d143d) with Amendment 1 (96847be8) and Amendment 2 (70ff416f). Every one was frozen before any N1DM arm trained or any number existed.
* **Arms:** N1DM_s0 and N1DM_s1: 3,000 steps on N1DH's recipe and init, with argv = N1D's argv plus exactly `--aux-route modes --w-aux-route 0.5`. Comparator: the banked N1D. Code of record 573ad839.
* **Evidence class:** MEASURED (ours). The pod trained the arms and made the dumps. The dev box ran the reads with zero training.
* **Tiers:** executor of record T1-kin/hb0.2 (kinematic rollout on the model's own actions); the old executor sits beside it; Z-POST and STRATEGIC at T0.
* **Estimator:** paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES"; training variance is the two seeds. The guard floor is F2 (the banked old-executor C0 replicate); no hb0.2 replicate exists.
* **Files:** `out_real/RESULT_N1DM_D1HO.json`, `eval141/out/RESULT_N1DM_EVAL141.json`, `RESULT_N1DM_ZPOST.json`, `KLIM_s0.json` / `KLIM_s1.json`, `RESULT_N1DM_s{0,1}_KOWNSPLIT.json`.

## The run (pod, MEASURED)
* G-SUITE PASS: 0 regressions, 0 new-test failures, 122 new tests pass.
  * Five tests went from skipped to failing. The cause is the pod tree's composition (the eval-141 grid file creates the v8F-mini directory, `test_v8f_wc.py:95`), not N1DM code.
* Launch gate: 8 of 8 checks PASS at both seeds.
* zerograd PASS: 258 of 258 tensors trained, including `route_head` and `route_br.0/1/2`.
* Clip 1.0 on every logged step.
* The arms trained beside RHK's seed-1 bank step under Amendment 2.

## Deciding cell: D1-HO kinematic TAC-LAT accuracy, N1DM - N1D (5,736 windows, 497 episodes, executor of record)
| seed | N1DM - N1D [interval] | bar | reading |
|---|---|---|---|
| 0 | -0.0237 [-0.0358, -0.0129] | 0.0502 | separated, the losing direction; inside the bar: no gain, no cost |
| 1 | +0.0096 [-0.0031, +0.0220] | 0.0502 | not separated: no gain |

* The bar is max(0.0383, F_HO 0.0424, |N1D draw| 0.0169, |N1DM draw| 0.0502), so N1DM's own seed spread sets it.
* **Beside, no bar:**
  * N1DM - N1DH: s0 -0.0207 (separated), s1 -0.0054.
  * N1DM - N1R: s0 +0.0007, s1 +0.0263 (separated).
  * T0 TAC-LAT: s0 -0.0183 [-0.0256, -0.0112], s1 +0.0012.
* **Z-POST** (T0, D1-HO): route head N1DM_s0 0.6832 [0.6637, 0.7028] and N1DM_s1 0.7109 [0.6919, 0.7284], against the logistic 0.685. Decoded plan 0.6083 / 0.6362.
* **Z-USE** (real minus branches zeroed): +0.0230 [+0.0100, +0.0370] and +0.0239 [+0.0115, +0.0375].
* **Z-UNIF** (real minus uniform p): +0.0106 [+0.0019, +0.0195] and +0.0132 [+0.0026, +0.0240].
* So the branches and the per-row selection both act inside the trained model, but the total does not beat N1D.
* **K-OWN-SPLIT** (no bar; eval-141 probe of 472 rows): the head is +0.030 / +0.042 more accurate on teacher-forced rows, about the same as N1DH (+0.034 / +0.038).

## Guard: eval-141 (2,399 windows, 141 episodes), N1DM - N1D, executor of record, floor F2: GUARD-COST-BOTH-SEEDS
| family | cell | s0 | s1 |
|---|---|---|---|
| LONGITUDINAL | operative 0-2 s along_mae | +0.0123 m [+0.0061, +0.0217] COST (floor 0.0005) | +0.0124 m [+0.0068, +0.0211] COST |
| | operative 0-2 s speed_mae | +0.0084 m/s [+0.0033, +0.0155] COST (floor 0.0031) | +0.0081 m/s [+0.0029, +0.0147] COST |
| | operative 0-2 s accel_mae | -0.0142 (within floor 0.0225) | -0.0258 [-0.0364, -0.0161] GAIN |
| LATERAL | operative 0-2 s cross_mae | +0.0015 m [+0.0000, +0.0029] COST (floor 0.0004) | not separated |
| | tactical 2-6 s heading | not separated | -0.52 deg (within floor 1.57) |
| TACTICAL | TAC-LON | -0.0275 [-0.0488, -0.0079] COST (floor 0.0150) | not separated |
| (all) | operative 0-2 s ade | +0.0122 m [+0.0061, +0.0212] COST (floor 0.0005) | +0.0124 m [+0.0067, +0.0209] COST |
| STRATEGIC | ROUTE-COST (RC-A cells, T0) | none | none |

* Distance-keeping cells are computed; none separates at either seed.
* STRATEGIC is read in full (the layer is unpaused): no route cost, and no S1-slot route-completion cell separated at the 30, 50 or 80 m arcs.
* The old executor, beside, reads the same operative 0-2 s cells to the last digit, because the heading bound does not bind in 0-2 s. Its 13 longer-horizon cells differ.
* **Named risk read first** (the card's HYPOTHESIS: a more direct trainable path into O's plan moves operative along-track precision): N1DH's cost cells read about 5x larger here (along +0.012 m against N1DH's +0.0016 / +0.0027 m). That is consistent with the hypothesis, and it is the card's named risk, not a new finding.

## Label
* The card's table, in its precedence: N1DM-COST fires when the "guard cost at both seeds". COST holds at both, so the label is **N1DM-COST, not adopted**.
* The D1-HO + Z-POST part alone had fired POST-WEAK (s0 head 0.6832, a 0.0018 miss of 0.685, interval [0.6637, 0.7028]). The card makes COST override it, so it is reported beside the label, not as it.

## Stamp defect (no number affected)
The read tools' headers print `card PREREG_V8F_N1DM.md sha256 N1DM-DRAFT`: the card constant was never updated to 958d143d at the freeze. The card of record is 958d143d.

## What it says
* N1DH (a posterior input) and N1DM (per-mode output branches) both leave the decoded plan where N1D had it, while the route head reads the decision (0.68 to 0.71). Two different decoder paths for the decision have now given no gain.
* Both cost operative 0-2 s precision, and the more direct path (N1DM) costs more.

## Next lever (card section 3, N1DM-COST row, committed in advance)
The cost family's own card: a same-seed hb0.2 replicate of N1D (about 0.3 pod-h, ESTIMATED). It gives the operative cost cells a real floor at this executor. F2 comes from the old executor, and 0.0005 m is far below the 0.012 m read here. The replicate is a new arm and needs a PI go.
