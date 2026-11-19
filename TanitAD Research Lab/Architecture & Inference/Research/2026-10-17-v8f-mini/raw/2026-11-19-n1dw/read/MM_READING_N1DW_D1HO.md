# MM reading (interim): PREREG_V8F_N1DW on D1-HO. The roundabout-window weight gains at seed 0 at both weights and not at seed 1; every guard-dependent label is PENDING-GUARD (2026-11-19)

**Status:** INTERIM. The deciding D1-HO cells are final. The eval-141 guard and the STRATEGIC family are not computed yet.
* Their runner is staged on the dev box, started at 18:30Z, and waits for another session's read to free memory.
* No outcome label is assigned here. The verdict reading follows the guard.

**Inputs**
- Card: `../PREREG_V8F_N1DW.md` (sha256 59f2b7f9) with Amendment 1, frozen before any N1DW arm existed. Build record: `../build/MM_BUILD_N1DW.md`.
- Result: `RESULT_N1DW_D1HO.json` (sha256 6473f154), printout `READ_N1DW_D1HO.txt` (sha256 5c1d25a8), code in `code/` (`n1dw_ho_read.py` 4ff5e4e7, `read_lib_pod.py` 2dc6a973).
- Checkpoints and dumps: banked PRIVATE on HF `Sayood/tanitad-v8f-strat1-runs`, all hub-equal by sha256 (records in `bank/`):
  - `n1dw/runs/<arm>/ckpt_last.pt`: N1DW4_s0 994a4f33, N1DW4_s1 d5fe963f, N1DW2_s0 b9ec88e1, N1DW2_s1 07411aaf;
  - `n1dw/read/n1dw_d1ho_dumps.tar` (aca3571e).

**Evidence and estimator**
- MEASURED. Tier **T1-kin**: the action-closed rolled track.
- Population: D1-HO, 5,736 windows / 497 held-out episodes. Pod CPU.
- Estimator: paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only. Training variance is the two seeds; there is no inference variance (the planner does not sample).

## Controls: all read their known values
| control | value | status |
|---|---|---|
| K-RE (N1R / N1D / N1D-zero recomputed from dumps vs `RESULT_NAVRBD.json`) | 6 of 6 equal | PASS |
| K1 (N1R_s0 self-pair) | delta 0.0, CI [0.0, 0.0] | PASS |
| K-CONST (majority `turn_right`) | 0.338215 | PASS |
| K-CLIP (share of logged steps with grad_norm > 1.0) | 60 / 60 for every N1DW arm, as for N1D | printed beside every label, as the card says |

**Bar:** max(0.0383, F_HO 0.042364, |N1D seed draw| 0.0303, |N1DW_w seed draw|) = **0.0424** at both weights.

## Deciding cell: D1-HO TAC-LAT, N1DW - N1R (paired, same windows)
| arm | accuracy | N1DW - N1R [95% CI] | gain (>= 0.0424 and lo > 0) | beside: N1DW - N1D |
|---|---|---|---|---|
| N1DW2_s0 | 0.6353 | **+0.0448** [+0.0321, +0.0574] | **yes** | +0.0157 [+0.0061, +0.0252], separated |
| N1DW2_s1 | 0.6574 | +0.0246 [+0.0127, +0.0367] | no | +0.0075 [-0.0032, +0.0178] |
| N1DW4_s0 | 0.6358 | **+0.0453** [+0.0336, +0.0569] | **yes** | +0.0162 [+0.0061, +0.0264], separated |
| N1DW4_s1 | 0.6471 | +0.0143 [+0.0012, +0.0283] | no | -0.0028 [-0.0142, +0.0082] |

- References: N1R 0.5905 / 0.6328, N1D 0.6196 / 0.6499.
- Mean two-seed N1DW - N1R: w2 +0.0347, w4 +0.0298.
- **What it says:** at seed 0 the weight adds about +0.016 over N1D at both weights. At seed 1 it adds nothing separated. The seed-0 gains clear the bar by 0.0024 and 0.0029, less than one tenth of the bar.

## As far as D1-HO allows (card section 7)
- RBW-GAIN and RBW-GAIN-ONE-COST: not fired (no weight gains at both seeds).
- RBW-ONE-SEED: **PENDING-GUARD**. Its D1-HO part holds at both weights.
- RBW-COSTS: PENDING-GUARD. D1-HO shows no cost at both seeds at any weight.
- RBW-SPLIT, RBW-DOSE, RBW-NO-GAIN, RBW-FLOOR-HIGH: not fired.

## The four families on D1-HO (beside, no bar; full table in `READ_N1DW_D1HO.txt`)
- **TACTICAL.**
  - TAC-LON, N1DW - N1R: +0.025 to +0.027 at all four arms, separated.
  - TAC-LON, N1DW - N1D: separated at s0 only (+0.0188 at w2, +0.0176 at w4).
  - D2-HO turn recall and false-turn share: mixed. At N1DW4_s0 both read lower than N1R (-0.021 recall, -0.049 false-turn, both separated). At N1DW4_s1 both read higher (+0.014 and +0.020, both separated).
- **LATERAL**, N1DW - N1D:
  - at 2-6 s, heading and yaw-rate errors are lower at all four arms, separated (heading -1.03 to -1.19 deg, yaw rate -3.65 to -6.12 deg/s);
  - at 2-6 s, cross-track and curvature errors are lower at all four arms, separated in 6 of these 8 cells (not separated: N1DW4_s0 cross-track, N1DW2_s1 curvature);
  - at 0-2 s, all lateral errors are lower or not separated; cross-track moves by at most 0.0046 m.
- **LONGITUDINAL**, N1DW - N1D:
  - at 2-6 s, speed, along-track and acceleration errors are lower, separated, at all four arms;
  - at 0-2 s the along-track error is HIGHER at three arms, separated, by +0.0025 to +0.0060 m;
  - N1DW2_s1's 0-2 s ADE is higher by +0.0036 m [+0.0010, +0.0062].
  - Headway and time-gap cells: NOT-COMPUTABLE, n = 0 (no lead block for held-out clips).
- **STRATEGIC:** computed on eval-141 by the guard runner and reported in full there.

## Next
- When the guard runner finishes: the verdict reading, with the guard, all four families and STRATEGIC in full.
- If RBW-ONE-SEED holds, the card's next step is s10/s11 at that weight, which needs a PI go.
- The decoder-side lever that reads T's code directly (draft card N1DC) is held for the PI's go together with that verdict.
