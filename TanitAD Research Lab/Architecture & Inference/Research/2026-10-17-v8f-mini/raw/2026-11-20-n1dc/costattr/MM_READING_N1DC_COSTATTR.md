# Reading: PREREG_V8F_N1DC_COSTATTR (sha256 295db8eb). INVALID as the card is written: control K-ADD failed on its form. No label (2026-11-20)

MEASURED (ours). T1-kin/hb0.2, eval-141, 2,399 windows / 141 episodes. Paired episode-cluster bootstrap B 2,000 seed 0; it answers "another draw of EPISODES" only. Zero training: the zeroed-skip copies U_s{0,1}, sha256 11cfe22a / 4074f70f, equal to the pod's.

## Controls
* K-COPY PASS: 278 tensors per checkpoint; `o_code.weight` all zero; the other 277 equal N1DC.
* K-SAME / K-FL-HB PASS. K1 PASS (exactly 0.0). K-GUARD PASS (exact on every field).
* **K-ADD FAILED:** max abs 0.0492 at s0 and 0.0091 at s1; the bound was 1e-12.
  * The cause is the card's wording. It said "window means are linear", but every pair is read on the windows finite in BOTH of its arms (`cell_delta`).
  * The yaw-rate, curvature and headway / time-gap cells have NaN windows that differ by arm, so P1 + P2 is not N1DC - N1D there: 7 cells at s0 and 6 at s1, including the s1 cost cell `dk_hw_abs_err_m`.
  * On the windows finite in all three arms the identity reads exactly 0.0 on every cell (`kadd_diag.json`). The reader's bookkeeping is right; the control as written asked the wrong question.
* ⇒ By the programme's rule for a control that fails on its form, the card is reported as written: **INVALID, no label.** A changed form would need a new card and a new run.

## POST-HOC reading, beside, not of record
Each cell reads delta [95 % CI]; C = cost.

| cost cell (its seed) | P1 = U - N1D | P2 = N1DC - U |
|---|---|---|
| TAC-LON (s0) | -0.0263 [-0.0459, -0.0075] C | +0.0108 [-0.0054, +0.0296] |
| operative 0-2 s ade_m (s1) | +0.0020 [+0.0005, +0.0035] C | -0.0004 [-0.0010, +0.0002] |
| operative 0-2 s along_mae_m (s1) | +0.0020 [+0.0008, +0.0032] C | -0.0008 [-0.0012, -0.0003], separated, lower error |
| operative 0-2 s headway abs err (s1) | +0.0209 [+0.0026, +0.0456] C | -0.0034 [-0.0140, +0.0061] |

* On all four cells the cost stays, and is slightly larger, when the skip's output is removed. The skip's own output is not a cost on any of them. The pattern is the one the card would have named VIA-WEIGHTS: the cost sits in the weights that training with the skip moved, not in the skip's inference output.
* Three of the four cells (TAC-LON, ADE, along-track) have no NaN windows, so the K-ADD flaw does not touch them. The headway cell does.
* This reading carries no name and enters no claim. It goes into the N1DH card's guard-risk list as a HYPOTHESIS: "training with an extra input to O's decoder can move O's operative longitudinal precision".

## Next
* Nothing trains for this card. The N1DC verdict (d1dece12) is unchanged: not adopted.
* The decoder-side lever is the N1DH card (auxiliary route-decision head on O's plan feature), DRAFTED, awaiting the PI's go for pod time.
