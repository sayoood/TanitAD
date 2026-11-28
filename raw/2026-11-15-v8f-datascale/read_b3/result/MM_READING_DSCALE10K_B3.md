# MM reading: DSCALE10K block 3, the H-DATA-SCALE-1 sub-read (2026-11-18)

Source: `out/RESULT_DSCALE10K_B3.json` (controls: no problems; K1 self-pair 0.0 [0, 0]; K-RE B1 and B2 216 of 216 T1 rows reproduced; K-EVALX PASS on all 10 arm-variants). Stage A, **ONE seed (seed 0)**. 141 episodes / 2,399 windows. Paired episode-cluster bootstrap, B 2,000, seed 0: every interval answers "another draw of EPISODES" only, never another training run (`H-ESTIM-SEED-1`). The floor is ONE replicate pair (B_rep vs B_s0): a weak floor.

## 1. The label: NEITHER (both readings agree)

Rulings `MM_RULINGS_B3.md` were made before the read; the bar is unchanged.

| quantity (ADE6, T0, lower is better) | Δ [95 % interval] | vs 2F_A = 0.1656 m | label |
|---|---|---|---|
| U_s0 − BL_s0 (the literal "U − BL") | −0.0090 [−0.1055, +0.0897], not separated | below | NEITHER |
| UE_s0 − BL_s0 (compute-matched) | −0.0958 [−0.183, −0.0053], separated | below | NEITHER |

* Mean ADE6 (T0, m): B_s0 2.1819, U_s0 2.0539, B_rep 2.0991, BL_s0 2.0629, UE_s0 1.9671. F_A = 0.0828 m.
* Δ_data = U_s0 − B_s0 = −0.1280 [−0.2295, −0.0357]; Δ_ep = BL_s0 − B_s0 = −0.1190 [−0.2165, −0.0269].
* `H-DATA-SCALE-1`: **NEITHER at one seed.** Neither lever beats the other by more than 2F_A on ADE6.

## 2. What the four families say (stage A, per pair, never pooled)

| pair | LONGITUDINAL | LATERAL | TACTICAL | STRATEGIC |
|---|---|---|---|---|
| U − B_s0 (3.2x clips, same steps) | gain | none | **cost** (TAC-LAT) | none |
| BL − B_s0 (same clips, 3.2x steps) | gain | gain | **gain** (TAC-LAT, TAC-LON) | cost (S2 skill@80) |
| UE − BL (3.2x clips, both at 61,568 steps) | none | none | **cost** (TAC-LAT) | **gain** (S1 pos_err@80, S2 skill@80) |
| U − BL | none | none | cost (TAC-LAT, TAC-LON) | none |

**MEASURED, the finding block 3 was built for.** The 10k clips cost TAC-LAT by the same amount at both training lengths, beyond F_cell 0.0767 (T1-kin):
* U − B_s0 at 19,164 steps: −0.0871 [−0.1297, −0.0446].
* UE − BL at 61,568 steps: −0.0855 [−0.1216, −0.0492].

⇒ At this seed, the tactical cost does not come from fewer epochs per clip. It travels with the 10k clips' content. "More epochs per clip" is eliminated as the explanation of block 2's tactical cost (one seed, one-pair floor).
⇒ This is consistent with the dilution HYPOTHESIS: the 10k clips carry masked VLM goal bits, so the labelled share of tactical supervision falls from 100 % to 31 % (4,512 / 14,512). It is still a HYPOTHESIS. The F1 diagnostic could not separate it (power-limited).

**MEASURED, a lever in its own right.** More steps on B's own data (BL vs B_s0) gains three families:
* TAC-LAT +0.0817 [+0.0467, +0.1175] and TAC-LON +0.0550 [+0.0229, +0.0875];
* speed MAE and headway;
* yaw rate and curvature.

It costs STRATEGIC S2 skill@80. ⇒ B at 19,164 steps is under-trained (one seed).

**MEASURED.** The 10k clips gain STRATEGIC at matched steps (UE vs BL).

T's latcons: NOT T-DATA-LIMITED on every pair.

## 3. The next lever (Rule Zero: a NEITHER is not a stopping point)

1. **The mixture arm, the discriminating experiment for dilution.** It trains the union at 61,568 steps with tactical supervision rebalanced, so the 4,512 labelled clips keep their tactical gradient share. Either batch-balanced sampling or a per-clip tactical loss weight; the pre-registration picks one and commits both outcomes.
   * If dilution is the mechanism, the arm keeps BL's tactical gain AND UE's strategic gain.
   * If not, the TAC-LAT cost persists, and the content of the 10k clips (not their label share) is the defect.
   * Needs: a pre-registration, a gate PASS, and the PI's go for pod time.
2. **X_s0** (the 10k clips alone, the card's conditional arm): about 2.6 pod-h, waiting on the PI's go.
3. **Seeds for BL and UE**: every block-3 label is one seed against a one-pair floor. A second seed of BL_s0 and UE_s0 is the cheapest way to make BL's three-family gain quotable as a lever effect.

Not read here: the mechanism npz evals and dumps (about 96 MB) stay on F:, re-creatable from the HF-banked checkpoints (UE_s0.A, BL_s0.A sha256-verified at three copies).
