# PREREG_V8F_N1DC_COSTATTR: is the N1DC guard cost carried by the skip's output, or by the weights it changed in training? (zero training, dev-box CPU)

Status: FROZEN at the sha256 beside this file, before any number of this card exists. Parent: `raw/2026-11-20-n1dc/PREREG_V8F_N1DC.md` (sha256 2eae7964), outcome N1DC-COST, verdict `raw/2026-11-20-n1dc/read/MM_READING_N1DC_VERDICT.md` (landed d1dece12). The row's next step: "the cost family gets a card". This is that card, in its cheapest form.

## 1. Question
The N1DC eval-141 guard (T1-kin/hb0.2, N1DC - N1D, floor F2) carries four cost cells:
* s0: TAC-LON;
* s1: operative 0-2 s `ade_m`, `along_mae_m`, `dk_hw_abs_err_m`.

N1DC differs from N1D in two ways at once: the skip's output at inference, and the weights that 3,000 steps of training with the skip moved. Zeroing the skip's weight (`o_code.weight` = 0, an exact "output x 0" because `o_code` has no bias) removes the first and keeps the second. Which one carries each cost?

## 2. Arms (zero training)
* **U_s0, U_s1:** the zeroed-skip copies `N1DC_s{0,1}_zuse`, built on the pod 2026-11-20 by `n1dc_zuse_ckpt.py` (sha256 c9b72aad): `o_code.weight` all zero, the other 277 tensors equal N1DC's (`torch.equal`). Pulled by `pscp`, with the pod's own `sha256sum` recorded beside each.
* One eval-141 T1-kin/hb0.2 dump per arm, through the same chain step as the N1DC dumps (`n1dc_dumps.py`, hb wrapper `--kappa-max 0.2`, tree `eval141/tree`, the same argv except the checkpoint and the tag). No zero twin (not needed for the cells).
* Comparators, already dumped by the guard chain (not re-dumped): N1DC_s{s} and N1D_s{s} hb0.2 in `eval141/dumps141_hb02`.

## 3. Read
The STRAT1 24 cells per seed, the STRAT1 cost / gain rule, floor F2 (the guard's), paired episode-cluster bootstrap B 2,000 seed 0 ("another draw of EPISODES" only), 2,399 windows / 141 episodes. Two contrasts per seed:
* **P1 = U - N1D**: does the cost stay when the skip's output is removed?
* **P2 = N1DC - U**: the skip output's own effect, weights held fixed.

Each of the four cost cells is decided **at its own seed**. The other seed is printed beside, with no label.

## 4. Outcomes, per cost cell (committed now)
| label | condition |
|---|---|
| **VIA-WEIGHTS** | P1 is a cost on the cell (same rule, same floor) |
| **VIA-OUTPUT** | P1 is not separated on the cell, and P2 is separated in the cost direction |
| **UNRESOLVED** | otherwise |

* All four labels are printed together, with every P1 and P2 cell at both seeds.
* **Next, for any label:** nothing trains, and nothing enters the recipe. The labels go into the guard-risk list of the next decoder-side card (auxiliary route-decision head on O's plan feature).
  * VIA-OUTPUT names "an inference path into O's decoder" as the risk.
  * VIA-WEIGHTS names "training with an extra input to O's decoder" as the risk.

## 5. Controls (each with the value it must read)
* **K-COPY:** on the pulled U checkpoints, `o_code.weight` is all zero, and every other tensor equals the N1DC checkpoint in `eval141/ckpts` (`torch.equal`; 277 of 277). The sha256 equals the pod's.
* **K-SAME:** the U dumps' eid, clip_index, ws, g and v0 equal C0_s0, and their floor path `fl` equals the hb0.2 floor (K-FL-HB rule of the guard reader).
* **K1:** U paired with itself reads exactly 0.0 on every cell.
* **K-ADD:** P1 + P2 equals N1DC - N1D on every cell to 1e-12. Window means are linear, so this checks the bookkeeping, not the estimator.
* **K-GUARD:** N1DC - N1D recomputed by this reader equals `RESULT_N1DC_EVAL141.json` `guard_eval141_hb02_N1DC-N1D` exactly on every field.

Any control failure makes the result INVALID; no label is printed.

## 6. Cost
Two CPU dumps of about 4.5 min each, plus a read of about 2 min, on the dev box. One process at a time, commit headroom of at least 5.5 GB, 2 threads. No pod GPU; pod access is `pscp` of the two checkpoints only. No PI spend.

## Log
* 2026-11-20 ~20:00Z: written and frozen by the MM, before any U eval-141 dump existed.
