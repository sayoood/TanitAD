# RESULT: EP-R1ub (the registered next arm of EP-R1), f = 10 %, fit-side. CPU only. Dev gate NOT read, navtest NOT opened.

**Headline for the Master Mind (integration and decisions):**
1. **Outcome N (NONE-FEASIBLE).** No config of the 9-config grid satisfies the registered constraint. The EP-R1 line closes at the fit (R7). The **formal abandonment file is staged** (`EPR1_ABANDON_PROPOSED.json`, accepted by the landed `refe_head_confirm.read_abandon`); it takes effect only when you land it. Under R9 the shared look may then open for SR-TTC alone.
2. **Landing is needed:** the new code is STAGED, not landed (`code/`, hashes in `code/CODE_SHA256_EPR1UB.json`). The gate bound the landed parents to their blobs at `1afc54f` and the new files to their sha256 (`landed: false`); at landing the blobs must equal those hashes.
3. A **DRAFT** pre-registration of the NC/DAC-row resolution read is staged (`DRAFT_PREREG_NCDAC_RESOLUTION.md`), NOT run. It needs registration.
4. A post-hoc, descriptive noise-row control (unregistered, trains nothing) says the leak is real and not an artefact of P2's in-sample comparator (section 4). Read it as a diagnosis that supports the draft, not as a result.

## 1. What was registered and what ran

* **PREREG_EPR1UB.md sha256 `214f2d379dc7cded632817759e50b4a770ad26111081338abdad130409d5ed00`**, written 2026-10-19 05:54:48, before any code and before any fit. Amendment 1 (stricter controls, interface facts; no bar moved) sha256 `4e23b64dac8faba225b11b24e6488354bacc1a55e29f4043d334b039c42a235a`, written 06:13:46, before the gate was minted.
* **Grid (R7):** lr {1e-3, 3e-3, 1e-2} x u {0.03, 0.1, 0.3}, safe-anchor 0.1, 10 epochs, 256 sets per batch, seeds {0, 1}, f = 10 %, the f10 fit logs and folds. u = weight of an unsafe-proposal anchor with target 0 (class-mean BCE over the unsafe entries of the minibatch).
* **Selection (R7, fixed before any number):** c\* = argmax over the FEASIBLE configs of the mean-over-seeds OOF pick-level dPDMS; feasible iff the paired 97.5th percentile of the OOF unsafe-pick-rate delta vs P2 is <= 0 on BOTH seeds. Estimator: paired log-cluster bootstrap over 117 fit logs, 10,000 resamples, seed 20260927 (it answers "another draw of LOGS" only; the two fit seeds answer head-fit noise; inference is deterministic).
* **Gate:** PASS 6/6 on the REAL manifest, token `C:/Users/Admin/epr1ub/gate/f10_run/epr1ub_gate_token.json` (staged under `raw/f10/`), minted 2026-10-19 06:16 local by `epr1ub_gate.py` from the verified bytes. G-COMMIT (7 landed parents equal their blobs at `1afc54f6fa5e`; 3 staged files hashed), G-DVB (9 configs built = registered, every new lever named in the argv, the built selection function equals the registered behaviour on 11 canned tables), G-PATH, G-BANK (23 units sha256-verified, 117 selected logs all in the bank), G-LIVE (the landed C1/C6 plus the u probes on unit `S5a37f70fed`: u-only gradient non-zero on the EP row and EXACTLY 0.0 on the other five rows and the features; EXACTLY 0.0 with no unsafe entry; bit-identical under changed labels and under changed SAFE logits; u = 0 bit-identical to the landed loss; one Adam step moves the row by lr at 1e-3 / 3e-3 / 1e-2, and by exactly 0.0 at lr 0), G-SUITE (`test_epr1.py` 46 tests OK, `test_epr1ub.py` 39 tests OK on a clean export). `test_epr1ub_gate.py`: 17 tests OK (one deliberate-regression arm per check; logs in `raw/`).
* **Run:** launched 06:16:32 through `launch_epr1ub.ps1` (the launcher verifies the token and runs the argv the gate printed), finished 06:24:10, 456 s.
* **Identity arm (precondition of the grid): PASSED, bit for bit.** (lr 1e-3, anchor 0.1, u = 0) through the UB path reproduced the f10 fit: final rows sha256 `f775753f0817d53607b0044e9d33f6d2c5f5b01b73a8d20567d6394ca393fa95` (seed 0) and `636f6270f52ef462fb6a1a56a990aa8c683f3c30a3394091b77d10f439852fa8` (seed 1), the whole OOF read and the c\* bootstrap equal to the last bit.

## 2. The table (MEASURED, ours; SELECTION tier, fit-side OOF; `raw/f10/epr1ub_result.json` sha256 `10562ac7defe299b5bd051648a024b81c0334b503276a51f5e2beb120f49cafa`)

dPDMS x100 and dUnsafe are the new pick minus **P2's pick on the same sets, and P2 trained on those sets (IN-SAMPLE for P2, biased in P2's favour)**. P2: unsafe-pick rate 0.03036, PDMS x100 90.3648. Intervals are 95 % paired log-cluster bootstrap. "mean sigmoid" is the EP row's mean sigmoid on unsafe / safe OOF proposals (P2: 0.436 / 0.774; the f10 c\*, u = 0: 0.843 / 0.850).

| config | seed | pooled OOF concordance | dPDMS x100 [95 %] | dUnsafe-pick rate [95 %] | changed picks | mean sigmoid unsafe / safe | feasible |
|---|---|---|---|---|---|---|---|
| lr0.001_u0.03 | 0 | 0.9776 | -0.862 [-1.297, -0.495] | +0.0163 [+0.0119, +0.0212] | 12726 | 0.433 / 0.729 | no |
| lr0.001_u0.03 | 1 | 0.9776 | -0.877 [-1.311, -0.504] | +0.0159 [+0.0115, +0.0210] | 12768 | 0.438 / 0.734 | no |
| lr0.001_u0.1 | 0 | 0.9753 | -0.662 [-1.113, -0.246] | +0.0178 [+0.0138, +0.0224] | 12766 | 0.259 / 0.605 | no |
| lr0.001_u0.1 | 1 | 0.9754 | -0.653 [-1.110, -0.225] | +0.0178 [+0.0139, +0.0222] | 12731 | 0.267 / 0.614 | no |
| lr0.001_u0.3 | 0 | 0.9731 | -0.834 [-1.279, -0.449] | +0.0182 [+0.0147, +0.0224] | 13068 | 0.141 / 0.443 | no |
| lr0.001_u0.3 | 1 | 0.9728 | -0.798 [-1.236, -0.416] | +0.0175 [+0.0140, +0.0216] | 13037 | 0.148 / 0.456 | no |
| lr0.003_u0.03 | 0 | 0.9773 | -0.930 [-1.313, -0.592] | +0.0138 [+0.0098, +0.0184] | 12987 | 0.418 / 0.744 | no |
| lr0.003_u0.03 | 1 | 0.9772 | -0.915 [-1.296, -0.570] | +0.0135 [+0.0095, +0.0182] | 12993 | 0.422 / 0.749 | no |
| lr0.003_u0.1 | 0 | 0.9753 | -0.697 [-1.148, -0.292] | +0.0172 [+0.0131, +0.0219] | 12798 | 0.249 / 0.616 | no |
| lr0.003_u0.1 | 1 | 0.9752 | -0.684 [-1.133, -0.287] | +0.0166 [+0.0125, +0.0212] | 12754 | 0.260 / 0.634 | no |
| lr0.003_u0.3 | 0 | 0.9735 | -0.792 [-1.233, -0.392] | +0.0174 [+0.0136, +0.0219] | 12991 | 0.136 / 0.460 | no |
| lr0.003_u0.3 | 1 | 0.9731 | -0.740 [-1.187, -0.345] | +0.0170 [+0.0132, +0.0215] | 12905 | 0.145 / 0.481 | no |
| lr0.01_u0.03 | 0 | 0.9781 | -1.012 [-1.401, -0.652] | +0.0150 [+0.0110, +0.0196] | 13009 | 0.412 / 0.741 | no |
| lr0.01_u0.03 | 1 | 0.9780 | -1.135 [-1.508, -0.777] | +0.0149 [+0.0108, +0.0197] | 12991 | 0.424 / 0.757 | no |
| lr0.01_u0.1 | 0 | 0.9763 | -0.776 [-1.260, -0.347] | +0.0181 [+0.0138, +0.0230] | 12897 | 0.242 / 0.608 | no |
| lr0.01_u0.1 | 1 | 0.9758 | -0.706 [-1.175, -0.287] | +0.0164 [+0.0119, +0.0216] | 12866 | 0.254 / 0.635 | no |
| lr0.01_u0.3 | 0 | 0.9743 | -0.732 [-1.177, -0.359] | +0.0169 [+0.0131, +0.0214] | 13013 | 0.135 / 0.459 | no |
| lr0.01_u0.3 | 1 | 0.9738 | -0.586 [-1.010, -0.200] | +0.0159 [+0.0122, +0.0200] | 12916 | 0.141 / 0.490 | no |

## 3. Outcome row (committed in PREREG_EPR1UB s7): **N, NONE-FEASIBLE**

* In **all 18 (config, seed) cells** the unsafe-pick-rate delta is wholly above 0 (point +0.0135 to +0.0182; lower bound >= +0.0095; upper bound +0.0182 to +0.0230) and the dPDMS is wholly below 0 (point -0.586 to -1.135; upper bound <= -0.200). The identity arm (u = 0, lr 1e-3) reads +0.0102 [+0.0065, +0.0146] unsafe and -1.145 dPDMS (seed 0), as in f10.
* **The u anchor made the unsafe-pick rate WORSE, not better**, relative to the u = 0 identity arm at lr 1e-3 (mean over seeds +0.0136 to +0.0179 against +0.0102; u = 0 at lr 3e-3 and 1e-2 was not run, so the comparison is exact only at lr 1e-3, where it holds for every u), although it did what it was built for on the EP row's level: the mean sigmoid on unsafe proposals fell from 0.843 (u = 0) to 0.43 / 0.26 / 0.14 (u = 0.03 / 0.1 / 0.3 at lr 1e-3), about P2's level (0.436) at u = 0.03 and below it at u >= 0.1. dPDMS improved somewhat with u (best mean -0.657 at lr 1e-3, u = 0.1) and stayed negative.
* **The lr extension did not help either.** lr 3e-3 and 1e-2 (E1's remedy) give the same picture as lr 1e-3. Edge disclosure E-hi / E-lo: moot, no c\*.
* **Next arms (committed):** (1) the formal abandonment of EP-R1 (staged); (2) the DRAFT NC/DAC-row resolution read (staged, NOT run); (3) under R9 the look may open for SR-TTC alone once the Master Mind lands the abandonment. **The dev gate was NOT read for EP-R1ub; no dev reading exists.**

## 4. How to read N: the comparator caveat, and a post-hoc control (UNREGISTERED, descriptive, trains nothing)

PREREG s1 warned that NONE-FEASIBLE could reflect P2's in-sample pick as much as the mechanism: P2's pick is its own in-sample argmax, so any deviation might cost safety on these sets. To size that, `code/null_control.py` replaced P2's EP row by P2's row plus isotropic Gaussian noise (no learning), on the same 15,808 sets with the same estimator (`raw/f10_null/null_control.json`):

| noise sigma | noise seed | changed picks | dUnsafe [95 %] | dPDMS x100 |
|---|---|---|---|---|
| 0.001 | 0 | 0.013 | -0.0001 [-0.0002, +0.0000] | +0.003 |
| 0.001 | 1 | 0.017 | +0.0001 [-0.0001, +0.0004] | -0.014 |
| 0.003 | 0 | 0.040 | -0.0001 [-0.0004, +0.0001] | -0.007 |
| 0.003 | 1 | 0.049 | +0.0003 [-0.0001, +0.0007] | -0.016 |
| 0.01 | 0 | 0.132 | -0.0001 [-0.0006, +0.0005] | -0.026 |
| 0.01 | 1 | 0.150 | +0.0005 [-0.0002, +0.0012] | +0.013 |
| 0.03 | 0 | 0.355 | +0.0000 [-0.0011, +0.0012] | -0.117 |
| 0.03 | 1 | 0.379 | +0.0010 [-0.0003, +0.0024] | -0.055 |
| 0.1 | 0 | 0.728 | +0.0006 [-0.0009, +0.0024] | -1.207 |
| 0.1 | 1 | 0.691 | +0.0034 [+0.0007, +0.0059] | -1.373 |

Noise rows that change 69-73 % of the picks (the fitted rows change 80-83 % of OOF picks) move the unsafe-pick rate by only +0.0006 to +0.0034. The fitted rows move it by +0.0135 to +0.0182, four to thirty times as much. **So the comparator's in-sample bias explains at most a small part of N; the learned preference for progress genuinely moves the pick onto unsafe proposals, and a lower EP level on unsafe proposals (u) does not stop it.** That is the `H-REFE-RL1-NCRES` reading (NC / DAC resolution, or the weighting overpowering the heads), now seen in a third arm. Limits: the control is isotropic noise (a learned perturbation is anisotropic and could differ), two noise seeds, post-hoc after seeing N. It is not used for any decision.

## 5. Evidence classes and limits

* Every number above: MEASURED (ours, the paths in the manifest), SELECTION tier (fit-side OOF), a design read and not a result about driving. T0/T1: not applicable (no closed-loop claim).
* One head-fit seed pair; the intervals do not see training variance (H-ESTIM-SEED-1). Nine configs were scored on the data they would have been chosen on (optimistic for a chosen c\*; none was chosen).
* The fitted rows are never evaluated off the 117 fit logs here (no dev, no navtest).
* LC read for u > 0 not built; D-FLAT versus D-DATA would have been UNDETERMINED after a dev FAIL. Moot.

## 6. Proposed register rows (NOT written; `Project Steering` files are not touched by this agent)

**GOALS_AND_CLAIMS.md, new row `D-REFE-EPR1UB-1` (2026-10-19):** "EP-R1ub (lr extension and unsafe-proposal anchor, PREREG_EPR1UB sha256 214f2d37...) has NO feasible config at f = 10 %: outcome N. In all 18 (config, seed) cells the OOF unsafe-pick-rate delta vs P2 is wholly above 0 (+0.0135 to +0.0182) and the OOF dPDMS wholly below 0 (-0.586 to -1.135). The u anchor lowered the EP row's mean sigmoid on unsafe proposals to about P2's level and below (0.43 / 0.26 / 0.14 at lr 1e-3) and the unsafe-pick rate still rose (+0.0136 to +0.0179 vs +0.0102 at u = 0, lr 1e-3). Identity arm bit-identical to f10. Post-hoc noise-row control: noise rows with a similar changed-pick fraction (69-73 % vs 80-83 %) move the unsafe rate by only +0.0006 to +0.0034, so the in-sample comparator explains little. EP-R1 closes at the fit (R7); formal abandonment staged; dev gate unread. Next: the NC/DAC-row resolution read (`H-REFE-RL1-NCRES`), DRAFT staged. SELECTION tier, MEASURED, paired log-cluster bootstrap 10,000 / seed 20260927 (another draw of LOGS only), P2 in-sample, one head-fit seed pair." Also update `H-REFE-RL1-NCRES` ("a THIRD arm shows the mechanism") and `H-EPR1` (CLOSED AT THE FIT).

**BACKLOG.md, new items:** (A, 0-GPU) register and run the NC/DAC-row resolution read Part A (CPU, seconds); (A) land `code/` of this package; (A) an LC stage for u > 0 only if the EP-R1 line is ever reopened; (A) re-state Part A on OOF picks with a gated fit if it is inconclusive.

## 7. Deviations and disclosures (everything that is not exactly as registered)

* Staged, not landed (the agent never commits). The gate's G-COMMIT binds the landed parents to blobs at `1afc54f` and the new files by sha256 with `landed: false`.
* The post-hoc noise-row control (section 4) is unregistered and descriptive. It was run after N was known and decides nothing.
* The E-lo waiver file was not needed (no c\*). The C2 stage and the dev applier were not run (no c\*).

## 8. Deliverable manifest

See `MANIFEST.md` (same folder) and `paths.json` in the staging root: repo targets under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-ep-r1ub/`. Run dir (disk only until landed): `C:/Users/Admin/epr1ub/run/f10` and `C:/Users/Admin/epr1ub/run/f10_null`; gate dir `C:/Users/Admin/epr1ub/gate/f10_run`; code dir `C:/Users/Admin/epr1ub/code`.
