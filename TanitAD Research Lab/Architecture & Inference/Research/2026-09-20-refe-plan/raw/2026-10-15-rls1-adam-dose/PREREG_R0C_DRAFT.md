# PREREG DRAFT: R0c -- anchor the EPO term to arm A's policy instead of the frozen base

DRAFT, 2026-10-15, Research Lab agent (Sonnet). NOT launched, NOT committed, nothing implemented in the trainer. Written for the Master Mind / PI. Provision, spend and the trainer edit are theirs.
Note on scope: the brief asked for an R0c draft IF H-ADAM held. It does NOT hold (`PREREG_H_ADAM.md`, Amendment 2), so this draft is for the lever the measurements point to instead (Rule Zero), not for a split optimiser.

## 1. Question and the ONE lever

R0 and R0b breached g5 on TTC against the control (B - A = -1.00 at R0 u1000; -0.57 / -0.70 at R0b u250 / u500). The same dumps decomposed against the start (MEASURED, `ab_vs_base.json`):
B - base TTC is -0.17 [-0.71, +0.38] (R0 u1000) and -0.03 [-0.60, +0.52] (R0b u500), both consistent with 0, while the BCE-only control gains +0.83 [+0.13, +1.58] / +0.67 [+0.03, +1.35] over the start. Arm A's selection policy sits at KL 1.14
from the base (J 0.828) and arm B's at 0.15-0.26 (J 0.847-0.865) (ESTIMATED, `late_probe.json`), because the EPO term's KL anchor pulls toward the FROZEN BASE and scales with lam.

**Lever:** `epo_loss`'s reference logits `sx_ref` (today `sx0`, the frozen base's, scorer_finetune.py `epo_term`/`train_losses`) become arm A's DETACHED logits of the same micro-batch (`--epo-ref armA`; default `base` = R0 exactly). Nothing else changes.

## 2. Arm

| field | value |
|---|---|
| arm | **R0c** (one A40 arm, ~3.5 A40-h by the section 10.3 cost model; its own arm A is the same-pass control, C0 is the banked second control) |
| argv | R0's banked argv (`raw/2026-10-14-rls1-r0-kill`, sha256 `24451b62...`) **+ `--epo-ref armA`**; `--epo-lam 1.0`, `--epo-beta 0.1` and every other flag as R0 (so R0 vs R0c differs in the reference ONLY) |
| guards | K3 / g5 / g3 / g4 / K0-K6 AS REGISTERED in `eval/PREREG_RL_S1.md` sections 4-5, unchanged; the control for every delta is the arm's own A; RL-val is the selection tier |
| trainer edit (NOT done) | a new flag `epo_ref` in `EPO_DESTS`, `epo_refusals` and `build_parser`; `train_losses` passes `sxA.detach()` instead of `sx0` to `epo_term` when `epo_ref == "armA"`; a G-DVB entry reading the BUILT reference |

## 3. Reachability and controls that must read known values BEFORE launch (the 2026-09-10/11 class)

1. **Trace flag -> loss:** argparse -> namespace -> `train_losses` argument -> `epo_loss(..., sx_ref, ...)` -> `logpi0` -> `kl` -> `backward()`, `file:line` at each hop.
2. **Identity at step 0:** with A = B = base, R0c and R0 give bit-identical loss, KL (exactly 0) and gradient on the same micro-batch (EXACT 0.0 / bit equality). This also proves the flag changes nothing it should not.
3. **Non-zero where it must be:** after one B-only update (B != A), the gradient of the KL term w.r.t. B is non-zero and DIFFERENT from the base-anchored one; A's parameters receive NO gradient from it (A detached; max abs grad on A's parameters EXACTLY 0.0 from the EPO term).
4. **Mutation:** replace `sxA.detach()` by `sxA` -> A receives an EPO gradient -> the check in 3 must go RED; replace it by `sx0` -> the check in 2 passes but the check in 3 must go RED.
5. **Default:** `--epo-ref` must be NAMED in the arm's argv (a default of `base` silently reproduces R0).
6. `launch_gate.py` PASS token bound to commit, argv sha256 and data manifests; a flag without a G-DVB entry is refused by the gate.

## 4. Predictions and outcomes committed in advance (reads: u250 / u500 / u750 / u1000, RL-val, B - A and B - base, paired log-cluster bootstrap 10,000 / seed 20260927, sub-scores x100)

The mechanism claims: (M1) B - A TTC rises from about -1.0 toward B - base's -0.17 because B is no longer pinned to the start; (M2) KL(pi_A || pi_B) stays small (logged `epo_kl` now measures A-to-B, expected well below the 0.15-0.26 base-to-B values) while PDMS B - A stays positive.

| outcome | condition (RL-val, as read) | meaning | next lever (Rule Zero) |
|---|---|---|---|
| **O1 ADOPT-candidate** | g3-g5 hold at every check AND PDMS B - A lower bound > 0 at u500 and u1000 AND the section 4 bar of `PREREG_RL_S1.md` | the reference was the defect; EPO's ascent does not harm TTC once it is not pinned to the start | the replicate arm (same flags, other seed) per H-ESTIM-SEED-1 BEFORE any claim; then the full-navtest endpoint |
| **O2 safe but flat** | g5 holds at every check, PDMS B - A lower bound <= 0 | anchoring to A leaves no ascent budget (B ~ A) | one pre-registered pair on the A-anchored axis: `--epo-beta` 0.1 -> 0.03 (anchor weaker, ascent unchanged); PI decision on budget |
| **O3 breached with a small KL** | g5 breached at two consecutive checks AND mean `epo_kl` (A to B) <= 0.2 at the breach | the EPO ascent ITSELF lowers TTC relative to A, whatever the anchor: the reference hypothesis is REFUTED | EPO restricted to the progress heads' parameters with a frozen trunk. The banked displacement lives 99 % in `score_dec` and 0.0 % in `score_head`, so this is a different regime, and its fraction of R0's J gain on cached trunk features is a CPU job to run FIRST (PI decision for the parameter-group change) |
| **O4 breached with a large KL** | g5 breached AND `epo_kl` > 0.2 | B drifted from A faster than the anchor holds it | `--epo-beta` up (0.1 -> 0.3), one pair; K3 is the guard |
| **O5 other kill** | K0-K2, K4-K6 or an NC / DAC bound | per the section 5 table | per section 5 |

A result that is separated on the paired interval is NECESSARY, not sufficient (H-ESTIM-SEED-1): the one training seed per arm stays a selection-tier read until the replicate runs. Inference replicates (the iCEM sampler) do not apply to this scorer-only arm, which uses the argmax rule.

## 5. Evidence and limits of this draft

* The decomposition (section 1) is MEASURED on banked RL-val dumps; the KL / J values are ESTIMATED (substituted sctx, V 0.996-0.997 at each state; 256-set pool). The mechanism is a hypothesis with a sign test, not a result.
* In the closed-loop simulation (50 updates) the A-anchor cannot be distinguished from the base anchor: A has barely left the base by then (identity at step 0, section 3 item 2). Only the A40 arm discriminates; there is no cheaper CPU test of the reference choice.
* Cost and authority: ~3.5 A40-h and a trainer edit. RL-S1b's authorised budget stops at the R0b kill (`PREREG_RL_S1.md` 10.3), so this needs a PI decision.
