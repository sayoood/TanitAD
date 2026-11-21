# PREREG: EP-R1ub, the registered next arm of EP-R1 (lr extension AND unsafe-proposal anchor), fit-side selection with a safety constraint

**Written 2026-10-19 (EP-R1ub agent), BEFORE any EP-R1ub code exists and BEFORE any EP-R1ub number exists.** This file's sha256 is in `PREREG_EPR1UB.sha256`, written before the first fit. Changes after registration are appended as `Amendment n`; the body is never edited.

**Design authority:** `MM_RULINGS_HEADROWS.md` section B, ruling **R7** (Master Mind, 2026-10-19 ~05:45, before any dev read and before the navtest look), and `PREREG_EPR1.md` (sha256 `29e79bd61cbbb209bc08aa8b42315736ee47fdaf89abd8ad0731a4d4d4f8f80a`) rows **E1** and **D-LEAK**. Nothing below changes a PREREG_EPR1 rule that this text does not name. Where this file is silent, PREREG_EPR1 holds.

## 0. What was read, and what was not

| read | used for | class |
|---|---|---|
| `raw/f10/epr1_result.json` (sha256 `eb573404bfd02bb8243cbb18cba954e2136bdcd0e3874cc21fb93e647e142463`, the f = 10 % fit) | the E1 and D-LEAK trigger values; the nine parent configs' OOF concordance, pick-level dPDMS, unsafe-pick rate (point values); the identity reference | MEASURED (ours, that path) |
| `MM_RULINGS_HEADROWS.md` B / R7, `PREREG_EPR1.md` s3-s8, `MM_DECISIONS_EPR1.md`, landed `epr1.py` (sha256 `a751c63e...38d8`), `epr1_gate.py`, `epr1_dev.py`, `launch_epr1.ps1` | design, code to import, interfaces the dev applier consumes | design / code |

**Not read:** any dev-gate (P2 held-out table) value, any navtest value, any value from a UB config. No grid number of this arm exists. The f10 numbers are SELECTION-tier fit-side reads (PREREG_EPR1 s5 row 7), not dev outcomes.

## 1. What this rests on

| fact | value | where | class |
|---|---|---|---|
| E1 fired: c\* = lr 1e-3 (the parent grid maximum), anchor 0.1, `c_star_at_grid_edge` true, pooled OOF concordance 0.9808 | | `epr1_result.json` | MEASURED |
| D-LEAK fired on both seeds. dPDMS x100 vs P2 at c\*: -1.145 [-1.473, -0.840] (seed 0), -1.174 [-1.503, -0.858] (seed 1). Unsafe-pick rate delta: +0.0102 [+0.0065, +0.0146] / +0.0101 [+0.0065, +0.0143] | paired log-cluster bootstrap, 10,000, seed 20260927 | `epr1_result.json` `pick_level_at_c_star` | MEASURED |
| across ALL nine parent configs the unsafe-pick rate of the new pick is above P2's (P2: 0.030364): from +0.0077 (lr 1e-4, anchor 0.3) to +0.0165 (lr 3e-4, anchor 0.03), point values, no interval computed off c\* | | `epr1_result.json` `grid[*].per_seed[*].pick_level` | MEASURED (point) |
| in the parent grid pooled OOF concordance rises monotonically with lr (0.962 at 1e-4 to 0.981 at 1e-3) while pick-level dPDMS FALLS (about +0.07 at lr 1e-4, anchor 0.1, to -1.17) | | same | MEASURED |
| the EP row's mean sigmoid on UNSAFE proposals approaches (and at anchor 0.3 passes) that on SAFE ones as lr rises: lr 1e-4, anchor 0.1: 0.689 unsafe vs 0.782 safe; lr 1e-3, anchor 0.1: 0.843 vs 0.850; lr 1e-3, anchor 0.3: 0.873 vs 0.865 (seed 0). The conditional target gives violators no signal, so the row extrapolates | C7 | same | MEASURED |
| in the banked labels EP is exactly 0 on 98.4 % of NC/DAC-violating proposals (the rest have NC = 0.5): a target of 0 on unsafe proposals matches P2's own unconditional label there | | `D-REFE-EP-1`, PREREG_EPR1 s1 | MEASURED (SELECTION tier, descriptive) |

⚠️ **The comparator is biased in P2's favour, and this changes what "feasible" can mean.** The pick-level read compares the new row's out-of-fold pick against **P2's pick on the same sets, and P2 TRAINED on those sets** (PREREG_EPR1 s5 row 7, "IN-SAMPLE for P2, biased in P2's favour, so conservative against the new row"). A constraint "OOF unsafe-pick delta upper CI <= 0" therefore asks the new row to beat an in-sample baseline on safety. **NONE-FEASIBLE is a live outcome and may reflect that bias as much as the mechanism**; it is reported with that caveat (section 7, row N).

## 2. Hypothesis and the two levers (single registration, R7)

**H-EPR1ub.** An unsafe-proposal anchor with target 0 (lever u), together with a learning-rate range that can reach the level shift (lever lr), yields at least one EP-row configuration whose fit-side OOF unsafe-pick rate is not above P2's (paired upper 95 % bound <= 0 on both seeds), and among those a positive OOF pick-level dPDMS. This is a design hypothesis, not a result.

* **Lever lr (E1 -> EP-R1b):** lr in **{1e-3, 3e-3, 1e-2}**. The parent maximum 1e-3 becomes the lower edge.
* **Lever u (D-LEAK -> EP-R1u):** weight **u in {0.03, 0.1, 0.3}** of an unsafe-proposal anchor with target 0.
* **Fixed (R7):** safe-anchor weight **0.1** (the f10 c\*'s), `pair_w` 1.0, K = 8, Adam (0.9, 0.999), eps 1e-8, no weight decay, no schedule, **10 epochs, 256 sets per batch**, **seeds {0, 1}**, **f = 10 %**, the same ranked fit logs (sha256 `a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1`), the same folds (`rank mod 5`), the same S0 bank and manifest, the same P2 checkpoint, the same PathGuard (the dev-gate and navtest tables are never read by the fit).

**Grid (9 configs; the order is part of the registration, ties go to the first):** lr ascending, then u ascending:
`(1e-3, 0.03) (1e-3, 0.1) (1e-3, 0.3) (3e-3, 0.03) (3e-3, 0.1) (3e-3, 0.3) (1e-2, 0.03) (1e-2, 0.1) (1e-2, 0.3)`.

## 3. The u anchor (exact; frozen before any number)

Loss per minibatch: `total = pair_w * pair + anchor_w * anchor_safe + u * anchor_unsafe`, where `pair` and `anchor_safe` are **the landed `epr1.py` terms, called unchanged**, and

* **anchor_unsafe = mean over the UNSAFE entries of the minibatch of BCE_with_logits(z_ep, 0)** (equivalently softplus(z_ep)). The target is the **constant 0**, not the banked EP label (which is 0 on 98.4 % of unsafe proposals and 0 or the label on the rest). *unsafe* is the exact complement of the landed `safe` mask (NC < 1 or DAC < 1 on the clipped labels). All 64 positions of every set count. **The denominator is the number of unsafe entries in the minibatch (clamped at 1)**, the same class-mean convention as the safe anchor, so u does not depend on the batch's unsafe fraction.
* **u = 0 is the landed loss, bit for bit** (the weight multiplies a finite term; the term is still built, so the identity is a check and not a branch).
* *Disclosure on the gloss in D-LEAK:* the row says the anchor "interpolates between this arm (u = 0) and P2's unconditional label (u = 1)". With class-mean normalisation u = 1 is not literally P2's pooled label (each unsafe entry carries u / n_unsafe, each safe entry anchor_w / n_safe). u = 1 is not built and not read. The gloss is the original row's wording, not a constraint on this grid.
* The EP logit of a SAFE proposal receives **no** gradient from the u term; the EP logit of an UNSAFE proposal receives none from the two landed terms (C6 of PREREG_EPR1 still holds for them).

## 4. Fit, scoring and the identity precondition

* The fit is the landed `epr1.fit_row` (Adam on `row_w` [256], `row_b` [1], features detached), with the loss above. The OOF run is the landed `oof_run` body with the loss routed through; at u = 0 it is bit-identical to the landed one (tested).
* Scored exactly as the f10 fit: pooled and fold-mean safe-contender concordance (reported, never selecting), per-log sums of the pick-level read, C7 leak read. **The pick-level read is computed for ALL nine configs and both seeds** (the parent computed it only for c\*), each with the paired log-cluster bootstrap over the 117 fit logs: **10,000 resamples, seed 20260927, percentile 2.5 / 97.5** (`epr1.boot_draws`, `epr1.paired_delta`, the landed estimator; it answers "another draw of LOGS" only; the two fit seeds answer head-fit noise; inference is deterministic).
* **IDENTITY ARM (a precondition of the grid, run first inside the same gated stage).** (lr 1e-3, anchor 0.1, **u = 0**) through the UB code path must reproduce the f10 fit **bit for bit**: (i) the final rows fitted on ALL fit-set sets, per seed, have **sha256 `f775753f0817d53607b0044e9d33f6d2c5f5b01b73a8d20567d6394ca393fa95` (seed 0) and `636f6270f52ef462fb6a1a56a990aa8c683f3c30a3394091b77d10f439852fa8` (seed 1)** (the registered `ep_row_seed{0,1}.npy` of the f10 fit); (ii) the OOF read per seed (pooled, fold-mean, per-fold, pairs, changed picks, pick-level, C7) equals the f10 grid row `lr0.001_anchor0.1` exactly; (iii) the c\* bootstrap equals f10's `pick_level_at_c_star`. **Any difference stops the run before the grid (outcome A).** Thread count 8 as in the f10 fit.
* Reference files for the identity arm are read from `C:/Users/Admin/epr1/run/f10` (read only) after their sha256 is checked against the pins above and `eb573404...` for the result.

## 5. Selection (fit-side OOF only; fixed now)

For each config: `dPDMS_s` = OOF pick-level mean PDMS x100 of the new pick minus P2's pick (seed s, pooled over the OOF sets); `dU_s` = OOF unsafe-pick rate of the new pick minus P2's, with the paired bootstrap interval; `hi_s` = its 97.5th percentile.

* **Feasible(config)** iff `hi_s <= 0.0` on **BOTH** seeds (a NaN bound is infeasible).
* **c\* = argmax over the feasible configs of the MEAN over seeds {0, 1} of `dPDMS_s` (the point values)**; ties go to the first in grid order. Concordance is **not** the criterion (R7; at f = 10 % it rose to 0.98 while the pick got worse). It is reported for every config.
* No feasible config: **no c\***, status `NONE-FEASIBLE`.
* The sign of `dPDMS` does **not** gate the dev read (R7: "if a config is feasible, the EP-R1 dev gate is read ONCE on it"). It is disclosed (row F-WEAK). The dev bar of PREREG_EPR1 s7 is unchanged: paired out-of-sample gain >= +0.30 PDMS x100 on BOTH seeds, C0, C1, C2, C4 pass.
* `c_star_at_grid_edge` = (lr = 1e-2). **Disclosed only**, no edge rule is added and the grid is never extended inside this arm (the same position as the Master Mind took for SR-TTC, R8).

## 6. Controls (each is a gate; a miss voids the arm, it is not a result)

Every new lever has a **G-DVB entry** (the built object equals the registered one: the lr set, the u set, the safe-anchor weight, the target 0, the class-mean normalisation, the selection constants, the criterion name; the argv must NAME every one of them), a **reach proof** and a **RED mutation arm**:

| lever | reach proof (non-zero where it should act; EXACTLY 0.0 where it must not) | mutation arms that must go RED |
|---|---|---|
| **u anchor** | u-only gradient non-zero on the EP row; **exactly 0.0** on the other five rows and on the features; **exactly 0.0** on a batch with no unsafe entry; **bit-identical** when every EP label is replaced (the target is the constant 0) and when the stored EP logits of SAFE entries are replaced; **changes** when the stored logits of UNSAFE entries are replaced (positive control); on an all-unsafe batch the full-loss gradient is **bit-identical** to the u-only gradient (the safe terms contribute exactly 0.0); the zero-weight control reads exactly 0.0; **u = 0 gradient bit-identical to the landed `total_loss`**; u > 0 moves the fitted row differently from u = 0 through `fit_*` (reach to the optimiser) | delete the u term; invert the mask (`safe` for `~safe`); drop the mask; target = banked label; ignore the weight (always 1); the fit path drops `cfg.u` |
| **lr extension** | one Adam step on the real fit path moves the row by `lr` within 1 % at each of 1e-3, 3e-3, 1e-2; at lr = 0 exactly 0.0 | the fit path ignores `cfg.lr` |
| **selection criterion** | on canned tables with known answers: the highest-dPDMS INFEASIBLE config is not chosen; one seed infeasible makes the config infeasible; `hi` exactly 0.0 is feasible and `hi` = +epsilon is not; ties go to the first in grid order; none feasible gives no c\*; the end-to-end `run` routes through `select_cstar` (a sentinel replacing it is what the run reports) | `<=` -> `<`; `all` -> `any` over seeds; drop the constraint; read `lo` for `hi`; argmax of concordance; argmin; the run bypasses `select_cstar`; the identity precondition removed |
| **identity arm** | a perturbed reference makes the run stop before the grid | the precondition removed |

Inherited and re-run in the gate: the landed C1 / C6 reachability on the real bank unit, G-PATH, G-BANK.
**C3 replicate:** two fit seeds, both required downstream. **C5 estimator:** as section 4.

## 7. Outcomes, committed in advance, each with its next arm (Rule Zero)

| id | trigger | meaning | next arm |
|---|---|---|---|
| **A** (build void) | the gate does not mint; the suite is not green; any reach proof or mutation arm fails to go RED; **the identity arm differs from f10 in any bit** | the instrument is wrong | fix and re-gate; no grid number is read; no selection is made |
| **FEASIBLE** | at least one config is feasible | the unsafe anchor (with the lr range) removes the violator leak against P2's in-sample baseline on both seeds | (1) refit c\* on all fit-set sets per seed; write `epr1_result.json` in the **landed schema** (so the frozen `epr1_dev.py` accepts it) and its sha256 BEFORE any dev read; (2) EP-R1's **C2 stage** at c\* (own gate token); (3) the **dev gate ONCE** with `epr1_dev.py` (`--registered-result-sha256` the value in (1); **no `--ttc-run-dir`**; SR-TTC's dev read belongs to another stream); (4) then the landed PREREG_EPR1 s8 rows decide: D-PASS -> navtest only under MM_DECISIONS_EPR1 D1 (both dev gates read, or one arm abandoned); D-TRADE -> EP-R1t; D-FLAT / D-DATA need an LC read that is NOT built for u > 0 and is then UNDETERMINED (named, BACKLOG) |
| **F-WEAK** (disclosure, fires with FEASIBLE) | c\*'s mean OOF dPDMS <= 0, or <= 0 on either seed | the row is safe against P2 but does not pick better in-sample-biased OOF | the dev gate is still read ONCE (R7); the report states that the fit-side evidence gives no positive support and that the comparator is in-sample for P2 |
| **E-hi** (disclosure) | c\* has lr = 1e-2 | the lr range may still not bound the level shift | disclosed only; a further extension would be a NEW registration |
| **E-lo** (disclosure) | c\* has lr = 1e-3 | the frozen `epr1_dev.py` refuses an lr = 1e-3 c\* unless a waiver exists (its E1 text) | EP-R1b, the lr extension E1 named, has now run inside this arm. The waiver file cites R7 verbatim ("If a config is feasible: the EP-R1 dev gate is read ONCE on it") and is written by this agent as a documentary record, not a new decision; the Master Mind may countermand it before the dev read |
| **N** (NONE-FEASIBLE) | no config is feasible | D-LEAK confirmed against P2's in-sample baseline, **with the caveat of section 1** | (1) the **formal abandonment file** of EP-R1 in `refe_head_confirm.read_abandon`'s format ({arm: "EP-R1", abandoned: true, register_row, reason, decided_by}), `decided_by` = "MM_RULINGS_HEADROWS.md R7 (the abandonment is R7's own committed consequence)", staged for the Master Mind to adopt; (2) a **DRAFT** pre-registration of the NC/DAC-row resolution read (`H-REFE-RL1-NCRES`) on the same bank, written and **not run**; (3) the shared navtest look may then open for SR-TTC alone (R9) |
| **R** (pattern disclosure) | any outcome | | the report lists, per config and seed, concordance, dPDMS with its interval, the unsafe-rate delta with its interval, changed picks and C7, so the trade between the three can be read without re-running |

No row ends at a verdict.

## 8. Estimator discipline

* Every interval is the paired log-cluster bootstrap over **117 fit logs** and answers "another draw of LOGS". It is blind to training variance (H-ESTIM-SEED-1); two fit seeds are the only replicate and both must satisfy the constraint. It is blind to inference variance only because the pick is deterministic.
* The comparator (P2) is in-sample on these sets, in P2's favour. **Every quoted unsafe-rate delta and dPDMS from this arm carries that sentence.**
* Nine configs are scored and one is chosen on the same data it is scored on: the selection is **optimistic** for the chosen config's OOF dPDMS. It is a fit-side selection read (SELECTION tier), not a result about H-EPR1ub.
* **Evidence classes:** the fit-side table is MEASURED (ours, the run dir paths), SELECTION tier. A dev-gate reading, if one is taken, is MEASURED on the P2 held-out table, SELECTION tier on **SEEN scenes** (`D-L1-G0-1`: that table is held out from P2-s0's PDM fine-tune only and is seen at the scene level by the model P2 warm-starts from), a gate and not a claim (MM_DECISIONS_EPR1 D3). Only the navtest primary is quotable.

## 9. Cost (ESTIMATED; arithmetic)

CPU only. f10 took about 1 minute per (config, 2 seeds, 5 folds). Identity arm about 2 minutes, nine configs about 9-12 minutes, two refits seconds. C2 stage about 2 minutes. Dev gate minutes. No GPU, no new bank, no navtest.

## 10. What this does not claim

Nothing about driving beyond the fit-side OOF pick-level read and, if taken, the dev gate. Nothing about SR-TTC. Nothing about the NC / DAC rows (their resolution is the next lever if row N fires). The LC read for u > 0 is not built here.

## 11. Not built in this package (flagged, not silently dropped)

* An LC stage for u > 0 (D-FLAT versus D-DATA is UNDETERMINED after a dev FAIL that is neither D-TRADE nor D-LEAK).
* The confirm applier for a UB row (the landed `refe_head_confirm.py` is not touched; a navtest read of this row is not authorised by this file).
* Code lands only through the Master Mind: the new files are STAGED, never committed. The gate therefore binds the **landed parents** to their blobs at the landed commit and the **new files** to their sha256, recorded in the token and in `CODE_SHA256_EPR1UB.json`. At landing the blobs must equal those hashes. This is a stated deviation from "bound to a landed commit" and is the only one.
