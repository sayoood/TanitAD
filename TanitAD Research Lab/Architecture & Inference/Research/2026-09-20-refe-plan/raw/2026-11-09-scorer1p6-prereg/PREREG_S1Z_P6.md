# PREREG_S1Z_P6: train the scorer on the near-twin failures it misses (FROZEN 2026-11-09)

**Status.** Frozen by the Master Mind, by sha256, before any Batch 1 arm has a navtest or E2 number and before this arm trains. It does not change `PREREG_SCORER1P6.md`, `PREREG_P6_SELECTORS.md` or `PREREG_P6_SELECTORS_ARM.md`.

**Authorisation.** PI, 2026-11-09: *"Go ahead with the TTC and zero-tail arm first in Batch 2"*. A launch-gate PASS token is still required; tokens are never hand-written. P7 is paused for this arm only (section 7).

⛔ navtest is never used for training or for choosing anything in this card.

## 0 · Why this arm (read before the freeze; no bar below is derived from it)

Evidence: `F:/tanitad_scratch/p6sel/why/why_miss.json`, banked with this card as `why_miss.json`. E1 = C3, 1,058 navtest tokens, stratum-weighted, already read and descriptive. FIT = 8,489 navtrain sets, in-sample for P6. No intervals; this informs the design and decides nothing. E2 and E3 were not read.

| P6, picking from its own 64 proposals | E1 | FIT |
|---|---|---|
| pick, best of 64 | 91.92, 97.88 | 94.49, 98.10 |
| regret, points | 5.96 | 3.61 |
| regret on tokens whose pick has PDMS 0 | 2.74 (3.06 % of tokens) | 0.90 (1.01 %) |
| regret on tokens whose pick has TTC 0 and PDMS above 0 | 1.04 (2.62 %) | 0.99 (2.26 %) |
| regret on all other tokens (only EP is lost) | 2.18 | 1.72 |
| when the pick fails: median fully clean proposals of 64 | 11 | 21 |
| when the pick fails: tokens with no clean proposal | 18.8 % | 4.0 % |
| ADE from the failing pick to its nearest clean proposal, median | 0.51 m | 0.36 m |
| ADE from the failing pick to a typical other proposal, median | 1.23 m | 1.16 m |
| head probability, failing pick against its nearest clean twin: NC | 0.99 / 0.97 | 0.99 / 0.97 |
| same: DAC | 0.94 / 0.86 | 0.96 / 0.90 |
| same: TTC | 0.95 / 0.91 | 0.94 / 0.90 |
| PDMS if the truth chose among the rule's top 3 | 94.52 | 95.81 |

Fully clean = NC 1, DAC 1, TTC 1, valid. On failing picks the heads rate the failing plan as safer than its clean neighbour about half a metre away.

**The gap in the current training.** L_pair of `PREREG_SCORER1P6` keeps NC, DAC and TTC pairs only when the two plans differ in speed by at least 15 % of v0, so C0 has no pair loss and S1E excludes the near-twin pairs of equal speed by construction. Zero-GPU reading of P6's outputs (nine selectors, `H-P6-SELECT-1`) found no gain, and neighbour rules on the logits (descriptive, `neighbour_rule.py`, E1 deltas −0.89 to −9.20 for smoothing and worst-neighbour, +0.06 and +0.08 for the medoid of the top 3 and top 8) do not supply the missing information.

## 1 · Arm (starts from P6's `model_final.pt`, sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1`)

C0's recipe exactly (`--lr 1e-4 --b-mode expreward --lam 1.0 --require-pdm`, batch 8 × accum 8, `--max-updates 400`, seed 0, the same 60,000 train sets, FIT logs excluded), read only at update 400, model B, plus one loss:

**L_twin, on arm B's head logits, for heads h in {NC, DAC, TTC}:**
1. *Clean set* C = valid proposals with NC 1, DAC 1 and TTC 1 (PDM truth).
2. *Hard negatives* of head h: valid proposals j with true y_h,j < 1 that are among the **8 highest** proposals of the current arm-B rule NC·DAC·(5EP+5TTC+2C)/12 (sigmoid of the logits, computed without gradient).
3. *Pairs:* (i, j) with i in C, j a hard negative of h, and ADE(i, j) ≤ **1.0 m**, the mean over the 20 raw poses of the Euclidean xy distance of the two proposals.
4. *Loss:* −log σ(s_h,i − s_h,j), meaned per head over its kept pairs, then over the heads with at least one kept pair in the micro-batch. No speed-contrast filter and no F/S balancing. A set with no pair contributes none.
5. **λ_z** is set once by the gradient-norm rule ‖∇L_twin‖ = 0.5·‖∇L_BCE‖ over the first 50 training micro-batches (`--pair-calibrate`, forbidden in a launch argv). Never tuned on held-out data.

⚠️ One new lever, against C0. EP is not in L_twin. The 8 and the 1.0 m are fixed here and are not searched.

## 2 · Launch gate (profile `refe_s1z_p6`, a NEW dict; existing profiles and shared constants are not edited; all must PASS)

G-DVB on each new lever (twin radius, top-K, heads, λ). G-LIVE: non-zero scorer gradient from L_twin, control exactly 0.0 with the weight at 0, share of sets with at least one kept pair reported per head, and 0 on any head is a blocker. Reachability traced flag → namespace → loss → `backward()` at `file:line`. Deliberate-regression arms that must turn RED: the twin radius filter removed; the top-K mining removed; L_twin dropped from `loss`; a default that disables the term left out of the argv. Non-scorer weights bit-equal to P6's. A navtest log or token in the training data is refused, RED-tested. CPU preflight on a tiny 6-row model first, then the real preflight on the L40.

## 3 · Evaluation sets (navtest, eval only, as `PREREG_SCORER1P6` section 3)

E2 (2,000 tokens) and E3 (12,146 tokens). G0-E2 and G0-E3 apply. Logits on all E3 tokens; the harness scores only tokens whose pick differs from P6's; unchanged tokens carry P6's banked value.

## 4 · Endpoints and bars (committed)

**Estimator.** Paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the set's own logs: "another draw of logs" only. Training variance is read from F = |C0r − C0| on PDMS, E3, measured under `PREREG_SCORER1P6`. The planner is deterministic. **Tier:** T1-class open loop on E3; selection tier on E2.

| id | endpoint | bar |
|---|---|---|
| **P1, primary** | PDMS × 100 on E3 | **S1Z vs C0:** CI-lo > 0 and \|S1Z − C0\| > 2·F. **S1Z vs P6:** CI-lo > 0. |
| **M2, mechanism** | failure share on E3: share of tokens whose pick has PDMS 0 or TTC 0 | S1Z − P6: CI-hi < 0, and S1Z − C0: CI-hi < 0 |
| **G1, guard** | TTC, NC, DAC sub-score × 100 on E3 | each point ≥ −0.5 against P6 |
| **G2, guard** | ranking regret on E2 | CI-hi of Δregret against P6 < +0.5 points |
| **G3, guard** | CRUISE overspeed share (plan v4/v0 > 1.15) on E2 | CI-hi of the difference against P6 < +0.10 |
| **G4, guard (new: a safety loss can make the scorer timid)** | CRUISE slow-pick share on E2, same definition as `P6_RESULT.md` | CI-hi of the difference against P6 < +0.10 |
| **Goal status** | PDMS on E3 | GOAL MET if the point ≥ 94.55; GOAL NOT EXCLUDED if CI-hi ≥ 94.55; else BELOW GOAL |
| reported, not deciding | EP sub-score, each head's within-token AUC on E2, near-twin AUC on E2 (pairs as in section 1 without top-K), Shapley split of the remaining regret on E2, top-K curve | |

**Four families, each with its CI on the same tokens (binding):** as `PREREG_SCORER1P6` section 4. STRATEGIC is **PAUSED** (PI 2026-09-19), not a pass.

**Adopted.** S1Z if it clears P1 and G1 to G4. **Family clause** as `PREREG_SCORER1P6`: none adverse is ADOPT; one or more adverse is PDMS GAIN WITH COST, every adverse component named.

**Read count.** S1Z's E3 read counts in the programme's K ledger (`PROGRAMME_P6_SCORER.md`). A best-of-K statement uses a Bonferroni-adjusted interval and prints K. P1 above is the arm's own pre-registered test.

## 5 · Committed outcomes (each names its next lever)

- **Z1 GOAL MET** (adopted, point ≥ 94.55): train S1Z seed 1 as the replicate, then the closed-loop read of `PREREG_SCORER1P6` section 6.
- **Z2 GAIN, BELOW GOAL** (adopted, point < 94.55): the Shapley split on S1Z on E2 (zero GPU) names the largest head left; the next lever targets it. Stacking S1Z's logits with an adopted Batch 1 arm or a selector is a counted candidate under its own card.
- **Z3 HARM** (clears P1 or M2, fails a guard): if G4 failed, the same loss with speed-matched pairs only; if G1 failed, the failing head is removed from L_twin. Each under its own card.
- **Z4 NULL** (fails P1 and M2): the scorer cannot learn the twin distinction from its current inputs at this dose. Next: explicit relative features between proposals, under its own card, and then the 1,600-update dose.
- **Z5 SWAMPED** (a CI-lo > 0 inside 2·F): NOT PROVEN. Next: S1Z seed 1 and a second C0 replicate.
- **Z6 MECHANISM ONLY** (M2 holds, P1 fails, no guard caused it): the failures fell and the gain was spent elsewhere. Next: the Shapley split on E2.
- **Unnamed:** labelled as such; the Shapley split on E2 names the next lever.
- ⛔ A miss is reported as FAILED or NOT PROVEN as written. The bars do not move after a read.

## 6 · What is not claimed

No closed-loop claim. No claim about the human's behaviour. Near-twin statistics on E1 and FIT are design inputs. The tracker-made share of failures is measured on u400 only (`D-REFE-ZEROTAIL-3`, 42.0 % of DAC-only selection zeros); plans the tracker alone fails cannot be fixed by any scorer that does not model the tracker, and that limits this arm's reach.

## 7 · Cost and order (ESTIMATED; the run's own measurement replaces it)

P7 paused once, at a checkpoint: preflight and three mutants about 15 min, gate about 3 min, training about 1.15 h (S1E MEASURED 1.12 h), about 1.5 L40-h in all. The same pause runs the FIT arm logits of `PREREG_P6_SELECTORS_ARM.md` (about 20 min) before training starts. Then P7 resumes. Navtest logits on the dev box about 4.3 h GPU unless shared with another pass.
