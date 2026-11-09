# PREREG_SCORER1P6: can a scorer-only fine-tune on P6 close the gap to 94.55 PDMS? (FROZEN 2026-11-09)

**Status.** Frozen by the Master Mind, by sha256, before any P6 label exists, before any arm trained, and before any E2 token was read. It is a new card and does not amend `PREREG_SCORER1.md` (sha256 `1f839af2...`), which keeps u400 as its base.

**Authorisation.** PI, 2026-11-09, on pausing P7 to run these arms first: *"Go for pausing as you recommend"*. A launch-gate PASS token is still required; tokens are never hand-written.

⛔ navtest is never used for training. Evidence class of every number below is stated where it is used.

## 0 · Why this experiment, and what was already read

**Committed by two verdicts.** `PREREG_P6` read NOT PROVEN and BELOW GOAL (PDMS 91.20 [90.41, 91.98]); `H-P6-SCORER-SPEED-1` read REFUTED. Both name a scorer fine-tune on P6 (`raw/2026-11-09-p6-navtest-read/`).

**The headroom, MEASURED on E1 (C3, 1,058 tokens, stratum-weighted to navtest; descriptive, one checkpoint, no interval), `ceiling/p6_ceiling_output.txt`.** The E1 tokens were already read before this card; no bar below is derived from them. They informed the arm design only (the EP rule in section 2).

| what is picked, using P6's own 64 proposals | PDMS | EP | TTC | DAC | NC |
|---|---|---|---|---|---|
| P6's pick | 91.92 | 87.52 | 96.45 | 97.79 | 99.14 |
| best true proposal of 64 | 97.88 | 95.49 | 99.65 | 99.81 | 99.94 |
| mean of the 64 | 67.59 | | | | |

The E1 weighted estimate for P6 (91.92) differs from the full navtest value (91.20) by 0.72; E1 is an estimator with its own error.

So the generator already holds proposals above the 94.55 target, and the regret of 5.96 points is a selection error. Exact Shapley over which heads are replaced by their truth (rule NC·DAC·(5EP+5TTC+2C)/12): **TTC 2.25, EP 1.89, DAC 1.32, NC 0.50**, sum 5.96. Replacing the comfort head alone by truth changes nothing (91.92).

**The EP label contains the violation terms.** On all E1 proposals with NC or DAC at 0 (39.75 % of proposals) the EP label is exactly 0; on clean proposals it is never 0 and has mean 0.869. Replacing the EP head by truth is worth +1.80 on clean proposals only, +2.52 on violating proposals only, and +3.11 on all.

**The head defect on CRUISE, MEASURED (T0, 154 tokens, 51 logs, `P6_SPEED_RESULT.md`):** gap in within-token Spearman against speed, predicted minus true: TTC −0.190 [−0.272, −0.102], NC −0.157 [−0.239, −0.079], DAC −0.154 [−0.266, −0.050], EP −0.139 [−0.204, −0.087]. Overspeed share 0.535 [0.356, 0.694]; plan v4 minus GT v4 +0.84 m/s [−0.15, +1.68].

## 1 · No blocking Stage 0

Stage 0's question (is the gap present off navtest?) cannot be asked cleanly on P6. All 24 held-out logs (`splits/sft_heldout_logs.txt`) are in `navtrain.yaml` and P6's trainer has no exclusion, so P6 very likely trained on them (UNVERIFIED on the pod bank). No other navtrain carve exists.

- The held-out sets stay a **trainer monitor only**. Any statistic read on them is labelled in-sample for P6 and decides nothing.
- The human gates `STAGE0_READING` and `P6_SPEED_READING` of `scorer1_chain.sh` do not apply. The new chain's gates are this card's sha256 and the launch-gate PASS token.

## 2 · Arms (all start from P6's `model_final.pt`, sha256 `968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1`)

All train the 6-output scorer only; the generator and trunk are frozen, so proposals are bit-identical to P6's. All share the SFT-4 recipe without its lane terms: `--lr 1e-4 --b-mode expreward --lam 1.0 --require-pdm`, batch 8 × accum 8, `--max-updates 400` with the trainer's cosine schedule, read **only at update 400**. Labels: PDM truth on P6's own proposals (about 60,000 train sets + 3,137 held-out sets; the exact counts are pinned in the launch-gate profile before the gate runs), from rows `train_grow_cmd_3c` that P6 trained on.

| arm | difference | role |
|---|---|---|
| **P6** | no training | zero point |
| **C0** | recipe, seed 0 | does more on-policy listwise training help P6? |
| **C0r** | C0, data-order seed 1 | training-variance floor F = \|C0r − C0\| (`H-ESTIM-SEED-1`) |
| **S1E** | C0 + λ_p · L_pair over NC, DAC, TTC, EP, seed 0 | the lever |
| **ENS3** | mean of the logits of C0, C0r, S1E, equal weight; no training | secondary, read after the verdict (section 5) |
| **S1** (conditional) | as S1E without the EP head | runs only under O3 |

**L_pair.**
- **NC, DAC, TTC:** exactly `PREREG_SCORER1` section 2 steps 1 to 3: true y_i > y_j, |v4_i − v4_j| / max(v0, 1) ≥ 0.15, F/S count-balanced at seed 20261107, a set with an empty side contributes none.
- **EP (new):** all pairs with y_EP,i − y_EP,j ≥ 0.05, with **no speed-contrast filter and no F/S balancing**. Violating proposals enter, with their label 0. The balancing that removes "fast is good" would delete the signal the head lacks.
- Why natural pairs and not progress-only pairs: truth on all proposals is worth more than truth on clean proposals alone (+3.11 against +1.80 on E1). Natural pairs carry both lessons the label contains: faster clean beats slower clean, and slower clean beats faster violating.
- **Per-proposal validity:** invalid proposals are excluded from every head's pairs.
- **Loss:** −log σ(s_i − s_j) on the head logits, meaned **per head** over its kept pairs, then meaned over the heads that have at least one kept pair in the micro-batch. This replaces the pooled mean of `PREREG_SCORER1` Amendment 1 item 2, because unbalanced EP pairs would otherwise outnumber the balanced ones.
- **λ_p:** fixed before any read by the gradient-norm rule: ‖∇L_pair‖ = 0.5 · ‖∇L_BCE‖ over the first 50 training micro-batches, measured by `--pair-calibrate` (forbidden in a launch argv), L_BCE being pointwise BCE without the expected-reward term. Never tuned on held-out data.

**Launch gate (profile `refe_scorer1p6`, a NEW dict; `PROFILES["refe_scorer1"]` and shared constants are not edited; all must PASS):**
- G-DVB on every pair lever, including the per-head balance and speed-contrast settings, and a built `nav_mode=command` fact.
- G-LIVE: non-zero scorer gradient from L_pair, control exactly 0.0 with `--pair-w 0`; share of sets with at least one kept pair reported per head, and 0 on any head is a blocker.
- Mutations that must turn RED: no balancing on a safety head; balancing applied to EP; L_pair dropped from `loss`.
- A navtest log/token refusal on the training data, RED-tested.
- The 6-output preflight has never run: a CPU run on a tiny 6-row model first, then the real preflight on the L40.

## 3 · Evaluation sets (navtest, eval only)

- **E1 = C3, 1,058 tokens.** Reported, never decisive. Read above.
- **E2, 2,000 tokens,** token-disjoint from C3, uniform over navtest, seed 20261108. The truth table of P6's 64 proposals is computed once on CPU from `proposals_p6final.npz`. **G0-E2:** the table reproduces P6's picks and per-token PDMS (`refe_p6final.csv`) on 100 % of its tokens, or the read stops.
- **E3, the full navtest, 12,146 tokens.** Each arm's logits on all tokens; the harness scores only tokens whose pick differs from P6's; unchanged tokens carry P6's banked value. **G0-E3:** each arm's 64 proposals are bit-identical to P6's on every token, or the read stops.

## 4 · Endpoints and bars (committed)

**Estimator.** Paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the set's own logs. It answers "another draw of logs" only. Training variance is read from F. The planner is deterministic (`PREREG_P6`'s bit-identity check on the second pass stands behind this). **Tier:** T1-class open loop on E3; selection tier on E2.

| id | endpoint | bar |
|---|---|---|
| **P1, primary** | PDMS × 100 on E3 | **C0 vs P6:** CI-lo > 0 and \|C0 − P6\| > 2·F. **S1E vs C0:** CI-lo > 0 and \|S1E − C0\| > 2·F, and S1E vs P6 CI-lo > 0. F on PDMS, E3. |
| **M1, mechanism** | Δρ = ρ(pred_h, v4/v0) − ρ(true_h, v4/v0), within-token Spearman, average ranks, on E2 CRUISE, a token enters h only if its true labels vary, n per head reported, mean over TTC, NC, DAC | S1E − C0 ≥ +0.5·\|Δρ_C0\| and CI-lo > 0 and \|S1E − C0\| > 2·\|C0r − C0\| |
| **G1, guard** | TTC, NC and DAC sub-score × 100 on E3 | each point ≥ −0.5 against P6 |
| **G2, guard** | ranking regret (best of 64 minus pick) on E2 | CI-hi of Δregret against P6 < +0.5 points |
| **G3, guard** | CRUISE overspeed share (plan v4/v0 > 1.15) on E2 | CI-hi of the difference < +0.10 against P6, and for S1E also against C0 |
| **Goal status** | PDMS on E3 | GOAL MET if the point ≥ 94.55; GOAL NOT EXCLUDED if CI-hi ≥ 94.55; else BELOW GOAL (DriveZero Table A13) |
| reported, not deciding | EP Δρ (the EP mechanism), each head's Δρ, CRUISE slow-pick share, Shapley split of the remaining regret on E2, top-K oracle curve | |

**Four families, each with its CI on the same tokens (binding).** LONGITUDINAL: EP, TTC, target-speed accuracy, overspeed, speed and along-track errors; distance-keeping is UNAVAILABLE at this tier and is read in section 6. LATERAL: DAC, heading, curvature, yaw rate, cross-track. TACTICAL: decision accuracy, selection quality (random, actual, best of 64), command adherence. STRATEGIC: **PAUSED** (PI 2026-09-19), not a pass.

**Adopted arm.** S1E if it clears P1 against C0 and G1 to G3. Otherwise C0 if it clears P1 against P6 and G1 to G3. Otherwise none.

**Family clause, kept and made visible.** `PREREG_P6` required that no longitudinal or lateral component separate adversely. Here each component is tested with a paired CI against P6:
- none adverse: verdict **ADOPT**;
- one or more adverse: verdict **PDMS GAIN WITH COST**, every adverse component named. Most of those components are distances from the human driver, which PDMS does not score. A PDMS GAIN WITH COST arm may be carried forward and may be used for the goal status, and it is never described as the better driver.

## 5 · Committed outcomes (each names its next lever)

- **O1 GOAL MET** (an adopted arm, point ≥ 94.55): confirm on a replicate (C0r already exists for C0; train S1E seed 1 for S1E), then the closed-loop read of section 6.
- **O2 GAIN, BELOW GOAL** (an adopted arm, point < 94.55): recompute the Shapley split on the adopted arm on E2 (zero GPU), then run the lever for its largest head. ENS3 is read first, as it costs no training.
- **O3 HARM** (S1E clears P1 or M1 and fails a guard): run **S1** (no EP), which isolates the EP term. If G3 is the failed guard and S1 clears G3, the next lever is the EP term with a pre-registered overspeed penalty.
- **O4 NULL** (no arm clears P1 against P6 and M1 fails): this recipe does not move the selection. Next: a four-fold larger dose on C0 (1,600 updates), then privileged-teacher distillation (`SCORER_GYM_DESIGN.md` section 2.7).
- **O5 SWAMPED** (an effect's CI-lo > 0 but inside 2·F): NOT PROVEN. Next: S1E seed 1 and a second C0 replicate.
- **O6 MECHANISM ONLY** (M1 holds, no arm is adopted, and no guard caused it): the safety heads moved and the picks did not gain. Next: the Shapley split on S1E and C0 on E2, then the lever for the largest head.
- **Unnamed** (any result matching none of O1 to O6): labelled as such; the Shapley split on E2 names the next lever.
- **ENS3** is read after the verdict and cannot change it. Bar against the adopted arm (or P6 if none): CI-lo > 0 on E3, plus G1 to G3. Its three members already exist, so it needs no training; it has no replicate floor, and that is stated with its result.
- ⛔ A miss is reported as FAILED or NOT PROVEN as written. The bars do not move after a read.

## 6 · Closed-loop read (after O1 or O2; descriptive, not part of the bar)

On the simulation session's rig, tier 1 scenes, with the adopted arm against u400's banked baseline: free-road speed hold (first-call plan v4/v0), time gap, junction behaviour, jerk, lane keeping, collisions and TTC, plus the deliberate-regression arms that must read their known values. The simulator's not-at-fault NC defect (`raw/2026-11-09-t2-tier1/TIER1_REPORT.md`) is stated with any number that carries it.

## 7 · Cost and order (ESTIMATED; each run's own measurement replaces it)

1. Dump P6's proposals on the pod: about 1.2 L40-h, with PDM labelling in parallel: about 3.9 h CPU. The launch gate's real-data check needs these labels, so **P7 is paused from the start of this step**, not from the PASS token.
2. Launch gate, then 3 arms × about 3.0 L40-h. Total about 10.5 L40-h for arms and held-out work, about 12 with the dump. Calibration, preflight and mutants are not included.
3. P7 resumes when the arms finish.
4. E2 table: about 2.5 h CPU. Arm logits on E3: GPU, measured at about 4.3 h per pass on the dev box (P6's seam), to be reduced if several scorers can share one backbone pass and P6's picks are then reproduced bit-identically.

## 8 · Differences from `PREREG_SCORER1`, stated

Base P6, not u400. Primary endpoint is PDMS on E3, because P6's defect is smaller and the goal is PDMS. Slow-pick share becomes reported (P6 is at 0.224). EP joins L_pair by its own rule. The loss is per-head mean. M1 is the three-head form of Amendment 2. Guards are read against P6. An overspeed guard is added. No lane head, no lane arm. Stage 0 is not blocking.
