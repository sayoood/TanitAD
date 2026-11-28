# PREREG_DATASCALE: SAFE-HEAD and EP-HEAD re-run on the 185-log navtrain cache, with a noise-margin CV rule (`H-REFE-DATASCALE-1`)

MM, 2026-11-28. **STATUS: DRAFT, not frozen.** It freezes (sha256 into `PREREG_DATASCALE.sha256`) once the build in 9 passes its tests, and **before any fit opens `index_185.csv`**. Parent cards: `PK/raw/2026-11-27-safehead/PREREG_SAFEHEAD.md` (SH), `PK/raw/2026-11-27-l2next/PREREG_EPHEAD.md` (EPH), `PREREG_SHMLP.md`. Everything not listed in 8 is the parent's, unchanged.

**State at drafting (disclosed):**
- **Parent outcomes (facts, MEASURED, banked):**
  - SAFE-HEAD: NULL-HELDOUT (`ed5a10ad`). SH-MLP: NULL-HELDOUT (`2708bca7`).
  - EP-HEAD: **ADVERSE** (`951e526c`). CV chose step 500 on both seeds, with an OOF gain of +0.113 / +0.046 on 2,272 tokens and 35 logs. Navtest went the other way: Delta0 -0.1237 [-0.2848, +0.0313] and Delta1 -0.1822 [-0.3288, -0.0409]. EP was separated worse on both seeds, and all four families were worse.
  - L3 EP-GEOM: NULL-HELDOUT.
- **Each parent's next arm is this card.** SH and SH-MLP named DATA-SCALE (card 2) on NULL-HELDOUT. EPH named "DATA-SCALE (8.1) with the adverse sub-scores carried into its guard" on ADVERSE.
- **The cache does not exist yet.** The x150 chain (PI go 2026-11-27, +150 logs) is building it at `F:/safehead_cache/x150/`. When this was drafted, about 15 of 150 logs were DONE. **No x150 token has been opened by any fit, selection or read.** The counts in 2.2 are the plan's (`raw/2026-11-28-safehead-data-x150/plan/PLAN.json`), not measured.

## 1. Question
The SAFE-HEAD diagnostic (`diag_veto/NOTE.md`) showed this pattern: on 29 fit logs the safety head cuts misses 69 -> 41, but on the 6 held-out logs they go 18 -> 20. EP-HEAD's navtrain OOF gain did not transfer to navtest.

**Does the same head-only recipe, fitted on about 5x more navtrain logs and selected by a rule that cannot open on noise, raise C1's navtest T1 PDMS?** And, descriptively: **does the out-of-fold gain grow with the number of fit logs?** That second answer decides whether more data is worth fetching.

## 2. Arms, anchor and data
| arm | recipe | rows trained | selection | role |
|---|---|---|---|---|
| **C1** | checkpoint `f09949e2...a34a4`, banked seam pick | none | none | comparator |
| **DSH_s0 / DSH_s1 / DSH_red** | SAFE-HEAD 3 (linear, qfl on the tracked safety labels) | 0, 1, 3, 5 | `cv5m` (4.3) | lever A: decision arm / R / CTRL-RED |
| **DEP_s0 / DEP_s1 / DEP_red** | EP-HEAD 3 (qfl on soft `tr_ep`, gamma 2.0) | 2 | `cv5m` (4.3) | lever B: decision arm / R / CTRL-RED |

- Seeds: s0 `--seed 0`; s1 `--seed 1`; red `--seed 0 --permute-labels-seed 20261127`.
- Lever A and lever B are read separately. ⛔ **No joint arm.** The composition DSH+DEP is named in 7.2 and needs its own card after both levers read.
- SH-MLP is not re-run: it is capacity, and this card moves data only. It is named in 7.3.

### 2.1 The anchor
The parents' 2.2, unchanged. It is a table read on `full_C1_s*of2.jsonl` (agg32 reproduces C1's picks; mean 92.55477798611808), with no new seam.

### 2.2 The data: `index_185.csv`, sealed
- **Contents.** The parent cache's 35 logs and 2,272 tokens (`F:/safehead_cache/out/`) plus x150's 150 logs, planned at 143 train + 7 held-out and 9,541 tokens before label drops (`F:/safehead_cache/x150/out/`).
- **Build.** `build_index_185.py` re-hashes every npz and fails on any duplicate log or token. `check_contract.py` and `sh_ctrl.py logit` run over x150 before sealing.
- **Sealing.** The parents' rule: a sha256 per file, re-hashed at load, and the launch token binds the index.
- **EP label caveat.** The parents' 2.3 caveat holds: `tr_ep` is a rebuilt metric cache and is approximate. Every navtest number is the harness-exact table's.
- **⛔ Folds.**
  - All N logs (N about 185), sorted by NAME, get fold = index mod 5. This is `safehead_data.cv_folds`, unchanged; only N grows.
  - The 13 held-out logs (6 + 7) are inside the folds, as in EPH 2.3. There is no separate held-out rule; both levers use 4.3.
- **⛔ CTRL-LEAK, unchanged.** No train or held-out log is a navtest log, and no token is shared.

## 3. The training recipe
The parents' 3, **unchanged in every value**: lr 3e-4, batch 32, 2,000 steps, saved steps 0/250/500/1000/2000, gamma 2.0, Adam, CPU-deterministic. Only the data grows.
⚠ At 2,000 x 32 = 64,000 token visits, the fit runs about 5.4 epochs of about 11,800 tokens, against 28 epochs in the parents. That is disclosed, not tuned. The learning curve (M6) and the per-step OOF table show whether 2,000 steps is short, and **a longer schedule is a named next arm (7.4), never a change here.**

## 4. The read
- **4.1 Logits and 4.2 selection.** EPH 4.1 / 4.2 generalised to the trained rows (DSH: rows 0, 1, 3, 5; DEP: row 2). Float32 argmax of agg32.
- **4.3 `cv5m`: the cross-validation rule WITH a noise margin and a guard. This is the deviation that matters.** For each fold k, a fresh model (same seed) trains on the other four folds and scores fold k at every saved step. That gives every token an out-of-fold pick per step, as in EPH 4.3. Then:
  - **OOF gain(s)** = the pooled OOF `tr_pdms`(s) minus `tr_pdms`(0), x100. Its CI is the paired **log-cluster bootstrap over the N navtrain logs** (`p5c_read.py` `Boot`, 10,000 draws, seed 20260927, 95 % percentile).
  - **OOF guard(s).** For each of NC, DAC, TTC, EP and C, the OOF sub-score delta at s vs 0 must not be separated worse: CI-hi >= 0. **EP is in this guard because EPH failed G on EP** (the "carried into its guard" clause). The same holds for every lever.
  - **A step s > 0 is ELIGIBLE iff CI-lo(OOF gain(s)) > 0 AND the OOF guard(s) holds.**
  - **The chosen step is the eligible step with the largest OOF gain** (ties go to the earliest). **If no step is eligible, the outcome is NULL-CV**: the arm is C1 and navtest is not read for a decision.
  - Also written, descriptive only: the parent's no-margin argmax step, so the effect of the margin itself is visible.
  - **The final head** trains on all N logs to the chosen step (prefix-deterministic, as EPH 4.3).
  - s1 applies the same rule to its own run. Everything is written to `cv_select.json` before navtest is touched.
- **4.4 Score.** The table's `tr.pdms` at the new picks. Tier T1 (NAVSIM v1.1 PDMS, LQR-tracked), 12,146 tokens / 136 logs, the paired log-cluster bootstrap. Which variance each interval answers: the navtest bootstrap answers LOGS, s1 answers TRAINING, and inference variance is zero.

## 5. Measures, bars and the committed reading (per lever, A and B independently)
- **M1-M4.** EPH 5's measures, unchanged, including **the four families in full** (LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC, per the PI 2026-11-16 unpause). These run through `run_fam_chain_eph.sh`'s **role-mapped** form, never the landed `safehead/run_fam_chain.sh`, which has the known tactical role bug.
- **M5.** EPH 5's M5, plus the CV-side guard table.
- **M6 (new, descriptive, does NOT decide): the learning curve.** Nested fit sets of n = 35 (exactly the parent's logs), 70, 110 and N logs, where each larger set adds x150 logs in sha256(log name) order. Each set runs the s0 recipe and 5-fold CV within itself and reports the OOF gain(s) with its log-cluster CI at every saved step. This is n x 5 extra CPU fits per lever.
- **F, P2.** F = |PDMS(s0) - PDMS(s1)|; F_REFe = 0.1158; **P2 = 2 x max(F, F_REFe)**.
- **Bars** (EPH 5, unchanged):
  - **P1:** CI-lo(Delta0) > 0.
  - **P2:** Delta0 >= P2.
  - **G:** no separated-adverse judged sub-score (NC, DAC, TTC, EP, C) on s0.
  - **R:** Delta1 > 0, and no separated-adverse PDMS or judged sub-score on s1.
  - Every control passes.

**Outcomes and next arms, per lever, in precedence order.** Rule Zero applies: each non-ADOPT outcome names its next arm, and the learning curve picks between them.
1. **INVALID.** A control failed. Fix it under a dated amendment and re-read.
2. **NULL-CV.** No step is eligible.
   - If M6's best-step OOF gain rises monotonically in n and its point at N exceeds its point at 35: **next arm 7.1, DATA-SCALE-2** (more logs; a PI decision on download and storage).
   - Otherwise: **next arm 7.5, Level 2** (the head on frozen `s` is exhausted; this is a PI decision on storage and GPU).
3. **ADVERSE.** CI-hi(Delta0) < 0, or G fails. The verdict names the sub-scores. **Next arm: 7.5, Level 2.** If the navtrain-eligible gain was separated and still did not transfer, that is a navtrain-to-navtest shift, so the fit data, not its size, is the defect.
4. **ADOPT** = P1, P2, G and R. The lever's rows become C1's read-out via `safehead_data.export_full`, with a SPEC amendment and a registry row, both landed by the MM. The goal status is printed as written: **BELOW GOAL** if the point is below 94.55. **Next arm: 7.2, the composition**, if the other lever also adopts; otherwise 7.3.
5. **SEPARATED-SMALL** = P1 and G, but P2 or R fails. Not adopted. **Next arm: the same lever with 3 seeds.**
6. **NULL.** The CI of Delta0 contains 0. Not adopted. Next arm as in outcome 2's learning-curve rule.
- **INCOMPLETE:** a family output is missing. That is a work item, never a result.

## 6. Controls (each can FAIL; any FAIL = INVALID)
The parents' controls, unchanged (CTRL-ID, DET-NT, FEAT, FEAT-NT, S, SET, T2, SHA, REACH, LEAK, CODE, RED), with these changes:
- **CTRL-CV** checks N logs and 5 folds by sorted name, and **(e) the chosen step equals `cv5m` recomputed from `cv_select.json`'s per-log OOF table.**
- **CTRL-RED** must choose step 0 under `cv5m`, **and its no-margin argmax is reported.**
- **CTRL-MARGIN (new).** A mutation that drops the CI condition, choosing argmax OOF gain with no margin, must change at least one synthetic fixture's chosen step; that turns the build test RED. A mutation that drops EP from the guard must also turn RED, on a fixture where EP alone is separated worse.
- **CTRL-PARENT (new).** At n = 35, M6's no-margin argmax step on lever B must reproduce EPH_s0's banked choice, **step 500**, with the same OOF PDMS to 1e-9. If it does not, the rebuilt pipeline is not the parent's.
- **CTRL-INDEX (new).** `index_185.csv` holds exactly the parent's 2,272 tokens bit-identical (by sha) plus x150's rows, and no log appears twice.

## 7. Named next arms (specified now; ⛔ none is computed under this card)
- **7.1 DATA-SCALE-2.** More navtrain logs, sized from M6's curve. A PI decision on download and storage.
- **7.2 DSH+DEP composition.** Both levers' rows together. Own card.
- **7.3 TRK+DS / ENS+DS compositions.** After the L1 TRK and L1b ENS reads. Own cards.
- **7.4 SCHEDULE.** The same data, with more steps if the OOF table peaks at 2,000. Own card.
- **7.5 Level 2.** Train `score_dec`, `score_q_mlp` and `score_head` on cached `visual_ctx` (SAFE-HEAD 8.3). Blocked on storage (3.93 MB per token) and a GPU; a PI decision.

## 8. Deviations from the parents (the complete list)
1. The data is `index_185.csv`, not the 35-log index.
2. SH's 6-log held-out rule is replaced by `cv5m` (4.3), for both levers.
3. `cv5m` adds a noise margin (CI-lo > 0) and an OOF guard on NC, DAC, TTC, EP and C. The parent EPH rule had neither.
4. M6, the learning curve, is new.
5. CTRL-MARGIN, CTRL-PARENT and CTRL-INDEX are new; CTRL-CV and CTRL-RED are extended.

## 9. Build (staged; the MM lands; no commit by any agent)
- **Code.** `refe/safehead_train.py --select cv5m` with the per-log OOF table. `refe/safehead_data.py` gains a bootstrap helper that reuses `p5c_read.Boot`, never a new estimator. An M6 driver. `refe/safehead_gate.py` gains profiles `refe_ds_{sh,ep}_{s0,s1,red}`, each new flag gets its G-DVB entry, and the new mutation arms are added. `launch_gate.py` is not patched.
- **Tests.** Synthetic fixtures for CTRL-MARGIN; and `test_safehead.py`, `test_safehead_next.py` and the pinned profile fingerprints unmoved (CTRL-SH-UNCHANGED for SH and EPH under `cv5`).
- **Freeze.** The card is frozen after the build passes and before any fit on `index_185.csv`.

## 10. UNVERIFIED at drafting
- The final token count after label drops.
- x150's EP parity against the harness. It is checked by a descriptive M5 rank-correlation, as in EPH 2.3.
- That 2,000 steps is not too short (M6 and 7.4).
