# SAFEHEAD: the CPU sim cannot test a safety head on TTC (V-NO), and the real R0 states say most of the TTC loss is NOT in the safety rows

Research Lab agent, 2026-10-16. PI decision executed: "experiment with the separate safety head". CPU only (0 GPU; `devbox_gpu.lock` never opened).
Pre-registrations (all hashed before the numbers they govern; re-verified): `PREREG_SAFEHEAD.md` `2103a1a3d8f8…0830ca` (18:30Z), A1 `PREREG_SAFEHEAD_A1_TTCPEN.md` `ffcfbc1e058f…70773b` (18:41Z, after controls, before any SH number), A2 `PREREG_SAFEHEAD_A2_REALSTATES.md` `4a935bfc726d…7931c5` (20:01Z, after V, before any A2 number).
Evidence class of every sim number: **ESTIMATED** (real scorer weights, labels, trajectories and the trainer's own closures; substituted visual-context cache; CPU fp32; offline selection read; not T1; not a driving claim). Estimator everywhere: paired log-cluster bootstrap, 10,000, seed 20260927, over the pool's logs, training state FIXED ("another draw of evaluation sets", never "another training run" nor "another inference run"; single seed, single data order).

## 1. Headline and outcome rows

- **Validity row: V-NO.** The sim's FULL arm does not reproduce R0's TTC loss at R0's own u250 horizon: regime T, K = 250, FULL lam 1 - FULL lam 0, pooled 364 sets, **P2 TTC +0.23 [-0.50, +1.04], P1 TTC +1.65 [+0.56, +3.05]** (x100; K = 50 -0.59 [-1.13, -0.10], K = 100 +0.12). The real R0 u250 state reads **-1.53** on the same 364 sets (phead control (d)). Same numbers as the MM's `k250T_read.json`.
- **Committed outcome, every design: SH-UNTESTABLE-TTC.** Under V-NO the sim cannot falsify a safety design on TTC. Per the prereg, the only TTC test is the A40 arm of the best-f design, SH_A (PI decision). **But read section 1.1 before spending it.**
- **f (regime P, K = 50, vs matched control; f_min = min(train, val))**: **SH_A 0.774** (train 0.777 [0.70, 0.87], val 0.774 [0.33, 1.00]); **SH_SG 0.687** (0.839 / 0.687, val CI [-0.37, 0.97]); **SH_F 0.418** (0.757 / 0.418, val CI [-0.46, 0.82]). All three keep far more than the head-only restriction did (phead best 0.164), because B keeps `score_dec`.
- **Sim safety proxies (not decision-grade under V-NO, reported as the prereg requires), P K = 50, vs matched control, pooled x100**: every design reads TTC **worse than FULL**, not better. FULL +0.27 / +0.16 (P1 / P2). SH_A -0.55 / **-0.85 [-1.70, -0.06]**, SH_SG -0.55 / -0.62, SH_F -0.27 / -0.50. NC within +-0.3 for all; DAC >= 0 pooled for all (DAC proxy UNVALIDATED, s.3).

### 1.1 The cheapest discriminating experiment left, run (A2: the REAL states, zero training)

Read the banked REAL R0 / R0b / C0 scorer states (A and B; these do lose TTC) through the safety-head selections on the 364 sets. P2 TTC x100, B-side view minus its control, pooled:

| state | FULL (B vs A) | SH_A-sel (B progress + A safety, vs A) | SH_F-sel (B progress + base safety, vs A progress + base safety) | f_sel SH_A / SH_F |
|---|---|---|---|---|
| R0 u250 | -1.53 | **-0.78** | -0.93 | 0.75 / 0.58 |
| R0 u500 | -1.25 | **-0.88** | -1.14 | 0.72 / 0.67 |
| R0 u1000 | -1.07 | **-0.75** | -0.99 | 0.73 / 0.70 |
| R0b u250 | -0.93 | -0.64 | -0.71 | 0.69 / 0.55 |
| R0b u500 | -1.25 | -0.84 | -1.05 | 0.70 / 0.67 |
| C0 u250 / u500 / u1000 (floor) | +0.00 / +0.00 / +0.00 | +0.00 / -0.00 / -0.00 | +0.00 / -0.00 / -0.00 | |

CIs (P2 TTC, pooled): R0 SH_A-sel [-1.53, -0.16] / [-1.82, -0.13] / [-1.73, +0.07] vs FULL [-2.67, -0.58] / [-2.42, -0.28] / [-2.34, -0.01]. P1 TTC (pick): R0 FULL -2.20 / -2.47 / -2.47 -> SH_A-sel -1.37 / -2.20 / -1.92. SH_A-sel NC P2 -0.15 / -0.20 / -0.17; pick PDMS FULL +1.53 / +1.31 / +2.32 -> SH_A-sel +1.47 / +0.97 / +2.36. The FULL column reproduces phead (d) to <= 2.1e-6 (x100); C0 floor holds in every view (`realstates.json` verdict: C0_floor_ok, identity_ok).
**Row A2 (as written): A2-PROGRESS-ROWS for SH_A-sel (0 / 3 R0 states >= -0.5) and for SH_F-sel (0 / 3).**
- Meaning: swapping in a BCE-trained (or frozen) safety head removes **30-49 %** of the real TTC loss (u250 49 %, u500 30 %, u1000 30 %) and keeps 0.72-0.75 of the J gain. **The other 51-70 % lives in arm B's PROGRESS rows**: the pick is pushed toward faster proposals by EP / C / lane logits, which a safety head cannot touch.
- The limit, stated in the prereg: B's progress rows here were trained with the safety channel open. A real SH run trains them differently, and the sim says in which direction: with the safety channel closed, EPO's progress push costs MORE TTC (every SH arm below FULL at P K = 50), and a diagnostic on FULL's own state (s.5) shows EPO's moves of the safety rows ADD TTC (+1.53 P2). So the selection-level 30-49 % is, if anything, an upper bound on what a safety head buys.
- Consistent with the MM's new input (`eval/RESULT_P2REP.md` @ 70a9b73, INHERITED, not re-verified here): P2's SUPERVISED seed replicate shows the same trade (TTC -0.49, EP +1.80). A progress-for-TTC trade that appears without RL is a progress-side trade, which is what A2 locates.

## 2. Design (from source, d06546a)

- Scorer `score_q_mlp` (81,408) -> `score_dec` (4 CrossBlocks, 4,213,760) -> `score_head` Linear(256, 7) (1,799) (`model.py:551-563`, `:699-709`). Rows NC 0, DAC 1, EP 2, TTC 3, C 4, DDC 5, lane 6. Three scorers live in every run: base (frozen, `scorer_finetune.py:629-631`, sx0 `:738-739`), arm A (model's own, BCE, `:728`, `:1359-1361`), arm B (`:634-637`, `:737`, `:1360-1362`); one backward of (lA + lB)/accum (`:1381`), per-arm clip + AdamW (`:1390-1395`).
- Selection `sel_score = agg_v1(p[:6]) * p[6]` (`:750-752`, `agg_v1 = NC*DAC*(5EP+5TTC+2C)/12` `:80-82`); EPO policy pi = log_softmax(log(sel_score(sigmoid(sx)) + 1e-6)/0.1) (`:188-191`), J exact over 64 (`:218-231`), reward `rl_reward.reward` (`rl_reward.py:152-189`).
- **SAFE = {NC, DAC, TTC}** = `rl_reward.safe_mask` (`rl_reward.py:87-89`), the reward's own safety gate; NC, DAC are multiplicative gates in the selection, TTC is the term the R0 family loses. Lane, C, DDC stay on the progress side.
- **Three designs (all run)**: comb = `where(SAFE, S.detach(), sxB)` feeds pi, J and the deployed pick; S = base (**SH_F**, frozen copy), arm A (**SH_A**, BCE-trained copy), arm B itself (**SH_SG**, stop-grad; its trunk still gets EPO's gradient, so its safety OUTPUTS are not EPO-immune). Init: S(theta_0) = sxB(theta_0) = base, so every design starts on FULL's exact policy. **Parameter cost: 0 new trainable parameters for all three**; inference +1 scorer forward (4,296,967 stored, 3 head rows used) for SH_F / SH_A, none for SH_SG. Variant named, not run: SH_A3 (copy trained on the 3 safety rows only; same EPO-immunity, costs a third scorer).
- Gradient diagram and the trainer flag: `SPEC_SAFETY_HEAD_FLAG.md` (`--safety-head {off, frozen, armA, stopgrad}`, default off; one changed argument in `train_losses`; the deployed pick in `evaluate()` `:849` and `ScoreDump` must read comb; G11-S zero-gradient probe + mutation; G-DVB entry).

## 3. Identity controls (all passed BEFORE any safety-head J number; `controls.json`, `logs/controls.log`)

| control | result |
|---|---|
| (e) view evaluator vs phead `Evaluator.evaluate`, every per-set array | equal (`np.array_equal`) |
| (a') `sh_losses` sep OFF vs the trainer's `train_losses` closure, one micro-batch at a moved state | lB, ep, lA equal (`==`), arm-B gradient max abs diff **0.0**, all 3 designs |
| (a) K = 3 closed loop, P, lam 1: sep OFF vs FULL | arm B max abs diff **0.0** (SH_F, SH_A, SH_SG); sep ON differs (6.0e-4): the separation is not inert |
| (a'') co-trained arm A vs banked FULL lam 0 `closed_P_0.0.pt` D[25], D[50] | rel L2 **0.0**, max abs 0.0 (bit-identical) |
| (b) EPO gradient on the safety source, lam_pg 1 | **exactly 0.0**: SH_F base copy (78 tensors), SH_A arm A (78), SH_SG B head rows 0, 1, 3. Same breath: B's EP row 0.041 / 0.027 / 0.019, B's `score_dec` L2 0.37 / 0.27 / 0.21 |
| (c) mutation `SAFEHEAD_MUTATE_NO_SEP=1` (detach deleted) | **RED**: 0.094 / 0.113 / 0.032 |
| (d) proxy validity (`proxy_validity_safety.json`, phead's rule + a real reference) | TTC **VALIDATED**, NC **VALIDATED** (P2 -0.42 / -0.47 / -0.38 vs real -0.20 / -0.23 / -0.33), DAC **UNVALIDATED** (no real R0 DAC B - A; it may block, never grant) |
| A1 lever reach (`ttcpen_reach.json`) | lam_ttc 0: ep and gradient equal the unpatched closure (0.0); 0.25: differ (0.097); the evaluator's R unchanged |

## 4. Numbers (regime P, K = 50; per pool train 302 sets / 30 logs in-sample, val 62 / 13 out-of-sample)

| arm | f train [CI] | f val [CI] | f_min | TTC P1 / P2 | NC P1 / P2 | DAC P1 / P2 | EP P1 | PDMS P1 | KL(base‖pi) | vs FULL lam 0 (g5 as registered): TTC P1 / P2 |
|---|---|---|---|---|---|---|---|---|---|---|
| FULL (ref, vs FULL lam 0) | 1 | 1 | 1 | +0.27 / +0.16 | +0.27 / +0.32 | +1.10 / +1.33 | +4.54 | +2.69 | (phead) | = |
| SH_A | 0.777 [0.70, 0.87] | 0.774 [0.33, 1.00] | **0.774** | -0.55 / **-0.85** [-1.70, -0.06] | +0.27 / -0.02 | +0.82 / +1.04 | +4.47 | +2.16 | 0.38 / 0.40 | -0.55 / -0.85 |
| SH_SG | 0.839 [0.77, 0.92] | 0.687 [-0.37, 0.97] | **0.687** | -0.55 / -0.62 [-1.42, +0.16] | +0.27 / +0.06 | +0.82 / +1.03 | +4.75 | +2.28 | 0.34 / 0.37 | -0.55 / -0.62 |
| SH_F | 0.757 [0.55, 1.05] | 0.418 [-0.46, 0.82] | **0.418** | -0.27 / -0.50 [-1.29, +0.23] | 0.00 / -0.08 | 0.00 / +0.44 | +4.04 | +1.57 | 0.14 / 0.15 | -0.82 / -0.85 |

FULL Delta J: train +0.0407, val +0.0333 (phead `full_ref.pkl`). P K = 25 and the per-pool proxies: `summary.json`. Pooled x100 with CI; P1 on val moves in 1.6-point steps.
Rows if V had been V-OK (for the record only, NOT the outcome): SH_A and SH_SG fail TTC at P (SH-NO), SH_F PARTIAL-on-P.

## 5. Diagnostic, not pre-registered, not a decision input (`diag_full_safety_rows.json`)

FULL lam 1's own state read with its NC / DAC / TTC rows swapped for the base's: P2 TTC -1.37 (vs +0.16 with its own rows), P1 -1.65 (vs +0.27) at K = 50; own-minus-swapped +1.53 / +1.92. In the sim, **EPO's moves of the safety rows ADD TTC** (the reward contains TTC at 5/12, so EPO raises the TTC logit of TTC-safe, high-R proposals). Closing that channel, which is what a safety head does, removes a protective move and leaves the progress push. HYPOTHESIS for the mechanism; it explains why every SH arm reads below FULL on TTC.

## 6. Applied rows, next lever (Rule Zero), what blocks it

- **Outcome (as written): SH-UNTESTABLE-TTC for SH_A, SH_SG, SH_F (V-NO). A2: A2-PROGRESS-ROWS.**
- **Recommendation (PI decision): do NOT spend the A40 on a safety-head arm alone.** The prereg names SH_A as the only A40 test; A2 says that even a perfect safety head reaches at most -0.75 to -0.88 P2 TTC on the real R0 states (g5 bound -0.5), and the sim says a real SH run would push progress harder.
- **Next lever = the objective** (prereg SH-NO / A2-PROGRESS next): (a) TTC-penalised reward R' = R - 0.25 (1 - ttc) (A1, queued; its sim T reading is OBJ-UNTESTABLE-TTC under V-NO, so it yields f only); (b) constrained EPO (maximise J s.t. E_pi[TTC] >= E_piA[TTC] - 0.005, dual weight per update). Both need an A40 arm to test TTC; blocked on the PI. Because P2's supervised replicate shows the same trade, the objective lever should be posed on the SELECTION, not only on RL.
- **Cheapest zero-training lever not run here (budget)**: the A2 harness with a TTC-heavier DEPLOYED selection (e.g. `5EP + w TTC`, w in {7.5, 10}) on the real R0 / P2 states: does it buy back TTC at a measurable PDMS cost? One `eval_views` pass per state (~1.5 min each); needs its own prereg; adjacent to `raw/2026-10-16-selrule-dac` (coordinate ownership).
- **The sim as an instrument**: it cannot carry R0's TTC mechanism (FULL +0.23 at K = 250). Anything that must be judged on TTC goes to real states (A2-style) or the A40.
- **Blocked, named**: the A40 arm (PI decision; also needs `SPEC_SAFETY_HEAD_FLAG.md` implemented, a matched control, a launch-gate PASS).

## 7. Still running (detached, below-normal priority, one python run at a time; `HANDOFF.md`)

T K = 250 arms (SH_SG running since 19:56Z, then SH_A, SH_F), then A1 (FULL + TTC penalty, P and T). **Under V-NO they can only add f at T; none can change a row.** The MM may stop them to free the CPU: kill by explicit PID — queue-1 bash 32048 / 27864 / 34780, python 38184 / 20424 (SH_SG T); queue-2 bash 31272 / 35468 (still waiting).

## 8. Deviations and disclosures

- The banked NC / DAC proxy values and R0c's real sub-scores were seen before the prereg (disclosed there). A1 and A2 are addenda written after a result that motivated them (controls; V), each before its own numbers.
- A2's first run hit a 595 s tool timeout after 6 of 8 states (`logs/realstates_attempt1_timeout.log`); the full rerun (`logs/realstates.log`, `realstates.json`) reproduces those 6 lines character for character (diff empty).
- A2 ran as a second process beside the queue's T arm (one short zero-training read, ~11 min, 15 GB free at start), a deliberate exception to "one run at a time".
- Thread counts: the sim arms used 6 threads, phead's `full_ref.pkl` 4: the banked-state cross-check differs by <= 8.0e-7 in J per set (`out/banked_P.pkl` xcheck), immaterial against Delta J ~ 0.03.
- The SH_F control at P is the banked FULL lam 0 state read through base safety rows (pre-registered); SH_A / SH_SG share FULL's control (their lam-0 run is B == A, bit-identical: (a'')).
- Four metric families: LONGITUDINAL partial (EP, TTC only; speed / headway n = 0, no instrument in a scorer-only sim); LATERAL n = 0 and TACTICAL n = 0 (the scorer selects among fixed proposals); **STRATEGIC: PAUSED** (PI ruling 2026-09-19, item 25), not a pass.

## 9. Manifest (the tree is a plain archive of d06546a, not a git repo: nothing staged; the MM lands the package from this path)

Package `C:/Users/Admin/AppData/Local/Temp/safehead/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-16-safety-head/`:

| artifact | note |
|---|---|
| `PREREG_SAFEHEAD.md` + `.sha256`, `PREREG_SAFEHEAD_A1_TTCPEN.md` + `.sha256`, `PREREG_SAFEHEAD_A2_REALSTATES.md` + `.sha256` | pre-registrations |
| `RESULT.md`, `SPEC_SAFETY_HEAD_FLAG.md`, `HANDOFF.md` | this report, the trainer spec, the running-queue handoff |
| `summary.json`, `controls.json`, `realstates.json`, `proxy_validity_safety.json`, `diag_full_safety_rows.json`, `ttcpen_reach.json` | results |
| `out/*.pkl` (P arms, banked controls; T / A1 arms land here) | per-set arrays, the inputs of every number |
| `sh_common.py`, `sh_closed.py`, `sh_controls.py`, `sh_banked_eval.py`, `sh_summarize.py`, `sh_realstates.py`, `sh_proxy_validity.py`, `sh_diag_full_safety_rows.py`, `sh_ttcpen.py`, `sh_ttcpen_reach.py`, `run_queue.sh`, `run_queue2.sh` | the instrument; reuses the phead + H-ADAM harness unchanged |
| `logs/` | every run log + `queue.txt` |
| scratchpad only: gradient banks `bank_s{0,1,2}.pt`, pools `phead/pool_*.pkl` | regenerable (H-ADAM / phead) |

## 10. SELTTC — a TTC-heavier DEPLOYED selection on the real states (Master Mind, 2026-10-16 ~20:55Z)

Pre-registration `PREREG_SELTTC.md` (sha256 verified against `PREREG_SELTTC.sha256` before the run). The agent's first attempt stopped after the
three R0 states (`logs/selttc_attempt1_stopped.log`); the Master Mind re-ran `sh_selttc.py` unchanged over all 8 states (`logs/selttc.log`,
`selttc.json`, 534 s, 4 threads). The three R0 lines of the re-run reproduce attempt 1 to the printed digit (deterministic read).
Evidence class: ESTIMATED (real banked scorer states, cached context, CPU fp32, offline selection; not T1). Estimator: paired log-cluster
bootstrap, 10,000, seed 20260927, 364 sets, training state FIXED (answers "another draw of evaluation sets" only).

**Committed outcome, as written: `CONTROLS_FAILED`.** Reach control (ii) requires every candidate to change the pick on >= 1 set of R0 u250
arm B; **S3 veto (tau = 0.5) changed 0** (S1 lin 94, S2 gate 103). Identity (i) and mutation (iii) PASS. No row is drawn.
- Why S3 did not reach: the veto changes a pick only when the original pick has TTC probability < 0.5 AND some other proposal has >= 0.5;
  that happened on 0 of 364 sets. Which non-binding case holds (every pick already >= 0.5, or every proposal < 0.5) is UNVERIFIED. **Class: built, tested, UNSATISFIABLE** (the R42 class): a candidate's parameter was fixed without deriving whether the quantity it
  thresholds can cross it on these states. ⇒ derive a veto threshold from the distribution it thresholds before committing it.
- **Diagnostic, NOT a row (controls failed):** had controls passed, every candidate reads **SEL-NO** — the TTC bar (B_new - A_new P2 TTC
  >= -0.5) is met at **0 / 3** R0 states for all three. P2 TTC x100, R0 u250 / u500 / u1000: original -1.53 / -1.25 / -1.07; S1 lin (TTC
  weight 5 -> 10) **-1.44 / -1.28 / -1.13**; S2 gate (x TTC) -1.48 / -1.38 / -1.24. Retention rho 1.05-1.15 (the gain is kept, even grown);
  C0 floor holds (all |TTC| < 0.005). R0b u250 / u500: S1 -0.87 / -1.24.
- **Reading.** Re-weighting TTC in the selection moves the real TTC loss by at most +0.09 (x100). Together with A2 (a separate safety head
  recovers 30-49 %, the rest sits in B's progress rows), the selection is not where the trade can be repaired.

**Next lever (Rule Zero), and what blocks it.** The objective: (a) constrained EPO (maximise J s.t. E_pi[TTC] >= E_piA[TTC] - 0.005, a dual
weight per update) or (b) the TTC-penalised reward R' = R - 0.25 (1 - ttc) (addendum A1). Both are trained levers: they need an A40 arm,
a trainer flag with its G-DVB entry, a matched control and a launch-gate PASS. **Blocked on the PI (compute spend) and on implementation
(0 GPU, ~1 day).** The A40 is idle while P6 waits for its twin bank.
