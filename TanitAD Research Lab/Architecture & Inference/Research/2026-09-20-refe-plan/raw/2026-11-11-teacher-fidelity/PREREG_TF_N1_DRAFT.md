<title>PREREG TF-N1 draft</title>

# PREREG TF-N1 (DRAFT): navtest confirmation of the tracker-aware DriveRL teacher target under NAVSIM's LQR tracker

**Status: DRAFT, not frozen, not run.** Written 2026-11-11 by the TF design agent. The Master Mind (MM) freezes it:
1. resolve the candidate rule in section 4 from `tf2/RESULT_TF2.md`, which is navtrain-only;
2. record the sha256 of the frozen text;
3. append the K-ledger entry of section 8;

and only then may the first navtest row be written. The rule binds the order: freeze, then ledger, then run. Every
navtest read so far is exploratory. This read is registered.

## 1. Question and what is already known

**Question.** Take the teacher's own banked navtest rollouts, convert them to the P5-S tracker-aware label (Tc20s) or
to the TF-2-selected label (section 4), and score them with NAVSIM v1.1's standard LQR-tracked pipeline. Do they
(i) **confirm** the navtrain lever on navtest, and (ii) reach the **paper level** (DriveRL, Table 4: PDMS 95.8,
DAC 99.9)?

**Known before this read. All of it is exploratory on navtest, MEASURED unless marked.**

| what | PDMS | DAC | source |
|---|---|---|---|
| T, the registered teacher (8-pose plan, LQR) | 89.81 [88.56, 91.00] | 95.23 | `RESULT_P4` §2, `p4_result.json` |
| F2, the same plans with 8-pose tracker compensation (p4_tracker_comp, 3 iterations) | 94.10 [93.59, 94.60] | 99.01 | `f2_result.json`, paired +4.30 [+3.33, +5.32] |
| human, through the same harness | 94.55 [93.78, 95.23] | 100.0 | `RESULT_P4` C2 |

On navtrain held-out, Tc20s and Tc8 (F2's method) have the **same** sim8 failure rate (0.98 %), and their labels differ
by a median of 0.0056 m (`p5s_cubic.json`). ⇒ **ESTIMATED: Tc20s on navtest ≈ F2 = 94.1 ± 0.3.** N1 is a confirmation
of a lever that is already exploratory-measured. It is **not expected to reach paper level** unless the TF-2 label is the
candidate.

## 2. Population

The **12,110 primary navtest tokens, 136 logs**, of D-REFE-TEACHER-NAVTEST-1: 12,146 minus the 36 C1-known
DB≠OpenScene tokens. This is identical to P4 and F2, so every comparison is paired.

All 12,146 tokens are scored. The harness is run with `--tokens all` because a token list overflows the Windows
32 KB command line (WinError 206, `score_p4.py`). The read uses the 12,110. An all-12,146 read is reported as a
sensitivity check.

## 3. Inputs (banked; nothing is re-rolled)

* **Teacher rollouts: banked.**
  * Where: `PK/raw/2026-10-05-teacher-navtest/full/rows_s{0..3}of4.jsonl`. The raw files are on E: (32 MB); the
    `.jsonl.xz` copies are in the repo.
  * What: 12,146 rows. Each `traj20` holds 20 rear-axle poses at 0.2 s from the **10 Hz** sim (`dt` 0.1, P4 protocol,
    C3 reproduces the bank to 8e-7 m).
  * ⚠ **Terminology.** Tc20s needs 20 *poses* on the 0.2 s grid, not a 20 Hz simulation. A 20 Hz-sim roll is the
    eliminated protocol: it fails C3's mutation arm, and E1 found it slightly worse.
  * Re-roll, only if ever needed: `eval/teacher_navtest_seam.py` via `eval/run_p4.py`. It runs at **1.42 s/token**
    (MEASURED, mean of `sec` over the 3,034 rows of shard 0), so 12,146 tokens take **4.8 CPU-h**.
* **Token order and fingerprints.** The registered T seam `refe_p4_teacher_navtest.npz` (`token`, `fingerprint`,
  `poses` [12146, 8, 3], `sampling`).
* **t0 ego state and map.**
  * The nuPlan test DB `E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test`, read through
    `navtrain_scenarios.build_scenarios_for_log`, exactly as F2's `p4_tracker_comp.py` does. Maps:
    `E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0`.
  * The token-to-log map: `F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz`.
    ⚠ The `EXPORT` constant in the P4 scripts still points to `E:/Archive/...`. It must be repointed. D: is gone.
* **Harness.** `PK/eval/harness/score_e.py` + `w3e/`, pinned by `HARNESS_PINS.json`, and driven by `eval/score_p4.py`
  (RAM-guard retries). The seam format is 8 poses at 0.5 s: the W3 agent refuses any other sampling (`w3_agents_v1.py:94-97`).

## 4. Arms (resolve the candidate rule BEFORE freezing)

| arm | what | role |
|---|---|---|
| **N1-C** | the candidate label → `to_navsim` → 8 poses at 0.5 s | **primary** |
| N1-W | Tw: the wrong-sign iterate, P5's negative control, produced by the same `iterate20` call as Tc20s | deliberate regression |
| N1-T | the registered T seam, re-scored unchanged | identity |
| human | banked `refe_p4_human_navtest.csv` | no new read |

**Candidate rule, mechanical, from TF-2 (navtrain only, so no navtest data informs it):**
* if TF-2 SECONDARY = WORKS, the candidate is CERT;
* else if TF-2 PRIMARY = A, the candidate is M (margin-shaped Tc20s);
* else the candidate is Tc20s.

**Resolved on 2026-11-12 from `tf2/RESULT_TF2.md`:** PRIMARY = C and SECONDARY = NOT SHOWN, so **the candidate is
Tc20s**.
* If the MM first runs TF-2b (CERT on fresh held-out logs, the same bar) and it reads WORKS, then re-resolve the rule
  before freezing. The candidate becomes CERT, and the cost in §10 becomes about 5 CPU-h.
* Do not freeze with CERT as the candidate on the strength of TF-2 alone.

⚠ **Admissibility of the candidate.**
* M and CERT read the static map, the t0 ego state and NAVSIM's public tracker, never the log's future. That is a label
  privilege the PI admits (2026-08-03).
* CERT selects its label by the tracked DAC rule, which is part of the test metric, computed from the map alone. Its
  navtest DAC is therefore partly guaranteed by construction. ⇒ **a CERT or M result is a label-quality result, not
  "our checkpoint equals the paper's".**
* The paper-level read below must name which arm it is.

## 5. Bars and outcomes (committed before any number)

**Primary contrast.** N1-C − N1-T, paired PDMS on the 12,110 tokens.

* **CONFIRMED.** Requires all three: the ΔPDMS point is ≥ +3.0; the 95 % CI lower bound is > 0; N1-C DAC is ≥ 98.5.
  ⇒ the candidate becomes the registered teacher-target protocol (new row D-REFE-TEACHER-NAVTEST-2). REFe's target and
  P5-C decisions use it.
* **NOT CONFIRMED.** Anything else. ⇒ the navtrain → navtest transfer failed. Next step: a D1-style anatomy of N1-C's
  DAC failures, on CPU, before any further lever. No GPU.

**Paper-level read.** Separate from the primary contrast, and committed. It is read on point values, because the paper
publishes point values, with our interval printed beside it.

* **PAPER LEVEL:** N1-C PDMS ≥ 95.8 **and** DAC ≥ 99.9. ⇒ the TF navtest goal closes; move to the student.
* **NEAR:** PDMS ≥ 95.3 and DAC ≥ 99.5. ⇒ run the first unrun rung of `TF_DESIGN.md` §4.
* **BELOW:** anything else. ⇒ report the gap decomposed by one-at-a-time sub-score swaps into the PDMS formula, then
  take the next rung of the ladder.
* *Descriptive only, never a bar.* Our harness reads the human at 94.55, where the paper's table has 94.8. The
  human-anchored paper level is therefore 95.55 (ESTIMATED; it assumes the offset is a property of the harness).

**Expected (ESTIMATED, written before any read).**
* Tc20s: 94.1 ± 0.3, so CONFIRMED and BELOW.
* M or CERT: +0.3 to +0.7 on top of that (navtrain TF-2, scaled by navtest's class-A rate of 0.59 %), so CONFIRMED and
  most likely BELOW.

## 6. Controls (a failure voids the read; no verdict is then taken)

* **C-DERIV.** The N1 build script, pointed at the 1,018 navtrain rows, must reproduce the banked P5-S Tc20s labels
  within 1e-9 m (TF-2 C2 already shows this for `iterate20` cubic). Run before the navtest seam is built.
* **C-SEAM.**
  * `to_navsim(traj20)` (float32) must equal the T seam's `poses` on all 12,146 tokens.
  * The N1-C and N1-W seams must carry the T seam's token order and fingerprints.
* **C-ID.** N1-T must reproduce the banked teacher CSV per token. The bar is |Δ| = 0 on every sub-score, and the
  PDMS must equal 89.7861 on all 12,146.
* **C-REG.** N1-W must satisfy PDMS(N1-W) ≤ PDMS(N1-T) − 10 (on navtrain, Tw − T was −36.5).
* **C-COV.** The harness status must read PASS, with 12,146 / 12,146 tokens and 0 misses.
* **C-PIN.** The `HARNESS_PINS.json` sha256 values must match before scoring.

## 7. Estimator

* **Interval.** Paired log-cluster bootstrap over the 136 logs (`proxy_eval.boot`), 10,000 draws, seed 20260927, 95 %
  percentile interval.
* **What it answers.** It answers *"another draw of LOGS?"*. The teacher, labels, tracker and harness are deterministic,
  so there is no training-seed or inference-seed variance on these arms.
* **Bonferroni.** Next to every navtest interval, print the Bonferroni-adjusted interval at K = 41: two-sided
  α = 0.05/41, i.e. the 0.061 and 99.939 percentiles (PROGRAMME_P6_SCORER rule 4).

## 8. K-ledger entry (the MM appends it at freeze, before the first row)

```json
{"key": "TF-N1:<sha256 of the frozen prereg>:E3", "n": 41, "time": "<freeze UTC>", "nominee": "TF-N1",
 "family": "TF", "set": "E3", "smoke": false, "arms": ["N1-C (candidate)", "N1-W (control)", "N1-T (identity)"]}
```
**MM to rule.** The draft counts this as one read event, because the two controls can never be selected. If the MM
rules instead that every arm counts, K becomes 41 to 43.

## 9. Four metric families (binding)

* **LONGITUDINAL:**
  * EP and TTC;
  * from `ev6e/families6.py`: speed MAE/RMSE, acceleration MAE, target-speed accuracy and headway.
* **LATERAL:**
  * DAC;
  * heading, curvature and yaw-rate MAE;
  * cross-track error.
* **TACTICAL:** the families6 lateral and longitudinal decision accuracy and κ, and the 5-way manoeuvre score against
  the human.
* **STRATEGIC:** **PAUSED** (PI, 2026-09-19, PI_DECISION_QUEUE item 25). Not a pass.
* ⚠ **Read the families on the EXECUTED trajectory (primary).** That trajectory is the LQR-simulated 8 poses (P5's
  `ego_frame` on `navsim_dac.simulated_states`). The seam's pose reference is read as descriptive only, because for
  N1-C it is deliberately pre-distorted (P4 F2-ii). This is **PI decision (a) from P4, which is still open**; the draft
  commits only to the order of reporting.

## 10. Script and cost

* **To write: `tf_n1_build.py`.** It is `tf2_score.py`'s derivation path, minus the scorer:
  1. iterate the T seam's tokens, grouped by log;
  2. build the scenario from the test DB;
  3. run one `iterate20` call (cubic), which yields Tc20s and Tw together; for M, run `shape()` and then `iterate20`;
  4. for CERT, run `navsim_dac.simulated_states` and the DAC rule on [Tc20s, T, M];
  5. write W3 seams `refe_tfn1_{cand,w}.npz` and a report json carrying C-SEAM.

  Shard it 3 ways and launch detached.
* **Cost** (ESTIMATED from P5-S's 269.9 s per 1,018 rows; the TF-2 per-row time is in `RESULT_TF2.md`):
  * Tc20s or Tw derivation: about 0.3 to 0.5 s per token, about **1 to 1.7 CPU-h**.
  * M or CERT: about 1.5 s per token, about **5 CPU-h**.
  * Harness: 1,204 to 1,439 s per seam (MEASURED, P4 status files) × 3 seams, run one at a time because of the RAM
    guard, about **1.2 h**.
  * **Total about 2.5 to 6.5 CPU-h, 0 GPU.**
