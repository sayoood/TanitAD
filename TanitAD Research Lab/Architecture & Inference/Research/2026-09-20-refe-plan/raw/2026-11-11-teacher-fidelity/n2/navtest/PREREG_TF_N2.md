<title>PREREG TF-N2</title>

# PREREG TF-N2 (READY TO FREEZE, not frozen, not run): the teacher re-rolled with its own positive-jerk limit 5.0 (RJ5), converted with P5-R, on navtest through the pinned harness

**Status: FROZEN by the Master Mind 2026-11-12T14:22:36Z** (sha256 in `PREREG_TF_N2.sha256`). Written 2026-11-12 by the TF navtest implementation agent, from `../TFN2_EVIDENCE.md`, `../reroll/` and the MM rulings below. No navtest token of any N2 arm had been rolled or scored at freeze.
- No navtest token was rolled, scored or opened by this agent. It read only the published N1 result files.
- Two fields are filled by the MM at prep, before the freeze: the tip sha (§11) and `N2_PINS.json`'s sha256 (§11). Nothing else is open.
- Freeze order, mechanically checked by the chain as in N1/A1:
  1. The MM runs `n2_nt_prep.py`, fills the two fields, and records sha256(this file).
  2. The MM appends the §9 K entry and lands it, then copies the landed ledger to `<run>/inputs/ledger_K.json`.
  3. The MM runs the `-CheckOnly` launch, then launches the chain.

**MM rulings applied (2026-11-12):**
- **The 5 % label warp is NOT adopted.** It is not an arm, a plan, a control or a fallback in this card. The earlier draft's RW105 and its warped regression arm are gone. It is mentioned here only as refused.
- **The candidate is the teacher re-rolled with positive-jerk limit 5.0 (RJ5)**, converted with **P5-R** (lambda* 3.0). RJ5 is the teacher's own policy setting (`max_jerk_long` 5.0 in its release yaml).
- Navtrain evidence: RJ5 + P5-R minus R0 + Tc20s = **+1.12 [0.40, 1.98]**. The 5 Hz arm (+0.79) is secondary at most and is not run here. Goal speed 8 is refuted (−0.19 [−0.82, +0.53] with P5-R, −0.45 [−0.94, −0.06] with Tc20s).

## 1. Question
On navtest, through the pinned W3 harness, does the RJ5 teacher with P5-R labels beat TF-N1's N1-C (Tc20s of the registered teacher) by a pre-committed margin, and does it reach the paper level?
- The paper level is DriveRL Table 4: PDMS **95.8**, DAC **99.9** (PUBLISHED).
- Known before this read (MEASURED, banked, `../n1/run_a1/RESULT_N1_A1.md`): N1-C **94.10** [93.60, 94.59], DAC **99.01**, EP 88.76. The gap to the paper is EP first (+1.13), then DAC (+0.84).
- Navtrain held-out, 469 tokens, 13 logs, W-TRANSP rebuild, EXPLORATORY (`../reroll/run/result/TFN2_REROLL_RESULT.json`):

| form (paired against R0 + Tc20s, the N1-C form) | ΔPDMS [95 % CI] | ΔEP | ΔDAC | PDMS |
|---|---|---|---|---|
| R0 + P5-R | +0.55 [0.09, 1.22] | +0.29 | +0.43 | 94.42 |
| RJ5 + Tc20s | +0.56 [0.14, 0.98] | +2.38 | −0.21 | 94.44 |
| **RJ5 + P5-R (the candidate)** | **+1.12 [0.40, 1.98]** | **+2.78** | **+0.21** | **95.00** |
| 5 Hz + P5-R (secondary, not run here) | +0.79 [0.09, 1.48] | +1.26 | +0.21 | |
| goal speed 8 + P5-R (refuted) | −0.19 [−0.82, +0.53] | −1.30 | 0.00 | |

- **Expected (ESTIMATED, written before any read):** N2-R about **94.7 to 95.3**, DAC about **99.0 to 99.2**. That is most likely **BELOW** the paper level, with NEAR (95.3) at the edge of the range. PAPER LEVEL would need +1.7, more than the navtrain point estimate (+1.12).
- The two levers are about additive on navtrain (+0.56 and +0.55 on their own, +1.12 together).

## 2. Population
The 12,110 primary navtest tokens in 136 logs. This is N1's population, with the same 36 C1-known exclusions (`frame_control_navtest_full.json`). All 12,146 tokens are rolled and scored and reported as a sensitivity read. Every comparison is paired with N1-C on the same tokens.

## 3. Inputs (banked; the only new data is the RJ5 re-roll itself)
All pinned by sha256 in `N2_PINS.json` (§7 C-PIN). The run tree is built from the git tip by `n2_nt_prep.py`: the N1 run directory was deleted and is not needed.
- **Banked teacher rollouts** `inputs/rollouts/rows_s{0..3}of4.jsonl.xz` (12,146 ok rows, mean 1.35 s/token, MEASURED from their own `sec`). They are N1's inputs and the R0 reference.
- **The T seam** `inputs/refe_p4_teacher_navtest.npz`, its banked harness CSV, the banked human CSV, and the known-36 file.
- **N1-C's banked harness CSV** `inputs/n1_csv/refe_a1_tfn1_c.csv`. It is the comparator and is **not re-scored in full**; its plan is re-derived and a 200-token subset is re-scored (C-ID-C).
- **Code, all from the tip, unchanged unless stated:**
  - the registered navtest roller `eval/teacher_navtest_seam.py`;
  - the engine-knob hook `refe/build_teacher_rollouts.py` (the TF-N2 re-roll patch, empty by default);
  - `tfn2_nt_roll.py` (this card; wraps the registered roller, §4);
  - P5-R `p5r_labels.py` (frozen with PREREG_P5R, lambda* = 3.0), `iterate20` with cubic interpolation, `tracked8`;
  - the A1 harness route `eval/score_p4_a1.py` → `eval/harness/score_e.py` (`HARNESS_PINS`, 21 files);
  - the A1 estimator `tf_n1_analyze_a1.py` (`Boot`).

## 4. Arms
Plans are 8 poses at 0.5 s (the T seam's tokens and fingerprints). The roller is the **registered navtest teacher protocol, unchanged**: 10 Hz, CPU, rank 0, step 0, history_rows 1, future_rows 80.

| arm | plan | role |
|---|---|---|
| **N2-R** | the RJ5 rollout's 20 poses → P5-R (lambda* 3.0, warm start Tc20s) | **PRIMARY candidate** |
| N2-C | the RJ5 rollout → Tc20s (`iterate20`, cubic) | isolates the jerk lever from P5-R; CERT2 input |
| **N1-C** | TF-N1's N1-C: the registered rollout → Tc20s | **comparator**: the banked harness CSV; its plan is re-derived for the families and verified by C-ID-C |
| T | the registered T seam, re-scored through the harness | identity (C-ID-H); CERT2 input |
| N2-W | the same call's wrong-sign iterate (`iterate20` [1]) of the RJ5 rollout | deliberate regression (C-REG) |
| CERT2 | per token, the first of [N2-R, N2-C, N1-C, T] whose **harness DAC = 1**; if none passes, N2-R | **SECONDARY, decides nothing** |
| human | the banked NAVSIM human CSV | descriptive |

**RJ5 and R0 are the only two roll arms, and nothing on the command line can add a third.** The arm table in `tfn2_nt_roll.py` is `ARM_JERK = {"R0": None, "RJ5": 5.0}`; there is no knob flag. RJ5 sets `positive_jerk_limit_when_nonnegative_acc` 1.0 → 5.0.

**Admissibility, stated per arm:**
- N2-R, N2-C and N2-W read the teacher's own rollout, the t0 ego state and NAVSIM's public tracker. They read no log future, no score and no map-derived label.
- CERT2 selects by the harness DAC, which is map-only (N1 ruling: partly guaranteed by construction). It decides nothing.
- ⚠ **RJ5 is a different teacher setting from the registered protocol** (jerk limit 1.0). Whether the paper's teacher ran 5.0 is not established. Every N2-R number is quoted as **"RJ5 teacher + P5-R"**, never as "the teacher" and never as checkpoint identity. CERT2 is quoted as "RJ5 teacher + P5-R + map-certified selection".
- ⚠ **The label warp is refused (MM ruling). It does not appear in any arm, control or fallback.**

**Reachability (the class "built, tested, unreachable from its caller").** The override is applied before the planner is built and re-applied after every `planner.initialize` (which overwrites it from the release yaml). After every roll the knobs of the **built** planner are read back and compared with the request; a mismatch makes the row an error row. Every ok row stores `knobs`. The chain fails a roll stage unless every token has an ok row with the right jerk value. Mutation arms (§7) prove each of these goes RED.

## 5. Scorer of record
The **pinned W3 harness** (`score_p4_a1.py` → `score_e.py`), exactly TF-N1-A1's route, each seam over all 12,146 tokens: the T re-score, N2-R, N2-C, N2-W, plus the 200-token N1-C subset. Its own status line must read PASS with the full CSV row count.
N1-C and the human are the banked harness CSVs. CERT2 is assembled per token from the arm CSVs; on a none-pass token PDMS = 0 for every arm, so the fallback cannot move PDMS or DAC.

## 6. Bars and outcomes (committed before any navtest number)

**PRIMARY.** N2-R − N1-C, paired PDMS (×100) on the 12,110 tokens. **CONFIRMED** requires all three:
1. Δ ≥ **+0.5**;
2. the 95 % CI lower bound > **0**;
3. ΔDAC (N2-R − N1-C) ≥ **−0.25**. The lever is ego progress and must not be paid for in drivable-area compliance.

Otherwise: **NOT CONFIRMED.**

*Why +0.5 (the MM may overrule it before the freeze).* The navtrain point estimate is +1.12 with a lower bound of +0.40. That estimate is the best of three knob arms and was read on 13 logs, so it carries a winner's-curse and small-n haircut. Halving it gives about +0.56, and the bar sits just below that and just above the lower bound. A real effect of the size navtrain suggests clears it. An effect that is only noise-level, or that the lever buys only on the 13 logs, does not. The CI condition then asks that navtest's 136 logs separate it from zero. Both outcomes are committed: Δ ≥ +0.5 with a separated CI is CONFIRMED whether or not the paper level is reached. Missing it is a FAIL, reported as written.

**PAPER-LEVEL read** (N2-R, on point values, with the interval beside them; unchanged from N1):

| outcome | PDMS | DAC |
|---|---|---|
| **PAPER LEVEL** | ≥ **95.8** | ≥ **99.9** |
| **NEAR** | ≥ **95.3** | ≥ **99.5** |
| **BELOW** | anything else | anything else |

On any level the gap is decomposed by one-at-a-time sub-score swaps to the paper's NC 99.8, DAC 99.9, EP 91.5, TTC 99.1 and C 99.0, with the residual printed (descriptive, never a bar).

**Four-family guard (PROPOSED to the MM, §10).** It never changes the PRIMARY verdict. It decides whether N2-R may become a student label.

**Secondary (decides nothing):** N2-C − N1-C (the jerk lever alone), N2-R − N2-C (P5-R on RJ5), CERT2 − N2-R and CERT2 − N1-C with the choice counts, the human-anchored paper level.

**What each outcome sends the ladder to (Rule Zero: no outcome ends at a verdict):**

| outcome | next step |
|---|---|
| CONFIRMED, PAPER LEVEL | the TF navtest goal closes; the PI rules on the guard flag (§10) before the label goes to the student |
| CONFIRMED, NEAR or BELOW | the next lever is the remaining EP gap, in this order: the 5 Hz policy interval combined with RJ5 (navtrain +0.79 alone, same re-roll code, one more arm), then CERT2's map-certified selection. Each is run first on navtrain with the re-roll chain (CPU, no GPU), then gets its own card |
| NOT CONFIRMED, DAC paid | a D1-style anatomy of the tokens where N2-R loses DAC against N1-C; if the DAC loss sits on tokens CERT2 repairs, CERT2 becomes the primary candidate of the next card |
| NOT CONFIRMED, no gain | the navtrain gain did not travel; the anatomy of EP by speed band on navtest tells whether the shortfall is the one navtrain showed (4 s path 0.94 of PDM-Closed's); then the re-roll levers not yet run |

## 7. Controls (a failure voids the read; the analysis exits before any N2-R, N2-C or CERT2 aggregate exists)

**Before the RJ5 roll:**
- **C-PIN.** Every code and input file matches `N2_PINS.json` (the chain's guard, and again in the read).
- **C-R0 (R0 re-roll identity).**
  - The **R0 subset is fixed in advance: the navtest tokens with sha1(token) mod 50 = 0** (about 243 of the 12,146; the chain refuses a subset outside 200 to 300).
  - The same roller, with **no override**, rolls them. Each token's 20-pose `traj20` must equal the banked rollout's to **1e-6** on x, y and heading, on **all** subset tokens, and every row must carry jerk 1.0.
  - **FAIL stops the chain before the RJ5 roll.** It means the re-roll environment is not the one that built the bank, so "N2-R minus N1-C" would mix engine drift with the lever.
  - Navtrain evidence: max |d| **0.0 m** on 469 tokens. A navtest roll of the R0 subset is an extra read of a teacher rollout, never of a score; it is logged under this K entry.

**After the build, before any harness score:**
- **C-BUILD.**
  - Every RJ5 token is derived, with 0 errors.
  - The RJ5 token set equals the T seam's token set (12,146).
  - `to_navsim(banked traj20)` as float32 equals the seam's poses bit for bit.
- **C-DERIV (navtrain).** The build, pointed at the TF-N2 re-roll's navtrain RJ5 rollouts, must reproduce the re-roll's scored plans (`arms.R` and `arms.C` `fam.reference8`) to **1e-9 m**. MEASURED on all 469 navtrain tokens: max deviation **0.0 m** for both R and C, 0 error rows (`raw/smoke/cderiv469_report.json`).

**After the T re-score, before any other arm is scored:**
- **C-ID-H.** The re-scored T seam equals the banked `refe_p4_teacher_navtest.csv`: |Δ| ≤ **1e-6** on NC, DAC, DDC, EP, TTC, C and PDMS, on all 12,146 tokens, 0 missing. **FAIL stops the chain.**
- **C-ID-C.** The re-derived N1-C plan on the **200 tokens with the smallest sha1(token)** is re-scored through the harness and must equal `refe_a1_tfn1_c.csv` within 1e-6 on all 7 metrics. It proves the comparator's plan (used for the families) is TF-N1's plan. **FAIL stops the chain.** It is an extra read of an already-read arm.

**After scoring (in the read):**
- **C-COV.** 12,146 tokens, 12,110 primary, every arm on every token, 0 error lines.
- **C-EST.** The draws reproduce `eval/proxy_eval.boot` to ≤ 1e-9.
- **C-REG.** PDMS(N2-W) ≤ PDMS(N1-C) − **10.0** on the 12,110 (TF-N1's N1-W read 52.38 against N1-C 94.10).
- **Diagnostic, never voids.** The human CSV is re-read; the count of tokens over 1e-6 is printed.

**Mutation arms (each control has one that must go RED; `test_tfn2_nt_chain.py` 55 tests, `test_tfn2_nt_roll_reach.py` 6 tests, and the smoke):**
- override dropped with the request kept → every RJ5 row is an error row, the chain's `roll_RJ5` FAILs;
- a banked rollout perturbed by 2e-6 → `ctl_R0` FAILs and no RJ5 row is ever written;
- a missing banked token → C-BUILD-COV FAILs; a 1e-6 deviation in a derived plan → C-DERIV FAILs;
- a 2e-6 breach of one token in the T CSV or the 200-token subset → C-ID-H and C-ID-C FAIL; one missing token fails both;
- the regression arm made as good as N1-C → C-REG FAILs; one token missing from an arm → C-COV FAILs; a pinned file edited → C-PIN FAILs;
- the verdict rule: a candidate equal to the comparator → NOT CONFIRMED; progress bought with DAC → NOT CONFIRMED.

## 8. Estimator
- Paired log-cluster bootstrap over the 136 logs, 10,000 draws, seed 20260927, 95 % percentile (`Boot` of `tf_n1_analyze_a1.py`, which reproduces `proxy_eval.boot`'s scheme exactly).
- Bonferroni at **K = 43** beside every navtest interval.
- It answers *another draw of LOGS* only. The policy is argmax, the CPU rollouts are deterministic, and the plans, labels, tracker and harness are deterministic. No training seed exists and there is no inference variance. The R0 identity (C-R0) is the check on that claim for the re-roll.

## 9. K ledger
**N2 is ONE entry, n = 43.** The ledger at tip `ed440366` (`raw/2026-11-09-scorer1p6-prereg/selectors/ledger_K.json`) holds 42 entries, max n = 42, the last being `TF-N1-A1:c26f7b8a…`. If another entry lands first, n and K shift: the MM corrects n here and the constant in the chain and the analysis, re-runs prep and re-freezes. The chain refuses a ledger whose top n exceeds 43.

```json
{"key": "TF-N2:<sha256 of the frozen card>:E3", "n": 43, "time": "<freeze UTC>", "nominee": "TF-N2", "family": "TF", "set": "E3", "smoke": false,
 "arms": ["N2-R (candidate, PRIMARY)"],
 "non_selection_reads": ["R0 re-roll identity subset (C-R0, rollouts only)", "N2-C (jerk lever alone)", "T (harness re-score, C-ID-H)",
                         "N1-C 200-token re-score (C-ID-C)", "N2-W (deliberate regression, C-REG)",
                         "CERT2 (pre-registered SECONDARY, decides nothing)", "human (descriptive, banked NAVSIM CSV)"]}
```

## 10. Four metric families (binding; STRATEGIC PAUSED)
**Instrument.** The pinned `transp_score.py` (W-TRANSP, as TF-N1-A1) rebuilds the scenery from the nuPlan DB, then `families_run.py`, on **N2-R, N2-C, N1-C (re-derived plan) and human**, over the 12,110 primary tokens. Frame **executed_tracked**. Values are labelled **"W-TRANSP scenery, not harness-identical"**: that route differs from the harness by 1 to 15 cm in object positions (`../n1/a1/WTRANSP_DIAG.md`). The A1 `--metric-cache-root` repair is NOT used: its command-line path is untested and no navtest token may be scored to test it. It is an MM option, outside this card. Every arm of a token is scored on the same scenery, so the contrasts are internally consistent.
- Each family is reported per family and never pooled, with its n. The read computes point values; no paired family CI exists (the N1 work item).
- **STRATEGIC: PAUSED (PI 2026-09-19).** It is reported as paused, never as a pass and never omitted. Longitudinal, lateral and tactical keep full force.

**The speed-bias cost of RJ5, reported explicitly.** RJ5 raises the teacher's progress, and it drives faster than the human.
- Navtrain reference (MEASURED, `TFN2_REROLL_RESULT.json`, RJ5 + P5-R against R0 + Tc20s):
  - speed bias vs human +0.64 m/s against +0.36 (**+0.28**);
  - speed MAE 0.876 against 0.699 (**+0.18**);
  - accel MAE 0.520 against 0.456 (**+0.06**);
  - heading MAE 2.69 against 2.61 deg.
- **Proposed longitudinal guard** (a proposal to the MM, not a bar; it is evaluated by the chain's `famread` and printed as `ZZN2_GUARD`). N2-R is flagged **"label-quality regression: goes to the PI before N2-R may become a student label"** if ANY of these holds, N2-R minus N1-C on the same tokens unless stated:

| check | bound | navtrain value |
|---|---|---|
| speed bias (signed, vs human) difference | ≤ +0.50 m/s | +0.28 |
| speed MAE difference | ≤ +0.30 m/s | +0.18 |
| accel MAE difference | ≤ +0.15 m/s² | +0.06 |
| N2-R's own speed bias vs human (absolute) | ≤ +1.00 m/s | +0.64 |

The bounds are about 1.4 to 2.5 times the navtrain values (ESTIMATED margin for a 13-log read). The flag changes no PRIMARY verdict. It decides only whether the label is handed on.
- Also printed beside the verdict: lateral heading, yaw-rate, curvature and cross-track errors, the tactical decision accuracies and the goal-point error, N2-R against N1-C.

## 11. Run (for the MM; CPU only, at most 2 processes, 0 GPU)
**Prep (before the freeze):** `python n2_nt_prep.py --run-dir F:/tanitad_scratch/tfn2_nt/run --mirror C:/Users/Admin/tanitad-push/.git --tip <landed sha>`. Tip sha: `54954b0f10d9a7c1c826c7bb5897b988804b491a`. `N2_PINS.json` sha256: `791247be0d1eaa5410c252a5ca08b9d1a9e5e27b1809b579314f09fe30c09482` (938 files; harness_pins OK, 21 files, caller NT_FIX1). It exports code and inputs from the tip, verifies the harness with its own pins, and pins every file.

**Chain** `tfn2_nt_chain.py` (guards, then stages, each with `markers/<stage>.done|.FAIL`; resumable):

| stage | what it does | stops the chain on |
|---|---|---|
| guards | card sha, ledger entry `TF-N2:<sha>:E3` n = 43, TF-N1-A1 n = 42 present, no n > 43, `N2_PINS.json` sha and every pin, no unpinned .py/.json/.yaml file, procs ≤ 2 | any refusal |
| `roll_R0` | the registered roller, no override, on the R0 subset, 2 shards | any missing or wrong-knob row |
| `ctl_R0` | C-R0 | FAIL (RJ5 never starts) |
| `roll_RJ5` | the same roller, jerk 5.0, all 12,146 tokens, 2 shards; error rows are dropped on a resume so they are retried | any missing, error or wrong-knob row |
| `build` | `tfn2_nt_build.py`: N2-R, N2-C, N2-W, N1-C plans → W3 seams and plan files; C-BUILD | any failed control |
| `score_T`, `cidh` | the T re-score through the harness; C-ID-H | FAIL (no N2 arm is ever scored) |
| `cidc` | the N1-C 200-token subset through the harness; C-ID-C | FAIL |
| `score_N2` | N2-R and N2-C together, then N2-W | a seam without PASS and the full CSV |
| `read` | `tfn2_nt_analyze.py read`, controls first | a failed control (no aggregate) |
| `families` | `transp_score.py` (N2-R, N2-C, N1-C, human) → `famprep` → `families_run.py` × 4 → `famread` (the guard) | a missing family file |

- Launch: `launch_tfn2_nt.ps1` (`Invoke-CimMethod Win32_Process Create`, `CreateFlags 0x08000200`, stdin from NUL; no `-ExecutionPolicy Bypass`). `-CheckOnly` runs the guards alone; `-Print` shows the command. The same command resumes. The chain waits for FreeVirtualMemory > 10 GB before each child; the harness waits for 6.5 GB of free RAM.
- Disk: about 3 GB on F: (rollouts, seams, harness CSVs, family rows). Free 36 GB at writing.

**Cost.** Measured pieces first.

| stage | basis | CPU-h |
|---|---|---|
| R0 subset roll | about 243 tokens × 1.4 s (the 5-token smoke: R0 1.68 s, RJ5 1.42 s per token, MEASURED) | 0.1 |
| RJ5 roll | 12,146 × 1.35 s (the bank's own mean, MEASURED) to 1.7 s | 4.5 to 5.8 |
| build | 5-token smoke: 0.51 s per token, MEASURED (2 × `iterate20` + P5-R) | 1.7 |
| harness | T, N2-R, N2-C, N2-W: 0.5 s per token each (TF-N1-A1's smoke, ESTIMATED), plus the 200-token subset | 6.8 |
| W-TRANSP rows | 5-token smoke: 0.92 s per token for 3 arms + human, MEASURED | 3.1 |
| `families_run` ×4 | not measured at scale | about 0.5 |
| **total** | | **about 17 to 19 CPU-h, about 10 to 11 h wall on 2 processes** (ESTIMATED; the T and N2-W harness runs are single-process) |

The earlier 9.4 s per token for a re-roll (TF-2b's record) is not what this machine shows warm; the 32 CPU-h figure would hold only under that load.

## 12. Evidence and its limits
- **Navtrain evidence is exploratory.** 469 tokens in 13 logs, W-TRANSP rebuild (not harness-identical). The knob values were fixed before the run, not tuned; RJ5 was the best of three knob arms, hence the haircut in §6. TF-2b showed CERT's navtrain gain is log-dependent. Standstill-heavy logs were not in the navtrain set.
- **The harness stages cannot be smoke-tested on a non-navtest token** (the harness has only the navtest metric cache), and this agent was not to score navtest. They are TF-N1-A1's code unchanged (`score_p4_a1.py`, the pinned harness), whose C-ID-H passed on all 12,146 tokens. The controls C-ID-H and C-ID-C are the guards; the chain's comparison and the read's controls are unit-tested with mutations.
- **C-R0 is the single point where the re-roll environment is tied to the bank on navtest.** It is a subset (about 243 tokens, all 136 logs sampled by token hash), not the full set.
- **What this read does not show:** that a student trained on N2-R labels gains this much (P5-C's question), and that RJ5 is the paper's setting.
- **Refused or out of scope:** the 5 % label warp (MM ruling); goal speed 8 (refuted); the 5 Hz arm (secondary, next card); the W-TRANSP harness-cache repair (MM option).

## 13. MM rulings at freeze
1. **The §6 PRIMARY bar is accepted unchanged:** Δ ≥ +0.5, CI lower bound > 0, and ΔDAC ≥ −0.25.
2. **The §10 longitudinal guard is accepted as written** (all four checks and their bounds). It is flag-only and never changes the PRIMARY verdict. A flagged N2-R goes to the PI before it may become a student label. "PROPOSED" in §6 and §10 now reads ACCEPTED.
3. **RJ5 is not the registered protocol.** Every quotation of an N2 number names it as "teacher with positive-jerk limit 5.0", never as the paper's teacher.
4. **K = 43** (the ledger at tip `54954b0f` holds max n = 42). The entry is appended in the same land as this file.
5. **Order:** guards → `roll_R0` → C-R0 → `roll_RJ5` → … ; any control FAIL stops the chain before an N2 PDMS exists.
