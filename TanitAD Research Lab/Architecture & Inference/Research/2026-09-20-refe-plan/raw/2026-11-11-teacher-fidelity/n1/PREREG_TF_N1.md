<title>PREREG TF-N1</title>

# PREREG TF-N1: navtest confirmation of the tracker-aware DriveRL teacher target (Tc20s) under NAVSIM's LQR tracker

**Status: READY TO FREEZE, not frozen, not run.** Prepared 2026-11-12 by the Teacher Fidelity agent from
`../PREREG_TF_N1_DRAFT.md` (sha256 `0511e485…fed9a6`, unchanged, also in the repo at
`PK/raw/2026-11-11-teacher-fidelity/`), with the Master Mind's (MM) rulings of 2026-11-12 written in. Section 13 lists
every change from the draft.

**Freeze procedure. The order binds, and the chain checks it mechanically:**
1. The MM records the sha256 of this file. The file is frozen from that moment.
2. The MM appends the section 9 entry to the K ledger.
3. Only then does the MM launch the chain.

`tf_n1_chain.py` refuses to start unless sha256(this file) equals `--prereg-sha256` and the ledger holds
`TF-N1:<sha256>:E3`. No navtest score of any N1 arm exists before the freeze: this agent read none, and the build has
not been run on navtest.

## 1. Question and what is already known

**Question.** Take the teacher's own banked navtest rollouts. Convert them to the P5-S tracker-aware label Tc20s and
score them with NAVSIM v1.1's standard LQR-tracked pipeline. Two questions:
* (i) Does the result **confirm** the navtrain lever on navtest?
* (ii) Does it reach the **paper level**? DriveRL, Table 4: PDMS 95.8, DAC 99.9.

**Known before this read.** Every number below is exploratory on navtest and MEASURED unless marked.

| what | PDMS | DAC | source |
|---|---|---|---|
| T, the registered teacher (8-pose plan, LQR) | 89.81 [88.56, 91.00] | 95.23 | `PK/raw/2026-10-05-teacher-navtest/p4_result.json` |
| F2, the same plans with 8-pose tracker compensation | 94.10 [93.59, 94.60] | 99.01 | `f2_result.json`, paired +4.30 [+3.33, +5.32] |
| human, through the same harness | 94.55 [93.78, 95.23] | 100.0 | `p4_result.json` C2 |
| navtrain held-out, TF-1 (1,018 rows, 11 logs): Tc20s tracked | 94.36 [93.07, 95.79] | 99.02 | `tf1/RESULT_TF1.md` |
| navtrain held-out, TF-2b (687 fresh rows, 13 logs): Tc20s tracked | 94.80 [94.35, 95.46] | 100.0 | `tf2b/RESULT_TF2B.md` |

⇒ **ESTIMATED: Tc20s on navtest ≈ F2 = 94.1 ± 0.3.** N1 is a confirmation of a lever that is already
exploratory-measured. It is not expected to reach the paper level.

## 2. Population

**Primary population.** The 12,110 primary navtest tokens in 136 logs (D-REFE-TEACHER-NAVTEST-1). That is 12,146 minus
the 36 C1-known DB≠OpenScene tokens, the `over_bar` keys of `PK/raw/2026-10-04-navtest-final/frame_control_navtest_full.json`.
* All 12,146 tokens are scored.
* All 12,146 are reported as a sensitivity read.
* The population is identical to P4 and F2, so every comparison is paired.

**Rollout coverage. MEASURED 2026-11-12 by this agent on two paths** (`rollout_verify.json`):
* The 4 banked shards hold 12,146 rows, one per token, with 0 duplicates.
* Their token set equals the T seam's.
* All 12,110 primary tokens are covered, in all 136 logs.
* Every `traj20` is a finite [20, 3] array, and `dt` is 0.1 on every row (the 10 Hz protocol).
* The two paths:
  * the raw `.jsonl` on E: (`PK/raw/2026-10-05-teacher-navtest/full/`);
  * the repo `.jsonl.xz` at the tip `dbb7f5c4`.
  * Each xz has the same sha256 on both, and each decompresses byte-equal to its raw file.

## 3. Inputs (banked; nothing is re-rolled)

| input | path (`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`) | sha256 |
|---|---|---|
| teacher rollouts, 20 poses at 0.2 s from the 10 Hz sim | `PK/raw/2026-10-05-teacher-navtest/full/rows_s0of4.jsonl.xz` … `s3of4` | `4c74ac3a…a776`, `f3d5c0a8…8edb`, `ed683563…84ad`, `07096341…dde5` |
| the same, raw (E: only) | `…/full/rows_s{0..3}of4.jsonl` | `8037b4ae…ab91`, `7040c1b8…12f3`, `63e6b459…fc93`, `5eccbd39…2b19` |
| the registered T seam: `token`, `fingerprint`, `poses` [12146, 8, 3] f32, `sampling` | `PK/raw/2026-10-05-teacher-navtest/refe_p4_teacher_navtest.npz` | `29c8129f…e113` |
| **the banked per-token teacher scores** (C-ID-H reference) | `PK/raw/2026-10-05-teacher-navtest/csv/refe_p4_teacher_navtest.csv` (repo blob, LF) | `2c5ccb91…0960` |
| the same CSV as the harness wrote it (CRLF) | `E:/Projects/TanitAD/data/refe_navtest/score/refe_p4_teacher_navtest/refe_p4_teacher_navtest.csv` | `23b6a41d…a120` |
| banked human scores (diagnostic) | `…/csv/refe_p4_human_navtest.csv` (repo, LF) | `8526e00f…5930` |
| the 36 C1-known tokens | `PK/raw/2026-10-04-navtest-final/frame_control_navtest_full.json` | `1137f44a…d385` |
| token → log | `F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz` | `b0dc05f3…a990` |
| C-DERIV labels (navtrain) | `PK/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz` | `0a0866fe…6072` |

**Notes on the inputs:**
* The two teacher CSVs differ only in line endings. Their rows are equal, and both reproduce PDMS **89.7861** on all
  12,146 and **89.806** on the 12,110.
* The t0 ego state and the map come from the nuPlan test DB `E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test`,
  read through `navtrain_scenarios.build_scenarios_for_log` with `history_rows=1, future_rows=80`, as
  `p4_tracker_comp.py:99` and `transp_score --split navtest` do. The maps are at
  `E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0`.
* `N1_PINS.json` pins every input above, the code, and four files of the vendored NAVSIM devkit @ 3e8291b.

## 4. Arms

**The candidate. MM ruling 1: Tc20s is the PRIMARY candidate.** It was resolved mechanically from the draft's rule:
* TF-2: PRIMARY = C, SECONDARY = NOT SHOWN;
* TF-2b, on fresh logs: CERT NOT SHOWN, because CERT − Tc20s = 0.000 [0.000, 0.000] and Tc20s already reads DAC 100.0;
* ⇒ the candidate is Tc20s.

| arm | plan (`plans20`, ego frame, t = 0.2 … 4.0 s) | role |
|---|---|---|
| **N1-C** | **Tc20s** = `iterate20(ND, T, to_navsim(T), ego, ctl, None, "cubic")[0]` (P5-S code unchanged; = TF-2 `tc_re`) | **PRIMARY** |
| N1-T | T = the banked `traj20`, bit for bit (C-BUILD-T-IS-BANKED) | identity (control i) |
| N1-W | Tw = the same call's `[1]`, the wrong-sign iterate | deliberate regression (control ii) |
| N1-M | M = TF-2 `shape(ND, T, ego, U, +1)` then `iterate20(…, "cubic")[0]` | **CERT input only**: reported descriptively, decides nothing |
| **CERT** | per token, the first of [N1-C, N1-T, N1-M] whose **tracked** `dac` (the scorer's own, same row) == 1. If none passes: N1-C | **pre-registered SECONDARY: decides nothing** |
| human | W-TRANSP `--human-arm`: the DB's logged future (8 poses for tracked, 20 for transparent) | descriptive reference |

**The plans.** The scorer receives `to_navsim(plan20)`, the 8 poses at 0.5 s, for the tracked arm, and `plan20` itself
for the transparent arm. For N1-T, `to_navsim(traj20)` cast to float32 equals the registered seam's `poses` on 12,146 of
12,146 tokens (MEASURED 2026-11-12, C-SEAM pre-check). The tracked N1-T arm therefore receives the registered seam bit
for bit.

**CERT. MM ruling 1.**
* CERT is quoted only as **"teacher + map-certified selection"**, never as the teacher alone and never as checkpoint
  identity.
* It is TF-2's L3 rule (`tf2_analyze.cert`) with one disclosed change. When no candidate passes, TF-2 broke the tie by
  the largest tracked map margin; N1 uses N1-C instead, because the margin is not in the rows.
* On a none-pass token every candidate has DAC = 0, and therefore PDMS = 0. ⇒ The change **cannot move CERT's PDMS or
  DAC on any token**. It can move CERT's NC, EP, TTC and C means and its families on those tokens, and their count is
  printed.

**Admissibility.**
* M and CERT read the static map, the t0 ego state and NAVSIM's public tracker. They never read the log's future (PI,
  2026-08-03: a label privilege).
* CERT selects by the tracked DAC, which is part of the test metric (map only). Its navtest DAC is therefore partly
  guaranteed by construction.

## 5. Scorer of record

**The scorer.** W-TRANSP `PK/eval/instruments/transp_score.py` (D-REFE-INSTR-1), tracked arm:
* NAVSIM v1.1 PDMScorer;
* the LQR tracker plus the kinematic bicycle model, vendored @ 3e8291b;
* one scorer call per arm on [PDM-Closed, plan tracked, plan transparent];
* EP normalised against PDM-Closed;
* PDMS = NC·DAC·(5 EP + 5 TTC + 2 C)/12, the same aggregation as `pdm_scorer._aggregate_scores`.

**The condition.** W-TRANSP is the scorer of record **on the condition that C-ID-H passes**. C-ID-H proves it equal to
the pinned W3 harness, which produced 89.81, on every token.

**What W-TRANSP also supplies:**
* the tracker-transparent score of every arm (MM ruling 3), descriptive only, with E2's caveat that transparent comfort
  is never a driving result;
* the family payload.

**Tier.** NAVSIM v1.1 PDMS: a non-reactive pseudo-simulation of a 4 s open-loop plan, against logged agents.

## 6. Bars and outcomes (committed before any number; unchanged from the draft)

**PRIMARY contrast.** N1-C − N1-T, paired PDMS (×100), tracked, on the 12,110 tokens.
* **CONFIRMED** requires all three:
  * ΔPDMS ≥ **+3.0**;
  * the 95 % CI lower bound > **0**;
  * N1-C DAC ≥ **98.5**.
  * ⇒ Tc20s becomes the registered teacher-target protocol, as a new row D-REFE-TEACHER-NAVTEST-2. REFe's target and
    P5-C decisions then use it.
* **NOT CONFIRMED:** anything else.
  * ⇒ The navtrain → navtest transfer failed.
  * Next: a D1-style anatomy of N1-C's DAC failures, on CPU, before any further lever. No GPU.

**PAPER-LEVEL read (N1-C, on point values, with our interval printed beside it).**
* **PAPER LEVEL:** PDMS ≥ **95.8** and DAC ≥ **99.9**. ⇒ The TF navtest goal closes; move to the student.
* **NEAR:** PDMS ≥ **95.3** and DAC ≥ **99.5**. ⇒ Run the first unrun rung of `TF_DESIGN.md` §4.
* **BELOW:** anything else. ⇒ Report the gap decomposed by one-at-a-time sub-score swaps, then take the next rung.
  * The swaps are made in the PDMS formula on the arm's means, to the paper's NC 99.8, DAC 99.9, EP 91.5, TTC 99.1 and
    C 99.0. The residual (actual minus formula) is printed.

**Descriptive only, never a bar:**
* The human-anchored paper level is 95.8 − (94.8 − our human). With the harness's 94.55 it is 95.55 (ESTIMATED).
* N1-C − human, paired.

**SECONDARY, CERT: no bar, decides nothing.**
* Printed: CERT − N1-C, CERT − N1-T, CERT DAC, the choice counts, the none-pass count, and CERT's PDMS read against the
  paper-level thresholds.
* All of it carries the label "teacher + map-certified selection".

**Expected (ESTIMATED, written before any read, unchanged).**
* Tc20s: 94.1 ± 0.3, so CONFIRMED and BELOW.
* CERT: +0.3 to +0.7 on top, so most likely BELOW.

## 7. Controls (a failure voids the read: the analysis exits before any N1-C, N1-M or CERT aggregate exists)

### Before any navtest score (navtrain, or plan geometry)

* **C-DERIV (chain stage `cderiv`).** The build, pointed at the 1,018 navtrain held-out rows, must reproduce the banked
  P5-S labels within **1e-9 m**: Tc20s against `p5s_cubic` column 1, Tw against column 2.
* **C-DAM (validation, same stage).** On the same 1,018 rows, the M label built from
  `PDMDrivableMap.from_simulation(map_api, ego, pdm_closed._map_radius)` must be **identical** to the M label built from
  `metric_cache_parts()[5]`. The drivable union and the ego state must also be identical. This licenses the cheap map
  route on navtest.
* **C-BUILD-COV / C-BUILD-T-IS-BANKED / C-SEAM (chain stage `build`).** On navtest:
  * 12,146 of 12,146 tokens are derived, with 0 errors;
  * N1-T is the banked `traj20` bit for bit;
  * `to_navsim(traj20)` cast to float32 equals the seam's `poses` on every token;
  * the rows' `plan8` equals `to_navsim(traj20)` to 1e-12;
  * the token sets are equal.
* **C-PIN.** Every file in `N1_PINS.json` matches its sha256. The chain checks before starting, and the analysis checks
  again.
  * `N1_PINS.json` itself has sha256 `53a8e80e1424798df9a67effdb39b91e1d59a479220126357d59d9a823bdcc73`.
  * It holds 27 files, built from the tip `dbb7f5c4`, and was verified by a second fresh export (`ZZN1_PREP PASS 27 pins`).

### After scoring, all checked before any N1-C aggregate

* **C-COV (MM ruling 4 iii).**
  * 12,146 tokens, 0 error rows;
  * every arm (N1-T, N1-C, N1-W, N1-M, human, CERT) present on every token;
  * 12,110 primary tokens.
* **C-EST.** The analysis draws reproduce `eval/proxy_eval.boot` on N1-T PDMS to ≤ 1e-9.
* **C-ID-H (MM ruling 4 i).** N1-T tracked equals the **banked per-token teacher harness run**,
  `refe_p4_teacher_navtest.csv`.
  * The bar: |Δ| ≤ **1e-6** on each of NC, DAC, DDC, EP, TTC, C and PDMS, on **all 12,146 tokens**, with 0 tokens
    missing.
  * The aggregate then reads **89.7861** on all 12,146 and **89.806** on the 12,110.
  * ⭐ This doubles as W-TRANSP's open control "tracked == pinned W3 harness" (`INSTRUMENTS.md`; PREREG_P5C §10.5 (i)).
* **C-REG (MM ruling 4 ii).** PDMS(N1-W) ≤ PDMS(N1-T) − **10** on the 12,110. On navtrain, Tw − T was −36.5.
* **Diagnostic, never voids.** The human arm is compared with the banked human harness CSV, and the count of tokens
  over 1e-6 is printed. The banked human is NAVSIM's human; W-TRANSP's is the DB future.

### ⭐ PRE-REGISTERED FALLBACK, if C-ID-H ALONE fails

W-TRANSP is then **not** harness-identical on navtest. The primary verdict comes instead from the **pinned W3 harness**:
1. Score `refe_tfn1_tc20s.npz`, `refe_tfn1_tw.npz` and `refe_tfn1_m.npz` (written by the build, with the T seam's tokens
   and fingerprints), plus a **re-score of the registered T seam**, through `eval/score_p4.py` → `eval/harness/score_e.py`
   (`HARNESS_PINS.json`).
2. Read them with `tf_n1_analyze.py read --harness-csv …`, which uses the same code path. C-ID-H is then the re-scored T
   seam against the banked CSV, |Δ| ≤ 1e-6. A failure there voids the read.

**Where each read comes from under the fallback:**
* Transparent scores and families still come from the W-TRANSP rows, flagged "not harness-identical".
* The choice of route depends only on N1-T (C-ID-H), never on an N1-C number.
* It is still the same ONE K entry.

⚠ **UNVERIFIED operability.** `score_p4.py`'s `SP` constant still points at an old scratchpad as the harness's working
directory, and must be repointed at `PK/eval/harness/`. This agent did not run the harness, because any harness run on
an N1 seam is a navtest read. The fallback adds about 4 × 0.4 h.

## 8. Estimator

* **Interval.** Paired log-cluster bootstrap over the 136 logs, 10,000 draws, seed 20260927, 95 % percentile interval.
  * The draw scheme is `eval/proxy_eval.boot`'s, exactly: sorted tokens, groups by sorted log, one
    `rng.integers(0, G, G)` per draw.
  * C-EST checks it against `proxy_eval.boot` itself.
* **Bonferroni.** Beside every navtest interval, the Bonferroni interval at **K = 41**: the percentiles 2.5/41 and
  100 − 2.5/41, i.e. 0.061 and 99.939 (PROGRAMME_P6_SCORER rule 4).
* **What it answers.** *"Another draw of LOGS?"* only. The teacher rollouts, labels, tracker and scorer are
  deterministic, so these arms carry no training-seed or inference-seed variance.
  * The rollouts are banked, and re-rolling reproduces them to 8e-7 m (C3).
  * In the smoke (§14), a second derivation reproduced every label bit for bit. The N1 path also reproduced TF-1's and
    TF-2's earlier, separate scoring runs bit for bit.

## 9. K ledger (MM ruling 2)

**N1 is ONE K entry.** N1-T (identity), N1-W (regression), N1-M (CERT input) and CERT (secondary) are logged under it as
**non-selection reads**: none of them can be selected.

* **The ledger.** `PK/raw/2026-11-09-scorer1p6-prereg/selectors/ledger_K.json`. It holds n = 40 at the tip `dbb7f5c4`.

The MM appends:
```json
{"key": "TF-N1:<sha256 of this file>:E3", "n": 41, "time": "<freeze UTC>", "nominee": "TF-N1", "family": "TF", "set": "E3",
 "smoke": false, "arms": ["N1-C (candidate, PRIMARY)"],
 "non_selection_reads": ["N1-T (identity, C-ID-H)", "N1-W (deliberate regression, C-REG)", "N1-M (CERT input)",
                         "CERT (pre-registered SECONDARY, decides nothing)", "human (descriptive)"]}
```
⚠ If another entry lands first, n and K shift. Then K in section 8 is the n of this entry, and the MM corrects both
before freezing.

## 10. Four metric families (binding; MM ruling 3)

* **Instrument.** W-FAM `PK/eval/instruments/families_run.py` on `rows_cert_primary.jsonl` (the 12,110), for N1-C,
  N1-T, N1-W, CERT and human.
* **Estimator.** Log-cluster bootstrap with eid = log, B = 10,000, per family. Families are never pooled, and each
  carries its n.
* **Frames.** **executed_tracked**, the LQR states at 0.5 s, is **PRIMARY**. **reference**, the plan as submitted, is
  diagnostic: for N1-C it is deliberately pre-distorted. executed_transparent is also diagnostic.
* **LONGITUDINAL:**
  * EP and TTC (scorer);
  * speed MAE, bias and RMSE;
  * target-speed accuracy within 0.5, 1 and 2 m/s;
  * along-track error;
  * acceleration MAE;
  * distance-keeping (min headway, time-gap and TTC) where a lead exists. NO_LEAD windows are counted and excluded.
* **LATERAL:** DAC (scorer); heading, curvature and yaw-rate MAE; cross-track error. Steps below 0.5 m/s are excluded
  and counted.
* **TACTICAL:**
  * lateral and longitudinal decision accuracy, κ and confusion, from the programme's trajectory-derived labeller
    against the human path;
  * **not computable** here, with n printed: tactical goal-setting and command adherence. NAVSIM carries no manoeuvre
    label beyond the driving command, and REFe has no tactical goal head.
* **STRATEGIC: PAUSED** (PI, 2026-09-19, PI_DECISION_QUEUE item 25). Printed as PAUSED. Not a pass.
* **Not built.** A paired family CI across arms. Family contrasts between arms are descriptive, read from the per-arm
  CIs, and decide nothing here: the bars are PDMS and DAC.

## 11. The E3 caveat (MM ruling 6)

**Where E3 stands.** E3, the checkpoint-identity check, is **in progress**:
* runA = **94.848** on nuPlan Test14-random non-reactive closed loop, 261 scenarios
  (`../e3_identity/e3_runA3_score.json`), against the paper's Table A10 **94.50**;
* runB is running.

**The dependency.** N1's **teacher-level interpretation** depends on E3. **The bars and outcomes in section 6 do not
change under any E3 outcome.** How each outcome changes the reading:

| E3 outcome (TF_DESIGN §4 rung 4) | what an N1 PAPER LEVEL / NEAR / BELOW then means |
|---|---|
| **A**: both runs within about 1 point of 94.50 | Our checkpoint and closed-loop protocol reproduce the paper's teacher. An N1 gap to 95.8 is then a **navtest-protocol** gap: tracker interface, conversion, route or EP normalisation, not the checkpoint. A BELOW sends the ladder to rung 3, the EP lever, from the navtest side. |
| **B**: both runs ≥ 2 points below 94.50 | "Same teacher" is closed as NOT SHOWN. An N1 BELOW is then at least partly the checkpoint or rollout protocol, and the EP lever is not fixable without training. A PAPER LEVEL or NEAR would be a label-protocol result that exceeds what the checkpoint shows closed-loop, and must be re-examined before it is quoted. |
| in between, or the runs disagree by more than their spread | Reported honestly as in between. No new bar. N1's verdict stands as registered, with the interpretation marked "pending E3". |

**In every case.** CERT is never checkpoint identity. N1-C is "our checkpoint + the Tc20s tracker-aware target", not
"the paper's protocol".

## 12. Run (for the MM; CPU only, at most 2 processes, 0 GPU)

### Order

1. Freeze.
2. Append the ledger entry and land it.
3. Run prep in the foreground (about 1 min):
   `bash <n1 folder>/n1_prep.sh C:/Users/Admin/n1_run origin/agent/arch-inf-20260803`. It must print
   `ZZN1_PREP PASS`.
4. Put the landed ledger in the run directory:
   `git -C C:/Users/Admin/tanitad-push show "origin/agent/arch-inf-20260803:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-09-scorer1p6-prereg/selectors/ledger_K.json" > C:/Users/Admin/n1_run/inputs/ledger_K.json`.
5. Launch the chain:
   `powershell -NoProfile -ExecutionPolicy Bypass -File <n1 folder>/launch_n1.ps1 -Sha <frozen sha256>`. It launches
   detached through WMI, with `CreateFlags = [uint32]0x08000200` (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) and
   `cmd /c "python tf_n1_chain.py … >> chain.log 2>&1 < NUL"`. It runs at most 2 children, sharded by the stable SHA-1
   token hash.

### Output layout

Everything is under `C:/Users/Admin/n1_run/`. Do not write to F:, which is at its floor.

| dir | contents |
|---|---|
| `markers/` | `<stage>.done` / `<stage>.FAIL` |
| `cderiv/` | stage 0, navtrain: `derive_s{0,1}of2.jsonl`, `out/n1_build_report.json` (C-DERIV, C-DAM) |
| `build/` | `derive_s{0,1}of2.jsonl`, `out/n1_plans_{T,Tc20s,Tw,M}.npz`, `out/refe_tfn1_{tc20s,tw,m}.npz`, `out/n1_build_report.json` |
| `score/` | `rows_s{0,1}of2.jsonl`, the W-TRANSP rows (about 110 MB); `rows_cert.jsonl`; `rows_cert_primary.jsonl` |
| `fam/` | `fam_{N1C,N1T,N1W,CERT,human}.json` |
| `out/` | `n1_result.json`, `read.log` |
| top level | `chain.log`, which ends `ZZN1_CHAIN DONE` or `ZZN1_CHAIN STOP` |

### Resume and polling

* **Resume.** Relaunch the same command. Finished stages are skipped, and the children skip finished tokens and retry
  errored ones.
* **Waiting.** Use one script-based wait on `markers/read.done` or `markers/*.FAIL`, never a model loop.

### Standalone analysis (the chain runs it as stage `read`)

```
C:/Users/Admin/navsim-crun/venv/Scripts/python.exe n1/tf_n1_analyze.py read --rows score/rows_cert.jsonl --known36 inputs/frame_control_navtest_full.json
  --build-report build/out/n1_build_report.json --identity-csv inputs/refe_p4_teacher_navtest.csv --human-csv inputs/refe_p4_human_navtest.csv
  --pins n1/N1_PINS.json --pin-root . --proxy-eval w/pk/eval --prereg-sha256 <sha>
  --families N1C=fam/fam_N1C.json N1T=fam/fam_N1T.json N1W=fam/fam_N1W.json CERT=fam/fam_CERT.json human=fam/fam_human.json --out out/n1_result.json
```

### Cost

The figures are ESTIMATED from the navtrain smoke (MEASURED per-token times, uncontended, single process) and from
INSTRUMENTS' contended figure.

| stage | per token | CPU-h |
|---|---|---|
| cderiv (1,018, with the map check) | 2.5 s | about 0.7 |
| build (12,146) | 0.76 s derivation plus about 0.2 to 1 s scenario | about 3 to 4.5 |
| score (12,146, 5 arms) | 1.8 s scoring plus scenario, 2.0 to 3.4 s | about 7 to 11.5 |
| families (5 arms × 12,110) | — | about 0.5 to 1 |
| **total** | — | **about 11 to 18 CPU-h, about 6 to 9 h wall on 2 processes, 0 GPU** |

* RAM: about 1 to 1.5 GB per process. The chain waits for FreeVirtualMemory > 10 GB before every child.
* This is above the draft's 2.5 to 6.5 CPU-h. The draft costed the pinned harness, whose PDM-Closed comes from the
  metric cache. W-TRANSP recomputes PDM-Closed.
* After the run, bank `score/`, `fam/`, `out/` and `build/out/` (rows as `.xz`) into `teacher_fidelity/n1/run/`, verify
  them by sha256, then delete `C:/Users/Admin/n1_run` by its literal path and log it.

## 13. Changes from the draft (`../PREREG_TF_N1_DRAFT.md`, sha256 `0511e485…fed9a6`, kept unchanged)

1. **§4 candidate: resolved to Tc20s** (MM ruling 1, after TF-2b). The draft's "if TF-2b reads WORKS, CERT becomes the
   candidate" branch is closed: TF-2b read NOT SHOWN.
2. **New arm: CERT, a pre-registered SECONDARY arm that decides nothing**, with its quoting rule (MM ruling 1). It is
   selected after scoring from the scorer's own tracked DAC. When no candidate passes, it uses N1-C instead of TF-2's
   margin tie-break; this cannot move CERT's PDMS or DAC.
3. **New arm: N1-M**, the input CERT needs. It is descriptive only.
4. **Scorer.** The draft scored W3 seams through the pinned harness (`score_e.py`). This card scores through W-TRANSP
   `transp_score.py`, which adds the transparent scores and the family payload in the same pass (MM ruling 3). It is
   conditional on the new **C-ID-H**, with a pre-registered harness **fallback** (§7). The build still writes W3 seams
   for that route.
5. **C-ID changed.** It was "N1-T re-scored through the harness equals the CSV". It is now C-ID-H: "W-TRANSP N1-T equals
   the banked per-token harness CSV, |Δ| ≤ 1e-6 on 7 sub-scores × 12,146 tokens" (MM ruling 4 i). The exact artifact
   and both sha256 values are named.
6. **New controls:** C-DAM, C-BUILD-COV, C-BUILD-T-IS-BANKED and C-EST. C-PIN now pins the N1 code, the instruments, the
   inputs and four devkit files (`N1_PINS.json`), not only `HARNESS_PINS.json`.
7. **C-SEAM, extended.** It also checks the rollouts' own `plan8`, and it was **pre-checked** on 12,146 of 12,146 tokens
   (plan geometry only, no score).
8. **§2: rollout coverage verified on two paths, with sha256** (MM ruling 5).
9. **§9 K ledger.** ONE entry, with the controls listed as non-selection reads (MM ruling 2). The ledger path and its
   current n = 40 are named. The draft's "K becomes 41 to 43" alternative is dropped.
10. **§10 families.** executed_tracked is primary and reference diagnostic, by MM ruling, via W-FAM on the 12,110.
    Distance-keeping uses the W-FAM lead block. STRATEGIC is PAUSED (MM ruling 3).
11. **New §11, the E3 caveat** (MM ruling 6).
12. **§12 run.** A chain with mechanical freeze and ledger guards, a WMI detached launch, the output layout, and a
    re-costed estimate of about 11 to 18 CPU-h (draft: 2.5 to 6.5).
13. **Paper-level BELOW decomposition.** Made explicit: mean-field swaps to the paper's five sub-scores, with the
    residual printed.
14. **Unchanged:** the population, every bar (+3.0 / CI-lo > 0 / DAC 98.5; 95.8 / 99.9; 95.3 / 99.5), C-REG's −10, the
    estimator, K = 41, and the expectations.

## 14. Evidence that the path is right (navtrain held-out ONLY; no navtest token was scored)

**The smoke.** 43 TF-2 held-out rows, deliberately enriched with failures: all 10 TF-1 Tc20s DAC-zero rows and 25 rows
that M shifts, in 4 logs (`smoke/`). They went through `tf_n1_build.py` → `transp_score.py` → `tf_n1_analyze.py`.

**Bit-for-bit (max |Δ| = 0.0, 8 sub-scores, 43 of 43 tokens):**
* N1-T tracked against TF-1 and TF-2 `T_tracked`;
* N1-T transparent against TF-2;
* N1-C tracked against TF-1 and TF-2 `Tc20s_tracked`;
* N1-C transparent against TF-1;
* N1-W tracked against TF-1 `Tw_tracked`;
* N1-M tracked against TF-2 `M_tracked`;
* human tracked against TF-1 and TF-2 `H_tracked`;
* the CERT choice against `tf2_analyze.cert`, 43 of 43 (Tc20s 33, T 4, M 6, none-pass 0).

**One definitional difference, not a defect.** TF-2's `M_transparent` executes the shaped path *before* `iterate20`.
W-TRANSP's N1-M transparent executes the M *label*, as it does for every arm. On the 18 unshifted rows, N1-M transparent
equals TF-1 `Tc20s_transparent` exactly.

**Build controls:**
* C-DERIV: Tc20s max deviation 1.5e-14 m, Tw 6.3e-11 m (bar 1e-9).
* C-DAM: 43 of 43 identical.
* Without `--check-dam` the labels are identical to the run with it.

**Readout:**
* All controls PASS on the smoke, including C-EST at max deviation 0.0 against `proxy_eval.boot`.
* The `--harness-csv` fallback mode gives identical outcomes on the same scores.
* The smoke's PDMS values are those of a failure-enriched subset. They are not a navtrain estimate and must not be quoted.

**The chain itself (`smoke/chain_smoke_wmi.log`).**
* It was launched detached by `launch_n1.ps1` (WMI, `0x08000200`) on a fresh `n1_prep.sh` export, with the 43-token
  `p5s` subset, test pins and a test ledger, and `-StopAfter cderiv`.
* The guards refuse correctly on a stale pin, a wrong sha256 and a ledger without the key.
* The memory gate held the start while FreeVirtualMemory was ≤ 10,240 MB.
* It then ran 2 shards in parallel, merged, and wrote `cderiv.done` (C-DERIV and C-DAM PASS on 43 of 43). The process
  exited cleanly, without the silent `^C` deaths TF-2's WMI launches had.
* The `build`, `score`, `families` and `read` stages ran here only as individual commands (above). They did not run
  inside the chain, because their inputs are navtest.

## 15. Master Mind rulings at freeze (2026-11-12T06:48:28Z, before any N1 build, score or read on navtest)

These bind this card as frozen. Where they conflict with an earlier section, they win.

1. **The C-ID-H fallback route (§7, "PRE-REGISTERED FALLBACK, if C-ID-H ALONE fails") is WITHDRAWN.** A frozen card does not rely on a route that is untested: `score_p4.py`'s `SP` constant points at a stale scratchpad (§7's own UNVERIFIED note).
   - If C-ID-H fails, **N1 is VOID**. The analysis exits before any N1-C, N1-M or CERT aggregate exists (§7 heading), and no arm number is reported.
   - The instrument is then diagnosed, and any rerun needs an amendment written and hashed **before** any re-score.
   - `tf_n1_analyze.py read --harness-csv` must not be used for N1.
   - The K entry stays consumed: a VOID read is still a read.
2. **The CERT tie-break is ACCEPTED** (§13 item 3: when no candidate passes DAC, CERT takes N1-C). It cannot change CERT's PDMS or DAC.
3. **Memory and order:** the chain starts after E3 run B4 has finished and released its 4 processes. The chain's own FreeVirtualMemory > 10 GB wait stays.
4. **Launch method:** `launch_n1.ps1` uses `-ExecutionPolicy Bypass`, which this session does not use (execution policy is the PI's setting). The MM starts the IDENTICAL command line (§12 step 5: `tf_n1_chain.py --run-dir --prereg --prereg-sha256 --ledger --procs 2`, appended to `chain.log`, `< NUL`) through `Invoke-CimMethod Win32_Process Create` with `CreateFlags 0x08000200`, from its own PowerShell. The chain code and arguments are unchanged.
5. **K:** N1 is ledger entry **n = 41** (the ledger held n = 40 at tip `dbb7f5c4` at freeze), so K = 41 in §8 stands.
6. **E3 state at freeze (§11):** run A = **94.848** on Test14-random NR, against Table A10's 94.50. That score was aggregated post hoc because the run's output path lacked the challenge name; run B4 is running with the native aggregator as a cross-check. The bars do not change. §11 maps each E3 outcome to the reading.
