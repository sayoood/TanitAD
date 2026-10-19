# READY_P6_READ: the day P6's checkpoint lands (P6-READ-PREP, Research Lab agent, 2026-10-19)

**What this is.** The exact commands, in order, to read a candidate checkpoint (P6, or R0e-pick, or any later EP-led change) through the EP-ADMISSION gate. Everything below was exercised on synthetic banks, on EP-RELIAB-PICKS' real composite written as a bank, and on a 3-token CPU smoke of P2. No GPU was used and nothing was read on any real candidate.

**Evidence classes.** MEASURED = ours, artifact named. ESTIMATED = a stated approximation. UNVERIFIED = not run.

**Tier.** SELECTION tier, SEEN scenes, label-lookup PDMS: T0-class. A PASS is a necessary condition for a navtest look, never a capability claim. The interval answers "another draw of LOGS" only (`H-ESTIM-SEED-1`): not another training run, not another inference run.

Code directory below: `K=<repo>/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-p6-read-prep/code` (dev-box staging: `C:/Users/Admin/par18/p6prep/out/<same path>/code`). Interpreters: `PYB=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe` (bank builder, GPU) and `PYA=C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (admission; `driverl-eval` also imports it). Always `-I`.

---

## Part A. Do these now (no candidate needed)

1. **Re-run the instrument checks** (each 0 GPU; RAM gate 3 to 4 GB):
   ```
   cd "$K"
   $PYA -I test_ep_admission.py            # 53 tests, expect OK (about 40 s)
   $PYB -I test_build_candidate_bank.py    # 12 tests, expect OK (about 12 s)
   $PYA -I ep_admission.py branch-table    # every decision function's branch table and RED mutations: 35 of 35 caught
   ```
2. **Replicate EP-RELIAB through this tool** (the gate must read the known-bad EP-led composite as FAIL-A1; minutes, not timed separately):
   ```
   $PYA -I ep_admission.py selftest-real --p2-bank C:/Users/Admin/srttc_s0/work/bank --ranked-logs C:/Users/Admin/srttc_s0/run/plan/ranked_fit_logs.tsv --arrays-npz C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz --landed-result C:/Users/Admin/par18/epreliab/run/results/result.json --out selftest_real.json
   ```
   Result on 2026-10-19 (MEASURED, `raw/results/selftest_real.json`): PASS. The loader equals `arrays.npz` bit for bit; every EP-RELIAB number is reproduced with max difference 0.0; EP-R1's composite reads **FAIL-A1** (D_M +0.1349 [+0.1114, +0.1640] seed 0; +0.1331 [+0.1101, +0.1618] seed 1) and A2 ADMIT (NC [0.664, 1.168], DAC [0.837, 1.084], TTC [0.762, 1.089]); P2 as the candidate reads exactly 0 (C-ID).
3. **Read what the A1 bar means before sealing it** (reads no bank):
   ```
   $PYA -I ep_admission.py oc --half-width 0.0263 --bar 0.02
   ```
   ESTIMATED (analytic, normal approximation; the half-width 0.0263 is MEASURED on EP-R1's composite at 77.9 % changed picks):

   | bar | point that just ADMITS | P(ADMIT-A1) at true D_M = -0.02 | -0.01 | 0 | +0.01 | +0.02 | +0.05 |
   |---|---|---|---|---|---|---|---|
   | +0.02 (the draft) | -0.006 | 0.85 | 0.61 | **0.32** | 0.11 | 0.03 | 0.00 |
   | +0.03 | +0.004 | 0.96 | 0.85 | 0.61 | 0.32 | 0.11 | 0.00 |
   | +0.05 | +0.024 | 1.00 | 0.99 | 0.96 | 0.85 | 0.61 | 0.03 |

   Read: with an interval this wide, the draft's bar ADMITS a perfectly calibrated candidate (true D_M = 0) only about a third of the time. The empirical false-FAIL rate of A1 on shuffled picks is 0 of 200 (max null upper CI 0.0185), but shuffled picks do not have the structure of a real candidate's picks, so it does not bound this. The half-width depends on the share of changed picks: a candidate whose picks differ from P2's on few sets gets a diluted D_M; `C-POWER` then VOIDs it by resolution rather than ADMITting.

---

## Part B. The day of: build the bank (GPU), in this order

Set `R=D:/Projects/TanitAD-artifacts/p6_bank` (a directory on D:; `prepare` refuses C: and fewer than 12 GiB free). `P6=<path of the P6 checkpoint on D:>`.

| # | action | command | gate / expected |
|---|---|---|---|
| B1 | get the checkpoint onto D: and record its sha256 | `sha256sum "$P6"` | write the sha256 and the file size down; this is the candidate's identity |
| B2 | architecture check (CPU, seconds) | `$PYB -I "$K/build_candidate_bank.py" preflight --ckpt "$P6" --out preflight_p6.json` | `PASS`, same tensors and shapes as P2, `meta_differs_from_P2` empty. If `score_head` is not 256 to 6 the landed gate refuses every unit: STOP and report. If only names or shapes differ, run `smoke` on P6 first (section 5 of `build_candidate_bank.md`) |
| B3 | lay out the run root | `$PYB -I "$K/build_candidate_bank.py" prepare --root "$R" --ckpt "$P6" --ckpt-sha256 <sha from B1> --candidate-id p6` | `PREPARE_OK`; the 9 landed S0 files hash to their pins; S0's plan is copied (ranked_fit_logs sha256 `a61275037a3d...`) |
| B4 | rebuild the fit-set stores | `$PYB -I "$K/build_candidate_bank.py" sets --root "$R"` | **needs the pod idle and a coordinator OK for the pod fetch**; it runs the landed `srttc_s0_podsets.py`. `--skip-fetch` re-uses pulled data |
| B5 | the stores equal S0's | `$PYB -I "$K/build_candidate_bank.py" verify-sets --root "$R"` | `PASS`, 15,808 sets equal to S0's labels and `ckpt_step`. A FAIL means the pod's stores changed since S0: STOP before any GPU hour |
| B6 | dry run | `$PYB -I "$K/build_candidate_bank.py" dump --root "$R" --dry-run` | imports the landed driver with the four overrides, prints `SRC / CKPT / sha256 / OWNER`, takes no lock |
| B7 | launch, detached | `$PYB -I "$K/build_candidate_bank.py" launch --root "$R"` | prints the WMI PowerShell body. **GPU launch needs the coordinator's approval and the shared dev-box lock**; PASTE the body into a PowerShell prompt (script execution is disabled; `-ExecutionPolicy Bypass` is never used). The same command RESUMES |
| B8 | wait by a script, not by a model turn | `$PYB -I "$K/build_candidate_bank.py" status --root "$R"` | `S0_DONE` marker, 8,922 tokens, 15,808 sets. About 4 h GPU (ESTIMATED from S0's 1.589 s/token); any run longer than 4 min is waited by a script, and the session ends with a `HANDOFF.md` (token-budget rule) |
| B9 | bank manifest | `$PYB -I "$K/build_candidate_bank.py" manifest --root "$R"` | refuses without `S0_DONE` and exact counts; **prints the bank manifest sha256: copy it** |
| B10 | verify the bank | `$PYB -I "$K/build_candidate_bank.py" verify --root "$R"` | `PASS`: keys and labels bit-equal to P2's, `ctrl_fanpath_logit_maxabs` <= 1e-4, the bank's `ckpt` is the candidate's file, and `same_logits_as_p2_on_any_set` = 0. A nonzero value means the dump ran on P2: STOP |

`verify` compares logits only for that last equality and for the identity controls; it computes no EP, A1 or A2 statistic. That is the reason the registration can be sealed after the bank exists. The tool cannot prove that nobody looked at the bank; it proves the registration names THIS candidate and THIS bank and was hashed before the run.

---

## Part C. The Master Mind seals the EP-ADMISSION registration (after B9 / B10, before D1)

Draft: `raw/2026-10-19-ep-reliab-picks/DRAFT_PREREG_EP_ADMISSION.md` (dev-box: `C:/Users/Admin/par18/epreliab/DRAFT_PREREG_EP_ADMISSION.md`). File name the tool expects to find: `PREREG_EP_ADMISSION_P6.md`, sealed with a sidecar `PREREG_EP_ADMISSION_P6.sha256` whose first token is the file's sha256 (the tool looks for the same name with `.sha256`).

**The sealed text must contain** (the tool checks the first two; a run without a sealed registration is `authority: NONE`):

1. the candidate checkpoint's full sha256 (B1), and
2. every candidate bank manifest sha256 (B9), one per bank when several seeds are read.
3. The constants, copied from the draft: A1 bar +0.02; A2 band [0.8, 1.25]; A3 reported, not gated; B RECOVERS iff dPDMS x100 lower CI >= 0 and the NC and DAC unsafe-rate-difference upper CIs <= +0.001; paired log-cluster bootstrap B = 10,000, seed 20260927, 117 logs.
4. **The open choices below, each fixed in writing.** The draft leaves them open, the tool defaults are in the last column, and the data after each is what I measured.

| open choice | tool flag, default | what the evidence says |
|---|---|---|
| A1 statistic form | `--a1-form mass`; `auto` = EP-RELIAB's rule (ON-SCALE row -> Delta, else D_M) | `mass` is the draft's literal D_M and needs no scale call; EP-R1's row was off-scale (within-set slope 0.31), P6's is unknown |
| A1 bar | +0.02 | see the table in Part A: at a half-width of 0.026 the bar admits a calibrated candidate only about 32 % of the time (ESTIMATED). Choices that keep a rule honest: keep +0.02 and accept the risk, or fix a different bar NOW with its operating-characteristic row quoted. Do not move it after the bank is read |
| A2 semantics | "every CI overlaps [0.8, 1.25]" | as drafted this refuses only a significant miscalibration above 25 % (CI half-widths here are about 0.25 for NC, 0.12 for DAC, 0.16 for TTC on EP-R1); it is weak, and two-sided (an over-conservative row also refuses). Say whether two-sided is intended |
| B slot prior | `--slot-source label` (default) or `row` | on the EP-R1 composite (smoke, NO authority): `label` dPDMS x100 -0.725 [-1.01, -0.41]; `row` -0.211 [-0.529, +0.098]. The choice moves B's verdict; `lam` fitted ~0.30 per fold, out of fold, not clipped |
| number of banks | repeat `--cand-bank` | the label is ADMIT only if EVERY bank passes A1 and A2; a single-seed ADMIT is necessary, not sufficient (the bootstrap is blind to training variance) |
| in-sample status | stated text | P2 is in-sample on S0 and that biases the realised gain DOWN, toward OVER. State whether P6 was trained on S0's tokens or logs; if it was, A1 will read optimistic for it |
| what a VOID does | `NEXT_ARM["VOID"]` | the failing control is reported and the instrument repaired first; no verdict |

**Recommendation for the Master Mind (a suggestion, not a ruling).** Seal in two steps: the decision rule and the open choices NOW, as a hashed design file; then, after B9, a one-page `PREREG_EP_ADMISSION_P6.md` that quotes the design file's sha256 and names the checkpoint and manifest sha256. That fixes the bar before the bank exists at all. The tool accepts either way; it checks only the final file.

---

## Part D. Run the admission and decide

```
cd "$K"
$PYA -I ep_admission.py run \
  --p2-bank C:/Users/Admin/srttc_s0/work/bank \
  --cand-bank "$R/work/bank" --cand-id p6 \
  --ranked-logs C:/Users/Admin/srttc_s0/run/plan/ranked_fit_logs.tsv \
  --prereg <path>/PREREG_EP_ADMISSION_P6.md --candidate-ckpt-sha256 <sha from B1> \
  --a1-form mass --slot-source label \
  --out admission.json
```
About 90 s on S0's geometry (MEASURED: 93 s on the disk replication). Several seeds: repeat `--cand-bank <dir>` and `--cand-id <id>`. A run without `--prereg` refuses unless `--allow-unsealed`, and then it says `authority: NONE`: never quote it as a gate result.

**Read `admission.json` in this order:**

1. `controls_ok` and `controls`: C-KEYS / C-LABELS (the alignment), C-ID (P2 as candidate reads exactly 0), C-DECISION (35 RED mutations), C-REACH, C-POWER (a planted +0.10 over-prediction must FAIL A1, a neutral bank must ADMIT), C-NULL (5 shuffled-pick reps; VOID iff at least 2 fire). A failing gating control means VOID, no verdict.
2. `warnings` (the half-width versus the bar, a diluted D_M, an off-scale row, a `lam` outside [0, 1]).
3. `verdict.label`, `verdict.A1`, `verdict.A2`, `verdict.next_arm`.
4. `per_candidate.<id>.G.D_M` (point, lo, hi), `.A1_resolution` (half-width, the analytic operating characteristic at this bank's half-width, `D_M_changed_only`), `.A2.heads` (NC / DAC / TTC with their side: ABOVE = over-confident row, BELOW = over-conservative row).
5. `per_candidate.<id>.A3`: dPDMS x100, NC and DAC unsafe-rate differences, with CIs. Reported, not gated.
6. `B`: `recovers_all_banks`, `slot_source`, per-bank dPDMS lower CI and unsafe-rate upper CIs (secondary; the draft's prior is that it mostly reverts toward P2).

**Decision per outcome** (the tool's `NEXT_ARM`, from the draft's "committed next arms"):

| `verdict.label` | meaning | next arm |
|---|---|---|
| `ADMIT` | A1 and A2 pass on every bank | the candidate's shared navtest look, under `MM_RULINGS_HEADROWS.md` R1-R9. A1 / A2 are necessary conditions, not a gain; A3's CI is what is reported next to it |
| `REFUSE-A1` | the candidate's EP gain is not real on the selection tier | NO navtest look. Attribute the cause with the (D) decomposition of EP-RELIAB-PICKS and return the lever to the trunk |
| `REFUSE-A2` | the safety rows are mis-calibrated at the candidate's picks | NO navtest look. Register a calibration arm for THAT candidate, with reliability at the picks pre-registered |
| `REFUSE-A1+A2` | both fail | NO navtest look. Attribute the EP gain (D), register the safety-row calibration arm for THAT candidate, return the lever to the trunk |
| `VOID (...)` | a gating control failed, or C-NULL fired on 2 or more of 5 | no verdict; report the failing control and repair the instrument first |

Rule Zero applies to every refuse row: the turn does not end at the label. The next lever, with its result or its named blocker, goes in the same report.

**Same-turn duties after the read** (Master Mind): bank `admission.json`, the run logs and the registration with its sidecar into the repo package; write the D-row in `GOALS_AND_CLAIMS.md` and the registry line with the tier stamp (T0-class, selection tier, seen scenes); append `RETRACTION_LOG.md` if the read overturns an earlier claim.

---

## Part E. Disclosures (read before quoting any result)

- **The A1 bar is inside the noise** for an EP-R1-like interval (half-width 0.026 against a bar of 0.02): a calibrated candidate passes A1 about a third of the time (ESTIMATED). A REFUSE-A1 therefore needs its D_M point and interval read, not only its label.
- **A2 is weak as drafted**; it refuses only significant miscalibration above 25 %.
- **P2 is in-sample on S0**; it biases toward OVER. Whether P6 is, is a fact of P6's training set.
- **Interval scope**: the log-cluster bootstrap resamples logs with the models fixed. Another training run of P6 could read differently. Name which question a separated interval answered (`H-ESTIM-SEED-1`).
- **NOT reproduced**: the set path of the bank builder (`traj3` only on the pod) on CPU. The fan path was reproduced to 1.28e-3 against a 5e-3 tolerance (3 tokens).
- **UNVERIFIED**: the dump on an A40 (the pod environment for the nuPlan devkit and planner stack, and the lock protocol, are dev-box specific). The GPU path measured is the RTX 4060: 1.589 s/token, 3.94 h for 8,922 tokens.
- **UNKNOWN until B2**: that P6's checkpoint has P2's architecture and a 256 to 6 score head.
- **Dependencies**: `ep_admission.py` imports the landed `epreliab_lib.py` (sha256 `72f493fd150ef5165b5476d270138088046c6172216a207497174969e9f74878`, found through `EPRELIAB_CODE_DIR`, the repo layout, or the dev-box staging copy) and refuses a different copy. All commands above were run on the dev box layout; the repo layout lookup is covered by the path list but was not exercised.
- **Integration needs the Master Mind's sealing decisions** (Part C); until then every run is `authority: NONE`.

## UPDATE 2026-10-19 (Master Mind): EP-ADMISSION-R1 replaced two files of this package
- `code/ep_admission.py` is now sha256 `f0193372085022ab7bfc6241f2ca5638fca710cbd4170a19ce0ae95dde6637c4` and `code/test_ep_admission.py` is `f1eed39135f2a4b3f600f7de01607ca93889334da97027ad4efbe445046768c0` (76 tests).
- `--rules {draft,r1}` defaults to `r1`; the draft rules are reproducible bit-for-bit under `--rules draft`.
- The table above and `code/CODE_SHA256.json` describe the PREP build and stay as its record. Nothing verifies against them.
- Package, simulations and ruling: `raw/2026-10-19-ep-admission-r1/` (`RESULT.md`, `MM_RULING_EPADM_R1.md`).
- ⚠️ For the day-of read: the A1 bar is Revision 1's (FAIL iff D_M lower CI > 0 or point > +0.05, any seed), NOT the +0.02 quoted above. A2 is FAIL iff any NC / DAC / TTC observed / predicted lower CI > 1.05, pending EP-ADMISSION-R2-CONFIRM (else 1.10). B's slot source is `row`. Name `--rules r1` in the sealed registration.
