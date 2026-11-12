# P5-R HANDOFF (2026-11-12, run launched 08:11:01Z; the agent ended while the run was still in progress)

## State
- **Card FROZEN:** `PREREG_P5R.md`, sha256 `cfd23e0651d66e497238a0708c717ba00acbcb2189295efcfb15dad657922a42`,
  frozen 2026-11-12T08:03:43Z (`PREREG_P5R.sha256`). There is no addendum. ⛔ Never edit the card.
- **Code** (hold, LF):
  - `p5r_labels.py`: the solver; pure numpy; the tracker is injected.
  - `p5r_run.py`: the driver. Its stages are prep, controls, tune, select, eval, verdict and all.
  - `test_p5r_labels.py`: **12/12 PASS**, including C-R1 GREEN / drop-lambda mutation RED, the zero control, C-R2,
    C-R3, the call-count formula and no-scipy.
  - Run with: `C:/Users/Admin/navsim-crun/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider test_p5r_labels.py`.
- **Prep (done, MEASURED):**
  - 482 eligible logs: 123 TUNING / 359 EVAL.
  - Samples: TUNING 250 per rank, EVAL 1,200 per rank. 0 sha mismatches, 0 rows with `sim_hz` ≠ 10.
  - Full-bank raw median max heading rate: rank 0 **4.052 °/s**, rank 1 8.966. Acc: 1.438 / 1.456 m/s².
- **Smoke** (TUNING rows only; plumbing, read for errors only):
  - All controls PASS on 4 rows.
  - Timing: a = 16.4 ms/call, b = 2.93 ms/proposal, t_bld 1.35 s/row, K̄(λ=1) = 9.5.
  - Projected 1.70 CPU-h, inside the 3.0 budget.
- **Running:** `p5r_run.py all --work F:/tanitad_scratch/p5r_agent/run`. It was detached through WMI: the cmd
  wrapper is PID 35180, python is PID 2260 (venv launcher pair 2260/21564). The log is
  `F:/tanitad_scratch/p5r_agent/run/chain.log`.
  - The chain order: controls (100 rows) → GATE check → budget rule → tune (rank 1 in a child, rank 0 inline) →
    select (writes `lambda_star.json` BEFORE any EVAL row) → eval → verdict → copies the raw files to the hold's `raw/`.
  - ETA ≈ 1 h wall.

## How to check it (no polling loops: one look)
```
tail -5 F:/tanitad_scratch/p5r_agent/run/chain.log ; tail -2 F:/tanitad_scratch/p5r_agent/run/{tune,eval}_r1.log
ls -la F:/tanitad_scratch/p5r_agent/run/   # controls.json, lambda_star.json, eval_r{0,1}.jsonl (line counts), verdict.json
powershell: Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -like '*p5r_run*' }
```
The terminal markers are:

| marker | meaning | next step |
|---|---|---|
| `ZZP5R_CHAIN_DONE` | the run finished | write RESULT_P5R.md |
| `ZZP5R_GATE_FAIL controls [...]` | no verdict may be read | fix under a dated `PREREG_P5R_ADDENDUM_1.md` written **before** rerunning; delete `controls.json` to rerun |
| `ZZP5R_STOP budget` | over the CPU budget | addendum |
| a Python traceback | the chain is resumable | — |

**Resuming:** `tune_r*.jsonl` and `eval_r*.jsonl` skip done rows, and stages skip when their output exists. Relaunch
with the same WMI command:
`cmd /c ""C:\Users\Admin\navsim-crun\venv\Scripts\python.exe" "E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5r\p5r_run.py" all --work F:/tanitad_scratch/p5r_agent/run > F:\tanitad_scratch\p5r_agent\run\chain.log 2>&1 < NUL"`
with `CreateFlags=0x08000200` and `CurrentDirectory F:\tanitad_scratch\p5r_agent`.

## Exact next step: write `RESULT_P5R.md` from `raw/verdict.json`, `raw/lambda_star.json` and `raw/controls.json`
1. **Controls (card §8):** every gate in `controls.json.gates` (13 booleans) must be true. Quote `S3_max`, the
   C-R3/C-R4 `B4` CIs, and `median_J_lam0` vs `median_J_track_tc20s_warm`.
2. **λ\* (card §4):** `lambda_star.json` (λ\*, status, the TUNING table per λ and rank).
   - Also `tc20s_on_tuning_descriptive`: the Tc20s projection check against the MM's ESTIMATED ≈ 2.09.
3. **Verdict per rank (card §6):**
   - `per_rank.k.verdict`, `failing_bars`;
   - B1 `TB6_p5r.R_acc/R_hr` (+ `ci_*`, descriptive);
   - B1p `R_*_proj` (**ESTIMATED**);
   - B2 / B3 / B4 CIs with the estimator string;
   - sanity S1–S5.
   - **P5-R PASSES iff rank 0 PASSES.** Rank 1 feeds P5-C §4.3.
4. **Comparators (§9):** raw / Tc20s / P5-R sim8, E_maxxy and TB-6. Tsel-s is reported as SECONDARY and decides
   nothing.
5. **Families:**
   - LONGITUDINAL and LATERAL from `families`;
   - TACTICAL not computable (n = 0, the reason given);
   - STRATEGIC PAUSED.
   - Descriptive: interp8 (Tc20s paid 13 % in P5-S; check whether P5-R pays less) and comfort.
6. **Cost (§2.3):** fill the formula from `budget.json`: a, b, t_bld, t_scr, K̄, and `bank_scale`.
   - The smoke's preliminary bank-scale figure is ~77 CPU-h (P5-R only), ~0.75 h wall on 128 cores. It is ESTIMATED.
   - t_bld dominates it. The pod build already pays t_bld for `iterate20`, so P5-R's marginal cost is about
     `t_row1 − t_bld`.
7. **Consequences (§10), as committed:**
   - **PASS** ⇒ the PREREG_P5C amendment list in §10 applies, but only if P5-C's trigger fires (the pod's full-table
     TB-6 rank-0 FAIL on Tc20s).
     - Needs an MM decision: confirm the PI ruling of 2026-11-11 covers P5-R.
     - Needs an MM decision: TB-0's tolerance. LAPACK in the LM means pod/dev-box bit identity is not guaranteed.
   - **FAIL** ⇒ draft the P5-T (selective label) §9 entry **in the same turn** (Rule Zero), with a learnability guard
     on the touched rows.
8. Evidence class and tier stamp, exactly as in card §7. Never "navtest", never "PDMS".

## Cleanup (PI rule, 2026-11-07), once RESULT_P5R.md and `raw/` are verified in the hold
- `F:/tanitad_scratch/p5r_agent/run/` (samples.json 10.7 MB, bank_learn.npz 2.4 MB, jsonl, smoke files, logs) can be
  re-created from the bank and code with `p5r_run.py prep`. Delete it **after** checking that `raw/` holds
  verdict/lambda_star/controls/budget/prep/eval/tune.
- `F:/tanitad_scratch/p5r_agent/vsmoke/` (~2.5 MB; a verdict-path smoke on TUNING smoke rows). Its deletion was
  REFUSED by the permission system for this agent. Leave it for an MM-authorised sweep; it is temporary and owned by
  this agent.

## Do not touch
- The MM's N1 chain, `C:/Users/Admin/n1_run`, the GPU lock, `p5c/gates/`.
- The bank `F:/tanitad_scratch/p5c_tb6/bank` (read-only).
