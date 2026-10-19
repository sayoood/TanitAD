# Amendment W-R8-2 — fp64 re-check of a wrapper-clause miss (PI decision, 2026-10-17 ~21:40 Berlin)

**What failed.** The step-20,000 fixed wrapper clause (W-R8-1 probe, GPU) read FAIL. Only one term was over the bar:
`lat_tac`, max wrapper rel **1.0647e-5** against a bar of max(3*floor, 1e-5) = 1e-5. W1 and W2 were still detected, and the unfixed
control reproduced the G0 terms exactly (`wrapper_clause_r8fix_gpu.json`). At 5,000 / 10,500 / 15,000 the same clause read
3.0e-6 / 3.3e-6 / 2.2e-6 (PASS).

**Diagnosis (MEASURED on CPU, same 4 rows, same splits, the unmodified probe; `wrapper_fp64/RESULT.md`).**

| arm | `lat_tac` wrapper rel | absolute difference |
|---|---|---|
| F32 | 1.0647e-5 | 2.98e-8 (reproduces the GPU) |
| F64 | 1.98e-14 | 5.55e-17 |

* Both differences are exactly 128 ulps of the value in their own precision, and the ratio 5.4e8 is about 2^29.
* In F64, 0 of 369 terms exceed 1e-9, W1 and W2 are still detected, and the rows-alone runs show no row-mixing op.
* `lat_tac` is a CE of 0.0028 on logits up to about 61, so a 1-2 ulp logit wobble reads as about 1e-5 relative.
* ⇒ The miss is fp32 reduction order, not a coupling.

**The amendment (PI: "fp64 re-check rule").**
1. The fp32 clause and its bar are UNCHANGED: max(3*floor, 1e-5) per term; W1 and W2 must be detected.
2. A term over the bar is **re-judged in float64** on the same rows, the same micro-batch splits and the same neutralised draw
   sites (`wrapper_fp64/wrapper_fp64_probe.py`). It passes only if its F64 wrapper rel is **≤ 1e-9**, and only if W1/W2 stay
   detected in F64.
3. A term still above 1e-9 in F64 is a **real coupling**: the clause FAILS, and the coupling is localised (rows alone, then by
   module).
4. Every re-check reports the modules that could not run in F64. If a term's own path did not run in F64 end to end, the re-check
   is INCONCLUSIVE, which counts as FAIL.

**Application.** Step 20,000: G0 as coded FAIL (A7). Its only gate reasons are wrapper-clause terms. The W-R8-1 clause leaves one
term over the bar, `lat_tac`, which passes the F64 re-check (1.98e-14). M1 is detected. ⇒ **G0 PASS under A7 + W-R8-1 +
W-R8-2.**

**What this does NOT change.** The step-5,000 detection-term miss (R3_PARTIAL) stands, so step 5,000 stays PROVISIONAL. Every
battery number remains T1 open loop, single training seed.
