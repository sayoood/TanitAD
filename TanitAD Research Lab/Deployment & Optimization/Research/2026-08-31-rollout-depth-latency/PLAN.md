<title>PLAN — pricing sequential rollout depth</title>

# PLAN — E-LAB-DEPLOY-0831

1. **Anti-duplication preflight.** Confirm the deployment line has priced ONE predictor pass
   (`2026-07-18-predictor-cudagraph-and-numerics-sweep`: 6.08 -> 2.36 ms, 2.57x, same 4060) and
   quantisation (three packages), but **never sequential depth**. Use those as context; do not
   re-measure them.
2. **Commit the criterion BEFORE measuring** (SPEC §2): linear iff `p50(K)/K` is ~constant over
   K in [1, 120].
3. **Benchmark** two proxy predictors (7.1 M, 37.8 M) at K in {1,2,4,8,16,30,60,120}, batch 1,
   fp16, 60 reps after 15 warmup, p50/p95/min.
4. **Controls that must read known values:** an empty loop (`C0_nowork`) must NOT scale with K;
   a trivial add (`C0b_add1`) must scale, and must stay orders below the signal. Without both,
   a flat ms/step could be the harness rather than the model.
5. **Convert to the decision**: flat-K per band at dt=0.1 (20 / 60 / 300) against a 10 Hz budget.
6. Bank raw JSON with `meta` (GPU, torch, dtype, reps) and an explicit `what_this_is` scoping
   note, so the proxy can never be quoted as ours.

⚠️ Step 6's scoping note is why this package survived being finished by a different author: the
data described its own limits.
