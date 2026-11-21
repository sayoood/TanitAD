# â›” RUN FAILED â€” no arm was scored

Run: `D:/Projects/TanitAD/taniteval/results/bench/navsim_v2/navhard_two_stage/20260920T201038Z-navsim_v2-refcv4b_b1_v72_40k-d3c2b2`

A `summary.json` EXISTS, and it is not a result: the suite writes one for a failed run too,
with every arm `status: FAILED` and every headline `UNAVAILABLE`. â›” Nothing in this run is
quotable. The per-arm failure records are the evidence:

```
CV    RAM_GUARD_ABORT    rc=3 wall=4277.1s retryable=True
       summary line 'Number of successful scenarios' ABSENT — cannot confirm work
       failed scenarios = None
       no result CSV found
STOP  RAM_GUARD_ABORT    rc=3 wall=0.5s retryable=True
       summary line 'Number of successful scenarios' ABSENT — cannot confirm work
       failed scenarios = None
       no result CSV found
ECHO  RAM_GUARD_ABORT    rc=3 wall=0.3s retryable=True
       summary line 'Number of successful scenarios' ABSENT — cannot confirm work
       failed scenarios = None
       no result CSV found
A1    RAM_GUARD_ABORT    rc=3 wall=0.3s retryable=True
       summary line 'Number of successful scenarios' ABSENT — cannot confirm work
       failed scenarios = None
       no result CSV found
```

## What survived, and what must NOT be redone

* model seam **A1**: 5462 model rows, 450 CV stand-ins, 11232.6 s — INTACT, adopt it with `--reuse-seams`
