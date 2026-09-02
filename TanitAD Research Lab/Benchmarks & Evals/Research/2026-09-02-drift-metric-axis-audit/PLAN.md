# PLAN — E-BE-DRIFT-1

`2026-09-02 · Benchmarks & Evals · compute budget: 0 GPU, seconds of CPU`

## Priority order (a killed run still yields value at every step)

| # | step | yields on its own | state |
|---|---|---|---|
| 1 | Locate the drift metric and read its **source** | F1 — FS-1's premise is wrong about our instrument; this alone prevents a mis-targeted port | ✅ done |
| 2 | Name the axis where FS-1's concern does apply | F2 — the blind spot is on `k`, not on a sequence | ✅ done |
| 3 | Tabulate drift vs `k` over every banked read, with controls asserted first | F3 — flat across 15×, six pairs, all controls clean | ✅ done |
| 4 | Check whether "drift" is used for other quantities | F4 — three distinct meanings | ✅ done |
| 5 | Write FS-1's cheap replacement with both outcomes pre-committed | the deliverable | ✅ done |

⭐ Step 1 was deliberately first: it is the cheapest step and the one that could
cancel every later step. Had drift been a start-vs-end measure, steps 2–5 would
have been replaced by "proceed with the port as written."

## Compute

* **0 GPU.** Re-analysis of 16 banked JSONs; no checkpoint loaded, no forward pass.
* ⛔ Thor and the A40 pod (REF-C v3 H-arm, ~35 h remaining): **not touched**.

## Owners / integration

* **Owner:** Research Lab (daily pass 2026-09-02).
* ⛔ **The Lab does not re-rank a backlog row.** FS-1's re-scope is escalated to the
  Master Mind in `COMMS.md`, together with the P5/L3-relevant result and the
  vocabulary finding for the EvalFlyWheel.

## Reproduce

```bash
python code/drift_axis_audit.py --out raw/drift_axis.json
```

The script **excludes** any read whose `constant (control)` column is not exactly
0.0 and reports the exclusion, so a broken source read cannot enter the table
silently. Source directories are constants in the script and echoed into the
artifact.

## The follow-on this package specifies (for whoever runs it)

```
k-sweep: k in {1, 2, 4, 8, 15, 30, 45, 60}
arms   : postrain30k, k8clip05p30k, k60clip05p30k
bands  : PCA [0,8] and [8,16]
report : drift r, t, AND n_rows per k  (n falls with k: 7,680 -> 3,200 between the
         banked endpoints, and a curve that hides a shrinking sample is not a curve)
read   : monotone/flat  => the single-k summary stands
         dip or peak    => every single-k drift number we hold is INCOMPLETE
```

Runs on the dev-box 4060 with the existing probe; no new instrument is needed.
