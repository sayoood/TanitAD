# PRE-REGISTRATION — H4: is the REF-C trunk's scene content data-limited?

**Written 2026-09-16 before any H4 number is read.** Dev box, RTX 4060, no downloads, no pod.
Companion to `Project Steering/REFCV6_CLARIFICATION.md` §6.4.

## Hypothesis

**H4.** The scene content of a REF-C trunk **accrues with training exposure**. If it does, a frozen-trunk BEV
occupancy probe reads higher at step 40,284 than at step 9,500 of the *same run*.

## The one variable

`ckpt_step` ∈ {9,500 · 40,284} of **refcv4b** (`C:/Users/Admin/navcomp/ckpt/ckpt_step9500.pt`,
`C:/Users/Admin/refcv4b_final/ckpt_40284.pt`). ⛔ Identity is asserted before use: both checkpoints must carry the
same `argv` arm name and the same 396 `core.encoder.*` keys, loaded **strict**; the step is read from the
checkpoint's own `step` field, never from a filename.

**Held constant:** the extractor (`p4a_extract_tokens.py`), the probe head and its 6,000-step schedule, the 139 B1
eval clips and their stacked rows, the content-blind sha12 split, the seed, fp32 inference.

## Splits

P4's registered split: test `sha12 %4 == 0`, val `%8 == 1` carved from the fit clips only. No hyper-parameter is
selected on test.

## Controls, each with the value it must read

| control | must read |
|---|---|
| `const` | the test prevalence **exactly** (0.2762 on this grid) |
| `prior` | the per-cell train marginal |
| `shuffled` (targets deranged across clips) | ≤ `prior` + 0.02 |
| `pixel` floor (same head on `pix64`) | — (the arm must beat it to mean anything) |
| seed replicate (probe seed 1 at 40,284) | the floor every step-effect is read against |

## Committed outcomes

- **SUPPORTS H4** — AP(40,284) − AP(9,500) > 0, the clip-cluster bootstrap CI excludes 0, **and** the gap is
  ≥ 3× the probe-seed floor.
- **REFUTES H4** — the CI covers 0, or the gap is under 3× the floor. Reading: exposure is not the lever within
  this run's range, which points at hypothesis H2 (planning-only supervision does not ground a trunk) instead.
- **VOID** — any control misses its value above, or the two checkpoints differ in more than the step.

## Scope declared in advance

⚠️ Two checkpoints of **one** run are `n = 2` points on one trajectory: this measures *exposure within refcv4b*,
not "data scale" in general, and it cannot separate exposure from optimisation progress. ⚠️ refcv4b, not
refcv5-v2, because only refcv4b has a mid-training checkpoint on this box. ⚠️ A linear/attention probe bounds
what a *probe* recovers, not what the trunk contains.

**Budget:** ≤ 1.5 GPU-h, sequential (a second GPU job beside it is MEASURED to crawl).
