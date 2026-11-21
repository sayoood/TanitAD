# E-BE-DRIFT-1 — audit the drift metric before porting a replacement for it

`Research Lab · Benchmarks & Evals · daily pass 2026-09-02 · 0 GPU`

## Why this exists

Backlog row **FS-1** proposes work on the strength of a claim about one of our own
instruments:

> ⭐⭐ **Port WorldRoamBench's SEGMENT-BASED drift metric into `taniteval` and
> re-score the banked v7 rollout dumps.** Our drift is read **start-vs-end**, which
> by their construction **cannot see non-monotonic mid-sequence collapse**.
> **Pre-committed read: monotone drift ⇒ the current instrument stands;
> non-monotone collapse ⇒ every drift number we hold is a LOWER BOUND and the
> instrument must change.** — **0 GPU**, the dumps exist.

Drift is load-bearing: it carries **P3** (the 3.3× seed-stable effect) and feeds
**P5/L3**. Before re-scoring anything against a foreign metric, the right first
step is to establish **what our metric measures, from its source** — because the
row's premise is a claim about our code, not about WorldRoamBench.

*(This is the same discipline the programme applies to numbers: a metric quoted by
its NAME rather than its definition is the `step_s` / `df` scope-error class.)*

## Hypotheses, both outcomes committed in advance

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| **H-DRIFT-1** | Our drift metric is a start-vs-end trajectory measure, as FS-1 states. | FS-1 is well-targeted; proceed to the port. | ⛔ It is something else ⇒ FS-1's premise is wrong and the port would not correct the number it claims to correct. |
| **H-DRIFT-2** | Whatever it is, it has FS-1's *class* of blind spot somewhere. | Name the axis and measure it on banked data. | It does not, and FS-1 should be retired. |
| **H-DRIFT-3** | Drift degrades with horizon `k` — a transporting latent should become harder to predict further ahead. | Expected behaviour; the single-`k` summary is at least informative about dynamics. | ⚠️ It does **not** degrade ⇒ the statistic may be reading a static property of the latent space, which bears directly on P5/L3. |

## Success criteria, committed in advance

1. ⛔ **The metric's definition is quoted from its own source**, not inferred from
   its name or from prose about it.
2. Every read used is checked against its **controls before it is scored**:
   the `constant (control)` column must read **exactly 0.0**, or that read is
   excluded and the exclusion is reported.
3. `n` is printed with every number. `k = 60` roughly halves the usable windows,
   and a comparison across `k` that hides that is comparing two sample sizes.
4. ⛔ **No interval is quoted unless its estimator is named.** This instrument has
   no episode-cluster bootstrap, so **no CI is offered rather than a wrong one** —
   and the resulting weakness is stated, not hidden.
5. An absence claim ("we have no X") is written as **"not found at the searched
   locations"** unless it survives the two-probe rule.

## Method

`code/drift_axis_audit.py` reads every banked `latentmotion_*.json` from the two
MM-E19 packages, asserts each read's constant control, and tabulates drift `r`
against `k` per (arm, PCA band).

⛔ **One trap disarmed explicitly:** the `k8-attribution` files store their arm
under the internal key `"k60clip05p30k"` — the `--arm-name` default that package
documents (§6a), resolved only by checkpoint md5. The audit remaps it via a named
constant; reading by the name in the file would silently swap two arms and invert
the result.

**Tier:** `T0-DIAGNOSTIC`, inherited from the source reads. ⛔ Nothing here is a
driving claim. The four-metric-families rule binds *capability* evals; this is an
instrument audit, and no family is being reported or omitted.
