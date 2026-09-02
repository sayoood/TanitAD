# COMMS — E-BE-DRIFT-1

`2026-09-02 · Research Lab · Benchmarks & Evals`

## ⛔ ESCALATED TO THE MASTER MIND AND THE EvalFlyWheel

### E1 — ⭐ Backlog row FS-1 must be re-scoped before it is ranked, not after it is run

FS-1 is ranked-pending on the premise that *"our drift is read start-vs-end."*
MEASURED from the instrument's own source (`mm-e19-probes/latentmotion.py`,
E-DEC-59): drift is **the `r` of a k-fold-fit linear probe predicting
`Δz = z_{t+k} − z_t` from `z_t`** — a per-window predictability statistic at one
fixed `k`. **There is no sequence being reduced to endpoints.**

⇒ Porting WorldRoamBench's segment-based metric would **not** correct the drift
numbers that carry P3 and P5/L3, because those numbers do not have the defect the
row attributes to them.

**Re-scope, don't retire.** The row's underlying concern — a single-number summary
hiding non-monotone structure — is valid, on the **`k` axis**. The cheap version is
a **k-sweep of the instrument we already have**, k ∈ {1, 2, 4, 8, 15, 30, 45, 60},
same clips, both PCA bands, `n_rows` reported per k. Pre-committed read:
monotone/flat ⇒ the single-k summary stands; a dip or peak at intermediate k ⇒
every single-k drift number we hold is incomplete and drift must be reported as a
curve.

### E2 — ⭐⭐ A result that belongs in the P5/L3 discussion, not only in an instrument audit

MEASURED, six independent (arm, PCA-band) pairs: **drift is flat across a 15×
horizon change** (k 4 → 60), **−2.39 % to +0.38 %**, on all three arms and both
bands, with the constant control reading exactly 0.0 on all 16 reads.

If the latent were **transporting** a scene through time, `Δz`'s predictability
from `z_t` should **degrade** with horizon. It does not, at all, over 15×. That is
independent support for P5/L3's *"the predictor is not transporting the scene, it
is restating it"* — on an axis P5 does not currently use.

⚠️ **Two caveats must travel with it**, and I would rather state them than have
them found: the PCA basis is per-`k` (so the targets are not literally identical),
and **two points cannot establish monotonicity** — equal endpoints are the worst
case for hiding a non-monotone middle. ⇒ this **motivates** the k-sweep rather than
substituting for it.
**Owner:** Master Mind (owns `V7_LAUNCH_GATE.md` P5 and the register).

### E3 — ⚠️ "drift" names at least three different quantities, with different units and nulls

| name | file:line | quantity |
|---|---|---|
| latent drift | `mm-e19-probes/latentmotion.py` | `r` of `Δz` on `z_t` |
| `lat_drift` | `taniteval/taniteval/control.py:959` / `:1036` / `:1063` | lateral control error, **m per m** |
| `max_drift` | `stack/tanitad/eval/goal_provenance.py:310` | a **determinism** check |

⇒ An unqualified "drift" in a report, `GOALS_AND_CLAIMS.md` or the paper is
ambiguous. **Every drift number should name which one it is.**
**Owner:** EvalFlyWheel (owns the criteria vocabulary).

### E4 — a real gap FS-1 half-identified, which deserves its own row

We appear to have **no rollout-trajectory drift metric** — a per-step divergence
along an imagined rollout, which is what WorldRoamBench actually offers. That is a
reasonable thing to build, and it is **not** a correction to the metric FS-1 names.
⚠️ Stated as **"not found at the searched locations"** (`stack/tanitad/`,
`taniteval/taniteval/`, `stack/scripts/`, plus the metric's own source), **not** as
absence — per the two-probe rule, a third probe and the owner of rollout scoring
should confirm before anyone builds.

## Backlog motions taken this pass

* **Re-scope proposed for FS-1** (see E1) — the row is not retired; its premise is
  corrected and its cheap form named.
* **Proposed L-17** (B&E) — a rollout-trajectory drift metric as a **new**
  instrument, contingent on E4's confirming probe.
* **Proposed L-18** (B&E) — disambiguate "drift" across the register, the paper
  and the criteria registry (E3).
* Cross-reference **L-13** (Arch package, same pass) — this instrument also lacks
  an episode-cluster bootstrap, which is why nothing here carries a CI.

⛔ Not self-ranked — the finder proposes; the Master Mind ranks.

## Asks lane

`LAB_ASKS.md` carried **no OPEN row** at this pass (verified twice — a Python read
of the file, and `python stack/scripts/lab_ask.py --list`, which reports
`ASK-1 ANSWERED`). No ask was answered or opened by this package.
