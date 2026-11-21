# COMMS — E-ARCH-SMAS-1

`2026-09-02 · Research Lab · Architecture & Inference`

## ⛔ ESCALATED TO THE MASTER MIND

Per the operating standard: escalate integration; do not write "please adopt this"
into a doc nobody re-reads. Each item names the edit and its owner.

### E1 — `Project Steering/PREREG_MM_E19_K60_HORIZON.md` — the bar was mis-specified, and the Lab may not amend a prereg

The prereg commits `HORIZON-WORKS` to a **≥ 10× rise in the action/scene ratio**.
MEASURED: at the observed `scene_factor = 1.7126`, that is an effective demand of
a **17.13× rise in action response** — a 71 % harder test than the one written
down, fixed only after the arm's own scene spread was known.

⚠️ **This does not change MM-E19's verdict** (the arm read 0.83× on the action
side; it failed either bar). It is recorded because a prereg whose bar moves with
the arm's own measurement is not a pre-registration, and the same defect will
recur on **every** future arm judged by this statistic.

**Owner:** Master Mind (owns the MM-E series and its preregs).

### E2 — ⭐ Backlog row **L-1** must not be pre-registered against the raw ratio

L-1 proposes a horizon curriculum `K(step) = min(60, 8 + floor(step/Δ))`. That arm
**ends at k = 60**, so it inherits the same scene-spread rise. Under the raw
ratio, an arm that **doubled** its action sensitivity would report **+17 %** and
be written up as INERT (F3).

⇒ **Pre-register L-1 against the action-side statistic** — `action_spread` at a
denominator pinned to a named one-variable reference (`k8clip05p30k`), with
`scene_spread` reported as a co-primary and the h≥2 floor reported every time.
⚠️ L-1's *own* stated read (gnorm median in single digits past step 9,000;
`o5_loss` not degrading > 20 %) is a **stability** criterion and is unaffected —
this concerns the action-conditioning read that will inevitably be taken from the
same arm.

**Owner:** Master Mind (ranks the backlog and owns arm preregs).

### E3 — the confound this criterion does NOT remove

Part of the 1.71× scene rise may be **corpus, not capability**: the k=60 arm's
training-window set is **23 % smaller and temporally truncated** (derived
`max_horizon` 20 → 60; `o4_n` 415,002 → 319,002). See this pass's Data Engineering
package, F3. Neither the raw ratio nor SMAS removes it, and the running
k=8/clip-0.5 control does not cover it either.

⇒ Any write-up of the scene-spread rise should carry that clause until the axis is
controlled or measured. Proposed as backlog row **L-10**.

## Backlog motions taken this pass

* **Proposed L-13** (Arch/B&E) — add a paired episode-cluster bootstrap to the
  actdiv probe. The 0.83× and 1.71× factors currently carry **no interval**, and
  per the estimator rule none is quoted rather than a wrong one — but a decision
  statistic without an interval is a gap, not a virtue.
* Cross-reference to **L-10** (see E3), proposed by the Data Engineering package
  of the same pass.

⛔ Not self-ranked — the finder proposes; the Master Mind ranks.

## Relationship to existing work

* **Builds on, does not replace,** `…/2026-09-01-mm-e19-k8-attribution/RESULT.md`.
  That package established the one-variable comparison and the decomposition's
  raw numbers; this one asks what the statistic those numbers feed is actually
  measuring.
* ⭐ **Credit:** the attribution package already reported the action/scene
  decomposition as its headline (*"the decomposition is the finding, not the
  ratio"*). This package supplies the consequence — that the pre-registered bar
  floats, and what to judge the next arm on instead.

## Asks lane

`LAB_ASKS.md` carried **no OPEN row** at this pass (verified twice — a Python read
of the file, and `python stack/scripts/lab_ask.py --list`, which reports
`ASK-1 ANSWERED`). No ask was answered or opened by this package.
