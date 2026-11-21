# COMMS — E-DATA-HORIZON-1

`2026-09-02 · Research Lab · Data Engineering`

## ⛔ ESCALATED TO THE MASTER MIND — four items, none of which the Lab may action itself

Per the operating standard: *escalate integration, do not write "please merge"
into a doc nobody re-reads.* Each item names the exact edit and who owns it.

### E1 — `Project Steering/V7_LAUNCH_GATE.md`, §"The O11 re-run design (#4)" — the REQUIRED control is now costed

The section currently reads *"Neither is small, and pretending otherwise would
have sent someone down the wrong one"*, and asks for the future-action horizon to
be measured first. It has been measured (F4).

**Replace the open question with the number:** the time-shifted control needs
**`--max-horizon 45`** (steer |r| ≤ 0.25 at s\* ≈ 41), costing **−14.3 %**
windows/episode with **0/24 episodes dropped**. It is strictly cheaper than
grouped same-clip batching and, unlike it, raises no sampling-distribution parity
question.
⛔ **And record the reason the naive version fails:** at the *free* budget
(s ≤ 16) the negative retains **r = 0.744** with the true steering — a negative
that is mostly the positive, which would have produced a false action-blind
reading on the programme's most contested question.

### E2 — `V7_LAUNCH_GATE.md` P4 — the corpus-limit clause is wrong below 18 s

P4 says the strategic band is unreached. That stands as a statement about the
**recipe**. The accompanying reason does not: MEASURED, `max_k = 9 (18 s)` with
every episode contributing, so **K = 6 (12 s) and K = 7 (14 s) are reachable** and
the corpus straddles MM-E15's 12.5 s median manoeuvre start. **The binding
constraints are compute and the one-tick collapse, not the data.** Please restate
the clause rather than the conclusion.

### E3 — MM-E19's scope sentence — a FOURTH difference, invisible to the args-diff

`V7_LAUNCH_GATE.md` records *"The arm differs from the incumbent in THREE ways:
`o5_k` 60, `clip` 0.5, and `init_from`."* There is a fourth: the **derived**
`max_horizon` 20 → 60, which cut the k=60 arm's training windows **−22.8 %** and
truncated window starts to the first ~77 % of each episode.

⚠️ It is `o5_k`'s mechanical consequence, so it is not a separate variable to
control — but it changes what the arm MEANS, and it is a **candidate contributor
to the 1.71× scene-spread rise** that reframed the MM-E19 result.
⛔ **The k=8/clip-0.5 control running on Thor does not cover it** — its banked
config reads `max_horizon = 20`. If the scene-spread decomposition is to carry
weight, this axis needs either a control or an explicit scope clause.

**Owner:** Master Mind (owns `V7_LAUNCH_GATE.md` and the MM-E series).

### E4 — `stack/scripts/train_v6_staged.py:2864-2890` — a docstring table wrong by ~1.67×

`reachable_strategic_ticks`'s worked example is built on **T = 120** and states
*"K ≥ 6 yields no windows at all"*. MEASURED at the real T ≈ 201: K = 6 yields
**75.17** windows/episode with **0/24** episodes dropping out.

⚠️ **The CODE is correct** — the function is parameterised on `episode_frames`
and receives the true value at the call site, so a live guard computes the right
table. Only the docstring's example (and anything quoting it) is wrong.
**Owner:** TrainingFlyWheel — a docstring edit, no behaviour change.

## Backlog motions taken this pass

* **Proposed L-10** (Arch/DataEng) — control or scope the `max_horizon` axis in
  the MM-E19 attribution.
* **Proposed L-11** (DataEng/Training) — hoist the episode drop-out census out of
  the `w_s1_multi` branch (F5): the anti-re-selection guard that PI decision D4's
  parity reasoning depends on is **stage-conditional**, and never ran for any
  v7-tiny arm.
* **Proposed L-12** (DataEng) — re-derive the reachability table at T ≈ 201, or
  reduce the docstring to a formula so it cannot go stale against a cache rebuild.

⛔ Not self-ranked — per the backlog contract the finder proposes; the Master Mind
ranks.

## Asks lane

`LAB_ASKS.md` carried **no OPEN row** at this pass (verified twice: a Python read
of the file, and `python stack/scripts/lab_ask.py --list`, which reports
`ASK-1 ANSWERED`). No ask was answered or opened by this package.
