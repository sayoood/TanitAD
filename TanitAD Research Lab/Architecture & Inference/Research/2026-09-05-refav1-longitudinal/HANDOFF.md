# HANDOFF — the refav1 LONGITUDINAL arms, and how to land them

*Written 2026-09-05 while the arms were still queued behind the sibling
stream's dev-box lane, so a fresh context can finish this without re-deriving
anything. Read `RESULT.md` first: every arm's BOTH outcomes are already
committed in §5, before the numbers exist.*

## 1. One command

```
bash <scratchpad>/finalize_lon.sh
```

It is idempotent, zero GPU, and it **NAMES the arms whose record is absent**
rather than silently thinning the panel. It writes, into the scratchpad and
then into `raw/` here:

* `lon_arms_present.txt` — which arms landed and which did not, with a pointer
  to each missing arm's log
* `lon_emitted_all.txt` — the emitted longitudinal control per arm (this is the
  diagnostic that says whether the lever moved the plan **at all**)
* `pd_lon.md` / `pd_lon.json` — the four-family **paired** deltas of every new
  arm against the same named baseline `wk15`, plus both floors, plus the
  arm-against-itself known-value control that must read `+0.0000 [0, 0]`
* `lon_oracle_all.txt` — the vocabulary-expressivity table re-read on each arm's
  own dump
* it also copies each arm's `rec_*.json` **and its whole dump** under
  `raw/arms/`, because until it runs they are single-copy off-repo

## 2. What is queued, and the gate

`raw/queueLON2.sh`, running from `<scratchpad>/queueLON2.sh`. Priority order, so
a killed queue still yields value:

| # | arm | the ONE variable against `wk15` |
|---|---|---|
| 1 | `lonshift` | `--a-sustain-mode a0_shift` — **D2**, the best measured design |
| 2 | `lonseam` | `--jerk-seam a0` — the cost lever alone |
| 3 | `lonshift_s1` | `lonshift` with `--plan-seed 1` — **the replicate; mandatory** |
| 4 | `loncomb2` | D2 + the jerk seam |
| 5 | `lonvocab` | `--a-sustain-mode a0` — D1, kept for ATTRIBUTION |

The baseline `wk15` is **already banked** (`ccos`,
`(0.0, 15.11245, 64.29715042415070)`, `--plan-seed 0`, shipped vocabulary), so
no baseline re-run is needed and every comparison is window-for-window.

⛔ **THE GATE COUNTS DISTINCT `--out` TARGETS, NEVER PROCESSES.** M28 (3): one
arm is a parent *and* its child, both carrying the full command line, so a gate
on "processes ≤ 1" can never open and the lanes silently serialise. Two
concurrent arms is the proven ceiling on the 8 GB 4060 (~2.5 GB each).

⛔ **AND A PROCESS SEARCH SELF-MATCHES.** My first attempt to kill the previous
queue used a pattern that appeared in my own command line, so it matched its own
wrapper shells and killed my shell instead (exit 255). Build the pattern from
disjoint pieces (`'*queue' + 'LON2*'`) and verify zero with a same-breath control
that must read non-zero.

## 3. Reading the result — the bar, stated before the numbers

1. ⛔ **A separated CI is NECESSARY AND NOT SUFFICIENT.** `D-REFAV1-CG-SEEDFLOOR`
   measured `separated` on **4 of 10** paired family metrics between two arms
   differing **only** in `--plan-seed`. The admissible form is *"the lever's
   paired delta exceeds the seed pair's delta on the same metric"*. On the
   longitudinal metric that floor is **`wk151 − wk15` = +0.0076 [−0.0134,
   +0.0282]**, against a gap of **+0.4862** — 64× the headroom needed, so the
   binding question is SIZE, not detectability. `lonshift_s1` exists so
   `lonshift`'s own noise floor is read on `lonshift`'s own rig.
2. **Four families or it is incomplete.** ADE alone would read `M27`'s paradox
   as progress: seven separated family improvements produced by the planner
   *stopping*.
3. **"Beats the floor" must be conjoined with "acts".** Report the emitted
   `mean|a|` and the `a[0] == a_goal[0]` fraction beside every ADE — that is what
   `lon_emitted_all.txt` is for. An arm that ties `ha0_ext` by emitting nothing
   has not driven.
4. **Every arm carries its cost triple, its vocabulary mode and its seed**, or
   it is not quotable. ⚠️ The third entry of the triple is a **DEAD TERM**
   (`target_speed` is never passed; `test_C1` pins the call site), so quote it
   but never attribute anything to it.
5. ⛔ **The expressivity tables in `RESULT.md` §1.5 / §1.9 are NOT planner
   results.** They roll a control profile through the unicycle integrator.
   Whether the search emits it is exactly what these arms measure, and mixing
   the two is retraction #29's error.

## 4. If the arms are not enough — the next levers, in order

1. **The 9 non-maintain windows.** They carry a 2.1× larger longitudinal error
   and **all** of the ADE loss (`raw/lon_attribution.txt`). D2 addresses them;
   if it does not move them at the arm level, the question becomes why the
   search will not follow a goal it copies elsewhere.
2. **`GOAL_REACH_S`.** Every token realises only `0.6513 ×` its named `dv` inside
   the 2 s plan window, because the profile decays with a 2 s time constant. A
   shorter reach time makes a token deliver its own semantics inside the
   optimised window. Not measured; one variable; cheap.
3. **Arm `W_VEND`** — ⛔ **PI DECISION, not an agent's.** `test_C1` pins the call
   site and `test_C2`'s docstring reserves it verbatim. The measurement that
   motivates it is `RESULT.md` §1.2–§1.4; the admissible target is
   `max(0, v0 + a0·plan_horizon_s)` from the measured t0 state.
4. **A LON level set + a trained chooser.** Only after 1–3: the current design
   deliberately has **no free parameter**, and adding one re-opens the
   oracle-vs-realised problem that M19/#29 already cost the programme once.

## 5. Two things that are NOT this agent's to decide

* **`a_shift` / `a_sustain` / `jerk_seam_a0` becoming refav1's DEFAULTS.** All
  three are implemented, pinned, and OFF by default; every default is
  bit-identical to the pre-2026-09-05 planner. These arms MEASURE them; they do
  not authorise them.
* **Arming `W_VEND`** — see §4.3.

## 6. Housekeeping this turn exposed

* `stack/tests/test_refav1_kin_contract.py::test_A6` is **RED and it is not this
  stream's**: it asserts `manifest["tiers"]` as an exact dict that predates
  `ha0_ext`'s registration (`refav1_arm.py:189, :672`). A stale exact-dict pin,
  escalated to the kin-contract stream rather than edited mid-flight.
* **`robocopy` from the G: mount stalled past 120 s** on two files. The working
  route is to run the same anchored patcher against **both** trees — it is
  deterministic and idempotent, so the trees converge byte-for-byte (verified:
  172,369 and 162,089 chars in each).
* The two sibling arms that landed at 21:05 / 21:11 UTC (`kamm07`, `l3ladder`)
  were **harvested here** against the same baseline rather than left for later —
  their own `HANDOFF.md` asked for exactly that. Both show the same trade and
  neither closes the gap (`raw/pd_lonbase.md`).
