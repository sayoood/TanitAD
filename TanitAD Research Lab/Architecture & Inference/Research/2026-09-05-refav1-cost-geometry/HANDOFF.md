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

* ⛔ **RETRACTED, in this same document, before it could mislead anyone:** I
  first recorded `test_refav1_kin_contract.py::test_A6` as a PRE-EXISTING red
  test belonging to the kin-contract stream. It **does not reproduce** — 26
  passed alone, 70 passed in the identical five-file selection that failed, and
  **647 passed / 1 skipped / 0 failed** over the whole blast radius. I had
  observed it **while my own patch was half-applied**. ⇒ **a red test seen
  during your own multi-step edit is evidence about your edit, not about the
  test, and blaming a sibling stream is the cheapest wrong explanation
  available.** Nothing is owed to the kin-contract stream.
* **`robocopy` from the G: mount stalled past 120 s** on two files. The working
  route is to run the same anchored patcher against **both** trees — it is
  deterministic and idempotent, so the trees converge byte-for-byte (verified:
  172,369 and 162,089 chars in each).
* The two sibling arms that landed at 21:05 / 21:11 UTC (`kamm07`, `l3ladder`)
  were **harvested here** against the same baseline rather than left for later —
  their own `HANDOFF.md` asked for exactly that. Both show the same trade and
  neither closes the gap (`raw/pd_lonbase.md`).

---

## 7. STATE AS OF 2026-09-05T19:45Z — four levers answered, two arms left

| lever | arm | verdict |
|---|---|---|
| L1 `W_KAPPA` | `wk15` / `wk151` / `cos_wk` | **ANSWERED — the accuracy lever, clean interior optimum at 15.11245.** ADE 1.3272 -> 0.8934, best lateral family of any arm; costs the longitudinal family (separated, 14-20x floor); 100 % rung and the SHIPPED weights both collapse to the do-nothing plan |
| L2 `ccosh` hold branch | `ccosh_w000` | **ANSWERED — a clean NULL.** Repairs the cost on 25 % of windows, changes not one plan (`+0.0000 [0,0]` on all ten family metrics; control `wk15` reads 1.508e+01 m) |
| L3 seed ladder | `l3ladder` | **ANSWERED — a NULL that settles the question.** Not one window of 40 realises a rung; `wk15` reaches the same band with no ladder ⇒ **the binding constraint is the COST, not the candidate set** |
| L4 Kamm cap | `kamm07` | **ANSWERED — the safety lever, and it is FREE.** `peak_g` max 3.262 -> 0.707, violation rate halved, `turn_left` recall and GT-turn ADE bit-identical, both LON deltas BELOW their noise floors |
| L5 retrain goal head | — | **not taken, and the evidence says do not**: a perfect goal is significantly WORSE (`cl - cl_oracleseed` ADE -1.2181 separated, 20x the floor) |

**Still running** (started 19:11:10Z and 19:13:04Z, ~5 episodes each remaining):

* **`combined`** = `ccos` + ladder + `--kamm-mu 0.7` (`raw/queueG.sh`). ⚠️ Given L3's
  null its expected value is now *the Kamm arm*, because a CONSTRAINT cannot make a
  rung attractive. Run it anyway — the prediction is registered and a surprise here
  would be worth more than the confirmation.
* **`wk15_ladder`** = `ccos` + `W_KAPPA 15.11245` + the same five rungs
  (`raw/queueH.sh`) — **the arm L3's null actually asks for**, and one variable
  against `wk15`. If the rungs are chosen here and not in `l3ladder`, the "cost, not
  candidate set" reading is confirmed from a third direction.

**The next experiments after those, in order:**

1. **A seed replicate of `wk15`** (`--plan-seed 1`, nothing else changed). Rule 2
   above makes this the cheapest thing that converts the package's best arm into a
   quotable one; the tactical family needs it most (25.6 % seed drift).
2. **`wk15` + `--kamm-mu 0.7`** — the two levers that each work, and they are
   complementary rather than overlapping (a penalty and a constraint; their
   straight-window gains differ, +0.1957 vs +0.4304 against `ha0_ext`).
3. **The LONGITUDINAL analogue, which does not exist yet.** Every lever in this
   package is lateral. The blocker is that 29/40 windows decode
   `ADAPT_SPEED_FOR_CURVE`, whose canonical control is `a == 0`, so the plan holds a
   constant acceleration while `ha0_ext` holds the measured `a0`. Neither `W_KAPPA`
   nor the cap touches it, and `W_JERK` is not it either (median realised
   `mean(jerk^2)` is already **0.0**).
4. **`wk15` on the full 282-window v7.2 EVAL grid**, so the result stops being
   conditional on a turn-DENSE panel.

## 8. TWO PROCESS LESSONS FROM THE END OF THIS TURN

* **`mktree_commit.py`'s compare-and-swap refused two commits** because a sibling
  agent moved HEAD while the tree was being built (~40 s for 225 paths on this
  mount). **The fix is to commit only the CHANGED paths** — 29 of 225 here, which
  won the race on the first attempt. Never `force`.
* **A conditional import is not a module import, and an argparse help string is
  CODE.** Both defects I introduced in `refav1_arm.py` are recorded in
  `RESULT.md` §7.5.1 with their root-cause class.
