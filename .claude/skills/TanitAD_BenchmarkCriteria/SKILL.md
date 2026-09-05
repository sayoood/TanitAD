---
name: TanitAD_BenchmarkCriteria
description: Keep TanitAD eval criteria COMPLETE and CONSTANT. Diffs an eval artifact (or the whole results corpus) against the binding criteria registry — the four metric families, the leak guards, the tier stamp, the estimator rules — and reports every missing criterion as a WORK ITEM rather than letting it pass as a silent omission. Use when running or reviewing any eval, adding a metric, adopting a community benchmark (NavSim, nuScenes), or auditing whether a result is admissible.
---

# /TanitAD_BenchmarkCriteria

**Owner:** TanitAD_EvalFlyWheel · **Products:** P7 TanitEval
**Registry:** `products/P7-TanitEval/CRITERIA_REGISTRY.json`
**Tool:** `tools/criteria_check.py` · **Tests:** `tools/tests/test_criteria_check.py`

## Why this skill exists

The charter states it plainly (TANITAD_PROGRAMME.md §2, EvalFlyWheel §5):

> *Criteria completeness is enforced by machinery, not memory — we measurably
> failed to keep them complete and constant by hand.*

Three ADE-only reports went out **after** the four-families rule was made binding
(PI, 2026-08-02). A rule in prose decays; C126 measured that decay directly, where
a correction living only in a report was re-counted by every later census because
**no prose correction can reach a glob**. So the rule ships as a tool plus a test.

## The three states — the whole point

| state | meaning | verdict |
|---|---|---|
| **PRESENT** | the metric is in the artifact | fine |
| **REFUSED** | the artifact's `refused` block names the criterion **and gives a reason** | **admissible** — and it becomes a visible WORK ITEM |
| **ABSENT** | neither | ⛔ **VIOLATION** — the silent omission the doctrine forbids |

An eval that cannot compute headway because no lead-agent state exists is doing
the *right thing* when it says so, with its `n` and its reason. An eval that
simply never mentions the tactical family is hiding a gap — and at a glance it
looks identical to an eval that has no gap. **This instrument makes those two
look different.** A `refused` entry with an empty reason is a shrug, not an
acknowledgement, and is treated as ABSENT.

## Procedure

### 1. Run the check
```
python tools/criteria_check.py <artifact.json>                    # one artifact
python tools/criteria_check.py --all taniteval/results/           # corpus census
python tools/criteria_check.py <artifact.json> --strict           # exit 1 on ABSENT
python tools/criteria_check.py --all taniteval/results/ --json report.json
```
Wire `--strict` into any eval pipeline that must not ship an incomplete result.

### 2. Read the scope line FIRST
The census opens with `N artifacts read -> X in scope, Y out of scope, Z UNKNOWN`.
Scoring a *profiling* artifact against *driving* criteria manufactures a violation
count that is pure noise — the same failure class as `df` on a pod, `free` on Thor,
and `memory.usage_in_bytes` on a cgroup: **a probe that reports the wrong scope is
worse than no probe, because it looks like an answer.** Anything `UNKNOWN` needs a
human decision; it is never counted as compliant.

### 3. Triage what comes back
- **VIOLATION (ABSENT)** → either implement the metric, or refuse it *explicitly
  with a reason and its `n`*. Those are the only two admissible outcomes. Adding a
  refusal is not a way to make the number go away — it converts a hidden gap into
  a tracked one.
- **WORK ITEM (REFUSED / PARTIAL)** → carry it into the report and the backlog.
  A missing family is a work item, never an excuse.
- **UNKNOWN scope** → classify the artifact, then extend
  `applicability` in the registry so the next run knows.

### 4. Report per-family, never pooled
⛔ Never present a horizon sweep of ADE as "the result" — it is one row of four
families. Each family carries its own estimator (paired episode-cluster bootstrap,
`taniteval/ci.py`) and its CI, on the same windows as the ADE it accompanies.
⛔ `overlapping_holdout_se` is forbidden as a decision-grade interval; it may appear
only under an explicit deprecation label, which the checker verifies by context.

### 5. Extending the registry
Add a criterion when a new metric becomes binding, a benchmark is adopted, or a
retraction (`C<N>`) teaches that something must always be reported.
- Give it an `id`, a `label`, its `keys` (dotted JSON paths, alternatives allowed),
  and `refused_as` names if the harness can legitimately decline it.
- ⭐ **Ship the registry change with its test in the same commit** (programme rule
  §6.5). `tools/tests/test_criteria_check.py` carries a **deliberate-regression arm**
  per family: it deletes a family from a compliant fixture and asserts the checker
  catches it. **A guard that has never been shown to fail proves nothing** — if you
  add a family, add its regression arm.

### 6. Community benchmarks
`benchmarks.navsim` / `benchmarks.nuscenes` are skeletons marked `pending: true` and
are **not enforced** until filled from the banked protocol specs
(`products/P7-TanitEval/benchmarks/`). When adopting one: run its protocol
**verbatim**, cite it, bank the primary (`tools/kb_add.py`), then remove `pending`
and fill `criteria`. Comparisons to other implementations use *their* published
protocol, named — never ours relabelled.

## What this skill does NOT do

It checks **completeness and admissibility**, not correctness. It cannot tell you a
number is *right* — only that the artifact reports what doctrine requires, stamps
its tier, names its estimator, and declares its gaps. Pair it with
`/TanitAD_ValidateAIDesign` for the gates and `/TanitAD_RunEval` for the run itself.

## Current state (MEASURED 2026-09-05, `tools/criteria_check.py --all taniteval/results/`)

Registry **v2.7.0**. 87 artifacts read → **39 in scope, 28 out of scope, 20 UNKNOWN**;
**493 violations, 61 work items** over the 39.

- TACTICAL: **5/39 present**, 34 missing (all three criteria).
- STRATEGIC decision + route/goal: **4 present, 1 refused**, 34 missing.
  nav-COMPLIANCE and its two controls: **0 present, 1 refused**, 38 missing.
- LATERAL yaw-rate: **5 present**, 34 missing. Curvature: 5 present, 27 refused.
- LONGITUDINAL distance-keeping: 4 present, **28 refused with a reason** — a tracked
  work item on the family that owns ~88.7 % of the oracle gap.
- Tier stamps: **27 T0, 7 UNSTAMPED, 5 T1**.

⚠️ **CORRECTION — the line this block used to carry, "Tier stamps: 27 T0, 7 unstamped.
Zero T1 driving artifacts exist.", was wrong on BOTH counts and is retracted.**

1. It was **already stale when written against the 2026-08-23 corpus**: four T1
   artifacts sat in `taniteval/results/` under the *unchanged* registry
   (`openloop-suite-DRYRUN-refcv3-fixture`, `openloop-suite-refav1-21109`,
   `openloop-suite-refcv3-30k-ckpt30000`, `refcv3-40284-openloop` — 3 violations
   each). The count came from a census taken before those were banked and was then
   re-quoted as a standing fact. Root-cause class: **a measured number quoted past
   its measurement date** — the same class as "REF-B v2 died at 22,600".
2. It is **now wrong by a much larger margin**, because the registry could not SEE
   the refav1 records at all: they nest the four families PER ARM at
   `arms.<arm>.four_families.*`, matched no in-scope marker, and every one read
   `UNKNOWN_SCOPE`. Fixed in registry v2.7.0.

**MEASURED after the fix:** **25 refav1-shaped records banked in-repo** (under
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-*/raw/`
plus `taniteval/results/refav1-21109-openloop.json`) — every one previously
UNKNOWN, now **IN_SCOPE, stamped T1, with 0 violations** and 3–4 work items each
(distance-keeping, and the three nav-compliance criteria the record declines with a
reason). Only the **`cl` planner arm** is scored; `ha` / `ha0` / `ha0_ext` are
controls and `ol` is T0. See `arm_scoring` in the registry.

⇒ The instrumentation gap is real but **smaller than this block used to claim**, and
its shape has changed: the four families ARE emitted by the refav1 path. What is
still missing across the older corpus is what the counts above show — and
**⛔ do not re-quote any of these numbers without re-running the census**, which is
the mistake being retracted here.
