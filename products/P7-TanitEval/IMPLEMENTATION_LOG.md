# P7 + P9 — implementation log, 2026-08-23

`Written by TanitAD_EvalFlyWheel after five parallel implementation streams. ⚠️ ALL FIVE
STREAMS DIED — four on a 600 s stall watchdog, one on an API error — and every one of them
died at the SAME phase: writing its long closing markdown to the G: Drive mount. The code
and tests they had already written survived, and are verified below by RUNNING them, not by
trusting the agents' reports (which do not exist). This log replaces those five reports.`

---

## 0. Verified state — measured, not inherited

| suite | result | how run |
|---|---|---|
| `tools/tests` + `stack/tests/test_resim.py` | **406 passed** | `PYTHONUTF8=1`, `PYTHONPATH=<wt>/stack;<wt>/taniteval` |
| `tools/tests/test_criteria_check.py` | 48 passed | incl. per-family regression arms |
| `tools/tests/test_release_gate.py` | 35 passed | incl. a regression arm per criterion |
| `taniteval/tests/test_navsim_adapter.py` | 50 passed | synthetic fixtures, no dataset |
| `stack/tests/test_resim.py` | 55 passed | ⚠️ see §1 |
| `taniteval/tests/test_no_jack_in_gates.py` | 16 passed | guard now scans 36 files, was **0** |

## 1. ⚠️ THE TRAP THAT NEARLY PRODUCED A FALSE FAILURE REPORT

`stack/tests/test_resim.py` first reported **16 failures**, seven of them the new
prediction-slot tests. The work was not broken: a bare `import tanitad` resolves to the
**PRIMARY checkout** (`<repo>/stack/tanitad/__init__.py`), not the worktree, because of the
venv's editable install. The tests were importing a tree that does not contain the new module.

```
PYTHONPATH="<worktree>/stack" python -m pytest stack/tests/test_resim.py -q   # 55 passed
```

This is the documented editable-install trap, and it bit inside this session. ⇒ **Always
re-check which tree was imported** (`python -c "import tanitad; print(tanitad.__file__)"`)
before recording any test failure in this repo.

## 2. What each stream delivered

### 2.1 Release gate — COMPLETE
`tools/release_gate.py` (51 KB), `tools/tests/test_release_gate.py` (35 tests),
`products/P7-TanitEval/RELEASE_GATE.md`, dry run banked at
`products/P7-TanitEval/release_gate_dryrun_2026-08-23.json`.
21 criteria, four verdict states including **CANNOT-RULE**. First dry run on our best banked
T1 arms: **RELEASE-BLOCKED**, 8 blocking FAILs, 0 CANNOT-RULE. The gate was not weakened to
let our own arms through. Detail in `RELEASE_GATE.md` §2.

### 2.2 NavSim adapter — COMPLETE (synthetic-tested)
`taniteval/adapters/navsim.py` (60 KB) + `__init__.py`, `taniteval/tests/test_navsim_adapter.py`
(50 tests) — built against the `ff_rescore.load_dump` seam, frame conversion tested with a
deliberate wrong-convention regression arm, `pdm_score`-as-EPDMS refused in code, and the
families NavSim cannot supply refused inline with reason + n.

⛔ **Open by design: the interval.** NavSim's resampling unit is a scene token, not an
episode, so our episode-cluster bootstrap does not transfer. The adapter emits the point
estimate and an explicit `UNAVAILABLE` for the CI rather than inventing one. **Do not report
a NavSim CI until this is settled.**
⛔ **No dataset on disk** — a real run needs a PI decision (hundreds of GB).

### 2.3 TanitResim — the correctness defect is FIXED
`stack/tanitad/viz_standard.py` (new), plus changes across
`resim/{export,sample,README,static/app.js,static/style.css}` and `replay/{arms,engine}.py`.
55 tests pass, including new ones asserting that a GT-derived input can never occupy a
prediction slot — the nav-echo class. The viz standard is now a CONTRACT: a frame with a
silently missing element does not render, and an unavailable element is drawn as
`unavailable: <reason>`.

### 2.4 Forbidden estimator — ⚠️ PARTIAL, and the finding is bigger than the fix
**Delivered:** the C134 guard-vacuity fix (§3) and an **arithmetic-shape detector**
(`gate_guard.scan_*_shapes`) that catches the `1.96·std/√n` shape under ANY name. Verified
live: it flags `stack/scripts/driving_diagnostic.py:172` — the unnamed clone that two prior
audits missed.

**NOT delivered — the three actual call-site fixes:**
- `stack/scripts/driving_diagnostic.py:167-177` — the clone itself, still live.
- `taniteval/taniteval/bench.py:303` — still emits the legacy block under the bare key
  `"heldout"`, which `gate_guard` cannot see; `runner.py:409-410` still falls back to it.
- `taniteval/recompute_ci.py:91` — still outside `ENFORCED_ROOTS`.

⛔ **Deliberately not rushed:** `mean_ci` computes published INTERVALS. Changing it changes
published numbers, and the standing rule is to stop and report rather than quietly restate.
**PI decision.** The class is now closed by machinery (detected) even where the instances
remain.

### 2.5 Eval pipeline — NOT DELIVERED, but it left one real fix
The stream died before writing `run_eval_pipeline.py`. It did find a latent defect in
`tools/criteria_check.py`: dotted-path resolution split naively on `.`, so a field whose NAME
contains a dot (`n_excluded_goal_below_0.5m`, emitted inside `goal_setting`) was unreachable
and read as ABSENT — the C133-b failure mode again. **Fixed and pinned** with three tests.

**Still open:** intervals on every LON/LAT component, the mandatory-tier pipeline entry point,
and the `pred_dense` KeyError on sparse-only dumps.

## 3. C134 — the guard that policed nothing

`gate_guard.scan_paths` matched skip tokens against the **absolute** path while its caller
passed `"/.claude/"`. Every agent works under `<repo>/.claude/worktrees/<name>/`, so the token
matched the checkout root and the scan skipped **373 of 373 files**. It is correct from the
primary checkout and vacuous from every worktree — it worked for the PI and not for the
programme.

⇒ **A scanner must assert on its own INPUT COUNT.** A scan that examined zero files is a
FAILED scan, not a clean one. Full entry in `RETRACTION_LOG.md`.

## 4. Deliverable manifest

| artifact | where | state |
|---|---|---|
| `tools/release_gate.py` + tests + `RELEASE_GATE.md` + dry-run JSON | repo | staged, 35 tests green |
| `taniteval/adapters/navsim.py` + `__init__.py` + tests | repo | staged, 50 tests green |
| `stack/tanitad/viz_standard.py` + resim/replay changes | repo | staged, 55 tests green |
| `tools/criteria_check.py` (+ `audit_keys`, `--recursive`, `_dig` fix) | repo | staged, 48 tests green |
| `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.2.0 | repo | staged |
| `taniteval/taniteval/gate_guard.py`, `ci.py` | repo | staged, 16 tests green |
| `products/P7-TanitEval/VISION.md`, `TANITEVAL_AUDIT.md`, `benchmarks/*.md` | repo | staged |
| `products/P9-TanitResim/SPEC.md` | repo | staged |
| census JSONs (repo-wide, registry v2.2.0) | `products/P7-TanitEval/raw/` | staged |

⛔ Nothing committed, nothing pushed.

## 5. Next, in order

1. **Declare inference inputs + parity in the emitter** — closes RG-07, RG-08 and RG-12
   without touching a model. Cheapest real gain available.
2. Intervals on every LONGITUDINAL and LATERAL component (§2.5).
3. The three estimator call sites — **needs the PI's word**, they move published intervals.
4. Emit `<arm>_fan_err` / `<arm>_sel_idx` so tactical anchor selection becomes scoreable.
5. NavSim: settle the cluster unit, then the dataset decision.

## 6. Addendum — full-suite verification and the protocol contract

**Full suites, run after every change above:**

| suite | result |
|---|---|
| `taniteval/tests` | **1235 passed**, 0 failed (223 s) |
| `tools/tests` + `stack/tests/test_resim.py` | **406 passed** |
| affected subset re-run after the last patch | 305 passed, then 53 passed |

**Item 1 of §5 is now DONE.** `all_families()` takes a `protocol=` dict and emits
`_protocol` + `_protocol_undeclared`. It **never guesses**: an emitter that invented
`vision_only: true` would be manufacturing compliance, which is worse than the gap, so an
absent declaration is written out as
`UNDECLARED — pass \`protocol=\` to all_families(); NOT assumed compliant`.

Same contract as the tier fix: the gap is visible, never silent. Three regression arms pin
it, including one asserting that declaring SOME fields does not silence the others. Callers
can now close RG-07, RG-08 and RG-12 by passing what they already know — **no model change,
no re-training, no re-scoring.**
