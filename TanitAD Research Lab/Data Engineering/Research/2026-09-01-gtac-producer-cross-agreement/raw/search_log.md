<title>search log - g_tac producer cross-agreement</title>

# raw/search_log.md — E-LAB-DATA-0901, 2026-09-01

This package is **source-and-measurement led**. No web search was run: the question is entirely
about two files in our own repo. Recording that explicitly so the absence of external citations
is not later read as an omission.

## Probe log

| step | what | outcome |
|---|---|---|
| 1 | Locate both producers | `stack/tanitad/data/tactical_goals.py` (A) and `stack/tanitad/data/g_tac_geom.py` (B) — **both in the main tree**, not only in worktrees |
| 2 | ⚠️ Second probe for B (absence discipline) | `find` also returned `.claude/worktrees/exciting-kepler-4d4449/…/g_tac_geom.py`. The main-tree copy is the one imported; the worktree copy was NOT compared and may differ. Flagged, not assumed identical. |
| 3 | Read both entry-point signatures | **identical contract**: `poses [T,4]` = (x,y,yaw,v) + key index + the same 2–6 s band. This is what makes agreement well-posed — checked before measuring, not after |
| 4 | Confirm A is not being starved of inputs | `derive(poses, *, key, hz, stop_reason=None, lon_vocab)` — geometry-only by signature. **"A was missing its VLM leg" was the obvious way for this result to be an artifact; it is ruled out in source.** |
| 5 | Import both from the G: tree | succeeded (`PYTHONPATH=<repo>/stack`) — the Errno-22 editable-install trap did not bite on this path today |
| 6 | Look for real parity poses on the dev box | **EMPTY** — see below |
| 7 | First run | A returned `None` for every token: I used `ga.lat`/`ga.lon`, but the dataclass fields are `lat_token`/`lon_token`. **Caught because every row read `None`, which is not a plausible label.** Fixed and re-run. |
| 8 | Second run | the reported result |

⚠️ **Step 7 is worth keeping on the record.** A silently-wrong attribute name produced eight rows
of `A(lat=None, lon=None)` against a fully-working B. Had A's dataclass happened to expose a
*different but valid* attribute, the run would have produced plausible-looking garbage instead of
obvious `None`s. The generalisable guard is the one that caught it: **B's column was populated and
sensible while A's was uniformly empty — an asymmetry no real disagreement produces.**

## Named EMPTY searches

| id | looked for | result |
|---|---|---|
| **E1** | real parity poses on the dev box, long enough for the 2–6 s band | **EMPTY.** `taniteval/results/windows_*.pt` hold only `pred/gt/cv [881, 4, 2]` — **4 waypoints**, far short of the ~60 samples the band needs. Probed a second way (the dump's own `wp_steps` field = 4), so this is a two-probe absence. |
| **E2** | a local episode cache with full pose tracks | not pursued after E1 — the parity corpus is on the pod/Thor side and the Lab does not touch either. The corpus-rate run is escalated as a proposed row rather than faked locally. |

⛔ **Standing note:** do not re-probe E1 on the dev box. The corpus run needs parity poses, which
means it belongs to a FlyWheel with pod access, not to the Lab.

## What was NOT done, and why

- **No corpus agreement rate.** See E1. The scenario-set rate (1/7) is reported as such
  throughout, and `RESULT.md` §5 states plainly that it must never be quoted as a corpus rate.
- **No re-verification of B's own refutation numbers** (oracle p50 1.80 m, 21.7 % theta band).
  Those are quoted as **INHERITED** from B's docstring and explicitly flagged as not re-measured
  here. F4 reaches the same conclusion independently, which is worth more than re-running B's study.
- **No threshold tuning.** Out of scope by the SPEC, written before the run.
- **No comparison against the worktree copy of `g_tac_geom.py`** (step 2). If that copy has
  diverged, this comparison describes the main tree only.
