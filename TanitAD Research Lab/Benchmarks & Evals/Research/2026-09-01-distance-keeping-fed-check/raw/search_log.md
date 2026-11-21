<title>search log - distance-keeping fed check</title>

# raw/search_log.md — E-LAB-BE-0901, 2026-09-01

Source-and-census led. **No web search was run** — the question is entirely about our own eval
path and our own artifacts. Recorded explicitly so the absence of external citations is not later
read as an omission.

## Probe log

| step | what | outcome |
|---|---|---|
| 1 | Locate `lead_metrics.py` | present in the main tree at `taniteval/taniteval/lead_metrics.py` (also in 5 worktrees, and in the Lab's `Implementation/incoming/2026-08-03-longitudinal-distance-keeping/`) |
| 2 | Find its consumers | `four_families.py` computes `distance_keeping.mean_headway_min_m` / `mean_time_gap_min_s` / `mean_min_ttc_s`; `driving.py` documents the retired refusal |
| 3 | Read `driving.py:65` and `:613-620` | **"REFUSAL RETIRED 2026-08-18"**, and the old reason explicitly labelled *"a STALE ABSENCE-CLAIM"* |
| 4 | Census run 1 over `taniteval/results/*.json` | 75 files; **0 numeric**, 28 non-numeric, 43 no-key, **4 OSError** |
| 5 | ⛔ Declared run 1 INADMISSIBLE | 4 Errno-22 reads. Per CLAUDE.md a G: failure count needs **zero** Errno 22 in the run |
| 6 | Attempted local mirror (retry loop, 6 tries/file) | **abandoned** — only 2 of 75 files copied in 6.6 min; the mount was in a slow wave |
| 7 | Identified the 4 failing files exactly | `driving_flagship-30k` / `-nospeed` / `-speed` / `-v16-ab-ft`.json, 32–33 KB each |
| 8 | Targeted retry, 8 spaced attempts each | first two: **STILL UNREADABLE** after 8 attempts |
| 9 | **Second, independent API** (.NET `File.ReadAllText` via PowerShell) | all four fail with `MethodInvocationException` |
| 10 | Census run 2 (the banked one) | reproduced run 1 **exactly**: 0 / 28 / 43 / 4 |

⭐ **Step 9 is what makes F4 a finding rather than a complaint.** Python `open()` failing is one
probe; the operating standard requires a second path before writing "X cannot be read". Two
independent APIs, three runs, ~25 minutes apart, all agree.

⭐ **Step 10 matters too**: the census was run twice, by two different code paths (an ad-hoc
inline script and the banked `code/dk_census.py`), returning identical counts. A census that ran
once is a number; a census that reproduces is a measurement.

## Named EMPTY searches

| id | looked for | result |
|---|---|---|
| **E1** | any banked artifact carrying a NUMERIC distance-keeping value | **EMPTY — 0 of 71 readable.** This is the pre-committed criterion for H-DK-1, and it fails to close the row. |
| **E2** | more than one distinct unavailability reason string | **EMPTY — exactly 1**, identical across all 28. A single string means a single upstream site, which is good news for the fix. |

⚠️ **E1 is bounded by F4**: 4 artifacts are UNREAD, so the honest statement is "0 of 71
readable", never "0 of 75". Those 4 remain capable of overturning the headline in principle.

## What was NOT done, and why

- **The 4 unreadable artifacts were not recovered.** Two APIs and 8+ retries failed. Recovering
  them likely needs a different host (they may be readable from a machine with a healthy mount),
  which is escalated rather than attempted here.
- **`win["lead"]` was not wired.** That is a `taniteval` code change on the Eval FlyWheel's
  surface, not a Lab deliverable; the Lab measured the gap and named the cheapest fix.
- **`lead_metrics.py` was not re-validated.** D-LEAD-1's admission is quoted as **INHERITED** from
  a source comment; re-running that control was out of scope and is not claimed.
- **No comparison against the worktree copies** of `lead_metrics.py` (5 exist). The main-tree copy
  is the one the consumer imports.
