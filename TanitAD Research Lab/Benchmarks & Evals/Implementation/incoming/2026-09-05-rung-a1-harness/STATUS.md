# RUNG A1 — REF-C harness: `ha0_ext` port + the missing ablation CLI flags

**Stream:** Benchmarks & Eval FlyWheel · **Opened:** 2026-09-05 (Europe/Berlin) · **GPU cost: ZERO**
**Branch:** `agent/arch-inf-20260803`
**Package:** `TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-09-05-rung-a1-harness/`

⛔ **`refcv4b-b1-v72-40k` is LIVE on pod `tanitad-refcv3` (ETA ~2026-09-06 08:00 UTC). Nothing in this
rung touches it. No job is launched from this package.**

⚠️ **PATH NOTE / ESCALATION (see §Escalations).** The brief named the **singular**
`Benchmarks & Eval/…`. The binding PI directive of 2026-08-27 makes the **plural**
`TanitAD Research Lab/Benchmarks & Evals/` the live spelling and calls the singular the dead one.
This package therefore lands at the **plural** path. A *singular* `Benchmarks & Eval/` tree was
created under the Lab at 2026-09-05 04:01 and is cited by `PREREG_REFCV4B_HIERARCHY_EVAL.md` §7 —
that is a live drift, raised below rather than silently propagated.

---

## STATUS — updated in place as the rung lands

| # | deliverable | state |
|---|---|---|
| 0 | this STATUS header, banked before any work | ✅ committed |
| 1 | `ha0_ext` ported into `taniteval/tools/refcv3_arm.py` as the SAME shared call `refav1_arm.py` makes; in `arms`, in `ARM_TIERS` (**T1**), in the analysis record beside `ha`/`ha0`; pinned by a cross-harness equality test | ⏳ not started |
| 2 | the missing ablation CLI flags on `refcv3_arm.py` for every arm registered in `Project Steering/PREREG_REFCV4B_HIERARCHY_EVAL.md` §3, each **stamped into the arm's own record**; pinned by a test that every prereg arm has a flag AND that the flag changes the forward path | ⏳ not started |

---

## The two MEASURED absences this rung closes

Both re-verified with a **same-breath control** in the same command, per CLAUDE.md's false-absence
rule (four false-absence claims were made on this mount on 2026-09-04).

**A1-a — `ha0_ext` is unreadable on the REF-C surface.**

```
grep -c 'ha0_ext'      taniteval/tools/refcv3_arm.py   ->  0     (the target)
grep -c 'add_argument' taniteval/tools/refcv3_arm.py   -> 28     (SAME-BREATH CONTROL: the read works)
grep -c 'ha0_ext'      taniteval/tools/refav1_arm.py   -> 12     (the sibling harness HAS it)
```

`MEASURED (ours, this session, 2026-09-05)`. The control reading 28 is what makes the 0 a fact about
the file rather than about the mount. Consequence: **half of refcv5's acceptance bar — "beat both
`ha` and `ha0_ext`" — cannot be read on the REF-C surface today**, because the harness that produces
that surface never computes the control. `stack/tanitad/eval/echo_gate.py::ha0_ext` already exists
and `taniteval/tools/t1_eval.py` already calls it; this is a **wiring gap, not a missing instrument**.

**A1-b — most of the 12 registered ablations have no CLI flag.**

The prereg's own §7 escalation says so verbatim: the switches *"are NOT yet implemented as CLI flags
of `refcv3_arm.py`"*. ⚠️ **And that escalation list is itself incomplete** — it names seven
(`gstr_zero, gstr_shuffle, e7_off, e9_off, h19_off, ego_zero, sel_refined`) and **omits the
frame-blind deliberate regression**, which is the arm the panel's validity rests on: a gate that has
never been shown to FAIL an image-blind arm certifies nothing (`H-ECHO-4`: an ADE-scored gate once
passed an echoing arm). **The authoritative list is §3 of the prereg, not §7.**

---

## Discipline carried

* `Project Steering/AGENT_OPERATING_STANDARD.md`: stage, never push, never commit to `main`, never
  `git add -A`; deliverable manifest; escalate integration; bank incrementally.
* Commits ONLY via `python stack/scripts/mm_commit.py <msgfile> <path> [...]` (positional), verified
  by a **length-guarded** blob comparison (both oids 40 chars, else INCONCLUSIVE).
* Absence asserted only with a same-breath control; existence only by `git cat-file -e HEAD:<path>`.
* Suite run from the off-Drive mirror `C:\Users\Admin\tanitad-wt` with `PYTHONPATH=<mirror>/stack`;
  pre-existing failures demonstrated pre-existing on an **unpatched** copy before being reported as such.
* `Project Steering/GOALS_AND_CLAIMS.md` updated in the same turn, by **byte append with read-back**,
  never a str-split/rejoin (`insert_rows.py` was retired 2026-09-04 for exactly that bug).

## Escalations

1. **Lab-tree spelling drift (open).** Singular `Benchmarks & Eval/` exists under the Lab and is
   cited by the staged prereg §7; the PI's 2026-08-27 directive makes the plural binding. Two
   parallel trees is exactly the failure mode that memory entry was written about. Needs a Master
   Mind decision: re-path the singular tree (and the prereg citation) to plural, or amend the
   directive. **Do not resolve by size/count comparison — `Hub`/`Lab` and `Eval`/`Evals` renames are
   near-byte-identical; diff CONTENT.**
