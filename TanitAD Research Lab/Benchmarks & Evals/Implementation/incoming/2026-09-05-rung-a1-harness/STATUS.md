# RUNG A1 — REF-C harness: `ha0_ext` port + the missing ablation CLI flags

**Stream:** Benchmarks & Eval FlyWheel · **2026-09-05 (Europe/Berlin)** · **GPU cost: ZERO**
**Branch:** `agent/arch-inf-20260803`
**Package:** `TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-09-05-rung-a1-harness/`

⛔ **`refcv4b-b1-v72-40k` is LIVE on pod `tanitad-refcv3` (ETA ~2026-09-06 08:00 UTC). Nothing in this
rung touched it. No job was launched from this package.**

⚠️ **PATH NOTE / ESCALATION E3.** The brief named the **singular** `Benchmarks & Eval/…`. The binding
PI directive of 2026-08-27 makes the **plural** `TanitAD Research Lab/Benchmarks & Evals/` the live
spelling and calls the singular dead. This package therefore lands at the **plural** path. A
*singular* tree was created under the Lab at 2026-09-05 04:01 and is cited by
`PREREG_REFCV4B_HIERARCHY_EVAL.md` §7 — a live drift, raised rather than silently propagated.

---

## STATUS — ✅ BOTH DELIVERABLES LANDED AND VERIFIED IN `HEAD`

| # | deliverable | state | commit |
|---|---|---|---|
| 0 | this STATUS header, banked before any work | ✅ | `37af3bc` |
| 1 | `ha0_ext` ported as the SAME shared call refav1 makes; in `arms`, `ARM_TIERS` (**T1**), `ARM_MEANING`, the manifest and two paired blocks; cross-harness equality pinned | ✅ **12 tests pass** | `5cd86fd` |
| 2 | all twelve prereg §3 ablations runnable from the CLI, each stamped into the arm's own record; bijection + "the flag bites" pinned | ✅ **28 pass, 1 skip** | `0954934` |
| 3 | register rows `D-RUNGA1-1..8`, byte-appended with read-back + sentinel re-assertion | ✅ 9,593 bytes; 870,628 → 880,221 | this package's commit |
| 4 | `RESULT.md` + `raw/` (replayable patch scripts, the divergence measurement, both suite logs) | ✅ | this package's commit |

Every commit was verified in `HEAD` by a **length-guarded blob comparison per path**, not by `git log`.

---

## Headline findings

1. ⭐ **The acceptance bar is readable.** `grep -c ha0_ext refcv3_arm.py` was **0** (same-breath
   control `add_argument` → **28**; sibling `refav1_arm.py` → **12**), so half of *"beat BOTH `ha` and
   `ha0_ext`"* could not be computed on the REF-C surface. It now can.
2. ⛔ **There are TWO `ha0_ext` kinematics in the programme and they differ by up to 1.862923 m over
   6 s** (0.540642 m at 2 s on the worst state; exactly 0.000000 m on the degenerate one). That is the
   same order as the whole model margin, so *"call the same shared kinematic"* has two readings and
   only one preserves cross-harness comparability. **ESCALATION E1.**
3. ⛔ **The prereg's registered `H19-OFF` mechanism removes nothing on the checkpoint it is registered
   for** — every REF-C v3/v4 build is `factored_maneuver=True`, so `maneuver_to_anchor` is `None` and
   the live prior is `lat_to_anchor` + `lon_to_anchor`. **ESCALATION E2: the prereg needs an erratum.**
4. ⚠️ **The prereg's §7 escalation list omits the frame-blind deliberate regression** — the arm the
   panel's validity depends on. §3 is authoritative.
5. ✅ **Suite: 21 failed / 5,883 passed / 8 errors, and all 29 failures are PRE-EXISTING —
   0 regressions**, demonstrated by re-running the same selection with the pre-Rung-A1 files restored.
6. ⛔ **`mm_commit.py` could not land a commit tonight** (25 attempts, ~40 min, all dead at
   `read-tree`). `stack/scripts/mktree_commit.py` — added here — landed both commits first time.
   **ESCALATION: adopt it as the default committer while the mount behaves this way.**

See `RESULT.md` for the evidence, the escalations E1–E4, and the deliverable manifest.
