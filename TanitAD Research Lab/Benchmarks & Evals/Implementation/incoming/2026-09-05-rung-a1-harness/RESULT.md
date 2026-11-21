# RUNG A1 — RESULT: the REF-C harness can now read the whole acceptance bar, and every pre-registered ablation is runnable

**Stream:** Benchmarks & Eval FlyWheel · **Date:** 2026-09-05 (Europe/Berlin) · **GPU: ZERO**
**Branch:** `agent/arch-inf-20260803` · ⛔ **Nothing was launched. `refcv4b-b1-v72-40k` was not touched.**

Closes both halves of `D-REFCV5-LADDER-2`, the two blockers the reconciled ladder registered as
rungs. The post-training hierarchy panel can start the moment the checkpoint lands.

---

## 1. What was wrong, MEASURED with same-breath controls

Every absence below is asserted only beside a control read in the **same command**, because four
false-absence claims were made on this mount on 2026-09-04 and a search tool reports "no matches"
for a file it could not open.

| absence | probe | control (same command) |
|---|---|---|
| `ha0_ext` is not an arm of the REF-C harness | `grep -c ha0_ext taniteval/tools/refcv3_arm.py` → **0** | `grep -c add_argument` same file → **28**; `grep -c ha0_ext taniteval/tools/refav1_arm.py` → **12** |
| the ablation switches are not CLI flags | `PREREG_REFCV4B_HIERARCHY_EVAL.md` §7 verbatim: the switches *"are NOT yet implemented as CLI flags of `refcv3_arm.py`"* | §3's twelve rows read from the prereg; three flags existed (`--with-navflip`, `--no-navshuf`, `--no-navzero`), FULL needs none |

`MEASURED (ours, 2026-09-05)`. Consequence of the first: **half of refcv5's acceptance bar — "beat
BOTH `ha` and `ha0_ext`" — was unreadable on the REF-C surface**, because the harness that produces
that surface never computed the control. Consequence of the second: the panel could not be run.

---

## 2. Deliverable 1 — `ha0_ext`, ported as ONE implementation

**The port calls `refav1_arm.hold_ext_controls` — the same function the refav1 harness calls —
generalised with an explicit `stride` whose default reproduces refav1's read bit-exactly.**
refcv3 calls it at the corpus's native 0.1 s tick (`stride=1`, `dt=0.1`, `loader=None`, since the
loader argument is read only for its `dt`). Nothing was re-derived in `refcv3_arm.py`; a test asserts
that by `__code__.co_filename`.

The arm reaches the **record**, not only the loop: `ARM_TIERS` (**T1**), `ARM_MEANING`, the DEFAULT
arm list, the manifest (the shared function's own docstring plus the literal call), the action-unit
block (the steer→kappa conversion applies to `ha0_ext` exactly as to `ha`), and two paired blocks —
`paired_os_minus_ha0ext` (the bar) and `paired_os_navzero_minus_ha0ext` (its deployment-relevant
form, without the oracle nav).

### 2.1 ⛔ A load-bearing choice, and the measurement that decides it

`D-REFCV5-LADDER-2` says the port *"must call the **same** `echo_gate.ha0_ext`"*. **There are two
`ha0_ext` kinematics in the programme and they are not interchangeable:**

* `tanitad.eval.echo_gate.ha0_ext` → `refc_v3.kinematic_goal_extrapolation` — the **closed form**,
  in the goal label's frame. Correct for the goal label, which is what it was built for.
* `refav1_arm.paths_from_controls` → `refa_v1_plan.unicycle_paths` — the **discrete 0.1 s unicycle**,
  which is what `ha`, `ha0` and refav1's own `ha0_ext` are integrated through.

**MEASURED (ours, 2026-09-05), over 6 s on plausible ego states:**

| v0 (m/s) | a0 (m/s²) | k0 (1/m) | max &#124;Δxy&#124; over 6 s | @2 s | @6 s |
|---|---|---|---|---|---|
| 10.0 | 0.00 | 0.000 | **0.000000** | 0.000000 | 0.000000 |
| 10.0 | 0.50 | 0.010 | 0.460776 | 0.125153 | 0.460776 |
| 8.0 | −0.40 | 0.030 | 0.338136 | 0.168056 | 0.338136 |
| 15.0 | 1.00 | −0.020 | **1.862923** | 0.540642 | 1.862923 |
| 5.0 | 0.00 | 0.050 | 0.340827 | 0.123703 | 0.340827 |

⇒ the two agree **exactly** on the degenerate straight-constant-speed case and diverge by up to
**1.86 m at 6 s** otherwise — *the same order as the entire model margin* (refcv3 `os` ADE ≈ 0.44 m
@2 s). Calling the closed form here would have produced a REF-C `ha0_ext` **incomparable with
refav1's** while looking like the more principled choice, and a cross-harness comparison would then
have measured the integrator instead of the model. The brief's instruction — *the same shared call
the refav1 harness makes* — is the one that survives measurement, and the divergence is now pinned by
a test so nobody "simplifies" the port later. **⇒ ESCALATION E1 below.**

### 2.2 A retraction applied where it had been sitting unapplied

`stack/tanitad/eval/echo_gate.py` retracts, **by name**, `refcv3_arm.py`'s line calling `ha0` *"the
echo test's real bar"* (`ha` is 2.2× harder, and the bar is `ha` **and** `ha0_ext` together). The
retraction had never reached the string a reader of the record actually sees. It has now, and a test
asserts the phrase is gone.

### 2.3 A diagnostic the port makes necessary

`ha0_ext_vs_ha0` reports the fraction of windows on which the echo control is **bit-identical to the
constant-velocity floor**. Where that fraction is high, *"beat BOTH"* has quietly collapsed into
*"beat `ha0`"* and the bar holds one control, not two — a reader must be told that as a number rather
than assuming two controls exist because two columns do. `ha0_ext` is excluded from **VOID-RISK** for
a **different** reason than `ha0`: its trivial fraction is a property of how straight and steady the
ego was at t0 on this corpus, not a defect of the arm, and escalating on it would make the loudest
warning in the tool fire on a control and be learned as noise.

*(On the synthetic fixture the instrument reads `ha0_ext` straight 0.0000 / const-speed 0.0714 /
CONSTANT-VELOCITY 0.0000 over 42 windows — i.e. a genuinely distinct arm there.)*

---

## 3. Deliverable 2 — every pre-registered ablation is runnable

`--ablate {gstr_zero, gstr_shuffle, e7_off, e9_off, h19_off, ego_zero, sel_refined, frames_blind}`,
plus `--ablate-frames` as its own named flag, plus `--gstr-bank` / `--gstr-shuffle-seed`.
With the three that already existed and FULL, the twelve §3 rows now have twelve routes, pinned as a
**bijection** by `test_the_registry_is_a_bijection_with_the_prereg_twelve`.

⚠️ **The prereg's own §7 escalation lists only seven of the eight missing switches — it omits the
frame-blind deliberate regression**, which is the arm the panel's validity rests on: *a gate that has
never been shown to FAIL an image-blind arm certifies nothing* (`H-ECHO-4`, where an ADE-scored gate
once passed an echoing arm). §3 is authoritative; the tool's `ABLATIONS` registry mirrors §3 and says
so in its own comment.

### 3.1 ⛔ A DISCREPANCY IN THE PRE-REGISTRATION, found by implementing it

**`H19-OFF` as registered would have removed nothing on the checkpoint it is registered for.**
§3 specifies `decoder.maneuver_to_anchor = None`. **MEASURED 2026-09-05:** every REF-C **v3/v4**
build — including `refcv4b` — sets `core.factored_maneuver = True` (`refc_v3.py:437`, `:615`), and
`refc.py:1214-1221` then builds the **factored pair** `lat_to_anchor` + `lon_to_anchor` and leaves
`maneuver_to_anchor` **None**. The switch therefore removes whichever anchor-prior heads the build
actually carries (`refc.py:1572-1581`'s `terms`), records **which**, and refuses only when the build
carries none. **⇒ the prereg needs an erratum, not the code a workaround. ESCALATION E2 below.**

### 3.2 The rules the flags obey

1. **Every flag changes the forward path** — pinned by rolling each ablation on a tiny CPU fixture
   and asserting at least one banked array moves against the FULL roll. *"A flag that parses but does
   nothing is worse than a missing one": it produces a table that reads like a knockout and is a copy.*
2. **Where a switch would be inert, the tool REFUSES** rather than running:
   `sel_refined` at diffusion `steps == 0` (`refc.py:1596` leaves `refined is conf` **by
   construction**; `:1630` gates `score_emitted` on `steps > 0`); `ego_zero` on a build fed no ego
   block; `gstr_shuffle` without a bank; a hier-only edge on a flat build; `h19_off` on a decoder
   with no anchor-prior head.
3. **`e9_off` reports itself INERT when the gate never opened.** `goal_gate` is zero-init, so on a
   checkpoint whose gate stayed at 0.0 the knockout **cannot** change the forward. That is a
   **finding about the checkpoint** (Caveat-B), not a licence to write *"no effect, therefore the
   seam is inert at eval"* — and the record says so in the arm's own block.
4. **`gstr_shuffle` permutes ACROSS WINDOWS, never across batch rows.** This harness's batch rows are
   the **nav conditionings of one window**, so a batch permutation would permute nav, not windows —
   a different intervention wearing the same name (the `--with-navzero` defect this tool already
   carries a warning about). It therefore reads `gstr_nav_true` from a banked FULL dump
   (`--gstr-bank`) and injects the permuted window's goal through the hook, with the inverse of the
   model's own normalisation so the emitted `g_str` is the banked value exactly.
5. **The injection is VERIFIED, not assumed.** On the first window the tool compares the model's own
   emitted `g_str` against what was injected and **refuses to roll** if they differ by > 1e-4. A hook
   registered on the wrong module raises nothing and reports nothing.
6. **`ego_zero` withholds v0 at the core too.** `refc.py` derives `keep` from `v0 is not None`, so a
   pre-zeroed `v0` would arrive with `keep = 1` — the file says so itself. The model-free controls
   keep the measured v0; they are controls, not arms.
7. **The stamp travels to every arm block.** The manifest carries `ablation`, `analyze_refcv3` copies
   it onto **every** `rec["arms"][x]`, and a dump whose manifest predates the stamp reads
   **UNKNOWN**, never *"none"* — *absence of the field is absence of the record*.
8. **Two regimes in one `--dump-dir` are refused**, and an `ABLATION.txt` marker names the regime.

---

## 4. Tests

| file | tests | what it prevents |
|---|---|---|
| `stack/tests/test_refcv3_ha0_ext_shared.py` (new) | **12** | a second implementation of the shared kinematic; a moved refav1 number; a `ha0_ext` computed but not emitted; the retracted "real bar" line; the wrong integrator |
| `stack/tests/test_refcv3_ablations.py` (new) | **28 passed, 1 skipped** | a missing switch; a switch that parses and does nothing; a silently inert switch; an unstamped result; two regimes in one dump dir |
| `stack/tests/test_refcv3_arm.py` (updated) | **20** | three pre-existing pins failed correctly on the new arm space and were updated to assert the NEW contract |

The one skip is the **bonus** cross-check of the tool's registry against the prereg **file**: the
off-Drive mirror does not carry `Project Steering/`, and the G: copy was errno-22 unreadable at the
time (12/12 attempts). The bijection pin itself — the twelve §3 arm names written into the test — ran
and passed; the file check only adds protection against a future prereg edit.

### 4.1 Suite, run from the off-Drive mirror, reported honestly

`C:\Users\Admin\tanitad-wt`, `PYTHONPATH=<mirror>/stack`, CPU only.

**Targeted modules — the ones this rung can break — are GREEN:**
`test_refcv3_arm.py` + `test_refcv3_ha0_ext_shared.py` + `test_refav1_arm.py` = **46 passed** after
the D1 pins were updated; `test_refcv3_ablations.py` = **28 passed, 1 skipped**;
`test_refcv3_ha0_ext_shared.py` alone = **12 passed** with the integrator-divergence pin added.

**A PRE-EXISTING collection error, demonstrated pre-existing rather than asserted:**
`stack/tests/test_closedloop_floor.py` fails at **collection** with
`ImportError: cannot import name 'FLOOR_ARMS' from 'closedloop_drive'`
(`stack/experiments/alpasim-gsplat/closedloop_drive.py`). Neither file is touched by this rung.
**Evidence:** the identical error reproduces with the **unpatched** copies of `refcv3_arm.py` and
`refav1_arm.py` swapped back in (`*.UNPATCHED.py`, md5-verified restored afterwards). ⇒ a work item
for whoever owns the AlpaSim closed-loop driver, not a regression here.

⚠️ An earlier sweep read *21 failed, 5,882 passed, 8 errors* but **overlapped** the swap above, so it
was discarded rather than quoted. The clean re-run below is the one reported.

### 4.2 The repo-wide sweep — 29 failures, **all 29 PRE-EXISTING, 0 regressions**

`pytest stack/tests -q --continue-on-collection-errors` from the mirror, CPU:
**21 failed · 5,883 passed · 113 skipped · 2 xfailed · 8 errors · 659.6 s.**

⛔ **"Pre-existing" is demonstrated here, not asserted.** The same selection was re-run with the three
modified files swapped back to their pre-Rung-A1 copies and the two new test files moved aside
(`raw/preexisting_check.py`; the restore is md5-verified):

| run | result |
|---|---|
| **A** — with Rung A1 applied | 21 failed, 284 passed, 8 skipped, **8 errors** |
| **B** — pre-Rung-A1 files restored | 21 failed, 284 passed, 8 skipped, **8 errors** |
| set difference | failing in BOTH: **29** · only with Rung A1: **0** · only without: **0** |

The 29 are, by module: `test_bev_consumer_fov` (1), `test_build_parity_guard` (1),
`test_decision_check` (5), `test_eval_contamination` (5 + 7 errors),
`test_launch_closure_audit` (1), `test_refav1_kin_contract` (1), `test_runbook_commands` (2),
`test_secret_scan` (1), `test_text_encoding_is_explicit` (1), `test_v6_chain` (2),
`test_v6_st_launch_fixes` (1), `test_closedloop_floor` (collection error).

⚠️ **One of them sits inside this rung's blast radius and still reproduces without it:**
`test_refav1_kin_contract.py::test_A6_adding_ha0_moves_no_existing_arm`. It fails identically with
the **unpatched** `refav1_arm.py`, so the `stride` generalisation did not move it — and the
bit-exactness of the legacy `stride=2` read is independently pinned by
`test_default_stride_reproduces_the_legacy_refav1_formula_bit_exactly`. It is a **work item for the
refav1 stream**, not a Rung A1 regression.

Raw logs: `raw/preexisting_patched.txt`, `raw/preexisting_unpatched.txt`.

---

## 4.3 A tooling escalation this rung had to solve to bank anything

⛔ **`stack/scripts/mm_commit.py` COULD NOT LAND A COMMIT TONIGHT — 25 attempts over ~40 minutes, all
dead at `read-tree`.** MEASURED 2026-09-05: `read-tree HEAD` walks all ~8,950 tracked entries across
the G: mount and hit `exit 0xC0000006 (mount paged out)` followed by a run of
`fatal: not a git repository` on **8/8** internal retries, on **every** driver attempt — while
`git rev-parse HEAD`, `git status` and `git cat-file -e HEAD:CLAUDE.md` all returned **0 in the same
second**, and a write probe to the repo root succeeded. ⇒ **the mount drops in bursts SHORTER than a
full-tree read but LONGER than a single-object read**, so the cost model, not the retry count, is the
problem. Six consecutive control reads of `CLAUDE.md` succeeded throughout, so this was **not** a
wedge and the PI's standing Drive-restart authorisation did **not** apply.

**The fix, and it is cheap:** `stack/scripts/mktree_commit.py` (added by this rung) builds the commit
with `hash-object` + `ls-tree -z` + `mktree`, rebuilding **only the ancestor directories of the named
paths** and reusing every sibling subtree SHA verbatim — O(depth × entries-per-directory) instead of
O(tracked files). It keeps mm_commit's compare-and-swap (`update-ref HEAD <new> <old>`), adds a
positive per-path assertion in the new tree AND after the commit, and refuses if a rebuilt directory's
sibling count moves. **Both Rung A1 commits landed first time with it** (`5cd86fd`, `0954934`), on the
same mount, minutes after mm_commit's 25th failure.

⇒ **Recommend it as the default committer while the mount behaves this way.** It is additive: no
existing script changes.

---

## 5. Escalations — integration, not a note in a doc

**E1 — `D-REFCV5-LADDER-2` says "the same `echo_gate.ha0_ext`"; the port calls
`refav1_arm.hold_ext_controls`.** The measurement in §2.1 (up to **1.86 m** divergence at 6 s) says
the harness integrator is the correct reading and the closed form would have broken cross-harness
comparability. The register row should be amended to name the harness derivation, or the difference
recorded as deliberate. **Master Mind decision.**

**E2 — `PREREG_REFCV4B_HIERARCHY_EVAL.md` §3's `H19-OFF` mechanism does not exist on `refcv4b`.**
See §3.1. An erratum to the prereg is needed; the code already does the right thing and records the
correction inside the arm's own registry entry. **Master Mind decision.** ⚠️ The prereg's falsifiable
object is its **git blob id at staging time**, so an erratum must be a NEW, separately-staged
statement — never a silent edit of the staged file.

**E3 — Lab-tree spelling drift.** A **singular** `TanitAD Research Lab/Benchmarks & Evals/` tree was
created at 2026-09-05 04:01 and is cited by `PREREG_REFCV4B_HIERARCHY_EVAL.md` §7, while the PI's
binding directive of 2026-08-27 makes the **plural** `Benchmarks & Evals` the live spelling and calls
the singular dead. This package therefore lands at the plural path. Two parallel trees is precisely
the failure that directive was written about. ⛔ **Do not resolve by comparing sizes or counts** —
`Hub`/`Lab` and `Eval`/`Evals` renames are near-byte-identical; **diff content.**

**E4 — the panel still needs one thing this rung does not provide.** `gstr_shuffle` requires a
banked **FULL** dump on the same episodes and window grid. The post-training panel must therefore
roll **FULL first** and point `--gstr-bank` at it; the tool refuses loudly otherwise, but the run
order is a runbook fact and belongs in the panel's launch script.

---

## 6. Deliverable manifest

| artifact | where it lives |
|---|---|
| `ha0_ext` port (roll, registry, manifest, paired blocks, diagnostic) | `repo:taniteval/tools/refcv3_arm.py` |
| the ONE shared derivation, `stride`-generalised | `repo:taniteval/tools/refav1_arm.py` |
| cross-harness pin (12 tests) | `repo:stack/tests/test_refcv3_ha0_ext_shared.py` |
| the eight ablation flags + registry + stamping + guards | `repo:taniteval/tools/refcv3_arm.py` |
| ablation pins (28 tests) | `repo:stack/tests/test_refcv3_ablations.py` |
| updated pre-existing pins | `repo:stack/tests/test_refcv3_arm.py` |
| this result + status | `repo:TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-09-05-rung-a1-harness/` |
| the replayable patch scripts (exact-match, read-back verified) | same package, `raw/` |
| register rows | `repo:Project Steering/GOALS_AND_CLAIMS.md` § `D-RUNGA1` |

⛔ Nothing is stranded on a pod, in a worktree, or in an agent's context. Everything above is in the
repo working tree and staged by commit.
