# Drift instrument hardening — the entry-point set was the hand-list, and the nightly checker was 81 % wrong

**2026-08-18 · branch `agent/arch-inf-20260803` · HEAD `14623d7` · zero GPU, read-only on Thor**
**All numbers MEASURED unless stamped otherwise. Raw artifacts in `raw/`.**

---

## Headline

1. ⭐ **The closure tool's own entry-point list is fixed — the entry points are now DERIVED.**
   120 files (7 hand-listed entries) → 134 (14) → **161 from 52 derived entries**, a **strict
   superset of both**. Re-running against Thor over the widened set found **4 more genuinely stale
   or absent files** that C102/C105 could not see, including one **absent from the box entirely**.
2. ⛔ **And the widening immediately exposed a defect in the tool itself: it compares the WORKING
   TREE, not HEAD.** `stack/tanitad/models/v6.py` came back **DRIFT** while Thor's copy is
   **byte-identical to HEAD** — a sibling agent's *uncommitted* `FROZEN_EXTERNAL_*` work had moved
   the dev box. **`--ship` would have pushed work-in-progress onto a box running a 5-day job.**
   Now annotated and **held back by default**.
3. **`pod_git_drift.py` is REPAIRED, not deleted** — with the reasoning below. On the live fleet it
   would have printed **361 DRIFTED of which 293 = 81.2 % are artifacts** (227 line-ending,
   66 basename collisions). It also found **45 genuinely stranded TanitAD files on Thor** that
   nothing has been looking for since the RunPod fleet was released.

**Suites, MEASURED (mine, not inherited):** `stack` **3892 passed / 0 failed / 7 skipped /
2 xfailed** (596 s) · `taniteval` **1136 passed / 0 failed** (168 s).
Raw: `raw/suite_stack.txt`, `raw/suite_taniteval.txt`.

⚠️ **The +24 against the brief's baseline does not fully reconcile, and I am not going to pretend it
does.** Attributable: **+16 mine** (my two test files collect **41**, against **25** at HEAD —
verified with `--collect-only`) and **+9** from a sibling agent's **untracked**
`stack/tests/test_v6_frozen_external.py`, which pytest collects straight out of the working tree.
That accounts for +25 against a baseline of **3868 that I inherited and never measured**, leaving
**−1 unexplained**. ⇒ **The MEASURED number is 3892/0; the 3868 is INHERITED and is not quoted here
as fact.** *(Same root as the `v6.py` false DRIFT: with several agents live, the tree under your
feet is not the tree your baseline was taken from.)*

---

## JOB 1 — the entry-point list was a hand-list (C105, unapplied to the tool itself)

### What changed

The entry points are read out of **launch sources** — files that *emit or document* launch command
lines — instead of being remembered:

| launch source | why it is one |
|---|---|
| `stack/scripts/v6_chain.py` | the ladder launcher; every S-* command line is emitted here |
| `…/2026-08-07-hierarchical-wm-redesign/V6_GO_PACKAGE.md` | the operator's 3 a.m. runbook; §2's lines are already pinned against `v6_chain.py commands` by `stack/tests/test_runbook_commands.py` |
| `Project Steering/GATE_PROTOCOL.md` | the gate battery — where `run_gate.py` / `gate_emitters.py` come from |

The walk is a **fixed point, not a single pass**: a script named by a launch source is itself a
launch source. That is load-bearing, not tidiness — `train_v6_staged.py` **subprocesses**
`taniteval/tools/eval_four_families.py`, `seam_probe.py` and `t1_eval.py`, which **no import walk
can see** (they are argv, not imports), and all three were on C105's stale list.

### MEASURED

| root set (`--entry-mode`) | entries | closure | launchable scripts covered |
|---|---|---|---|
| `ladder` — C105's original hand-list | 7 | **120 files** | 24 / 158 |
| `fixed` — C105's widened hand-list | 14 | **134 files** | 30 / 158 |
| **`derived` — this work (DEFAULT)** | **52** | **161 files** | **54 / 158** |
| `executable` — every `__main__` script, the paranoid ceiling | 158 | 298 files | 158 / 158 |

The `executable` ceiling is the useful control: it says the derived set reaches **161 of a possible
298** files, so the remaining gap is a **number**, not a shrug.

`raw/closure_ladder7_local.json` · `raw/closure_fixed14_local.json` ·
`raw/closure_derived_local.json`. Pinned by
`test_the_real_derivation_strictly_widens_both_hand_lists`, which asserts
`ladder7 ⊂ fixed14 ⊂ derived` — **a fix that trades one file for another is not a fix.**

### ⚠️ Three honest limits, each stated rather than designed away

1. **The recursion bottoms out at the launch-source list.** It has to. The difference is that a
   launch-*source* list changes when the launch **mechanism** changes (rare, loud), where the
   entry-point list changed every time an instrument joined the ladder (frequent, silent).
   ⇒ **The tool now PRINTS its root set on every run** — sources, entries, each entry's provenance,
   the sys.path roots and the coverage gap. *A root set nobody sees is a root set nobody checks;
   that is exactly how the 7 survived.*
2. ⛔ **A derivation must WIDEN, never NARROW.** The derivation reaches 48 entries but **not**
   `watch_gates.py`, `t1_summary.py`, `run_spectral.py` or `v5_guard.py` — four instruments C105
   carried that **no current launch source names** (`t1_summary.py` is invoked from a *shell* chain;
   `run_spectral.py` only from `run_orthogonality.py`). Silently dropping them would be this tool
   regressing coverage while looking more principled. They are **kept as a floor and printed
   `FLOOR ONLY (no launch source names it)`** — audited on inherited authority, and said so.
3. **Coverage is now a measured fraction, not an assumption:** `54/158 launchable scripts covered ·
   104 NOT covered` (`raw/closure_derived_local.txt`). The remaining 104 are `__main__`-guarded and
   off the derived launch path. The denominator exists so "we audit the launch path" stops being an
   unexamined claim — and it **moves as siblings add scripts**, which is the point: it is measured
   each run, not remembered.

Scope unchanged from C105 and restated: only the Python closure; a script invoked through a
runtime-assembled name, or through a shell script no launch source names, is invisible.

### Re-run against Thor over the widened set — 4 new real findings

`raw/closure_derived_thor.json` · `SAME=88 CRLF_ONLY=68 DRIFT=4 MISSING_REMOTE=1`

| file | verdict | in the 134-closure? | what the box is missing |
|---|---|---|---|
| `stack/scripts/sel_winners_curse_law.py` | **MISSING_REMOTE** (18,266 B) | ❌ new | absent from Thor entirely |
| `stack/scripts/lf0_bev_lead.py` | **DRIFT** 345 → 544 ln | ❌ new | `corridor_fov_census` |
| `stack/scripts/p8_bev_reel.py` | **DRIFT** 291 → 406 ln | ❌ new | `NOFOV`, `NOFOV_HIT`, `iou_pair` |
| `stack/scripts/refc_train.py` | **DRIFT** 1339 → 1373 ln | ❌ new | (no top-level symbol delta) |
| `stack/tanitad/models/v6.py` | DRIFT | ✅ already covered | ⛔ **FALSE — see below** |

All four real ones are in the derived-only **+27**, i.e. **invisible to both hand-lists.**
⚠️ **Not shipped.** None is on the S-T launch path and Thor is 5 days from finishing; a ship is not
genuinely required, so this is an escalation, not an edit.

### ⛔ The defect the widening exposed: WORKING TREE vs HEAD

`stack/tanitad/models/v6.py` reported **DRIFT** (263,615 vs 252,018 B, missing 7 symbols
`FROZEN_EXTERNAL_FLAG`, `assert_frozen_external`, `declare_frozen_external`, …).

**MEASURED:** Thor's copy is **byte-identical to HEAD** (`git show HEAD:…` LF-md5 equal). The dev
box's tree is **ahead by an unstaged sibling-agent edit** (`stack/tanitad/models/v6.py` is ` M`,
with an untracked `stack/tests/test_v6_frozen_external.py` beside it).

⇒ **ROOT-CAUSE CLASS: the probe compared a different side than the question assumed** — the same
family as `df` on a pod, `free` on Thor, and cgroup `usage_in_bytes`. It is a **standing**
false-positive generator, because several agents live is this programme's normal state. And it is
worse than a wrong number: **`--ship` would have pushed a sibling's half-finished module onto the
training box.**

**Fixed:** every row carries `local_dirty_vs_head` / `remote_matches_head`; the run prints
`compared side: WORKING TREE (each row also checked against HEAD)`; a dirty DRIFT row is tagged
*"this is YOUR uncommitted edit, NOT box staleness"*; and **`--ship` holds those files back unless
`--ship-dirty` is passed deliberately.** Pinned by
`test_a_dirty_local_file_is_not_reported_as_box_staleness`.

⚠️ **Method note on how this was nearly missed:** my first `git status --short` was piped through
`head -30`, which cut the `stack/` rows. *Absence found at one location is not absence* — the
truncation, not the repo, produced the clean read.

---

## JOB 2 — `pod_git_drift.py`: REPAIR, and the argument for it

### The decision

**REPAIRED, not deleted.** Deleting a misleading instrument is legitimate, and I considered it
seriously — but it fails on one fact: **`pod_git_drift.py` answers a question no other instrument
in the programme answers.** `launch_closure_audit.py` only inspects files that are *in the repo's
import closure*, so a **box-only file — the REF-B v2 / TanitEval failure mode, the reason the
Operating Standard exists — is invisible to it by construction.** Deleting would leave the
programme's *documented dominant failure mode* with **no instrument at all**, and the Standard
would cite a deleted file.

The repaired run **immediately justified the choice**: it found **45 genuinely stranded TanitAD
files on Thor** (below), which nothing has been looking for since the RunPod fleet was released on
2026-08-15.

**Callers checked, two probes:** (a) repo-wide grep across `*.sh|*.yml|*.yaml|*.json|*.toml|*.bat|
*.ps1`; (b) `.claude/{skills,hooks,settings*.json}` and `Project Steering/AgentSchedule`.
**Result: no automated caller anywhere** — only `stack/tests/test_pod_git_drift.py`, the Operating
Standard's cadence line, historical reports, and md5 manifests from a 2026-07-26 audit. It has been
**doctrine that never ran**, which is precisely why four defects accumulated unseen.

### The four defects, MEASURED

| # | defect | measurement |
|---|---|---|
| 1 | ⛔ the index walked **`.claude/worktrees/`** — 14 stale full repo copies | **8,079 indexed files against the repo's 2,132 (3.8×)**; ambiguous basenames **5.6 % → 89.4 % of names**. Every `IN_GIT`/`DRIFTED` verdict was decided against a pool of stale duplicates. |
| 2 | ⛔ matched by **basename anywhere in the repo** | even worktree-free, `__init__.py` has dozens of copies; **66 rows on Thor** were unrelated same-named files called drift |
| 3 | ⛔ **no CRLF normalisation** | **28.1 % of `stack/` and 19.3 % of `taniteval/`** sources contain CRLF → **227 rows on Thor** |
| 4 | ⛔ `DEFAULT_PODS` = four **dead machines** | released 2026-08-15; it would print four unreachable hosts and a reassuring `TOTAL: 0` |

Plus the fifth from `PRODUCED_GOAL_PATH.md`: **`-maxdepth 3`** put `/root/TanitAD/stack/tanitad/**`
(depth 5) below the horizon — the trees that were later found 52 % wrong were **never scanned**.

⚠️ **The brief's "~50 % false" was the right order and the wrong denominator, and the distinction
matters.** Two honest numbers, both stated: **~26 % of all inspected files** would be falsely
called drift (the CRLF rate); **81.2 % of the rows it actually prints as drift** are artifacts.
On the live checkout alone the ratio reaches **227 of 251 = 90.4 %**, reproducing C105's 94 % on a
different tool. *The rate rises with how well-synced the box is — a well-synced box makes the
denominator almost pure artifact.*

### What the repair does

Verdicts are now `IN_GIT` · **`CRLF_ONLY`** (its own category, never drift) · `DRIFTED` (a repo file
at the **same longest path suffix** differs) · **`NAME_ONLY`** (basename-only hit — weak evidence,
explicitly *not* drift) · `HOST_ONLY`. Content agreement still outranks path agreement, so a
rescued file relocated in the repo is still `IN_GIT`. Defaults repointed at `tanitad-thor-wifi`;
`maxdepth 6`; package caches excluded; **an unreachable host is a loud non-zero exit, not a quiet
zero** — *absence of a finding is not a finding of absence*.
⚠️ **A sixth instance of that same class, found while testing the repair:** `--hosts` with no values
scanned nothing and printed `TOTAL HOST-ONLY FILES: 0`, **exit 0** — the dead-pods failure in
miniature. It is now a hard error (exit 2), pinned by a test. Output leads with a per-directory
shape (265 files in one vendored tree is one fact, not 265) and prints **what the unrepaired tool
would have said**, so the repair cannot be quietly undone by someone who thinks the old numbers
looked more thorough.

The docstring now opens with **what it does NOT answer** and points at `launch_closure_audit.py`
for the converse. ⇒ **`Project Steering/AGENT_OPERATING_STANDARD.md` §"Standing cadence" is
corrected** to name **both** tools, their directions, and the rule that they are converses.

### Live result — `raw/drift_thor_after.json` / `.txt`

`1600 files · 929 IN_GIT · 227 CRLF_ONLY · 68 DRIFTED · 66 NAME_ONLY · 310 HOST_ONLY`

**HOST_ONLY, grouped:** **265** are the vendored `alpasim` tree (third-party, not ours). **45 are
genuine TanitAD work living in exactly one place:** `/home/nvidia` root **19**, `nurec_work/` **16**,
`rq_out/` **4**, `s1_climbout/` **2**, and `nurec-gsplat/`, `lambda_findability/`, `parity_verify/`,
`_s1_backup/` **1** each.

**DRIFTED:** 44 are in deliberate backup/clone trees (`_thor_backup_*`, `tanitad_cl/`).
**24 are in the live `/home/nvidia/TanitAD` checkout** — outside the launch closure (which C105
left at DRIFT 0): `stack/tests/*` ×8, `stack/scripts/*` ×9 (`ph0_sam3`, `ph0_v2`,
`aug120_pipeline`, `ph1_fuse`, `ph0_rich_overlay`, …), `taniteval/taniteval/*` ×3, `taniteval/
tests/*` ×2, `Paper/figures/*` ×1. ⚠️ **The tool cannot say which side is newer** — that is a real
limit of a content comparison and is not guessed at here.

---

## ⛔ Escalations — decisions, not doc notes

1. 🔴 **45 stranded files on Thor.** Chiefly `nurec_work/` (16 — the NuRec/gsplat investigation
   that produced the msgpack + 492 FPS findings), `rq_out/` (4), and 19 loose scripts at
   `/home/nvidia`. **Each exists in exactly one place on earth, on a box.** This needs a rescue
   pass; I did not do it here because it is a different work item and Thor is training.
2. 🟠 **4 stale/absent files on Thor** (`sel_winners_curse_law.py` absent; `lf0_bev_lead.py`,
   `p8_bev_reel.py`, `refc_train.py` drifted). Not on the S-T path, **not shipped** — ship them with
   the next legitimate sync, not mid-run.
3. 🟠 **24 DRIFT rows in Thor's live checkout outside the launch closure.** Direction unknown.
   Worth one pass to decide which side is authoritative before either is overwritten.
4. 🟢 **`AGENT_OPERATING_STANDARD.md` edited** (standing-cadence section). Flagged because it is a
   steering doc: it now names both tools and their directions.

---

## Deliverable manifest

| artifact | where it lives | note |
|---|---|---|
| `stack/scripts/launch_closure_audit.py` | `repo:` **staged** | derived entry points, root-set banner, coverage gap, dirty-vs-HEAD attribution, `--ship-dirty` guard |
| `stack/tests/test_launch_closure_audit.py` | `repo:` **staged** | 19 → **29** collected (13 → 23 fns + a 7-way parametrize) |
| `stack/scripts/pod_git_drift.py` | `repo:` **staged** | repaired: CRLF, path-suffix match, worktree exclusion, live fleet, loud unreachable, empty-host-list is an error |
| `stack/tests/test_pod_git_drift.py` | `repo:` **staged** | 6 → **12** tests |
| `Project Steering/AGENT_OPERATING_STANDARD.md` | `repo:` **staged** | standing-cadence correction |
| `DRIFT_INSTRUMENT_HARDENING.md` (this file) | `repo:` **staged** | |
| `raw/closure_{ladder7,fixed14,derived}_local.json/.txt` | `repo:` **staged** | the 120 / 134 / 161 comparison |
| `raw/closure_derived_thor.json/.txt` | `repo:` **staged** | the 4 new findings + the dirty-vs-HEAD attribution |
| `raw/drift_thor_after.json/.txt` | `repo:` **staged** | the live fleet run |
| `raw/suite_stack.txt`, `raw/suite_taniteval.txt` | `repo:` **staged** | 3882 / 1136 |

**Nothing produced by this work exists in only one place.** Nothing was written to Thor: every
remote call was `ssh -n` running `find`/`sha256sum`/a staged read-only hash probe. **No commit, no
push, no branch switch.**

**Live run untouched** — no GPU work, no writes to the run directory, no process signalled.
MEASURED after all remote probes: PID **25477 alive** (`/proc/25477`, elapsed 181,174 s ≈ 50.3 h),
`v6F-SW-30k` at step **13,000**, `step_s` **26.4736** — progressed normally from the ~12,950 the
brief reported, at an unchanged step time. ⚠️ The `ps` check was done **by explicit PID**, and the
one filtered probe assembled its token with `printf "pyth\x6fn"` so the filter could not match its
own command line (the self-match trap, four measured instances).
