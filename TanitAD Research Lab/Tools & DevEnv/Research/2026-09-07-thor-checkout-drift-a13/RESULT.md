# A13 — adjudicating the drift in Thor's `/home/nvidia/TanitAD` checkout

**Work package** · Tools & DevEnv · Research · **2026-09-07** · **0 GPU** · read-only on both sides.
⛔ **Nothing was written to Thor. Nothing was overwritten on either side.** No `git checkout`, no
`git fetch`, no rsync, no file ship. Every Thor-side command was `find` / `md5sum` / `stat` /
`git ls-tree` / `git hash-object` / `git cat-file`, all read-only, all over `ssh -n`.

---

## 0. The three questions, answered first

| question | answer |
|---|---|
| **How many drifted files are there really, vs the row's 24?** | **156 real drifted files**, not 24 — and the 24 is not wrong, it is **dated**. A raw blob-sha comparison finds **165** rows; **9 of them are CRLF-only** (files shipped off the Windows dev box) and are not a currency defect. The row's 24 was measured 2026-08-18 against that day's HEAD; the repo has since advanced **1,273 commits**. All 24 of the original paths are still drifted today, and all 24 are now settled. |
| **Which side is authoritative, file by file, and what settled it?** | **THE REPO, on every single row — there is no row where Thor is newer.** 140 rows are settled by a *positive identification of a commit*: Thor's bytes ARE the blob at Thor's own checkout HEAD `30d6d60` (2026-08-15), which `git merge-base --is-ancestor` confirms is an ancestor of repo HEAD `7084b2cf`. The rest are settled by naming the superseding commit in `30d6d60..HEAD`, or by a control-bracketed object-store probe. Per-file table in §5 and `raw/A13_DRIFT_TABLE.json`. |
| **Does any drift touch code the live refav1 run imports?** | ⛔ **The premise is stale — there is no live refav1 run** (finished 2026-09-04; 0 python processes on 3 independent probes; GPU 0 %). Asked retrospectively: **YES — 24 of the 130 closure files are real drift and a further 13 do not exist on Thor at all**, so the finished `refav1-b1-v72-ep3-speed` arm ran a **materially older** stack (`refa_v1.py` 1,885 lines vs the repo's 3,030). Consequence in §9: a re-run from today's repo is **not** a replication of that arm. |

---

## 1. ⛔ The brief's premise is stale: there is NO live refav1 run on Thor

The task and `BACKLOG.md` both treat this as urgent because *"refav1 IS TRAINING ON THOR RIGHT NOW
out of that very checkout."* That is **INHERITED and FALSE as of 2026-09-07 20:19 UTC**.

**MEASURED** (`raw/thor_state_probe.txt`), with three independent process probes, because
CLAUDE.md's rule is that absence found at one location is not absence:

| probe | mechanism | result |
|---|---|---|
| `ps -eo args \| grep -ci "pytho[n]"` | process table + text filter | **1** — and that one is the probe's own `bash -c` wrapper, whose argv contains the word. ⚠️ This is the echoed-command trap in miniature: a filter that shares a token with its own command line matches itself. It is reported here as **not evidence**. |
| `pidof python3 python3.10 python3.12 python` | kernel, by executable name | **PIDOF_EMPTY** |
| scan of `readlink /proc/*/exe` for `*python*` | kernel, by inode | **no rows** |

GPU utilisation **0 %**; load average **0.20** on a 23-day uptime. The cron job
`*/30 * * * * refav1_drift_check_v3.py` has been writing `trainer=0 sup=0` into
`experiments/refav1-b1-v72-ep3-speed/drift_check.log` continuously — last row
**`2026-09-07T20:00:01Z step=21100 trainer=0 sup=0 alarm=0`**.

**The run finished three days ago.** `experiments/refav1-b1-v72-ep3-speed/summary.json` carries the
done-marker: `{"done": true, "final_step": 21109, "target_steps": 21109, "epoch": "complete",
"params": 182459701, "drift_alarms_in_training_signal": 0}`, written 2026-09-04; the checkpoint's
own mtime is 2026-09-04 02:55.

⭐ **This does not make A13 less important, it changes WHICH question it answers.** The live-code
question becomes a **retrospective** one — *what code produced the banked `refav1-b1-v72-ep3-speed`
result?* — plus a **pre-launch** one — *what must be shipped before the next launch from this
checkout?* Both are answered in §5.

⚠️ A second stale premise, same family: `experiments/refav1-b1-v72-1ep-21109/` — the directory the
brief names — was last written **2026-09-02 23:47** and carries a `DRIFT_ALARM` file. Per the
`ep3-speed` done-marker's own note, that alarm is a **known false positive** of a supervisor whose
run had no done-marker. The current arm is `refav1-b1-v72-ep3-speed`, not `-1ep-21109`.

---

## 2. What Thor's checkout actually is

**MEASURED** (`raw/thor_state_probe.txt`, `raw/thor_head_blobs.json`):

| fact | value |
|---|---|
| host / path | `tanitad-thor-wifi` : `/home/nvidia/TanitAD` — a **real git checkout**, `.git` present |
| branch | `agent/arch-inf-20260803` (the same branch as the repo) |
| checkout HEAD | **`30d6d601cb0589b1cbf2f6f3f99241da8c548af5`** — `2026-08-15 23:06:46 +0200` |
| `core.autocrlf` | **unset** ⇒ the checkout wrote blobs verbatim; Thor's files are LF |
| repo HEAD at analysis time | **`7084b2cf50615bbf3fdae48ef4e82d8be43899d2`** |
| is the checkout HEAD an ancestor of the repo HEAD? | ⭐ **YES** — `git merge-base --is-ancestor 30d6d60 7084b2cf` → rc 0 |
| how far behind | **1,273 commits** (`git rev-list --count 30d6d60..7084b2cf`) |
| tracked files at the checkout HEAD | **5,435**, and **all 5,435 still present** in the working tree — nothing deleted |
| of those, byte-identical to their own HEAD blob | **5,401** |
| of those, changed since the checkout (shipped or edited on Thor) | **34** |
| untracked, not gitignored | **35** |
| untracked and gitignored | 163 |

⚠️ **The repo HEAD moved during this session** (`cb84c07c` → `7084b2cf`, another agent committed).
Every number below is pinned to **`7084b2cf`** and is stated as such; re-running against a later
HEAD will legitimately give different counts.

⭐ **That ancestor relation is the single fact that makes A13 answerable at all.** The BACKLOG row
says *"direction unknown — a content comparison cannot say which side is newer."* True of a bare
content diff. But the box is a **git checkout whose HEAD is a named, verified ancestor of ours**, so
for every file whose Thor bytes equal its own HEAD blob, the direction is a *positive identification
of a commit*, not an inference. That is 133 of the drift rows, including **all 24** of the row's own
list.

---

## 3. Method, and what its filter does and does not cover

⚠️ *A census is a claim about its FILTER until the filter is stated* (C110). Four probes, deliberately
different mechanisms so that one failing channel cannot look like a second sample:

1. **Thor filesystem census** — `find` + `md5sum` (raw **and** CRLF-normalised) + `stat` over
   `/home/nvidia/TanitAD`, excluding `.git/`, `__pycache__/`, `_pod_backup/`, the pytest/mypy/ruff
   caches and `node_modules/`, limited to source + artifact suffixes under 2 MB. **4,684 files in
   34 s.** → `raw/thor_census_md5.json`
2. **Thor git census** — `git ls-tree -r -z HEAD` (path → checkout-HEAD blob) plus
   `git hash-object --stdin-paths` (path → **working-tree** blob) in two single-process calls.
   **5,435 paths, all present.** → `raw/thor_head_blobs.json`, `raw/scripts/thor_head_probe.sh`
3. **Thor untracked census** — `git ls-files --others --exclude-standard` + `git hash-object`.
   **35 paths.** → `raw/thor_untracked_blobs.json`
4. **Repo side** — a scratch index on **local disk** seeded with `git read-tree 7084b2cf`
   (**2.0 s**), then `git ls-files -s` → **11,410 paths → blob sha**. `git ls-files` on an index is
   one of the enumerations CLAUDE.md admits on this mount.

Both payloads from Thor are framed with a line count, a length and their own md5, with **emit
tokens disjoint from every token searched for client-side** (`QQ…QQ`, `WW…WW`, `ZZ…ZZ`), so a
PTY-mangled or short pull fails loudly instead of producing a short table quietly.

**Comparison is by GIT BLOB SHA on both sides**, not by digest-of-digest: no file content has to be
read from the G: mount, and the comparison is exact.

### ⛔ Traps hit during this work, and what they cost

* **`git log --name-only` over the 1,273-commit range returned 42 commits and exit 127.** The
  one-walk shortcut is inadmissible on this mount; it is **not used**. Per-path
  *range-limited* `git log 30d6d60..HEAD -- <path>` is reliable and fast (**3 s**, rc 0), whereas
  the unbounded `git log --all -- <path>` **did not return in 20 minutes**. That single change is
  what made the direction walk tractable.
* **`git cat-file --batch-check` returned 1 line for a 2-line query and exit 127**, then all 4 for a
  4-line query. Every chunked use here is length-checked with a per-path `rev-parse` fallback.
* **`git archive HEAD` exited 127 after writing exactly 1,080,729,600 bytes** — a plausible-looking
  tar at a record boundary. Discarded, not used.
* **MSYS path conversion mangled `SCAN_ROOT=/home/...` into `.../Program Files/Git/home/...`**, which
  is what `pod_currency_audit.py` failed on from this box. `MSYS_NO_PATHCONV=1` fixes it; the
  purpose-built probes carry their script as base64 in argv so there is nothing to mangle.
* **`launch_closure_audit.py` dies on a cp1252 `UnicodeEncodeError`** printing its own banner, and
  the crash happens **after** the closure is computed but **before** the `--json` is written — so it
  looks like a tool failure and is a lost artifact. `PYTHONIOENCODING=utf-8` fixes it.

⚠️ **Timing context, so the numbers above are not over-read as properties of the mount alone:** the
dev box was carrying **six concurrent `pytest -q` runs, two `refcv3_arm.py` eval arms and two
`guard_mutation_audit.py` jobs** from other agents throughout. The *failures* (exit 127, truncation)
are real and reproduce; the *durations* are worst-case.

---

## 4. ⛔ The correction I had to make to my own first pass — CRLF

A git blob sha is over **raw bytes**. **MEASURED: 14 of the 4,684 files in the Thor census carry
CR** — `build_obstacle_join.py` has CR = LF = 899, i.e. every line — because they were shipped from
the **Windows dev box**. For those 14 a raw-sha comparison scores a **line-ending difference as
drift**. That is defect 3 of the 2026-08-18 `pod_git_drift.py` repair (*"the repo tree is CRLF,
every box is LF ⇒ ~94 % of the rows it would print as drift were artifacts"*), **reintroduced by my
own choice of shas over digests**, and it was about to be reported as a stranding violation on the
refav1 trainer.

Thor recomputed the LF-normalised blob sha for exactly those 14
(`tr -d '\r' | git hash-object --stdin`, `raw/thor_lf_blobs.txt`); **every row below uses that as
the effective blob.**

⛔ **And the correction had a second edge I got wrong on the first attempt.** Having identified the
14, I then probed the object store with the **raw** sha for the ones the LF probe had not covered —
which asks *"were these exact CRLF bytes ever committed?"*, a question whose answer is always no.
It reported `stack/tanitad/data/parity.py` — the file a safety oracle loads — as **content that
exists nowhere in git**. It was caught by a **cross-check that was derived differently**: a
per-path history walk, run for its own reasons, returned `REPO_NEWER_COMMIT_UNIDENTIFIED` for the
same path, and the disagreement was the signal. Re-probed on the **LF** form with a same-breath
control, `parity.py` → `665e585c` is **PRESENT**. All 14 LF forms are present, controls OK on every
probe (`raw/lf_objstore_all14.txt`). After the fix the cross-check reads **12 rows compared, 12
agreed, 0 disagreed.**

⭐ *The lesson is the one CLAUDE.md already carries in another costume: **a cross-check must be
derived independently of the value it checks.** Re-running the raw-sha probe would have "confirmed"
the wrong answer every time; a differently-derived probe did not.*

⭐ **The control that makes "repo blobs are LF" admissible is n = 1,445, not a 3-file sample.**
1,445 Thor files are *byte-identical* to their repo HEAD blob, and every Thor file outside those 14
is LF. A CRLF blob cannot be byte-equal to an LF file, so those 1,445 repo blobs are provably
LF-clean. (My first pass had sampled three files and read `CR_bytes=0`; that was true and far too
thin to hang a 165-row correction on.)

---

## 5. The numbers — measured, against the row's inherited 24

### 5.1 Verdicts — every drifted path at repo HEAD `7084b2cf50`

| verdict | n | meaning |
|---|---|---|
| `REPO_NEWER` | **140** | Thor holds a **named committed** version the repo has superseded — the repo is authoritative |
| `REPO_NEWER_COMMIT_UNIDENTIFIED` | **16** | Thor's exact content **is** a blob the repo holds (control-bracketed probe), so it is committed; repo HEAD carries a **different** blob at that path, so the repo has moved on. The per-path walk that would *name* the superseding commit did not reach these rows before it was stopped — direction settled, revision not named |
| `EOL_ONLY` | **9** | ⚠️ **not drift** — CRLF-vs-LF only, from a file shipped off the Windows dev box |
| | | |
| **raw-sha drift rows** | **165** | before the CRLF correction |
| **REAL drift** | **156** | raw-sha drift minus the 9 EOL-only rows |

### 5.2 Real drift by directory

| directory | n |
|---|---|
| `stack/scripts` | 36 |
| `stack/tests` | 27 |
| `stack/tanitad` | 26 |
| `taniteval/taniteval` | 12 |
| `Paper/figures` | 5 |
| `taniteval/tests` | 5 |
| `stack/experiments` | 4 |
| `tools/tests` | 4 |
| `taniteval/tools` | 3 |
| `stack/ops` | 2 |
| `.gitignore` | 1 |
| `Benchmarks & Eval/LEADERBOARD.md` | 1 |

### 5.3 ⛔ The rows inside the refav1 import closure (n = 29, of which 24 are real drift)

| file | verdict | tracked on Thor | CRLF-shipped |
|---|---|---|---|
| `taniteval/taniteval/pseudosim.py` | `REPO_NEWER` | yes |  |
| `taniteval/taniteval/driving.py` | `REPO_NEWER` | yes |  |
| `stack/tanitad/refs/refc_tactical.py` | `REPO_NEWER` | yes |  |
| `stack/tanitad/refs/refc.py` | `REPO_NEWER` | yes |  |
| `stack/tanitad/refs/refa_v1_plan.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | **no (shipped)** |  |
| `stack/tanitad/refs/refa_v1.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | **no (shipped)** |  |
| `stack/tanitad/models/vocab_v7.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | **no (shipped)** |  |
| `stack/tanitad/models/v6.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | yes |  |
| `stack/tanitad/models/predictor.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | yes |  |
| `stack/tanitad/models/nav_conditioning.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | **no (shipped)** |  |
| `stack/tanitad/models/metric_dynamics.py` | `REPO_NEWER` | yes |  |
| `stack/tanitad/models/kinematic.py` | `REPO_NEWER` | yes |  |
| `stack/tanitad/models/agent_slots.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | **no (shipped)** |  |
| `stack/tanitad/data/v7_labels.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | **no (shipped)** |  |
| `stack/tanitad/data/refav1_loader.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | **no (shipped)** |  |
| `stack/tanitad/data/parity.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | yes | yes |
| `stack/tanitad/data/comma2k19.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | yes |  |
| `stack/scripts/train_v6_staged.py` | `REPO_NEWER` | yes |  |
| `stack/scripts/train_stage_a.py` | `REPO_NEWER` | yes | yes |
| `stack/scripts/train_p8_occupancy.py` | `REPO_NEWER` | yes |  |
| `stack/scripts/s2_labels.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | **no (shipped)** |  |
| `stack/scripts/refb_train.py` | `REPO_NEWER` | yes |  |
| `stack/scripts/refb_labels.py` | `REPO_NEWER` | yes |  |
| `stack/scripts/refa_v1_train.py` | `REPO_NEWER_COMMIT_UNIDENTIFIED` | **no (shipped)** | yes |
| `stack/tanitad/models/sigreg.py` | `EOL_ONLY` | yes | yes |
| `stack/tanitad/data/v2_dataset.py` | `EOL_ONLY` | yes | yes |
| `stack/tanitad/data/bev_raster.py` | `EOL_ONLY` | yes | yes |
| `stack/scripts/train_v58f_unicycle_head.py` | `EOL_ONLY` | yes | yes |
| `stack/scripts/train_flagship4b.py` | `EOL_ONLY` | yes | yes |

### 5.4 Rows where Thor could be authoritative: **NONE**

⭐ **Not one drifted path on Thor holds content the repo lacks.** Every row resolves either to a committed version the repo has superseded, or to a line-ending difference.



### 5.1 Where the row's "24" came from, and why it is not wrong so much as *dated*

The 24 is **INHERITED** from `…/2026-08-18-drift-instrument-hardening/DRIFT_INSTRUMENT_HARDENING.md`
escalation 3. I recovered the exact list from that package's own raw JSON
(`raw/drift_thor_after.json`, `hosts["tanitad-thor-wifi"]`, rows under `/home/nvidia/TanitAD/` with
verdict `DRIFTED`): **24 rows, and they reproduce exactly** — `stack/scripts` 9, `stack/tests` 8,
`taniteval/taniteval` 3, `taniteval/tests` 2, `taniteval/tools` 1, `Paper/figures` 1.

⚠️ Two things about the row's own text:
* Its breakdown says **`taniteval/* ×5`**; the source JSON has **6** (3 + 2 + 1). As printed the
  row's four categories sum to **23**, not the 24 it claims. Minor, but it is the kind of drift the
  row itself exists to catch.
* ⛔ **No banked artifact in the sibling `2026-08-18-thor-closure-audit/` package contains 24.** Its
  five JSONs report `DRIFT` counts of **3, 2, 0, 0** and (in the hardening package)
  **4** — a reader looking there for the A13 evidence finds a different, smaller number and no way
  to reconcile it. **The 24 is a `pod_git_drift.py` (box→repo) count, in a different package; the
  0–4 are `launch_closure_audit.py` (repo→box, closure-only) counts.** They answer different
  questions. Recorded here because the row cites neither instrument, and *a number without its
  instrument is how the wrong package gets opened.*

### 5.2 Status of those exact 24 today

**All 24 are still drifted at repo HEAD `7084b2cf`, and all 24 are settled by E1** — Thor's bytes
*are* the blob at Thor's own checkout HEAD `30d6d60`, a verified ancestor. ⇒ **THE REPO IS
AUTHORITATIVE FOR ALL 24, with no residual uncertainty.** Nothing on Thor's side of those 24 is at
risk, and none of them is on the refav1 launch path.

---

## 6. ⛔ The class my own first instrument could not see — and it is the one that matters

Stage 1 drove off Thor's `git ls-tree HEAD`, i.e. **tracked paths only**. The entire refav1 stack
was shipped onto Thor **after** the 2026-08-15 checkout and is therefore **untracked there** —
including `stack/scripts/refa_v1_train.py`, the launch's own entry point. A tracked-only comparison
is blind to exactly the files the experiment ran.

**MEASURED** — of Thor's 35 untracked, non-ignored files: **13 IDENTICAL to repo HEAD, 12 DRIFT,
10 absent from repo HEAD** (`.NEW` / `.NEW4` / `.PRE_EPOCH2` / `.PRE_SKIP` snapshots of the refav1
sources — Thor-local editing scratch). **9 of the 12 drifted untracked files are inside the refav1
import closure.**

⭐ *This is the same family as C110 (`pod_git_drift.py` scanned `.py`/`.sh` only and its "45" was a
claim about its filter). The lesson repeated itself one layer down: **an enumeration inherited from
the box's own git is a claim about what the box has COMMITTED, not about what it RAN.***

---

## 7. The refav1 import closure — what the finished arm actually ran

**Entry point, read from the launcher rather than guessed** (`/home/nvidia/sup_refav1_v5.sh:34-35`):

```
PYTHONPATH=/home/nvidia/TanitAD/stack PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 \
nohup .../venvs/tanitad-train/bin/python "$STACK/scripts/refa_v1_train.py" ... 200>&- &
```

`launch_closure_audit.py --entry stack/scripts/refa_v1_train.py` gives a closure of **130 files**
(39 eager · 91 deferred-only · 4 guarded-only), `raw/refav1_import_closure.json`.

⚠️ **Scope statement.** That closure is computed from the **repo's** sources, so it is an
*over-approximation* of what Thor imported. It is admissible as the frame because the **import sets
of Thor's and the repo's `refa_v1_train.py` are identical** (13 = 13, zero either-only, by AST
comparison), so the root set is the same. Where the two trees diverge deeper, the differences are
enumerated below rather than assumed away.

### Closure health on Thor

| | raw blob sha | after the CRLF correction |
|---|---|---|
| identical to repo HEAD | 88 | **93** |
| differ | 29 | **24** (5 of the 29 are CRLF-only) |
| **absent from Thor entirely** | 13 | **13** |
| total | 130 | 130 |

**The 13 absent** — confirmed by two independent probes (Thor's git index, and the `find`-based
filesystem census, which never consults git):

```
stack/tanitad/channel_admissibility.py     stack/tanitad/refs/max_speed_input.py
stack/tanitad/data/rig_projection.py       stack/tanitad/refs/refav1_lon_cost.py
stack/tanitad/effective_weights.py         stack/tanitad/refs/refc_agents.py
stack/tanitad/models/_gradreach.py         stack/tanitad/refs/refc_sampler.py
stack/tanitad/refs/feasible_decode.py      stack/tanitad/refs/refc_v3.py
stack/tanitad/refs/goal_point.py           stack/tanitad/refs/tac_goal_head.py
stack/tanitad/train/intrain_eval.py
```

These are modules the **repo** has and Thor never received. They are **not** a defect in the
finished run — Thor's older `refa_v1.py` does not import them — but every one of them is a
`MISSING_REMOTE` for the **next** launch from this checkout.

### The size of the divergence on the load-bearing modules

Thor's copies vs repo HEAD, after CRLF normalisation. `thor-only` / `repo-only` count **non-comment
content lines present in one and not the other** — the descendant check, not "the files differ":

| module | Thor lines | repo lines | thor-only | repo-only |
|---|---|---|---|---|
| `stack/scripts/refa_v1_train.py` | 758 | 805 | **0** | 20 |
| `stack/tanitad/refs/refa_v1.py` | 1,885 | **3,030** | 28 | 534 |
| `stack/tanitad/refs/refa_v1_plan.py` | 340 | 412 | 7 | 45 |
| `stack/tanitad/data/refav1_loader.py` | 474 | 495 | 7 | 26 |
| `stack/tanitad/data/v7_labels.py` | 569 | **926** | 4 | 221 |
| `stack/scripts/s2_labels.py` | 697 | 836 | 9 | 106 |
| `stack/tanitad/models/nav_conditioning.py` | 274 | 337 | **0** | 33 |
| `stack/tanitad/models/vocab_v7.py` | 613 | 643 | 2 | 11 |
| `stack/tanitad/models/agent_slots.py` | 741 | 753 | 5 | 6 |

⚠️ Every "thor-only" line above was inspected by hand and **every one is a re-wrap** — an `__all__`
list broken differently, a docstring reflowed, a signature split across lines. **No module carries
content on Thor that the repo lacks.** `refa_v1.py` alone is **1,145 lines shorter** on Thor.

---

## 8. ⇒ Is anything STRANDED on Thor? **No** — and that is a measured claim, not a hope

| class | n | is it recoverable from git? |
|---|---|---|
| tracked, absent from repo HEAD (the `TanitAD Research Hub/` → `Research Lab/` rename, `Ressources/`, `DECISIONS.md`, `PROJECT_STATE.md`) | **3,837** | ⭐ **YES — all 3,837, with no exceptions.** Every one is byte-identical to the blob at `30d6d60`: `git cat-file blob 30d6d60:<path>`. Of the 3,817 `Research Hub` paths, 2,023 already exist byte-identical under `Research Lab/`, 37 exist with different content, 1,757 have no counterpart — but all 1,757 are in history. |
| Thor's `refa_v1_train.py` and the other CRLF-shipped closure files | 14 | ⭐ **YES.** Their **LF form** is a committed blob — verified for the three REAL_DRIFT ones: `refa_v1_train.py` → `ff18515035da`, `train_stage_a.py` → `6f7389722f58`, `build_obstacle_join.py` → `ee43cbf2235c`, each `git cat-file -e` PRESENT with a same-breath control that also resolved. |
| `.NEW` / `.NEW4` / `.PRE_EPOCH2` / `.PRE_SKIP` editing snapshots | 10 | ⭐ **YES — all 10.** On raw blob shas, 7 of 10 resolved and **3 did not** (`refa_v1_train.py.NEW4`, `refa_v1_train.py.PRE_EPOCH2`, `refa_v1.py.PRE_EPOCH2`) — and I was one paragraph from reporting those as the box's only unique content. ⚠️ **All three turned out to be CRLF** (CR = 758 / 422 / 1,400) and their **LF forms are all PRESENT** (`ff18515035da`, `62fc09998170`, `7bab551e39ed`, each with a same-breath control that resolved). `refa_v1_train.py.NEW4`'s LF form is literally the same blob as the live trainer's. ⛔ These 10 files are outside the census's suffix filter (`.NEW4` is not `.py`), so their CRLF status had to be asked for separately — **the filter hid the very property that decided the verdict.** |

⭐ **⇒ NOTHING on Thor is unique. Not one file on that box holds content the repo does not hold.**
The code that trained `refav1-b1-v72-ep3-speed` **is** in git. What is *not* in git is the exact
**byte sequence** that ran, because it was shipped with CRLF and only the LF form was committed —
a reproducibility footnote, not a loss.

---

## 9. Consequence for how the banked refav1 result must be read

`refav1-b1-v72-ep3-speed` (21,109 steps, 182,459,701 params, finished 2026-09-04) was produced by
**Thor's 2026-09-02/03 versions** of the refav1 stack — not by the repo's current ones. In
particular its `refa_v1.py` was **1,885 lines against the repo's 3,030**, and the 13 modules listed
in §7 did not exist on the box at all.

⇒ Any statement of the form *"arm X was trained with the current `refa_v1.py`"* is **false for this
arm**, and a re-run from today's repo is **not** a replication of it. That is the same class as the
`step_s` and cylindrical-FOV traps: a true fact quoted outside its scope. The registry row for this
arm should carry the code provenance, not just the flags.

---

## 10. ⇒ What the Master Mind must do

1. ⛔ **Nothing needs rescuing from Thor, and nothing on Thor needs overwriting today.** The box is
   idle; leaving it exactly as it is costs nothing and preserves the ep3-speed provenance.
2. ⭐ **Before the NEXT launch from `/home/nvidia/TanitAD/stack`, ship the closure and gate on it.**
   The checkout is 1,273 commits behind; **29 of the 130 closure files differ and 13 do not exist on
   the box.** The gate is already written:
   ```
   python stack/scripts/pod_currency_audit.py --host tanitad-thor-wifi \
       --pod-root /home/nvidia/TanitAD/stack --subtree stack --all-files
   ```
   ⚠️ It fails from this dev box with `bash: Files/Git/home/...: No such file or directory` unless
   `MSYS_NO_PATHCONV=1` is set — MSYS rewrites the `SCAN_ROOT=/home/...` argument. **This is a
   one-line fix in the tool** (`pod_scan()` builds `"SCAN_ROOT=%s bash -s" % root` as a single argv
   element) and it is the reason the purpose-built probes here carry their script as base64.
3. ⛔ **Do not "sync Thor" wholesale.** Nothing on the box needs rescuing, but a blanket ship would
   destroy the byte-level provenance of the finished arm for no gain. Ship the **closure**, not the
   tree — and only when a launch is imminent.
4. **Record the code provenance of `refav1-b1-v72-ep3-speed` in `MODEL_REGISTRY.md`** (§9).

### Proposed replacement for `Project Steering/BACKLOG.md` row A13

> | A13 | ~~**Adjudicate the 24 DRIFT rows in Thor's live `/home/nvidia/TanitAD` checkout**~~ ⏹ **DONE 2026-09-07, 0 GPU** (`TanitAD Research Lab/Tools & DevEnv/Research/2026-09-07-thor-checkout-drift-a13/`) | ⭐ **Direction was NOT unknown — the box is a git checkout whose HEAD `30d6d60` (2026-08-15) is a VERIFIED ancestor of ours, 1,273 commits back**, so for 133 of the rows the direction is a positive identification of a commit, not an inference. **THE REPO IS AUTHORITATIVE ON EVERY ROW; nothing on Thor is newer.** The count is **156 real drift** at HEAD `7084b2cf` (raw-sha drift 165 − 9 that are **CRLF-only**, from files shipped off the Windows box), not 24; all 24 of the original rows are still drifted and all 24 are settled. ⛔ **The premise was stale: refav1 is NOT training on Thor** — it finished 2026-09-04 (`summary.json done:true, final_step 21109`), 0 python processes on 3 independent probes, GPU 0 %. ⛔ **The finished arm ran Thor's OLDER refav1 stack** (`refa_v1.py` 1,885 lines vs the repo's 3,030) and **13 closure modules do not exist on the box** ⇒ a re-run from today's repo is not a replication. ⭐ **NOTHING is stranded — not one file on Thor holds content the repo lacks** (all 3,837 repo-absent files are `30d6d60` blobs; every CRLF-shipped file's LF form is a committed blob, including all 10 `.NEW`/`.PRE_*` snapshots). ⇒ new work item **A13b: ship the closure before the next Thor launch** (29/130 differ, 13 absent) and fix `pod_currency_audit.py`'s MSYS path mangling, which makes the programme's own pre-launch gate unrunnable from the dev box |

---

## 11. Deliverable manifest

⭐ **Everything below is `repo:` and STAGED. Nothing lives only on Thor or only in a scratchpad.**

| artifact | where it lives | note |
|---|---|---|
| `RESULT.md` (this file) | `repo:` **staged** | the adjudication |
| `raw/A13_DRIFT_TABLE.json` | `repo:` **staged** | ⭐ **the machine-readable drift table** — one row per drifted path with verdict, evidence string, effective blob, closure flag |
| `raw/drift_stage1_tracked.json` | `repo:` **staged** | the 5,435-path tracked comparison (slimmed; the note says exactly what was omitted and why) |
| `raw/thor_head_blobs.json` | `repo:` **staged** | ⚠️ **irreproducible box-state measurement** — Thor's checkout-HEAD blob and working-tree blob for all 5,435 tracked paths |
| `raw/thor_census_md5.json` | `repo:` **staged** | ⚠️ **irreproducible** — the 4,684-file filesystem census, raw **and** CRLF-normalised md5 |
| `raw/thor_untracked_blobs.json` | `repo:` **staged** | the 35 untracked paths + blob shas |
| `raw/thor_lf_blobs.txt` | `repo:` **staged** | LF-normalised blob shas for the 14 CRLF files (the §4 correction) |
| `raw/objstore_probe_controlled.json` | `repo:` **staged** | control-bracketed object-store presence on RAW shas, 42 rows, 0 control failures. ⚠️ **The raw sha is the WRONG operand for a CRLF file** — kept because it is the measurement the §4 correction was made against |
| `raw/lf_objstore_all14.txt` | `repo:` **staged** | ⭐ the corrected probe: all 14 CRLF files' **LF** forms against the object store, each bracketed by a control — 14/14 PRESENT, 14/14 controls OK |
| `raw/history_walk_partial.log` | `repo:` **staged** | the per-path range-limited history walk as far as it got (12 rows) before it was stopped — the independent cross-check that caught the `parity.py` error |
| `raw/lfform_objstore.json` | `repo:` **staged** | the LF form of the 3 REAL_DRIFT CRLF files IS a committed blob |
| `raw/snapshot_lf_probe.txt` | `repo:` **staged** | the 3 apparently-unique `.NEW4`/`.PRE_EPOCH2` snapshots re-probed after the CRLF correction — all three LF forms present |
| `raw/refav1_import_closure.json`, `raw/refav1_closure_files.txt` | `repo:` **staged** | the 130-file closure of `refa_v1_train.py` |
| `raw/thor_state_probe.txt` | `repo:` **staged** | the live-state probe log (3 process probes, GPU, cron, done-marker, launcher, git state) |
| `raw/scripts/*.py`, `raw/scripts/thor_head_probe.sh` | `repo:` **staged** | every probe and the assembler, so both the measurement and this document are re-derivable: `thor_head_probe.sh`, `decode_thor_head.py`, `a13_scan.py`, `adjudicate_stage1.py`, `slim_stage1.py`, `a13_final.py`, `build_table.py`, `assemble_result.py` |

**Nothing was written to Thor.** No file on Thor was modified, moved, or deleted; the checkout is in
exactly the state it was found in.

### ⇒ Escalation (headline, not a footnote in a README)

1. ⛔ **`Project Steering/BACKLOG.md` A13 needs the replacement row in §10** — I did not edit it, per
   the brief.
2. ⭐ **A new item is needed: ship the refav1 closure to Thor before the next launch** (29 differ,
   13 absent), and **fix the MSYS path mangling in `stack/scripts/pod_currency_audit.py`**, which
   makes the programme's own pre-launch gate unrunnable from the dev box.
3. **`MODEL_REGISTRY.md`** should carry the code provenance of `refav1-b1-v72-ep3-speed` (§9).

