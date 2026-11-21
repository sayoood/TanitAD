# Pod ↔ repo currency audit — `tanitad-refcv3`, 2026-09-04

**Question.** *Is the box running the repo?* Not "did the files arrive" (presence), not "are the
bytes intact" (md5), not "does it import" (loading) — **is every file the version that is in git.**

**Answer.** Mostly, and the exceptions were expensive-looking. The live 40,284-step run's own
import closure was **39/41 current**; the two stale files were on paths the live trainer never
executes. But one file outside that closure — the **anchor-vocabulary builder** — was missing a
refusal guard added *the same morning*, and that guard exists specifically to prevent a repeat of
the failure that burned refcv3's 40,284 steps.

Host `tanitad-refcv3` (69.30.85.211:22001) · pod tree `/workspace/TanitAD/stack` ·
repo ref `HEAD` = `0f6d5ff1a2acd71f8c26f3591e3ad6cd246d5b7f` on `agent/arch-inf-20260803`.
Live run `refcv4b-b1-v72-40k` (`refc_v3_train.py --arm hier --size base … --steps 40284`),
started 2026-09-04 11:40 UTC, at step ~1,350 throughout this audit, **never disturbed**
(7 processes + 1 supervisor before and after; `train.stderr.log` 0 bytes throughout).

---

## 0. The one fact that frames everything

⛔ **THE POD HAS NO GIT REPOSITORY.** `git -C /workspace/TanitAD rev-parse HEAD` returns nothing —
there is no `.git` at all. `MEASURED`.

So there is no commit to compare, no `git log` to read, and **no way for anyone to ask the box what
version it is on.** Every file arrived by ship, and the only evidence of currency is content. This
is *why* drift here is silent: not because someone forgot to check, but because until now the check
did not exist.

It also means CLAUDE.md's warning is not merely a caution here, it is unconditional: a chain step
doing `git fetch && git checkout` on this pod would hang on the fetch and, if the fetch failed while
the checkout ran, destroy every shipped file. **The audit therefore never runs git on the pod.**

---

## 1. The drift map (`MEASURED`, raw JSON banked)

`.py` sweep, 830 selected paths (skipping `__pycache__`, `.pytest_cache`, `results`).
Pod tree holds 767 `.py` of 888 files total.

| state | n | meaning |
|---|---:|---|
| `IDENTICAL` | **738** | pod bytes == HEAD blob (line-endings normalised — pod files are CRLF, git blobs LF; that is a ship artefact, not staleness) |
| mismatched | **14** | pod bytes differ from HEAD — dissected in §2 |
| `POD-MATCHES-DIRTY` | **15** | pod == the **uncommitted worktree**, not HEAD |
| `REPO-ONLY` | **63** | in git, absent from the box |
| `POD-ONLY` | **0** | nothing on the box is missing from git |

`POD-ONLY = 0` is the good news and it is worth stating plainly: **nothing of value is stranded on
this pod.** The box→repo direction (`pod_git_drift.py`) is clean. It is the repo→box direction that
was never being asked.

### ⚠️ `POD-MATCHES-DIRTY` is a rename sweep that was never committed

All 15 are the PI-binding `TanitAD Research Hub/` → `TanitAD Research Lab/` rename (2026-08-27),
present in the worktree and on the pod but **not in HEAD**. Verified on
`scripts/refc_train.py`: LF-normalised diff = **2 lines**, a docstring path string. Cosmetic, zero
functional blast radius — but it is an uncommitted sweep living in one worktree, which is the
stranded-work class. **Escalated, not fixed here** (it is not this agent's file set).

### ⚠️ `REPO-ONLY` is not automatically benign — and the `.py` filter hid the proof

`tanitad/data/v72_eval_clip_digests.json` is **at HEAD and was absent from the box** — and it is the
*data* that a safety oracle in `parity.py` loads. A `.py`-only sweep cannot see it. That is why
`--all-files` is now a documented per-box step and not an option nobody uses.

---

## 2. Blast radius, ranked — a stale file only matters if something reads it

The live trainer's import closure was computed **on the pod, from the pod's own bytes**, by AST walk
(no torch, no GPU): **41 files**, banked at `raw/live_import_closure_refc_v3_train.json`.
39 were `IDENTICAL`. Ranking is by *what executes it*, not by how stale it is.

### Rank 1 — in the LIVE closure (2 files)

| file | state | does the live run execute it? |
|---|---|---|
| `tanitad/data/parity.py` | behind by 49 lines | **imported and called** — but only `assert_v2_parity_cache` |
| `tanitad/models/metric_dynamics.py` | behind (function body rewritten) | **imported, never called** by this trainer |

**`parity.py`** — the pod was missing §10d entirely: `V72_EVAL_DIGESTS_PATH`,
`v72_eval_clip_digests()`, `clips_in_v72_eval()`. That block is the **v7.2 eval-split membership
oracle**, and its own docstring names the hazard: the B1 train cache holds 4,713 clips and **141 of
the v7.2 eval split's 147 records have their pixels inside it** — a v7 trainer pointed at that cache
with no exclusion trains on the evaluation split, nothing crashes, and every later T1 number is
quietly contaminated.

⭐ **PROVEN SAFE TO SHIP, and the proof is the point.** HEAD is a **strict superset**: the pod's
2,447 lines are byte-for-byte the first 2,447 lines of HEAD — `md5(common prefix)` =
`bcab2ed98939…` on **both** sides, with **zero** pod-only lines. So every line the live run executes
is provably unchanged. That is a positive content assertion, not an eyeballed diff.
The live trainer's only use is `parity.assert_v2_parity_cache(...)` at `refc_v3_train.py:1200`;
it never names `clips_in_v72_eval`. **Shipped** (§3).

**`metric_dynamics.py`** — the pod lacks `bptt_truncate` in `rollout_transitions`. This is **not** a
pure addition: 38 pod-only lines (the old signature and body). Equivalence at the default
`bptt_truncate=0` is asserted by the new docstring and pinned by `tests/test_metric_dynamics_bptt.py`
— but that is `INHERITED`, not something re-verified against this pod's torch. The trainer imports
the module (via `tanitad/models/__init__.py`) and **never calls the function**. Zero benefit to this
run, non-zero behaviour risk on a supervisor relaunch. **ESCALATED, not shipped.**

### Rank 2 — ⛔ a NAMED defect that a future launch from this box resurrects

`bptt_truncate` **appears nowhere on the pod** — not in `metric_dynamics.py`, not in
`train_v6_staged.py`, not in any pod file (`MEASURED`, grep over `tanitad/` and `scripts/`).
The pod's `train_v6_staged.py` is **488,471 bytes against HEAD's 601,585 — 113 KB, ~19 % behind**,
and HEAD names `bptt_truncate` **10 times**, including the `--bptt-truncate` flag, the two
`rollout_transitions(..., bptt_truncate=…)` call sites, and the preflight that refuses an inert value.

So the box is **stale-but-internally-consistent**, which is the worst shape: nothing errors. A v7 /
`train_v6_staged.py` launch from `tanitad-refcv3` runs the O5 rollout with the **full unbounded BPTT
chain** — the MEASURED failure `PREREG_MM_E19`: gnorm **2.1e9**, run killed at step **9,000**. And
an operator who *knows* to pass `--bptt-truncate` gets `unrecognized arguments`, which at least fails
loudly; an operator who does not gets the divergence.

**ESCALATED — do not ship mid-run.** 113 KB of a trainer this pod is not running, whose compatibility
with any future v7 config cannot be verified from here. It must be shipped *as part of* a v7 launch
preflight, with the currency audit as the gate.

### Rank 3 — ⭐ the anchor builder: a loaded gun, and the guard was added the same morning

`scripts/build_refc_anchors.py`, pod **6,707 bytes vs HEAD 9,593** — behind by exactly the guard
committed on 2026-09-04. HEAD is a **PURE ADDITION**: 45 lines added, **0 removed**.

What the pod was missing:

* the MEASURED result that **FPS at the 6 s / 8-slot grid scores 0.7666 oracle-in-vocabulary ADE
  against a straight line's 0.6843** — i.e. *worse than drawing a straight line* — on 19,602 held-out
  windows over the 141 B1-v7.2 eval clips;
* the hard `SystemExit` refusing any `>4`-slot build unless `--i-know-fps-loses-at-6s` is passed;
* the pointer to the validated k-means/slot-normalised builder that scores **0.3796**.

⭐ **Why this outranks the two live-closure files.** The guard's own docstring states the lesson:
*"refcv3 trained 40,284 steps on the synthetic fallback because nobody passed `--anchors`, and a
perfect chooser on that fan still lost to a hold-action control. A VOCABULARY IS A CEILING — no
selector, however good, can pick a trajectory the fan does not contain."* The stale builder on this
box would hand back exactly such a fan, silently, on the very machine where that already happened
once. **Shipped** (§3) — it is outside the live import closure (verified: the closure's only
`scripts/` entry is `refc_v3_train.py`), so live-run exposure is zero and the change can only refuse
work, never alter it.

✅ **This run's own anchors are NOT affected.** `anchors.build.json` shows a **parametric**
13 × 9 = **117** `(a_lon × a_lat)` grid built through
`tanitad.refs.refa_v1_plan.unicycle_paths → models.kinematic.rollout_unicycle`, with a held-out gate
(`oracle_in_vocabulary_ade_0_2s_m` **0.1987**, 4,823 windows / 141 episodes) — the validated refcv4
path, not FPS, and matching the live `--n-anchors 117`. `MEASURED`.

### Rank 4 — the refav1 arm, badly behind

| file | pod | HEAD | shape |
|---|---:|---:|---|
| `tanitad/refs/refa_v1.py` | 1,227 lines | 2,155 lines | **928 lines behind**; 39 pod-only lines (also `REPO-DIRTY`) |
| `tanitad/data/refav1_loader.py` | 455 lines | 496 lines | 56 HEAD-only, 15 pod-only |
| `scripts/refa_v1_train.py` | — | — | mismatched |
| `tests/test_refa_v1_target_space.py` | — | — | mismatched |

Not in the live closure; nothing on this box runs refav1 today. **ESCALATED** — none is a pure
addition, so none can be shipped on a superset proof, and `refa_v1.py` being simultaneously
pod-stale *and* repo-dirty means the three-way state must be resolved in the repo first.

### Rank 5 — dead on this box

`experiments/alpasim-gsplat/{closedloop,openloop}_drive.py` (need a renderer this pod does not run),
`scripts/dinov3_fp8_encode_ship.py`, `scripts/fp8_l2_gate.py`, `tests/test_refc_v3.py`,
`tests/test_v6_s1_multitick.py`. Reported for completeness; no action.

### ✅ The 2026-09-04 launch-agent fix is genuinely on the box

`tanitad/refs/refc_tactical.py` is **byte-identical to HEAD** (22,067 bytes, verified by full-content
pull and `diff -q`, not merely by md5 agreement), as is `scripts/refc_v3_train.py`. The width
assertion `c37fa68` — *"Defect A: a positional contract that was never asserted, feeding the live
anchor prior"* — **is present and running.** The stale-pod problem that opened this audit is closed
for those two files.

---

## 3. What was shipped, and how it was verified

⛔ **The safety rule this obeys.** The live trainer has already imported its modules, so overwriting
a `.py` cannot affect the running process — **but the supervisor relaunches on crash, and a relaunch
imports whatever is on disk at that moment.** Shipping is therefore never free: it changes what the
run *becomes*. Only files with a positive compatibility proof were shipped.

| file | proof | verification at both ends |
|---|---|---|
| `tanitad/data/parity.py` | strict superset; `md5(first 2,447 lines)` = `bcab2ed98939…` on both sides, 0 pod-only lines; live call path `assert_v2_parity_cache` provably unchanged | backup `parity.py.bak.20260904-currency-audit` = `0b91523f314d…` (old); in place `e1923a6be3be…` = **HEAD blob md5**; `ast.parse` OK; **real import executed** (stdlib-only, no torch, no GPU) and the oracle exercised: `v72_eval_clip_digests()` → **147** digests, `clips_in_v72_eval(["fake"])` → 0 |
| `tanitad/data/v72_eval_clip_digests.json` | **new file** — cannot break any import | in place `7903622f3576…` = HEAD blob md5; parses; **147** entries, independently matching the documented v7.2 eval-split size |
| `scripts/build_refc_anchors.py` | **pure addition** (45 added / 0 removed); outside the live import closure | backup `…bak.20260904-currency-audit` = `7b40b580b0f9…`; in place `d00d4cb2dfbc…` = HEAD blob md5; `ast.parse` OK; guard string `i-know-fps-loses-at-6s` present **3×** |

Transport: `xz` + base64 over ssh, decoded to `/tmp`, md5-checked, `ast.parse`-checked, then moved
into place with `mv` (**atomic within one filesystem**, so no partial file can ever be imported).
Every backup is retained on the pod.

**Run health across the whole operation:** 7 trainer processes and 1 supervisor before and after;
`train.stderr.log` **0 bytes** throughout; steps 1150 → 1350 progressing normally.

### Escalated, NOT shipped — with the reason

| file | why not |
|---|---|
| `scripts/train_v6_staged.py` | 113 KB behind; carries the `--bptt-truncate` wiring. Belongs to a v7 launch preflight, not to a mid-run patch of a pod running a different trainer. **Its absence is a live hazard for the next v7 launch from this box — see Rank 2.** |
| `tanitad/models/metric_dynamics.py` | not a pure addition (38 pod-only lines); default-equivalence is `INHERITED` from a docstring + a test not run against this pod. Imported but never called → zero benefit, non-zero relaunch risk. |
| `tanitad/refs/refa_v1.py`, `tanitad/data/refav1_loader.py`, `scripts/refa_v1_train.py` | none is a pure addition; `refa_v1.py` is 928 lines behind **and** repo-dirty — the three-way state must be settled in the repo first. |
| the 15 `POD-MATCHES-DIRTY` files | the uncommitted `Research Hub`→`Research Lab` rename. Cosmetic, and committing another agent's sweep is not this agent's call. |

---

## 4. The checker — `stack/scripts/pod_currency_audit.py`

Reusable, takes a host alias and a subtree, prints the drift map, **exits non-zero on drift**.

```bash
python stack/scripts/pod_currency_audit.py --host tanitad-refcv3            # .py sweep
python stack/scripts/pod_currency_audit.py --host tanitad-refcv3 --all-files \
    --json "<bank>/drift_map.json"                                          # incl. data sidecars
python stack/scripts/pod_currency_audit.py --host thor \
    --pod-root /home/nvidia/TanitAD/stack --fail-on pod-behind
```

It is the **third** drift tool and it answers what neither of the other two can:

| tool | direction | question |
|---|---|---|
| `pod_git_drift.py` | box → repo | what lives only on a box? |
| `launch_closure_audit.py` | repo → box | is *one launch command's* import closure current? |
| **`pod_currency_audit.py`** | repo → box | **is the WHOLE subtree the version that is in git?** |

Design decisions that are load-bearing rather than stylistic:

* **Compares against the ref's blobs, never the worktree** — and reports `POD-MATCHES-DIRTY` as a
  distinct third state, so "the pod runs an uncommitted change" cannot hide inside "the pod is stale".
* **Direction is never guessed.** A mismatch is only `POD-BEHIND` if the pod's bytes equal an
  *ancestor commit's* blob for that path — the report names the commit and the distance. Otherwise
  `POD-DIVERGED`. No `-n` cap on the history walk, because a cap makes "no match" ambiguous between
  "diverged" and "older than the window".
* **Framed transport.** The pod emits `ZZLINES…ZZ` / `ZZB64MD5…ZZ` / `ZZB64LEN…ZZ` and a gzip+base64
  payload; a short or corrupt pull **raises**. The GOTTY PTY drops ~5 of every 14 lines on bulk pulls
  (MEASURED 2026-08-12), and a silently short drift map is worse than none.
* **Disjoint tokens.** Values are computed pod-side and wrapped in opaque markers, so the PTY echoing
  the command back cannot match the filter — the failure MEASURED three times in this programme.
* **Never runs git on the pod** (test-asserted: `"git " not in POD_SCAN_SH`).

Pinned by `stack/tests/test_pod_currency_audit.py` — **14 tests, all passing**, no pod required.

### ⚠️ Two defects the tool had *in this very audit*, both now fixed and both test-pinned

1. **A failed probe was being reported as a verdict.** The first revision called git with
   `check=False`, so a G:-mount flap returned an empty log and **all 14 mismatches came back
   `POD-DIVERGED`** — a confident classification manufactured out of an outage, for files that are
   in fact `POD-BEHIND`. This is CLAUDE.md's rule in tool form: *an empty result is a claim about the
   SEARCH, not about the CONTENT.* Now every git call is `check=True`, a probe that cannot complete
   raises, the row reads **`HISTORY-UNKNOWN`**, and that state **fails the audit by default** —
   a probe that could not run must never be scored as clean.
2. **The retry policy protected only part of the git surface.** `run_git` retried patiently while
   `cat-file --batch` / `--batch-check` went through a bare `subprocess.run`; two audit runs died
   there while `run_git` was riding out the identical outage. *A retry policy is only as good as its
   least-protected call site.* All git now goes through retrying wrappers, and `merge-base
   --is-ancestor` distinguishes rc=1 ("not an ancestor", a real answer) from a mount failure.

⚠️ **A third trap, worth recording because it produced a silent no-op.** The repo file, freshly
written to the G: mount, became cloud-only: `head` reported `Invalid request code` while `ls` showed
its exact byte count — and **`python <that file>` exited 0 with no output at all.** Python running a
file it cannot read looks exactly like a program that chose to print nothing. Same family as the
2026-09-04 note about `.claude/worktrees/` never hydrating: metadata resolves, content does not, and
every tool reports success. The tool is therefore run from a local copy while the mount settles.

---

## 5. Runbook change

`Project Steering/AGENT_OPERATING_STANDARD.md` § *Standing cadence* previously named **two** drift
tools and warned that *"presence proves transfer, md5 proves bytes, a successful import proves
loading — none of them proves currency."* It now names **three**, with the currency audit as a
**pre-launch gate**, and carries the `--all-files` requirement with the evidence for it (the
`v72_eval_clip_digests.json` case). Because that file is quoted verbatim into every subagent brief,
the step travels with every brief rather than living in a doc nobody re-reads.

---

## 6. Deliverable manifest

| artifact | location |
|---|---|
| the checker | `repo:stack/scripts/pod_currency_audit.py` |
| its pins (14 tests) | `repo:stack/tests/test_pod_currency_audit.py` |
| runbook step | `repo:Project Steering/AGENT_OPERATING_STANDARD.md` (§ Standing cadence) |
| this report | `repo:TanitAD Research Lab/Tools & DevEnv/Implementation/incoming/2026-09-04-pod-currency-audit/POD_CURRENCY_AUDIT.md` |
| drift map, `.py` sweep | `repo:…/2026-09-04-pod-currency-audit/drift_map_py.json` |
| drift map, all files | `repo:…/2026-09-04-pod-currency-audit/drift_map_all.json` |
| raw pod md5 table (888 files) | `repo:…/raw/pod_scan_tanitad-refcv3_20260904.tsv` |
| live import closure (41 files) | `repo:…/raw/live_import_closure_refc_v3_train.json` |
| shipped files | `tanitad-refcv3:/workspace/TanitAD/stack/{tanitad/data/parity.py, tanitad/data/v72_eval_clip_digests.json, scripts/build_refc_anchors.py}` |
| pre-ship backups (pod only) | `tanitad-refcv3:…/{tanitad/data/parity.py.bak.20260904-currency-audit, scripts/build_refc_anchors.py.bak.20260904-currency-audit}` |

Nothing produced by this audit lives in only one place, **except the two pod-side backups**, which
are deliberately pod-local: their content is the pre-ship pod state, and both are reconstructible
from git history.
