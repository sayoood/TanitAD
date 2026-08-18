# Thor closure audit — C99's class, measured and closed

**Date:** 2026-08-18 · **Host:** `tanitad-thor-wifi` (`thor6`), `/home/nvidia/TanitAD`
**Evidence class: MEASURED** unless stated. Raw artifacts: `raw/*.json` beside this file.
**Instrument (the deliverable):** `stack/scripts/launch_closure_audit.py` + `stack/tests/test_launch_closure_audit.py`.

---

## 0. Headline

| | |
|---|---|
| **The hand-list C99 shipped** | **3 files** |
| **The computed import closure of the same ladder** | **120 files** (7 entry points) — **40×** |
| **The closure when the T1 / four-families instruments are added** | **134 files** (14 entry points) |
| **Genuinely stale on Thor** | **7 files** — 5 DRIFT, 2 absent entirely (one of them **46,905 B**) |
| **CRLF artifacts that a naive md5 would have called drift** | **47 of 50** apparent-drift rows in the briefed closure — **94 % false** |
| **Final state** | **DRIFT 0 · MISSING_REMOTE 0 · 131/134 modules import for real on Thor**, remaining 3 adjudicated below |
| **Cost to the live run** | **zero** — step 12,800, `step_s` 26.4742 unchanged, PID 25477 never touched |

⛔ **The finding that matters most is not the seven files. It is that the briefed entry-point
list was itself a hand-list, and widening it found three more stale files** — including
`v0_antiecho.py`, 46,905 B, **absent from Thor entirely**, which is the anti-echo instrument the
four-metric-families rule leans on. *A hand-listed dependency set is a guess about what a launch
touches — and so is a hand-listed set of entry points.*

---

## 1. What was actually stale (MEASURED)

All digests below are **`md5_lf`** — md5 after `\r\n` → `\n` and nothing else. Backups of Thor's
originals are at **`/home/nvidia/_thor_backup_2026-08-18-closure/`**, verified by md5 before a byte
of the destination was touched.

| # | file | reach | Thor before | repo | what Thor was MISSING |
|---|---|---|---|---|---|
| 1 | `stack/scripts/train_v6_staged.py` | **eager** | 234,845 B (4,076 ln) | 252,691 B (4,374 ln) | `GATE_APPLICABILITY`, `SEL_GAP_TIER_NOTE`, `UNMEASURED_BY_CONSTRUCTION`, `arm_record`, `probe_applies` |
| 2 | `stack/tanitad/data/anchor_goal.py` | deferred | **absent** | 10,945 B | the whole module |
| 3 | `taniteval/taniteval/loaders.py` | deferred | 7,972 B (158 ln) | 9,793 B (191 ln) | `resolve_labels_v2` |
| 4 | `taniteval/taniteval/rollout.py` | deferred | 19,387 B (363 ln) | 20,130 B | *no symbol change* — body-only drift |
| 5 | `taniteval/taniteval/four_families.py` | deferred | 64,296 B (1,131 ln) | 71,011 B (1,252 ln) | `KAPPA_VERDICT_LADDER`, `_KAPPA_BAND_GLOSS`, `_anti_echo`, `kappa_band`, `kappa_verdict` |
| 6 | `taniteval/taniteval/hierarchy.py` | deferred | 67,783 B (1,257 ln) | 79,712 B (1,445 ln) | `MANEUVER_LABEL_KEY`, `PER_WINDOW_KEYS`, `_per_window` |
| 7 | `taniteval/taniteval/v0_antiecho.py` | deferred | **absent** | 46,905 B | the whole module |

⚠️ **Row 4 is the one to remember.** `rollout.py` differed by 1,116 bytes with **zero top-level
symbol change** — every symbol a hand-check would look for was present. Only the digest saw it.
**A symbol census explains drift; it cannot detect it.**

⭐ **Row 1 is the S-T launch blocker.** `train_v6_staged.py` is the trainer *and* the module
`v6_chain.py` imports the S-T adjudicator from (`assert_stage_precondition`, `build_parser`,
`read_ckpt_provenance`, `_save_ckpt`). Thor's copy predated the five S-T launch-path fixes at HEAD —
`GATE_APPLICABILITY` and `probe_applies` are exactly that machinery. **An S-T launch from Thor's
tree would have run a pre-fix ladder**, which is the pod2 `0f93b98` failure verbatim.

**⛔ Escalation, not a doc note — row 2 is C99 repeating one level down.** `anchor_goal` is imported
lazily at `stack/tanitad/models/v6.py:1825`, behind `cfg.anchor_goal != "none"`. The live S-W run
never takes that branch, so **the box looked perfectly healthy while the file did not exist.** Any
S-T arm configured with an anchor goal would have died *after* the model was built.

---

## 2. ⚠️ The CRLF trap, at scale

This repo's working tree is **mixed**, Thor is LF. Every comparison here carries two digests —
`md5_raw` (bytes as they sit) and `md5_lf` (`\r\n` → `\n`, **nothing else**: no whitespace strip, no
final-newline fixup, no encoding round-trip) — and the verdict names which one decided it.

| briefed closure (120 files) | before ship | after ship |
|---|---|---|
| `SAME` (raw digests equal) | 69 | 72 |
| **`CRLF_ONLY`** (raw differs, LF agrees — **not drift**) | **47** | 48 |
| `DRIFT` (LF digests differ) | **3** | **0** |
| `MISSING_REMOTE` | 1 | **0** |

**A naive raw-md5 comparison would have reported 50 drift rows. Three were real.** That is
**94 % false**, against the 70 % (7 of 10) measured on 2026-08-16 — the artifact rate *rises* with
the size of the compared set, so the trap gets worse exactly as the audit gets more thorough.

⭐ **An arithmetic self-check that settles the classification without trusting the tool.**
`rollout.py` post-ship reads local 20,503 B vs Thor 20,130 B — a difference of **373 bytes**, and the
file is **373 lines**. One `\r` per line, exactly. The `CRLF_ONLY` verdict is not a judgement call.

---

## 3. ⛔ Two instrument failures I hit, both of which produced clean, plausible, WRONG answers

Recording these because each would have shipped as a finding.

### 3.1 MSYS argument mangling reported **120/120 MISSING_REMOTE**

Invoked from Git Bash on Windows, `--remote-root /home/nvidia/TanitAD` reached Python as
**`C:/Program Files/Git/home/nvidia/TanitAD`**. Every remote `isfile` returned False and the audit
reported **every file in the closure missing from Thor** — while the trainer was at that moment
executing `/home/nvidia/TanitAD/stack/scripts/train_v6_staged.py`.

The tell was not in the table; it was in the header line echoing the mangled root. **The finding was
catastrophic, internally consistent, and entirely an artifact** — the `df`-on-a-pod shape: a probe
answering a different question than the one asked.
⇒ Fixed twice over: run with `MSYS_NO_PATHCONV=1`, **and** `_demangle_posix()` now strips it with a
warning, so the tool cannot silently produce that answer again. Pinned by
`test_msys_path_mangling_is_undone`.

### 3.2 My own probe self-matched, twice, in one command

A `ps` filter for `supervise` and `train_v6_staged` matched **my own ssh command line**, which
contains both literals — reporting a supervisor that does not exist and **2** trainers where there
is 1. This is the documented PTY-echo trap in a `ps` costume.
⇒ Re-probed with tokens assembled from `chr()` codes so the command line never contains the string
it searches for, plus explicit exclusion of my own PIDs. **Corrected result: 0 supervisors, exactly
1 trainer (PID 25477).**

⚠️ **The 0-supervisors result is load-bearing**, and it is why shipping row 1 was safe: with no
supervisor there is no process that could relaunch the trainer onto newly-shipped code mid-run.

### 3.3 The `stack` suite reported **4 failures that are a console codepage**, not a regression

`pytest -q` under Git Bash on this dev box: **4 failed, 3857 passed**. The brief's MEASURED figure
was **0 failed**, so this read as a regression I had caused. The tracebacks say otherwise —
every one is `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f` while a **cp1252** reader
consumes a subprocess's stdout containing the repo's own `⛔ ⚠️ ⭐` glyphs.

Discriminating experiment, cheapest available — re-run the same four with UTF-8 IO forced:

```
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 pytest tests/test_e_wc2_sigma_star.py::test_cli_print_contract \
    tests/test_ff_v58f.py::test_tool_REFUSES_the_biased_estimator_by_name tests/test_bev_consumer_fov.py -q
→ 37 passed
```

**All four pass.** The failures are the shell's encoding, not the code, and not the two files I
added. ⇒ **A "green suite" claim on this box is only meaningful with the encoding stated.** Banked
runs here are `PYTHONUTF8=1`: `stack` **3861 passed / 0 failed / 7 skipped / 2 xfailed**
(`raw/suite_stack.txt`), `taniteval` **1136 passed / 0 failed** (`raw/suite_taniteval.txt`), the
`stack` figure including my **19** new tests.

*(Fifth instance tonight of the same family: a probe answering a different question than the one
asked — after the MSYS root, the two `ps` self-matches, and the md5-vs-function gap that opened C99.)*

---

## 4. The reach axis — why a stale box looks healthy

The instrument now classifies every closure member by **how it is reached**:

| | briefed (120) | wide (134) |
|---|---|---|
| **eager** — imported at process start, fails fast and loud | 50 | 60 |
| **deferred-only** — reachable ONLY through a function-level import | **70** | **74** |

**58 % of the launch closure is invisible to any startup check.** Six of the seven stale files were
in that set, and so was C99's `refc_dump_latents` and the `t1_eval.py` disaster (both arms, 40
episodes, 6,844 windows each rolled — then `ImportError` in `analyze()`). **This is the class,
quantified: deferred imports are where staleness hides, because nothing exercises them until the
compute is already paid for.**

---

## 5. Real-import verification — the only sufficient evidence

md5 agreement proves transfer, not function. Every closure module was imported **for real** on Thor
with the training venv and the launch `PYTHONPATH`
(`/home/nvidia/venvs/tanitad-train/bin/python`, `OMP_NUM_THREADS=2`):

| | before | after |
|---|---|---|
| modules imported OK | 116 / 120 | **131 / 134** |
| `tanitad.data.anchor_goal` | ❌ `ModuleNotFoundError` | ✅ **resolved** |

The three remaining failures are **environment**, not drift, and each is adjudicated from source —
**none blocks the S-T launch**:

| module | missing dep | verdict | evidence |
|---|---|---|---|
| `lead_state_gate` | `pandas` | **GUARDED — tolerated by design** | its only closure import site, `stack/scripts/probe_latent_state.py:117-124`, is a `try/except ImportError` with documented fallback constants (`LEAD_LAT_M`, `LEAD_MAX_GAP_M`, `VEHICLE_CLASSES`) and the comment *"fallback for hosts without pandas"* |
| `tanitad.data.physicalai` | `pandas` | **not on the ladder** | reached only from `stack/tanitad/train/train_worldmodel.py:161`, inside `if data == "physicalai"` — the corpus-**build** path. No v6 entry point references `train_worldmodel` at all |
| `tanitad.lake.catalog` | `pyarrow` | **not on the ladder** | reached only from `stack/tanitad/lake/enrich.py:377`, the lake-enrichment path |

⚠️ **I deliberately installed nothing.** `uv pip install <anything>` has twice silently pulled torch
from the default index and left a wheel the driver cannot run; doing that to the **training** venv
beside a live 5-day run to satisfy a guarded import would be a bad trade. **If the PI wants them
anyway, the safe form is `--no-deps`, torch reinstalled last from the pinned index, verified with a
real `conv2d`.** Flagged, not done.

⭐ **The guard classification is itself an anti-false-positive fix.** A direct `import_module()`
probe bypasses a `try/except ImportError` the launch relies on. Before I taught the tool to detect
guards, `lead_state_gate` read as a blocker — **I would have reported a fabricated S-T blocker in an
audit whose entire subject is fabricated findings.**

---

## 6. The live run was not disturbed (MEASURED)

| | |
|---|---|
| PID | **25477**, alive throughout (`kill -0` OK), state `Ssl`, etime 2-00:16 → **2-00:50** |
| step / `step_s` | **12,750 → 12,800 → 12,850**, `step_s` **26.4735 → 26.4742 → 26.4745** — the run kept advancing across all seven ships at an unchanged rate |
| log schema | 258 training rows + **64 spectrum rows** — spectrum rows carry no `loss` field, so this parse selects on `"loss" in row` rather than `.get("loss", 0)`, which would manufacture zeros |
| supervisors | **0** (disjoint-token probe) — no process could relaunch onto new code |
| trainer `PYTHONPATH` | `…/stack:…/stack/scripts` — **`taniteval` is NOT on it** |
| `dump_seam_plan` | **absent from the run's `config.json` args** |

⇒ Six of the seven shipped files are under `taniteval/`, which **the live process cannot import at
all** — confirmed by two independent facts (PYTHONPATH, and the seam-dump flag that is the only
`taniteval` call site in the trainer). The seventh, `train_v6_staged.py`, is the already-loaded
`__main__`; replacing it on disk cannot affect a running interpreter, and with no supervisor there is
nothing to relaunch it.
⚠️ **A manual resume would now resume on the NEW trainer.** If that is ever unwanted, Thor's exact
pre-ship binary is one `cp` away at `/home/nvidia/_thor_backup_2026-08-18-closure/stack/scripts/train_v6_staged.py` (md5 `8b7c91d4…`, 234,845 B).

⚠️ **A latent landmine found while establishing the above, reported not fixed:**
`train_v6_staged.py:3074-3090` imports `SeamDumpError` **inside** the `try` whose `except` catches
`SeamDumpError`. If `taniteval` is off `PYTHONPATH` — which is the live run's exact configuration —
a launch with `--dump-seam-plan` raises `ImportError`, the handler's name is unbound, and **the
trainer dies at a checkpoint boundary** rather than logging "NOT banked". Same family as C99 and the
`t1_eval.py` analysis-time import.

---

## 7. The instrument (this is the integration — it is code, not a description)

**`stack/scripts/launch_closure_audit.py`** — re-runnable, and the next launch runs it instead of
hand-listing:

```bash
# audit only (read-only, zero GPU, ssh -n throughout)
MSYS_NO_PATHCONV=1 python stack/scripts/launch_closure_audit.py \
    --host tanitad-thor-wifi --remote-root /home/nvidia/TanitAD --json raw/closure.json

# ship exactly the DRIFT + MISSING_REMOTE rows, then prove the box can run them
MSYS_NO_PATHCONV=1 python stack/scripts/launch_closure_audit.py \
    --host tanitad-thor-wifi --remote-root /home/nvidia/TanitAD \
    --ship --backup-dir /home/nvidia/_thor_backup_$(date +%F)/ \
    --import-python /home/nvidia/venvs/tanitad-train/bin/python --verify-import
```

What it does that a hand-list cannot:

1. **Closure by AST**, transitively, including **function-level** imports, `importlib.import_module("literal")`, relative imports, and every package `__init__.py` on the path.
2. **Two digests always**, so CRLF can never be reported as drift.
3. **Symbol census on DRIFT rows** — names *what* the box is missing.
4. **Reach classification** (eager / deferred-only) — the risk axis.
5. **Guard detection** (`try/except ImportError`) — so the probe cannot fabricate a blocker.
6. **Backup-first, atomic, LF-normalised ship**, md5-verified against `md5_lf` on both sides.
7. **Real-import verification** on the box with the venv interpreter and launch `PYTHONPATH`.

**Probe hygiene is built in, not remembered:** `ssh -n` on every call; the remote computes its own
answer and emits one opaque `ZZ…ZZ` frame; nothing greps the raw stream for a token the command
contains; payloads travel through a **chunked pusher** because Windows `CreateProcess` caps a command
line at 32,767 chars and fails with a bare `WinError 206` that names no cause (measured — the 4-file
ship died there).

**`stack/tests/test_launch_closure_audit.py`** — **19 tests, all green.** They pin the closure
following function-level imports, CRLF ⇒ `CRLF_ONLY` never `DRIFT`, guard detection (incl. that a
`try/else:` branch is *not* protected), MSYS demangling, `ZZ…ZZ` framing, that the shipped default
entry points and roots all exist, and that the v6 ladder closure stays far larger than a hand-list —
asserting by name that **`refc_dump_latents.py`, the exact file C99 missed, is in the computed
closure.**

### ⛔ Escalations (decisions, not doc notes)

1. **`stack/scripts/pod_git_drift.py` is stale doctrine.** Its `DEFAULT_PODS` are
   `tanitad-pod/-pod2/-pod3/-eval` — **all four are dead machines** (the RunPod fleet was released
   2026-08-15; the fleet is Thor + the dev box). It also matches by **basename anywhere in the repo**
   and does no CRLF normalisation, so on this repo it would report drift on ~half of what it
   inspects. It answers a different question (pod-only files) and is worth keeping, but its defaults
   need repointing at Thor. **Not done here — it is a different tool with a different contract.**
2. **`pandas` / `pyarrow` in `tanitad-train` on Thor** — PI call, safe recipe in §5.
3. **The `SeamDumpError` handler bug** in §6 — a two-line fix (move the import above the `try`), not
   made here because the file is the live trainer's `__main__` and I would rather it land with the
   S-T change than as a surprise edit mid-run.

---

## 8. Deliverable manifest

| artifact | where it lives | notes |
|---|---|---|
| `stack/scripts/launch_closure_audit.py` | `repo:` **staged** | the instrument — new |
| `stack/tests/test_launch_closure_audit.py` | `repo:` **staged** | 19 tests, green |
| `THOR_CLOSURE_AUDIT.md` (this file) | `repo:` **staged** | |
| `raw/closure_audit_before.json` / `_after.json` | `repo:` **staged** | briefed 120-file closure, both sides |
| `raw/closure_audit_wide.json` / `_wide_after.json` | `repo:` **staged** | wide 134-file closure |
| `raw/ship_result.json` / `_wide.json` | `repo:` **staged** | per-file backup md5, post-ship md5, byte counts |
| `raw/import_probe_before.json` | `repo:` **staged** | the pre-ship `anchor_goal` failure, as evidence |
| `raw/suite_stack.txt` | `repo:` **staged** | full `stack` suite output |
| **Thor's pre-ship originals** | `tanitad-thor-wifi:/home/nvidia/_thor_backup_2026-08-18-closure/` | **5 files, ONE place only** — deliberate: they are superseded versions kept as a rollback path, and the current versions are all in git |
| 7 shipped files | `tanitad-thor-wifi:/home/nvidia/TanitAD/…` | all identical to the repo modulo CRLF; nothing on Thor is unique |

**Nothing produced by this work exists in only one place**, except the Thor backup directory, which
is by design a rollback copy of content that git already holds in its newer form.
