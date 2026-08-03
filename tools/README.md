# `tools/` — repo-level dev tooling (agent-facing, not `stack/` model code)

Cross-cutting scripts every TanitAD session/agent runs. These are **not** MVP model
code, so they live at the repo root (not under `stack/`) and are maintained directly
by the Tools & DevEnv agent — no intake round-trip. Stdlib-only, ASCII-clean stdout
(the Windows cp1252 console lesson), OS-agnostic (a `.py` core + a `.ps1` Windows
wrapper; on the pod call the `.py` directly). `gpu_tripwire.py` is the one exception
to stdlib-only — it needs `torch` and the `stack/` package, by definition.

| Tool | What it does | When to run |
|---|---|---|
| `ci_gate.py` / `ci.ps1` | One-command test gate: fails on failure, **collection error**, slow test, wall blow-out, a missing/red tripwire node, a **suite below its manifest floor**, a total-collected count under `--min-total`, or a CUDA parity failure. | **Before every commit/push** (protocol G-E). |
| `gpu_tripwire.py` | CUDA device-parity probes on the real model (encode/imagine CPU-vs-GPU, I2 on device, backward-finite) + the batch-1 encode latency (I8 proxy). | Via `ci_gate --gpu-smoke`, or standalone on any GPU box. |
| `session_guard.py` / `.ps1` | D-026 stranded-work guard: **blocks** on uncommitted hub deliverables; **warns** on uncommitted `stack/`+`tools/` source, unmerged `agent/*` branches vs tip, and stale INTAKE verdicts. | **Session end**, every agent (protocol G-F). |
| `safe_commit.py` | The **only sanctioned commit path**. Never emits the segfaulting `git commit -- <pathspec>` form; instead makes the whole-index commit safe by declaration, refuses staged secrets, and clears the phantom `index.lock`. | **Every commit.** |
| `registry_lint.py` | Keeps `MODEL_REGISTRY.md` honest: **pointer drift** against the raw eval JSON, and a **multiline retracted-claim sweep over section headers**. | Before quoting the registry; after any retraction. |
| `repo_janitor.py` | Worktree/`incoming/` sprawl report — safe-to-delete worktrees (**report-only** unless `--delete`), a dated incoming ledger, and future-dated items. | Weekly, and whenever a search result looks truncated. |
| `fleet_probe.py` | Fleet liveness by **discovery**, never by hardcoded log names: finds jobs in `ps`, binds each to its own log via the launcher's stdout redirect, cross-checks the GPU against the process table, catches freezes (`LOG_STALE`, `STEP_NOT_ADVANCING`) and measures disk with a real `dd`. Exit `0`/`1`/`2` = GREEN/AMBER/RED. | Before claiming the fleet is healthy; behind the `fleet-status` skill. |

Tests: `pytest tools/tests/` — count re-measured each run; see the current figure in
`Tools&DevEnv/Research/STATE.md`. Each drives a throwaway git repo, a synthetic pytest
project, or a captured `ps`/`nvidia-smi` payload end-to-end; the CUDA-specific ones skip
loudly on a CPU-only box.

## fleet_probe

```bash
python tools/fleet_probe.py                     # table, all four hosts, ~10 s
python tools/fleet_probe.py --json              # machine-readable
python tools/fleet_probe.py --hosts pod1 --no-dd
```

It exists because the previous monitor greped **hardcoded** run names
(`p0-sB01-realmix.log`, `arm_base.log`, `pgrep -fc train_worldmode[l]`) that had been
renamed away. A grep matching nothing prints nothing, and printing nothing was scored as
health — the fleet lost 2 of 4 GPUs behind dead trainers **four times** under a monitor
that reported no anomaly. The one rule the tool enforces everywhere:

> **Absence of evidence is an ALARM, not an all-clear.**

A verdict starts at `UNKNOWN` and needs positive evidence to reach GREEN; a running job
whose log cannot be discovered is **AMBER — "liveness UNVERIFIED"**, never green. Roles
matter: `role=burst` (the eval pod) is *supposed* to sit idle, `role=train` at 0 % is RED.

Two Windows traps are baked in, both measured the hard way (see the 2026-07-21 note):
remote bash payloads are sent as **LF bytes** (`text=True` would translate `\n` to CRLF
and every `fi` would arrive as `fi\r`), and on win32 it drives
`C:\Windows\System32\OpenSSH\ssh.exe` — git-bash's MSYS `ssh.exe` **deadlocks** under
`subprocess` pipes, and does so only against the *busy* hosts, which looks exactly like an
outage and is not one. Override with `--ssh`.

## ci_gate

```bash
python tools/ci_gate.py --rootdir stack                  # the standard gate
python tools/ci_gate.py --rootdir stack --gpu-smoke require
python tools/ci_gate.py --rootdir stack --json gate.json # for the orchestrator
python tools/ci_gate.py --rootdir stack -- -k comma2k19  # pytest passthrough
```

Windows: `.\tools\ci.ps1` (activates the off-Drive venv; `-GpuSmoke require`, `-Json`).

- **Exit 0** = GATE PASS. **Exit 1** = one or more gate reasons, all printed.
  **Exit 3** = pytest could not be launched at all.
- **Why a suite manifest and not just node tripwires:** a named-node tripwire only
  guards nodes somebody thought to name. `SUITE_MANIFEST` pins the load-bearing
  modules (instrument doctrine, calib trio, the three reference arms, eval/metric
  surfaces) to a collected-count **floor**, so a module that is deleted, renamed, or
  quietly halved fails the gate. Adding tests is always fine; removing them has to
  edit that dict on purpose. `--min-total` (default 390) is the same idea for
  wholesale loss, e.g. a broken `conftest.py` deselecting half the tree.
- **Budgets** (measured 2026-07-20): full suite **60.2 s / 531 tests** on the Drive
  tree, **39.3 s / 396 tests** in an off-Drive worktree; tall pole
  `test_replay_app_test_mode_and_regression_gate` 7.2–7.9 s. Defaults are 15 s
  per-test / 150 s wall — comfortably inside the 5-minute ceiling, so **no sharding
  is needed** at current suite size.

## gpu_tripwire

```bash
PYTHONPATH=stack python tools/gpu_tripwire.py            # human report
PYTHONPATH=stack python tools/gpu_tripwire.py --json g.json --require-cuda
```

The `stack/tests` suite is **100 % CPU-only** (`grep -rl cuda stack/tests` returns
nothing, measured 2026-07-20) while every trainer, eval and deploy tick runs on a
GPU — so device/dtype/NaN regressions were invisible to CI. Four probes close that:

| Probe | Asserts |
|---|---|
| `P1_encode_parity` | `WorldModel.encode` CPU vs CUDA, `max abs dev <= --tol` (1e-3) |
| `P2_imagine_parity` | operative predictor, every horizon, same tolerance |
| `P3_i2_on_device` | I2 batch-1 vs batch-B encoder consistency, **run on CUDA** (1e-4) |
| `P4_backward_finite` | one `loss.backward()` on CUDA; every gradient finite |

Measured on the local RTX 4060 (torch 2.11+cu128, fp32, 2026-07-20): all four pass
in **1.7 s**, worst deviation **9.5e-07**, batch-1 encode **1.26 ms**. No CUDA
visible → `--require-cuda` fails, otherwise a loud skip.

## session_guard

```bash
python tools/session_guard.py            # gate the current worktree
python tools/session_guard.py --strict    # branches, source + stale INTAKEs also block
python tools/session_guard.py --base origin/main   # tip = a different ref
python tools/session_guard.py --json      # machine-readable report
```

Windows: `.\tools\session_guard.ps1` (activates the off-Drive venv, same flags).

- **Exit 0** = clear to end the session (warnings may still print).
- **Exit 1** = a BLOCKING condition — uncommitted deliverable under `TanitAD Research
  Hub/`, `Project Steering/`, `PROJECT_STATE.md`, or `DECISIONS.md`. Commit or discard,
  then re-run until `RESULT: PASS`.
- **Exit 3** = not a git repo / git unavailable.
- The **source check** (`stack/`, `tools/`) warns rather than blocks — a mid-work tree
  is legitimately dirty — and lists untracked files separately, because an untracked
  module has no copy anywhere. It was added on 2026-07-20 after the shared Drive tree
  was found holding 40 uncommitted `stack/` paths, 22 of them untracked (12 test
  modules = 135 tests, 9 `tanitad/lake/*` modules) while the hub check said "clean".
- Status is read with `--untracked-files=all`: the git default collapses a wholly
  untracked directory to one `?? stack/` row, which would hide exactly those modules.

The "tip" defaults to `HEAD` (the worktree's current integration point) because
`origin/main` is intentionally diverged in this repo; pass `--base` to override.

---

## safe_commit — *use this instead of `git commit`*

```bash
# the normal call: declare what you staged
python tools/safe_commit.py -p tools/ -p "TanitAD Research Hub/Data Engineering" \
    -m "tools: wave-1 ops tooling"

python tools/safe_commit.py --print-index                 # look FIRST (always free)
python tools/safe_commit.py -p tools/ -m "..." --dry-run   # every guard, no commit
python tools/safe_commit.py --paths-from staged.txt -F msg.txt
python tools/safe_commit.py --accept-index -m "..."        # whole index, declared
```

It encodes `CLAUDE.md` §"Git hygiene". Four properties, each earned by an incident:

- **It never emits `git commit -- <pathspec>`.** That is git's *partial-commit*
  path and it **segfaults intermittently on this repo** (exit 139 under MSYS,
  `0xC0000005` native — so not the shell; fsmonitor was already off). Three
  root-cause theories were asserted and falsified in one session
  (`RETRACTION_LOG` 07-25, C8), so the only defence is to never use the form.
  A test asserts the emitted argv is exactly `git commit -F <file>`.
- **Pathspec-free means the WHOLE INDEX**, which is how `60265d3` swallowed the
  eval tooling and `3d41bd0` swallowed REF-C v1.2's rescorer. So you *declare*
  what you staged with `-p`, and anything else in the index **aborts the commit**
  and is printed with its diffstat. `--accept-index` is the sanctioned override:
  it proceeds and **names the foreign paths in the commit message** rather than
  hiding them (splitting them out would need the crashing form).
- **Secrets are refused on three independent probes** — git's own `check-ignore`
  verdict (the only way `Keys.txt` reaches the index is `git add -f`, which
  leaves that fingerprint), the filename shape, and the staged content. Findings
  print **redacted**; a test asserts the token body never appears on stdout.
  The brief's loose `hf_[A-Za-z0-9]+` is **advisory only** — it matches this
  repo's own committed `hf_export.py` / `hf_repo_state_*.json`; the blocking
  pattern is the real 34-char token shape.
- **The phantom lock.** Every crash leaves `.git/index.lock`, and the next
  attempt reports *"Another git process seems to be running"* — contention
  wording for a corpse. It confirms no `git` process is alive (matched on the
  **image name**, never `pgrep -f`, which self-matches), removes the lock, and
  retries up to `--retries` (default 3). **HEAD movement, not the exit code, is
  the success signal**: a crash can land the commit and still return 139, and
  retrying there would double-commit.

Also refuses to commit on `main` (`--allow-main` to override) and never passes
`--amend` or `--no-verify`. Exit codes: 0 committed / dry-run OK · 1 REFUSED ·
2 usage · 3 no git · 4 commit failed after retries.

## registry_lint

```bash
python tools/registry_lint.py                       # the standard sweep
python tools/registry_lint.py --strict               # body + boilerplate hits also fail
python tools/registry_lint.py --self-test            # 5 red/green falsifiers
python tools/registry_lint.py --file "TanitAD Research Hub/Benchmarks & Eval/LEADERBOARD.md"
python tools/registry_lint.py --json lint.json
```

**CHECK 1 — pointer drift.** Every number in the registry is hand-transcribed
from raw eval JSON and nothing links it back. A pointer restores the link:

```markdown
<!-- src: taniteval/results/driving_flagship-30k.json#headline.ade_0_2s.mean near="full-set" -->
| **1=** | **Flagship v1 (speed+jerk) FINAL** | ... *(full-set 0.4271, boot [...])* |
```

The tolerance is **the coarser of the two written precisions** — `0.4271`
tolerates 5e-5, `0.452` tolerates 5e-4, and a prose `0.43746` against a stored
`0.4375` passes, because either side may legitimately have been rounded. A
dangling path, a missing field, or a non-numeric field is itself a finding.
Pointers may also live in the anchor-keyed sidecar `tools/registry_pointers.jsonl`
(**5 seeded** on the rank-1 leaderboard tie and the §1.4b retraction), which is
how the mechanism gets added without editing a file six agents have open; an
anchor that matches ≠ 1 line is reported, because a pointer that silently
relocates is worse than one that is missing.

**CHECK 2 — the retracted-claim sweep.** Claims quoted in `RETRACTION_LOG.md` are
searched against a **whitespace-collapsed token stream of the whole document**
with a token→line map, so a claim that wraps across a newline matches exactly
like one that does not — that is the instance which walked through a line-based
grep on 07-25. Matching tolerates `--gap` inserted tokens (default 1) because the
header that survived four days said *"best **ADE** in the program"* while the log
quoted *"best in the program"*: **at `--gap 0` the tool reproduces the miss.**

- A hit touching a **section header** is an **ERROR**; body prose is a **WARN**,
  because a body legitimately quotes the claim it is retracting.
- Suppressed: hits within `--context` tokens of a retraction marker
  (`retracted`, `corrected`, `NOT`, `superseded`, …), and any line carrying
  `<!-- lint-ok: reason -->`.
- `--rare-max` (default 25) demotes a header hit whose every word is house
  vocabulary to a WARN. MEASURED on the registry (26,459 tokens): the neutral
  title `### 4.4 REF-C CLOSED-LOOP …` matches a retracted claim on
  ref=122/c=81/loop=60/closed=32, while the real stale headline carries best=23
  and program=13. Nothing is hidden — the demotion still prints, and `--strict`
  makes it fatal.

Exit 0 clean · 1 a finding · 2 usage/IO.

## repo_janitor

```bash
python tools/repo_janitor.py                          # report everything
python tools/repo_janitor.py --fast                    # skip the per-worktree status probe
python tools/repo_janitor.py --ledger-out LEDGER.md    # the dated incoming ledger
python tools/repo_janitor.py --delete                  # remove ONLY safe candidates
python tools/repo_janitor.py --json janitor.json
```

The point is not tidiness, it is **search reliability**. On 07-24 two modules
were reported stranded and absent from main; both were in main. `Glob` is
mtime-sorted and truncated at 100, and with 51 worktrees + ~95 incoming bundles
the freshly-touched worktree copies filled the window (`RETRACTION_LOG` 07-24, C8).

- **Worktrees.** `git worktree prune` first, then each worktree is scored for
  commits the tip does not have, plus a working-tree status probe. Only
  `0 ahead AND clean` becomes a **CANDIDATE**, and candidates are **reported, not
  deleted** — `--delete` is required, never uses `--force`, and refuses to run
  under `--fast` because a clean-tree check is mandatory before removing anything.
- **Incoming ledger.** Every `**/Implementation/incoming/<date>-<slug>/` bundle
  with age, file count, size and INTAKE-verdict state, oldest first.
- **Future-dated items.** Tracked files whose mtime is ahead of the clock and
  bundles whose date slug is. This repo's narrative clock genuinely runs ahead of
  wall-clock, so these are an **artifact to be aware of**, not corruption — but a
  future stamp sorts first in every mtime-ordered tool, which is the mechanism
  behind the false claim above.

Exit 0 always unless `--fail-on {future,stale,any}` is set: this is a reporter,
not a gate.
