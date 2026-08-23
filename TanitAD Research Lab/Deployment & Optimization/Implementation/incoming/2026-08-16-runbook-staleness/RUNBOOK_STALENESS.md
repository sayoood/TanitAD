# The 3 a.m. runbook was stale ELEVEN ways, one of them since the day it was written — and now a test executes it

**2026-08-16 · branch `agent/arch-inf-20260803` · base `eca7106`.**
⛔ **CPU only. No GPU touched, Thor not contacted, no stage launched.** The live v6F S-W run
(pid 25477, ~step 6,400/30,000) is unaffected: nothing here changes a `state_dict` key, a shape, an
optimiser group or a schedule.

---

## 0. The one-line answer

> **The brief said "stale in five ways". It is stale in ELEVEN, I checked each against the code, and
> the two worst were not on anyone's list** — because they were inherited from another document's
> count rather than re-derived. **⭐ One of them was never true at all: §2.0's emphatic, ⚠️-marked,
> "MEASURED 2026-08-12" instruction to md5 THREE files was WRONG THE DAY IT WAS WRITTEN** — the two
> missing dependencies were already top-level imports at `2b8d09e`, the exact commit that row cites.
>
> ⇒ **The fix is not a corrected document. It is that the document stopped being a source of
> truth.** §2's launch lines are now the *output* of `scripts/v6_chain.py commands`, and
> `stack/tests/test_runbook_commands.py` (16 tests) **executes** the runbook's preflight command and
> fails the suite the moment the doc and the generator disagree.

**Suite:** `3170 passed / 0 failed / 17 skipped / 2 xfailed` — MEASURED, `raw/stack_pytest.txt`.
Briefed baseline at HEAD was **3154**; **+16 are this file's** and there are **0 failures**.

---

## 1. ⭐ THE HEADLINE: A STALENESS COUNT IS ITSELF AN INHERITED CLAIM

The brief carried "§2 is stale in five ways", correctly sourced to `V6_STAGE_CHAIN.md` §8.4 and
Escalation 3. That doc named four concretely — `--batch 16`, the `v6-SW-30k` naming, no admission
gate, no arm handling — and the fifth is "it should point at `v6_chain.py commands`". **Every one of
those is real and is fixed here.**

**But the count was produced by an author who was looking at the chain, not at the runbook.** Read
§2 line by line against the code and there are eleven, and **the two highest-blast-radius items are
absent from that list of five**:

| | the two the inherited count missed | why it is worse than the five |
|---|---|---|
| **A** | §2.0(a) tells you to md5 **three** files; the trainer's import-time closure is **four** `scripts/` files (+ `v6.py` = **five artifacts**) | a file-ship that follows it **dies at import** on the pod — `ModuleNotFound: train_stage_a` — after the transfer, after the md5 verification, at the moment of launch |
| **B** | §2.1 reads *"the only stage that can start tomorrow"* and §1 rows 10–14 read `⛔ BLOCKED — PI`; **S-W has been LIVE for days** | the other ten defects make a command *fail*. This one makes it **succeed**: an operator "unblocks" S-W and starts a **second trainer on the fleet's only GPU**, next to a 7.4-day run |

⇒ **The lesson, and it is the same lesson twice today (C70, C70b):** a count of defects is an
INHERITED claim like any other. It travels, it decays, and it caps the audit at whatever the last
author happened to see. **Re-derive the count; never adopt it.** *(This report's own "eleven" is
MEASURED per item in §2 below, and should be re-derived by the next reader too.)*

---

## 2. THE ELEVEN, EACH CHECKED AGAINST THE CODE AS IT IS NOW

Ordered by what happens to the operator, not by section number. Every "code" cell is **MEASURED
(mine 2026-08-16)** unless marked.

### 2.1 ⛔ LAUNCH-FATAL — the command does not work

| # | the doc said | the code says | consequence |
|---|---|---|---|
| **S1** | "⚠️ **THREE files, not two**: `v6.py`, `train_v6_staged.py`, `train_v58f_unicycle_head.py`" | the import-time closure is **`train_v6_staged` + `train_stage_a` + `stage_a_probes` + `train_v58f_unicycle_head`** — `train_v6_staged.py:114` and `:117` are module-level `from` imports. Confirmed two ways: a runtime `sys.modules` filter after `import train_v6_staged`, and a static module-level AST walk. **Both give 4.** | `ModuleNotFound: train_stage_a` at launch |
| **S2** | every path is `/workspace/experiments/v6-SW-30k`, `v6-ST-10k`, … | the live run is **`~/experiments/v6F-SW-30k`** — two compounding errors, the `v6F` prefix *and* the `/workspace` → `/root` root. `v6_chain.py:376` pins `sw_dir="v6F-SW-30k"` | §2.2's `--init-from …/v6-SW-30k/ckpt.pt` points at nothing; and `--out …/v6-SW-30k` would start a **duplicate S-W** |
| **S3** | §3's gate table: S-S required = **`STRATEGIC_family`** | `STAGE_GATE_SPEC["S-S"]["required"]` is **`("STRATEGIC_family", "sel_gap_revalidated", "TACTICAL_revalidated")`** (`train_v6_staged.py:365-372`) | plan one gate probe, discover two more **after ~2.5 GPU-days** — and an S-S gate that omits them reads INCONCLUSIVE, which S-J refuses |

⭐ **S1 is not staleness — it is a defect that shipped.** `git show 2b8d09e:stack/scripts/train_v6_staged.py`
has those imports at the same line numbers. The author measured **one** dependency
(`train_v58f_unicycle_head`), wrote "two → three", and never enumerated the closure.

> **Root-cause class, and it is a NEW one worth logging:** **`absence found at ONE location is not
> absence` has a mirror image, and the mirror is not covered by the rule as written —
> PRESENCE FOUND AT ONE LOCATION READ AS THE WHOLE SET.** The existing rule guards against
> concluding *"X does not exist"* from one probe. It says nothing about concluding *"the set is
> {X}"* from finding X. Both are one-probe generalisations; only one of them has a rule. **The
> operational form: when a claim is about a SET — dependencies, required probes, affected files,
> call sites — enumerate it with a tool that can see all of it, and say which tool.** A grep that
> found one hit is evidence about that hit and nothing else.
>
> *(Sibling: `CLAUDE.md`'s own "2 of 36 features" correction to 4 — a stale absence-claim living
> inside the rule warning about stale absence-claims. Same shape: a count asserted from a partial
> read of a set.)*

### 2.2 ⛔ DANGEROUS — the command works, and that is the problem

| # | the doc said | reality | consequence |
|---|---|---|---|
| **S4** | §2.1 *"the only stage that can start tomorrow"*; §1 rows 10–14 `⛔ BLOCKED — PI (D1 cost, D2 pod)` / `NOT-STARTED` | **S-W is LIVE on Thor** — pid 25477, ops loop pid 29587, `~/experiments/v6F-SW-30k`, ~step 6,400/30,000 *(INHERITED from the coordinator's fleet read; not re-probed — Thor holds the only GPU and this turn was CPU-only)* | a **second trainer** beside a 7.4-day run on the fleet's only GPU |

### 2.3 ⚠️ WASTEFUL — the command works and quietly costs you

| # | the doc said | the code says | consequence |
|---|---|---|---|
| **S5** | `--batch 16` on all four stages | `THOR_BATCH = 8` (`v6_chain.py:122`): Thor's 20 SMs **saturate at 8**, throughput FLAT at 12.3–14.1 windows/s across a 6× batch range. The trainer's default is 16 — the A40 instinct | buys **nothing**, costs unified memory |
| **S6** | `--v2-lru` never passed | trainer default is **64** (`train_v6_staged.py:2210`); the chain emits **6** (`THOR_V2_LRU`) | a 64-entry frame cache competing with the model, on a box where **host RAM IS device memory** |
| **S7** | §4: `step_s` ESTIMATED **21–35 (A40)**, ladder in A40-hours, D1 branch table built on that band | **MEASURED 27.18 s/step on Thor**, batch 8 (marginal over steps 6300→6400; three statistics with different startup exposure agree to 0.5 %). A40 = 20.46 | 27.18 is **not "in band"** — it is a different measurement of different silicon. Reading it as in-band is how a re-cost rule becomes decoration |
| **S8** | `nohup python3 … > train.out 2>&1 &` | the chain emits `setsid nohup python3 -u … > train.out 2>&1 < /dev/null &` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | without `setsid` the job shares the ssh session's process group; without `< /dev/null` it is exposed to the nested-stdin trap |

### 2.4 ⚠️ MISLEADING — the doc documents something that is not true

| # | the doc said | the code says |
|---|---|---|
| **S9** | §2.5 lists `--i-know-this-is-the-control-arm` as the working escape hatch | it was **INERT** until 2026-08-16 (defect D1): `main()` registers `dest="control_arm_ack"` while `preflight` read `getattr(a, "i_know_this_is_the_control_arm", False)` — an attribute argparse **never creates**. ⇒ **the pre-registered inert-scorer control arm was unlaunchable**, and `V6F_PLANNER_DESIGN` §4.1's "Pre-registered controls" list documents a flag that did not work. Fixed at `train_v6_staged.py:2280-2281` |
| **S10** | §2 carries no selector story at all | **SEL-1 is REFUSED** (E-WC2, 2026-08-16: σ/ADE 9.9915 [7.4492, 13.5119] vs a pre-registered refusal line of 3.0). S-T's default is `--selector none --w-select 0`, and `assert_selector_admissible` **refuses at launch** — it runs *first* in `assert_may_launch`. The doc's emitted commands were *accidentally* correct (the trainer's defaults match), but nothing told the operator that adding `--selector` now refuses, or why |
| **S11** | §2.0 *"Expect … byte-for-byte the same three lines"* | **six** lines today — the dry-run also writes `stage_gate.json` (INCONCLUSIVE, `_dry_run`) and names it in the OK line. MEASURED by executing it. *(The `params 87.89 M` half is still exactly right.)* |

**Also fixed but not counted as §2 staleness** (they are §1/§3/§4 rows §2 leans on): row 19's "23
standing pytest failures" — **now 0**; §5.1's *"R2 is the one risk with no built-in defence"* — it
now has one; §4.2's `--batch 8` cost lever — **not a lever on Thor, it is the default**.

---

## 3. ⭐ THE CLASS FIX — a runbook that fails the suite instead of the operator

**`stack/tests/test_runbook_commands.py`, 16 tests, NEW.** A runbook goes stale silently because
nothing tests it. This tests it.

### 3.1 The design decision that matters

⛔ **The runbook stopped being a second source of truth.** §2.2's four launch lines are the verbatim
output of `v6_chain.py commands`, and
`test_runbook_launch_lines_are_exactly_what_v6_chain_emits` compares **argv token lists**. That one
assertion pins, permanently and without anyone remembering to: `--batch 8`, `--v2-lru 6`,
`--n-candidates` on every step, the `v6F-*` directory names, `setsid`/`-u`/`< /dev/null`,
`PYTHONPATH`, `OMP_NUM_THREADS=6`, and `--init-from` + `--prev-gate` on every stage above S-W.

*(Contrast the alternative — asserting each fact separately. That is a list someone must extend
every time the chain learns something, i.e. exactly the mechanism that failed.)*

### 3.2 ⛔ AST/PARSER-BASED, NOT REGEX — and the guard cannot match its own documentation

The brief's warning was specific: *a regex guard written today matched its own documentation at
176:0.* That failure mode is designed out at four levels:

| step | mechanism | why not a pattern |
|---|---|---|
| extract | ```` ```bash ```` fence detection | structural markdown — asks *where the code block is*, never *what it says* |
| lex | **`shlex.split`** | a real POSIX lexer. Handles the quoting the chain deliberately emits |
| validate flags | argv fed to the trainer's **actual `build_parser()`** | **argparse is the authority.** An unknown flag raises `SystemExit(2)` and fails the test |
| validate admissibility | the trainer's own **`preflight()`**, imported | not a re-implementation. Covers the S-W `--selector` refusal, the S-S `--w-select` refusal, `--v2-cache` required, `--o5-k ≤ --plan-steps`, `--init-from` mandatory |
| validate the ship list | **module-level AST walk**, transitive | breaks statically the moment someone adds a top-level import, on a box with no torch |

⭐ **`test_the_extractor_cannot_match_its_own_documentation` makes 176:0 unrepeatable.** §2.5's
control-arm **table** names `--per-layer-encoders`, `--no-isolate-planner`, `--uplink` and
`--i-know-this-is-the-control-arm` in prose. The test asserts those flags appear in **zero**
extracted commands — a pattern-matching guard would "find" them, report the runbook as covered, and
never look at a command.

⚠️ **Two traps I hit while building it, both worth carrying forward:**

1. ⛔ **`build_parser()` alone is NOT the trainer's CLI.** `main()` adds
   `--i-know-this-is-the-control-arm`. A guard checking only `build_parser` reports a **correct**
   runbook as stale — *the same shape as the defect that made that flag inert.* The test builds the
   parser the way `main()` does.
2. ⛔ **`ast.walk` over-counted the import closure by 12 files** (`train_flagship_v4`, `refb_train`,
   … are all *function-local* imports, resolved lazily). An over-broad closure is **not a safe
   error**: it would make §2.0(a) list a dozen files nobody needs, and a list nobody believes is a
   list nobody follows. The walk descends module-level `if`/`try` only, never into `def`/`class` —
   and `test_the_static_closure_matches_the_runtime_one` pins it against the number MEASURED by
   actually importing the trainer.

### 3.3 The runbook's own preflight, EXECUTED

`test_step_zero_dry_run_actually_runs` runs §2.0(e) as a subprocess on CPU (~10 s, no corpus, no
GPU) and asserts exit 0 + `dry_run.json` + `stage_gate.json`. **A runbook command that has never
been run is a hypothesis.** *(The four stage lines cannot execute without the corpus, so they go
through `preflight()` instead — stated rather than glossed.)*

### 3.4 ⭐ Every guard here has been SEEN TO FIRE

*A guard nobody has seen fire is a guard nobody knows about.* Two tests replay the **actual
pre-2026-08-16 text** and assert it is rejected:

| replayed | verdict |
|---|---|
| the old S-T line (`--batch 16`, `v6-SW-30k`, no `--v2-lru`, no `--n-candidates`) | ⛔ rejected |
| the old three-file `md5sum` block | ⛔ rejected — the closure is not covered |

Without these, every other assertion could be **vacuously true** on an empty extraction — which is
precisely how the 176:0 guard passed.

---

## 4. THE WIDER SWEEP — 11 operator-facing runbooks audited

Priority is **"would someone paste this under pressure"** × **"does its host answer"**.

| file | pressure | verdict | headline item |
|---|---|---|---|
| ⭐ `TanitAD Research Hub/Production & Optimization/THOR_DEPLOYMENT_RUNBOOK.md` | **HIGH** | **commands CURRENT, but DANGEROUS** | ⛔ **FIXED** — §4's TRT build block ran on `thor6` with **no busy-GPU precondition**, and thor6 is training v6F S-W right now |
| `Project Steering/STOP_2026-08-15_RESUME_RUNBOOK.md` | **HIGH** | STALE | ⛔ **FIXED** — §3 "resume on a fresh pod" would start a **second** trainer; §3.6's two watchers carry dead `/workspace` paths |
| `stack/ops/runs.d/README.md` | **HIGH** | STALE | ⛔ **FIXED** — the stated contract was wrong in **both** directions |
| `stack/RUNPOD_RUNBOOK.md` | **HIGH** (title) | STALE | ⛔ **FIXED** — prescribed `pkill -f`, which two other docs here ⛔-ban; and tailed a log nothing writes |
| `Project Steering/POD_MIGRATION_RUNBOOK.md` | HIGH | wholly superseded | ⛔ **FIXED** (banner) — every host released |
| `stack/ops/POD_ACCESS_2026-08-04.md` | HIGH-looking | premise gone | ⛔ **FIXED** (banner) + **escalated**: a live SSH **private key** secret |
| `Project Steering/HANDOVER_TO_LOCAL_2026-08-15.md` | MED | STALE (priorities) | ⚠️ **LISTED** — 3 cheap items |
| `Project Steering/POD_HANDOVER_2026-08-13.md` | MED | STALE (framing) | ⚠️ **LISTED** — self-contradicts on checkpoint size by **2.7×** |
| `Project Steering/GATE_PROTOCOL.md` | MED | **CURRENT** | every flag verified present. 2 cosmetic path items |
| `Project Steering/CONTINUATION_PROTOCOL.md` | LOW | CURRENT, 2 drifts | reports dir + no HF row |
| `Project Steering/PRE_FLIGHT_VALIDATION.md` | LOW | historical | its pass criterion (261 M ±5 %) cannot apply to a 336.5 M model |

### 4.1 ⭐ THE FINDING THE SWEEP PRODUCED, which is not "N docs are stale"

> **A runbook's danger is not how stale it is. It is whether its host is still alive.**

Nine of the eleven point at released RunPod hosts and **fail safe** — you paste, nothing answers,
you stop and think. **`THOR_DEPLOYMENT_RUNBOOK.md` fails loud and expensive**, because its host
answers: its §4 TensorRT build allocates and compiles on the box currently 97 % busy with a
7.4-day training run, and it carried **no warning at all**. It was also the *least* stale document
in the set by flag-count — every command verified CURRENT.

⇒ **Staleness triage by "how old is it" would have ranked it LAST. It is first.** The fix (a ⛔
banner + a `/proc`-based precondition) took two minutes.

### 4.2 Fixed in this turn

| file | what changed |
|---|---|
| `…/Production & Optimization/THOR_DEPLOYMENT_RUNBOOK.md` | ⛔ busy-GPU STOP banner at §4, with a `/proc`-based precondition (**not** `pgrep -f`) and the unified-memory probe rule |
| `Project Steering/STOP_2026-08-15_RESUME_RUNBOOK.md` | §3 SUPERSEDED banner + the two watchers' dead paths + the `"aug120.py" in cmd` guard that cannot match `aug120_pipeline.py` |
| `stack/ops/runs.d/README.md` | contract corrected (`WORKDIR` is **not** required, defaults to `.`; `OPS_DIR` defaults to `/workspace/ops` and is silently wrong on Thor); launch line now copies the manifest out of the repo; `pgrep -f` verification replaced; `v6_chain.py manifests` documented; the missing live-run manifest recorded as a 🟥 gap |
| `stack/RUNPOD_RUNBOOK.md` | HISTORICAL banner; `pkill -f` → kill by explicit PID; `p0-sB01.log` → `p0-sB01-realmix.log` |
| `Project Steering/POD_MIGRATION_RUNBOOK.md` | HISTORICAL banner — **and an explicit "keep §3 and §5"**, because its doctrine outlives its hosts |
| `stack/ops/POD_ACCESS_2026-08-04.md` | dead-endpoint banner; MEASURED core preserved; `TANITAD_POD_SSH_KEY` escalated |

### 4.3 Left undone, deliberately, with the reason

1. 🟥 **The live `v6F-SW-30k` run has no supervisor manifest.** `v6_chain.py manifests` generates the
   three downstream stages' — but the **live** one needs `TRAIN_CMD` read **verbatim from
   `/proc/25477/cmdline` on Thor**, and I was instructed not to contact Thor. ⛔ **I did not
   generate it from here**: a manifest built on a guess about Thor's `$HOME` is *worse* than none —
   it is "a run row that lies about what moved" in a new costume. Recorded as a 🟥 gap in the README.
2. **`pbattery_watcher.py` / `hf_push_loop.py` portability** — 5 hardcoded `/workspace` paths and a
   real substring bug. Documented in the STOP banner; the code fix is REAL WORK and touches ops
   scripts other streams may be holding.
3. **`TANITAD_POD_SSH_KEY` rotation / `pod-exec.yml` retirement** — credentials are the PI's.
4. **A v6-era `PRE_FLIGHT_VALIDATION.md`** — its axes are 261 M/4-brain/comma-mix. A rewrite is a
   design task, not a correction.
5. The MED/LOW cheap items in §4 (POD_HANDOVER's 2.7× size self-contradiction, HANDOVER's three
   stale priorities, GATE_PROTOCOL's `taniteval/taniteval/` paths, CONTINUATION's reports dir).
   **Listed, not fixed** — each is one line, and none of them is on a path an operator runs under
   pressure. ⚠️ *Listing is not fixing; these should be swept by whoever owns `Project Steering/`.*

---

## 5. WHAT I DID NOT DO / OPEN

1. ⛔ **Nothing ran on a GPU, no stage was launched, Thor was never contacted.** The live-run facts
   (pid 25477, pid 29587, step ~6,400) are **INHERITED** from the brief and the coordinator's fleet
   read — **not re-probed**, and labelled as such everywhere they appear in the runbook.
2. ⚠️ **The test pins the runbook to ONE config** (`/root/experiments`, `/root/TanitAD/stack`, the
   two canonical caches). If Thor's `$HOME` is not `/root`, §2.2's paths need regenerating — the
   doc says so and gives the command. **The chain refuses a `~` rather than expanding it**, because
   `~` on the generating box is not `~` on the running box.
3. ⚠️ **`test_runbook_names_every_probe_S_S_requires` and the SW-threshold test use substring
   containment**, not parsing — the only two that do. Their polarity is the safe one: the names come
   from `STAGE_GATE_SPEC` / `SW_LATENT_ADMISSION` in **code**, so adding a probe forces a doc edit
   and never the reverse. Stated rather than hidden.
4. ⚠️ **I fixed §1/§3/§4 rows that §2 depends on, and left the rest of those sections alone.** §5's
   risk register is still largely a 2026-08-12 document (R1, R3–R12 re-read and still accurate; R2
   and §5.1 updated for SEL-1).
5. **The eleven is mine and should be re-derived.** §1's whole point is that inherited counts decay.

---

## Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| ⭐ **the corrected runbook** (§2 rewritten; masthead, §1 rows 9–16/19, §3 S-S gate row, §4.2/4.3, §5.1) | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-07-hierarchical-wm-redesign/V6_GO_PACKAGE.md` | **staged** |
| ⭐ **the class fix — 16 tests that execute the runbook** | `stack/tests/test_runbook_commands.py` | **staged** (new) |
| this writeup | `…/Production & Optimization/Implementation/incoming/2026-08-16-runbook-staleness/RUNBOOK_STALENESS.md` | **staged** (new) |
| full-suite log | `…/2026-08-16-runbook-staleness/raw/stack_pytest.txt` | **staged** (new) |
| the generated chain commands, as banked | `…/2026-08-16-runbook-staleness/raw/v6_chain_commands_thor.txt` | **staged** (new) |
| the import-closure measurement, both ways | `…/2026-08-16-runbook-staleness/raw/import_closure.json` | **staged** (new) |
| Thor deployment runbook — busy-GPU STOP banner | `TanitAD Research Hub/Production & Optimization/THOR_DEPLOYMENT_RUNBOOK.md` | **staged** |
| stop-runbook SUPERSEDED banner | `Project Steering/STOP_2026-08-15_RESUME_RUNBOOK.md` | **staged** |
| pod-migration HISTORICAL banner | `Project Steering/POD_MIGRATION_RUNBOOK.md` | **staged** |
| pod-access dead-endpoint banner + key escalation | `stack/ops/POD_ACCESS_2026-08-04.md` | **staged** |
| manifest contract corrected | `stack/ops/runs.d/README.md` | **staged** |
| RunPod runbook banner + `pkill`/log fixes | `stack/RUNPOD_RUNBOOK.md` | **staged** |

**Nothing was committed and nothing was pushed. Nothing produced here lives only on a pod or only in
a worktree.**

---

## Escalations — requests, not notes

1. ⭐ **`RETRACTION_LOG.md` warrants a NEW root-cause class, and I do not own that file.** *"Absence
   found at ONE location is not absence"* has a **mirror with no rule: PRESENCE FOUND AT ONE
   LOCATION READ AS THE WHOLE SET.** §2.1 has the text ready to lift, with two instances (§2.0's
   three-file list; the inherited "stale in five ways"). ⚠️ This is not a variant of the existing
   entry — the existing rule guards *"X does not exist"* and is silent on *"the set is {X}"*.
2. ⭐ **`V6F_PLANNER_DESIGN` §4.1's "Pre-registered controls" list documents a flag that did not
   work** (`--i-know-this-is-the-control-arm`, inert until today). Any claim that those control arms
   were *reachable* before 2026-08-16 is false. The design doc is the orchestrator's/PI's.
3. 🟥 **The live v6F S-W run has NO supervisor manifest and, per the stop runbook, no supervisor.**
   7.4 days remain. It cannot be fixed from here without Thor — and the failure mode it guards
   against (a death that nobody notices) is the one that costs the most. **This needs a Thor-side
   turn**, using `/proc/25477/cmdline` verbatim, and `OPS_DIR` set explicitly (it defaults to
   `/workspace/ops`, which does not exist on Thor).
4. ⚠️ **`TANITAD_POD_SSH_KEY` is a live SSH private key in repo secrets for hosts that no longer
   exist, and `.github/workflows/pod-exec.yml` is still checked in.** Rotation/retirement is a PI
   call.
5. ⚠️ **`V6_GO_PACKAGE.md` lives in `…/incoming/2026-08-07-…/`** — a dated research folder — while
   functioning as **the** operator runbook. Dated `incoming/` folders are *records*; a living
   runbook in one is how it got read as historical and left to rot. **It probably belongs in
   `Project Steering/`.** I did not move it: the move breaks every citation and the test's path, and
   that is a call for whoever owns the hub layout. *(If it moves, update `RUNBOOK` in
   `stack/tests/test_runbook_commands.py` — the test asserts the path exists rather than skipping,
   precisely so a silent move cannot disable the guard.)*

**Evidence class:** every code claim in §2 and §3 is **MEASURED (ours)** and reproduced by
`stack/tests/test_runbook_commands.py`. The live-run state (pids, step, host) is **INHERITED**. The
SEL-1 numbers are **INHERITED** from E-WC2 (REF-C surface, T0-DIAGNOSTIC). The 27.18 s/step is
**MEASURED** (another agent's, on the live run) and every wall-clock derived from it is
**ESTIMATED**. The §4 sweep's per-file verdicts are **MEASURED** (flags AST-checked, paths probed).
