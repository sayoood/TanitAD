# P0-REPLICATE armed — the boundary, the preconditions, and a defect found while arming

**Agent:** TanitAD_TrainingFlyWheel · **2026-09-20** · authorisation: Master Mind, *"ARM THE RUNNER
FOR **P0-REPLICATE ONLY** … it must not roll on to P1 unattended."*

⛔ **No arm has run.** The runner is alive and WAITING on A7; the GPU is still held at ~3,950 MiB by
the PI's servers and the gate is untouched.

---

## 1. Where the boundary lives, and how it is proven

`bounded_next(out_dir, stop_after)` is the authorisation, as a pure function:

* while `P0-REPLICATE` is not VALID it is the answer (**within the bound**);
* once `P0-REPLICATE` is VALID there is **no next arm at any gate state** — `plan()` returns
  **DONE**, not RUN, and the same state *unbounded* would have returned RUN (the test asserts both
  halves, so the bound is shown to be what stops it);
* an **unknown** bound runs **nothing** rather than falling back to unbounded;
* after the arm finishes the runner prints `ZZP-BOUND-REACHED` and **returns** — it never loops
  back into `plan()`.

Pinned by 10 new tests and **4 new mutations** (`mutate_p_runner.py`), all caught, control green:
`M6` lets the bound expire · `M7` accepts an unauthorised arm · `M8` silently drops a flag from the
replicate · `M9` lets a missing input pass preflight. Full proof: **9/9 caught, control GREEN**,
`raw/mutation_proof_p_runner.json`.

⚠️ **One mutation's expectation was MINE and was wrong.** `M8` listed `test_build_argv_ADDS_NOTHING`
as a must-go-RED; that test's fixture carries no `--w-box3d`, so a *drop* mutation cannot move it.
The proof correctly read ⛔ FAILED and the fix was the expectation, never the guard. Recorded in
the harness beside the mutation so it is not "corrected" back.

## 2. ⛔ A DEFECT FOUND WHILE ARMING — the self-match trap, inside the function that warned about it

`count_a7_procs()` asked PowerShell for processes whose command line matches
`'*a7-imagenet-knockout*a7_run.sh*'`. **The literal pattern is itself part of the querying
process's command line**, so the probe matched **itself** — and every shell that had ever typed
the pattern. MEASURED: it returned **1–4 while the A7 launcher was demonstrably still WAITING** at
3,950 MiB, i.e. while A7 was **not running at all**.

⭐ **Why this was worth catching before the launch and not after:** the failure is *fail-safe in
direction* (the runner never starts when it shouldn't) and therefore **invisible** — the runner
would simply have waited **forever**, reporting a healthy-looking `A7 panel still running` line
every two minutes, and P0 would never have launched. A wait that never ends looks exactly like a
wait that is working.

⚠️ The function's own docstring claimed to avoid this ("never by a pattern that could match this
runner's own command line"). **A docstring is not a guard.** Fix, per the documented recipe —
make the emitted token disjoint from the searched token:

* the pattern is **assembled inside PowerShell from parts** (`$a`, `$b`, `$c`, `$t`), so the whole
  string never appears in any command line;
* the querying process **excludes itself by `$PID`**;
* the count is emitted as an opaque **`ZQ<n>ZQ`** marker and parsed from that, so the parser cannot
  match the words the query contains;
* it also matches the **trainer** (`refc_v3_train.py` with the A7 out dir), not only the panel
  script, so a panel whose shell has exited but whose trainer is alive still blocks.

**Verified by a discriminating control:** with A7 demonstrably not running the probe now reads
**0** (it read 1–4 before), and the runner's decision changed from the phantom
`A7 panel still running` to the true blocker `A7 arms not VALID yet: [4 arms]`.

## 3. What P0 will actually run, and one thing that is NOT reproducible

`build_argv` copies **A8's own recorded `argv` verbatim** (82 tokens) and changes exactly two
things: `--out` and `--seed 0 → 1`. **Nothing is added** — in particular no `--eval-window-dump`,
because an added flag makes it a different experiment and the arm-to-arm difference stops being a
floor. §3.1 of the prereg defines P0 as *"A8's exact flags, a different `--seed`, same steps"*.

⛔ **A8's exact CODE is not recoverable, and this must travel with the floor.** A8's `config.json`
carries **no commit, no code hash and no trainer md5** (checked: no `git`/`commit`/`code`/`md5`
key exists in it). P0 runs from the **pinned** `C:/Users/Admin/tanitad-a7-run` tree
(`refc_v3_train.py` md5 `e7ac9ec33ff44568a5e931619087cec9`), which is the tree the later P arms
will use — a floor measured on code the levers won't run is the wrong floor. The repo's copy of
the same file already differs from the pinned one by **313 diff lines**, so "the repo" is not a
reproducible target either.

⇒ **The P0-vs-A8 difference is `seed + an unquantified code delta`.** It is therefore an
**UPPER BOUND on the run-to-run seed floor**, not the seed floor itself, and it must be reported
that way. It stays *usable* because it is conservative in the direction that matters: a lever must
clear a **larger** margin to be called SUPPORTED. ⚠️ But it cannot separate a systematic code shift
from noise on one sample, so it may not be quoted as "the seed floor".

⭐ **The cheapest experiment that would give the pure floor** is a second replicate on the **same
pinned tree** (P0b, same flags, different seed, ~11 h). That is one card slot, not a redesign, and
I have not run it — it is outside the P0-only authorisation. Flagging it rather than assuming it.

⚠️ **A live risk, named rather than worked around:** A8's argv points `--v7-labels` at
`C:/Users/Admin/tanitad-wt/_s2build/…`, the **mirror**, whose resync deletes repo-absent files. I
kept the path because changing it would break replicate fidelity; the preflight **refuses to
launch** if the file is missing or its md5 is not `eefc38d1453bd1c73802d44d45affced`. If the
mirror wipes it, P0 refuses rather than training on different labels — and I will raise it.

## 4. Preconditions checked before the card is spent

`preflight()` refuses unless **all** hold, and an unreadable file is a REFUSAL, never a pass:

| check | value |
|---|---|
| the arm is the **authorised** one | `P0-REPLICATE`; anything else refuses |
| an arm was authorised at all | no `--authorise-arm` ⇒ launches nothing |
| pinned trainer md5 | `e7ac9ec33ff44568a5e931619087cec9` |
| v8 label md5 | `eefc38d1453bd1c73802d44d45affced` |
| A7 complete | 4/4 arms VALID **and** zero live A7 processes |
| boxstat gate | GPU ≤ 2,500 MiB **and** host ≥ 8 GB, **unchanged** |

## 5. The arm check — what makes this resumable, and why it had to exist

`p_check_arm.py` writes `p_arm_check.json`. ⛔ Without it a **finished 11-hour arm reads as
PARTIAL and the next invocation moves it aside.** It asserts on the **artifact**, never on the
trainer's exit code: `run/config.json`'s recorded argv must carry the expected seed and steps; the
last eval row must sit at the expected step; the prereg's label-density identity
(`n_matched == n_target`) must hold; and the headline must be finite.

**Validated against a REAL finished arm, both directions** (on a scratch copy — nothing was written
into another arm's directory): A8's own run reads **VALID** with
`headline_eval_box3d_centre = 13.92835` and `identity_matched_equals_target: true`; the same run
checked against a **wrong expected seed** reads **INVALID** with `seed mismatch: config says 0,
expected 1`. A checker that only ever passes is not a checker.

**13.92835 is the value P0 replicates against**, at step 5000, seed 0.

## 6. State

| | |
|---|---|
| runner | **ALIVE**, PID 23224 (+child 46172, one launch), polling every 120 s, 72 h ceiling |
| current decision | `ZZP-WAIT P0-REPLICATE \| A7 arms not VALID yet: [4 arms]` |
| GPU | **3,950 MiB** held by the PI's servers — the gate is doing its job, not failing |
| A7 | launcher armed and waiting; **not started** |
| tests | **47 passed bare** (test_p_runner 32, test_prebuild_p3_targets 15), no ambient `PYTHONPATH` |
