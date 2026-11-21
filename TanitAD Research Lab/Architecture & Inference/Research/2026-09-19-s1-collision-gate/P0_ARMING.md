# P0-REPLICATE armed — the boundary, the preconditions, and a defect found while arming

**Agent:** TanitAD_TrainingFlyWheel · **2026-09-20** · authorisation: Master Mind, *"ARM THE RUNNER
FOR **P0-REPLICATE ONLY** … it must not roll on to P1 unattended."*

⛔ **No arm has run.** The runner is alive and WAITING on A7; the GPU is still held at ~3,950 MiB by
the desktop session (see §8 — NOT the PI's servers, which was my own unverified assumption)
and the gate is untouched.

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
| runner | **ALIVE**, PID 52324 (+child 46140, one launch), polling every 120 s, 72 h ceiling. ⚠️ Superseded the PID-23224 process: stopped it, verified **0** live by PID, edited, re-proved, restarted — never edited under a live run |
| current decision | `ZZP-WAIT P0-REPLICATE \| A7 arms not VALID yet: [4 arms]` |
| GPU | **3,111–3,958 MiB** held by the **Windows desktop session**, not by any training job (§8). The gate refuses correctly — but it can never clear in this state |
| A7 | launcher armed and waiting; **not started** |
| tests | **52 passed bare** (test_p_runner 37, test_prebuild_p3_targets 15), no ambient `PYTHONPATH` |
| mutation proof | **12/12 CAUGHT, control GREEN** (29 test functions) |
| bound | **P0 -> P0b, then STOP** (was P0 only) |

---

## 7. AMENDMENT (same day) — the bound is extended to **P0 -> P0b**, then STOP

The Master Mind accepted §3's limitation and authorised the fix rather than living with it:
*"extend the bound to P0 -> P0b, then stop … |P0 − P0b| on the same tree, differing only in seed,
IS the seed floor."* ~22 h of card for two arms, against a 3.2–3.6 day panel whose every result
depends on the floor being real.

**The reporting rule, committed NOW so it cannot drift once the numbers exist:**

| quantity | how it is reported |
|---|---|
| **\|P0 − P0b\|** | same pinned tree, seeds 1 vs 2, nothing else moved ⇒ **THE SEED FLOOR**. Quote this. |
| **P0 vs A8** | ⛔ an **UPPER BOUND**, never called the floor. Reported beside it, labelled — the gap between the two **is the size of the code delta**, which is worth knowing on its own. |
| ordering | the floor is reported **BEFORE** anything is interpreted against it, and **P1 does not start until both have landed**. |

**What changed in the runner:**

* `P0B-REPLICATE` inserted as `ARMS[1]` — **both replicates precede every lever**, so no lever can
  run before the floor exists. Seeds are **1** and **2**; A8 is **0**, and a test asserts all three
  are distinct (*a replicate sharing A8's seed is not a replicate*).
* `--authorise-arm` now **repeats**, and it is a **SECOND, INDEPENDENT LOCK**: an arm must pass the
  ordering bound **and** appear in the explicit list. A test proves each refuses alone — P0b is
  refused when the list omits it even though the bound allows it.
* The runner no longer `return`s after an arm; it `continue`s **back through `plan()`**, so a
  second arm can only start by passing **every** lock again from scratch. An INVALID check still
  returns STOP there.
* ⛔ **New livelock guard, and it protects 11 h of card.** An arm that finished but whose check did
  not read VALID comes back as **PARTIAL** — and `plan()` would hand it back, whereupon
  `move_aside` would rename away the run just paid for, forever. One attempt per arm per process;
  a second is a **STOP** with the reason named. *(This hazard only appeared once the runner was
  allowed to loop; the single-arm build returned before it could bite.)*
* `build_argv` is asserted to make the two replicates **differ in `--out` and `--seed` ONLY** —
  every remaining token identical.

**Debt 1 discharged:** the name **`pgrep -f` self-match trap** is back in `count_a7_procs`, now
beside a real guard rather than in place of one. Debt 2 (`--max-wait-h` 48 -> 72) stands as a
deliberate ceiling change, and is 72 in this build.

⚠️ **The live process was NOT edited under itself.** Sequence: `TaskStop` → **verified 0 live
by PID** (the "successfully stopped" message is a claim, the PID count is the evidence) → edit →
re-prove → restart. Nothing was running, so the restart cost nothing. The landed copy is again
byte-identical to the tree the new process runs from.

**Re-proof after the amendment:** **52 tests pass bare**, and the mutation proof is **12/12
CAUGHT with a GREEN control** (29 test functions) — three new mutations for the new locks:
`M10` the two replicates share a seed · `M11` an empty authorisation list launches anyway ·
`M12` P0b is dropped so the floor collapses to one arm.

---

## 8. ⚠️ CORRECTION + a blocker that will NOT clear on its own (MEASURED 2026-09-20 10:15 CEST)

⛔ **I reported the card as "held by the PI's servers" in three places. That was INHERITED, never
verified, and it is WRONG.** `nvidia-smi --query-compute-apps` names every holder, and not one is a
training job:

`dwm.exe` (the desktop compositor) · `explorer.exe` · `SearchHost` · `StartMenuExperienceHost` ·
`ShellExperienceHost` · `ShellHost` · `CrossDeviceResume` · **`NVIDIA Overlay.exe` ×2** ·
**`msedgewebview2.exe`** · `GCC.exe`.

**Total 3,211 MiB of 8,188 MiB; ~4,977 MiB free; GPU utilisation 15 %.** This is the **Windows
desktop session**, browser and NVIDIA overlay — nothing of the PI's, and nothing I may assume is
transient. *(Evidence class: the earlier claim was INHERITED and is retracted; this one is
MEASURED, artifact = `nvidia-smi` compute-apps + per-GPU query.)*

⛔ **AND THE GATE HAS NEVER BEEN CLOSE.** Over the launcher's **19** gate samples spanning ~9.4 h:
**min 3,111 MiB**, median 3,922, max 3,958, and **0 of 19 (0 %) at or below the 2,500 MiB gate** —
never within **600 MiB** of clearing. The P-runner has logged **172** consecutive WAITs.

⇒ **This is not a wait, it is a deadlock.** `A7-launcher` carries a **24 h ceiling** and expires
**~2026-09-21 00:52** (~14.5 h from this measurement); the P-runner's 72 h ceiling expires
~2026-09-23 04:20. On the current trajectory the launcher times out, A7 never runs, and P0/P0b then
wait out two more days for an A7 that cannot come. ⭐ **A wait that never ends looks exactly like a
wait that is working** — the same sentence as §2, now about the *resource* rather than the *probe*.

**Two ways out, and NEITHER is mine to take:**

| option | whose call | note |
|---|---|---|
| **quiesce the desktop** (close the Edge WebView / NVIDIA overlay, or sign the session out) | the **user's** — it is their desktop | would need to free ≥ 611 MiB from the observed minimum |
| **re-calibrate the gate** for a desktop box rather than a headless one | the **Master Mind / PI** | ⛔ I was told twice not to lower it, and I have not |

⚠️ **What I could NOT determine, stated rather than guessed:** the arm's actual VRAM need. A8's
`metrics.jsonl` logs **no GPU-memory key** — `ga_box_memory` is the box-memory module's loss/param
count (91,136 params), **not** VRAM, and reading it as memory would be exactly the wrong-scope
error this programme keeps logging. So whether an arm fits in the ~4,977 MiB that remains is
**unmeasured**, and the gate cannot be re-calibrated on evidence until it is.

## 9. MEASURED — the arm's peak VRAM, and what it says about the gate

§8 said the arm's footprint was **unmeasured** and that the gate could not be re-calibrated on
evidence until it was. It is now measured. *(PI chose this option directly, 2026-09-20.)*

**Instrument:** `code/vram_probe.py`. The real arm config from the **pinned** tree, argv built by
**`p_runner.build_argv`** — the same function P0 will use, so the probe cannot drift from the arm
it measures — with **seed 99** (never 0/1/2 = A8/P0/P0b) into a scratch directory, 12 steps and a
forced eval at step 10. Artifacts: `raw/vram_probe/`.

⛔ **Only the in-process allocator counter is admissible.** `nvidia-smi`'s total includes the
desktop, which swings **847 MiB** on this box; you cannot answer a question that turns on ~600 MiB
by subtracting a baseline that moves by more. The trainer's own
`gp_cuda_max_mem_gb = torch.cuda.max_memory_allocated()` is the right probe, and it is cumulative
since process start, so the step-12 row also covers the eval path.

| quantity | value | class |
|---|---|---|
| allocator peak, step 10 | **2.5122 GB = 2,573 MiB** | **MEASURED** |
| allocator peak, step 12 (after the eval) | **2.5122 GB** — *identical* | **MEASURED** |
| total process footprint on the card | **~2,939 MiB** (a mid-run `nvidia-smi` 6,140 minus the 3,201 post-exit baseline ⇒ ~366 MiB of CUDA context/reserve above the allocator figure) | **ESTIMATED**, single sample |
| probe wall-clock | 338 s | MEASURED |

⭐ **The peak PLATEAUED**: identical at step 10 and step 12, across the eval boundary. The
dominant allocations (weights, optimizer state, activations) are all taken by step 10.
⚠️ It remains a **LOWER BOUND** for a 5,000-step run — 12 steps cannot exclude a later
fragmentation peak — but the flatness is the evidence against one, and it is stated rather than
assumed.

### The gate arithmetic

Card **8,188 MiB**. Desktop baseline over 19 samples: **min 3,111 / median 3,922 / max 3,958**.

| desktop | + arm (~2,939) | headroom of 8,188 |
|---|---|---|
| min 3,111 | 6,050 | **2,138 MiB** |
| max 3,958 | **6,897** | **1,291 MiB (15.8 %)** |

⇒ **THE ARM FITS AT EVERY OBSERVED DESKTOP LEVEL.** The `≤ 2,500 MiB used` gate is not measuring
the quantity that matters: **the desktop alone exceeds it**, so the gate rejects a configuration
that would in fact run. It was calibrated for a headless box, and this box has a desktop on it.

⛔ **I have NOT changed it.** A gate is a safety device and re-calibrating one is the PI's call,
not the measuring agent's. For that decision, the evidence supports expressing it as **FREE VRAM**
rather than used: `free ≥ arm (2,939) + margin (~900, the desktop's own observed swing)` ⇒
**free ≥ ~3,850 MiB, i.e. used ≤ ~4,300 MiB**, which would have cleared at **all 19** observed
samples (max 3,958).

⚠️ **The residual risk, named:** at the worst observed desktop level the headroom is 1,291 MiB
over an ~11 h run. A desktop spike mid-run (more browser tabs) could still OOM the arm. That is a
real trade the PI is entitled to weigh — it is not a reason to pretend the current gate is right,
and not a reason for me to move it.
