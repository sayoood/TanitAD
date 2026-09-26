# Can every standard eval actually RUN? — a reachability audit of `python -m taniteval.bench`

**Package** `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-suite-runnability-audit/`
**Owner** EvalFlyWheel orchestrator · **Date** 2026-09-26 (Europe/Berlin)
**Asked by** the PI: *"assure that all the standard evals can be run in our eval suite"*; and by the
Master Mind, relaying the PI 2026-09-23: *"prepare the required standard tests including the navsim
suite for refcv6"*.

⛔ **The question is RUNNABILITY, not existence.** `CLAUDE.md` records five — now six — instances of
*"built, tested, and unreachable from its caller"*, three of them found only after an arm had been
trained on the unreachable thing. So every row below is a **verdict from the real CLI**, not a reading
of the source.

## §1 The audit — MEASURED 2026-09-26 by invoking the CLI

`python -m taniteval.bench --help` exposes **navsim_v2, navsim_v1, nuscenes_ol, internal_t1** plus
`submit / validate / reaggregate / gpu-gap / tombstone`. ⚠️ It must be run with cwd `taniteval/` (or
`PYTHONPATH`): from the repo root `taniteval` resolves to the outer namespace directory and dies with
`No module named taniteval.bench` — the namespace-shadow trap, and the thing that killed W7's
post-run step on 09-20.

| benchmark | verdict | evidence |
|---|---|---|
| **navsim_v2** `warmup_two_stage` | ✅ **DRY_RUN_PASSED** 1.2 s | floors CV+STOP added by rule |
| **navsim_v1** `navtest` | ✅ **DRY_RUN_PASSED** 3.0 s | *"12146/12146 scenes in the metric cache"* |
| **nuscenes_ol** `val` | ⚠️ **REFUSED — by design** | *"runs ONE convention per run and has no default: set `TANITAD_NUSCENES_PROTOCOL`"*. A deliberate refusal (ST-P3 and UniAD are different quantities), **not** a defect. The real blocker is data: the download needs the PI's registration. |
| **internal_t1** | ✅ reachable; needs `--ckpt --episodes --labels` | device gate passed on CPU (*"explicit --device cpu and no training process is alive"*); ran 539.8 s before **my own `timeout`** killed it |

⚠️ **`internal_t1` is the refcv6-critical one and it is NOT too slow because of a defect — it is too
slow on CPU.** 2 episodes did not finish in 9 minutes at 416x1024, so the 139-clip split needs the
GPU. See §4.

⭐ **All four splits' data is present on the dev box.** `v2ep-eval139-416x1024cyl` holds **139/139**
clips (11 GB) and `s2_labels_v7.2_eval.jsonl.gz` is banked — so refcv6's held-out eval needs **no**
transfer from Thor.

## §2 ⛔ The one real defect, and it was on refcv6's critical path — FIXED

`DEFAULT_MEM_LIMIT_MIB = 1024` was hardcoded, and **`limit_mib` was reachable in every signature and
passed by nobody.** Verified with same-breath controls that must read non-zero:

| probe | result | control |
|---|---|---|
| CLI flag for the limit | **0** | `--ram-floor-mb` → **1** |
| `os.environ`/`getenv` in `gpu_gap.py` | **0** | `OVERRIDE_FLAG` → **8** |
| call sites passing `limit_mib` | **0 of 2** (`contract.py:466`, `navsim/benchmark.py:292`) | — |

⇒ the gate always ran at 1,024 MiB. MEASURED: with **no training process at all**, this desktop's
WDDM compositor holds **1,175–6,085 MiB**, so the gate could never open — W7 sampled **0 gaps in 12
samples** and every model arm silently fell back to CPU, which for a 416x1024 ResNet is ~100x too slow.

**Fix:** `--gpu-mem-limit-mib` + `$TANITAD_GPU_MEM_LIMIT_MIB`, resolved by
`resolve_mem_limit_mib()` (explicit > env > 1024) at the impure boundaries only — ⛔ `gap_verdict`
stays the pure, literal-tested predicate and never reads the environment.

⭐ **The proof is DIFFERENTIAL, which is stronger than a pass/fail.** Same command, same box, one
flag changed:
* `--gpu-mem-limit-mib 4300` → *"training process alive: pids [...]; **GPU memory used 6058 MiB >= limit 4300 MiB**"*
* `--gpu-mem-limit-mib 7000` → *"training process alive: pids [...]"* — **the memory clause is gone and nothing else changed.**

**Controls (19 tests, `taniteval/tests/test_gpu_gap_limit_reachable.py`), all literal expectations:**
a malformed / `0` / `inf` env value is IGNORED (a gate reading 0 refuses forever; one reading `inf`
never refuses); the CLI default is `None`, not 1024, so the env still applies; **5,812 MiB is still
refused at limit 4300**, so the gate remains a real threshold; and ⛔ **the discriminating control —
a live training pid REFUSES at `limit_mib=10**9` with zero foreign memory.** That invariant was also
confirmed in production, not only in the test: four `diag_onpolicy_train.py` pids are alive on this
box right now and both flag values refuse.

## §3 ⚠️ Two gates now answer one question — a convergence item, not a bug

The refcv6 battery ships **its own** gate, `battery/code/gpu_gate.py`, and it is the better-specified
one: `used < 4300 MiB` **and** no other python compute on the GPU **and** free host RAM >= 8 GB — the
PI's policy exactly, plus a documented fix for a self-gating bug (it had waited on its own CUDA
context; `evaluate()` now drops the caller's pids — the `pgrep -f` self-match family).

Its live verdict, MEASURED: `{"ok": false, "gpu_used_mib": 6085, "python_compute": [2 foreign pids],
"free_ram_gb": 7.15}` — a correct WAIT.

⇒ The battery does **not** have §2's defect. But `gpu_gap.py` calls itself *"the ONE definition"* for
training processes while a second definition of *"may I use the GPU"* now lives beside it. §2 makes
convergence possible (the suite can finally be told 4,300); ⛔ the suite gate still has **no host-RAM
clause and no foreign-compute clause**, which the battery's does. **Escalated to W1 / the Master
Mind** rather than written into a README.

## §4 ⛔ What actually blocks refcv6's eval tomorrow — the box, not the harness

refcv6-r101-s0 finishes on Thor ~2026-09-27 19:30 Berlin. Both gates refuse **right now**, correctly:

* GPU **6,085 MiB** used of 8,188 — held by another session's `diag_onpolicy_train.py` (refe-plan, 4 pids);
* free host RAM **7.15 GB** < the battery's 8 GB floor.

⇒ The final battery will **WAIT**, not fail. The harness is ready; the *box* is not. This needs a PI
scheduling decision: finish/pause the refe on-policy training, or provision a pod. ⚠️ CPU is not a
fallback here — §1 measured the CPU rate.

## §5 The leaderboard omits the community headline it was asked about

The Master Mind quoted **DiffusionDrive navtest PDMS 88.1**. ⭐ **That number is correct and banked** —
`products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md:593`, primary `2411.15139` Tab. 1, cross-checked
against the live leaderboard at **88.0157**. ⚠️ I nearly reported it as misattributed: my first probe
searched only `published_results.json`, where DiffusionDrive appears **only** under
`EPDMS_v2_navtest_single_stage` (84.5 / 85.5 / 87.5, camera **+ LiDAR**). A second probe found the v1 row.
*Absence found at one location is not absence* — again.

⛔ **But the asymmetry is a real gap:** `published_results.json` is what the leaderboard GENERATOR
consumes, and it holds **16** `PDMS_v1_navtest` rows with **DiffusionDrive not among them** (Hydra-MDP
is, at 91.3). So the generated leaderboard omits the headline. **Work item for W4.**
⚠️ And flag, do not adjudicate: our banked Hydra-MDP is **91.3** while DiffusionDrive's paper claims
**+1.6 over Hydra-MDP** at 88.1 — implying ~86.5. Different variants/backbones are the likely
explanation; it needs the primaries side by side, not a guess.

## §6 My own errors this turn — one class, three instances

⛔ **A filter's output read as the world.**
1. I reported `internal_t1`'s `--episodes`/`--labels` as **absent from the CLI** and called it "a real
   gap on tomorrow's critical path". **False.** Both are exposed; my `sed` range had truncated
   `--help` and I read the truncation as absence. The grep control (episodes 2, labels 1, vs `--arms`
   2) settles it. Its `FAILED` verdict was correct behaviour — it named exactly the missing flags.
2. "5 scoring processes alive" and "waiter = 5 procs" were **my own shells and the query itself**
   matching the search string — the `pgrep -f` trap, twice.
3. I wrote the new test into `taniteval/taniteval/tests/` (inside the package) instead of the real
   suite at `taniteval/tests/`, then ran `pytest taniteval/tests` from cwd `taniteval/` and saw
   **19 passed** — my file alone — and nearly recorded that as "the suite is green". ⭐ The tell was
   arithmetic: the suite collects **2,382** tests. Moved; 54 pass with the 35 pre-existing gate tests.

⇒ In all three the probe's scope was mistaken for the fact. The cheap discriminator each time was a
**control that had to read a known non-zero value**, and it is the same lesson as §5.

## §7 Manifest

| artifact | where |
|---|---|
| this audit | `repo:FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-suite-runnability-audit/RESULT.md` |
| the fix | `repo:taniteval/taniteval/bench/gpu_gap.py`, `cli.py`, `contract.py`, `navsim/benchmark.py` — staged, **not committed** |
| 19 tests, literal + mutation + the training-invariant control | `repo:taniteval/tests/test_gpu_gap_limit_reachable.py` |
| refcv6 battery (SIBLING stream, already built — do not duplicate) | `repo:FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery/` — 610 files, pre-registered SPEC `32625b9a…`, G0-A2 PASS at step 5000, chain queued for 30000 + FINAL |
| navhard stage-1 frames (the 09-20 unlock) | **dev box only**: `C:/Users/Admin/tanitad-caches/navhard-stage1-frames-20260920` — `COMPLETE: 3375/3375`, 76/76 logs, 32/32 receipts |

## §8 ⛔ THE SUITE IS NOT GREEN — 32 failing tests on the tip, and this gates the COMMIT decision

MEASURED 2026-09-26, the full run (43m 09s): **31 failed · 2,380 passed · 27 skipped**
(32 failing node ids in `.pytest_cache`).

⛔ **And the wrapper reported `exit code 0`.** The command ended `… | tail -15`, so the status came
from `tail`, not from pytest — the `$?`-through-a-pipe trap that `CLAUDE.md` documents twice. I was one
sentence from reporting the suite green off a number that was never pytest's. ⚠️ The log file is also
**tail-truncated** (1,500 bytes, showing 9 of 31), so the enumeration below comes from
`.pytest_cache/v/cache/lastfailed`, not from the log.

| failing file | node ids |
|---|---|
| `tools/tests/test_release_gate.py` | **17** |
| `taniteval/tests/test_render_openloop_video.py` | **9** |
| `tools/tests/test_registry_paths_allow.py` | 2 |
| `taniteval/tests/test_benchreport_legacy_compat.py` · `tools/tests/test_library.py` · `test_library_tracking.py` | 1 each |

**Not caused by §2's fix**, established positively rather than by inference:
* **0 of the 7** failing files import `gpu_gap`, `bench.cli`, `bench.contract` or `navsim.benchmark`
  — with the control that **all 7** import *something*, so the grep is working;
* the fix's own neighbourhood (gate + contract + reuse-seams + reuse-floors) passes **112 tests**;
* the release-gate failure is a fixture/policy mismatch — the gate prints
  `VERDICT: ADVISORY-FAIL (1 registry-named criterion)` under the PI's 2026-08-28 advisory ruling while
  `test_a_clean_release_exits_zero` expects 0 — and the nine others are video-rendering tests.

⭐ **Why this was invisible: every recent report measured a SUBSET.** W7 banked *"323 taniteval and 186
checker tests pass"*, W3 *"the suite is 94/94 green"* — both true of what they ran, and neither is the
suite. `CLAUDE.md` requires `pytest -q` green before any commit, so **the PI's pending commit decision
now carries a precondition**: 32 tests to fix or explicitly waive, in a tree whose HEAD has not moved
in five days (`37645fc`) while **2,820 paths sit staged**.

⚠️ I have NOT fixed them: they span seven unrelated concerns, none is on refcv6's critical path, and
guessing at another stream's fixtures under a deadline is how the release gate got an expected value
of *"whatever the code does"* in the first place. They are named here as a work item with their
enumeration, which is the honest form.

## §9 ⛔⛔ THE A16 RESUME SILENTLY CHANGES THE EVAL'S LABEL CLOCK — flag before the FINAL, not after

**Context (INHERITED from the Master Mind, 2026-09-26):** the PI has ruled refcv6-r101-s0 **stops and
resumes with the A16 fixes** (commit `82c2331`); F3's cascade loss *"had never run"* and tactical labels
were *"0.37 s early"*. Checkpoints from the switch step onward are a **hybrid**.
⚠️ `82c2331` is **NOT present locally** (HEAD is `37645fc`, five days old), so its exact file list is
**UNVERIFIED by me**. Everything below is read from the frozen-trunk audit's own artifact.

⭐ **"F3's cascade loss had never run" is the SEVENTH instance of the `CLAUDE.md` class this audit is
about** — built, tested, and unreachable from its caller — and it ran for ~3 days of GPU before anyone
noticed. That is the same defect family as §2, in a trainer instead of a gate.

### The eval consequence nobody has stated

The 0.37 s defect is **in `V3Dataset`**, not in the trainer:
*"`V3Dataset` evaluates the label band at `t_now = (t + w - 1) * 0.1` … while the labels live on the RAW
clip timeline"* — and its fix is also in `V3Dataset`
(`t_now = grid_start_s + (t + w - 1 + n_stack - 1) * dt_s`, plus a new `tanitad/data/clip_clock.py`).

⛔ **The eval path SUBCLASSES that class:** `taniteval/tools/refcv3_arm.py:1707` —
`class EvalV3Windows(tr.V3Dataset)`. The battery's `refcv6_loader.py`, `refcv6_roll.py`,
`check_pairing_surface.py` and `reproduce_inrun_eval.py` all reference it too.

⇒ **The eval's tactical label clock changes the moment the battery runs against `82c2331`** — for
**every** checkpoint it scores, including the pre-switch milestones (5,000 and 30,000) that were
*trained* under the OLD clock. Magnitude, already MEASURED by the audit on the eval split:
**598 / 5,699 windows (10.5 %)** change band membership (train side: 19,044/179,129 = 10.6 % outside
the true band, 13,315 = 7.7 % wrongly unsupervised).

| milestone | trained under | will be scored under | consequence |
|---|---|---|---|
| 5,000 · 30,000 | old clock (0.37 s early) | **corrected** clock | train/eval MISMATCH — penalised on the tactical family specifically |
| FINAL (hybrid) | part old, part corrected | corrected | closest match, but not a single arm |

⇒ ⛔ **The TACTICAL family — one of the four mandatory families — is not comparable across the switch
unless the label clock is pinned as a recorded arm property.** And a cross-milestone learning curve is
no longer one trajectory: it spans two experiments, the *"a derived constant silently changed the
experiment, so the reproduction is not a replay"* class.

**Recommendation to the battery stream (its SPEC amendment discipline is already correct — every
amendment so far was written BEFORE the measurement it governs):**
1. Record the **label clock** (old / corrected, with the `dt_s` source) in every battery result, beside
   the tier and loop stamps. An arm property that is not in the manifest is invisible in six weeks.
2. Score **all** milestones with the **corrected** clock so the EVAL is at least internally consistent,
   and state plainly that the pre-switch checkpoints were trained under the old one.
3. ⛔ Do not publish a tactical comparison across the switch step without that stamp, and do not fit a
   learning-curve exponent across it at all.
4. Name the switch step explicitly; "from the switch step onward is a hybrid" is not a number.

## §10 Master Mind items 4 and 5 — DONE

**Item 5 — the one-copy frames now have a checksum manifest.**
`…/2026-09-20-navhard-stage1-extract/raw/MANIFEST_stage1_frames.json`, written by `code/write_manifest.py`:
**3,375 files · 76 logs · 711,722,316 bytes**, sha256 per file, keyed by the loader's own relative
`data_path` so a future check verifies what the scorer will actually open. Independently cross-checked
the same hour: `--verify` still reads `COMPLETE: 3375/3375`. ⭐ The two answer DIFFERENT questions and
both are kept — `--verify` = *"is every file the scorer asks for present and a valid JPEG?"*; the
manifest = *"are the bytes still the bytes we extracted?"* Moving the set anywhere (e.g. private HF) is
a quota/spend question for the PI and has NOT been done.

**Item 4 — DiffusionDrive added to `PDMS_v1_navtest`, and upgraded from INHERITED to PRIMARY.**
`products/P7-TanitEval/benchmarks/published_results.json`, row `v1.diffusiondrive`, 113 → **114** rows
with a superset guard: **0 of the 113 original rows changed** (compared by sorted-key serialisation, not
by "the file differs"). The file has no HEAD version — it is new since 09-19 — so my three-blob check
correctly read INCONCLUSIVE; index == worktree.

⭐ **Verified in the banked PDF rather than inherited from our own prose:** `2411.15139`, p.1 abstract
and p.2 both state **88.1 PDMS**, p.2 naming the **navtest split** and the **aligned ResNet-34
backbone**; **Tab. 1 is on p.6**, captioned *"Comparison on planning-oriented NAVSIM navtest split with
closed-loop metrics"*. Cross-checked against the live leaderboard at **88.0157**.

**The Hydra-MDP question, recorded and largely resolved — but not adjudicated in the data file.**
`NAVSIM_PROTOCOL.md` carries the variant ladder two lines above DiffusionDrive: `Hydra-MDP-V8192` 83.0,
**`Hydra-MDP-V8192-W-EP` 86.5**, `Hydra-MDP++` 86.6 (camera only) — all `2406.06978v4` Tab. 1. So
DiffusionDrive's *"+1.6 over Hydra-MDP"* at 88.1 matches **86.5 = V8192-W-EP**, consistently. ⚠️ What
remains open is **this file's own Hydra-MDP row at 91.3**, whose source is `2406.15349` Tab. 3 p.9 — the
**NAVSIM** paper, a different artifact. Which number belongs in the column is W4's to settle from the two
primaries; the row's `notes` field says exactly that and adjudicates nothing.

## §11 ⛔ Correction to §8 — the count is 31, and my "order-dependence" claim was WRONG

I wrote in §8 that `test_leaderboard.py` had one failure, and then — seeing it pass **76/76** in
isolation — concluded that *"at least one failure is order-dependent"*. **Both were wrong, from the same
cause: I enumerated from `.pytest_cache/v/cache/lastfailed`, which carries entries from EARLIER runs.**

MEASURED: the cache holds **32** node ids; **exactly 1 no longer resolves** —
`test_leaderboard.py::test_the_print_convention_block_keeps_the_probes_own_verdict_and_scope`, whose
function was **renamed** to `…_block_prints_…` (control: the file defines 69 `test_` functions, so the
grep works). **32 − 1 stale = 31, which reconciles exactly with the run's `31 failed`.**

⇒ **`test_leaderboard.py` is fully green (76/76) and was never among the failures.** There is **no
evidence of order dependence** — I invented it to explain a discrepancy whose real cause was a stale
cache. The corrected table is six files, 31 ids: release_gate **17**, render_openloop_video **9**,
registry_paths_allow **2**, and benchreport_legacy_compat / library / library_tracking **1 each**.

⭐ **Class — the third instance in this one audit, and they are all the same mistake:** a truncated
`--help` read as a missing flag (§6.1), a process list matching my own shells read as running jobs
(§6.2), and now a stale test cache read as this run's failures. **The artifact of a probe is not the
world.** Each time, the cheap discriminator was arithmetic or a must-read-nonzero control — here, that
31 ≠ 32 was visible from the start and I let it pass unchased for two messages.
⚠️ `tools/tests/test_library_tracking.py::test_every_banked_pdf_is_in_head` is very likely a GENUINE and
fully explained failure: HEAD is five days old while PDFs have been banked since, so "every banked PDF
is in HEAD" is false **because of the commit backlog**, not because of a defect. Classifying the
remaining 30 is still the work item; ⛔ a count is not a diagnosis.

## §12 ⭐ A5 (dual-clock tactical scoring) IS implementable — costs ZERO extra inference, and has one trap

MEASURED 2026-09-26 in the A16 fix package
(`…/2026-09-26-refcv6-frozen-trunk-audit/code/fix/stack/scripts/refc_v3_train.py`):

1. ⭐ **The old clock is still reachable.** `_now_s` (`:3005`) carries
   `if self.legacy_label_clock: return r * float(self.v7_dt)` — **exactly** the historical defect
   `(t + w - 1) * 0.1`. ⚠️ I first concluded the opposite: `--clip-clock-sidecar` is the only
   clock-matching CLI flag, and without it dt still comes from the clip's poses, i.e. the CORRECTED
   formula either way. The legacy path is a **dataclass field**, not a flag.
2. ⛔ **`legacy_label_clock: bool = False` (`:2983`) and there is NO CLI flag** — `add_argument` hits
   for it: **0** (control: 205 `add_argument` calls in the file). It is reachable programmatically only.
3. ⭐ **The clock moves LABELS ONLY, not model inputs.** `_now_s` has exactly **two** call sites
   (`:3097`, `:3117`), both on the label-band path. ⚠️ Stated as a positive assertion over call sites,
   **not** an exhaustive trace of `enable_nav_from_v7`; nav is not among them, which is consistent with
   the nav input being clock-independent but does not prove it.

⇒ **A5 costs NO extra rollout.** The model's outputs do not depend on the clock, so one inference pass
per checkpoint is scored TWICE — once with `legacy_label_clock=False`, once `=True`. What A5 needs is
for the battery to SET that field (or for a flag to be added), which is the one wiring step.

### ⛔ The trap, and the control that catches it

`legacy_label_clock` defaults to **False** with no CLI route — the textbook *"a flag whose default
disables it"* hazard from `CLAUDE.md`. If the battery forgets to set it, **A5 still produces a
"both clocks" table: the same clock run twice, printing two identical numbers that look like
agreement.** That is worse than a missing row, because it reads as evidence.

⭐ **Required control: the two clocks MUST disagree on a known, non-zero number of windows.** The
frozen-trunk audit already measured the expected magnitude on this very split — **598 / 5,699
(10.5 %)** change band membership. So the assertion is a literal, not a vibe:

* legacy vs corrected band assignment must differ on **~598** eval windows (and on **> 0** always);
* if the two columns are identical, the run must be REFUSED, not published — the switch did not take.

⚠️ And the mirror-image control is just as cheap: scoring the SAME clock twice must produce
**bit-identical** tactical numbers, which proves the comparison harness is deterministic and that any
difference seen above comes from the clock and not from run-to-run noise.

## §13 The §11 work item, CLOSED — and ONE test is 61 % of the suite's runtime

Isolation pass (each node id run alone), MEASURED 2026-09-26:

| node id | alone | verdict |
|---|---|---|
| `test_every_banked_pdf_is_in_head` | **FAILED in 1,585.8 s (26m 25s)** | genuine |
| `test_real_registry_has_no_dead_or_malformed_citations` | FAILED in 32.4 s | genuine |
| `test_importing_the_package_is_cheap` | FAILED in 0.9 s | genuine |
| `test_the_print_convention_block_keeps_…_verdict_and_scope` | **no tests ran** | the stale cache id (§11) |

⇒ **Every failure that resolves also fails in isolation. There is no order-dependence anywhere** —
which closes §11's retraction properly: the appearance of it came entirely from the stale cache entry.

### ⭐ And here is the root cause of the thing §8 complained about

**`test_every_banked_pdf_is_in_head` alone takes 26m 25s of the full suite's 43m 09s — 61 % of the
total runtime in ONE test.** That is why *"every recent report measured a SUBSET"*: the suite is not
usable as a pre-commit gate at that cost, so each stream ran the slice it cared about and the 31
failures stayed invisible. ⛔ **The subset habit is not carelessness; it is a rational response to a
43-minute gate**, and it will not change by asking people to be more thorough.
⇒ Work item, and it is the one with leverage: make that test cheap (it appears to shell out to git
per banked PDF — a single `git ls-tree`/`cat-file --batch-check` over the set would answer the same
question in seconds), or mark it `slow` and run it in CI rather than pre-commit. Fixing this one test
is what makes `pytest -q` a gate anybody will actually run.

⚠️ **Do not re-run it on this checkout to see if it is fixed.** It asks *"is every banked PDF in
HEAD?"* and my local HEAD is `37645fc`, five days stale — the Master Mind has since landed the Library
backlog (`1767cfb`) and my batches (`57a2a835`) upstream. Locally it must fail for the backlog reason
regardless, so 26 minutes here would buy a knowably-wrong answer. It should be re-read on a tree that
has fetched.

## §14 The slow gate, FIXED — and it was a CORRECTNESS bug wearing a performance costume

**Batch 5.** `tools/tests/test_library_tracking.py`: the per-file `git cat-file -e` loop is replaced by
ONE `git cat-file --batch-check` over every banked path.

| | before | after |
|---|---|---|
| `test_every_banked_pdf_is_in_head` | **1,585.8 s (26m 25s)** | — |
| the HEAD membership query itself | — | **0.53 s** for 552 paths, one process |
| the WHOLE file (now 7 tests) | — | **2.13 s** |

⇒ ~**750x** on the file, and the 43-minute suite loses 61 % of its runtime.

### ⛔ But the speed was a symptom. The real defect was the classifier.

The old code called a path "missing" only when stderr contained `does not exist` or
`Not a valid object`. **git's actual message for the case this test exists to catch is:**

```
fatal: path 'TanitAD Research Lab/Library/papers/<x>.pdf' exists on disk, but not in 'HEAD'
```

which matches **neither**. So every genuinely-unbanked PDF was filed as **UNDECIDED** and the gate
failed with *"the mount was flapping … re-run"* — when the true, actionable diagnosis was
**"commit these 86 files"**. ⭐ It failed loudly and named the wrong cause: a
**true-but-wrong-for-the-reader** failure, the class already in auto-memory. And the misclassification
IS the slowness: 86 missing x 6 attempts x 3 s = **25.8 min of pure sleeping**, which reconciles with
the measured 26m 25s to within the subprocess overhead.

⭐ **The durable part of the fix is that it no longer parses English.** `--batch-check` answers in a
protocol — `<sha> blob <size>` or `<input> missing` — so no wording, locale or git-version change can
reintroduce this. ⛔ And not `ls-tree -r`, per `CLAUDE.md`: it truncates on the G: mount, exits 0, and
truncates *consistently*, so the answer would depend on where the tree sits.

### The arms that keep it honest (all 6 green; the 7th is the gate itself)

* ⛔ **Deliberate regression:** a real on-disk PDF absent from HEAD must read **missing**, never
  undecided. ⭐ Written against a REAL temp file, because a fabricated name produces a *different* git
  message — a fictitious path would not reproduce the bug and the arm would pass while the defect lived.
* ⭐ **Control that the batch ANSWERED:** present + missing + undecided must equal the input count, and
  deciding **0 of 552** must REFUSE as INCONCLUSIVE rather than read as "nothing is missing" — the
  2026-09-05 false finding in reverse.
* **Control that it is not inert:** `present > 0`, and a paper matching a `library.json` key is found.
* **A speed bar** at 60 s: loose enough for a slow CI box, far below the 25.8 min sleep storm.

### Verdict on each tree, stated separately

* **My tree (HEAD `37645fc`, five days stale):** the gate **FAILS**, naming **86** PDFs — correct, and
  now correctly diagnosed.
* **The true tip (`4c94f4ee` via the mirror):** **552/552 present, 0 missing ⇒ WOULD PASS.**

⚠️ **A principled deviation from the brief, surfaced not hidden.** I was asked to build and time on a
fresh `git archive` of the tip. ⛔ **The "before" cannot be measured there:** the 26 minutes is caused
by files MISSING from HEAD, and a fresh checkout has none — the old test would have run in ~21 s,
understating the fix ~75x and hiding the correctness bug entirely. The stale tree is the only place the
defect reproduces, and it is the *same* tree before and after, which is what the timing comparison
required. The tip was then used for the thing it IS authoritative about: the gate's verdict.

## §15 Re-run after all five batches landed — the gate is now RUNNABLE; 0 regressions

⛔ **Which HEAD, stated first.** The D: tree's HEAD is `37645fc`, **155 commits** behind the true tip
(`048ae3b9`, read from the mirror BY NAME). Its shared worktree is a MIX — probed on 195 files changed
in the last 8 tip commits: my batch files are **byte-identical** to the tip, while e.g.
`battery/SPEC.md` is genuinely different and LARGER than the tip (unlanded edits ahead of it).
⚠️ A first `git hash-object` probe read **138 of 195 "differ"** — mostly the EOL-filter artifact
(CRLF worktree vs LF blobs); raw-byte comparison showed my files SAME. So this run measures **the
current shared worktree**, which is neither my HEAD nor the landed tip, and it is labelled as such.

MEASURED, exit code written to a FILE (not read through a pipe):

| | first run (09-26 morning) | re-run |
|---|---|---|
| failed / passed / skipped | 31 / 2,380 / 27 | **31 / 2,388 / 27** |
| wall | **43m 09s** | **9m 40s** |
| real exit code | reported 0 — **it was `tail`'s** | **`PYTEST_RC=1`** |

* ⭐ **4.5x faster** — the gate is now something a stream will actually run before committing. Almost
  all of it is batch 5 (one test: 26m 25s → 0.53 s).
* **Same 31, file for file** (release_gate 17, render_openloop_video 9, registry_paths_allow 2,
  library_tracking 1, library 1, benchreport_legacy_compat 1) — taken from this run's full 76 KB output,
  not the pytest cache. ⇒ **0 regressions from everything landed; 0 of the 31 fixed.**
* **+8 passed = exactly mine** (5 style pins + 3 library arms; the 19 GPU-gate tests were already in the
  first run's 2,380).
* The library gate is RED here **for the stale-HEAD reason** and now says so correctly (86 MISSING, not
  "mount flapping"); at the true tip it reads 552/552. ⇒ **expected count at the tip: 30.**

## §16 ⛔ A leaderboard column we can see but cannot enter

`navsim_v2 --split navtest_single_stage` → **REFUSED**: *"not supported by navsim_v2 here; supported
['navhard_two_stage', 'warmup_two_stage'] (navtest single-stage is not wired — PDMS_v1 navtest is W3's
navsim_v1)"*. The profile EXISTS in `navsim/profiles.py`; the benchmark does not accept it.

⇒ `published_results.json` carries **18 external rows** under `EPDMS_v2_navtest_single_stage`
(DiffusionDrive-V2 at 87.5, among others) and **we have no way to produce our own number in that
column.** ⭐ The refusal is the RIGHT behaviour — it names the missing wiring instead of running
something else — so this is a gap, not a defect.
**What it would take (ESTIMATED, not measured):** the navtest frame bank (32/32 shards) and the v1
metric cache already exist, but EPDMS needs v2 cache fields (DDC, TLC, LK, HC, EC) that the v1.1 PDMS
cache does not carry — so a **v2 navtest metric cache** must be built (CPU, hours) before the profile
can be wired. It is the only NAVSIM protocol on our leaderboard we cannot currently run.

## §17 nuScenes, first contact (PI 2026-09-26: *"yes I accepted the terms, run the meta tier"*)

**Download.** `v1.0-trainval_meta.tgz`, **461,678,030 B**, public bucket `motional-nuscenes`, 191 s,
receipt `raw/nuscenes/RECEIPT_meta.json` (`accepted_terms_by: Sayed`). Landed packed at
`D:/Archive/devbox-C/nuscenes/archives/` — **off-repo, one copy**, re-fetchable with the same command.

⛔ **The publisher's own checksum file is WRONG for this object** — and it would have sent us re-downloading
a perfect file. Our md5 `537d3954…` ≠ the bucket's `md5.checksum` entry `3eee6988…`, at an exactly matching
byte count. Three independent probes settled it: `gzip -t` (CRC-32 over the whole stream) **OK**; the tar
index lists 21 members cleanly; and ⭐ **S3's multipart ETag recomputed from OUR bytes (8 MiB parts) =
`fd4ea76d8701fb567a67a65025d0b022-56`, identical to the live object's** ⇒ our file is byte-for-byte the
object served today. `md5.checksum` (uploaded 66 s before the archive, 2024-01-30) does not describe it.
⚠️ **Work item before the 45 GB `planning` tier:** verify each archive by **multipart ETag**, not by
`md5.checksum` — the script fetches that file but never compares against it, so today it verifies nothing
beyond byte count.

**Content, by independent statistic:** 13 tables parse — **850 scenes, 34,149 samples, 1,166,187
annotations, 23 categories, 4 maps**, matching nuScenes' published trainval figures.

**Floors, MEASURED, non-claim-bearing (H-EVAL-6), no interval (no pre-registered cluster unit):**

| protocol | pipeline | n (pre-registered) | GT L2 | STOP L2 | CV L2 | GT-collision floor |
|---|---|---|---|---|---|---|
| `nuScenes_OL_L2_uniad` | UniAD | **6,019** ✅ | 0.0 | 9.349 m | 1.365 m | **0.413 %** (PARA-Drive Tab. 8: 0.36 %) |
| `nuScenes_OL_L2_stp3` | VAD | **5,119** ✅ | 0.0 | 6.591 m | 0.818 m | ⛔ **REFUSED** (was 0.359 %; see below) |

Two of W6's three pre-registered counts reproduced exactly; the third (**4,819**, AD-MLP's 8-future rule)
belongs to a construction not yet run. ⭐ The two conventions also show *why* the suite refuses to run
without naming one: the same CV floor reads **1.37 m** (value AT t) vs **0.82 m** (mean UP TO t).

### ⛔⛔ The VAD collision column was computed over the WRONG OBJECTS — found, proven, fixed

`occupancy_vad` selects colliding agents by raw `category.json` **index** — `{2..8}` pedestrian,
`{14..23}` vehicle (`VAD_HUMAN_INDEX`/`VAD_VEHICLE_INDEX`) — correct only for the **32-entry lidarseg**
ordering (W6's F10). The base metadata has **23** entries. The audit, run for the first time:

* "pedestrian" `{2..8}` selects wheelchair, stroller, personal mobility, police officer, construction
  worker, **`animal`** and **`vehicle.car`** — and **misses `human.pedestrian.adult` and `child`**;
* "vehicle" `{14..23}` selects construction, ambulance, police, trailer, **barrier, traffic cone,
  pushable, debris, bicycle rack** — and **misses car, truck, both buses, motorcycle and bicycle**;
  index **23 does not exist**.

⭐ **External confirmation, not just a code reading:** that grid gave a VAD-protocol GT-collision floor of
**0.359 %** against **PARA-Drive Table 8's published 0.96 %** — **2.7x too low**, precisely the direction
dropping most road users predicts. UniAD is unaffected: `occupancy_uniad` selects by **name**.

⛔ **Eighth instance of "built, tested, unreachable from its caller":** `vad_category_index_audit` existed
for exactly this first contact, and its comment says it *"prints it on first contact"* — but it had **zero
call sites**. First contact happened and it said nothing.

**Fix** (`taniteval/adapters/nuscenes_planning.py`): `vad_category_order_ok()` gates the VAD pipeline
BEFORE the grid is built; on a False, `kernel_vad` gets `occ=None` and returns collision
**UNAVAILABLE with the reason** (a new branch mirroring `kernel_stp3`'s), and the audit is written into
`summary.json → controls.vad_category_audit` on **every** VAD run, pass or refuse. **7 tests**
(`taniteval/tests/test_nuscenes_vad_category_gate.py`): refuse on the real 23-entry file with the exact
missed classes; a SYNTHETIC intended-ordering fixture as the positive control (so "always refuse" fails);
L2 bit-identical with and without the refusal; and ⛔ a refused collision is never published as a fake
0.0 %. **64 pass** with the existing 57. ⭐ **Proven reachable end-to-end, not only in tests:** the real
re-run reads collision `UNAVAILABLE` with the reason, the audit is in the summary, and **L2 is bit-for-bit
unchanged** (STOP `6.590993453736307`, CV `0.8181846342993767`). The pre-fix run is **tombstoned**
(`TOMBSTONE.json`, directory kept, its 0.359 % retained as the record of the defect).

**Next lever, and it has a real acceptance target now:** computing VAD collision as published needs the
**32-entry lidarseg `category.json`** (inside the nuScenes-lidarseg expansion — a new download, PI's call)
or a name-based reimplementation. ⛔ Not by guessing the 32-entry order — that assumption IS the bug.
⭐ Whichever route: **it must reproduce PARA-Drive's 0.96 % VAD GT-collision floor** before any VAD
collision number is quoted.
