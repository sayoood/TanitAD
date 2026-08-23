# The stale-import guard — closing a defect that returns a PLAUSIBLE WRONG NUMBER, not an error

*2026-07-27. Closes the two escalations in `…/incoming/2026-07-27-small-validation/SMALL_VALIDATION.md`
§7.1 (items 1 and 5) and the frame-number error in the v5 documents.*

⛔ **Nothing here launches v5.** ⛔ **pod1 was NEVER CONTACTED.** ⛔ **pod2 was probed READ-ONLY** —
the small validation's trainer (PID 3695401), chain (3695397) and armed eval chain (3696611) were
not touched, no process was signalled, and no GPU or RAM load was added. pod3 and the eval pod were
probed read-only too.

---

## 0. HEADLINE

⭐⭐ **1. THE GUARD EXISTS, IT IS AUTOMATIC, AND IT IS DEMONSTRATED FAILING.**
`taniteval.stack_guard` refuses to let `taniteval` import a `tanitad` the caller did not ask for.
It fires **with no env var and no doc change**, because every published command already carries
`PYTHONPATH=/workspace/TanitAD/stack` and that is enough to state intent. RED→GREEN on a
deliberately stale tree is banked at `raw/stale_import_demo.json`, **7/7**.

⭐⭐ **2. THE CAUSE IS REMOVED, NOT PAPERED OVER: 52 hardcoded
`sys.path.insert(0, "/root/TanitAD/stack")` lines in 28 modules are gone.** ⚠️ It was **28 modules,
not the "~15" the escalation reported** — I re-counted rather than inheriting.

⛔ **3. THE RED CASE IS REAL AND IT IS A NUMBER, NOT A CRASH.** MEASURED: same host, same command
shape, guard off → the stale tree answers **HFOV 120.0** where the pinned tree says **117.0**, exit
0, no warning. That is the artifact-grade wrong number, reproduced.

⚠️ **4. `PREFLIGHT: OK` WAS PRINTED FOR A COMMAND THAT CANNOT RUN — and the RED half of that is run
against the SHIPPED code at `40aa6ff`, not a simulation** (`raw/preflight_path_demo.json`). The fix
is not another hand-written check but an **exhaustiveness contract** over the parser: `--poses-*` and
`--labels-*` were never checked either, so the hand-written list would have been incomplete from day
one.

⭐ **5. v5's frame is 117.000° × 32.131°, 429 tokens — not 120°.** 21 occurrences read and judged
individually (§5). **Most are correct and were left alone**; three described the *arm* and were
corrected.

---

## 1. 🔴 THE MECHANISM — and why the obvious check could never catch it

### 1.1 What was there

**MEASURED (ours), `raw/hardcoded_insert_rewrite.json`:** 28 modules of `taniteval/taniteval/`
contained **52** literal `sys.path.insert(0, "/root/TanitAD/stack")` / `.../scripts` lines:

```
ab bench blind_baseline cam_overlay closedloop corpus_overlay corridor data
direct_overlay driving efficiency flagship_overlay generalization hierarchy
imagination lateral loaders pathspeed plan_fan plan_fan_clips planner_p2
planning refb_eval refc_eval refc_rerank rollout runner strategic_probes
```

`insert(0, …)` puts that tree **in front of PYTHONPATH**. So importing *any* of them re-points
`tanitad`.

### 1.2 What `/root/TanitAD/stack` actually is — three hosts, three different answers

**MEASURED (ours), read-only, `raw/fleet_stack_inventory.txt`:**

| host | `/root/TanitAD/stack` | `/workspace/TanitAD/stack` | has `heldout_gate.py`? |
|---|---|---|---|
| **pod2** | ✅ **12 MB, NO `.git`, `resolve_v2_frames` grep = 0** | ✅ 54 MB, git, `resolve_v2_frames` = 3 | ⛔ **only the workspace one** |
| **pod3** | ❌ absent | ✅ 60 MB, git | ❌ neither (pod3 is itself pre-v5) |
| **tanitad-eval** | ✅ 5.8 MB, git — **the ONLY tree on the host** | ❌ absent | ❌ (older tree; it is the deployed eval host) |

⇒ **only pod2 is ambiguous, and pod2 is where v5 evaluates.** The two trees there differ by a whole
release.

⚠️ **The same shape exists one level up, for the harness itself:** pod2 carries **`/root/taniteval`
(22 MB, no `.git`, 107 `.py`)** *and* `/workspace/TanitAD/taniteval` (29 MB, git, 115 `.py`), and
`V5_GATEABLE.md` §5.4's corridor command pointed `PYTHONPATH` at the **stale** one. The stack guard
**cannot** catch that — it pins `tanitad`, not `taniteval` — so it had to be fixed in the command
(§6, correction 5).

### 1.3 ⚠️ Why "verify with a real `import tanitad`" was an insufficient instruction

**MEASURED**, `raw/stale_import_demo.json` scenarios S2 / S2b — two processes, one host:

| process | command | resolves to |
|---|---|---|
| the sync proof | `python3 -c "import tanitad"` | ✅ **GOOD** |
| the eval | the stale insert happens first (as the shipped code did), then the identical `import tanitad` | ⛔ **STALE** |

`import tanitad` alone is *load-order-dependent* and the sync proof runs in the wrong load order by
construction. **It cannot detect this defect and never could.**

⭐ **The corrected mechanical understanding, worth recording because I got it wrong first:** once
`tanitad` itself has been imported from the good tree, its **submodules follow `__path__`, not
`sys.path`**, and are safe. The shadow bites only when `tanitad`'s **root** import happens after the
stale insert. That is exactly what `taniteval/__init__.py`'s override import already exploited — and
why the cure works at all.

### 1.4 ⛔ The RED that matters is not the crash

The escalation's transcript showed `ModuleNotFoundError: tanitad.train.heldout_gate` — the **lucky**
case. **MEASURED here, S1:**

```
guard OFF, stale tree at sys.path[0], PYTHONPATH -> the good tree
  import tanitad.geometry  ->  {"tag": "STALE", "hfov": 120.0}      exit 0
  the pinned tree would have said                    117.0
```

**No exception. No warning. A number.** Every GREEN below is admissible only because this RED
reproduces.

---

## 2. ⭐ THE GUARD — three layers, and the demonstrated failure of each

`taniteval/taniteval/stack_guard.py` (new, 567 lines) + `taniteval/taniteval/stack_check.py` (a
23-line entry point, so the copied command does not print a `runpy` warning) +
`taniteval/taniteval/__init__.py` (rewired).

### 2.1 What "the one the caller intended" means — stated, because the whole guard rests on it

`resolve_intended_stack()`, in order. **Every source except the last is caller-controlled:**

| # | source | class |
|---|---|---|
| 1 | an explicit argument | explicit |
| 2 | `TANITEVAL_STACK_OVERRIDE` | explicit |
| 3 | an already-imported `tanitad` | explicit (someone got there first) |
| 4 | the first `sys.path` entry providing `tanitad`, **excluding** the legacy tree | explicit (PYTHONPATH / cwd) |
| 5 | `/root/TanitAD/stack` — the deployed tree | ⚠️ **fallback: nothing was named** |
| 6 | an installed/editable `tanitad` via `find_spec` | fallback |

⭐ **Therefore the guard can only fire when the caller NAMED a stack and a different one won.** When
nothing is named, the deployed tree *is* the intent and there is nothing to violate. **That is the
property that makes turning this on safe for the eval host that produced every published closed-loop
number** — and it is pinned by a test, not asserted (§2.5).

### 2.2 Layer 1 — the cause. `ensure_stack_on_path()`

Replaces the 52 hardcoded lines. Semantics:

* **nothing named** → the legacy tree and its `scripts/` go to the **front**, byte-for-byte the old
  behaviour;
* **something named** → *that* root and its `scripts/` go to the front and **the legacy tree is not
  added at all** — not even appended. A silent fallback to a pre-v5 tree is the same wrong-number
  failure one step later; a loud `ImportError` is the better outcome.

### 2.3 Layer 2 — the tripwire. `StackSentinel`, a `sys.meta_path` finder

⭐ **Why a finder and not a one-shot assert:** a one-shot check is defeated by the *next*
`sys.path.insert`, and there were 52 of them in this package alone plus every driver script and
every copied command. The finder is consulted on **every** `tanitad` import for the life of the
process, so it cannot be outrun. It delegates to the finders that would have served the import
anyway and only inspects the answer, so it changes **no** resolution semantics when nothing is
shadowed.

⚠️ **One near-miss worth recording (class: a guard that looks armed and is inert).** My first version
delegated to `importlib.machinery.PathFinder` only. On this dev box `tanitad` is an **editable
install served by a custom meta-path finder and is not on `sys.path` at all** — so `PathFinder`
returned `None`, the sentinel inspected nothing, and `report()` said
`sentinel_installed: false`. **Caught by printing the report rather than trusting the install.** It
now iterates all of `sys.meta_path`.

### 2.4 Layer 3 — the capability probe. `assert_stack(require="v5")`

Identity is necessary, not sufficient: a tree can be the right *path* and still be pre-v5 (a pod
never `git pull`ed — **that is pod3 and the eval pod today**, MEASURED §1.2). `V5_CAPABILITIES`:

```
tanitad.train.heldout_gate:PRIMARY_NAME
tanitad.train.heldout_goal:make_goal_kwargs_fn
tanitad.data.parity:register_v2_geometry_sibling
tanitad.geometry:frame_from_args
train_flagship_v4:resolve_v2_frames
```

⚠️ **A probe list that does not describe the real tree is a rubber stamp**, so
`test_this_repo_stack_satisfies_the_v5_capability_set` checks all five against this checkout's own
`stack/`.

CLI: `python3 -m taniteval.stack_check --require v5 --json <f>` → **exit 0 / exit 2**.

### 2.5 ⛔ THE DEMONSTRATED FAILURES — 7 scenarios, 17 tests

`raw/stale_import_demo.json` (standalone, `code/stale_import_demo.py`) and
`taniteval/tests/test_stack_guard.py` (17 tests, subprocess-isolated, two independent routes to the
same claims). **All MEASURED (ours).**

| # | scenario | result |
|---|---|---|
| **S1** ⛔ RED | stale tree, guard off | **exit 0, `hfov 120.0`** — the wrong number, silently |
| **S2** ⚠️ RED | bare `import tanitad` (the sync proof) | **passes**, says GOOD |
| **S2b** ⚠️ RED | the same import after the stale insert | **STALE** |
| **S3** ⭐ GREEN | **no env var at all**; PYTHONPATH is the intent, stale inserted later | **refuses**, `STACK SHADOWING` |
| **S4** GREEN | `TANITEVAL_STACK_OVERRIDE` **set but ineffective** (typo / moved checkout) | **refuses** — it used to print its success line over a stale resolution |
| **S5** GREEN | capability probe on a pre-v5 tree | **exit 2**, names `heldout_gate` |
| **S6** GREEN | capability probe on a post-v5 tree | **exit 0** |
| **S7** GREEN *(test only)* | pod2's real layout: two trees, **neither named** | **SHOUTS** `AMBIGUOUS STACK`, does not refuse (§3.3) |

Plus, in the test file: an already-imported stale `tanitad` is caught at sentinel-install time;
`warn` mode shouts and records without raising; **`off` mode cannot hide itself** (`report()` always
carries the mode, so no artifact can look clean while the guard was disabled); the **deployed
legacy-only layout is unchanged**; and a **regression pin** fails if any hardcoded
`sys.path.insert(0, "/root/TanitAD/stack")` comes back.

---

## 3. ⚖️ VERDICT ON THE HARDCODED `sys.path.insert` — remove the shadow, keep the fallback

### 3.1 The decision

**REMOVED as an unconditional insert; KEPT as a last-resort resolution target.** Concretely: no
module inserts `/root/TanitAD/stack` any more; `ensure_stack_on_path()` *resolves* to it only when
the caller named nothing, and then places it exactly where the old lines did.

### 3.2 Why not delete it outright — MEASURED, not assumed

`tanitad-eval` **is** a `/root/TanitAD/stack`-only host: 5.8 MB, git, and **`/workspace/TanitAD/stack`
is absent** (`raw/fleet_stack_inventory.txt`). Its harness is `/root/taniteval`, which contains no
`tanitad`. Deleting the fallback would leave `import tanitad` unresolvable there for any command that
does not set `PYTHONPATH` — i.e. it would break the deployed eval path, which the brief forbids and
which produced every published closed-loop number.

⇒ **`test_legacy_only_layout_is_unchanged` pins exactly that layout**: provenance
`legacy-fallback`, legacy + `legacy/scripts` at the front, `tanitad` resolving there, guard inert.

### 3.3 ⚠️ What is NOT fixed by this, stated plainly

On a host with **two** trees where the caller names **neither** — pod2 with no `PYTHONPATH` and no
override — the legacy (pre-v5) tree still wins. **Resolution is deliberately not changed there**,
because any reordering that helps pod2 moves the eval host. Instead the silence is removed: that
case now prints

```
[taniteval.stack_guard] ⚠️ AMBIGUOUS STACK — you named none, so the deployed tree
'/root/TanitAD/stack' was used, but this host ALSO has ['/workspace/TanitAD/stack'].
```

and records `ambiguous_alternatives` in `report()`. **A shout, not a refusal** — refusing would be
the change that breaks the deployed path.

⇒ **The documented path is therefore still the override**, and §6's commands carry it. This is the
honest residual, not a claim of completeness.

### 3.4 Two more residuals I did not close

1. **`sys.path.insert(0, "/root/taniteval")`** survives in the same modules. It cannot re-import the
   `taniteval` package (already in `sys.modules` by then), but it *can* shadow top-level helper
   modules. **Separate class, separate decision, and pod2 has a stale `/root/taniteval`** — flagged
   in §8, not silently swept in.
2. **18 top-level scripts** outside the package (`taniteval/refc_scale_ab.py`, `cosmos_*.py`, …) still
   hardcode the path. They are drivers, not library code, and the sentinel still catches them the
   moment they import `taniteval`. Listed, not fixed.

---

## 4. `PREFLIGHT: OK` OVER A PATH THAT DOES NOT EXIST

### 4.1 The finding, and the RED against shipped code

`--anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt` is in **both** published v5
launch commands and **that directory is empty on pod2** (INHERITED from `SMALL_VALIDATION.md` §7.1
item 5, which measured it; I did not re-probe pod2's filesystem for it). `--print-launch` checked
argument *presence*, never path *existence*.

**MEASURED (ours), `raw/preflight_path_demo.json` — the RED half runs `git show
40aa6ff:stack/scripts/train_flagship_v4.py`, i.e. the actually-shipped code, with the SAME argv:**

| run | verdict | exit |
|---|---|---|
| **RED — shipped code, missing anchors** | **`PREFLIGHT: OK`** | **0** |
| **GREEN — fixed code, missing anchors** | **`PREFLIGHT: BLOCKED`** + `[PATH-PREFLIGHT] --anchors-dense …: DOES NOT EXIST on THIS host` | **2** |
| GREEN — fixed code, real anchors | `PREFLIGHT: OK` | 0 |

### 4.2 The decision: check every path, do not drop the flag

⛔ **Dropping `--anchors-dense` from the commands was rejected.** The arm needs an anchor vocabulary,
so the flag must stay and its *value* was wrong (`V5_GATEABLE.md` §5.0 had already corrected it;
`V5_EVALUABLE.md` had not). More importantly, deleting the flag fixes one command and leaves the
class — and **this is the second `PREFLIGHT: OK` over an unexamined input** (the first was
`--require-parity` against a cache whose `corpus_key_of` was `None`).

### 4.3 ⭐ Why the fix is a contract and not a check list

`preflight_path_problems()` checks kind and existence for every classified path.
**`PATH_ARGS ∪ NOT_A_PATH` must cover every free-form string argument the parser accepts**
(`stack/tests/test_preflight_paths.py::test_path_classification_is_exhaustive_over_the_parser`).

⚠️ **This is not theoretical: enumerating the parser turned up `--poses-train`, `--poses-val`,
`--labels-train`, `--labels-val` — four path arguments I would not have written down.** A
hand-maintained list would have shipped incomplete on day one. 13 path args are now checked; 2
(`--v2-subframe`, `--device`) are listed as non-paths **with the reason**.

`--out` is treated as an output: its **parent** must exist, so a legitimate fresh run directory is
not refused while `--out /no/such/place/run` is.

**11 tests**, including the RED (`test_preflight_BLOCKS_a_missing_anchors_file`) and its twin
(the same command with the real file → `PREFLIGHT: OK`), so it is not blanket-refusing.

---

## 5. 🔬 v5's FRAME IS 117°, NOT 120° — every occurrence read and judged

**The measurement (INHERITED from `SMALL_VALIDATION.md` §5.1, taken through the trainer's own
`resolve_v2_frames` on pod2 — not re-run by me, no pod load added):**

```
--v2-subframe none    -> TRAIN 256x640, f_ref 305.5775, HFOV 120.000 deg, cylindrical
--v2-subframe 176x624 -> TRAIN 176x624, f_ref 305.5775, HFOV 117.000 / VFOV 32.131,
                         rows [40:216], cols [8:632]   — a pure pixel slice
```

⇒ **the cache is 120°; the arm is 117°.** Both published launch commands carry
`--v2-subframe 176x624`, and preflight *refuses* a non-deployed v2 frame without one — so 117° is
not a possibility, it is the run.

### 5.1 The judgements — ⚠️ read individually, NOT sed'ed

| where | occurrence | verdict |
|---|---|---|
| `PREP.md` §3.7 «target 100–120°», §3.7 storage table «120° 256×640 = 112.9 GB» | the **build** decision | ✅ **left** — describes the cache |
| **`PREP.md` §3.7 «v5's GEOMETRY IS 256×640 CYLINDRICAL (the full 120°, 640 tokens)»** | ⛔ **describes the ARM** | 🔧 **corrected** — the PI's quote and decision are left verbatim; a marked correction records that the *cache* is 120° and the *arm* is the 176×624 / 117° / 429-token slice |
| `V5_EVALUABLE.md` §6.3 `"requested_hfov_deg": 120.0` (manifest JSON) | the **registered cache** entry | ✅ **left** — changing it would falsify a committed manifest |
| `V5_EVALUABLE.md` §7.1/§7.3, `V5_GATEABLE.md` §5.1/§5.3, `RIG_FIX_WIRING.md` §8 — `--frame-hfov 120` | the **flag value**, which describes the parent render the providers hand back | ✅ **left — it is correct**; a comment now says the scored arm is the 176×624 slice at 117° |
| `V5_EVALUABLE.md` §9 «24 clips … decoded in the 120° build» | the **build** | ✅ left |
| `RIG_CLEAN_FIX.md` (≈14 hits incl. «117° × 32.1° vs 120° × 45.5°», «the 120° request over-runs the sensor», «120 A / 120 B» clip counts) | this is the document that **established** 117° | ✅ **all left** — they are the parent frame, the request, or clip counts, and the doc already states the contrast correctly |
| `RIG_FIX_WIRING.md` §9 «whether 117° × 32.1° trains better than 120° × 45.5°» | the contrast, correctly stated | ✅ left |
| `RESOLUTION_GAIN.md` §4.7 reconciliation, E3/E4 measurements on `V5_640` | measurements made **on the 256×640 frame** | ✅ **left** — re-labelling them would falsify what was measured |
| **`RESOLUTION_GAIN.md` §6.1 table: «`R_640` … the chosen v5 frame»** | ⛔ **describes the ARM** | 🔧 **corrected** to "the chosen v5 **cache**", with the arm's slice named |
| **`RESOLUTION_GAIN.md` §7 «Train v5 at 256×640 / 120° cylindrical»** | ⛔ **describes the ARM** | 🔧 **corrected** to "**from** the … cache", + a note that the verdict is unaffected because px/deg (**5.3333**, set by `f_ref` 305.5775) is **unchanged by the slice** — only the field changes |
| `RESOLUTION_GAIN.md` E1 «the v5 frame's angular resolution … 5.3333 px/deg (120° cylindrical)» | ⚠️ judged and **left**: the *number* is slice-invariant and the parenthetical names the render it was computed from | ✅ left |
| `V5_TRAINER.md`, `WIDE_VAL_BUILD.md`, `WIDE_FOV_BUILD.md`, `RETRACTION_LOG.md` §C-… | all about **building the 120° cache** | ✅ left |
| `Project Steering/Reports/2026-07-27-1757`, `2026-07-28-0757` | dated snapshots that were **accurate when written** (pre-`284c591`) | ✅ **left** — a report is a record; §8 escalates the forward-looking fix instead |

**3 corrected · 18 judged correct and left.** ⚠️ `MODEL_REGISTRY.md` carries **no** v5 geometry entry
yet (v5 is untrained) — checked, so nothing there needs the correction.

---

## 6. ⭐ THE CORRECTED v5 COMMAND SET

⛔ **Nothing here launches anything. A launch is the PI's go.** Staged into `V5_EVALUABLE.md` §7 and
`V5_GATEABLE.md` §5 as well, so the fix lives where the commands are copied from.

```bash
# ───────── STEP 0 ⭐ THE STACK PIN. One second; it is the difference between a
#           wrong number and an exit 2. Run it FIRST, on the pod.
cd /workspace/TanitAD/stack && \
TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts:/workspace/TanitAD/taniteval \
python3 -m taniteval.stack_check --require v5 \
  --json /workspace/taniteval/results/stack_guard_v5.json

# ───────── TRAIN (unchanged except the anchors path)
cd /workspace/TanitAD/stack && PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
python3 -u scripts/train_flagship_v4.py \
  --v2-train-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache   /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --v2-lru 64 --require-parity \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --from-scratch \
  --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
  --out   /workspace/experiments/flagship-v5-w120-rigclean-30k \
  --steps 30000 --batch 8 --accum 8 --lr-head 1e-4 --lr-trunk 1e-4 \
  --warmup 2000 --workers 8 --eval-every 500 --save-every 1000 --rollout-k 4 \
  --heldout-gate --heldout-every 2000 --heldout-episodes 8 --heldout-patience 2 \
  --device cuda
#   ⭐ run it with --print-launch FIRST, ON THE POD: preflight now also refuses a
#     path that does not exist there.
#   ⚠️ --frame-hfov 120 is the PARENT cache. The arm is the 176x624 slice: 117.000
#     deg x 32.131 deg, 429 tokens.

# ───────── EVAL MODE A (harness check vs the known v1 number; stays on the RAW path)
cd /workspace/TanitAD/stack && TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts \
python3 scripts/eval_flagship_v4.py \
  --ckpt /workspace/models/flagship-30k/ckpt.pt --canary-only \
  --val-cache /workspace/data/physicalai-val-0c5f7dac3b11 \
  --key v1-validation --out /workspace/taniteval/results/v1-validation.json

# ───────── EVAL MODE B (the v5 arm)
cd /workspace/TanitAD/stack && TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts \
python3 scripts/eval_flagship_v4.py \
  --ckpt /workspace/experiments/flagship-v5-w120-rigclean-30k/ckpt_best.pt \
  --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --require-parity --v2-lru 64 \
  --key flagship-v5-w120-rigclean-10k \
  --out /workspace/taniteval/results/flagship-v5-w120-rigclean-10k.json
#   … and the DEPLOYABLE twin (no goal oracle) — add:  --goal-mode produced

# ───────── CO-PRIMARY PANEL  ⛔ override REQUIRED, and /root/taniteval was WRONG
cd /workspace/TanitAD/stack && TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/taniteval \
python3 scripts/gate_emitters.py corridor \
  --windows      /workspace/taniteval/results/windows_flagship-v5-w120-rigclean-10k.pt \
  --out-corridor /workspace/taniteval/results/corridor_flagship-v5-w120-rigclean-10k.json

# ───────── GATE CHECK  (override defensive here — see the table below)
cd /workspace/TanitAD/stack && TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack \
PYTHONPATH=/workspace/TanitAD/stack \
python3 scripts/run_gate.py check \
  --card gates/flagship-v5-w120-rigclean-30k.card.json \
  --log  /workspace/experiments/flagship-v5-w120-rigclean-30k/train_log.jsonl \
  --eval-json /workspace/taniteval/results/flagship-v5-w120-rigclean-10k.json \
  --corridor-json /workspace/taniteval/results/corridor_flagship-v5-w120-rigclean-10k.json
```

⚠️ **`TANITEVAL_STACK_OVERRIDE` must be set BEFORE `python3` starts** — it works by importing
`tanitad` from that root inside `taniteval/__init__.py`, so the `sys.modules` cache beats every later
insert. ⭐ **Since today it is verified, not trusted** (S4).

**Where it is load-bearing vs defensive — VERIFIED IN CODE, not assumed:**

| command | imports `taniteval`? | the override is |
|---|---|---|
| `eval_flagship_v4.py` | ✅ `taniteval.bench` (:1116), `taniteval.driving` (:1155) — **both carried the stale insert** | ⛔ **REQUIRED** |
| `gate_emitters.py corridor` | ✅ `taniteval.corridor`, `taniteval.rollout` (:356-357) — **both carried it** | ⛔ **REQUIRED** |
| `train_flagship_v4.py --heldout-gate` | ✅ lazily via `heldout_gate._taniteval()` → `taniteval.pseudosim` + `.ci` | ⛔ **REQUIRED** |
| `run_gate.py` | ❌ none — it mirrors `taniteval.ood` in pure arithmetic on purpose | defensive; kept so the leg is spelled one way |

---

## 7. SUITES — zero new skips

| suite | brief's baseline | **measured here, before my changes** | **measured here, after** | new skips |
|---|---|---|---|---|
| `stack/` | 1523 passed, 12 skipped | ✅ 1523 passed, 12 skipped (108.5 s) | ✅ **1534 passed, 12 skipped** (+11 new: `test_preflight_paths.py`) | **0** |
| `taniteval/` | 644 passed | ✅ 644 passed (68.4 s) | ✅ **661 passed** (+17 new: `test_stack_guard.py`) | **0** |

⚠️ **The `taniteval/` suite was re-run after the 28-module rewrite and before any new test was
added: 644 passed, unchanged.** So the rewrite is separable from the additions.

---

## 8. 🔴 ESCALATIONS — decisions, not documentation

1. ⭐⭐ **The 28-module rewrite touches SHIPPED library code that the running small validation's
   armed eval chain will import** (`pod2:/workspace/smallval/evalchain.sh`, PID 3696611). **It is
   staged in the repo and NOT pushed to pod2**, deliberately: pod2 was synced at HEAD and is running.
   ⇒ **someone must decide when pod2 re-syncs.** Until it does, pod2 keeps the old modules **and**
   `code/smallval_pseudosim.py:pin_stack()`, which already does the equivalent check — so the running
   validation is not exposed. **This is the integration decision, and it is here rather than in a
   README.**
2. 🔴 **pod3 and `tanitad-eval` are BOTH pre-v5** (no `heldout_gate.py`; `resolve_v2_frames` grep 0)
   — MEASURED §1.2. Any v5-era eval on either host will now be **refused by the capability probe**,
   which is correct, but it means **the only v5-capable eval host is pod2, and pod2 is the training
   host.** That is a fleet-capacity decision, not a code one.
3. ⚠️ **`/root/taniteval` (the harness, not the stack) is a second stale tree on pod2 (22 MB, no
   `.git`)** and was in a published `PYTHONPATH`. Fixed in the command; **the guard cannot cover
   this class** (it pins `tanitad`). If harness shadowing matters, it needs its own instrument.
4. ⚠️ **`SMALL_VALIDATION.md` §7.1 items 2, 3 and 4 are NOT mine and are still open** — the
   `"full step s"` label (an 8× sizing error), `git archive`'s CRLF non-faithfulness, and the
   **missing val uid digest** (the trainer's val parity check is count-only). I deliberately did not
   edit that document: its stream is live and banking into it.

---

## 9. DELIVERABLE MANIFEST

| artifact | where it lives | only one place? |
|---|---|---|
| `STALE_IMPORT_GUARD.md` (this) | `repo:…/incoming/2026-07-27-stale-import-guard/` **(staged)** | no |
| ⭐ `taniteval/taniteval/stack_guard.py` — the guard, all three layers (567 lines) | `repo:` **(staged)** | no |
| `taniteval/taniteval/stack_check.py` — the `python3 -m taniteval.stack_check` entry point | `repo:` **(staged)** | no |
| ⭐ `taniteval/taniteval/__init__.py` — rewired; the `TEMP assess` label removed | `repo:` **(staged)** | no |
| ⭐ **28 modules** of `taniteval/taniteval/` — 52 hardcoded inserts removed | `repo:` **(staged)** | no |
| ⭐ `taniteval/tests/test_stack_guard.py` — 17 tests incl. the RED | `repo:` **(staged)** | no |
| ⭐ `stack/scripts/train_flagship_v4.py` — `PATH_ARGS`, `NOT_A_PATH`, `preflight_path_problems` | `repo:` **(staged)** | no |
| ⭐ `stack/tests/test_preflight_paths.py` — 11 tests incl. the RED (159 lines) | `repo:` **(staged)** | no |
| `code/stale_import_demo.py` + `raw/stale_import_demo.json` — 7 scenarios | `repo:` **(staged)** | no |
| `code/preflight_path_demo.py` + `raw/preflight_path_demo.json` — RED vs shipped `40aa6ff` | `repo:` **(staged)** | no |
| `raw/fleet_stack_inventory.txt` — the read-only 3-host probe | `repo:` **(staged)** | ⚠️ **yes** (a probe transcript; reproducible in one command) |
| `raw/hardcoded_insert_rewrite.json` — every file and line replaced | `repo:` **(staged)** | no |
| `raw/suites.txt` — the four suite runs | `repo:` **(staged)** | no |
| **doc edits** — `V5_EVALUABLE.md` §7, `V5_GATEABLE.md` §5, `RESOLUTION_GAIN.md` §6.1/§7, `Project Steering/Gates/flagship-v5-retrain.PREP.md` §3.7 | `repo:` **(staged)** | no |

**I ran no `git commit`, no `git push`, and switched no branch.** Staging was verified with
`git ls-files --cached`, not exit codes (`raw/staged_files.txt`).

---

## 10. Provenance and evidence class of every number

| claim | class · tier | source |
|---|---|---|
| 28 modules / 52 hardcoded insert lines | **MEASURED (ours)** · DECISION-GRADE | `raw/hardcoded_insert_rewrite.json` |
| pod2 has both trees; `/root/TanitAD/stack` 12 MB, no `.git`, `resolve_v2_frames` = 0 | **MEASURED (ours)** · read-only probe | `raw/fleet_stack_inventory.txt` |
| pod3 has no `/root/TanitAD/stack`; `tanitad-eval` has ONLY it | **MEASURED (ours)** · read-only probe | same |
| pod3 + eval are both pre-v5 (`heldout_gate.py` absent) | **MEASURED (ours)** | same |
| ⭐ the stale tree returns 120.0 where the pinned one returns 117.0, exit 0 | **MEASURED (ours)** · DECISION-GRADE | `raw/stale_import_demo.json` S1 |
| the bare `import tanitad` sync proof passes while the eval resolves stale | **MEASURED (ours)** | same, S2/S2b |
| the guard refuses with no env var / on a set-but-ineffective override / on a pre-v5 tree | **MEASURED (ours)** | same, S3–S6 |
| ⭐ shipped `40aa6ff` prints `PREFLIGHT: OK` for a missing `--anchors-dense`; the fix exits 2 | **MEASURED (ours)** · DECISION-GRADE | `raw/preflight_path_demo.json` |
| `--poses-*` / `--labels-*` were unchecked free-form path args | **MEASURED (ours)** | parser enumeration; `test_path_classification_is_exhaustive_over_the_parser` |
| suite counts 1523/12 → 1534/12 and 644 → 661 | **MEASURED (ours)** | `raw/suites.txt` |
| **v5's frame is 176×624, 117.000° × 32.131°, 429 tokens** | **INHERITED** (`SMALL_VALIDATION.md` §5.1, measured there via `resolve_v2_frames` on pod2) — **not re-run by me**, to add no pod load | that doc |
| `--anchors-dense` path is absent on pod2 | **INHERITED** (`SMALL_VALIDATION.md` §7.1 item 5) — the *class* is MEASURED here, the pod2 filesystem fact is not re-probed | that doc |
| which scripts import `taniteval` | **MEASURED (ours)** · VERIFIED IN CODE | `eval_flagship_v4.py:1116,1155`; `gate_emitters.py:356-357`; `heldout_gate.py:177-185`; `run_gate.py` = none |
| pod1's state | **NOT PROBED** — never contacted | — |
| v1's 0.4271 | ⛔ **NOT USED** — it is `wm_fidelity_ade_2s` (the world model handed the TRUE actions), not a planning bar | — |

🔒 **Gated-confidential:** every artifact here carries **counts, paths, digests and module names
only**. No PhysicalAI-AV clip UUID appears anywhere in this deliverable.
