# OPENLOOP_SUITE — the binding-KPI open-loop suite, and how to fire it

**Instrument:** `taniteval/tools/openloop_suite.py` · **tests:** `stack/tests/test_openloop_suite.py`
(18 passed) · **owner:** Benchmarks & Evals FlyWheel · **written:** 2026-09-03

> ⛔ **THIS IS THE LAYER ABOVE `refcv3_arm.py`, NOT A SECOND COPY OF IT.** The arm tool
> defines and produces the arms and their per-arm four-families blocks. This tool obtains
> that record, adds the constant-only control, assembles ONE artifact the criteria registry
> can read, runs the checker against it, and renders the report. **It computes no geometry
> of its own** — every metre comes from `taniteval/four_families.py` and `taniteval/ci.py`
> through `t1_eval.analyze`. A second implementation of a metric is a second definition of
> it, and that is how two numbers wearing one name end up in a table.

---

## 1. ⛔ THE REGIME: everything this suite produces is OPEN LOOP

**PI ruling, 2026-09-02, verbatim:**

> "The open loop performance corresponds to the fact that the AI model is not controlling the
> vehicle in the world. The fact that the predictor is consuming the output of the planner of
> the world-model-based system is ALSO OPEN LOOP because the trajectory of the model is not
> affecting the new ego data (this will be fed from the eval ego data). Closed loop means the
> trajectory is controlling the vehicle in the simulation, e.g. in AlpaSim or a real test
> vehicle."

No arm here is scored in a simulator and none of them steers anything. The rendered report is
**checked mechanically** for the superseded phrase before it is written
(`_guard_loop_vocabulary`), and the write is **refused** on any occurrence.

⚠️ **The guard fired on this tool's own first report** — at the very table that *reports* the
stale labels. The entries in `KNOWN_STALE_LOOP_LABELS` therefore **describe** the stale text
instead of quoting it. That was the right repair: a guard with an exemption for "the section
that talks about the rule" is a guard that can be walked through.

**The tier LETTER (`T1`) is kept** — it is the doctrine's machine-readable stamp and the
criteria registry keys on it. Three places still carry the superseded PROSE for that row; they
are **REPORTED in §9 of every report and NOT edited** (a separate stream owns that correction):
`taniteval/tools/t1_eval.py` (`_TIER_NOTE` / `_tier_doctrine`),
`products/P7-TanitEval/CRITERIA_REGISTRY.json` (`tiers.values.T1.note`),
`Project Steering/EVAL_DOCTRINE.md` (the T1 row).

---

## 2. ⭐ THE ONE COMMAND — the final step-40,284 read

**Paths are MEASURED from the run's own `config.json`** (`argv`, banked at
`C:\Users\Admin\refcv3_diag\config.json`), not assumed:
`--eval-cache /root/data/eval`, `--eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz`,
`--v2-cache /root/data/train`, `--image-hw 256 640`, `--arm hier`, `--steps 40284`.

### ⛔⛔ THE CHECKPOINT IS `ckpt.pt`. `ckpt_40284_FINAL.pt` IS NEVER WRITTEN.

**Verified at source:** `refc_v3_train.py:103` sets `MILESTONES = (5000, 15000, 20000, 30000)`
and **40,284 is not in it**, so no `ckpt_<step>.pt` is produced at the end of the run.
`REFCV3_ARM.md` §3.1 and §5 both tell the operator to evaluate `ckpt_40284_FINAL.pt` —
**waiting for that file waits forever.**

The final checkpoint is plain **`/workspace/experiments/refcv3-b1-v72-30k/ckpt.pt`**, written by
`refc_v3_train.py:1191` (`if step % args.save_every == 0 or step == args.steps:` — the
`step == args.steps` branch fires at 40,284).

⚠️ **It is ALSO the ROLLING file** (`--save-every 500`), and **its name never changes.** Before
the run ends, that same path holds an earlier step — and a report of "the final read" computed
from step 38,450 is not wrong in any way a reader could detect. ⇒ **verify by CONTENT:**

```bash
python -c "import torch; print(torch.load('.../ckpt.pt', map_location='cpu')['step'])"
```

⭐ **The suite enforces this itself: pass `--expect-step 40284` and it REFUSES a checkpoint whose
own `step` differs.** The value comes from the arm tool's manifest — i.e. from the checkpoint —
never from the filename. It is ~**1.28 GB** (optimizer state included), unlike the 428 MB
milestone files which carry only `{"model", "step"}`.

⛔ **GATE 0 / GATE 1 FIRST** — `T1_CHECKLIST.md` in
`TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refcv3-epoch-conclusions/`:
`summary.json` reads `{"done": true, "step": 40284}`, the supervisor has **exited**
(`ps -eo args | grep -c 'sup[_]refcv3'` must read **0**), and **the pod is no longer training**.
Copying `ckpt.pt` to an immutable name and md5'ing it is still worth doing — but the copy's
*name* proves nothing either; `--expect-step` is what proves it.

### Step 1 — price it (two clips), then choose the stride

```bash
OMP_NUM_THREADS=6 python taniteval/tools/openloop_suite.py \
  --ckpt      /workspace/experiments/refcv3-b1-v72-30k/ckpt.pt \
  --episodes  /root/data/eval \
  --labels    /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --nav-source v72 --grid 2s --action-units steer --with-oracle-sel \
  --episodes-n 2 --window-stride 1 --n-boot 300 --expect-step 40284 \
  --dump-dir  /workspace/eval/openloop_probe_dump \
  --out-dir   /workspace/eval --tag refcv3-40284-probe \
  --tiers os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0
```

Read the arm tool's `[cost]` line from that run; it, not an assumption, sets `--window-stride`.

### Step 2 — ⭐ THE REAL READ (paste this)

```bash
OMP_NUM_THREADS=6 python taniteval/tools/openloop_suite.py \
  --ckpt      /workspace/experiments/refcv3-b1-v72-30k/ckpt.pt \
  --episodes  /root/data/eval \
  --labels    /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --lead-block "TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz" \
  --nav-source v72 --grid 2s --action-units steer --with-oracle-sel \
  --window-stride 5 --n-boot 2000 --seed 0 --expect-step 40284 \
  --dump-dir  /workspace/eval/refcv3_40284_dump \
  --out-dir   taniteval/results --tag refcv3-40284-openloop \
  --corpus    "PhysicalAI B1 v7.2 EVAL — 141 clips with pixels of 147" \
  --parity-status "NON-PARITY — MEASURED from the run's own config.json: v2_parity.parity false, checked false, corpus_key null; the trainer prints its own non-parity warning on line 2 of train.log" \
  --parity-key "eval slice; the run did NOT assert corpus parity" \
  --labels-md5 aa12c948f062181c3297265b51526ec5 \
  --train-eval-disjoint "ARITHMETIC EVIDENCE, not a check: config.json train cache /root/data/train holds 4,572 clips = 4,713 - 141 eval clips. v2_parity.checked is FALSE, so the trainer verified nothing — assert by episode id before quoting (GATE 3)." \
  --tiers os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0 \
  --strict
```

Emits, all three from one invocation:
`taniteval/results/refcv3-40284-openloop.json` · `.md` · `.html`.

### Re-analysis — ⛔ ALWAYS CHECK THIS BEFORE RE-RUNNING ANYTHING

```bash
OMP_NUM_THREADS=6 python taniteval/tools/openloop_suite.py \
  --analyze-only /workspace/eval/refcv3_40284_dump \
  --lead-block "…/b1_eval_lead_block.npz" --n-boot 2000 --seed 0 \
  --out-dir taniteval/results --tag refcv3-40284-openloop --strict
```

**Zero GPU.** The rollout is the only expensive part, and an analysis-time failure after a
completed rollout reads like a total failure while the compute is already paid for — that has
destroyed a finished 2-arm / 40-episode rollout once. The tool also **refuses** to roll out into
a `--dump-dir` that already holds a dump, rather than silently paying twice.

### Consuming another stream's banked record

```bash
python taniteval/tools/openloop_suite.py --arm-json <refcv3_arm record>.json \
  --dump-dir <its dump dir> --out-dir taniteval/results --tag <tag>
```

⚠️ Pass `--dump-dir` too. Without the per-window arrays the **constant-only control cannot be
built**, and the suite says so as a REFUSAL rather than quietly dropping it.

### ⚠️ `--stack` — when the editable install points at a flaky mount

```bash
python taniteval/tools/openloop_suite.py … --stack /workspace/TanitAD/stack
```

**MEASURED 2026-09-03, and it CORRECTS the remedy `CLAUDE.md` records for this trap.** The
dev-box venv's editable install of `tanitad` is **not** a `.pth` path entry — it is
`__editable___tanitad_0_0_1_finder.py`, a **MetaPathFinder**. `sys.meta_path` is consulted
**before** `sys.path`, so `PYTHONPATH=<clone>/stack` does **not** win: with the clone at
`sys.path[1]` and `[2]` (verified), `import tanitad` still resolved to
`G:\…\stack\tanitad\__init__.py`. A run that believes it is off-Drive is then one G: hiccup away
from an `Errno 22` **hours in**. `--stack` evicts the finder and **verifies the binding by
CONTENT** (`tanitad.__file__` must live under the given tree), refusing to run if it did not
take. Every run banks `provenance.resolved_trees` regardless — presence proves transfer, md5
proves bytes, a successful import proves loading, **none of them proves currency**.

---

## 3. What the artifact carries, and why each piece is load-bearing

| block | what it is | the failure it exists against |
|---|---|---|
| `headline` | *does the arm beat `ha0`, **per family**, paired* — WON / LOST / TIED per metric | a model number without its control is unreadable: a T1 arm once read 9.3697 m beside its own control's 0.4246 m on the same windows |
| `void_gates` | trivial profile **and** selection profile, before any family row | a 2.5 h read shipped "lateral planning already beats holding" while every plan was a straight constant-speed line; and a model can select ONE anchor on 100 % of windows while `trivial_frac` reads 0.0 — the two gates see different degeneracies |
| `controls.const0` | a predictor emitting the same output for every input, against **known values** | four probe failures in one afternoon each produced a confident, publishable number; three were caught **only** because a control read the same value as the thing measured |
| `four_families` | LON / LAT / TAC / STRAT promoted to the **top level**, never pooled | a perfectly complete *arm* record scores as UNKNOWN SCOPE, because the registry's keys are `four_families.<family>.<metric>` — and that silence reads exactly like compliance |
| `gaps` | every criterion this run could not measure, **named**, with reason + n | an eval that silently omits one reads at a glance exactly like an eval that has no gap |
| `criteria` | the checker's own verdict, embedded | criteria completeness is enforced by machinery, not memory |
| `provenance.resolved_trees` | which tree each package came from | see `--stack` above |

**Three defects in this tool were found by RUNNING the checker, not by reading it:** a top-level
`block` carrying an artifact *kind* was read as an unrecognised **tier** (`hyg.tier_stamp`
MISSING on a correctly stamped artifact); the disavowal audit trail **re-introduced the very
token it audits** at a fresh un-exonerated path; and `four_families.py`'s own explicit
disavowal wording is not in the checker's exoneration list, so a correct disavowal scored as a
use. All three are fixed **in the report**, never in the checker.

### The constant-only control's three readings

| check | expected | tolerance |
|---|---|---|
| `const0` paired against **itself** | `delta 0.0, [0.0, 0.0], separated False` | ⭐ **NONE — bit-exact** |
| `const0` ADE | mean over (window, slot) of ‖GT‖, derived **independently** in float64 numpy | rel 1e-4 — the production geometry path runs in **float32 torch**; the tolerance is stated, not hidden |
| `const0` LON speed MAE | mean GT speed (a zero path has speed 0 everywhere) | rel 1e-4, same reason |

⛔ **If any fails, the harness is wrong, not the model** — and the headline says so instead of
printing a family table.

---

## 4. DRY-RUN VALIDATION — what was actually run

**MEASURED 2026-09-03, dev box, CPU, 0 GPU.** The full path — rollout → dump →
`analyze_refcv3` → suite artifact → `criteria_check` → MD + HTML — over
`stack/tests/test_refcv3_arm.py`'s fixture: a **random-init** `RefCV3Model` at
`refc_v3_smoke_config(hier=True)` (encoder widened to 9 channels at 64 px) on a synthetic
3-clip v2 cache. **42 windows, 3 episodes, 6 arms + `const0`.**

Banked: `taniteval/results/openloop-suite-DRYRUN-refcv3-fixture.{json,md,html}`.

⛔ **THESE ARE INSTRUMENT CONTROLS ON A RANDOM-INIT MODEL. THEY ARE NOT EVIDENCE ABOUT
refcv3.** They are here because a control that reads a **known value** is the only thing that
shows the instrument works.

| what | read | the known value it had to read |
|---|---|---|
| criteria checker | **0 violations**, 1 work item (registry v2.5.0) | every required criterion PRESENT or REFUSED-with-reason |
| tier resolution | `T1` | the stamp, not the artifact kind |
| `const0` self-paired | `0.0 [0.0, 0.0]`, not separated | ⭐ bit-exact, no tolerance |
| `const0` ADE | 9.162552 vs expected **9.162551** (\|Δ\| 6.5e-07) | mean ‖GT‖, float64 reference |
| `const0` LON speed MAE | 7.283556 vs expected **7.283556** (\|Δ\| 0) | mean GT speed |
| `ha0` straight / const-speed | **1.0000 / 1.0000** on 42/42 | a zero-control unicycle rollout IS a straight line at constant speed |
| `os − ha0` ADE, paired | **+7.9571 [6.1530, 9.7625]**, separated → **LOST** | a random-init model must LOSE to constant velocity |
| selection profile | `n_distinct = 1`, modal #2 at **100 %**, entropy 0.0 → VOID-RISK | while `trivial_frac` read 0.0000 on the same arm |
| loop-vocabulary guard | **CLEAN**, 0 occurrences in either rendering | after refusing the first report at 4 sites |
| `pytest stack/tests/test_openloop_suite.py` | **18 passed** | — |
| `pytest stack/tests/test_refcv3_arm.py test_refav1_arm.py` | **35 passed** | no regression in the layer below |

### ⭐ A real UNVERIFIED item discharged for free

`REFCV3_ARM.md` §4.1 lists `rebuild_config` **via `argv`** as *"written and reviewed but never
exercised on a real run's `config.json`"* — and treats a `SystemExit` there as version skew that
would strand the final read. **MEASURED 2026-09-03 against the live run's own banked
`config.json`: it WORKS**, returning `(RefCV3Config, Namespace, str)` with
`horizons (5,10,15,20,30,40,50,60)`, `n_anchors 128 / pool 4096`, `image_hw (256, 640)`,
`in_channels 9`, `hier True`, `tac_vocab_version v7.0` — all matching §2.1. That path is no
longer a risk to tonight's read.

⚠️ It returns a **3-tuple, not a bare config** — a caller that treats the return value as the
config gets `AttributeError: 'tuple' object has no attribute 'core'`. Noted for anyone calling
it directly.

### What the dry run could NOT validate — say it plainly

* ⛔ **Nothing about refcv3 itself.** `ckpt_30000.pt` was **not reachable from this box**: every
  host in `~/.ssh/config` refuses on its advertised port (`tanitad-pod` 38.147.83.15:39198 and
  `tanitad-pod5` 69.30.85.106:22039 both `Connection refused`), and `tanitad-refcv3` is not in
  the config at all. The suite was therefore validated on the fixture, and **no number in the
  dry-run artifacts is a claim about the model.**
* **The lead-block join at real scale.** The fixture has no lead block, so
  `long.distance_keeping` is the single REFUSED work item — which is the correct state, and it
  is what proves the REFUSED path renders as a work item rather than vanishing.
* **`--arm-json` against a real banked record.** ⭐ The ROUND-TRIP **is** exercised: the arm
  record was banked standalone from the dump exactly as another stream would bank it, then
  consumed through `--arm-json … --dump-dir …` — **0 violations, `const0` OK, exit 0**. What is
  untested is that path against *refcv3's* record specifically; no other stream had banked one
  when this was built (`taniteval/results/` held none).
  ⚠️ Pass `--dump-dir` alongside `--arm-json`, or the constant-only control is REFUSED for want
  of the per-window arrays — the suite says so rather than dropping it.
* **Wall-clock at real scale.** Quote the arm tool's `[cost]` line from step 1, never a number
  from this file.

---

## 4b. ⛔ TWO THINGS THE REPORT MUST SAY, THAT A READER WOULD OTHERWISE ASSUME WRONG

### The run is NON-PARITY

MEASURED from the run's own `config.json`: `v2_parity.parity` **false**, `checked` **false**,
`corpus_key` **null** — and the trainer prints a non-parity warning on line 2 of its `train.log`.

⇒ **refcv3 is NOT cross-arm comparable with `refc-base` or `refc-xl`.** The programme's usual
parity does not hold here, and a reader must not assume it. The suite prints this as a banner in
§1 and again in §6 whenever `--parity-status` starts with `NON` — which is the **default**,
because assuming parity is the failure mode, so the tool assumes the opposite until told
otherwise.

⭐ **The margin over `ha0` survives this.** `ha0` is bit-identically defined across models
(`a = 0`, `κ = 0` at the measured `v0`, same integrator, same grid), so *each arm's margin over
its own `ha0`* stays comparable even when the corpora do not match. **A level against another
model's level does not.**

⚠️ Note the `train_eval_disjoint` value above is **arithmetic evidence, not a check**:
4,572 train clips = 4,713 − 141 eval clips is an exact identity, but `v2_parity.checked` is
**false**, so nothing verified it by episode id. That remains GATE 3, and it is the run's
property, not this tool's.

### The nav token is an ORACLE, so `os_navzero` gets EQUAL billing

`config.json`'s `nav_cmd_derivation` records the nav command as **oracle-derived** (provenance
`ego-future`), and the arm tool's own docstring says it *"will not exist at deployment"*. The
route input is therefore **optimistic by construction**.

⇒ the report prints **both margins side by side** — in the headline table, in every family
table, and in a dedicated §1.1:

| margin | what it is | when to quote it |
|---|---|---|
| `os − ha0` | the oracle-nav margin | the fair like-for-like against another arm that is also fed nav |
| `os_navzero − ha0` | the **deployment** margin, nav withheld | the one that answers *does this drive* |

⛔ **Quoting only the oracle-nav margin overstates the system.** Neither is a footnote.

⚠️ And `os_navzero` is a **LOWER BOUND on nav dependence**: under `nav_cmd=None` the E13
injection into the tactical and strategic states is removed entirely, but the **core** encoder
*collapses onto the majority token* (`follow`) rather than losing the input — so a model that
read nav only through the core would look *less* nav-dependent here than it really is.

---

## 5. Escalations for the Master Mind

0. ⛔ **`REFCV3_ARM.md` §3.1 and §5 name a checkpoint that is never written**
   (`ckpt_40284_FINAL.pt`; `MILESTONES` at `refc_v3_train.py:103` does not contain 40,284). The
   correction is §2 above. That document is another stream's, so it is **reported here and not
   edited**; the suite defends itself with `--expect-step`.
1. ⛔ **The tier ruling is still OPEN** (BACKLOG R30): does the doctrine admit as T1 a model
   that consumes **no actions**? The artifact stamps `T1` **and** carries
   `status: UNRULED` on every `os*` arm, and the **margin over the shared `ha0` floor** is
   meaningful whatever the ruling decides — so the numbers survive it either way.
2. **The three superseded regime labels** in §1 need a doctrine edit by whoever owns
   `EVAL_DOCTRINE.md` and the criteria registry. This tool reports them and edits nothing.
3. **`criteria_check._check_forbidden_estimator`'s exoneration list does not recognise
   `four_families.py`'s own disavowal wording** — a correct, explicit disavowal is scored as a
   use. Reported, not edited; the suite republishes with the canonical clause appended and lists
   every annotated site in `four_families._disavowal_annotations`.
4. **`CLAUDE.md`'s remedy for the editable-install trap is insufficient** as written —
   `PYTHONPATH=<clone>/stack` cannot beat a MetaPathFinder. `--stack` is the working remedy and
   the measurement is in §2.
