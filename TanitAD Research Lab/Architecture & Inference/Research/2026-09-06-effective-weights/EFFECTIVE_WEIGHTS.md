# The effective weight is not the argparse default — and nothing told the operator

**Date** 2026-09-06 · **Agent** effective-weight · **Branch** `agent/arch-inf-20260803`
**Evidence class** MEASURED (ours) unless marked otherwise · **Tier** n/a — this is a
*launch-time* instrument, not an eval. No capability claim is made here, so no T-tier applies;
the four metric families are likewise not engaged (nothing was evaluated).

---

## 1. The defect

`V6LossWeights.for_stage(stage)` in `stack/scripts/train_v6_staged.py` **rewrites the argparse
defaults per stage**. So

```
python3 scripts/train_v6_staged.py --stage S-T --w-o5 1.0 ...
```

**trains nothing on O5**, stamps `w_o5: 1.0` into the `args` block of `config.json`, and prints
not one word about it. The operator asked for a term; a layer they cannot see discarded it; the
run record advertises the term they asked for.

⭐ **This is the mechanism behind the already-measured v7f defect** — *42 of 138 optimizer tensors
received no gradient, 5,305,667 params = 52.2 % of a declared trainable budget* — because a
zero-weighted term is **guarded out** of the loss (`if w.seam_op:`) and its modules never enter the
autograd graph at all. ⭐ **`p.grad is None` is the sound discriminator; the weight's value never
was.**

### 1.1 The zeroing table, DERIVED from `for_stage` itself

MEASURED by pushing a probe whose every float term is 1.0 through the real
`V6LossWeights.for_stage`:

| stage | terms zeroed | which |
|---|---|---|
| **S-W** | **10** | `lambda_plan, s1_latent, seam_op, t1_latent, w_anchor, w_s1_multi, w_s2_goal, w_select, w_t2_contrast, w_t5_consist` |
| **S-T** | **10** | `o1_ctrl, o1_fact, o1_scene, o2_nearfield, o3_masked, o5_rollout, o6_sigreg, s1_latent, w_s1_multi, w_s2_goal` |
| **S-S** | **14** | the seven O-terms + `lambda_plan, seam_op, t1_latent, w_anchor, w_select, w_t2_contrast, w_t5_consist` |
| **S-J** | **0** | — |

⚠️ **Two corrections to the enumeration this work started from.** S-T zeroes **10** terms, not the
nine reported — **`w_s1_multi` was missed**. And **S-S was not enumerated at all**; it zeroes
**14**, the largest set of the four. Both are pinned in
`stack/tests/test_v6_effective_weights.py::test_the_measured_zeroing_counts`.

⭐ **The table is derived, never copied.** `stage_zeroed_terms(stage)` calls the real `for_stage`,
so editing `for_stage` moves the guard with it. A hand-copied list is a second source of truth,
and this programme has measured repeatedly that the copy rots — the "2 of 36 features" count went
stale **four times**, inside the very rule warning about stale counts.

---

## 2. What must be refused, and what must not

This is the whole design, and getting it wrong makes the guard worse than the silence it replaces.

| case | verdict | why |
|---|---|---|
| default is `0.0`, operator silent | `OFF_BY_DEFAULT` — **pass** | the zero defaults exist so that *"adding the seams to the code cannot change a run that does not ask for them"* |
| non-zero default, stage zeroes it, operator silent | `OFF_BY_LAYER` — **pass** | the staged ladder **working as designed**; refusing here fires on every honest launch |
| operator explicitly passes `0.0` | `OFF_BY_OPERATOR` — **pass** | declared and recorded |
| ⛔ operator explicitly passes non-zero, a layer zeroes it | **`DISCARDED` — REFUSE** | the launch line advertises a term that trains nothing |
| ⛔ effective > 0 but the module was never built | **`NO_GRAPH` — REFUSE** | a head "built, stamped, and supervised by nothing" |
| effective > 0, term rides a validity mask | `TRAINS_IF_MASK` — **pass, annotated** | see §4 |

⛔ **"Explicit" is read from `argv`, never from the value.** `explicit_dests()` re-parses the *same*
command line against a parser whose every default is a unique sentinel; anything not a sentinel was
typed. This is exact — it gets `--flag=value`, abbreviations, `nargs` and `store_true` right because
argparse itself does the parsing — and it matters because **`--w-o5 1.0` passes the default value**,
so a value comparison cannot tell it from silence, and the two must produce opposite verdicts.
Pinned by `test_explicitness_comes_from_argv_and_not_from_the_value`.

⚠️ When the parser/argv are unavailable the audit reports `explicit_source: unavailable` and
**refuses nothing**, rather than reading UNKNOWN as "nothing was explicit" — which is how a guard
becomes cover.

---

## 3. The two-sided mutation proof

⭐ *A guard that refuses everything gets deleted; one that refuses nothing measured nothing.*

**v6 — 8/8, `_preflight_effective_weights` only** (so unrelated preflight noise cannot make a
one-sided guard look two-sided):

| case | expect | got |
|---|---|---|
| REINTRODUCE `--w-o5 1.0` on S-T | REFUSE | REFUSE |
| CONTROL `--w-o5 1.0` on **S-J** (stage keeps it) | PASS | PASS |
| CONTROL silent on S-T (default 1.0 zeroed) | PASS | PASS |
| CONTROL `--w-o5 0` on S-T | PASS | PASS |
| REINTRODUCE `--w-o1-ctrl 1.0` on S-T | REFUSE | REFUSE |
| REINTRODUCE `--w-s1-multi 1.0` on S-T *(the term the enumeration missed)* | REFUSE | REFUSE |
| ESCAPE HATCH `--w-o5 1.0 --allow-discarded-weights` | PASS | PASS |
| REINTRODUCE `--w-t1 1.0` on S-W | REFUSE | REFUSE |

**refc — 8/8**, including the live arm:

| case | expect | got |
|---|---|---|
| ⛔ **LIVE refcv5** (`--sampler ddim --w-u0 0.5 --agents off`) | **PASS** | **PASS** |
| REINTRODUCE `--w-u0 0.5` with `--sampler none` | REFUSE | REFUSE |
| CONTROL `--sampler none`, `w_u0` at its 0.0 default | PASS | PASS |
| REINTRODUCE `--w-agent 1.0` with `--agents off` | REFUSE | REFUSE |
| CONTROL `--w-agent 1.0 --agents head --agent-join …` | PASS | PASS |
| REINTRODUCE `--goal-point-w 1.0` without `--goal-point-inject` | REFUSE | REFUSE |
| CONTROL `--goal-point-w 1.0 --goal-point-inject` | PASS | PASS |
| REINTRODUCE `--agent-w-project 1.0` with `--agents off` | REFUSE | REFUSE |

⚠️ **The controls earned their keep on the first run.** Three refc CONTROLS read as *refusals*
because `--arm` is required and `parse_args` raised `SystemExit(2)` **inside the `try`** — an
argparse exit swallowed as if it were my guard's. The harness looked two-sided while measuring
nothing. Parsing now happens **outside** the `try`. This is the same family as
*"a probe that tunes on the data it scores"*: only a control that must read a known value catches it.

⭐ **Flag design.** The refusal ships **with** `--allow-discarded-weights`, per the
`--refuse-unreached` / `--allow-unreached` precedent — a bare refusal that fires on honest launches
gets deleted. The override does **not** hide the finding: it is stamped as
`acknowledged_discarded: true`, so a run that overrode the guard says so in its own artifacts.

---

## 4. ⚠️ Distinguishing a masked term from a zeroed one

MEASURED this week: the route head's apparent zero gradient was a **validity mask, not a dead
head** — forcing `route_valid=True` moved the loss **0.0 → 0.687** and produced gradient. A zero
loss can come from **four** places, and the table names which:

| source | column that says so |
|---|---|
| the argparse weight | `OFF_BY_DEFAULT` / `OFF_BY_OPERATOR` |
| a stage override | `OFF_BY_LAYER` / `DISCARDED`, with `layer=for_stage('S-T')` |
| a missing structural precondition | `NO_GRAPH`, with `missing=…` |
| an all-invalid batch | `TRAINS_IF_MASK`, with `mask=…` — **the term is alive; check the mask** |

MEASURED, `--stage S-S --w-s2-goal 1.0 --s2-labels …` reads `TRAINS_IF_MASK`, `builds_graph=True`,
`mask=s2_valid (+ g_str_valid/a_str_valid; v7.2 arg L1 is ALL-ZERO masked by
V72_ARGS_SUPERVISED=False)`. The **same weight on S-T** reads `DISCARDED`, `builds_graph=False`.
Two different zeros, two different verdicts. Pinned by
`test_a_masked_term_is_distinguished_from_a_zeroed_one`.

---

## 5. ⭐⭐ Correctness and wiring are different claims

A guard was recently found in this programme **written, working when called, and called from one
launch path of two**. So there are tests whose entire job is asserting the preflight is *invoked*.

**v6** — `main()` gates both `dry_run(a)` and `train(a)` behind `preflight(a)`. MEASURED by
monkeypatching **both** entry points and launching a `DISCARDED` command: `main` returns **2** and
**neither** is entered, on the real path *and* under `--dry-run`. The positive half asserts the
ordering `preflight → dry_run` / `preflight → train` on a clean launch, so the negative test is
measuring a refusal and not a broken `main`.

⛔ **refc is where this was not theoretical.** `refc_v3_train.main` runs `preflight` **only** under
`--preflight` and otherwise calls `train` directly:

```python
args = ap.parse_args(argv)
if args.preflight:
    raise SystemExit(preflight(args))
train(args)          # <- the REAL path never touches preflight()
```

That is exactly why every existing guard in that file (`_check_nav_from_v7_args`,
`_check_goal_point_args`, `_read_anchor_artifact`, `_check_anchor_artifact_against_cfg`) is
**double-called**, and the new one is too. `test_refc_guard_is_on_the_real_path` monkeypatches the
guard to a recorder and asserts it fires from **`train()`** and from **`preflight()`** separately.

---

## 6. What each trainer got, and honest scope

### `train_v6_staged.py` — a live hole, closed

Five (stage, term) pairs had hand-written refusals (`w_t2_contrast`, `w_t5_consist`, `w_s1_multi`,
`w_select`, `w_anchor`). **The other ~29 did not** — including the headline `--w-o5` on S-T. The
new layer is general, derived, and exhaustive over `V6LossWeights`
(`test_every_float_loss_weight_is_in_the_audit` fails if a term is added without an entry).

⚠️ **`seam_op` has NO FLAG.** `_weights_from_args` never sets it, so it is always the dataclass
`1.0` and then zeroed in S-W/S-S. It is **listed** in the table rather than omitted, because a term
absent from the table reads as a term that does not exist.

### `refc_v3_train.py` — no live hole; earlier failure + auditability + a contract

⚠️ **Honest scope, MEASURED:** refc has exactly **five** weight-like flags — `--w-u0`, `--w-agent`,
`--agent-w-project`, `--agent-w-ground`, `--goal-point-w` — and **all five were already gated**.
Claiming otherwise would be manufacturing a defect. The layer earns its place three other ways:

1. **it fails earlier.** `w_u0 > 0` with no `control_head` is caught today by
   `assert_seams_are_built`, which runs *after* the corpus mounts and the model is built. Now it is
   caught in the preflight, in milliseconds — the `--gate-probes` lesson.
2. **the run record becomes auditable.** `config.json` stamped `w_agent`/`w_u0` as bare numbers,
   with nothing saying whether the operator typed them or whether the term builds a graph.
3. ⛔ **`REFC_WEIGHT_GATES` is exhaustive over the parser** — a new `--w-*` cannot ship ungated.
   That contract is the durable half; the refusals are the cheap half.

⛔ refc has no stage layer, so `DISCARDED` cannot arise there and **no `--allow-discarded-weights`
was added** — a flag that can never fire is not a feature.

---

## 6.1 ⛔⛔ A REFUSAL THAT CANNOT BE PRINTED HAS REFUSED NOTHING

MEASURED 2026-09-06, and found only because the suite was run as a controlled comparison rather
than trusted: on the **cp1252 dev box every preflight refusal in `train_v6_staged.py` died inside
its own `print`** --

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u26d4'
```

-- so the process exited **1 with a traceback** instead of **2 with the reason**. The guard was
correct, reached, and **mute**. ⚠ **Confirmed PRE-EXISTING**: the identical argv fails
byte-identically on the pre-change trainer, so this is not a regression. But it made **every**
refusal in the file invisible exactly where an operator meets it -- the new effective-weight one
included, which is the entire deliverable.

⚠ **And fixing the leading glyph alone was NOT enough.** With the marker made conditional the
very next run died on `'\u21d2'` **at position 322 -- inside another guard's message body**. The
messages in this file are full of arrows and warning signs, so the fix cannot be to sanitise
content one glyph at a time; it has to be **at the write**. `_print_refusal` now degrades to
backslash escapes instead of raising: the reason always reaches the operator and the exit code
stays **2**. Verified end-to-end through `main()` on cp1252 -- `rc=2`, four refusals printed,
**stderr empty**, including
`REFUSED: --w-o5 1 in --stage S-T: for_stage('S-T') forces the effective weight to 0.0 ...`.

⭐ Same family as `--refuse-unreached`: an instrument is not finished when it is *correct*, only
when it *lands*.

---

## 6.2 The suite, as a controlled comparison

⛔ **Nine failures appeared alongside `test_v6_effective_weights.py`'s 39 green. Exactly ONE was
mine, and I fixed it.** Each was classified against evidence, not assumed:

| failing test(s) | verdict | how it was settled |
|---|---|---|
| `test_v6_s2_loss::test_preflight_refuses_weight_without_labels_on_a_REAL_run` | ⛔ **MINE -- FIXED** | its second half asserts the OPPOSITE for a dry run (*"a dry-run may smoke the loss on synthetic keys without labels"*); my label precondition fired there |
| `test_v6_chain` × 2 | **pre-existing** | a **both-directions failure-ID diff** loading the pre- and post-change trainers side by side and comparing `preflight()` on the chain's own argv: **ADDED=0, REMOVED=0 on all four stages** -- both refuse on `--horizons (1, 2)` |
| `test_refc_v3` × 3 (param counts) | **pre-existing** | import-graph proof: a fresh interpreter importing exactly what that test imports leaves BOTH `tanitad.effective_weights` and `refc_v3_train` absent from `sys.modules`, with a same-breath control asserting `tanitad.refs.refc_v3` **is** loaded |
| `test_v6_st_launch_fixes` E4/E5, `test_v6_s2_loss::test_the_80_ex_lane_change_rows` | **G: mount flap** | `OSError: [Errno 22] Invalid argument` raised inside `importlib`/`pathlib`/the label reader -- the documented mount failure, not an assertion |

⭐ **THE ONE THAT WAS MINE IS THE INTERESTING ONE, AND IT SHARPENED THE DESIGN.** A **module**
precondition (`--selector none`: the scorer was never built) is a structural absence in **every**
mode. A **label** precondition (`--w-s2-goal` without `--s2-labels`) binds only on a run that
**trains** -- a dry run legitimately smokes the loss on synthetic keys. Conflating them is exactly
the false alarm §4 warns about, one level up. Pinned by
`test_a_LABEL_precondition_is_exempt_under_dry_run_but_a_MODULE_one_is_not`.

---

## 7. ⛔ No weight was changed

Flipping a default silently changes the recipe every banked arm was trained under, and the arms are
the comparison basis. **This work adds visibility and refusal only.** Where a zero weight looks
worth trying, it is named below as a pre-registered future arm, not enabled.

### Candidate arms (NOT run, NOT enabled)

| candidate | why it is interesting | what it would cost |
|---|---|---|
| `w_s2_goal > 0` on **S-S/S-J** with `--s2-labels` | the strategic goal supervision is the hierarchy thesis and is `0.0` everywhere today; S-S/S-J are the stages that keep it | one S-S arm + its replicate; needs a pre-registration and the s2 label join |
| `w_t2_contrast > 0` on **S-T** with `--t2-contrastive` | the manoeuvre contrastive is catalogued (F-7/T2) and never switched on | one S-T arm + replicate |
| `o10_psg > 0` with `--psg-labels` | the only term in the audit with a real per-clip mask; `psg_valid` is 1.0 only for clips in the train join | needs the obstacle join shipped to the training pod |

⛔ Each needs its own pre-registration with **both outcomes committed in advance**, and — per
`H-ESTIM-SEED-1` — a **replicate arm**, because a separated CI from a one-seed arm is necessary and
not sufficient.

---

## 8. The two INCONCLUSIVE items, resolved

### 8.1 `refc_agents.py` internal sub-weights — RESOLVED, and it was a **wrong-file** probe

The earlier pass reported that `presence`/`cls`/`centre`/`size`/`yaw` *"failed every read probe"*.
They are **not in `refc_agents.py` at all**. `agent_losses` delegates to `slot_set_loss`, which
lives in **`stack/tanitad/models/agent_slots.py`**, where:

```
SLOT_LOSS_W = {"presence": 1.0, "cls": 1.0, "centre": 1.0, "size": 1.0,
               "yaw": 1.0, "rates": 0.5, "occ": 0.5}
NO_OBJECT_W = 0.1
```

⭐ **Two findings that matter.** (a) **None is 0.0**, so none is an advertised-but-inert term.
(b) **None is operator-reachable**: `refc_v3_train.py` calls
`agent_losses(slots_ag, tgt_ag, core.agents, cam=…)` **without `weights=`**, so these are
*constants*, not launch flags, and correctly outside an audit about operator-supplied weights being
discarded. They are also already mask-disciplined: `slot_set_loss` reports every term **with its
`n`**, and *"a term with `n == 0` is reported as 0.0 WITH its count, never dropped"* — which is
exactly the §4 distinction, made independently and earlier.

⚠️ **Root-cause class: absence found at ONE location is not absence.** The original probe searched
the consumer and concluded the constants did not exist.

### 8.2 `refa_v1_train.py` — ENUMERATED (⛔ not edited: outside this agent's ownership)

Three weight flags, all defaulting `0.0`:

| flag | gate | verdict |
|---|---|---|
| `--w-cf` | `sanity()` in **`refa_v1.py`** (a *model* method, not a trainer function) refuses `w_cf` with `cf_negs < 1` and with `cf_at_step` outside `[1, op_steps]` | **gated** |
| `--w-sigreg` | none — but the term is **self-contained** (`if self.cfg.w_sigreg:` … no external target or head) | **ungated, correctly** |
| `--w-aux-head` | bounds only (`>= 0`); a `SystemExit` fires **only under `--smoke`** | ⚠️ **thinnest gate of the three** |

⭐ **`--w-aux-head` is this defect class in a different costume, and the source says so.** The
trainer's own comment records that the loop read
`out["loss"] + cfg.w_aux_head * torch.zeros(())` — *"the config carries `w_aux_head 0.1`, the launch
record would show it, and the term contributes EXACTLY nothing"* — and names it *"the defect
`sanity()` refuses for `w_cf`, sitting in the trainer."* The current guard refuses only under
`--smoke`, on the stated grounds that with a real `--cache` the loader's `(a, kappa)` **is** the
demonstration and the term is live. That claim is **INHERITED here, not re-verified** — verifying it
means checking `p.grad is not None` on the aux head under a real cache, which is a one-arm probe for
whoever owns `refa_v1_train.py`.

---

## 9. Artifacts

| what | where |
|---|---|
| the engine | `stack/tanitad/effective_weights.py` *(new)* |
| v6 audit + refusal + `--allow-discarded-weights` | `stack/scripts/train_v6_staged.py` |
| refc audit + refusal + `REFC_WEIGHT_GATES` | `stack/scripts/refc_v3_train.py` |
| 39 tests incl. both wiring tests | `stack/tests/test_v6_effective_weights.py` *(new)* |
| this note | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-effective-weights/EFFECTIVE_WEIGHTS.md` |

**The stamp.** `config.json` now carries `effective_weights`: per term the argparse default, what
the operator asked for, the effective value, **which layer changed it**, whether the operator typed
it, whether it **builds a graph**, its mask, and its missing precondition — plus `explicit_source`
and `acknowledged_discarded`. ⛔ A run record that does not carry its effective weights cannot be
audited afterwards, and three arm-substitutions have already been found in this programme.
**A gate row carries its arm.**

---

## 10. The one-line answer

**Can an operator still pass a weight that trains nothing without being told?**

⛔ **In `train_v6_staged.py` and `refc_v3_train.py`: no** — an explicitly passed weight that a stage
or a mode gate discards now **refuses at launch**, on **every** path that can start training, and
every run stamps its full effective-weight table. **Elsewhere: yes** — `refa_v1_train.py`'s
`--w-aux-head` is gated only under `--smoke`, and any trainer outside these two is unaudited.
