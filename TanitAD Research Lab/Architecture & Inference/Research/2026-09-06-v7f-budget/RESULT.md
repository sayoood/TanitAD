# v7f's declared trainable budget was 40 % fiction — and the biggest piece of it was the EMA teacher

**ArchInf FlyWheel · 2026-09-06 · 0 GPU** (CPU only; both GPUs busy — refcv5 on the A40,
an eval on Thor — neither touched. Dev-box RTX 4060 checked: no python compute, and not used.)
Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-v7f-budget/`

---

## 0. THE HEADLINE

My predecessor measured that **42 of 138 optimizer tensors — 5,305,667 params = 52.2 % of
v7-tiny's declared trainable budget — receive no gradient.** I reproduced that independently
(`raw/p1_fullcensus.json`, my own tree, same 42) and then asked the question the brief asked:
*is each one legitimately frozen, starved, or dead — and does the code say what is true?*

Three answers, all MEASURED:

1. ⭐⭐ **The 42 split 36 / 6, not 42 / 0.** Thirty-six are **STARVED** by objectives the recipe
   deliberately weights 0.0 (O1 → `step_readout_op`, O3 → `masked_cells`) and revive completely
   when those objectives are switched on. Six are **STRUCTURALLY DEAD** under *every* objective.
   **None** of the 42 is a legitimate stage freeze — the 130 stage-frozen tensors were already
   correctly outside the optimizer.
2. ⛔⛔ **At v7f's OWN production geometry there is a FIFTH family the v7-tiny census could not
   see, and it dwarfs all four: the O5 EMA teacher, 86,236,544 parameters, in the optimizer.**
   `_EmaCopy`'s own docstring says its parameters are *"buffers-in-spirit: `requires_grad=False`
   and **excluded from every optimiser**"*. Its constructor freezes them — and then
   `apply_stage_freeze` walks every named parameter, reads `ema_o5_enc.` → group `aux`, sees that
   S-W trains `aux`, and **un-freezes the teacher.**
3. ⭐⭐ **The mechanism was already named in this codebase, one guard over, and quantified at
   almost exactly the same number.** `assert_frozen_external`'s docstring: *"a frozen external
   encoder installed under the `encoder` group is UN-FROZEN by S-W (MEASURED, E-XENC-1 build:
   **86,580,480** foreign parameters would have trained…)"*. That guard is **fully built and
   called by NOTHING outside its own test file** — `grep -c reassert_frozen_external
   train_v6_staged.py` = **0**, `grep -c assert_frozen_external` = **0**, with `grep -c
   apply_stage_freeze` = **3** as the same-breath non-zero control. The trap it was written for
   was still live, on native modules instead of foreign ones.

⇒ **v7f, at the launch line `PREREG_V7F.md` §9 already carries, would have declared 234,909,315
trainable parameters and trained 140,192,128 of them — 59.7 %.** After this turn it declares
**143,949,315** and trains **140,192,128** — **97.4 %**. ⭐ **The effective number is IDENTICAL.
Nothing about what v7f trains changed. What changed is that the run stopped claiming 90,960,000
parameters no gradient could reach.**

---

## 1. THE 42, CLASSIFIED — (a) frozen / (b) starved / (c) dead

Arm: `v7tiny_postrain30k` (`run` `v6-staged-S-W`, stage **S-W**, horizons `[1,2,4]`, `o5_k` 8,
`o5_form` l1), at its own `config.json` weights: **O5 1.0 · O6_sigreg 0.1 · O1_ctrl/fact/scene
0.0 · O2 0.0 · O3 0.0**. Rebuilt from that config; geometry and freeze **bit-identical** to the
run's own record before anything was measured (`[ctrl]` line in `raw/p1_fullcensus.py`).

| family | tensors | params | group | **class** | justification, FROM SOURCE |
|---|---:|---:|---|---|---|
| `masked_cells.*` | 30 | **1,621,056** | `aux` | **(b) STARVED** | `v6_loss_step` guards the whole O3 block on `if w.o3_masked:` and its only consumer is `o3_masked_cell_loss(stack.masked_cells, …)`. The recipe sets `o3_masked 0.0`. **All 30 revive** under the mutation arm. |
| `step_readout_op.net.{0,1,3}` | 6 | **2,107,395** | `predictor_op` | **(b) STARVED — and the only EVAL HAZARD in the set** | reached only from `stage_a_losses(stack.predictor_op, stack.step_readout_op, …)`, inside the block guarded by `if w.o1_ctrl or w.o1_fact or w.o1_scene:`. All three are 0.0. **All 6 revive** under the mutation arm. |
| `predictor_op.heads.2`, `.4` | 4 | **1,052,672** | `predictor_op` | **(c) DEAD** | `OperativePredictor.trained_horizons = (1,)`; O5 reaches long horizons by applying head `'1'` autoregressively (`metric_dynamics.rollout_transitions`). **Survive the mutation arm** — dead under every objective. |
| `predictor_op.out_proj` | 2 | **524,544** | `predictor_op` | **(c) DEAD** | referenced **exactly once in the programme** — the line that constructs it. 989 readable `.py` files scanned; zero uses. **Survives the mutation arm.** |
| **(a) legitimately frozen** | **0** | **0** | — | — | the 130 stage-frozen tensors (`layer_tac`/`layer_str`/`planner`) were already outside the optimizer and were never part of the 42. |

**Mutation control** (`raw/p1_fullcensus.json`, key `on_mutation`): switching O1 and O3 on takes
the census **42 → 6 tensors, 5,305,667 → 1,577,216 params**, and the 6 survivors are exactly
`heads.2` + `heads.4` + `out_proj`. That is what makes (b) and (c) a measurement rather than a
reading.

### 1.1 ⛔ THE FIFTH FAMILY — visible only at v7f's own geometry

v7-tiny never passed `--o5-target ema`, so its census could not see this. v7f's launch line does
(`--o5-target ema --ema-decay 0.996`), and `build_stack_from_args` then constructs
`ema_o5_enc = _EmaCopy(stack.encoder)` and `ema_o5_ro = _EmaCopy(stack.readout)`.

| family | tensors | params | group | **class** | justification |
|---|---:|---:|---|---|---|
| `ema_o5_enc.*`, `ema_o5_ro.*` | 151 | **86,236,544** | `aux` | **(a) MUST NEVER TRAIN** | `_EmaCopy`: `forward` is under `@torch.no_grad`, `update` is `@torch.no_grad`. Its docstring: *"excluded from every optimiser … which is what makes `uplink="ema"` an X3-compliant target source rather than a second trainable path in disguise."* |

⚠️ **It was numerically harmless — and that is the trap, not the reassurance.** AdamW skips a
`None`-grad parameter *entirely*, decoupled weight decay included (pinned in
`test_grad_reach_census.py`), so the teacher never moved. The safety therefore rested on a
`no_grad` in one call site rather than on `requires_grad=False`. **One un-`no_grad` forward and
the teacher trains** — the exact X3 violation the class exists to prevent — with nothing in the
run's artifacts saying so, and with `config.json` reporting the 86 M as "trainable" throughout.

---

## 2. WHAT THE TRAINABLE COUNT READS, BEFORE AND AFTER

### 2.1 The fix, in one sentence

`apply_stage_freeze` set `requires_grad` from the **group map alone**. It now also honours a
**declaration a module makes about itself**: `models/_gradreach.py`'s
`declare_grad_unreachable(module, why)` — the same shape as the existing frozen-external flag,
kept **separate** because the two make different claims (*"someone else's weights"* vs *"no loss
in this ladder reaches this"*), and folding them together would change what
`assert_frozen_external`'s Direction-B check means. Three declarers:

* `_EmaCopy.__init__` — every EMA teacher, in its own constructor (**not** a `reassert_*` helper a
  caller must remember: that pattern is exactly the one that left the frozen-external guard
  unwired for months).
* `V6Stack.__init__` — `predictor_op.heads[k]` for every `k not in trained_horizons`.
* `OperativePredictor.__init__` — `out_proj`.

### 2.2 ⛔ THE SAME SCOPE ERROR, CAUGHT TWICE — once by reading, once by a suite regression

**First time, caught by reading.** My first version declared the untrained horizon heads in
**`OperativePredictor.__init__`**, which is where `trained_horizons` lives and reads like the
obvious home. It is wrong: **`refa_train.py:213` and `finetune_traj.py:248` both do `for k in
horizons: loss_pred += …`** — in those trainers heads 2 and 4 are genuinely trained. The
declaration would have silently frozen REF-A's multi-horizon objective, and REF-A's own tests
would still have passed (`test_refa.py` already carries a hand-written exemption asserting
`predictor.out_proj.*` has `grad is None`). `trained_horizons = (1,)` is a fact about **the v6
ladder**, not about the class — the `df`/Thor-`free`/`step_s` family, in a new costume.

⛔⛔ **Second time, caught by the SUITE — and I had reasoned my way past it.** I left `out_proj` in
the constructor on the argument that it is dead in *every* consumer, which is true. The controlled
suite run then produced **5 regressions**, and the decisive one was
`train_flagship_v4.py:1636`'s **not-frozen gate refusing every launch**:

> `TRUNK FROZEN — Sayed's hard requirement is that the encoder AND predictor train jointly (NO
> frozen part)`, `"trunk_tensors_frozen": 4`

plus three `test_v5_trainer_v2_val.py` gates. ⚠️ **The gate's INTENT is satisfied** — a tensor no
gradient reaches was never training, and the gate was previously *passing* on a model where those
tensors did not train. **Its PREDICATE is this turn's defect in miniature**: it reads
`requires_grad` as a proxy for "trains". But correcting a **PI hard requirement's** predicate is
not a v7f-budget deliverable, and silently tripping it is worse than leaving it.

⇒ **both declarations moved to `V6Stack.__init__`**, the consumer whose budget was wrong. REF-A,
flagship-v4 and `finetune_traj` keep byte-identical behaviour; their own (2-tensor) overstatement
is logged for their owners along with the predicate to fix. **Every headline number in this report
was re-measured after the move and is unchanged.**
`test_scope_a_BARE_predictor_keeps_every_horizon_trainable` now pins **both** halves — including
an explicit assertion that `out_proj` stays trainable in a bare predictor, so the flagship gate
cannot be re-broken quietly.

⭐ The lesson is not "I made a mistake twice". It is that **a true statement about a module is not
automatically true at the module's definition site**, and the only thing that caught the second
instance was a controlled suite run against a real baseline.

### 2.3 The numbers

**v7-tiny — `v7tiny_postrain30k` (`raw/p1_after.json`)**

| | before | after |
|---|---:|---:|
| total params | 19,300,297 | 19,300,297 *(unchanged)* |
| **declared trainable** | **10,169,731** | **8,592,515** |
| optimizer tensors | 138 | 132 |
| unreached tensors | 42 | 36 |
| unreached params | 5,305,667 | 3,728,451 |
| overstatement | **52.2 %** | **43.4 %** |

**v7f at `PREREG_V7F.md` §9's launch line, production geometry** (`raw/v7f_budget_before_after.json`;
both arms produced in ONE process, differing only in whether the declarations are present)

| | BEFORE | AFTER | AFTER + O1/O3 on |
|---|---:|---:|---:|
| total params | 245,588,568 | 245,588,568 | 245,588,568 |
| **declared trainable** | **234,909,315** | **143,949,315** | 143,949,315 |
| unreached | 94,717,187 (**40.3 %**) | 3,757,187 (**2.6 %**) | **0 (0.0 %)** |
| **EFFECTIVE trainable** | **140,192,128 (59.7 %)** | **140,192,128 (97.4 %)** | 143,949,315 (100.0 %) |
| optimizer tensors | 193/443 unreached | 36/286 unreached | 0/286 unreached |
| unreached, by module | `ema_o5_enc` 86,138,112 · `predictor_op` 4,723,456 · `step_readout_op` 2,107,395 · `masked_cells` 1,649,792 · `ema_o5_ro` 98,432 | `step_readout_op` 2,107,395 · `masked_cells` 1,649,792 | — |

### 2.4 The controls — every one of them positive, and each catching a different lie

1. ⭐ **The EFFECTIVE budget is bit-identical before and after: 140,192,128.** The change
   withholds a false claim; it does not change what trains.
2. ⭐ **`n_trainable_by_group_map_alone` reproduces the run's own recorded `freeze.n_trainable`
   EXACTLY** — 10,169,731 = 10,169,731. The new number is a withholding, not a different model.
3. ⭐ **The banked 30 k checkpoint still loads STRICTLY: `missing=[] unexpected=[]`.**
   `requires_grad` is not serialised. This is the property the whole design rests on.
4. ⭐ **Total parameter count unchanged** in every arm — nothing was deleted.
5. ⭐ **Withheld computed two independent ways and agreeing: 90,960,000** = (declared_before −
   declared_after) = the freeze audit's own `n_grad_unreachable`.
6. ⭐ **The residue is exactly the starved objectives and nothing else**: with O1/O3 on, the
   census reads **0 unreached**. If any structurally-dead tensor had crept back it would show here.
7. ⭐⭐ **AND A SEVENTH CONTROL FROM A COMPLETELY DIFFERENT MECHANISM — THE CHECKPOINTS
   THEMSELVES.** The census is a forward/backward argument; this one needs no model of the loss.
   md5 of the raw tensor bytes across **9 banked v7-tiny checkpoints**: `heads.1.weight` (the
   TRAINED control) has **9 distinct fingerprints**, while `step_readout_op.net.1.weight`,
   `out_proj.weight` and `heads.2.weight` take only **THREE** values total and **co-vary
   perfectly** — three *initialisations*, not three training outcomes. Six arms that trained to
   six completely different `heads.1` share one **bit-identical** `step_readout_op`. And
   `rdw8p30k`'s `step_readout_op.net.1.weight` and `heads.2.weight` are **BIT-IDENTICAL TO A
   FRESH SEED-0 INIT** after 30,000 AdamW steps at `lr 1e-4, wd 0.05`. Two independent probes,
   different mechanisms, same verdict — which is what *"repeated samples through one broken
   channel are one sample"* actually requires. Table in `raw/EXPOSURE_SWEEP.md` §1.

### 2.5 What the code now SAYS

* `apply_stage_freeze` returns `grad_unreachable` (subtree → **reason**), `n_grad_unreachable`,
  and `n_trainable_by_group_map_alone` — so the audit that ships in `config.json` says *what was
  withheld and why*, not merely that a number went down. An unexplained dead subtree is refused:
  `declare_grad_unreachable` raises on an empty reason.
* `grad_reach_census` now carries its own denominator — `trainable_numel`,
  `effective_trainable_numel`, `unreached_frac_of_trainable`. *(A percentage whose denominator is
  fetched from elsewhere is how `45,456/168,873` and `45,466/168,910` both came to read 26.9 %.)*
* The banner leads with the **budget**, not the tensor count: `[gradreach] EFFECTIVE trainable
  budget 140.19 M of 143.95 M DECLARED (97.4 %)`. "42 tensors" reads like a rounding error;
  "52.2 % of the declared budget" is the sentence that would have stopped v7-tiny being quoted
  as a 10.17 M-parameter arm.

### 2.6 ⭐ `--refuse-unreached` is wired into the launch line — with the companion that makes it survivable

`PREREG_V7F.md` §9 now carries `--refuse-unreached --allow-unreached step_readout_op masked_cells`,
plus a new §9.1 stating the measurement and the constraint it implies. **The `--allow-` half is
not a loophole; without it the flag is unusable and would be deleted.** §9's own do-not-add list
sets `--w-o1-* 0 --w-o3 0` for measured reasons, and those are the *only* weights that reach
`step_readout_op` and `masked_cells` — so a bare `--refuse-unreached` refuses that launch every
single time, and a flag that always refuses does not stay in a launch line. Naming the two
accepted modules turns a silent default into a **recorded decision**, and **anything nobody
decided about still refuses**. `test_refuse_ACCEPTS_only_the_modules_the_run_named` pins all three
branches, including the one that matters: a brand-new dead subtree refuses even while the two
known ones are allowed.

⚠️ **No objective, weight or criterion of the pre-registration was touched.** The edit adds a
diagnostic and writes down a decision §9 had already taken.

### 2.7 ⛔ THE HAZARD THIS TURN'S BIGGEST FINDING ACTUALLY IS

**`step_readout_op` is the METRIC TRAJECTORY READOUT** — latent transition → per-step Δpose — and
at O1 = 0 it is at **RANDOM INIT in every v7-tiny checkpoint and would be in every v7f arm
launched as pre-registered.** Meanwhile `V6Stack.roll_consistency`, `tanitad/eval/v6_probe_trunk.py`
and `stack/scripts/probe_saliency_p9.py` all decode through **the checkpoint's own copy of it**.

That is not a wasted-parameters problem. It is the `heads.2`/`heads.4` failure again: an untrained
module that is **READ** emits initialisation noise that looks like a measurement — which produced
one retracted action-divergence result (MM-E10 → MM-E14) and one false *"the model only imagines
0.1 s"* alarm. ⇒ the trainer now prints `[gradreach] ⛔ HAZARD step_readout_op: …` naming the
three consumers, and §9.1 states the reporting constraint: **no metric decode may be reported
from such an arm unless the readout is fitted at eval time and the report says so.**

### 2.8 The one migration cost, named rather than discovered later

Removing 157 tensors from the optimizer changes the optimizer's parameter-group size, so
`opt.load_state_dict` refuses a `--resume` from a **pre-2026-09-06** checkpoint — with torch's
own message, which points at the optimiser and reads like corruption. `load_resume` now checks
the slot counts first and raises a **named** `ResumeLineageError` saying what happened and what to
do (`--init-from` weights-only, or stay on the pre-change code), and explicitly refuses to
auto-repair: silently dropping the extra moments would land an Adam `exp_avg` on a different
parameter by list position, which is the exact failure `RESUME_CONTRACT` exists to prevent.

⚠️ **A second, smaller consequence, MEASURED and reported rather than papered over:** the per-stage
trainable-TENSOR counts move (S-W and S-J only). At the tiny test geometry **S-W falls 84 → 80 and
now COLLIDES with S-T**, so the *accidental* optimiser barrier no longer separates that pair.
`test_the_stage_counts_…_are_a_coincidence` existed to announce exactly this day, and its own
failure message said what to do: the real protection is `assert_resume_lineage`'s explicit stage
check, which **is** wired (`train_v6_staged.py:6285`, verified with a control, not assumed). The
test now **executes that guard on the colliding pair** instead of pinning the coincidence. At
v7f's production geometry the counts remain distinct (S-W 286 · S-T 76 · S-S 54 · S-J 416).

---

## 3. `predictor_op.out_proj` — THE PI DECISION, WITH ITS DEFAULT AND BLAST RADIUS

**What it is.** `self.out_proj = nn.Linear(state_dim, d)  # reserved: feed predictions back`.
1,573,632 params at v7f's geometry, 524,544 at v7-tiny's. **Referenced exactly once in the
programme: on the line that constructs it** (MEASURED: 989 readable `.py` files scanned; the only
other mentions are `nn.MultiheadAttention`'s unrelated `attn.out_proj`, REF-C's own different
`out_proj`, and — tellingly — `stack/tests/test_refa.py:103`, which already carries a hand-written
exemption: *"predictor.out_proj is reserved-but-unused in OperativePredictor.forward —
legitimately grad-free"*. **The codebase had already built a test exception AROUND the dead tensor
instead of retiring it.**)

### 3.1 DONE THIS TURN (non-destructive, no decision needed)

`out_proj` is declared grad-unreachable **by `V6Stack.__init__`**: it **leaves the optimizer and
the trainable count** for every v6/v7 arm, and **stays in the `state_dict`**. Every banked
checkpoint still loads strictly (control 3 above). The false claim is gone; the tensor is not.

⚠️ **Scoped to `V6Stack`, and here is the cost of that scoping, stated rather than hidden.** REF-A,
`train_flagship_v4.py` and `finetune_traj.py` also build an `OperativePredictor`, so **they still
count `out_proj`'s 2 tensors as trainable**. Declaring it in the constructor fixes them too — and
makes `train_flagship_v4.py:1636`'s not-frozen gate **refuse every launch** (§2.2). ⇒ **a work
item for those trainers' owners, not a silent breakage today**: the gate should test *"does
everything that CAN train, train?"*, which means excluding declared-unreachable subtrees from its
`requires_grad` count — the same one-line correction made here, applied to a PI hard requirement
that is not mine to touch.

### 3.2 THE DECISION THE PI OWNS: retire it, or wire it?

⛔ A tensor `predictor.py` calls *"reserved: feed predictions back"* is either **wired** or
**retired**. Leaving it counted-but-dead was the defect; leaving it *present* but dead is a
smaller, honest cost — and which of the three it should be is not mine to choose.

| option | cost | consequence |
|---|---|---|
| **DEFAULT — ⭐ KEEP AS DECLARED (what shipped today)** | zero | 1.57 M params ride in every checkpoint and in the sub-300 M budget forever, doing nothing. No load breaks. **Recommended unless the PI wants the budget back.** |
| **RETIRE (delete the tensor)** | one migration, below | recovers 1,573,632 params (0.64 % of v7f's 245.59 M) and ends the "reserved" fiction. |
| **WIRE (use it)** | a design change + its own pre-registration | it was reserved to feed predictions back into the predictor. That is an architecture decision, not a cleanup. |

### 3.3 BLAST RADIUS OF *RETIRE*, MEASURED — not asserted

* **11 of 11 banked checkpoints reachable on this box carry `predictor_op.out_proj` (2 tensors
  each)**: the ten `v7tiny_*` arms under `mm-e19-assets-20260901/` (`emao14_30k`,
  `emao14_30k_tauramp`, `k60clip05p30k` + its `ckpt_step17500_INTERIM`, `k8clip05p30k`,
  `o14fut30k`, `postrain30k`, `postrain30k_freeze`, `rdw8p30k`, `splitp30k`) **and REF-A's
  `refa-dinov2-4b/ckpt.pt`** — REF-A carries `out_proj` too, with no v6 heads and no
  `step_readout_op`, so a deletion in `OperativePredictor` hits REF-A as well. *(Programme-wide the
  count is larger — `predictor.py` itself says ~30 banked checkpoints; those live on Thor / pods /
  HF and were not reachable from this box. **INHERITED**, and it only makes the radius bigger.)*
* **MEASURED, by executing it:** with `out_proj` (and `heads.2`/`heads.4`) deleted from the class,
  a strict load of `v7tiny_postrain30k/ckpt.pt` **REFUSES** —
  `Unexpected key(s) in state_dict: "predictor_op.out_proj.weight", "…bias", "predictor_op.heads.2.*", "predictor_op.heads.4.*"` —
  and `strict=False` recovers it exactly: **missing = 0, unexpected = 6**.
* ⇒ **the migration is real but bounded**: a documented key-drop shim in `load_trunk_auto` /
  `load_stage_init` (drop those 6 keys, then strict-load) restores strictness. The cost is that
  **every** consumer that strict-loads an old checkpoint must go through the shim, and a consumer
  that is missed fails loudly rather than silently — which is the right failure direction.
* ⛔ **Do not couple the two.** `heads.2`/`heads.4` disappear on their own the moment a run passes
  `--horizons 1`; `out_proj` needs a class edit. Retiring `out_proj` is one decision, and the
  horizon default is a different one (§4).

---

## 4. ⛔ A GUARD THIS CODEBASE PROMISES AND DOES NOT HAVE

`predictor.py`'s own comment states: *"⇒ `train_v6_staged.py` refuses `--horizons` with any entry
!= 1 when CONFIGURING a run. See `_refuse_untrained_horizons` there."*

**MEASURED, two probes, each with a same-breath non-zero control** (`grep -c "def v6_loss_step"`
= 1 on both the local copy and the G: worktree):

| token | count |
|---|---:|
| `_refuse_untrained_horizons` | **0** |
| `allow_untrained_horizons` (ever passed) | **0** |
| `trained_horizons` (ever read by the trainer) | **0** |
| `--horizons` default | **`[1, 2, 4]`** |

⇒ the promised preflight **does not exist**, the authoritative declaration is read by nothing, and
the default still allocates two dead heads — 3,149,824 params at v7f's geometry. Today's change
makes them **untrainable and uncounted**, which removes the measurement defect. It does **not**
stop them being *allocated*: that needs either the missing preflight or a change to the
`--horizons` default, and changing a default silently changes the geometry of every new run.
⇒ **logged as a named gap with its cheapest fix, not smuggled in.** `--horizons 1` at v7f's
geometry: total **242,438,744** instead of 245,588,568, declared trainable **identical**
(143,949,315 — they were already withheld).

---

## 5. BLAST-RADIUS SWEEP — which claims rest on these arms

**82 rows swept** — every `GOALS_AND_CLAIMS.md` row naming a v7-tiny arm, plus the two mentioning
`step_readout` directly. Row-by-row table in `raw/EXPOSURE_SWEEP.md`. Scoped in the shape the
seed-noise rule (`H-ESTIM-SEED-1`) used: **a claim about a structural identity is not exposed the
way a number decoded through an untrained module is**, and ⛔ **nothing is retracted on this rule
alone.**

### 5.1 ⭐⭐ THE SCOPE CORRECTION THE SWEEP FORCED — THE DEFECT IS THE *RECIPE*, NOT THE ARCHITECTURE

The sweep surfaced an apparent contradiction, and chasing it produced the single most important
qualification of this whole finding. `E-DEC-20b` reports *"Group movement 2k→10k: `predictor_op`
0.1784, `readout` 0.1644, **`step_readout_op` 0.1095**"* for `splitfrz10k` — a module that, under
my finding, **cannot move**.

**MEASURED: the row is right, my finding is right, and they are about different recipes.**
`v6F-snapshots/sw_config.json` (`run` `v6-staged-S-W`, `out` `/home/nvidia/experiments/v6F-SW-30k`)
carries **`o1_ctrl 1.0 · o1_fact 1.0 · o1_scene 0.3 · o3_masked 1.0 · o5_rollout 1.0`** — the
**full objective set**. Every locally-banked **v7-tiny** `config.json` reads **`o1_* 0.0 ·
o3_masked 0.0 · o5_rollout 1.0`**, 4 of 4.

⇒ ⛔ **`E-DEC-20b` STANDS UNCHANGED, and the 52.2 % / 40.3 % starvation is a property of the
v7-tiny/v7f TWO-TERM RECIPE (O5 + O6) — NOT of the v6 architecture and NOT of the v6F family.**
Any claim resting on a v6F-era arm is outside this blast radius entirely. Had I not chased the
contradiction I would have over-claimed the radius by the whole v6F line.

### 5.2 The classification

| class | meaning | count | verdict |
|---|---|---:|---|
| **E1 — metric decode** | decoded to metres / Δpose / waypoints through the checkpoint's own `step_readout_op` (or `roll_consistency` / `v6_probe_trunk` / `probe_saliency_p9` / `eval_metric_rollout`) | **3** | **EXPOSED** |
| **E2 — dead-head read** | read off `heads.2` / `heads.4` / a multi-horizon prediction | **4** | **EXPOSED** |
| **E3 — budget quoted as trained** | the row quotes v7-tiny's count as an *effectively trained* number | **0** | — |
| **E4 — structural** | identity, exact zero, bit-exactness, argv/config audit, leak/disjointness fact, absence claim, latency | **39** | not exposed |
| **E5 — trained path** | O5/O6/O14 latent-space quantities: rank, participation, effective rank, drift, collapse, loss, probe-fitted ridge R² | **36** | not exposed |
| **E6 — cannot tell** | | **0** | — |

⭐ **E3 = 0 is itself a finding.** No row anywhere quotes an *effectively trained* count. The rows
that quote budgets use them correctly as **declared** values, and "~19 M-param" is accurate as a
**total**. What the register has never stated is the effective figure — **4,864,064** at v7-tiny.
That number is what this turn adds, and it is why the count had to be fixed in code rather than
in prose.

⭐ **E5 = 36 is the load-bearing half.** The starved modules sit on **neither** path any
latent-space claim uses: the encoder, readout, predictor blocks and `heads.1` are reached by
O5/O6/O14 and trained normally. The whole rank / participation / collapse / drift / decodability
body — `H-RANK-*`, `H-PROOF-1*`, `H-PROOF-4/6`, `E-DEC-1/3/4/55/67/69`, `MM-E6`,
`D-V7F-L1-MEASURED`, `D-V7F-DRIFT-NULL`, `D-V7F-L3-RULED` — is **NOT exposed**.

### 5.3 The 7 exposed rows

⭐ **`D-V7F-READOUT-DEAD` had already found the SYMPTOM; this turn supplies the MECHANISM.** That
row localised every T1 distance metric to a ≈0-motion readout, and
`D-T1-NO-SCALE-ON-EMISSION-PATH` had ruled out a harness scale error. The answer is **an untrained
random projection** — which also explains why `D-ROW3-CONSTANT-EMITTER-REFUTED` found the weight
term dominating the bias (so: not a constant emitter) and the module still emitting nothing
usable. A random projection is neither constant nor informative.

**E1 (3) — one causal chain, the programme's only T1 capability read:**

* **`D-T1-V7-READ`** (ade 14.069 / 13.879 m, fde 26.297, heading 94.63°, LON_speed 10.703). ⇒
  re-analyse the banked `thor:/home/nvidia/t1dumps/*/dumps/` rollouts with a readout **fitted at
  eval time on held-out windows**, and re-state as *latent decodability*, not driving skill; or
  re-run T1 on an arm trained at O1 > 0.
* **`H-ARCH-ACTINS`** (the "~1 %" closed-loop vs hold-action gap; `emao14_30k` ade +1.37 %). ⇒
  read the gap in **latent space** — the action-divergence probe this row itself committed to and
  `MM-E10` ran at h=1. That route never touches the readout.
* **`D-V7F-T1-NO-PAIRED-CI`** — only its *correction* of the "every distance metric" headline
  (cross-track 1.0722 / 1.1704, heading −0.4957, speed −0.1974). Its structural half —
  `paired_decision_grade` and `paired_legacy` are `{}` on all three arms — **stands untouched**.
  ⇒ compute the paired CIs in the **same pass** as a fitted-readout re-decode.

**E2 (4) — and the finding SHARPENS `MM-E14` rather than repeating it.** `MM-E14` said those heads
are unreached *in this recipe*; the mutation arm shows they are unreached under **every**
objective. ⇒ *"turn the loss on and re-read the heads"* **is not available**; the rolled-h=1 route
is the only one until the heads are wired. Rows: **`H-PROOF-2`** (fully), **`H-PROOF-3`** (h≥2
half only; the h=1 cos 0.3495 stands), **`H-PROOF-1B2b`** (horizon-decay half only; h=1 0.264×
stands), **`MM-E10`**'s `emao14_30k` h=2/h=4 cells (already formally withdrawn by `MM-E14`).

### 5.4 ⛔ ONE COMMITTED CRITERION IS UNSATISFIABLE BY CONSTRUCTION

**`MM-E11`'s `O1-WORKS` branch requires "h1 ratio rises ≥10× *and h2/h4 leave the floor*".**
`heads.2`/`heads.4` receive no gradient **even with O1 on**, so the second conjunct can never
fire and `o1ctrl30k` is pre-committed to at best `O1-INSUFFICIENT` on a criterion nothing could
meet. ⛔ **This is not a licence to move a goalpost after seeing data:** the correct action is to
record that the criterion was unsatisfiable *when it was written* and amend it **before** the arm
is adjudicated, not after.

---

## 6. WHEN v7f FINALLY GETS ITS GPU-DAYS

⭐ **At `PREREG_V7F.md` §9's launch line exactly as it now stands, v7f will train
140,192,128 of its 143,949,315 declared trainable parameters — 97.4 % — where the same launch
line before this turn would have trained the same 140,192,128 while declaring 234,909,315, i.e.
59.7 %; the missing 2.6 % is `step_readout_op` and `masked_cells`, starved by the pre-registered
`--w-o1-* 0 --w-o3 0` and now named on the record by `--allow-unreached`.**

---

## 7. EVIDENCE CLASSES AND ARTIFACTS

Every number above is **MEASURED (ours)** on CPU in `C:\Users\Admin\tanitad-v7fbudget`, a copy of
the repo tree verified byte-identical to the branch worktree by md5 (`train_v6_staged.py`
`d2ade650…`, `v6.py` `09cb5346…`, `predictor.py` `6656e296…`) before any edit, with
`PYTHONPATH` pointed at the clone and the imported tree re-checked (`tanitad.__file__`) — the
editable-install trap. The one **INHERITED** figure is flagged in §3.3 (the ~30 programme-wide
checkpoints).

| artifact | what it holds |
|---|---|
| `raw/p1_fullcensus.py` / `.json` | the 42, reproduced in my tree, with the O1/O3 mutation arm |
| `raw/p1_after.py` / `.json` | v7-tiny before/after, with the three controls (total unchanged · old rule reproduced · strict load OK) |
| `raw/v7f_budget_census.py` | the v7f launch line built through the trainer's own parser |
| `raw/v7f_budget_before_after.py` / `.json` | the v7f table, both arms in one process |
| `raw/EXPOSURE_SWEEP.md` | the row-by-row exposure classification |
| `raw/SUITE_BASELINE_VS_AFTER.md` | the controlled suite comparison |
