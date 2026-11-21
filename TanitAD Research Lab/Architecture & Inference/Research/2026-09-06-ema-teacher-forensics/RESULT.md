# The O5 EMA teacher was in the optimizer and NEVER TOOK A STEP — the defect is the BUDGET, not the algorithm

**ArchInf FlyWheel · 2026-09-06 · 0 GPU** (CPU only. The A40 is running refcv5's 44 h training —
not touched. Thor not touched. Dev-box RTX 4060 not used: all of P1–P3 is checkpoint forensics
plus a mutation test.)
Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-ema-teacher-forensics/`

---

## 0. THE ANSWER, IN ONE LINE

⭐⭐ **WAS THE v7 RECIPE EVER TESTED AS SPECIFIED? — On the EMA teacher: YES.**
The O5 teacher received **zero optimizer steps** in both banked `--o5-target ema` arms. Its target
space was genuinely non-self-generated, exactly as the recipe says. **My predecessor's un-freeze
finding is real and its severity is OUTCOME 2: a budget/accounting defect, not an algorithmic
one** — and per the brief I am not going to inflate it.

⚠️ **That is the answer for the TEACHER, and it is not a clean bill of health for the recipe.**
The predecessor's separate `step_readout_op` hazard (E1, 3 rows) is untouched by this turn and
still stands.

---

## 1. THE DISCRIMINATOR, PRE-STATED BEFORE ANY MEASUREMENT

Full text: `raw/PREREG_DISCRIMINATOR.md`, written before the first number was produced.

⛔ **The thing that had to be tested, and why a code reading would not do it.** The predecessor
concluded the teacher "never moved" because `_EmaCopy.forward` is under `@torch.no_grad`, so
`p.grad is None` and AdamW skips it. That is an inference from **one** call site to **every** call
site. `in the optimizer` does not imply `received updates`, and `no_grad here` does not imply
`no gradient anywhere`.

**The two hypotheses were separated on decoupled weight decay.** torch's AdamW collects a parameter
only `if p.grad is not None`, and applies `p *= (1 - lr*wd)` to **every parameter it collects**,
whatever the gradient's magnitude. So:

| | H_EMA (teacher skipped) | H_GRAD (teacher stepped) |
|---|---|---|
| **D1** `r = \|\|teacher\|\|/\|\|student\|\|` | `r ~ 1.0` | `r <= 0.8607` at the arm's own `lr 1e-4, wd 0.05, N 30,000` |
| **D2** frozen-tensor identity | rel. deviation `<= ~1e-6` | `~1.4e-1` |
| **D3** geometry (corroborating only) | `cos ~ 1` | weaker, explicitly not decisive |

**Committed decision rule:** median `r > 0.99` ⇒ H_EMA; `r < 0.95` on a majority ⇒ H_GRAD;
between ⇒ INCONCLUSIVE. ⛔ No threshold below that line was changed after seeing data.

### 1.1 ⭐ ONE PROBE WAS ADDED AFTER RECONNAISSANCE, AND IT IS DECLARED AS SUCH

Reconnaissance (C3) showed the checkpoints carry **`opt`, the optimizer's own state dict**. That
enables a **direct** test the pre-registration did not anticipate — **D0** — because torch's AdamW
`_init_group` creates `state[p]` **only** for parameters it collects, and `state[p]["step"]` counts
the steps actually taken. D0 is therefore *post-hoc-enabled*, it **replaces nothing**, and the
pre-registered rule is reported on its own terms below.

### 1.2 The controls, and what each of them actually read

| id | control | required value | MEASURED | verdict |
|---|---|---|---|---|
| **C1** | `predictor_op.heads.1.weight` md5 across the 9 banked arms | **DIFFERENT** | 9 arms, **9 distinct** | ✅ the reader reads real per-arm bytes |
| **C2** | `out_proj` / `heads.2` / `step_readout_op.net.1` md5 across arms | **IDENTICAL** | 9 arms, **3 distinct each** | ✅ the instrument can see "no gradient ⇒ no movement" |
| **C3** | `ema_o5_enc.*` present | **PRESENT** | **41 + 2 tensors**, 980,288 params, in 2 of 9 arms | ✅ D0–D3 can run |
| **C4** | the arm's own record says `ema` | **`--o5-target ema`** | `o5_target "ema"`, `ema_decay 0.996`, `lr 1e-4`, `wd 0.05`, `steps 30000`, `seed 0`, stage `S-W` | ✅ correct arm |
| **C5** | rebuild reproduces the run | **exact** | `n_trainable` **11,742,915 == 11,742,915** recorded | ✅ |

⭐ **C2 reproduces the predecessor's independent measurement exactly** (3 fingerprints across 9
arms for each dead column), which is what makes this instrument admissible for a null result.

⛔ **C4 matters because A GATE ROW CARRIES ITS ARM.** The two arms are
`v7tiny_emao14_30k` and `v7tiny_emao14_30k_tauramp` (differing only in `ema_decay_ramp: cosine`).
**They are the only banked arms that carry a teacher at all** — the other 7 have no `ema_o5_*` and
no `frozen_o5_*` module, so this finding's blast radius is exactly two checkpoints. v7f itself has
never been trained.

---

## 2. ⭐⭐ THE RESULT — OUTCOME 2, ON FOUR PROBES WITH DIFFERENT MECHANISMS

### 2.1 D0, at NAME level — the optimizer's own record (`raw/d0_namemap.json`)

The arm was rebuilt from its **own recorded args**, frozen by the **pre-2026-09-06 rule**
(group map alone — what actually trained it), and every optimizer slot mapped to its parameter name.

| | measured |
|---|---:|
| rebuilt trainable params (group map alone) | **11,742,915** — *identical to the ckpt's own `freeze.n_trainable`* |
| rebuilt trainable tensors | **185** — *identical to the ckpt's 185 optimizer slots* |
| shape alignment of the index→name map | **OK** (every state entry's `exp_avg` matches its named param's shape) |
| slots the optimizer **STEPPED** | **100**, all at `step = 30,000` |
| slots **NEVER STEPPED** | **85** |
| ⭐ **EMA-teacher tensors STEPPED** | ⭐ **0 of 43** |
| ⭐ **CONTROL — encoder STUDENT tensors stepped** | ⭐ **41 of 41** (0 skipped) |

**The 85 never-stepped, by module:**

| module | tensors | params |
|---|---:|---:|
| `step_readout_op` | 6 | 2,107,395 |
| `masked_cells` | 30 | 1,621,056 |
| `predictor_op` (`out_proj`, `heads.2`, `heads.4`) | 6 | 1,577,216 |
| **`ema_o5_enc`** | **41** | **972,032** |
| **`ema_o5_ro`** | **2** | **8,256** |

⭐ **The non-EMA rows sum to exactly 42 tensors / 5,305,667 params — the predecessor's headline
figure, reproduced from a completely different instrument** (their census ran a real backward; this
reads the banked optimizer state). Two mechanisms, one answer.

### 2.2 D0 again, NAME-FREE — the shape-multiset test

The name mapping depends on a rebuild. This variant does not depend on anything but the file.
Every encoder tensor shape occurs **twice** in the model (student + teacher):

| shape | student | teacher | state entries | EMA-only predicts | stepped predicts |
|---|---:|---:|---:|---:|---:|
| `(128, 9, 16, 16)` *(patch conv — unique to the encoder)* | 1 | 1 | **1** | 1 | 2 |
| `(1, 640, 128)` *(positional embedding)* | 1 | 1 | **1** | 1 | 2 |
| `(384, 128)` | 3 | 3 | **3** | 3 | 6 |
| `(128, 512)` | 3 | 3 | **3** | 3 | 6 |
| **all encoder-shaped** | 41 | 41 | **45** | **41** | **82** |
| readout-shaped | 2 | 2 | **2** | 2 | 4 |

*(45 rather than 41 because four generic shapes — `(128,)`, `(512,)` — are also carried by other
modules that DID train.)*

### 2.3 D1 — the PRE-REGISTERED PRIMARY test

| arm | median `r` | min | max | `r < 0.95` | `r > 0.99` |
|---|---:|---:|---:|---:|---:|
| `emao14_30k` | **0.9999981** | 0.9999923 | 1.0000027 | **0** | **43 / 43** |
| `emao14_30k_tauramp` | **0.9999996** | 0.9999913 | 1.0000062 | **0** | **43 / 43** |

⭐ The committed rule said `r > 0.99 ⇒ H_EMA`. **H_GRAD predicted 0.8607.** The measured deviation
from 1.0 is **1.9e-6** against a predicted **0.139** — a factor of **~73,000**.

⚠️ **Sharpening my own prediction against the arm, honestly.** The saved optimizer group records
`lr 0.0`, i.e. the run annealed to zero, so the *average* lr over training was roughly half the
peak and H_GRAD's true floor is nearer **0.93**, not 0.8607. That makes the prediction *weaker*,
and it is still **~37,000×** away from what was measured. Stated because a pre-registered number
quoted without its schedule is the same scope error this programme keeps logging.

### 2.4 D2 and D3

⛔ **D2's literal form DID NOT RUN, and I am not going to present it as though it did.** It needs a
tensor **frozen in the student**; at stage S-W the encoder trains, so **no qualifying tensor
exists** and `n_bit_identical = 0`. What the same measurement *does* show is the teacher–student
deviation: **6.87e-6 – 3.53e-5** (`emao14_30k`) and **1.72e-6 – 3.57e-5** (`tauramp`) — about four
orders of magnitude below H_GRAD's 1.4e-1, and exactly the size an EMA lag should be once the lr
has annealed.

**D3 (corroborating only):** median `cos = 1.0`; median relative distance **1.09e-5** / **5.97e-6**.

### 2.5 ⇒ the verdict, and its scope

**OUTCOME 2 — the teacher moved by EMA only.** Four probes with four different mechanisms — the
optimizer's own state (twice, once name-free), the weight-decay norm signature, and the
teacher/student geometry — and **not one of them is compatible with a gradient-stepped teacher.**

⇒ **The un-freeze is a BUDGET/ACCOUNTING defect.** Concretely, for the two affected arms:

| | value |
|---|---:|
| `config.json` **declared** `n_trainable` | **11,742,915** |
| **effectively trainable** (post-fix, same model) | **9,185,411** |
| overstatement | **2,557,504 = 21.8 %** |

⛔ **Nothing about what those two arms LEARNED changes.** No result is retracted by this turn.

### 2.6 ⛔⛔ BUT "ONE UN-`no_grad` FORWARD AWAY" WAS **NOT** HYPOTHETICAL — THERE IS A REAL CALL SITE

The predecessor wrote that the teacher's safety *"rested on a `no_grad` in one call site rather
than on `requires_grad`"* and that it was **one un-`no_grad` forward away** from training. I went
looking for whether such a forward exists. **It does.**

`v6_chain.py::write_dry_predecessor` builds a stack, applies the stage freeze, and then does

```python
tr = [p for p in stack.parameters() if p.requires_grad]
sum((p * p).sum() for p in tr).backward()
opt.step()
```

— a dummy loss over **every trainable parameter**. It never calls `_EmaCopy.forward`, so the
`no_grad` that made the un-freeze harmless in the trainer **does not protect it at all**.

**MEASURED by reproducing that exact pattern under both freeze rules** (`raw/p1b_unnograd_site.py`):

| freeze rule | teacher `enc` tensors MOVED | teacher `ro` MOVED | **CONTROL — encoder STUDENT moved** |
|---|---:|---:|---:|
| **pre-2026-09-06 (group map alone)** | ⛔ **12 of 17** | ⛔ **2 of 2** | 12 of 17 |
| **current (honours the declaration)** | ⭐ **0 of 17** | ⭐ **0 of 2** | 12 of 17 |

⇒ **the hypothetical was a real, reachable code path in this repo, and the predecessor's fix closes
it.** ⚠️ **Scope it honestly:** `write_dry_predecessor` fabricates a *synthetic predecessor*
`dry_ckpt.pt` for chain testing — it is **not** a training launch, so **no banked training
checkpoint is affected** and §2.5's verdict is unchanged. What it establishes is that the safety
margin was **zero**, not merely thin.

*(Incidental self-check: only 12 of 17 move because the zero-initialised biases have `grad = 2p = 0`
and decoupled weight decay multiplies zero by a constant. That is the same weight-decay mechanism D1
rests on, observed working — a small positive control I did not plan for.)*

---

## 3. P2 — THE DEAD GUARD IS WIRED, AND PROVEN TO FIRE

`assert_frozen_external` was built, pinned in both directions, quantified its own trap at
**86,580,480** parameters — and had **zero callers outside its own test file** (control:
`apply_stage_freeze` has many). That is why the trap it guards happened anyway, on native modules.

### 3.1 What shipped

* **`v6.py: assert_declared_freezes_hold(stack, stage, expect_n_trainable=None)`** — one preflight,
  three directions:
  * **A** frozen-external (delegates to `assert_frozen_external` — *this is the wiring the brief
    asked for*). ⚠️ **LATENT today**: `declare_frozen_external` has zero production callers, so
    nothing can leak yet. It arms the day v7f's `--enc-init-from` lands a foreign backbone.
  * **A'** grad-unreachable — **LIVE**, 3 declarers, **and this is the direction that actually
    failed.** New exception `GradUnreachableViolation`, deliberately distinct from
    `FrozenExternalViolation` because the two make different claims (PROVENANCE vs LOSS GRAPH).
  * **B** nothing-trains — **LIVE**.
* **Wired at BOTH launch paths** (`dry_run` and `train`), immediately after `apply_stage_freeze`,
  via `_declared_freeze_preflight`. The audit is written into **`config.json` and `dry_run.json`**,
  so the run's own artifact records that the declarations were *checked*, not merely *made*.
* **`--expect-n-trainable`** (default `None` = OFF) makes the runbook's exact-count assertion a
  launch precondition — ⛔ A GATE ROW CARRIES ITS ARM, enforced by the launcher instead of retyped
  into a report afterwards.

### 3.2 ⛔ THE MUTATION PROOF (`raw/p2_mutation.json`)

An assertion that passes on both the healthy and the broken model has measured nothing.

| case | v7-tiny | v7f (`PREREG_V7F` §9) |
|---|---|---|
| **HEALTHY, current code** | ✅ **PASSES**, `n_trainable` 9,185,411 | ✅ **PASSES**, `n_trainable` **143,949,315** |
| **MUTATION A'** — reintroduce the pre-2026-09-06 group-map-alone freeze | ⛔ `GradUnreachableViolation` — **49 tensors / 2,557,504** | ⛔ `GradUnreachableViolation` — **157 tensors / 90,960,000** |
| **MUTATION A** — foreign backbone under a trained group | ⛔ `FrozenExternalViolation` — 41 / 972,032 | ⛔ `FrozenExternalViolation` — 149 / **86,138,112** |
| **MUTATION B** — whole model frozen | ⛔ raises, naming the starved groups | ⛔ raises |
| **MUTATION C** — `expect_n_trainable` off by one | ⛔ *"not the arm it claims to be"* | ⛔ same |

⭐ **The mutation reproduces HISTORY, not just a synthetic break.** Under Mutation A' the v7-tiny
arm reads **11,742,915** — *exactly what `v7tiny_emao14_30k`'s banked `config.json` recorded*. And
at v7f it reads **90,960,000** withheld — exactly the predecessor's independently-derived figure.

### 3.3 It survives contact with a real launch

`--dry-run` at v7-tiny geometry with `--o5-target ema`: **EXIT 0**, and
`[v6] declared-freeze preflight OK (dry-run): 0 frozen-external + 3 grad-unreachable subtree(s);
directions A/A'/B checked`. ⭐ **That measurement is why the guard is wired as an unconditional
refusal rather than behind a flag** — the `--refuse-unreached` precedent is that *a flag that always
refuses gets deleted*, and its converse is that a guard which passes every honest launch should not
be opt-in.

### 3.4 The test that was actually missing

`stack/tests/test_declared_freeze_preflight.py` (12 tests) pins the mutations **and** adds
`test_the_TRAINER_ACTUALLY_CALLS_the_preflight_on_BOTH_launch_paths` — asserting the guard is
**wired**, and that it runs *after* the freeze it checks. **A guard's correctness and a guard's
wiring are different claims, and only the first one was ever tested.** That is the durable fix for
the whole failure class.

---

## 4. ⛔⛔ P3 — TWO THIRDS OF THE HORIZONS CLAIM IS REFUTED, AND THE REFUSAL ALREADY EXISTS

The brief said: *"`predictor.py` promises `_refuse_untrained_horizons`; it does not exist (grep 0,
control 1) … either implement the refusal or change the default."*

⭐ **MEASURED BY EXECUTION, NOT BY GREP: the refusal EXISTS AND FIRES.** A fresh launch with
`--horizons 1 2 4` **exits 2**:

> `[v6] ⛔ --horizons (1, 2, 4) declares [2, 4], which NO loss consumes … ⇒ pass --horizons 1 and
> set the HORIZON with --o5-k`

It lives inside **`preflight(a)`**, which `main()` runs before **both** `dry_run(a)` and `train(a)`.
Only the **identifier** `_refuse_untrained_horizons` was ever missing.

| part of the standing claim | verdict | evidence |
|---|---|---|
| *"the preflight DOES NOT EXIST"* | ⛔ **REFUTED** | fires on execution; and present in the **pre-edit baseline** tree (`train_v6_staged.py` md5 `d2ade650…`), so it **predates** the claim |
| *"`trained_horizons` is read by nothing"* | ⛔ **REFUTED** | `v6.py` reads it to decide which heads to declare grad-unreachable — a reader added by the **same turn** that wrote "read by nothing" |
| *"`--horizons` still defaults to `[1, 2, 4]`"* | ✅ **STANDS** | parser default is `[1, 2, 4]` |

⚠️ **This is the `true-but-wrong-for-the-reader` class exactly**: a correct measurement about a
**NAME**, stated as a claim about a **BEHAVIOUR**, and about to be actioned as "implement the
missing guard".

### 4.1 THE DECISION: keep the default, fix the comment, pin the guard — and why

⛔ **The default is deliberately NOT changed.** Changing it would silently change the **geometry**
of every new run that omits the flag, and a model built at `(1,)` cannot strict-load a banked
checkpoint (MEASURED by the predecessor: exactly **6** unexpected keys). The refusal already makes
the wrong default **un-launchable for a fresh arm** — loudly, with the fix in the message — while
**exempting RESUMES**, which is load-bearing: `k60p30k` (MM-E19) runs `--horizons 1 2 4`
*deliberately* as a matched control, and refusing its restart would make a 22 h arm unrecoverable.
**A guard that destroys the work it protects is worse than the defect.**

⇒ shipped: the dangling reference in `predictor.py` is corrected to point at where the check
actually lives, the refuted parts are recorded **in place**, and
`stack/tests/test_horizons_refusal_is_wired.py` (9 tests) pins the refusal, its **non-firing
control**, the **resume exemption**, the **wiring through `main()`**, and the default itself — so
*"does this guard exist"* is answered by running it, never again by grepping a name that was never
the guard's.

⛔ `heads.{2,4}` were **not** deleted. Nothing in this turn touches a banked checkpoint's loadability.

---

## 5. EXPOSURE CLASSIFICATION

Scoped the way the seed-noise rule (`H-ESTIM-SEED-1`) and the predecessor's 82-row sweep were:
⛔ **nothing is retracted on this turn's findings alone.**

| finding | class | rows exposed | what would re-establish / what it changes |
|---|---|---:|---|
| **F1** O5 teacher in the optimizer | **E3 — budget quoted as trained** | **0** | The predecessor's sweep already measured **E3 = 0**: no register row quotes an *effectively trained* count. The two EMA arms' `config.json` `n_trainable` (11,742,915) is an **overstatement of the declared budget by 21.8 %**, and is now corrected in code. **No capability claim rests on it.** |
| **F1** as an *algorithmic* risk | **not exposed** | **0** | The teacher took **0 steps of 30,000**. `uplink="ema"` was an X3-compliant, non-self-generated target as specified. Would be re-opened only by an arm whose `_EmaCopy` is called outside `no_grad` — which the wired A' direction now refuses. |
| **F2** the horizons-guard claim | **documentation / work-item** | **0** | It was a stated gap, never a result. Two thirds refuted; the surviving third (the default) is now a **recorded decision** with a test. |
| predecessor's **E1** `step_readout_op` metric decode | **unchanged — still EXPOSED (3 rows)** | 3 | Untouched by this turn. Still needs an eval-time-fitted readout, exactly as the predecessor specified. |
| predecessor's **E2** dead-head reads | **unchanged — still EXPOSED (4 rows)** | 4 | Untouched. |

⭐ **The load-bearing negative result:** every latent-space claim (rank, participation, collapse,
drift, decodability — the predecessor's E5 = 36 rows) rests on the **student** encoder/readout/
predictor path, which **stepped normally: 41/41 encoder tensors, all at step 30,000**. This turn
adds a *positive* confirmation for those rows rather than a doubt.

---

## 6. THE SUITE, AS A CONTROLLED COMPARISON

Two separate trees, so the control could not be contaminated by editing the tree it ran in.

| | tree | state |
|---|---|---|
| **BASELINE** | `C:\Users\Admin\tanitad-v7fbudget` | HEAD state (predecessor's fix in), untouched by me. ⭐ **Blob-verified against git HEAD before use** — `v6.py` `e9d733d0…`, `train_v6_staged.py` `46c4db1d…`, `predictor.py` `73dac418…`, each with the 40-character shape check that stops a mount outage reading as a match |
| **AFTER** | `C:\Users\Admin\tanitad-emaforensics` | a copy of it plus this turn's edits |

⚠️ Re-checked again after HEAD advanced twice mid-turn (siblings are live): all three files
**still matched**, so no sibling's work is being compared away.

Identical invocation in both, matching the predecessor's:
`cd stack && PYTHONPATH=<tree>/stack OMP_NUM_THREADS=4 PYTHONUTF8=1 python -m pytest -q
-p no:cacheprovider --continue-on-collection-errors`.

| | failed | passed | skipped | xfailed | errors | wall |
|---|---:|---:|---:|---:|---:|---:|
| **BASELINE** | **65** | 6,374 | 121 | 2 | **51** | 868 s |
| **AFTER** | **65** | **6,395** | 121 | 2 | **51** | 860 s |

* ⭐⭐ **FAILURE-ID SET DIFF: EMPTY IN BOTH DIRECTIONS.** `comm -13` (new failures) = **0**
  and `comm -23` (newly fixed) = **0**. **ZERO REGRESSIONS.** `failed` and `errors` are identical
  at **65** and **51**, and the diff is over the *ids*, not the counts, so an equal-count swap
  could not hide in it.
* ⭐ **The `passed` delta is +21, and it is accounted for EXACTLY.** This turn adds
  `test_declared_freeze_preflight.py` (**12** collected: 9 functions, one of them a 4-way
  parametrize) and `test_horizons_refusal_is_wired.py` (**9**: 6 functions, one a 4-way
  parametrize) = **21**. 6,374 + 21 = **6,395**. No existing test was quietly deleted or skipped
  to make the numbers work.
* ⚠️ **The baseline is NOT green, and that is why the verdict is the set diff.** Both trees
  are `stack/`-only copies without most repo-root data, so a large block of environment-dependent
  tests fails or errors in **both** columns and cancels out. Full table: `raw/SUITE_BASELINE_VS_AFTER.md`.

---

## 7. EVIDENCE CLASSES AND ARTIFACTS

Every number above is **MEASURED (ours)**, on CPU. The import origin was re-checked
(`tanitad.__file__` → the local clone, not the G: editable install — the documented trap).
⚠️ The G: mount was flapping throughout: `RESULT.md` failed 12 consecutive content reads while its
metadata resolved, and `git rev-parse` alternated between working and *"not a git repository"*.
**The predecessor's package was therefore recovered from the git OBJECT STORE by blob hash**, not
from the working tree, and every absence claim here was made only after a retry loop on the **same**
target.

| artifact | what it holds |
|---|---|
| `raw/PREREG_DISCRIMINATOR.md` | the discriminator + controls + decision rule, written before any measurement |
| `raw/ema_recon.py` / `.json` | C3/C4 gating controls over all 9 banked arms |
| `raw/ema_forensics.py` / `.json` | C1/C2 controls, D0 (optimizer state), D1, D2, D3, per-tensor |
| `raw/d0_namemap.py` / `.json` | D0 at name level: the rebuild, the 185→185 slot map, the 0-of-43 |
| `raw/p1b_unnograd_site.py` | the REAL un-`no_grad` call site, under both freeze rules, with its control |
| `raw/p2_mutation.py` / `.json` | the guard's mutation proof at both geometries |
| `raw/SUITE_BASELINE_VS_AFTER.md` | the controlled suite comparison |
