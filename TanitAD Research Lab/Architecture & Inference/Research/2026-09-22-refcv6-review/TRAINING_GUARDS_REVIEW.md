# refcv6 adversarial review — TRAINING RECIPE / HYPERPARAMETERS and the GUARD LAYER

**Reviewer:** independent adversarial reviewer (Arch+Inference stream), 2026-09-22/23.
**Repo:** `D:/Projects/TanitAD`, branch `agent/arch-inf-20260803`, session start
`37645fcc61b158a327e6597e64e0abb64dcbc46d`.
**Python:** `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, torch 2.11.0+cu128; `tanitad`
resolved from `D:\Projects\TanitAD\stack\tanitad\__init__.py` — **asserted, not assumed** (the
MSYS-PYTHONPATH false-pass class).
**Compute:** CPU only. The dev-box GPU showed 2,848 MiB used / 33 % util at session start.

⚠️ **THE REPO ADVANCED MID-REVIEW, AND IT IS ACCOUNTED FOR.** At 00:02:50 a sibling landed a
route-gating fix in `stack/scripts/refc_v3_train.py` (gating `ROUTE_WEIGHT * loss_route` on
`no_strategic`, +1 stamp key `route_loss_applied`), which shifted every line below ~3632 by +20.
**All line numbers in this document are re-derived against the CURRENT file**, and every mutation
anchor was re-validated after the edit: **13 of 13 still match exactly one line**, so no result
here is stale. (`raw/anchor_revalidation.txt`.)

**Scope (my lane):** the training recipe and hyper-parameters, and the guard layer. Trunk/input,
diffusion, tactical/nav and perception/data belong to four siblings.

**Authority:** `Project Steering/SPEC_REFCV6_V2.md` §2, §3(F6), §8, §9, §10.2, §10.7, §11, §12;
`Project Steering/PREREG_REFCV6_V2.md`; `Project Steering/ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md`
classes E, F, G.

---

## VERDICT AT A GLANCE

| # | question | answer |
|---|---|---|
| 1 | Is the recipe implemented? | **4 of 5 yes. The encoder lr ×0.5 holds for ZERO optimiser steps, and `--opt dd` — which carries the whole §2 row — is opt-in with nothing opting in.** (F-1) |
| 2 | Loss weights / non-zero gradient | **10 live, 0 dead on the reachable arm. 6 of 8 refcv6 weights REFUSE without their seam — the dead-weight gate layer is the strongest thing in this file.** 1 arm INCONCLUSIVE, 5 NOT ESTABLISHED. (§2) |
| 3 | Gradient clip = 100 at every site | **No — 2 of 4. The GRPO post-train library still defaults to 1.0, and nothing names it.** (F-2) |
| 4 | Provenance closure | `assert_seams_are_built` **refuses 7/7** constructed disagreements — and **6 further falsifications pass SILENTLY**, including the one field that says which of §10.2's two runs happened. `assert_knobs_stamped` is `assert f(x) == f(x)`; **16 of 42 knobs can vanish from `config.json` green.** (F-3, F-4) |
| 5 | Guards that cannot go red | **13 arms · 4 CAUGHT · 9 NOT CAUGHT.** Zeroing every learning rate passes **176** tests. Every one of the five hyper-parameter-surface mutations survived. (F-5) |
| 6 | Batch / Thor | **The Thor batch-8 rule does not apply — refcv6 targets a pod.** But the prereg holds a batch constant **without stating its value**, and the activation figures it would be sized from **carry no batch scope**. (§6) |

---

## F-1 ⛔⛔ THE `encoder lr ×0.5` IS BUILT CORRECTLY AND THEN DESTROYED AT EVERY TRAINING STEP

**Evidence class: MEASURED** — `raw/recipe_probe.json`, instrument `code/probe_recipe.py`.

SPEC §2 (still binding under §10.2's trunk swap): *"optimiser | **AdamW**, weight decay 1e-4,
**encoder lr ×0.5** of the heads, warm-up then cosine"*.

`build_optimizer` (`stack/scripts/refc_v3_train.py:5678`) gets it right: `:5715-5719` calls
`param_groups_dd`, which returns two groups. MEASURED at build on the real `RefCV3Model` with the
timm resnet34 trunk:

```
group encoder  lr=5.000000e-05 wd=0.0001 tensors=108 params=21,284,672
group head     lr=1.000000e-04 wd=0.0001 tensors=117 params=141,159
AT BUILD  encoder/head lr ratio = 0.5000      <- SPEC literal 0.5
```

The train loop then overwrites both. **`stack/scripts/refc_v3_train.py:7458-7459`**:

```python
for g in opt.param_groups:
    g["lr"] = args.lr * sched(step)
```

It writes `args.lr * sched(step)` — **not** `g["initial_lr"] * sched(step)` — into *every* group.
`initial_lr` occurs nowhere in the file (MEASURED: 0 hits in 9,058 lines), and there is exactly
**one** `g["lr"] =` assignment in the trainer. MEASURED by replaying that loop — recovered from the
trainer's source **by AST** and `exec`-ed verbatim, so it is the live code and not a restatement:

| step | sched | encoder lr | head lr | ratio |
|---|---|---|---|---|
| 0 | 0.000500 | 5.000000e-08 | 5.000000e-08 | **1.0000** |
| 1 | 0.001000 | 1.000000e-07 | 1.000000e-07 | **1.0000** |
| 1999 | 1.000000 | 1.000000e-04 | 1.000000e-04 | **1.0000** |
| 2000 | 1.000000 | 1.000000e-04 | 1.000000e-04 | **1.0000** |
| 15000 | 0.555982 | 5.559822e-05 | 5.559822e-05 | **1.0000** |
| 29999 | 0.000000 | 3.147195e-13 | 3.147195e-13 | **1.0000** |

⇒ **the ×0.5 holds for exactly zero optimiser steps.** 21.28 M encoder parameters — 99.3 % of the
model's trainable weight by count — train at the head rate for the whole run. At `resnet101`
(§10.2's PRIMARY arm, 42.5 M) the exposure is roughly twice as large.

⭐ **The docstring names the defect as the thing that does not happen.**
`refc_v3_train.py:5697-5700`:

> *"⛔ Warm-up + cosine is unchanged and lives in the `sched` lambda at the call site. It is a
> MULTIPLIER on each group's own `lr`, so the 0.5x survives the schedule instead of being
> overwritten by it — which is the silent way a paramwise config stops meaning anything."*

A correct statement of the requirement and a false statement about the code six pages away.
Nothing crashes; the run looks normal; `config.json` stamps `encoder_lr_mult: 0.5` (`:4784`) and
the launch line prints `encoder_lr=5.000e-05`. **The record asserts a rate the run never used.**

### Why no guard caught it — advisory class F, object swapped

`stack/tests/test_refcv6_trunk.py:371-387`
(`test_param_groups_carry_the_INTENDED_learning_rates`) asserts the ratio **at construction** and
stops — `assert [pg["lr"] for pg in opt.param_groups] == [3e-4, 6e-4]` — and never advances a step.
The guard tests the **constructor**; the defect lives in the **consumer**. That is the advisory's
*"every instrument built its inputs FROM the config it was testing … structurally incapable of
noticing that a CONSUMER disagrees"*, with config→optimiser.

⭐ **The programme already owns the correct form, in another trainer.**
`stack/tests/test_dinov3_seed.py:645-659` drives the scheduler across the warm-up window and
asserts *both* group lrs at every step:

```python
for step in range(warm + 5):
    seen.append((opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]))
    sched.step()
```

An independently authored in-repo reference. refcv6's guard simply lacks it.

### Two aggravations, both MEASURED

* **`--opt dd` is opt-in and NOTHING opts in.** `refc_v3_train.py:8766` — `default="adam"`. A
  scoped search over `stack/` and `Project Steering/` (`*.py *.sh *.md`) finds `--opt dd` **only in
  the flag's own help text and docstrings** — no launch script, no preflight, no test of the train
  loop. With the default, the arm runs plain `Adam`, **one group, weight decay 0.0**, so *none* of
  SPEC §2's optimiser row is in effect, not merely the ×0.5. MEASURED: `--opt adam` →
  `class = Adam, group g0 lr=1.0e-04 wd=0.0`.
* **No gate asserts the recipe is on.** `stack/scripts/refcv5_preflight.py` (759 lines) contains no
  check of optimiser, weight decay, encoder lr, warm-up or clip, and
  `stack/tanitad/train/prelaunch_v2.py:164-171` covers hypotheses / split overlap / tuning leak
  only. **SPEC §8's pre-pod gate is satisfiable with the entire §2 recipe off.**

**What is NOT wrong:** AdamW ✅, weight decay 1e-4 on both groups ✅, warm-up ✅ (0.0005 → 1.0 over
2,000 steps), cosine ✅ (1.0 → 3.1e-13 at 30 k) — *conditional on `--opt dd` being passed*.

### The fix, and the guard that must accompany it

```python
for g in opt.param_groups:
    g["lr"] = g.setdefault("initial_lr", g["lr"]) * sched(step)
```

⚠️ `setdefault` is what makes it resume-safe; a plain `g["initial_lr"]` read would `KeyError` on a
checkpoint restored from before the key existed. The guard must be **step-driven** — the build-time
assertion stays green either way, which is how this survived.

---

## §2 LOSS WEIGHTS — the enumeration, and what actually produces a gradient

**Evidence class: MEASURED** — `raw/loss_reach_base.json`, `raw/loss_reach_gp.json`, instrument
`code/probe_loss_reach.py`. Method taken from `refc_v3_train.assert_ground_prior_is_supervised`
(`:5059-5157`), the instrument that MEASURED `D-RC5-GROUND-DEAD`'s 1.164e-10.

### 2.1 The complete weight inventory

**Always-on module constants** (`refc_v3_train.py:129-158` plus the shared ones from
`refc_train.py`), assembled at `:3647-3653`: `TRAJ_WEIGHT`, `ANCHOR_CLS_WEIGHT`, `LAW_WEIGHT`,
`ROUTE_WEIGHT` (as of 00:02:50 gated on `no_strategic`), `LAT_WEIGHT/2`, `LON_WEIGHT/2`,
`GOAL_TAC_WEIGHT = 0.5` (`:129`), `GOAL_STR_WEIGHT = ROUTE_WEIGHT` (`:132`),
`SEL_V3_WEIGHT = 1.0` (`:135`).

**Twelve gated `--w-*/--agent-*` knobs, every one defaulting to 0.0** (`REFC_WEIGHT_GATES`,
`:1737`; MEASURED against the live parser):

| dest | flag | default | term |
|---|---|---|---|
| `w_u0` | `--w-u0` | 0.0 | u0 control-space x0 loss (WP-4) — **SPEC §3 F6 requires 0** |
| `w_agent` | `--w-agent` | 0.0 | GT-supervised detection set loss (WP-6) |
| `w_bev_aux` | `--w-bev-aux` | 0.0 | WP-D BEV auxiliary |
| `w_tac_goal` | `--w-tac-goal` | 0.0 | 22-token tactical goal set (BCE) |
| `agent_w_project` | `--agent-w-project` | 0.0 | agent projection consistency |
| `agent_w_ground` | `--agent-w-ground` | 0.0 | agent ground prior — **registered DEAD (`D-RC5-GROUND-DEAD`)** |
| `goal_point_w` | `--goal-point-w` | 0.0 | E15 geometric goal point |
| `w_map` | `--w-map` | 0.0 | refcv6 SAM3 BEV map soft-CE (§6) |
| `w_box3d` | `--w-box3d` | 0.0 | refcv6 3-D cuboid set loss (§6) |
| `w_tac_v6` | `--w-tac-v6` | 0.0 | refcv6 §4 behaviour decoder |
| `w_r7_wta` | `--w-r7-wta` | 0.0 | refcv7 WTA proposal decoder |
| `w_r7_scorer` | `--w-r7-scorer` | 0.0 | refcv7 disentangled scorer |

### 2.2 ⭐ THE STRONGEST THING IN THIS FILE — the dead-weight gate layer works

MEASURED: **6 of the 8** refcv6 weights I drove REFUSE at startup when their seam is absent, each
naming the M18 defect class. Verbatim:

* `--w-map 1.0` → *"needs `--trunk timm`. refcv6 perception reads the STRIDE-16 map, and the legacy
  REF-C ResNetEncoder exposes only stride 32 — `fmap_s16` would be None"*
* `--w-agent 1.0` → *"With no agent seam … every one of these parses, is STAMPED INTO config.json,
  and trains nothing — the run record would state a configuration that did not happen"*
* `--w-bev-aux 0.1` → *"No head is built, so `bev_logits` never appears in `out` and the term is
  SILENTLY SKIPPED while the weight is stamped"*
* `--w-tac-goal`, `--w-tac-v6`, `--agent-w-ground` likewise.

This is exactly the class the advisory says nothing else catches. It is exhaustive over the parser
(`tests/test_v6_effective_weights.py` fails if a new `--w-*` ships ungated) and called from **both**
launch paths (`:1944-1961`, pinned by `test_refc_guard_is_on_the_real_path`). It should be cited as
the model for the other arms, and arm **A11** below shows it goes red under mutation.

### 2.3 The gradient measurement, and its honest limit

On the reachable arm (`--smoke --arm hier`, synthetic episodes, 153 trainable tensors), with the
same-breath positive control `traj` at `2.737562e+03` over 81/153 tensors:

| term | value | grad_abs_sum | reaches |
|---|---|---|---|
| `law` | 60.64 | 2.362636e+04 | 85/153 |
| `cls` | 33.07 | 5.073353e+03 | 115/153 |
| `sel_v3` | 10.56 | 3.726746e+03 | 116/153 |
| `traj` (CONTROL) | 15.14 | 2.737562e+03 | 81/153 |
| `goal_tac` | 15.36 | 1.488857e+03 | 66/153 |
| `lat` / `lon` | 1.12 / 1.13 | 7.851e+02 / 7.048e+02 | 40/153 |
| `lat_tac` / `lon_tac` | 0.95 / 1.03 | 6.522e+02 / 6.545e+02 | 66/153 |

**10 live, 0 dead** at a 1e-9 floor. ⇒ **I found no second `D-RC5-GROUND-DEAD` among the terms I
could reach.**

⚠️ **One arm is INCONCLUSIVE and must not be read as a finding.** With
`--goal-point-w 1.0 --goal-point-inject`, `goal_point` reads **0.0, grad 0.0** — but the same dict
reports **`gp_label_rows = 0`**. The term is zero because **my synthetic corpus carries no
goal-point label**, which is indistinguishable from a dead term. `goal_point`'s gradient reach is
**NOT ESTABLISHED**; one forward on a real batch with `gp_label_rows > 0` settles it.

⭐ **That INCONCLUSIVE reading is itself a finding about the guard layer.** The arm launched with
`gp_label_rows = 0`. The trainer has the right instrument one seam over: `_lan_arm_preflight`
(`:4515`, the check at `:4574`) explicitly refuses when `any_valid_frac <= 0` *"because a `goal_str` computed over an
all-invalid label is 0.0 and looks like a pass"*. **The goal-point seam has no equivalent
live-label refusal.** Whether real-corpus coverage is non-zero is NOT ESTABLISHED here; the cheap
durable fix is to log `gp_label_rows` at step 0 and refuse 0 — the same refusal `--w-agent` already
gets, one level in.

⛔ **Five of the twelve knobs could not be driven on this box** (`w_map`/`w_box3d` need the timm
trunk plus SAM3/cuboid targets; `w_agent`/`agent_w_project` need the agent join; `w_tac_v6` and
`w_tac_goal` need `--v7-labels`). Their gradient reach is **NOT ESTABLISHED by me**. Each is
settled by a single `compute_losses_v3` forward on the 139-clip eval cache with the seam on — work
SPEC §10.6 already requires before any pod hour.

---

## F-2 ⛔ SPEC §10.7's `grad clip = 100` REACHED 2 OF 4 SITES

**Evidence class: MEASURED** — census `raw/clip_sites.json`, instrument `code/probe_clip_sites.py`
(AST over **1,101** `.py` files under `stack/`, 0 unreadable, with a same-breath control that aborts
the run INVALID if the known site in `refc_v3_train.py` fails to appear).

SPEC §10.7: *"RL gradient clipping: **100, not 1.0** … MEASURED on the banked 600-step logs of all
three arms: a 1.0 max-norm would bind on **600/600 steps**."*

**42 `clip_grad_norm_` call sites** under `stack/`; **7** supply a grad-clip default. The four that
matter:

| site | value | amended? |
|---|---|---|
| `stack/tanitad/rl/ddv2_il.py:104` `GRAD_CLIP = 100.0` | **100.0** | ✅ |
| `stack/tanitad/rl/ddv2_il.py:123` `IlSettings.grad_clip = GRAD_CLIP` | **100.0** | ✅ |
| `stack/tanitad/rl/config.py:225` `PostTrainConfig.grad_clip: float = 1.0` | **1.0** | ⛔ **no** |
| `stack/scripts/rl_refcv3_min.py:988` hardcoded literal | **1.0** | ⛔ **no** |

MEASURED live, not read from source:

```
L.GRAD_CLIP                    = 100.0
L.IlSettings().grad_clip       = 100.0
PostTrainConfig().grad_clip    = 1.0
make_cfg-shaped cfg .grad_clip = 1.0      <- ratio 100.0x
```

The last row is load-bearing. `stack/scripts/rl_refcv3_min.py:765-791` (`make_cfg`) builds the
`PostTrainConfig` for every GRPO arm and **never names `grad_clip`** — MEASURED: the string occurs
**0 times** in `rl_refcv3_min.py` and **0 times** in `rl_pilot_refc21.py`. The dataclass default
therefore stands and `stack/tanitad/rl/posttrain.py:454-455` applies it. Arm **A10** below confirms
the guard layer is blind to it: flipping that default to `0.0` — which makes
`if cfg.grad_clip:` falsy and **disables clipping entirely** — passes 70 tests.

⚠️ **Scoped honestly.** §10.7's 600/600 was measured on the **ddv2** arms
(`PREREG_DDV2_RL_VALIDATION.md` §11, AMENDMENT A-1). Whether 1.0 also binds every step on the GRPO
library is **NOT ESTABLISHED** — different objective, `freeze_trunk=True`, a different trainable
set. What IS established: the number the programme MEASURED to be an every-step 17–62× rescale on a
sibling RL trainer is still the unexamined default here, and the amendment written to remove it did
not reach it. **What settles it** is the measurement §10.7 already ran — log the pre-clip grad norm
for ~100 GRPO steps and count how many exceed 1.0.

### And the IL clip is an unmeasured, unstamped magic literal

`stack/scripts/refc_v3_train.py:7569`: `clip_grad_norm_(model.parameters(), 10.0)` — hardcoded, no
flag, and (MEASURED by reading `_seam_stamp`, `:4669-4840`) **not written into `config.json`**. The
refcv6 IL arm's clip is therefore not derivable from the run record, not settable without an edit,
and never measured against the gradients it will meet — the work §10.7 did for the RL side. Arm
**A6** shows the consequence: setting it to `1e-8`, which annihilates every gradient, passes 111
tests.

---

## F-3 ⛔ `assert_seams_are_built` REFUSES 7/7 REAL DISAGREEMENTS — AND IS SILENT ON THE FIELD THAT NAMES WHICH §10.2 ARM RAN

**Evidence class: MEASURED** — `raw/seam_probe.json`, instrument `code/probe_seams.py`. Method:
build a real model, take the real `_seam_stamp(cfg, args)`, falsify **one key**, call
`assert_seams_are_built` (`:5211`). Every arm asserts it actually changed the value, and a clean
stamp is re-checked after the run to prove no arm contaminated the shared object.

**Half A — the guard works, and works well.** 7 of 7 constructed record/model disagreements REFUSE,
each naming the right seam:

| arm | verdict | the refusal |
|---|---|---|
| `sampler: 'none' → 'ddim'` | **REFUSED** | *"decoder.control_head is None — the record would claim a denoiser the weights do not contain"* |
| `w_u0: 0.0 → 1.0` | **REFUSED** | *"no control_head for that loss to supervise"* |
| `trunk: 'refc' → 'timm'` | **REFUSED** | *"core.encoder is a ResNetEncoder"* |
| `cross_agent: False → True` | **REFUSED** | *"decoder layers [0, 1] have no agent attention"* |
| `agents: None → {...}` | **REFUSED** | *"core.agent_head is None"* |
| `ego_history: None → {enable: True}` | **REFUSED** | *"core.ego_hist is None"* |
| `goal_point: None → {inject: True}` | **REFUSED** | *"model.gp_head is None"* |

Bidirectional, model-vs-record (fact vs intent), and it does catch the 2026-09-06 defect it was
written for. This is a good guard.

**Half B — six falsifications pass silently.** Each proven to change the stamped value; the trunk
ones constructed on a real timm resnet34 with a RANDOM init, which is the arm they describe:

| arm | clean → falsified | verdict | why it matters |
|---|---|---|---|
| `trunk_name` | `'resnet34'` → `'resnet101.a1_in1k'` | ⛔ **SILENT** | **SPEC §10.2's only variable.** resnet101 is the PRIMARY arm, resnet34 the comparison. The record is the one thing that says which ran. |
| `trunk_pretrained` | `False` → `True` | ⛔ **SILENT** | the registered **ImageNet KNOCKOUT** arm. If the record can claim a prior the weights lack, *"the prior helps"* is unfalsifiable. |
| `trunk_in_channels` | `3` → `9` | ⛔ **SILENT** | §10.3's K-frame history is read off `in_channels`; the record would name a different input stack. |
| `opt` | `'adam'` → `'dd'` | ⛔ **SILENT** | SPEC §2's optimiser — see F-1. |
| `encoder_lr_mult` | `0.5` → `2.0` | ⛔ **SILENT** | the multiplier the run reports. |
| `weight_decay` | `1e-4` → `1e-2` | ⛔ **SILENT** | SPEC §2's wd — and under `--opt adam` the real wd is 0.0 while the stamp reads 1e-4 either way. |

⭐ **The last three are checkable and simply are not checked.** `opt = build_optimizer(model, args)`
is at `refc_v3_train.py:7069`; `assert_seams_are_built(model, _seams)` is at `:7126`. **The
optimiser object exists 57 lines before the guard runs**, so asserting `stamp["opt"]`,
`stamp["weight_decay"]` and `stamp["encoder_lr_mult"]` against `opt.param_groups` is a four-line
addition — and it would have caught F-1 at startup, in the record, on day one.

⚠️ **This probe's first run was itself a class-F failure, and I report it rather than quietly fix
it.** Run 1 scored `trunk_imagenet_norm=True`, `encoder_lr_mult=0.5` and `weight_decay=1e-4` as
"SILENT" while **the clean stamp already held those values** — three INERT arms reported as
findings. That is the advisory's *"a control that cannot come out the other way is not a control"*,
committed inside the instrument written to hunt it. The rerun asserts `new != clean` and **aborts
the run as INVALID** otherwise; the three arms were re-authored with values that genuinely differ,
and one (`trunk_imagenet_norm`) turned out to be **unconstructible from argv at all** — there is no
`--trunk-imagenet-norm` flag, so it is reported as NOT RUN rather than as silent.

---

## F-4 ⛔ `assert_knobs_stamped` IS `assert f(x) == f(x)` — AND 16 OF 42 KNOBS CAN VANISH FROM `config.json` GREEN

**Evidence class: MEASURED** — mutation arms **A7 / A7b / A8**, `raw/mutation_results.json`,
`raw/mutation_logs/`; live dest census below.

The guard's docstring claims *"every `--agent-*`/`--w-*` knob reaches `config.json`"*. Its two sides:

* `_seam_stamp:4961` writes `"agent_knobs": agent_knob_stamp(args)`
* `assert_knobs_stamped:5520` computes `want = agent_knob_stamp(args, parser)`

**The same function, called twice.** Advisory class F shape 1 verbatim: *"the expected value is an
expression over the code under test; `assert x == f(x)` measures determinism, not correctness."* It
catches a stamp that DROPS the block or a key (arm **A8: CAUGHT**, 2 failed — the control works).
It is structurally blind to a defect in `agent_knob_dests`'s **derivation** (`:5010`), because both
sides inherit it.

The only thing plugging that hole is a separate test with **literal** expectations —
`stack/tests/test_refc_v3_agent_provenance.py:383-394`:

```python
assert len(dests) >= 15
for must in ("w_agent", "w_u0", "agent_w_project", "agent_w_ground", "agent_queries"):
    assert must in dests, must
```

Five names out of **42**. MEASURED against the live parser:

* dropping `--w-` from the derivation → **CAUGHT** (A7: 2 failed) — the five literals bite.
* dropping `--bev-aux` and `--wp-index` → **NOT CAUGHT** (A7b: **85 passed, 1 skipped**).
  MEASURED: that mutation silently removes **16 of 42 dests** — `bev_aux`, `bev_aux_detach`,
  `bev_aux_dtok`, `bev_aux_hidden`, **`bev_aux_occlusion`**, `bev_aux_pos_weight`, `bev_aux_rmax`,
  `bev_aux_rng`, **`bev_aux_shuffle`**, `wp_index`, `wp_index_const_xy`, `wp_index_detach`,
  `wp_index_hidden`, **`wp_index_mode`**, `wp_index_radius_m`, `wp_index_scale_m` — leaving 26,
  which clears the `>= 15` floor with room to spare.

⭐ **Those are precisely the fields the code itself calls load-bearing.** `agent_knob_dests`'s own
docstring (`:5023-5033`) says the 2026-09-07 widening exists because *"`--bev-aux-occlusion` is
precisely what separates the pre-registered arm from its DELIBERATE-REGRESSION twin … A record that
cannot say which of the two ran makes the panel unfalsifiable."* And `assert_seams_are_built`
(`:5339-5343`) says a wrong `wp_index.mode` means *"a CONTROL arm would be reported as the
treatment, or the reverse."*

**This is advisory class F shape 5** — *an allow-list that excuses an absence nobody re-reads* — in
must-list form: the literal list was written before the widening and was never widened with it. The
fix is to derive the expectation from a **different consumer** (assert the dest set against the
parser's option strings, or an independently written registry) rather than from `agent_knob_stamp`
itself, and to raise the floor to the real count.

---

## F-5 ⛔⛔ THE MAIN EVENT — ZEROING EVERY LEARNING RATE PASSES 176 TESTS

**Evidence class: MEASURED** — `raw/mutation_results.json`, `raw/mutation_logs/*.log`,
`raw/mutation_console.txt`, instrument `code/mutation_harness.py`.

**Harness discipline** (it audits itself against the same advisory): whole-line anchors matched by
**byte equality** on a CRLF tree; an anchor matching 0 or >1 lines **aborts the run as INVALID**,
never a skip; the mutation is applied to the **source the guard reads**, never to a test; every arm
runs a **baseline that must be GREEN first**; subprocesses decoded `encoding="utf-8",
errors="replace"`; the verdict is read from **pytest's own summary line** (the artifact), not from
an exit code, and a run with no summary is **INCONCLUSIVE**; every file is restored and
**sha256-verified** after each arm. The tree under mutation is a **COPY** at
`C:\Users\Admin\AppData\Local\Temp\mt` — the harness refuses any `--tree` containing `Projects`, so
**the repo was never mutated** (proven below).

| arm | the real regression | tests | verdict |
|---|---|---|---|
| **A1** `param_groups_dd` drops `encoder_lr_mult` | the ×0.5 never built | 40 | **CAUGHT** (1 failed) |
| **A2** `g["lr"] = 0.0` for every group | **the model cannot learn at all** | **176** | ⛔ **NOT CAUGHT** |
| **A3** `g["lr"] = args.lr` | schedule computed and discarded — no warm-up, no cosine | 92 | ⛔ **NOT CAUGHT** |
| **A4** `--weight-decay` `1e-4 → 1e-2` | torch's default — the trap the flag's own help names | 60 | ⛔ **NOT CAUGHT** |
| **A5** `--encoder-lr-mult` `0.5 → 1.0` | SPEC §2's ×0.5 off at the launch default | 60 | ⛔ **NOT CAUGHT** |
| **A6** IL clip `10.0 → 1e-8` | every gradient annihilated | 111 | ⛔ **NOT CAUGHT** |
| **A7** `--w-` dropped from the knob derivation | the M18 defect | 107 | **CAUGHT** (2 failed) |
| **A7b** `--bev-aux`/`--wp-index` dropped | the M18 defect, 2026-09-07 half | 86 | ⛔ **NOT CAUGHT** |
| **A8** `agent_knobs` stamped as `{}` | control for A7 | 85 | **CAUGHT** (2 failed) |
| **A9** `LAT_WEIGHT/2 → LAT_WEIGHT` | restores the DOUBLE tactical budget refcv3 shipped with | 105 | ⛔ **NOT CAUGHT** |
| **A10** `PostTrainConfig.grad_clip 1.0 → 0.0` | GRPO clipping **disabled entirely** (`if cfg.grad_clip:` is falsy) | 70 | ⛔ **NOT CAUGHT** |
| **A11** `U0_WEIGHT_DEFAULT 0.0 → 1.0` | flips SPEC §3 **F6** by a module constant | 124 | **CAUGHT** (7 failed) |
| **A12** `--warmup` `2000 → 0` | no warm-up at all; the ImageNet prior meets a full-rate step 0 | 92 | ⛔ **NOT CAUGHT** |

**13 arms · 13 valid (every anchor unique) · 4 CAUGHT · 9 NOT CAUGHT.**

⛔ **A2 is the headline.** `g["lr"] = 0.0` freezes **every parameter of the refcv6 model for the
entire run** — a 30 k-step arm whose checkpoint is bit-identical to its initialisation — and **176
tests across six of the most refcv6-relevant files pass** (`test_refcv6_trunk`, `test_refc_v3`,
`test_refcv3_arm`, `test_refcv6_diffusion`, `test_refcv6_perception_training`,
`test_refcv6_tactical_training`). ⇒ **No guard in the refcv6 suite reads the learning rate the
optimiser actually steps with.** F-1 is the instance; this is the property.

⭐ **The pattern is sharper than any single arm: every mutation to the trainer's HYPERPARAMETER
SURFACE survived — A2, A3, A4, A5, A6, A9, A10, A12, eight for eight.** Everything CAUGHT (A1, A7,
A8, A11) is *structural*: is the group built, is the dest derived, is the block stamped, is the
default OFF. **The suite guards structure extremely well and guards the numbers that structure is
trained with not at all.** That is one gap, not nine, and one file closes most of it.

### The cheapest guard that closes A2–A6, A9 and A12

One test, driven over steps, asserting **literals**:

```python
# with --opt dd, replaying the trainer's own lr application
assert [g["lr"] for g in opt.param_groups] == [approx(5e-5), approx(1e-4)]       # at peak
assert opt.param_groups[0]["lr"] / opt.param_groups[1]["lr"] == approx(0.5)      # EVERY step
assert opt.param_groups[0]["weight_decay"] == 1e-4
assert sched(0) < sched(args.warmup - 1) == 1.0 > sched(args.steps - 1)
assert CLIP_MAX_NORM == 10.0 and LAT_WEIGHT / 2 * 4 == MANEUVER_WEIGHT
```

⛔ The expectations must be **literals**, not `args.lr * sched(step)` — that would be arm A2's own
expression and would go green against it.

---

## §6 BATCH SIZE / THROUGHPUT ON THOR

**Evidence class: MEASURED (source) + NOT ESTABLISHED (the number).**

⭐ **The Thor batch-8 rule does not apply to refcv6, and quoting it here would be a scope error of
exactly the kind this programme logs.** That finding is about *Thor's 20 SMs* saturating at batch 8.
refcv6 targets a **pod**: `PREREG_REFCV6_V2.md` §3 is titled *"The pipeline is validated on the 139
eval clips BEFORE any pod hour"*, and SPEC §12.5 records that **`resnet101` OOMs on the 8 GB dev
box at BOTH geometries, including `416 × 1024` at batch 1**. Thor is not named as a refcv6 host
anywhere I found (two probes: SPEC and PREREG).

What I can report:

* **The trainer default is `--batch 20`** (`refc_v3_train.py:8441`). MEASURED.
* **`PREREG_REFCV6_V2.md` §1 lists "the batch size" among the things held constant across arms A
  and B — and never states its value.** A constant that is asserted but not named cannot be checked
  by a reader or by a gate, and `prelaunch_v2` does not check it
  (`stack/tanitad/train/prelaunch_v2.py:164-171` covers hypotheses / splits / tuning only).
* ⚠️ **"Held constant" may be unachievable, and it is worth naming now.** Arm A is `resnet101` at
  1024/2048 channels (42.5 M); arm B is `resnet34` at 256/512 (21.3 M) — a 4× channel change at
  both strides. If A must drop its batch to fit and B does not, the prereg's "everything else
  identical" clause breaks and A-vs-B acquires a second variable. The prereg already declares
  A-vs-B *"a RECIPE comparison, not a controlled single-variable ablation"*, so this is not a
  contradiction — but the batch is currently the undeclared half of that admission.
* ⛔ **The activation figures carry no batch scope.** MEASURED: `1033.5` / `1662.9` occur in exactly
  two files, `Project Steering/SPEC_REFCV6_V2.md` §11 R1 and `Project Steering/PI_DECISION_QUEUE.md:602`
  — *"resnet101 K=3 activations | 645.9 MB | 1033.5 MB | **1662.9 MB (1.609x)**"* — and **neither
  names a batch size**. Under the programme's own units rule that makes them **inadmissible for a
  batch decision**. (They scale as the token counts 640/1024/1664 do, which is *consistent with*
  per-sample, but that is an inference, not the artifact's statement.)
* ⛔ **I measured no throughput or memory on any device.** CPU-only session.
* ⛔ **If anyone measures on Thor: only in-process `torch.cuda.max_memory_allocated()` is
  admissible there.** `mem_get_info`, `free`, `tegrastats` and `VmRSS` all lie on unified memory,
  in both directions.

**What would settle it:** one forward+backward at `416 × 1024` on the pod class actually
provisioned, reading `torch.cuda.max_memory_allocated()`, for `resnet101` and `resnet34` at the
candidate batch — then **write the number into the prereg**, since it is already declared a
held-constant.

⚠️ **A method note against myself.** My first pass at this search reported *"`1033.5` → 0 files"*
and I nearly wrote it up as an absence. It was a **timed-out grep**: the target queries ran over
`Project Steering` **plus** the 500 MB research tree with a 100 s cap, while the "control" ran over
`Project Steering` alone. **A control that runs a cheaper query than the target is not a control.**
Re-scoped to one identical scope, the strings are there, in two files. *(CLAUDE.md: "0 hits is a
claim about the SEARCH, not about the CONTENT.")*

---

## WHAT I COULD NOT ANSWER

1. **Gradient reach for `w_map`, `w_box3d`, `w_agent`, `agent_w_project`, `w_tac_v6`,
   `w_tac_goal`.** Their gates (correctly) refuse without SAM3 maps / cuboid targets / the agent
   join / `--v7-labels`, none of which are on this box. **NOT ESTABLISHED.** Settled by one
   `compute_losses_v3` forward per seam on the 139-clip eval cache — work SPEC §10.6 already
   requires before any pod hour.
2. **`goal_point`'s gradient reach** — INCONCLUSIVE, `gp_label_rows = 0` on synthetic episodes.
   Settled by one forward on a batch with a live label.
3. **Whether a 1.0 max-norm actually binds on GRPO gradients.** §10.7's 600/600 was ddv2. **NOT
   ESTABLISHED** for the GRPO library. Settled by logging pre-clip norms for ~100 steps.
4. **Any throughput or memory number, on any device.** CPU-only session; the dev-box GPU was busy
   and is the wrong device anyway (§12.5: resnet101 OOMs there at batch 1).
5. **`trunk_imagenet_norm` falsification** — there is no `--trunk-imagenet-norm` flag, so the arm
   could not be constructed from argv. Whether `assert_seams_are_built` would catch a record
   claiming normalisation it did not apply (advisory class A) is **NOT ESTABLISHED**; it needs a
   model built with the flag off, which means adding the flag first.
6. **Whether the sibling's 00:02:50 route-gating edit introduces anything in my lane.** I
   re-validated my anchors against it (13/13) but did not review the change itself — it is a
   tactical/nav-lane edit.

---

## DELIVERABLE MANIFEST

All paths under
`D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-refcv6-review/`.
⛔ **STAGED IN THE WORKING TREE. NOT COMMITTED. NOT PUSHED.**

| artifact | what it is |
|---|---|
| `TRAINING_GUARDS_REVIEW.md` | this document |
| `code/probe_recipe.py` | Q1 — builds the real optimiser; replays the trainer's own schedule recovered by AST |
| `code/probe_clip_sites.py` | Q3 — AST census of every `clip_grad_norm_` site and grad-clip default under `stack/` |
| `code/probe_seams.py` | Q4 — 13 constructed record/model disagreements vs `assert_seams_are_built`, each proven non-inert |
| `code/probe_loss_reach.py` | Q2 — per-term parameter-gradient reach with a same-breath positive control |
| `code/mutation_harness.py` | Q5 — the 13-arm mutation harness (CRLF whole-line anchors, INVALID on non-unique match, sha256 restore verification, refuses a repo `--tree`) |
| `raw/recipe_probe.json` | Q1 measurements |
| `raw/clip_sites.json` | Q3 census (1,101 files, 0 unreadable) |
| `raw/seam_probe.json` | Q4 measurements |
| `raw/loss_reach_base.json`, `raw/loss_reach_gp.json` | Q2 measurements |
| `raw/mutation_results.json` (18,124 B), `raw/mutation_logs/*.log` (26 files), `raw/mutation_console.txt` | Q5 per-arm baseline + mutated pytest output |
| `raw/anchor_revalidation.txt` | the 13/13 re-validation against the post-00:02:50 repo |

**Scratch (outside the repo, safe to delete):** `C:\Users\Admin\AppData\Local\Temp\mt` — the copied
`stack/` + `taniteval/` + `tools/` tree the harness operated on.

**Repo source files changed by me: NONE.** Verified by byte comparison against the pre-mutation
snapshot: `stack/tanitad/models/timm_trunk.py` **MATCH** (`d01415e1…`) and
`stack/tanitad/rl/config.py` **MATCH** (`3665b46f…`). `stack/scripts/refc_v3_train.py` DIFFERS — a
39-line diff that is the sibling's route-gating fix at 00:02:50 (`_route_on` gate + the
`route_loss_applied` stamp key), none of it at any anchor I touched.
