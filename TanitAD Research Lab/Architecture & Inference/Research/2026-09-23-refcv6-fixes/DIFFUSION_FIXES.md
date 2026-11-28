# refcv6 DIFFUSION DECODER — THE FIVE MEASURED DEFECTS, FIXED · PROVEN · EVALUATED

**Implementer:** senior implementer, dimension = *diffusion decoder*. **Date:** 2026-09-23.
**Repo:** `D:/Projects/TanitAD` (`agent/arch-inf-20260803`). **Compute:** CPU, dev box.
**Input:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-refcv6-review/`
(`DIFFUSION_PAPER_REVIEW.md` + `raw/`). **PI order:** *"Fix all findings and evaluate them."*

**Primary source for every conformance claim:** `…/2026-09-05-diffusiondrive-v2-analysis/raw/
ddv2_src/v1/transfuser_model_v2.py`, `blocks.py`, `multimodal_loss.py` — the **released
DiffusionDrive source**, quoted by line. No number below is cited to a summary.

---

## 0. VERDICT IN ONE TABLE

| # | finding | fixed | mutation | changes an existing arm? |
|---|---|---|---|---|
| **1** | coupling (1) BUILT and NEVER CALLED — 3 breaks | ✅ **all three**, 0 → **6** module calls | **M1 · M2 · M10 · M11 all RED** | ⛔ **NO** — bit-identical, max Δ **0.0** |
| **2** | the ranked score is BLIND to the emitted trajectory | ✅ telemetry lie fixed + **opt-in refusal** | **M3 · M4 · M9 RED** | ⛔ **NO** (tensors 0.0; one stamp value corrected on an F5 arm) |
| **3** | every v0 guard tests the DECLARATION, not the tensor | ✅ **both** guards now read `anchor_controls` | **M5 · M5b · M6 RED** | ⚠️ **only a provably degenerate build**, which now refuses |
| **4** | the bit-identity baseline rotted to a tautology | ✅ pinned to `cbadba5…` (`8c7d215^`) | **M7 RED** | ⛔ **NO** (test-only) |
| **5** | F8 clamps ONCE, outside the ladder | ✅ clamps **inside**, DD `:519` | **M8 RED** | ⚠️ **only the F8 arm, which has never been run** |

**Mutation harness: 12 / 12 arms CAUGHT** (`code/mutation_harness.py` → `raw/mutation_results.json`).
**`stack/tests/test_refcv6_diffusion.py`: 34 → 60 tests, all passing.**
**Covering subset (50 test files that import `refc` / the decoder / the chain): 1,137 passed,
2 skipped, 0 failed, 285.10 s** — `raw/covering_subset_after.txt`, re-run after the LAST edit.

⛔ **THE BINDING ANSWER FIRST, because it is the one that can cost a retrain.** MEASURED over
**64 windows at a pinned inference seed**, BEFORE = the `git HEAD` blob of BOTH `refc.py` and
`refcv6_diffusion.py` (the fixes are **staged, never committed**, so `HEAD` *is* the pre-fix
source), AFTER = the worktree, loaded as a **coherent pair** (`code/eval_fixes.py` →
`raw/eval_fixes.json`):

| arm | `state_dict` keys | weight mismatches | output tensors | `sel_tele` |
|---|---|---|---|---|
| **all nine flags off** | identical | **0** | **BIT_IDENTICAL**, max Δ **0.0** | **identical** |
| **the validated 2026-09-17 arm** (`f1 f2 f3 f4 f5_focal f6 f9`) | identical | **0** | **BIT_IDENTICAL**, max Δ **0.0** | **+2 stamp keys only** |
| `--f5-emitting-conf` on | identical | **0** | **BIT_IDENTICAL**, max Δ **0.0** | **+2 keys, and `sampler_ranks_the_fan` false → TRUE** |

⇒ **NO FIX CHANGES WHAT ANY TRAINED ARM COMPUTES.** The only differences reaching a banked arm
are two additive provenance keys (`f5_refuse_blind_rank`, `f8_clamp_in_ladder`) and one stamp
value that was **false while the code it describes was true** (§2). Every behaviour change is
behind a flag that is **default OFF** or in a configuration that is **provably meaningless**.

---

## 1. FINDING 1 — coupling (1) was built and never called

### 1.1 The fix, at all three breaks

**Break A — the sampler never received it (`refc.py`).** `_sample`'s `bev` is its **ninth**
parameter and the call passed **eight** positionals ending at `agent_pos`, so `_decode_ctrl`'s
correct-looking `bev` forward always forwarded `None`. **AST census of every
`_decode` / `_decode_ctrl` / `_sample` call site: 2 of 7 carried `bev` → 7 of 7.** The default
classifier `_decode`, the prefilter `k >= n` branch, the pre-v5 refine loop and the
`--sel-score-emitted` pass are all now fed.

⚠️ The refine loop is fed with `bev=` **only**. The documented agent asymmetry there (*"AGENT-FREE
AT HEAD, AND LEFT THAT WAY ON PURPOSE"*) is untouched — coupling (1) is a different seam, and its
address (`x_in`) moves on every pass of that loop too.

**Break B — no caller ever supplied the map.** `RefCModel.forward` declared `bev` and forwarded
it, and **no call site in `stack/` ever passed it**: `refc_v3.py`'s two `self.core(...)` calls pass
`scene_hook` / `bev_hook` / `bev_tokens`. Closed **inside `refc.py`**, where the §4 perception hook
already runs: when a build carries the coupling, the branch's **dense** `bev_feats`
(`refcv6_perception_branch.py:434`, `[B, C, X, Y]` from `bev_encoder.py:263`) becomes `bev`.

⚠️ **`bev_tokens` is NOT that map.** It is the §4 tactical seam, a `[B, T, d]` **sequence**. One
branch, two tensors, two consumers — and feeding either to the other's consumer is a shape crash at
best. Two suppliers for one tensor is **refused**, the same refusal `bev_hook` vs `bev_tokens`
already carries.

**Break C — no config route.** `DecoderConfig.bev_coupling` is assigned in exactly one file in the
repository, and `refc_v3_train.py` has no `--bev-coupling` argument. ⛔ **NOT MINE TO FIX** — the
exact patch is in §7.

**And the guard that makes it self-enforcing.** A coupling that is **built**, counted in
`param_breakdown`, stamped by `bev_coupling_provenance()` and then handed **no map** is exactly the
state the review measured — and from the outside it is indistinguishable from a coupling that helps
nothing. `RefCModel.forward` now **refuses** it before the compute, so no arm can ever publish
*"the BEV coupling does not help"* while never having had a BEV map.

### 1.2 PROVEN

`stack/tests/test_refcv6_diffusion.py`, **12** new tests (the table groups the last three). Every expectation is a **literal derived
from the architecture** (2 layers × (1 classifier + 2 denoise passes) = **6**), never an
expression over the code:

| test | kind | expectation |
|---|---|---|
| `…REACHES_the_sampler_and_the_classifier` | fails without the fix | fires **== 6**; ctx shapes `wp [B,N,4,2]`, `bev [2,6,120,64]` |
| `…fires_on_the_CLASSIFIER_ONLY_pass` | fails without the fix | fires **== 6** at `steps=0` |
| `…CONTROL_no_bev_means_no_call` | **control, known value** | fires **== 0** |
| `…CONTROL_direct_call_fires_exactly_once` | **control** | fires **== 1** — what the 9 existing tests measure, green throughout the defect |
| `…is_REMOVABLE_at_the_zero_init_gate` | removability | `traj` **bit-identical** |
| `…is_GATED_not_DEAD` | gated ≠ dead | `u0_hat` **and** `traj` both move |
| `…BUILT_coupling_with_NO_map_is_REFUSED…` | the self-enforcing guard | raises |
| `…map_REACHES_the_decoder_THROUGH_a_full_RefCModel…` | consumer-side | fires **== 6** |
| `BREAK_B…FEEDS_the_coupling` / `…TWO_suppliers` / `…is_SCOPED…` | Break B + controls | 6 · raises · Δ **0.0** |

⭐ **THE MUTATION HARNESS CAUGHT MY OWN GUARD BEING TOO WEAK, AND THAT IS THE MOST USEFUL THING
IT DID.** The first `…is_GATED_not_DEAD` watched only `traj`, and with the historical defect
restored it **STAYED GREEN** — because the CLASSIFIER pass still reads the map, its confidences
move, the argmax moves, and a *different candidate* is emitted. `traj` moved for a reason that has
nothing to do with the sampler. `u0_hat` is produced **only** by `_sample`, so it is the assertion
that can tell them apart. *A guard my own regression arm could not turn red was not a guard.*

⚠️ **THE INSTRUMENT ARTIFACT, restated because it produced two wrong readings in the review and one
here:** `control_head` and every `CascadeHeads.control_head` are **zero-init**, so at construction
the emitted trajectory is independent of every decoder weight. Any liveness probe that mutates an
upstream module and watches `traj` reads **0.0 on a perfectly wired model**. Every arm below carries
**both** states.

### 1.3 EVALUATED — `raw/eval_fixes.json`

| quantity | before | after |
|---|---|---|
| `BEVWaypointSampler.forward` calls, decoder, `steps=2` | **0** | **6** |
| … `steps=0` (classifier-only) | **0** | **6** |
| … through a full `RefCModel` with the §4 hook | **0** *(and it RAN)* | **6** |
| a BUILT coupling with NO map | **RAN silently, 0 fires** | **REFUSED** |
| **zero-init gate**: `traj` max abs Δ | — | **0.000000000 m** |
| **zero-init gate**: `u0_hat` max abs Δ | — | **0.000000000** |
| **gate OPENED** (emitting head un-zeroed, gate → 1.0): `traj` | — | **60.786 m** |
| **gate OPENED**: `u0_hat` | — | **81.527** |
| **gate OPENED**: `sel_idx` changed | — | **17 / 64 windows** |
| params (d=32, d_bev=6, 2 layers) | 2,826 built | 2,826 **used** |
| **CONTROL** — a no-coupling arm fed the same hook | Δ **0.0** | Δ **0.0** |

⇒ **The coupling is now LIVE, is REMOVABLE at its gate, and costs the banked baseline nothing.**
⛔ **Whether it HELPS is still NOT ESTABLISHED** — it has never trained. Settling it needs a paired
arm **with a replicate and an inference-seed repeat**, because this planner samples.

---

## 2. FINDING 2 — the ranked score was blind to the emitted trajectory

### 2.1 What was fixed, and what was deliberately NOT

⛔ **THE DEFAULT WAS NOT FLIPPED.** `SelectionConfig.refined` / `.score_emitted` stay `False`.
Flipping them would change what every banked arm computes, which the binding rule forbids, and
*"improve the geometry, keep the ranking"* is a legitimate arm the decoder already documents. What
was fixed is that the defect was **invisible and the telemetry actively denied it**:

1. ⛔ **`sampler_ranks_the_fan` was a LIE on an F5 arm.** It read `bool(self.sel.refined)` alone,
   while the ranked surface is chosen by `base = refined if (sel.refined or rv6.f5_emitting_conf)
   else conf`. A reader tells which arm they are looking at from this key; a key that contradicts
   the line it claims to describe is worse than no key. Now `bool(sel.refined or f5_emitting_conf)`.
2. ⭐ **A new OPT-IN refusal, `DiffusionFlags.f5_refuse_blind_rank` (default `False`).** On a
   sampler build whose ranked surface is the classifier, it refuses **before the compute**, naming
   `--f5-emitting-conf` (DD-faithful) and `--sel-refined` as the two ways out.
3. **The defect is pinned as a MEASUREMENT** so it can never be "discovered" again.

**PUBLISHED (primary):** DD gathers the emitted trajectory by the argmax of the **same** pass's
classification head — `mode_idx = poses_cls.argmax(-1); best_reg = gather(poses_reg, mode_idx)`
(`transfuser_model_v2.py:554-557`). **F5 ON is DD-faithful; F5 OFF is not.**

### 2.2 PROVEN — mutations M3 (drop the F5 term) · M4 (restore the lying stamp) · M9 (disarm the refusal), all RED

### 2.3 EVALUATED — perturb ONLY the sampler's own head, 64 windows, pinned seed

| arm | `traj` moved | `sel_score` moved | `sel_idx` changed | stamp |
|---|---|---|---|---|
| **DEFAULT** (no F5, no `--sel-refined`) | **68.377 m** | **exactly 0.000000** | **0 / 64** | `false` ✅ honest |
| **`--f5-emitting-conf`** | **88.295 m** | **3.613712** | **38 / 64** | `true` ✅ **was `false`** |

The stamp, all four corners, before vs after:

| | before | after |
|---|---|---|
| `f5=False, sel.refined=False` | false | false |
| `f5=False, sel.refined=True` | true | true |
| **`f5=True, sel.refined=False`** | ⛔ **false** | ✅ **true** |
| `f5=True, sel.refined=True` | true | true |

⇒ **The default is still blind, ON PURPOSE and now honestly stamped; the arm that closes it is one
flag away and refusable.** ⛔ **Whether F5-ON improves ADE is NOT ESTABLISHED** — a ranking that
moves is not a ranking that improves. Same settlement as §1.3: a paired arm, a replicate, an
inference-seed repeat.

⚠️ **ESCALATION (§7, item 2):** the 2026-09-17 pipeline-validation arm ran `f5_emitting_conf
FALSE`. Either every future refcv6 arm carries `--f5-emitting-conf`, or that arm's row in
`MODEL_REGISTRY.md` is documented as **anchor-bank-ranked**.

---

## 3. FINDING 3 — every v0 guard tested the declaration, not the tensor

### 3.1 The fix

Two guards, both now reading `anchor_controls`:

* `refc.py`'s WP-4 preflight gains a **tensor-level twin** of its own message — which already
  said *"a fixed-path bank carries `anchor_controls` of all zeros"* and **nothing read that
  tensor**. LATCHED (`_v0_controls_nonzero`): `anchor_controls` is a non-trained buffer, so one
  host sync settles it for the model's life.
* `assert_f9_vocabulary(n_anchors, v0_conditioned, expect_n, **anchor_controls=None**)` — the
  signature had **three declarations and no tensor**, so its third message was unenforceable by
  construction. Handed the tensor **only on a sampler build** (the "anchored Gaussian centred on
  'do nothing'" claim is about the sampler), so a classifier build with F9 on is unchanged.

The expectation is the literal `0.0`, and it is an **identity, not a threshold**: all-zero controls
roll every anchor to the *same* straight line.

### 3.2 PROVEN — M5 (decoder guard) · M6 (F9 assert) · M5b (both) all RED

⭐ M5 alone left `test_F9_reaches_the_tensor_THROUGH_THE_FORWARD` **green**, and the harness
scored the arm **NOT CAUGHT** — *the harness caught MY error*. That is **defence in depth, not
redundancy**: both guards must go to restore the full defect (M5b). The `expect_red` was corrected
and the reason recorded in the arm's note.

### 3.3 EVALUATED

| quantity | value |
|---|---|
| bank spread across anchors, **real controls** | **7.523445 m** |
| bank spread across anchors, **all-zero controls** | **exactly 0.000000000 m** |
| forward on a degenerate bank, **before** | **RAN SILENTLY** |
| `assert_f9_vocabulary` on zeros, **before** | **PASSED** |
| forward on a degenerate bank, **after** | **REFUSED** |
| `assert_f9_vocabulary` on zeros, **after** | **RAISED** |

### 3.4 ⚠️ THIS ONE DOES CHANGE BEHAVIOUR — say exactly where

A build that **declares** `v0_conditioned=True` and carries **all-zero** `anchor_controls` now
**refuses** instead of running. That state is reachable from argv as `--sampler ddim
--anchor-v0-conditioned` **without** `--anchors`, where the trainer prints a warning and does not
refuse. It is **provably meaningless** — every candidate is the same line — so the change is from
*a plausible-looking wrong experiment* to *a crash*. No arm with a real vocabulary is affected.

**Three test fixtures were hit and are ALSO fixed, because they were degenerate too:**

| file | test | what it was |
|---|---|---|
| `stack/tests/test_refcv6_diffusion.py` | `test_all_flags_off_is_BIT_IDENTICAL_on_64_windows` | compared two **degenerate** fans — valid but blind to how anchors are rolled |
| `stack/tests/test_refc_sampler.py` | `test_sampler_groups_gt_one_REFUSES…` | wanted F7's refusal, got the vocabulary's |
| `stack/tests/test_refc_sampler.py` | `test_metre_space_arm_is_REACHABLE` | the metre arm is about the sampler SPACE, so it needs a real vocabulary to be about anything |

⚠️ **`test_refc_sampler.py` IS OUTSIDE MY DECLARED OWNERSHIP.** Two fixtures gained
`anchor_controls.copy_(<the same ladder `_v0_decoder` uses>)` and a comment naming the reason. The
file was **clean in `git status`** at the time (no sibling editing it) and the alternative was to
leave the suite red. Flagged here rather than left to be found in an audit.

---

## 4. FINDING 4 — the bit-identity guard had rotted to a tautology

### 4.1 The fix

`_pre_refcv6_module()` materialised its baseline with `git show HEAD:…/refc.py` — the **same blob
as the worktree**. Now pinned:

```
_BASELINE_REV   = "cbadba5844a2db70a968172ab0b0bbe3a0140af0"   # == 8c7d215^
_BASELINE_CHILD = "8c7d215a18631f836c93d0f39b26f4ce79d70646"
```

`8c7d215` is the commit that introduced `refcv6_diffusion` into `refc.py` (2026-09-16,
`git log --reverse -S"refcv6_diffusion"`). **The full hash is written out** because `8c7d215^` is a
rev-expression over a short hash and must not silently re-resolve — the test asserts
`rev-parse <child>^ == _BASELINE_REV` with a **40-character shape check** on the result.

Four things are now asserted before the comparison runs, any of which **ABORTS as INVALID**:

* the baseline blob reads **non-empty** (`_git_blob` never returns an empty bytestring as content —
  an empty result from this mount is indistinguishable from a failed query);
* `refcv6` mentions in the baseline **== 1** (literal), in HEAD **>= 50** (floor);
* the same-breath **CONTROL** `class AnchoredDiffusionDecoder` reads **1 in BOTH** blobs, so a
  count of 0 can never be a claim about a failed read;
* `baseline != HEAD`.

⛔ A missing baseline **FAILS**; only a missing `git` binary skips. A skipped bit-identity test
reports as *"not failed"*.

### 4.2 PROVEN

* **M7** — point `_BASELINE_REV` back at `"HEAD"` → **RED**.
* `test_bitidentity_baseline_MUTATION_pointing_at_HEAD_goes_RED` — the same regression, in-process.
* `test_missing_baseline_ABORTS_rather_than_SKIPPING`.

### 4.3 EVALUATED

| | HEAD | `cbadba5` (`8c7d215^`) |
|---|---|---|
| `refcv6` mentions in `refc.py` | **83** | **1** |
| **CONTROL** `class AnchoredDiffusionDecoder` | **1** | **1** |
| blobs equal? | **no** |

⭐ **AND THE CLAIM SURVIVES THE REAL BASELINE.**
`test_all_flags_off_is_BIT_IDENTICAL_on_64_windows` now runs against `cbadba5` **and passes** —
identical `state_dict` keys, zero weight mismatches, zero tensor diffs over 64 windows, identical
`sel_tele`. *The guard was inert; the claim was true.* And it is now a **stronger** claim: the
fixture carries real `anchor_controls`, so it compares a real fan instead of two identical lines.

---

## 5. FINDING 5 — F8 clamped once, outside the ladder

### 5.1 The fix

DD clamps at the **top of every test-loop iteration** — `x_boxes = torch.clamp(img, min=-1, max=1)`
(`transfuser_model_v2.py:519`) — and denormalises the **clamped** state. Ours clamped once, above
the loop, and denormalised whatever `sched.step` produced. The clamp now runs at the top of every
ladder pass.

⭐ **The pre-loop clamp STAYS**, and it is a different statement: it is DD's **training** clamp
(`:474`), which fires once after `add_noise` before the single decoder call. `clamp` is
**idempotent**, so pass 0 is bit-identical either way and only passes ≥ 1 change.

⭐ **AND THE NEGATIVE HALF, stated so nobody "fixes" it later: DD does NOT clamp what it EMITS.**
`best_reg` is gathered from `poses_reg` straight out of the decoder (`:545, 554-556`); only the
sample fed back into `scheduler.step` is boxed. Our `u0_hat = denorm(x0_hat_n)` is the analogue of
`poses_reg` and is correctly left free. The probe separates the two **structurally** — a denorm is a
LADDER denorm iff a `_decode_ctrl` pass follows it — never by index, because a `[:-1]` slice would
silently re-pool them the day an F3 cascade adds per-stage exports inside the loop.

### 5.2 PROVEN

**M8 → RED.** ⚠️ **This anchor is why whole-line EQUALITY matters:** the identical text
`if rv6.f8_flat_waypoint_noise:` occurs at **8 spaces** (the training clamp, which stays) and at
**12 spaces** (the in-ladder clamp, which is the fix). A substring anchor hits the wrong one.

The test carries a **discriminating control**: the *pre-clamp* state must EXCEED the box, or
"≤ 1.0" is satisfied by a sample that was never outside it.

### 5.3 EVALUATED — `--sampler-space metre --f8-flat-noise`, ladder `[10, 0]`

| | pass 0 | pass 1 | `sched.step` max (pre-clamp) | outside DD's box |
|---|---|---|---|---|
| **before** | 0.980661 | **64.155** | 64.155 | **[64.155]** |
| **after** | 0.980661 | **1.000000** | 64.155 | **[]** |

⚠️ **Scope it honestly:** the 64.15 was produced with the emitting head deliberately un-zeroed at
σ 0.5, so it demonstrates the **mechanism**, not the size on a trained checkpoint. What is not
scope-dependent is the source: **DD clamps inside the loop and we did not.**

### 5.4 Blast radius

Only the **F8 arm**, and **F8 has never been run** (`…/2026-09-22-refcv6-review` §1.1: the
validated arm stamps `f8_flat_waypoint_noise FALSE`). Nothing banked is affected. There is no
second knob: two clamp semantics under one flag name is how an arm becomes unattributable. The
stamp records `"f8_clamp_in_ladder": true` so any future artifact says which semantics produced it.

---

## 6. WHAT I DID NOT DO, AND WHY

1. ⛔ **I did not flip `SelectionConfig.refined` / `.score_emitted`.** That changes every banked
   arm's computation; the binding rule makes it opt-in, and the flag exists.
2. ⛔ **I did not touch `stack/scripts/refc_v3_train.py`** — the parent owns it. Patches in §7.
3. ⛔ **I did not touch `stack/tanitad/refs/refc_v3.py`** — staged-modified by a sibling. Break B
   was closable inside `refc.py` instead, which is strictly better: the §4 hook already runs there
   and `bev_feats` exists at that instant.
4. ⚠️ **I DID touch two files outside my declared ownership**, both forced, both flagged:
   * `stack/tests/test_refc_sampler.py` — two degenerate fixtures (§3.4);
   * `stack/tanitad/rl/ddv2_refc_chain.py` — `capture_sampler_inputs`'s hook had **fixed arity 8**
     and my ninth `_sample` argument turned it into a `TypeError` three tests deep (8 tests red).
     It now accepts and forwards `bev`, and **REFUSES** a non-`None` one for the same reason it
     refuses agents: refcv5-v2 was trained with no BEV coupling, and a chain that silently dropped
     it on a coupling-built model would be a different model.
5. **Findings 4 and 5 of the review (`--f7-ack-eval-join`'s absent consumer; F7 staying refused)
   are NOT in this package** — the consumer is `taniteval/tools/t1_eval.py`, another agent's file.
   The review's escalation stands.
6. ⚠️ **The pre-existing failures I did NOT cause.** `stack/tests/test_refcv6_tactical.py`'s six
   `tflip`/`tzero`/`obedience`/`acceptance` tests fail **when that file is run alone** with
   `ModuleNotFoundError: No module named 'taniteval.refcv6_acceptance'`, and pass in the 50-file
   subset (another file's import puts `taniteval` on `sys.path`). **POSITIVE ASSERTION:** the
   traceback mentions `refc.py` / `refcv6_diffusion.py` **0 times**, while the same-breath control
   reads `refcv6_acceptance` **12 times** — so the 0 is a genuine absence, not a failed read.
   Evidence: `raw/preexisting_tflip_failures.txt`. (`test_refcv6_perception.py`'s
   `box3d_set_loss` failure, seen earlier in this session, was a sibling's in-flight edit and
   cleared at 00:30 when they landed their fix.)

---

## 7. ⛔ ESCALATIONS — exact patches for `stack/scripts/refc_v3_train.py` (the parent applies these)

None of these is optional if refcv6 is to run the fixed decoder. **Without Patch C/D the BEV
coupling cannot be built by any training run at all**, which is Break C.

### Patch 1 — route the new guard flag (`refcv6_flags_from_args`, ~`:5668`)

**BEFORE**
```python
        f5_emitting_conf=bool(getattr(args, "f5_emitting_conf", False)),
```
**AFTER**
```python
        f5_emitting_conf=bool(getattr(args, "f5_emitting_conf", False)),
        f5_refuse_blind_rank=bool(getattr(args, "f5_refuse_blind_rank",
                                          False)),
```

### Patch 2 — its argparse entry (group `g6`, immediately before `--f5-focal`, ~`:8818`)

**BEFORE**
```python
    g6.add_argument("--f5-focal", action="store_true",
```
**AFTER**
```python
    g6.add_argument("--f5-refuse-blind-rank", action="store_true",
                    help="F5 GUARD, not the feature. Refuse a sampler build "
                         "whose RANKED score is the CLASSIFIER surface -- "
                         "MEASURED 2026-09-22, a sampler-only perturbation "
                         "moves `traj` 12.20 m and `sel_score` EXACTLY 0.0, "
                         "i.e. the ranking judges the PROPOSAL, not the path. "
                         "Satisfied by --f5-emitting-conf (DD-faithful) or "
                         "--sel-refined. Default OFF: an arm that wants "
                         "'improve the geometry, keep the ranking' is "
                         "legitimate and must say so.")
    g6.add_argument("--f5-focal", action="store_true",
```

### Patch 3 — the `--bev-coupling` route (argparse, group `g6`)

**AFTER** (append to `g6`)
```python
    g6.add_argument("--bev-coupling", action="store_true",
                    help="refcv6 coupling (1): the BEV map read AT THE "
                         "CANDIDATE'S OWN WAYPOINTS, in every decoder layer "
                         "and on every denoising pass (DiffusionDrive "
                         "`blocks.py:88-108`), behind a zero-init gate. ⛔ "
                         "Requires --w-map > 0: the dense `bev_feats` it "
                         "reads come from the MAP BRANCH.")
    g6.add_argument("--bev-coupling-learned-offsets", action="store_true",
                    help="OUR EXTENSION, off by default. Each (query, "
                         "waypoint) also predicts a bounded metric offset. "
                         "The released DD samples AT the waypoints; the "
                         "provenance stamp reports which build ran.")
    g6.add_argument("--bev-coupling-offset-max-m", type=float, default=2.0,
                    help="bound on that offset, metres (tanh-scaled).")
```

### Patch 4 — the assignment (beside the WP-B block, after the `--wp-index` `else:` branch ends, ~`:1180`)

Needs one new import beside `_perc` (`:123`):
```python
from tanitad.models import refc_bev_coupling as _bevc  # noqa: E402
```

**AFTER** (new block)
```python
    # ---- refcv6 coupling (1) -- BREAK C, the missing config route --------- #
    # ⛔ `DecoderConfig.bev_coupling` was assigned in exactly ONE file in the
    # repository (`integration/verify_patch.py`), so NO TRAINING RUN COULD
    # BUILD THE SAMPLER AT ALL. `refc.py:3755` reads it and calls
    # `attach_bev_coupling`; only `enable` / `learned_offsets` / `offset_max_m`
    # are read from this object -- `d_model` and `n_points` are taken from the
    # decoder itself, which is why they are not set here.
    if bool(getattr(args, "bev_coupling", False)):
        if float(getattr(args, "w_map", 0.0) or 0.0) <= 0.0:
            raise SystemExit(
                "[v3] ⛔ --bev-coupling with --w-map 0. Coupling (1) reads the "
                "DENSE `bev_feats` the MAP BRANCH produces; with no map branch "
                "the sampler would be BUILT, counted in param_breakdown, "
                "STAMPED into config.json by `bev_coupling_provenance()` and "
                "handed no map -- the exact 0-forward-hook-fires state the "
                "2026-09-22 review MEASURED, and an arm in it would publish "
                "'the BEV coupling does not help' while never having had a "
                "BEV map. `refc.py` refuses it at the first batch; this "
                "refuses it before the GPU.")
        core.decoder.bev_coupling = _bevc.BEVCouplingConfig(
            enable=True,
            learned_offsets=bool(
                getattr(args, "bev_coupling_learned_offsets", False)),
            offset_max_m=float(
                getattr(args, "bev_coupling_offset_max_m", 2.0)))
        # ⛔ NOT A LITERAL: `d_bev` is the BEV ENCODER's `d_out` (96 today),
        # downstream of the stride-16 map whose own width is 1024 on
        # resnet101 and 256 on resnet34. Read it from the SAME config object
        # the branch will be built from, never written down.
        core.decoder.bev_coupling_d_bev = int(
            _perc.PerceptionBranchConfig(
                w_map=float(getattr(args, "w_map", 0.0) or 0.0),
                w_box3d=float(getattr(args, "w_box3d", 0.0) or 0.0)
            ).bev_cfg.d_out)
        print("[v3] refcv6 coupling (1) ON: d_bev=%d learned_offsets=%s "
              "offset_max_m=%.3g -> %d decoder layers"
              % (core.decoder.bev_coupling_d_bev,
                 core.decoder.bev_coupling.learned_offsets,
                 core.decoder.bev_coupling.offset_max_m,
                 int(core.decoder.layers)), flush=True)
```

### Patch 5 — stamp it into `config.json` (~`:4746`, beside `"wp_index"`)

**BEFORE**
```python
        "wp_index": (core.decoder.wp_index.as_dict()
                     if getattr(core.decoder, "wp_index", None) is not None
                     else None),
```
**AFTER**
```python
        "wp_index": (core.decoder.wp_index.as_dict()
                     if getattr(core.decoder, "wp_index", None) is not None
                     else None),
        # ⭐ refcv6 coupling (1), STRUCTURALLY -- and `provenance` in
        # particular, because an arm with `learned_offsets` on that reports
        # itself as "DiffusionDrive's coupling" is a MISLABELLED arm, not a
        # better one. `enabled: false` when the seam is off, so the two states
        # are distinguishable in the record and not merely by absence.
        "bev_coupling": model.core.decoder.bev_coupling_provenance(),
```
*(If `model` is not in scope at that site, `core.decoder.bev_coupling_provenance()` is the same
object — `core` is `model.core`.)*

### Patch 6 — not a patch, a DECISION

⛔ **Either every future refcv6 arm carries `--f5-emitting-conf`, or the 2026-09-17 arm's row in
`MODEL_REGISTRY.md` records that its ranking was ANCHOR-BANK-ONLY.** MEASURED: on that arm a
sampler-only perturbation moves `sel_score` by **exactly 0.0**.

⚠️ Also still open from the review and **not mine**: `refc_bev_coupling.py:58` cites
`tests/test_refcv6_bev_coupling.py`, which does not exist — the tests are in
`stack/tests/test_refcv6_perception.py:394-518` (plus the **12** consumer-side tests added here).

---

## 7bis. ⚠️ THIS DIRECTORY IS SHARED — DO NOT CONFLATE THE TWO PACKAGES

A **perception sibling** is banking into the same folder in the same hours. Theirs:
`PERCEPTION_FIXES.md`, `code/mutate_perception_fixes.py`, `code/eval_map_cells_recovered.py`,
`code/eval_box_supervision_recovery.py`, `code/refc_v3_train.PATCH.md`,
`raw/mutation_perception_fixes.json`, `raw/map_cells_recovered.json`,
`raw/box_supervision_recovery.json`. Mine are listed in §8.

⛔ **`code/refc_v3_train.PATCH.md` IS THEIRS, NOT MINE.** My trainer patches are **§7 of this
file** and nowhere else — I did not write a second `PATCH.md`, precisely so nothing overwrites
theirs.

⭐ **The two patch sets do NOT collide, checked rather than assumed.** Their argparse hunk anchors
on `--map-gt-root` (~`:8575`); mine on `--f5-focal` (~`:8818`) and an append to the same `g6`
group, which is order-independent. Their model-attach hunk is at the perception build (~`:6280`);
my assignment is at config time beside the WP-B block (~`:1180`) and my stamp at ~`:4746`. No
shared line.

---

## 8. DELIVERABLE MANIFEST

All paths are in the repo working tree at `D:/Projects/TanitAD`, **staged, NOT committed, NOT
pushed**. Nothing is on a pod, in a worktree, or only in this agent's context.

### Source changed (repo)

| path | what |
|---|---|
| `stack/tanitad/refs/refc.py` | findings 1, 2, 3, 5 — the `bev` arity at all 7 call sites, the Break-B feed, the built-coupling-with-no-map refusal, the honest `sampler_ranks_the_fan`, the `f5_refuse_blind_rank` refusal, the tensor-level v0 refusal, the in-ladder F8 clamp |
| `stack/tanitad/models/refcv6_diffusion.py` | `DiffusionFlags.f5_refuse_blind_rank`; `assert_f9_vocabulary(..., anchor_controls=None)`; stamp keys `f5_refuse_blind_rank` + `f8_clamp_in_ladder` |
| `stack/tests/test_refcv6_diffusion.py` | 34 → **60** tests: the pinned baseline (`_BASELINE_REV`, `_git_blob`) + 26 new guards, controls and in-process mutations |
| ⚠️ `stack/tests/test_refc_sampler.py` | **outside my ownership** — two degenerate fixtures given real `anchor_controls` (§3.4) |
| ⚠️ `stack/tanitad/rl/ddv2_refc_chain.py` | **outside my ownership** — `capture_sampler_inputs`'s fixed-arity hook accepts, forwards and refuses `bev` (§6.4) |

### Research package

| path | what |
|---|---|
| `…/Research/2026-09-23-refcv6-fixes/DIFFUSION_FIXES.md` | this report |
| `…/2026-09-23-refcv6-fixes/code/mutation_harness.py` | **12 deliberate-regression arms on a COPIED tree; whole-line EQUALITY anchors; an anchor matching ≠ 1 line ABORTS; baseline must be green; verdict read from pytest's summary artifact; restores sha256-verified** |
| `…/code/eval_fixes.py` | the BEFORE/AFTER evaluation: `HEAD` blobs of **both** files loaded as a coherent pair, 64 windows, pinned seed, every arm in gate-closed **and** gate-opened states |
| `…/raw/mutation_results.json` | **12/12 CAUGHT**, per-arm red sets |
| `…/raw/eval_fixes.json` | every number in §§1.3, 2.3, 3.3, 4.3, 5.3 + the binding table |
| `…/raw/covering_subset_after.txt` | the 50-file covering subset: **1,137 passed / 2 skipped / 0 failed** |
| `…/raw/preexisting_tflip_failures.txt` | the six sibling-owned failures, with the same-breath control |

### Commands that reproduce it

```bash
PYTHONPATH=D:/Projects/TanitAD/stack <py> -m pytest stack/tests/test_refcv6_diffusion.py -q
#   -> 60 passed

cd "TanitAD Research Lab/Architecture & Inference/Research/2026-09-23-refcv6-fixes/code"
<py> mutation_harness.py --json ../raw/mutation_results.json
#   -> 12/12 arms CAUGHT

PYTHONPATH=D:/Projects/TanitAD/stack <py> eval_fixes.py --json ../raw/eval_fixes.json
```
