# F-7 (manoeuvre contrastives) and F-8 (temporal consistency) — BUILT

**2026-08-18** · branch `agent/arch-inf-20260803` · base HEAD `6784455` ·
evidence class per claim below · dev-box RTX 4060 / CPU only, **no training run, Thor untouched**

---

## 0. Headline — and the two things that need a decision

**Both cells are built, measured, wired and pinned. The default build is UNCHANGED at
87,893,449 params / 405 keys** (MEASURED, `stack/tests/test_v6_t2_contrastive.py::
test_default_build_is_untouched_at_the_production_geometry`), so the live tensor-strict v6F S-W
resume is untouched by either.

⛔ **ESCALATION 1 — HALF OF CATALOG T2 IS NOT EXPRESSIBLE ON THIS ARCHITECTURE, AND IT IS NOT A
SMALL CAVEAT.** The catalog specs *"time-reversal **and** lane-mirror augmentations as hard
negatives"*. **`z_tac` is a function of the LAST FRAME ALONE** — MEASURED, two probes, §2. There is
no temporal extent at the tactical latent for a time reversal to act on: reversing the window hands
the tactical layer *the frame from W ticks earlier*, not "the manoeuvre played backwards". Worse,
using it as a hard negative would push apart the latents of nearby windows, which is **the direct
opposite of catalog T5 (F-8)**. ⇒ `lane_mirror` is the default hard negative; `time_reverse` is
built, measured, and **excluded from the default negative set** with the reason in code. **This is a
DIAGRAM/CATALOG-UPDATE decision, not something an implementation may silently resolve.**

⛔ **ESCALATION 2 — F-8 IS DEGENERATE ALONE AND THE GUARD IS NOW WIRED.** A constant control plan
scores **exactly 0** on T5. This is not hypothetical: **the unicycle emission outputs exactly zero
controls at initialisation** (MEASURED), so the term begins at its global minimum and its only
effect is to resist the plan objective. `v6_loss_step` **and** `preflight` both refuse
`w_t5_consist > 0` with `lambda_plan == 0`. Any future "T5 improved consistency" claim must be read
against the flat-plan control.

⚠️ **A defect in my own instrument, caught by its own test and fixed in the same change.** The T2
trivial-proxy control originally issued a verdict at any `n`. MEASURED at random init (where the
true ratio is 1 **by construction**), 5 seeds per cell:

| n per side | ratio range at the NULL |
|---|---|
| 4 | **0.397 – 3.361** |
| 16 | 0.595 – 1.471 |
| 64 | 0.949 – 1.281 |
| 256 | 0.829 – 1.036 |

⇒ a verdict from `n=4` is noise wearing a number's clothes. The control now **refuses below
`T2_CONTROL_MIN_N = 32`** and reports per-side SEM. **Root-cause class: a ratio quoted without its
n** — the same family as "never quote an interval without its estimator".

---

## 1. The spec, quoted — two independent locations per cell

Per the "absence found at ONE location is not absence" rule, each cell's spec was established at two
places before a line was written.

### F-7 / catalog T2

| source | quote |
|---|---|
| `…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:65` | *"T2 \| **manoeuvre-contrastive windows** (label-free): time-reversal and lane-mirror augmentations as hard negatives for the tactical predictor \| a lane change mirrored is the OPPOSITE manoeuvre — the predictor must not be invariant to it; teaches manoeuvre identity without manoeuvre labels \| TACTICAL family: confusion improves on the E4.1-derived … eval strata"* |
| `…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:56` | *"Needs: a T2 loss (label-free augmentations of the window + a contrastive head on `z_tac`) + a weight in `V6LossWeights`. Fix F-7"* |
| `…/DIAGRAM_CONFORMANCE.md:212` | *"F-7 \| P3 \| T2 manoeuvre contrastives (time-reversal + lane-mirror negatives on `z_tac`) — label-free, S-T-stage loss + weight."* |

### F-8 / catalog T5

| source | quote |
|---|---|
| `…/V6_TRAINING_MEASURES.md:68` | *"T5 \| temporal-consistency selection loss (momentum-aware, Drive-JEPA pattern) \| penalise plan flip-flop across consecutive windows (cross-frame comfort) \| LATERAL family: yaw-rate/curvature MAE at selection level; plan-switch rate reported"* |
| `…/DIAGRAM_CONFORMANCE.md:58` | *"no cross-window plan-flip-flop penalty; no plan-switch-rate logging. Needs consecutive-window batches (the current sampler draws windows independently) — a sampler + loss change. Fix F-8"* |
| `…/DIAGRAM_CONFORMANCE.md:213` | *"F-8 \| P3 \| T5 temporal-consistency selection loss — needs consecutive-window batches (sampler change) + plan-switch-rate logging (its gate row asks for LATERAL yaw/curvature MAE at selection level)."* |

⚠️ **The brief pointed at `stack/tanitad/models/v6.py` for `STAGE_MAY_INTRODUCE` and
`RESUME_CONTRACT`. They are not there** — both live in `stack/scripts/train_v6_staged.py`
(`:396` and `:299` respectively, pre-edit line numbers). Reported rather than worked around.

### What the spec does NOT say — the assumptions, named

| # | gap in the spec | narrowest defensible reading taken | where it is declared |
|---|---|---|---|
| A1 | **A contrastive loss needs a POSITIVE; the catalog names only negatives.** | a manoeuvre-**preserving** augmentation (`photometric` brightness/contrast — geometry, hence manoeuvre, untouched). Doubles as C1 nuisance-non-retention pressure. | `v6.py` T2 block; `--t2-positive` help text |
| A2 | which negatives are *easy* vs *hard* | in-batch other windows = easy; the anchor's own mirrored view = hard (the catalog's word) | `t2_contrastive_loss` docstring |
| A3 | "momentum-aware" is not defined operationally | the term compares the two plans **at the same absolute instants** (`plan_i[lag:]` vs `plan_j[:K-lag]`) — the model's own forward motion is what defines the alignment | `t5_consistency_loss` docstring |
| A4 | T5's space (position or control) | **control space** — justified by this programme's own measurement, §3 | the `F-8` block comment |
| A5 | T5's pair offset | `cfg.stride_tac` (5 steps = 0.5 s), overridable via `--t5-lag` | `--t5-lag` default 0 ⇒ stride_tac |

⛔ **The obvious "free" positive is DEGENERATE and that is MEASURED, not reasoned.** `uplink`
defaults to `"stopgrad"` (`v6.py:2778`), so `uplink_tac` returns `target = online.detach()` —
`z_tac_target` **is** `z_tac` with the graph cut. Through a unit-norm projector its cosine
similarity to the anchor is **exactly 1.0** for every window regardless of what the head learned,
so InfoNCE would collapse to a pure push-apart (flip-detector) objective. Pinned:
`test_z_tac_target_is_a_degenerate_positive`.

---

## 2. ⛔ THE ARCHITECTURAL FINDING — `z_tac` reads the LAST FRAME ONLY

**MEASURED**, by construction of the code path and then by execution:

1. `v6.py:3844-3847` — `encode_window` does `flat = frames.reshape(b*w, …); tok = self.encoder(flat)`.
   The encoder sees every frame **independently**; there is no temporal mixing on the path to `z_op`.
2. `v6.py:4197` — `z_op = z_op_win[:, -1]`.
3. `v6.py:4207` — `z_tac, z_tac_tgt = self.uplink_tac(z_op, own_tac)`.
4. `v6.py:3886` — inside `uplink_tac`, `online = self.adapter_tac(x)` with `x = z_op` under the
   default `shared_encoder=True`.

**Executed probes** (`test_z_tac_reads_the_last_frame_only`,
`test_time_reversal_is_an_earlier_frame_not_a_reversed_manoeuvre`):

| probe | result |
|---|---|
| replace every frame **except the last** with noise | `z_tac` **does not move** (`allclose`, atol 1e-6) |
| `z_tac(time_reverse(x))` vs `z_tac` of a window whose frames are all `x[:,0]` | **identical** (atol 1e-6) |
| `z_tac(lane_mirror(x))` vs `z_tac(x)` | **differs** — the mirror IS expressible |

**Consequences.**

- The **lane-mirror half of T2 is faithful and built**. It carries the catalog's entire stated
  justification (*"a lane change mirrored is the OPPOSITE manoeuvre"*).
- The **time-reversal half is not what the catalog means**, on this architecture. As a hard negative
  it would train *"the tactical latent at t must differ from the tactical latent at t−W"* — an
  anti-temporal-collapse objective, and **in direct opposition to T5/F-8**, which pulls nearby
  windows' plans together. Two terms of the **same stage** would fight.
- ⇒ built, exported and testable, but **not in the default negative set**. `--t2-negative
  time_reverse` exists so the arm can be measured the day the tactical path gains real temporal
  extent; the help text and the docstring both say why it is not the default.

⚠️ This is a **decision for the PI / diagram owner**, recorded here rather than resolved by fiat.

---

## 3. Why F-8 is in CONTROL space and needs no pose transform

**MEASURED, and it is this programme's own measurement** —
`…/2026-08-06-v1-defect-triage/results/TEMPORAL_STABILITY_RESULT.md` (40 OOD-val episodes,
6,794 consecutive pairs, stride-1, `flagship-v1arch-v2bal-30k` @ 29999):

| quantity | flagship v1 | GT floor |
|---|---|---|
| replan shift, mean | 0.0947 m | **0.0** |
| **replan accel jump, mean** | **1.1021 m/s²** | 0.0001 |
| manoeuvre toggle rate (per 0.1 s pair) | 0.1759 | — |

That document's own conclusion is that **"a small position shift hides a large acceleration
change"** — the commanded acceleration at the *same absolute instant* is revised by more than the
human's entire acceleration RMS (0.8048 m/s²). A position-space consistency term is blind to the
defect that actually exists.

⭐ **And controls are FRAME-INVARIANT**: acceleration and curvature do not depend on which ego frame
they are expressed in. So comparing `plan(t)` with `plan(t+lag)` needs **no relative pose, no pose
label, and no alignment approximation**, and the GT floor is **exactly** zero by construction (the
human's controls from `t+lag` are a suffix of those from `t`). Pinned by
`test_the_loss_is_in_control_space_and_needs_no_pose_transform`, which asserts the implementation
never reads `waypoints`.

---

## 4. MEASURED parameter cost — by building the module, never estimated

*(The brief's warning is honoured: a prior doc's "+41,089" was an ESTIMATE and the measured figure
was +33,801.)*

| build | params | state_dict keys |
|---|---|---|
| **default** (`V6Config()`) | **87,893,449** | **405** |
| `t2_contrastive=True` | 88,057,674 | 410 |
| **F-7 delta** | **+164,225** | **+5** |
| **F-8 delta** | **0** | **0** |

F-7's five keys: `t2_head.log_tau`, `t2_head.net.{0,2}.{weight,bias}`.
Arithmetic, pinned against the config so a geometry change cannot drift the constant:
`512·256 + 256` + `256·128 + 128` + `1` (log_tau) = **164,225**.

**F-8 adds nothing at all** — no parameters, no buffers, like `MpcRefiner`. That is why it needs no
`STAGE_MAY_INTRODUCE` entry and why it can be switched on over an existing checkpoint.

---

## 5. Stage contract — the earliest legal insertion point, stated plainly

`STAGE_MAY_INTRODUCE` (`train_v6_staged.py`) and `STAGE_GROUPS` (`v6.py:3138`) were read before
placing either cell.

| fact | value | consequence |
|---|---|---|
| `STAGE_MAY_INTRODUCE["S-W"]` | `()` | S-W starts the ladder — it can inherit nothing, so nothing may be *introduced* into an S-W **resume**. |
| `STAGE_MAY_INTRODUCE["S-J"]` | `()` | joint polish introduces nothing. |
| `STAGE_MAY_INTRODUCE["S-S"]` | `()` | S-T already carried everything S-S trains. |
| `STAGE_GROUPS["S-W"]` | `("encoder","readout","predictor_op","aux")` | **`layer_tac` is NOT trained in S-W** — so even a *fresh* S-W run would carry an untrained T2 head. |
| `STAGE_GROUPS["S-T"]` | `("layer_tac","planner")` | the stage that trains both cells' targets. |
| `encoder`/`readout`/`aux` in S-T and S-S | **frozen** | neither cell touches them; T2's gradient is cut at the uplink. |

⇒ **F-7's earliest legal insertion point is S-T**, via `STAGE_MAY_INTRODUCE["S-T"] += ("t2_head.",)`
— exactly the F-1 / F-18 path, and it does **not** require a fresh S-W run. The head rides in over
an S-W checkpoint that never had it and trains immediately, because `t2_head.` is grouped
`layer_tac` and S-T trains `layer_tac`.

⚠️ **Unlike `agent_slots.`, this entry means the ordinary thing.** `agent_slots.` is an
introduction-**only** entry (no ladder stage trains it). `t2_head.` is introduced *and* trained by
the same stage. The distinction is recorded at the entry.

⇒ **F-8 has no insertion point because it needs none.** Zero keys ⇒ nothing for the allowlist to
adjudicate ⇒ it may be enabled in **any** S-T or S-J run, including one resuming an existing
checkpoint tensor-strict.

**Carry rule (recorded, not chain-enforced — the same gap `agent_slots.` documents):** an S-T run
launched with `--t2-contrastive` writes `t2_head.*` into its checkpoint, so **S-S and S-J must be
launched with the flag too**, or those keys become UNEXPECTED and `load_stage_init` is fatal.
`v6_chain.assert_geometry_carry` enumerates its levers first-class and there is no `Step.t2_contrastive`.

---

## 6. The `06b8782` defect class — checked, and avoided by construction

The brief flags it correctly: commit `06b8782` **appended `interp` to `MODULE_GROUPS`**, which
changed what S-J trains **without touching the line that declares S-J**, because `STAGE_GROUPS["S-J"]`
was the bare alias `MODULE_GROUPS`. Tuple immutability protected against mutation, not against
meaning.

**Neither cell touches `MODULE_GROUPS`.** F-7 adds a `_GROUP_PREFIXES` entry
`("t2_head.", "layer_tac")` that maps into an **existing** group. Therefore:

| check | result (MEASURED) |
|---|---|
| `MODULE_GROUPS` membership | **unchanged** — 8 entries, identical tuple |
| `LADDER_UNTRAINED_GROUPS` | **unchanged** (`{"interp"}`) |
| `STAGE_GROUPS["S-J"]` contents | **unchanged** |
| `STAGE_GROUPS["S-J"] is MODULE_GROUPS` | **False** — the identity alias is gone, as the brief states |
| no stage's trainable set moved without its declaring line being edited | **holds** |

⇒ **the defect class does not apply to this change.** Adding a prefix into an existing group cannot
change any stage's trainable *set of groups*; it only assigns new parameters to a group whose stage
membership is already declared. Pinned by `test_the_head_is_layer_tac_and_trains_in_S_T_only`, which
asserts the per-stage `requires_grad` outcome directly rather than trusting the declaration.

---

## 7. Guards — wired in the same change, per "a guard that is never called is not a guard"

| guard | where it FIRES | test |
|---|---|---|
| T5 without a plan objective (**the degeneracy**) | `v6_loss_step` **and** `preflight` | `test_v6_loss_step_refuses_the_term_without_a_plan_objective`, `test_preflight_refuses_the_degenerate_and_unpaired_launches` |
| T5 without consecutive-window pairs | `v6_loss_step` + `preflight` | `test_v6_loss_step_refuses_the_term_without_pairs` |
| T5 lag with no overlap (0, ≥K, negative) | `t5_consistency_loss` | `test_the_loss_refuses_a_lag_that_leaves_no_overlap` |
| T5 empty pair set | `t5_consistency_loss` | `test_the_loss_refuses_an_empty_pair_set` |
| T2 weight with no projector | `t2_contrastive_loss` + `preflight` | `test_the_loss_refuses_a_missing_projector` |
| T2 in a stage that freezes `layer_tac` | `preflight` | `test_preflight_refuses_T2_without_its_projector_and_in_the_wrong_stage` |
| **T2 positive/negative SWAPPED** | `t2_contrastive_loss` | `test_the_loss_refuses_a_swapped_positive_and_negative` |
| T2 under the E-ENC arm (b) (half-augmented pair) | `t2_contrastive_loss` + `preflight` | — |
| T2 control below its sample floor | `t2_flip_detection_control` | `test_the_control_refuses_a_verdict_below_its_sample_floor` |

⚠️ **C95/C97 discipline: every refusal has its opposite pinned too.** A *legal* T2 launch and a
*legal* T5 launch are both asserted to raise **nothing** — this programme shipped a
rejects-everything guard and a passes-everything guard within one day.

⭐ **A bug the tests caught in the guard itself.** `preflight` read `a.shared_encoder`, which does
**not exist** on the namespace (the flag is `--per-layer-encoders`; `shared_encoder` is the *config*
field derived as `not a.per_layer_encoders`). It would have raised `AttributeError` at launch. Fixed,
and the note is at the fix site. *(Exit codes are not evidence — this was found by an assertion, not
by the code running.)*

---

## 8. Controls — positive AND trivial-proxy, for both cells

| cell | positive control | trivial-proxy control |
|---|---|---|
| **F-7** | `test_the_loss_falls_when_the_head_is_optimised_POSITIVE_CONTROL` — 25 Adam steps, loss must strictly fall. *(C79: D1 was withdrawn because a probe failed its positive control.)* | `t2_flip_detection_control` — splits by mean \|steer\|. Mirroring a **straight** window is manoeuvre-preserving; mirroring a **turning** one is not. **ratio ≈ 1 ⇒ FLIP DETECTOR**, not manoeuvre identity. Its own negative case is pinned: an untrained head **must** come back ≈1 at n=64. |
| **F-8** | `test_a_consistent_pair_scores_exactly_zero_POSITIVE_CONTROL` — when plan *j* genuinely is plan *i*'s lag-suffix the loss is 0, the same floor `TEMPORAL_STABILITY_RESULT.md` measured on real data. Paired with `test_an_inconsistent_pair_scores_above_zero`. | `test_a_flat_plan_scores_exactly_zero` — **a constant plan wins the term outright**. Any T5 result must be read against this. Reinforced by `test_the_emission_is_flat_at_initialisation_so_T5_starts_at_its_optimum`. |

⚠️ **C92 is the reason F-7 has a trivial-proxy control at all**: a headline died because a readout
was echoing ego speed. "The margin went up" is **not** evidence that T2 taught manoeuvre identity —
the ratio is.

---

## 9. What was built, file by file

### `stack/tanitad/models/v6.py`

- `lane_mirror_window` — horizontal flip + **lateral action channel negated only** (longitudinal and
  speed untouched: mirroring a scene does not change how fast the ego is going). Involution, pinned.
- `time_reverse_window` — built, **not default**, with the §2 finding in its docstring.
- `photometric_jitter_window` — the declared manoeuvre-preserving positive (per-frame affine
  intensity; pixel rank order **within each frame** preserved, pinned).
- `T2_AUGMENTATIONS`, `T2_MANOEUVRE_PRESERVING`, `T2_MANOEUVRE_REVERSING` — the partition as **data**,
  so a launch line that swaps positive and negative is *refused* rather than silently training the
  inversion of the catalog row. Exhaustive-and-disjoint, pinned.
- `ManoeuvreContrastiveHead` — 2-layer projector + learnable `log_tau`, unit-norm output.
- `V6Config`: `t2_contrastive=False`, `d_t2_proj=128`, `d_t2_hidden=256`, `t2_tau=0.1`.
- `V6Stack.__init__`: `self.t2_head = None` unless enabled, **built last** (default path draws no RNG,
  creates no key).
- `_GROUP_PREFIXES` += `("t2_head.", "layer_tac")`.
- `__all__` extended.

⭐ **`forward` is deliberately untouched.** The loss calls the head directly, so the output dict gains
no key and `test_v6_gstr_port.py::test_default_forward_is_bit_identical_and_emits_no_new_key` — which
has already caught one unconditional key — cannot be tripped by this cell.

### `stack/scripts/train_v6_staged.py`

- `V6LossWeights`: `w_t2_contrast=0.0`, `w_t5_consist=0.0`; both zeroed in `for_stage("S-W")` and
  `for_stage("S-S")`.
- `STAGE_MAY_INTRODUCE["S-T"]` += `"t2_head."` (with the F-8 "no entry needed" note at the site).
- `t2_contrastive_loss` — InfoNCE `[B, B+1]`: column *i* positive, `j≠i` easy negatives, column *B*
  the anchor's own mirrored view as the **hard** negative.
- `t2_flip_detection_control` + `T2_CONTROL_MIN_N`.
- `t5_consistency_loss` — control-space MAE over the overlapping absolute instants, weighted by the
  selector's softmax (**"at selection level"**, differentiable into the scorer).
- `t5_plan_switch_rate` — the *"plan-switch rate reported"* half, **per axis** (LAT/LON), the
  successor of the 0.1759 toggle rate.
- `v6_loss_step`: both terms wired, batch contract documented, degeneracy guard.
- **Sampler**: `--t5-pairs` builds a `partner` map from `ds_train.index` (precedent:
  `train_tactical_stage0.py:685-694`), zeroes O4 weight on unpartnered tail windows, and draws
  `batch//2` anchors + their `+lag` partners.
- `preflight`: six new refusals.
- CLI: `--t2-contrastive`, `--d-t2-proj`, `--d-t2-hidden`, `--t2-tau`, `--w-t2-contrast`,
  `--t2-positive`, `--t2-negative`, `--w-t5-consist`, `--t5-pairs`, `--t5-lag`, `--t5-w-kappa`.

⛔ **PARITY IS UNTOUCHED.** `--t5-pairs` re-selects **no episode**. It pairs windows *within*
episodes the parity key already chose (`physicalai-train-e438721ae894`, 2376 eps, skip-hash
`f09e44db`), and the tail windows it excludes are excluded from the **anchor draw only** — every
window remains reachable as a partner.

⚠️ **RNG discipline.** `gen` is shared between `InteractionSampler`, `sample_random_deltas` and
`v6_loss_step`, so an extra draw would move every other term bit-for-bit. With `--t5-pairs` **off**,
nothing in the new block runs and the stream is consumed exactly as before.

---

## 10. Suites

Run **separately** with `PYTHONUTF8=1` (combining them breaks rootdir resolution and exits 0 on a
collection error), and not concurrently with any other CPU job.

| suite | result |
|---|---|
| `stack/tests/test_v6_t2_contrastive.py` | **26 passed** |
| `stack/tests/test_v6_t5_consistency.py` | **26 passed** |
| `stack/tests/test_v6_stage_init_introduction.py` + `test_v6_ladder_edges.py` + `test_v6_chain.py` | **92 passed** |
| `taniteval` (full) | **1136 passed**, 139 s |
| `stack` (full) | **3991 passed, 7 skipped, 2 xfailed**, ~640 s |

⛔ **THE FIRST FULL `stack` RUN FAILED 3 AND STILL EXITED 0.** `test_v6_stage_init_introduction.py`
pins `STAGE_MAY_INTRODUCE["S-T"]` **exactly** — *"growing the allowance must fail here first and be
extended consciously"* — so adding `t2_head.` correctly broke it in three places. The pin was
extended with the reason at the site. **Had I trusted the exit code the three failures would have
shipped**; this is the seventh instance of the "exit codes are not evidence" class this session.

⚠️ **An edge case the sampler review caught** (§9): `InteractionSampler.__call__` falls back to
uniform weights when an episode's weights sum to zero (`v6.py:544-546`), which would defeat the
tail-window masking for an episode **all** of whose windows are unpartnered (any episode with
≤ `t5_lag` windows). The draw now filters to partnered anchors with a bounded retry and a **named
refusal** rather than a bare `KeyError` deep in the step loop.

---

## 11. Escalations — decisions this change does NOT make

1. ⛔ **The T2 time-reversal half** (§2) — catalog/diagram update, or an architecture change giving
   the tactical path real temporal extent. **PI / diagram owner.**
2. ⛔ **T5's weight is a declared decision, never a default.** Units are m/s² and 1/m; it is not
   commensurate with `w_select`/`w_anchor` (metres) or `w_t2_contrast` (nats). No numeric value is
   pre-registered and inventing one was out of scope.
3. ⚠️ **The `t2_head.` carry rule is not chain-enforced** (§5) — plumbing `--t2-contrastive` into
   `v6_chain` is the follow-on the moment a chain step wants the head. Same gap `agent_slots.` has.
4. ⚠️ **F-7's compute cost when ON: two extra encoder passes per step** (positive + hard-negative
   views). Zero when off. Not benchmarked on Thor — Thor is training and off-limits.
5. ⚠️ **Neither cell has been trained.** Everything here is construction-, contract- and
   control-level evidence. No claim is made about whether T2 or T5 *improves* anything.

---

## 12. Deliverable manifest

**Everything below exists in the repo working tree AND is staged. Nothing lives in only one place;
nothing is on a pod or in a worktree. STAGED, NEVER PUSHED — no commit was made.**

| artifact | location | state | staging verified by |
|---|---|---|---|
| F-7 augmentations, `ManoeuvreContrastiveHead`, config fields, group prefix, `__all__` | `repo:stack/tanitad/models/v6.py` | MODIFIED | index blob == worktree blob |
| F-7 + F-8 losses, controls, weights, `STAGE_MAY_INTRODUCE`, CLI, preflight, pair sampler | `repo:stack/scripts/train_v6_staged.py` | MODIFIED | index blob == worktree blob |
| allowlist pin extended (+ reason at the site) | `repo:stack/tests/test_v6_stage_init_introduction.py` | MODIFIED | index blob == worktree blob |
| F-7 suite — 26 tests | `repo:stack/tests/test_v6_t2_contrastive.py` | NEW | `git ls-files --cached` PRESENT |
| F-8 suite — 26 tests | `repo:stack/tests/test_v6_t5_consistency.py` | NEW | `git ls-files --cached` PRESENT |
| this report | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-f7-f8-cells/F7_F8_CELLS.md` | NEW | `git ls-files --cached` PRESENT |

⚠️ **Foreign entries are in the index from concurrent streams** — `CLAUDE.md`,
`Project Steering/MODEL_REGISTRY.md`, `Project Steering/RETRACTION_LOG.md`,
`…/2026-08-18-monitor-fixes/{MONITOR_FIXES.md,suite_full_clean.txt}`, and
`stack/scripts/scoped_commit.py` (staged **deleted** while present untracked — an odd state another
stream left). **Not mine, not touched.** Whoever commits must apply the CLAUDE.md git-hygiene rule:
check for foreign staged entries first, and prefer a pathspec-free `git commit -F <msgfile>` only
after confirming every index entry is intended.
