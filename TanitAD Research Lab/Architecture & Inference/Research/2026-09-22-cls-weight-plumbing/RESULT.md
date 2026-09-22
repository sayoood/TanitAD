# 2026-09-22 — `H-BOXCLS-1` plumbing, the stamp that states the TENSOR, and the two S1 controls

**Evidence class:** MEASURED (ours), CPU only, no checkpoint, no corpus, no GPU.
**Tier:** not applicable — nothing here is a driving number. These are implementation,
identities and controls.

---

## 1. What was built

`--agent-cls-weight {off,train2400}` on `refc_v3_train.py`, end to end:

| piece | where |
|---|---|
| the banked vector + its recipe | `stack/tanitad/data/agent_cls_weights_train2400.json` |
| loader, digest, refusal | `agent_slots.load_cls_class_weight`, `agent_slots.cls_weight_digest` |
| attach to the model | `refc_v3_train.py` beside `model._w_map` |
| **both** loss paths | `agent_losses` (2-D `core.agent_head`) **and** `box3d_loss_row` (3-D `perception.box_dec`) |
| the run record | `_cls_weight_stamp(args)` → `config.json["agent_cls_weight"]` |
| the guard | `assert_seams_are_built`, bidirectional + digest-aware |

⛔ **Both loss paths, not one.** The class collapse is MEASURED on BOTH heads — 2,000/2,000
and 320/320 slots emitting one class of ten — and `perception.box_dec` is the head
`s1_pass.py:307` actually scores. A fix wired to the 2-D seam alone would have left the
scored head untouched while the config said `train2400`.

⭐ **`off` is BIT-IDENTICAL to every arm trained before the flag**, not merely close.
`slot_set_loss`'s denominator FOLLOWS the weight, so `weight=ones` divides by the same count
`None` does and the two are exactly equal. Pinned as `==`, never `allclose`. That denominator
does a second job: it fixes the term's SCALE, so re-weighting moves exactly one variable
(relative emphasis) instead of also shifting the term against its siblings.

---

## 2. The stamp states the TENSOR, not the flag and not the file

`config.json["agent_cls_weight"]` uses the same three-fact shape as `tac_goal_tok_head` and
`max_speed_input`: `requested` (argv), the artifact's ten values + counts + provenance, and
**`built` = `cls_weight_digest(model._cls_class_weight)`** — the digest of the tensor the model
carries, filled in `train`.

⛔ **`built` is read off the model, never re-read from the artifact.** A slot that re-loads its
own source agrees with itself forever and is blind to the one defect it exists for: a flag that
parses and reaches nothing (`305debd`, `occ_from_geometry`).

The guard is bidirectional, and because `built` is a DIGEST rather than a boolean a third
failure is reachable that a boolean cannot see. MEASURED against the real function:

| state | outcome |
|---|---|
| `off`, nothing attached | ✅ passes |
| `train2400` attached, digest agrees | ✅ passes |
| record asks for it, model has nothing | ⛔ REFUSES |
| model has it, record says `off` | ⛔ REFUSES |
| record names a different vector | ⛔ REFUSES |
| **record self-consistent, model carries another vector** | ⛔ REFUSES |

---

## 3. Evidence: 23 tests, mutation-proven 9/9

`stack/tests/test_cls_weight_stamp.py`; prover `code/mutate_cls_stamp.py`; artifact
`raw/mutation_proof_cls_stamp.json`. Both target files restored **byte-identical**, final
clean run green.

| arm | defect reintroduced | RED |
|---|---|---|
| S1 | `built` re-reads the artifact | ✅ |
| S2 | the digest ignores class names | ✅ |
| S3 | the loader skips its own digest verification | ✅ |
| S4 | the `agent_losses` site drops the weight | ✅ |
| S5 | the `box3d_loss_row` site drops the weight | ✅ |
| S6 | the "record says off" branch disabled | ✅ |
| S7 | the digest-comparison branch disabled | ✅ |
| S8 | the "requested but not built" branch disabled | ✅ |
| S9 | an out-of-vocabulary label relabelled to class 0 | ✅ |

⚠️ **TWO ARMS ESCAPED FIRST, AND BOTH ESCAPES WERE THE TEST'S FAULT — which is the most
useful thing in this package.**

* **S6** originally mutated only a refusal MESSAGE. The test, which asserted that three
  substrings appeared in the trainer's source, stayed **green against a guard it had never
  exercised**. That is `A CHECK THAT SHARES THE DEFECT IT CHECKS FOR` one level up: the check
  shared the *source file* with the thing it checked. Fixed by making the arm disable the
  BRANCH and rewriting the test to CALL `assert_seams_are_built`.
* **S7** was masked by a second, redundant-looking branch. Isolating it required a case that
  did not exist until the mutation demanded it: **the record internally consistent — its stated
  artifact digest and its `built` slot agreeing — and both describing a vector the model does
  not carry.** Only a check that reads the LIVE TENSOR sees that.

⇒ **A mutation proof is not a formality.** Both escapes produced a better test than I would
have written unprompted, and the second produced the case that justifies `built` being a digest.

⚠️ Two process traps re-paid while building the prover, both already in `CLAUDE.md` and both
now closed in the tool itself: an anchor carrying a line terminator matches nothing in these
CRLF files, and — new — **an anchor matched as a SUBSTRING collides across indentation levels**:
the two loss call sites differ only by indent, so the 12-space anchor is a proper substring of
the 20-space line and matched both. The prover now matches **whole lines by equality**, and an
anchor that does not match exactly once ABORTS the run as INVALID rather than being skipped.

---

## 4. E11 / E12 — the two S1 controls, read on CPU before any GPU

`code/s1_controls.py` → `raw/s1_controls.json`. The checker is
`tanitad.rl.pdm_proxy.no_at_fault_collision`, **IMPORTED** (the prereg is explicit that
`taniteval/tools/fan_safety.py` uses a different collision model and is not used).

| arm | committed value | MEASURED |
|---|---|---|
| **`S1-GATE-CONST`** | recovery **exactly 0** | **0** — the gate changed the pick in **0 / 64** windows, `nc == 1.0` for every candidate |
| **`S1-RANDOM`** | **exactly** the candidate mean | max abs difference **0.00e+00** over 256 draws |

⭐ **AND THE CONTROL THAT MAKES THE ZERO MEAN SOMETHING.** "Recovery exactly 0 under empty
tracks" is satisfied *vacuously* by a gate that never masks anything under ANY tracks — which
would also make `S1-GATE-ORACLE` read 0 while looking like a clean control. The same code path
was therefore run against a stopped car on the ego's line: the gate moved the pick in
**64 / 64** windows, with `nc ∈ {0.5, 1.0}`. ⇒ the zero is a property of the **empty world**,
not of an inert gate.

⚠️ `S1-RANDOM` consumes **no RNG** and carries **no interval**: the arm is the uniform
*expectation* over the fan, so it is exact. A sampled version would have reported a CI around a
quantity that has none — the weaker artifact, and easy to produce by accident.

⛔ **Why these ran now rather than with the rest of S1.** `S1-GATE-PRED` is NOT RUNNABLE against
the collapsed box head. Had the controls only run alongside it, any deviation would have been
unattributable between *"the gate is wrong"* and *"the head is collapsed"*. Establishing them
first makes a future deviation attributable to exactly one thing.

---

## 4b. What the lever ACTUALLY does — exact, no GPU

`code/cls_gradient_mass.py` -> `raw/cls_gradient_mass.json`. §5 below used to name the
milder-exponent question and leave it; RULE ZERO says a turn that names the next lever and does
not run it is unfinished, so the part that needs no GPU was run — and it is the decisive part.

**The structural fact narrows the question to one variable.** For a slot with target `c`,
`d/d(logits)[w[c] · CE] = w[c] · (softmax − onehot)`. The weight is a SCALAR on that slot's whole
gradient, so ⛔ **weighting changes NO slot's gradient direction.** With the weight-following
denominator fixing the total scale, the entire intervention is a redistribution of **gradient
mass** across classes — computable exactly from the census counts, with no model at all.

| | `automobile` | `animal` |
|---|---|---|
| unweighted share of the `cls` term's gradient mass | **76.574 %** | **0.0715 %** |
| at full inverse frequency (α = 1) | **10.000 %** | **10.000 %** |
| change | ÷ 7.7 | **× 139.9** |

⭐ **The identity control is ANALYTIC, and it caught the one thing worth catching.** At α = 1,
`count · w` is constant, so every class must land on exactly `1/C`. Derived from the COUNTS the
deviation is **2.8e-17**; against the **BANKED** vector it is **5.5e-06**, because the artifact
stores 6 dp. The mathematics is exact, the ARTIFACT is not, and the gap is the artifact's
precision — negligible here (5.5e-06 on a 0.1 share) but *measured* rather than assumed. ⚠️ This
is deliberately the inverse of the failure `CLAUDE.md` records where a builder rounded its ladder
to 4 dp and then verified the shipped buckets **against that same rounded ladder**, reading
"0 % moved" while 57.5 % of the corpus moved against an independently derived one.

**And the next arm is left behind with its arithmetic already done.** `w ∝ count^(−α)` gives
class mass share `∝ count^(1−α)`, so α is the single knob:

| α | weight ratio maj:rare | `automobile` share | `animal` share | `animal` fold vs α=0 |
|---|---|---|---|---|
| 0.00 | 1.0 | 76.57 % | 0.07 % | ×1 |
| 0.25 | 5.7 | 63.42 % | 0.34 % | ×4.7 |
| 0.50 | 32.7 | 44.81 % | 1.37 % | ×19.2 |
| 0.75 | 187.2 | 24.59 % | 4.30 % | ×60.1 |
| 1.00 | 1071.1 | 10.00 % | 10.00 % | ×139.9 |

⇒ a milder arm is no longer a hunch; it is `--agent-cls-weight` at a named α, and the table says
what each one buys before any GPU is spent.

---

## 4c. Training readiness — the join is INSIDE parity, measured

⛔ **Parity is sacred**: *"anything that re-selects episodes breaks cross-arm comparability and
must be refused."* So "the train agent join exists" is not a readiness statement; "its clip set
is a SUBSET of the parity set" is.

| | |
|---|---|
| join episodes | **2,308** (2,308 distinct digests) |
| banked parity digest set (`is_full_corpus: true`) | **2,400** |
| join **inside** parity | **2,308** |
| join **OUTSIDE** parity | ⭐ **0** |
| parity clips absent from the join | **92** |

⭐ **The gap closes on its own arithmetic** — the join's meta names `no_obstacle` **79**,
`registration_failed` **10**, `bad_clip` **3**, and **79 + 10 + 3 = 92**, exactly the shortfall.
⇒ the join drops clips for three stated reasons and re-selects nothing. An `--agents` arm trains
on **96.17 %** of the parity clip set, entirely within it — a stated, attributable coverage
reduction, not a comparability break. *(Control: a non-clip string reads NOT in parity, so the
membership test discriminates.)*

⚠️ **Three layers, and a number must say which it counts** — the `"corpus" means the CLIP SET`
scope error one level over: parity **clip set 2,400** (MEASURED here), parity **episode build
2,376** (⚠️ INHERITED from `CLAUDE.md`, not re-measured), agent **join 2,308** (MEASURED).
⛔ 2,376 − 2,308 = 68 is **not** the 92, because the 92 is against the clip set. These two are
deliberately left unreconciled: the episode-build layer was not measured here.

⚠️ **Preflight is NO-GO and correctly so.** `refcv5_preflight.py` run bare reports 5 PASS /
2 FAIL / 9 INCONCLUSIVE. The INCONCLUSIVEs are all *"no `--v2-cache` / `--agent-join` /
`--anchors` given"* — it was run without data arguments, so they are honest unknowns, not
findings. Of the two FAILs: `taniteval.ci` is the known **namespace-package shadow** (the inner
package is `taniteval/taniteval/`, so the path must name that directory, not the repo root — it
resolves immediately when it does), and `agent_w_ground` is `D-RC5-GROUND-DEAD`, already
registered, already guarded by `assert_ground_prior_is_supervised`, and re-derived independently
here (§4d).

---

## 4d. `D-RC5-GROUND-DEAD` re-derived from a live run — NOT a new finding

⛔ Stated plainly because the temptation was to write it up as a discovery: the register already
carries `D-RC5-GROUND-DEAD`, `D-RC5-GRADGATE` and `H-RC5-GROUNDFIX`, and the trainer already
refuses the weight. I ran the preflight, saw `agent_w_ground grad=1.164e-10 DEAD`, chased it to
`_ground_terms`, and checked the register **after** — the right order, and the register had it.
The re-derivation agrees to the digit (**1.164e-10**).

**Two increments the single banked measurement could not give.**

*It is input-INDEPENDENT.* Over five distributions the residual never leaves float32 rounding
noise (eps **1.192e-07**): preflight's own **7.465e-08**, near field **6.457e-08**, far field
**7.352e-08**, wide lateral **8.427e-08**, and degenerate `cy = 0` **1.561e-15**. ⇒ a tautology
by construction, not a prior that happens to be satisfied on the probe's input — and only the
first justifies `H-RC5-GROUNDFIX`.

*The signature carries the missing ingredient.* `ground_range_prior(pred_box, cam,
z_center_m=0.75)` **declares `z_center_m` and never uses it.** Placing the foot at `z = 0` and
intersecting the ray with `z = 0` makes the two sides one computation; a foot at a stated box
height would have made the back-projection independent. The unused parameter is direct evidence
about the intended design.

⚠️ **Not fixed here, deliberately** — making the term real changes what an arm trains, which is
an architecture change, and `H-RC5-GROUNDFIX` is registered as needing a prereg and a param-band
ruling.

---

## 5. What this does NOT establish

* ⛔ **Nothing here says the re-weighting WORKS.** `H-BOXCLS-1`'s hypothesis is that weighting
  the `cls` term breaks the collapse; that needs a trained arm, and the arm needs the PI's GPU
  call. This package establishes only that the lever is correctly connected and honestly
  recorded.
* ⛔ **No T0/T1 number is produced or implied.**
* ⚠️ The weight vector is the TRAIN inverse frequency at imbalance **1,071.1 : 1**. §4b now
  quantifies exactly what that does to gradient mass and parametrises the milder-exponent
  family, but ⛔ **which α is best remains an empirical question that only a trained arm
  answers.** No α other than 1.0 is implemented, and none is pre-registered.

---

## 6. Deliverable manifest

| artifact | where |
|---|---|
| tests (23) | `stack/tests/test_cls_weight_stamp.py` |
| weight artifact + recipe | `stack/tanitad/data/agent_cls_weights_train2400.json` |
| loader, digest, refusal | `stack/tanitad/models/agent_slots.py` |
| flag, attach, stamp, guard | `stack/scripts/refc_v3_train.py` |
| 3-D loss forward | `stack/tanitad/models/refcv6_perception_branch.py` |
| corrected prose (3 sites) | `refc_agents.py`, `refcv5_preflight.py`, `test_refc_agents.py` |
| mutation prover | `code/mutate_cls_stamp.py` |
| mutation proof | `raw/mutation_proof_cls_stamp.json` |
| S1 controls probe | `code/s1_controls.py` |
| S1 controls result | `raw/s1_controls.json` |
| gradient-mass analysis | `code/cls_gradient_mass.py` |
| gradient-mass result | `raw/cls_gradient_mass.json` |
| retraction | `Project Steering/RETRACTION_LOG.md` — `RETR-2026-09-22-SELF-ATTESTING-DIGEST` |
| correction | `Project Steering/GOALS_AND_CLAIMS.md` — `CORR-2026-09-22-VOCAB-ATTRIBUTION` |

**Blocked, and it is the only thing:** the `H-BOXCLS-1` GPU call is the PI's.
