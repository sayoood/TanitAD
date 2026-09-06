# H19-STAMP-1 — the `man5 = None` guard: what it drops, what it does not, and the stamp

**Date** 2026-09-06 · **Branch** `agent/arch-inf-20260803` · **Compute** ZERO GPU
(CPU smoke rung + banked run records) · **Evidence class** MEASURED unless marked.

---

## 0. Headline — the escalated claim is HALF RIGHT, and the half that is wrong matters

> *escalated:* "Any non-`kin3` tactical vocabulary sets `man5 = None` and
> **silently drops the H19 lateral prior**. It is guarded, so it never errors."

⛔ **REFUTED in its strong form.** The H19 prior is **NOT dropped**. Under **both**
vocabularies the decoder receives a valid `maneuver_logits [B, 5]`, because
`refc.py`'s `reweight = maneuver_logits if maneuver_logits is not None else
man_logits` substitutes the **core's own** kin3-derived 5-way. A run under `v7.0`
still has a live anchor prior.

✅ **CONFIRMED in its true form, and it is worse than "lateral".** What the guard
drops is the **TACTICAL BRAIN's feed** into that prior — and the loss is an
**EXACT ZERO, not a degradation** — across **three** tensors, not one.

⭐ **And the record charge stands unqualified:** refcv4b's own `config.json`
carries **no H19 field of any kind**. `tac_vocab_version` is present, but
deriving H19 provenance from it requires knowing a guard in a third file.

---

## 1. The blast radius — every `man5 = None` path and every H19 consumer

**Two mechanisms, each with a same-breath non-zero control.**
M-A: `grep -rn man5` (POSIX, Bash tool) over `stack/`. M-B: ripgrep (Grep tool)
over the same tree. Controls, re-measured on the fast mirror rather than quoted
from memory: `grep -c "def " stack/tanitad/models/v6.py` = **142**,
`grep -c import stack/tanitad/refs/refc_v3.py` = **21**.
**Both mechanisms returned the same two sites.**

⚠️ **Search the FORM, not the concept.** A literal `grep "man5 = None"` finds
**only** the labeller line — the guard is written as a ternary
(`… if … == "kin3" else None`) and never contains that string. Both mechanisms
above searched the bare token `man5`, which is why they found it. An absence
claim from the literal form would have been an absence claim about the regex.

### Every `man5 = None`

| site | what it is |
|---|---|
| `stack/tanitad/refs/refc_v3.py` — `RefCV3Model._hook`, the `man5 = (… if self.tac_vocab_version == "kin3" else None)` expression | ⛔ **THE site.** The only one in model code. |
| `stack/scripts/refb_labels.py` — the labeller's per-window dict initialiser | ⭐ **UNRELATED.** A *label field* named `man5`, set immediately after from `classify_maneuver_v2`. Not the prior, not a guard. |

### Every H19 consumer, in order

1. `refc_v3.py::_hook` — emits `hook_out["maneuver_logits"] = man5` **only if
   `man5 is not None`**.
2. `refc.py::RefCModel.forward` — `if maneuver_logits is None: maneuver_logits =
   hk.get("maneuver_logits")` (the hook port).
3. `refc.py` aux-head block — computes the core's own `man_logits` from
   `lat_head`/`lon_head` (factored) or `maneuver_head`, **always**.
4. ⭐ **THE FALLBACK, and the reason the strong claim is wrong** —
   `reweight = maneuver_logits if maneuver_logits is not None else man_logits`.
5. ⚠️ **THE SECOND SEAM, missed by the escalation** — `if maneuver_logits is not
   None and self.cfg.factored_maneuver:` re-derives `lat_prior`/`lon_prior` via
   `tac.invert_man5`. So under `kin3` the tactical brain drives the **D-TAC1
   grafts too**; under `v7.0` that branch never runs.
6. `refc.py::AnchoredDecoder.forward` — `if self.maneuver_to_anchor is not None
   and maneuver_logits is not None:` … plus `lat_to_anchor` / `lon_to_anchor`.
   ⚠️ **In a factored build `maneuver_to_anchor` is `None`** and H19 is carried by
   `lat_to_anchor` (default init — refc.py: it *"inherits the LIVE H19 role"*)
   with `lon_to_anchor` zero-init. Reading `maneuver_to_anchor is None` as "no
   H19" is a trap; the stamp reads the built graft instead.
7. Grafts applied to **both** the classifier surface and the refined/sampled
   surface (`_apply_grafts` twice, deliberately).

---

## 2. What the live v7.2 arm actually lost — MEASURED, exactly zero

`raw/h19_blast_radius.py` → `raw/h19_blast_radius.txt`. CPU smoke rung, hier
build, `graft_maneuver=True`. Perturb **only** `lat_head_tac` / `lon_head_tac`
(the z_tac action heads — the tactical brain's decision surface) by
`+N(0, 5)` on weight and bias, and read the tensors **the decoder receives**:

| moved by the z_tac action heads? | `kin3` | `v7.0` |
|---|---|---|
| decoder `maneuver_logits` | MOVED max\|d\| **24.4632** | ⛔ **UNCHANGED 0.0** |
| decoder `lat_prior` (D-TAC1) | MOVED max\|d\| **21.8972** | ⛔ **UNCHANGED 0.0** |
| `anchor_logits` | MOVED max\|d\| **18.0365** | ⛔ **UNCHANGED 0.0** |
| decoder got a `maneuver_logits` tensor at all | **YES** `[B, 5]` | **YES** `[B, 5]` |

⇒ **Under `v7.0` the 8-wide v7 tactical action heads are trained by CE and have
NO inference-time influence on the trajectory decoder's anchor ranking.** They
reach it through **E7** (`target_latent`) and **E9** (goal selection) only.
That is a *structural zero*, not a noisy null — no seed, window count or
replicate arm changes it, so `H-ESTIM-SEED-1` does not apply to this row.

### The live arm did pay it — MEASURED from its own record

| fact | value | artifact |
|---|---|---|
| `tac_vocab_version` | **`v7.0`** | `C:\Users\Admin\refcv4b_viz\config.json` **and** `C:\Users\Admin\navcomp\ckpt\config.json`, 6,416 B each, agreeing |
| `v7_labels` | present (⇒ `_pin_trainer_cfg` pins `v7.0`) | same |
| any `h19_*` field | ⛔ **NONE** — `param_breakdown` (10 keys), `registered_delta`, and no H19 line | same |

⚠️ **There is no `--tac-vocab-version` CLI flag** (positive assertion: `grep
add_argument … | grep -i vocab` returns nothing against a same-breath control of
**88** `add_argument` calls). The vocabulary follows `--v7-labels`. The brief's
phrasing "refcv4b ran `--tac-vocab-version v7.0`" names a flag that does not
exist; the *conclusion* (refcv4b is non-kin3) is nonetheless correct.

⚠️ **The five banked local v4-design arms (`A_v3`, `B_v4_noguard`, `C_v4_drop`,
`D_v4_full`, `E_regress`) all ran `kin3` with `v7_labels: null`** — so the
tiny-rig ablation panel was on the **applied** side and is unaffected.

### ⛔ Can the magnitude of the loss be measured? NO — and the reason is decisive

The counterfactual "what would refcv4b have scored with the tactical feed on
under `v7.0`" is **not measurable, and not desirable**. The only implementation
that ever supplied that feed under `v7.0` was **provably wrong**: it fed 8-wide
heads to a `[B,3]×[B,3]` positional contract, reading `turn_left ←
LANE_CHANGE_L`, `turn_right ← LANE_CHANGE_R`, `accelerate ← YIELD_MERGE`,
`brake_stop ← FOLLOW`, with **10 of 16 classes never read** — and it ran that way
for all of refcv3 and refcv4's first 6,400 steps (`D-REFCV4-DEFECTA1`).

⇒ **The honest statement is: the arm lost a seam, not a capability that was
working.** Restoring "what was there" would restore a scrambled prior. Whether a
*correct* v7→5 carry is worth building is a separate, unpre-registered question
(§5).

⇒ ⭐ And the loss was **not** undocumented at programme level — `MODEL_REGISTRY.md`
§4.6 lever (5) and `…/2026-09-04-refcv4b-seam-state/SEAM_STATE.md` both state it.
**It was undocumented at RUN level**, which is the gap this work closes: prose one
document away is not a stamp on the artifact.

---

## 3. Is the prior even reachable? — YES, both branches are live launch paths

| | |
|---|---|
| `RefCV3Config.tac_vocab_version` class default | **`v7.0`** (PI v7 mandate) |
| what `_pin_trainer_cfg` pins | **`"v7.0" if --v7-labels else "kin3"`** |
| ⇒ `kin3` reachable? | ⭐ **YES** — it is what every launch *without* `--v7-labels` gets, including all five banked v4-design arms |
| ⇒ H19 ever dark? | ⛔ **NO.** `man_logits` is assigned on both branches of `factored_maneuver`, so `reweight` is never `None`. H19 goes dark only via `graft_maneuver=False`, which the vocabulary does not touch. |

⇒ **The cheaper/honest outcome the brief anticipated ("the fix is then a stamp and
a refusal, not a restoration") is the right one — but for a different reason than
expected.** Not because the prior is unreachable; because it is **always
reachable**, and the thing that changes is *who feeds it*.

---

## 4. The fix — STAMP FIRST, and it is proven two-sided

### 4.1 Landed (files I own, both clean in `git status` at edit time)

**`stack/tanitad/refs/refc_v3.py`**

* `h19_prior_stamp(model) -> dict` — **ONE PREDICATE, ONE CONSUMER**, reading the
  **BUILT OBJECT** (`model.tac_vocab_version`, the decoder's actual graft
  modules), never re-deriving a config condition. This is the pattern the file
  itself mandates for `tac_goal_tok_head` / `param_breakdown_v3`.
  Emits `h19_prior`, `h19_prior_source`, `h19_graft`, `h19_tac_vocab_version`.
* `cache["h19_tactical_feed"] = man5 is not None` **inside `_hook`**, one
  statement after the guard — a **bool**, in the same cache and the same style as
  the existing `ego_injected` / `nav_injected` / `goal_point_injected` bools, so
  it reaches `out` through the existing `out.update(cache)` and **no consumer
  meets a new value type**.

**`stack/tests/test_h19_prior_stamp.py`** — 9 tests, **9 passed**.

### 4.2 The two-sided proof (the acceptance criterion the brief set)

| build | `h19_prior` | `h19_tactical_feed` in `out` |
|---|---|---|
| `hier` + `kin3` | **`applied`** | **`True`** |
| `hier` + `v7.0` | **`dropped(tac_vocab_version='v7.0': derive_man5_logprobs is a [B,3]x[B,3] POSITIONAL contract and cannot read a 8-wide head (D-REFCV4-DEFECTA1))`** | **`False`** |
| flat | **`n/a(flat arm: no tactical cascade)`** | *(no hook, no flag)* |

⭐ `n/a` is stamped **distinctly from `dropped`** — a flat arm has no tactical
brain, so there is no feed to lose, and the two have different remedies.

### 4.3 Proven able to FAIL — source-level mutation, `raw/mutation_proof.txt`

| mutation | suite | wanted |
|---|---|---|
| **M0** pristine (CONTROL) | **PASS** | PASS ✅ |
| **M1** guard removed (`man5` always computed) | **FAIL** | FAIL ✅ |
| **M2** stamp hardcoded to `applied` | **FAIL** | FAIL ✅ |
| **M3** runtime flag hardcoded `True` | **FAIL** | FAIL ✅ |

**4/4.** ⭐ **M2 is the one that matters**: it reproduces exactly the sibling
defect shipped the same night — a stamp that reads identically on every arm — and
the suite refuses it. A stamp that cannot fail has measured nothing.

### 4.4 REACHED, not merely correct

`test_the_flag_reaches_a_real_forward[kin3|v7.0]` asserts the bool arrives in the
model's own output on an ordinary `model(frames, v0=…)` call — **no preflight, no
special entry point**. This is the deliberate answer to the sibling D-ROLL-1
finding that `refc_v3_train.main` runs `preflight` only under `--preflight` and
otherwise calls `train()` directly: a guard wired into `preflight` alone covers
one launch path of two. `test_the_runtime_flag_and_the_record_stamp_cannot_
disagree` pins that the forward's bool and the record's string are two *readouts*
of one guard, not two copies of one condition free to drift.

### 4.5 Escalated, NOT applied — `ESCALATION_refc_v3_train.md`

`stack/scripts/refc_v3_train.py` was **`MM`** (staged *and* dirty — a sibling has
it live) for this entire turn, so the `config.json` / preflight stamp is filed as
an **exact two-site diff** rather than edited. Both sites are named because the
trainer has two launch paths.

---

## 5. Restoration — deliberately NOT attempted, and why that is the finding

The brief authorised a flag, default OFF, bit-identical when off. ⛔ **I did not
build one, and building one would have been the wrong call**, on the brief's own
"evidence, not sentiment" rule:

1. **There is nothing correct to restore.** The v7→5 carry has never existed in a
   correct form; the only implementation was the scrambled positional read.
2. **Inventing one is a new, unvalidated mapping** — precisely what
   `D-REFCV4-DEFECTA1` chose option (a) to *avoid* ("no invented 8→5 mapping is
   shipped"). It would need its own pre-registration and its own bar.
3. **`refc_tactical.py` already refuses loudly** when a wider head reaches
   `derive_man5_logprobs` — the "refuse at launch rather than drop silently" half
   of the brief's remedy is **already implemented**; what was missing was only the
   record, which is now stamped on both the forward path and (escalated) the
   config.
4. ⚠️ A `v7 → kin3` projection **already exists** in `refc_tactical.py` (the "PI
   mandate 2026-08-27" block) and is the obvious substrate for a correct carry —
   **but wiring it into the prior changes the recipe of a live arm**, which the
   brief forbids without authorisation. Named as the next lever, not taken.

**⇒ Next lever, blocked on a PI/pre-registration decision, not on compute:**
carry the tactical brain into the anchor prior under `v7.0` by projecting the
8-wide lat / 8-wide lon posteriors through the existing `refc_tactical` v7→kin3
map, behind a flag defaulting OFF, proven bit-identical when off by
`torch.equal`, pre-registered with both outcomes committed. **Cost: zero GPU to
implement, one tiny-rig A/B to read.** It is not free: it re-opens the tactical
seam that `D-REFCV4-DEFECTA1` closed, so it needs a bar, not a preference.

---

## 6. Suite — CONTROLLED comparison

`test_refc_v3.py`, `test_refc_v3_rollability.py`, `test_refc_v4.py`,
`test_refc_tactical.py`, `test_refc.py`, `test_tac_goal_wiring.py`,
`test_refcv3_ablations.py`, `test_refc_v3_lan_preflight.py`,
`test_h19_prior_stamp.py`:

**170 passed, 2 failed, 1 skipped** —
`test_refcv3_ablations.py::test_each_ablation_moves_the_record[e7_off]` and
`::test_the_ablation_stamp_reaches_the_manifest_and_every_arm_block`, both
`taniteval/tools/refcv3_arm.py:1036: TypeError: 'NoneType' object is not
callable`.

⭐ **CONTROL RUN, same two tests, same environment, with `refc_v3.py` replaced by
`git show HEAD:…` (my edit removed): the SAME 2 tests fail with the SAME
traceback.** ⇒ **PRE-EXISTING, not mine** — the `refcv3_arm` rollability
regression a sibling owns and which the brief instructs me not to touch. My files
alone: **9/9**.

---

## 7. Deliverable manifest

| artifact | where |
|---|---|
| `h19_prior_stamp()` + `H19_DROP_REASON` + runtime flag | **repo** `stack/tanitad/refs/refc_v3.py` |
| 9 tests, two-sided + reachability | **repo** `stack/tests/test_h19_prior_stamp.py` |
| this report | **repo** `…/Research/2026-09-06-h19-prior/H19_PRIOR.md` |
| exact 2-site diff for the contended trainer | **repo** `…/2026-09-06-h19-prior/ESCALATION_refc_v3_train.md` |
| blast-radius probe + its output | **repo** `…/raw/h19_blast_radius.py`, `raw/h19_blast_radius.txt` |
| mutation proof + its output | **repo** `…/raw/run_mutation_proof.py`, `raw/mutation_proof.txt` |

⛔ Nothing is stranded on a pod or in a worktree. Nothing was committed to `main`;
nothing was pushed.

---

## 8. The one line

⛔ **No — a run can no longer drop the H19 tactical feed without saying so on the
forward path** (`out["h19_tactical_feed"]`, emitted on every hier forward,
mutation-proven to fail if hardcoded); **but it still can in `config.json` until
the escalated two-site diff is applied to `refc_v3_train.py`**, which is a
sibling's file and was `MM` throughout this turn.
