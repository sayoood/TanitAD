# D-ROLL-1 — every banked v7.0 checkpoint was unrollable for ~90 minutes; the construction is fixed, proven on the real weights, and proven to fail when reverted

**Date** 2026-09-06 · **Stream** Architecture & Inference (rollability regression) ·
**Evidence class MEASURED (ours)** · **0 GPU for the fix and every proof below** ·
**Surface** dev-box (x86_64, torch 2.11.0+cu128, CPU load), one process.
⛔ The A40 was not touched (it is reserved for refcv5).

---

## 0. One line

`refc_v3.py` built the D-TACGOAL-1 goal-token head on the **vocabulary alone**, adding
**11,286 params** to every rebuild of a checkpoint trained before that head existed; the head is
now **opt-in**, and **refcv4b@40284 plus all three local refcv3 checkpoints load again at
0 unexpected keys, with the recorded `param_breakdown` reproduced key-for-key**.

---

## 1. The defect, at source

`RefCV3Model.__init__` (commit `ee70bdb`, tonight):

```python
self.tac_goal_tok_head = None
if _vv != "kin3":                 # <- the vocabulary ALONE
    ...  TacGoalTokenHead(cfg.d_tac, n_tokens=len(_goal_toks))
```

`tac_vocab_version` is pinned to `"v7.0"` by `_pin_trainer_cfg` for **any** run launched with
`--v7-labels`, which is every current arm. So every rebuild grew a module the weights do not
contain, and `refcv3_arm.cross_check_config` — whose contract is *"every fact `config.json`
states about the model must hold for the rebuilt one"* — refused, naming both values:

```
[refcv3_arm] config.json CONTRADICTS the rebuilt model - refusing rather than guessing
  param_breakdown: config.json {... 'total': 107058488}
             vs rebuilt {... 'tac_goal_tok_head': 11286, 'total': 107069774}
```

⭐ **The guard was right. The construction was wrong.** Nothing here loosens a cross-check.

### 1.1 Blast radius — MEASURED, five records, none of which names the head

| record | `tac_vocab_version` | recorded `total` | carries `tac_goal_tok_head` line? |
|---|---|---|---|
| `refcv4b@40284` (`C:\Users\Admin\refcv4b_final\config.json`) | `v7.0` | 107,058,488 | **no** |
| `refcv3-rlmin@40284` | `v7.0` | 107,032,901 | **no** |
| `refcv3-ol@30000` | `v7.0` | 107,032,901 | **no** |
| `refcv3-viz@40284` | `v7.0` | 107,032,901 | **no** |
| ⛔ **`refcv5-ddim-b1-v72-40k` — the LIVE A40 run** | `v7.0` | 108,246,216 | **no** |

⛔⛔ **The live run is the part that was not in the escalation.** refcv5 rebuilds through this same
file on every supervisor relaunch, so the regression made the running A40 job **unresumable**, not
merely unrollable. That is also what settles the default: **ON would have broken it.**

---

## 2. The fix, and why this shape

`RefCV3Config.tac_goal_tok_head: bool = False`, and the gate becomes

```python
if _vv != "kin3" and bool(getattr(cfg, "tac_goal_tok_head", False)):
```

Three reasons this shape and not another:

1. ⛔ **`param_breakdown`'s contract is that the RECORD is the authority.** `cross_check_config`
   compares the recorded ledger against the rebuilt one and refuses on any difference. The only
   way a pre-drift checkpoint can rebuild is for the module not to exist — so the condition has to
   be something a pre-drift `argv` cannot switch on. A default-OFF config field is exactly that:
   absent from every recorded `argv`, therefore False on every rebuild.
2. ⭐ **It is the rule this config class already states twice.** `nav_args_inject`: *"Default OFF:
   no banked arm's recipe changes."* The v4 pins: *"BOTH DEFAULT FALSE … the live 40,284-step run
   resumes through this file, so a default that moved would silently change a training in flight."*
   D-TACGOAL-1 is the first module in this class to move a default, and it did so under a live run.
3. ⛔ **The head is NOT deleted.** It is constructed, exercised and tested exactly as the sibling
   built it — 22 sigmoids, multi-label, `mask_report`, `majority_control_scores`, all untouched.
   Only the *decision to build it* moved from the vocabulary to an explicit lever, which is what
   makes its capacity cost attributable at all.

⚠️ **The vocabulary remains NECESSARY.** `kin3` still refuses the head even when the flag is True —
pinned by `test_kin3_refuses_the_head_even_when_it_is_explicitly_asked_for`, which is a *stronger*
control than the sibling's original (that one could pass because the flag defaulted off; it now
asks for the head and is still refused, so it tests the vocabulary and nothing else).

---

## 3. ⛔⛔ THE BAR — real checkpoints, through the instrument's own strict loader

`refcv3_arm.load_model` (**not** a re-implementation — the function that refused before the fix),
`device=cpu`, one process. `raw/roll_loadmodel.json`, `raw/roll_proof.json`.

| checkpoint | step | **missing** | **unexpected** | rebuilt `total` | recorded `total` | cross-check | verdict |
|---|---|---|---|---|---|---|---|
| `refcv4b@40284` (md5 `99b573e8277d94a5e3bfbf630cb4d751`) | 40,284 | **0** | **0** | 107,058,488 | 107,058,488 | **PASS** | **ROLLABLE** |
| `refcv3-rlmin@40284` | 40,284 | 1 *(inert)* → **0** | **0** | 107,032,901 | 107,032,901 | **PASS** | **ROLLABLE** |
| `refcv3-ol@30000` | 30,000 | 1 *(inert)* → **0** | **0** | 107,032,901 | 107,032,901 | **PASS** | **ROLLABLE** |
| `refcv3-viz@40284` | 40,284 | 1 *(inert)* → **0** | **0** | 107,032,901 | 107,032,901 | **PASS** | **ROLLABLE** |

**`n_refused = 0` of 4.**

⚠️ **The single missing key on the three refcv3 rows is named, not glossed:** it is
`core.decoder.anchor_controls`, the **pre-existing, explicitly documented INERT-BUFFER tolerance**
banked 2026-09-05 (refcv4-b added the persistent buffer; `roll_bank` reads it only when the decoder
is `v0_conditioned`, which these are not). It **predates this regression**, it is what
`load_model` was already built to accept, and it is **recorded in the provenance, never silent**.
After that documented tolerance the count is **0 / 0 on all four**.

⭐ **Same-breath non-zero controls**, so a 0/0 cannot come from an empty read: 551 / 544 / 544 / 544
checkpoint tensors, and the returned model carries 107,058,488 / 107,032,901 live parameters.

---

## 4. ⛔ TWO-SIDED MUTATION PROOF — at the SOURCE, on the real refcv4b weights

`raw/run_mutation_proof.py` → `raw/mutation_proof.json`. It rewrites the gate in the **off-Drive
mirror only** (never the repo) and restores in a `finally` with a **shape-asserted** md5 compare.

| state | gate in source | `load_model(refcv4b@40284)` | `test_refc_v3_rollability.py` |
|---|---|---|---|
| **FIXED** | `if _vv != "kin3" and bool(getattr(cfg, "tac_goal_tok_head", False)):` | **LOADED** 0/0, total 107,058,488 | **GREEN** — 11 passed |
| **DEFECT** | `if _vv != "kin3":` | ⛔ **REFUSED** — `tac_goal_tok_head: 11286`, total 107,069,774 | ⛔ **RED** — 1 failed |

`md5` fixed `7aeed46252c26876c1f9e769028a997f` · defect `55ecdab097a86f8837013e72ddc82b95` ·
**RESTORE VERIFIED** back to the fixed hash. **`TWO_SIDED_PROOF: PASS`.**

⭐ A guard that cannot fail measures nothing, and a fix that cannot be shown to fail is not
evidenced. Both halves were run.

---

## 5. ⭐ THE PATH IS *REACHED*, not merely correct

MEASURED tonight and cited in the brief: `refc_v3_train.main` runs `preflight` **only under
`--preflight`** and otherwise calls `train()` directly, so a guard wired into `preflight` alone
covers **one launch path of two**. Correctness and wiring are different claims. Two tests, in
`stack/tests/test_refc_v3_rollability.py`:

* **`test_the_head_has_exactly_one_construction_site_and_it_is_gated`** — *structural*. AST over
  the whole `stack/tanitad/` tree: `TacGoalTokenHead` is instantiated in **exactly one** place, and
  that place is inside `RefCV3Model.__init__`, under the gate's exact text. ⇒ **there is no second
  path to cover.** ⛔ The scan asserts it **READ** its files (`scanned > 50`, `unreadable == []`) —
  0 hits from a mount that could not open a file is a claim about the SEARCH, not the content.
* **`test_every_config_production_path_reaches_the_gate`** — *empirical*. (a) AST proves both
  `preflight` and `train` build their config through **one** helper, `_pin_trainer_cfg`; (b) a
  counter on `TacGoalTokenHead.__init__` then measures the gate on a config **that helper actually
  produced** from a real recorded `argv`: **0 heads by default, exactly 1 when the flag flips.**
  The second half is the control — a counter that never increments would report "not reached" for
  a gate that is simply never exercised.

⚠️ **A third construction site exists and is named rather than hidden:** `_lan_arm_preflight` builds
a `RefCV3Model` **without** `_pin_trainer_cfg`. It is unaffected here precisely because the gate
lives in `__init__` rather than in a launch path — which is the whole argument for putting it there.

---

## 6. ⭐ How a SECOND stale consumer was prevented

The root-cause class the brief names: *the eval arm's comment claimed it computed `a_star` "exactly
as the trainer computes it" — true until the trainer was corrected on 2026-09-04, never updated
since.* **Two consumers, one convention, one stale.** A fix that introduces a build condition
creates exactly that risk, because a capacity ledger is the natural place to re-derive it.

**It was prevented by not creating the second copy.** `param_breakdown_v3` reports the line by
reading the **BUILT OBJECT** —

```python
if getattr(model, "tac_goal_tok_head", None) is not None:
    out["tac_goal_tok_head"] = cnt(model.tac_goal_tok_head)
```

— and **never** re-derives `_vv != "kin3" and cfg.tac_goal_tok_head`. The ledger cannot disagree
with the constructor because it is not making an independent judgement; it is reading the result of
the constructor's. Pinned by **`test_ledger_line_tracks_the_built_object`** over all four
(vocabulary × flag) combinations:
`("tac_goal_tok_head" in breakdown) == (model.tac_goal_tok_head is not None)`, plus the
sum-equals-total invariant in **both** states. If anyone ever adds the second copy, that test is
what fails.

---

## 7. Test fallout — every sibling assertion preserved, suite green

Two sibling files pinned the unconditional construction and went red (7 + 1 failures). Both were
updated **minimally**: only the BUILD states the lever; **not one assertion was changed, relaxed or
removed**, and one control was **strengthened**.

| file | change | result |
|---|---|---|
| `stack/tests/test_tac_goal_wiring.py` | `_model()` sets `cfg.tac_goal_tok_head = True`; the `kin3` control now **asks for the head anyway**, so it tests the vocabulary rather than the default | 17 passed |
| `stack/tests/test_refc_v3.py` | a local `_tg = lambda c: dataclasses.replace(c, tac_goal_tok_head=True)` on the four rung builds that price the head; **added** a control asserting the DEFAULT build carries neither head nor ledger line | 20 passed |
| `stack/tests/test_refc_v3_rollability.py` (**new, mine**) | 11 tests: the contract, the vocabulary refusal, the ledger/object pin, the three real banked records, both reachability tests | 11 passed |

**`test_refc_v3.py` + `test_refc_v4.py` + `test_tac_goal_wiring.py` + `test_refc_v3_rollability.py`
+ `test_goal_point_trainer_flags.py`: 103 passed.**

⭐ `test_banked_records_rebuild_to_their_recorded_param_breakdown` carries the **real recorded
`argv` and `param_breakdown`** of refcv4b, refcv3-b1-v72 and the **live refcv5** as fixtures, and
rebuilds each through the trainer's own `build_parser` + `_pin_trainer_cfg`. It is the rollability
bar, in CI, **without the 1.3 GB weights** — so this regression cannot recur silently.

---

## 8. ⛔ ESCALATION — the trainer opt-in flag (I do not own `refc_v3_train.py`)

New arms need a way to switch the head on. The config field exists; **the CLI flag does not**, and
`stack/scripts/refc_v3_train.py` is a sibling's file. Exact diff, to be applied by its owner:

```diff
@@ in build_parser() @@
+    ap.add_argument("--tac-goal-tok-head", action="store_true",
+                    help="D-TACGOAL-1: build the 22-token tactical-goal SET head "
+                         "(+11,286 params at d_tac 512). Requires --v7-labels; "
+                         "kin3 has no tactical goal vocabulary and refuses it. "
+                         "OFF by default so a banked checkpoint rebuilds with the "
+                         "parameter set it was trained with (D-ROLL-1).")

@@ in _pin_trainer_cfg(cfg, args) @@
+    # D-ROLL-1: OPT-IN, recorded in argv. Building this head on the vocabulary
+    # alone made refcv4b, three refcv3 checkpoints and the LIVE refcv5 run
+    # unrollable, because its 11,286 params are absent from every recorded
+    # param_breakdown and cross_check_config correctly refuses.
+    if getattr(args, "tac_goal_tok_head", False):
+        cfg.tac_goal_tok_head = True
```

⚠️ **Until that lands, the head can only be switched on programmatically** (a config object, or
`config.json`'s `refcv3_arm_model_cfg` adapter extension) — it cannot be switched on from a
training command line, so **no new arm can accidentally train it, and no banked arm can lose
rollability.** That is the safe failure direction.

---

## 9. `C3_os_reproduction` — can it be re-run?

⭐ **YES — the instrument-level blocker is gone.** `refcv4b@40284` loads 0/0 on this box in one
process, which is precisely what could not happen 90 minutes ago.

⚠️ **And one claim in the escalation that unblocked this work is superseded, which must be said
rather than quietly dropped.** `D-BANK-TEMP-1a` (appended earlier today) MEASURED
`C3_os_reproduction` **PASS** on the banked A40 dump — measured 0.2975 / banked 0.2975 /
`abs_diff` 2.6e-05 against a 0.001 tolerance. ⇒ ⛔ *"both refcv4b number families are unquotable"*
is **too strong**: the **A40 family (`os` 0.2975, `os_navzero` 0.3928) IS quotable**; the **Thor
family (0.2965 / 0.3926) is not**, and the 0.001031 FAIL is the **cross-hardware** roll only
(A40 x86_64 / torch 2.8.0+cu128 vs Thor aarch64 / torch 2.13.0+cu130, argmax tie-breaks,
`n_distinct` 50 → 51).

⇒ **What is genuinely open is the cross-hardware disagreement, not the tool**, and the decisive
experiment is the one the brief specifies: **every arm of the comparison rolled on ONE surface in
ONE process.** All three inputs are now local — the 141-clip eval cache
(`C:\Users\Admin\tanitad-data\refav1-eval141\eps`, 14 GB / 142 files), the v7.2 eval labels, and
the B1 lead block — so a dev-box roll is a **third independent surface**, and if it reproduces
0.2975 within tolerance it settles the disagreement in favour of the A40 family.

**Cost probe status:** priced with the RUNBOOK's own 2-clip probe on the free 4060 (the A40 is
reserved and was not touched); see §11.

---

## 10. Evidence-class discipline

| claim | class |
|---|---|
| the five records' `param_breakdown` and vocab | **MEASURED (ours)** — each run's own `config.json` |
| the 0/0 load counts | **MEASURED (ours)** — `raw/roll_loadmodel.json`, real weights |
| the two-sided mutation | **MEASURED (ours)** — `raw/mutation_proof.json` |
| refcv4b `os` 0.2975 / `os_navzero` 0.3928 | **INHERITED** (D-BANK-TEMP-1a) — not re-measured here |
| best-in-fan 0.1993 m vs `os` 0.2970 m | **INHERITED** (D-SELQ-ASTAR-10) — and an **EMPIRICAL ceiling, never a mathematical bound**: `a_star` minimises summed squared error, ADE is a different norm |

⛔ **Vocabulary:** the nav command is an **INPUT simulating the vehicle's nav system**, never an
oracle. ⛔ **No label set was emitted, rebuilt or regenerated** — the PI's standing prohibition.
⚠️ `H-ESTIM-SEED-1` does not bite on anything in §3–§7: these are **structural identities**
(a module exists or it does not; a key count is 0 or it is not), not differences between trained
arms, so no replicate arm is owed.

---

## 11. Deliverable manifest

| artifact | where it lives |
|---|---|
| the construction fix | **repo** `stack/tanitad/refs/refc_v3.py` (md5 `7aeed46252c26876c1f9e769028a997f`) |
| the new test suite (11 tests) | **repo** `stack/tests/test_refc_v3_rollability.py` |
| sibling test updates | **repo** `stack/tests/test_tac_goal_wiring.py`, `stack/tests/test_refc_v3.py` |
| this report | **repo** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-rollability/RESULT.md` |
| real-checkpoint proof (both harnesses + JSON) | **repo** `…/2026-09-06-rollability/raw/` |
| two-sided mutation harness + JSON | **repo** `…/2026-09-06-rollability/raw/run_mutation_proof.py`, `raw/mutation_proof.json` |
| C3 cost probe | **repo** `…/2026-09-06-rollability/raw/` (see §9) |
| off-Drive mirror used to run | `C:\Users\Admin\tanitad-wt\` (disposable; restored + md5-verified after the mutation) |

⛔ Nothing is stranded on a single disk: every artifact above is in the repo working tree and
staged.
