# GOAL-PROVENANCE AUDIT — the PI's admissibility check, computed and gated

**Date** 2026-08-18 · **Branch** `agent/arch-inf-20260803` · **Tier** T-NA (a
structural/graph audit — not an eval tier, so no T0/T1 stamp applies and none is
claimed) · **GPU** none (CPU only; dev-box RTX 4060 untouched, Thor untouched)

**The ruling this serves** (`CLAUDE.md`, Sayed 2026-08-03): a goal/route signal
is admissible at inference; the **output of the situation classifier is not, in
any form**. The mandated check — *"could this have been computed from the
situation classifier's output?"* — was, until this turn, **a question someone had
to remember to ask**.

---

## 0. Headline

| # | finding | class |
|---|---|---|
| 1 | **The instrument the backlog calls unbuilt was already built — and had ZERO call sites.** `goal_admissibility.py` (242 lines, 12 tests) is imported by nothing in `stack/` or `taniteval/`. Flagged as item 5 of the 2026-08-16 stale-blocker sweep, **open 12 days**. | MEASURED |
| 2 | **Its provenance clause is DECLARED, not computed** — `situation_disjoint` intersects two hand-written lists of symbol *names*. That is the C112/C113 class exactly: a non-overlap assumed from provenance. | MEASURED |
| 3 | ⭐ **The existing gradient probe is STRUCTURALLY BLIND to this question.** `V6Stack.assert_isolation` measures **backward** edges; admissibility is a **forward** question. `V6Stack.forward` routes every downward goal port through `detach()`. **A leak spliced in there carries the full signal and zero gradient — certified clean by the gradient probe.** Demonstrated on a wired arm, not argued. | MEASURED |
| 4 | ⭐ **Reverse direction: CLEAN, and now checked for the first time.** The situation classifier's entire input vector is `sub["img"]` — `sitclf_train.py:160`. No goal/route/nav symbol exists anywhere in its path. | MEASURED |
| 5 | ⭐ **`v6.py:62`'s prose claim is now VERIFIED BY COMPUTATION**: no goal **head** reads `v0` or `actions` on any of 6 arms; `v0` reaches exactly one node, the unicycle `emission`. | MEASURED |
| 6 | **Shared trunk: shared CLASS, not shared instance.** The situation substrate is `WorldModel.encode()` from a **frozen v1 flagship4b** checkpoint; v6 trains its own encoder. Common ancestor (vision), no path. | MEASURED |
| 7 | ⚠️ **Two live defects found in passing** — a substrate schema mismatch that means `sitclf_train.py` has only ever run on fixtures, and a default arm name that points at an **inadmissible ego arm**. §8. | MEASURED |
| 8 | ⚠️ **My own instrument failed twice during construction** — an inert input probe, and a dead-node rule that called disjointness an artefact. Both caught, both pinned by regression tests. §6 — recorded because the root-cause classes are the point. | MEASURED |
| 9 | ⚠️ **`stack/scripts/goal_provenance.py` ALREADY EXISTS** — same basename, different question (goal SOURCE: oracle-GT-future vs produced-from-vision). Found via `GATE_PROTOCOL.md:186` **after** I had probed `tanitad/eval/` and concluded the name was free. Absence at one location, again. §7. | MEASURED |

**Delivered:** the instrument (`goal_provenance.py`), 17 unit tests, **11
real-model gate tests that run under `pytest`**, and a CLI gate with exit codes.
The rule now has a mechanism.

---

## 1. What the goal path actually reads — from source

`V6Stack.forward` (`stack/tanitad/models/v6.py:4650`) accepts **exactly three
inputs**: `frames`, `actions`, `v0` (plus `own_frames_tac`/`own_frames_str`,
refused unless `shared_encoder=False`). The goal path, traced line by line:

| step | file:line |
|---|---|
| `z_op_win = self.encode_window(frames)` | `v6.py:4670` |
| `z_op = z_op_win[:, -1]` | `v6.py:4672` |
| `z_tac, _ = self.uplink_tac(z_op, own_tac)` | `v6.py:4685` |
| `z_str, _ = self.uplink_str(z_tac, own_str)` | `v6.py:4686` |
| `z_str_p = self._cut(z_str, cut)` | `v6.py:4700` |
| `g_str = self.goal_head_str(z_str_p)` | `v6.py:4703` |
| `e_g_str = self.cond_tac(g_str["probs"], g_str["args"])` | `v6.py:4705` |
| `g_tac = self.goal_head_tac(z_tac_p, cond=e_g_str)` | `v6.py:4708` |
| `e_g_tac = self._encode_goal(self.cond_op, g_tac)` | `v6.py:4726` |
| `plan = self.emit(z_plan, e_g_tac, v0, roll_ctx=roll_ctx)` | `v6.py:4790` |

**There is no situation-classifier port.** `V6Stack.named_children()` is
`encoder, readout, vocab_*, predictor_*, step_readout_op, cond_op, adapter_*,
goal_head_tac, act_head_*, goal_head_str, plan_proj, cand_queries, emission,
masked_cells, sigreg` — no situation module (MEASURED).

The module says as much in prose at `v6.py:51-55`, `:1641-1644`, `:1807-1810`.
**Prose is what this audit exists to replace**, so the claim was recomputed.

---

## 2. ⭐ Why a gradient probe cannot answer this — the discriminating result

`V6Stack.assert_isolation` (`v6.py:4956`) is a genuine autograd probe over the
X3 matrix. It is a good instrument and it is **pointed at a different question**:

| | `assert_isolation` | `goal_provenance` (new) |
|---|---|---|
| direction | **backward** (gradient) | **forward** (information) |
| answers | *what TRAINS what* | *what is READ at inference* |
| sees `detach()`? | **NO — a detached wire is invisible** | **YES — sees the value** |

This matters here specifically because `V6Stack.forward` is *saturated* with
`self._cut()` = `Tensor.detach()` (`v6.py:4341-4342`), applied at `:4698-4700`,
`:4750`, `:4757`, `:4766`, `:4773`. **Every downward goal port is detached.** A
situation-classifier output spliced in behind one of them would leak completely
and pass the gradient probe.

**MEASURED** on the same graph and batch
(`tests/test_goal_provenance.py::test_the_gradient_probe_MISSES_what_the_forward_probe_CATCHES`):

```
forward_information_path : True
backward_gradient_path   : False
probes_disagree          : True
```

⇒ The two instruments are **complementary, not redundant**. Neither supersedes
the other, and `assert_isolation` should not be read as covering admissibility.

---

## 3. The measurement — per arm, per input

`stack/scripts/audit_goal_provenance.py`, artifact
`goal_provenance_audit.json`. Six pre-registered `V6Config` arms, each with its
**own** positive control (C107: a control run once for a study left 33 of 165
rows with no control).

**Which goal nodes move when each input is intervened on** (`✔` = information
path measured; blank = bit-identical under all 5 perturbation kinds):

| arm | `frames` → heads | `actions` | `v0` |
|---|---|---|---|
| `default` | ✔ str, cond_tac, tac, cond_op | — | `emission` only |
| `planner-cut-off` | ✔ same | — | `emission` only |
| `uplink-cut-off` | ✔ same | — | `emission` only |
| `factored-goal` | ✔ str, cond_tac, tac, **lat/lon pair** | — | `emission` only |
| `goal-cat-args` | ✔ same | — | `emission` only |
| `tac-goal-cond` | ✔ same | — | `emission` only |

**Three readings, all MEASURED:**

1. **Every goal HEAD is a function of `frames` alone.** Vision-only holds for
   the goal decision on all six arms.
2. **`actions` reaches no goal node at all.**
3. **`v0` reaches exactly one node — `emission`, the unicycle.** This
   **verifies by computation** `v6.py:62`'s prose *"`v0` (initial speed) enters
   ONLY the unicycle"*. It is the single allowlisted non-vision edge
   (`EXPECTED_NON_VISION`), it names the exact `(source, target)` pair, and it
   carries the line that authorises it — an allowlist is where a violation
   hides, so it is pinned by a test against widening.

**Positive control: fired on all 6 arms.** The control splices a
situation-classifier stand-in into `goal_head_tac` **behind a `detach()`** —
the exact shape a real violation would take here. Without it firing, every clean
reading above would be unpowered (C109).

### ⚠️ One arm is UNPOWERED, and the gate says so

`tac-goal-cond`: **`cond_tac_dyn` is CONSTANT** — no perturbation of any input
moved it. That is the F-1 port, a **zero-init** `Linear` (`v6.py` F-1 note), so
on a freshly built model its output is identically zero and it reads as
independent of everything. **Disjointness is NOT established for that node**, and
the gate exits **3**, not 0. `--allow-unpowered` records the known state
deliberately.

This is the honest limit of a structural audit on an untrained model, stated
rather than buried: **a dead-valued path reads as absent.** The same caveat
applies to `emission`'s vision input (zero-init CV warm start, `v6.py:4534`),
which is why `emission` does not move under `frames` here.

---

## 4. ⭐ The reverse direction — checked for the first time

The ruling says *information-disjoint*, which is **symmetric**. Every check
written before this turn asked only *goal ← situation*.

**Answer: clean, by construction and by data availability.**

| evidence | file:line | class |
|---|---|---|
| the model input **is** `sub["img"]`, and nothing else | `sitclf_train.py:160` | MEASURED |
| substrate load reads only `("img","y","valid","clip_id")` | `sitclf_train.py:106` | MEASURED |
| declared `"inference_inputs": ["image_features"]` | `sitclf_train.py:194` | MEASURED |
| declared `"ego_at_inference": False` | `sitclf_train.py:197` | MEASURED |
| final flat vector = `flatten(img[t-7..t])`, one modality | `sitclf.py:60-85` | MEASURED |
| labels read **only poses** | `emit_situation_labels.py:53-58` | MEASURED |
| grep for `nav_cmd\|goal\|route\|g_str\|g_tac\|anchor\|waypoint` over the whole situation path → **4 hits, all prose comments** | — | MEASURED |

⚠️ **Evidence-class note.** The situation-path sweep was produced by a subagent,
which makes it **INHERITED** to me. The four load-bearing citations above
(`:106`, `:160`, `:194`, `:197`) were therefore **re-read first-hand** before
publishing and are stamped MEASURED on that basis; the remaining rows in this
table are INHERITED-but-corroborated by the structural gate in
`test_the_structural_half_passes_and_scans_a_nonzero_surface`, which recomputes
the absence over 39 sources on every `pytest` run.

And it is closed by **data availability**, not just discipline —
`sitclf_deploy.py:657-663` declares `STRATEGIC: UNAVAILABLE`, *"no route/goal/map
label exists on PhysicalAI-AV"*. There is no route signal in existence for the
situation path to read.

**The disjointness is already enforced elsewhere in the goal direction**, which
is worth recording because it shows the discipline is real:
`ph0_v2.py:688-689` strips `situations` from the record;
`e_ag1_anchor_floor.py:59-64` forbids the import, pinned by
`test_no_situation_classifier_path`; `test_v6_factored_goal.py:844-845` asserts
`"tanitad.data.situations" not in src`; `s2_labels.py:204` carries
`_DISJOINT_NEEDLES = ("situation", "sitclf")`.

---

## 5. ⭐ The shared trunk — disclosed, and why it is not a back door

The ruling requires this statement explicitly, and **"a shared trunk" was the
most likely honest finding**. It is subtler than expected:

**There is no shared trunk *instance*.**

| | goal path (v6) | situation path |
|---|---|---|
| encoder | `self.encoder = _Enc(cfg.encoder)` `v6.py:3925` | `WorldModel.encode` `fourbrain.py:470` |
| readout | `SpatialGridReadout` `v6.py:3927` | `SpatialGridReadout` `fourbrain.py:412` |
| weights | **trained live by v6** | **FROZEN v1 `flagship4b-speedjerk-30k`** ckpt, step 29999 |

Shared: the **classes** `ViTEncoder` / `SpatialGridReadout`, and the 2048-d
geometry. **Not shared:** the instances, the weights, or any runtime tensor. No
cross-import exists (MEASURED).

**The statement the ruling asks for.** Both paths descend from *vision*. That
makes vision a **COMMON ANCESTOR**, not a back door, and the distinction is
mechanical rather than rhetorical: a back door requires an information path
**from** the classifier's output **into** the goal, and the interventional probe
shows there is none — intervening on a situation output leaves every goal node
bit-identical. Covariation through a shared ancestor is not a path, and both
paths are independently entitled to read vision. `classify_edge()` returns
`DIRECT_PATH` / `COMMON_ANCESTOR` / `INDEPENDENT` precisely so this cannot be
fudged; a **correlational** provenance test could not tell the two apart, which
is why the instrument is interventional.

⚠️ **The conditional that is NOT yet true, and must not rot into a claim.**
`tactical.py:186-192` already contemplates a classifier trained as a read-only
head off the same trunk. That stays admissible **only** while information flows
trunk→head. **Coupling a classifier gradient into a shared UNFROZEN trunk
(stage 2+) needs a fresh admissibility argument** — and at that point
`assert_isolation` becomes the relevant instrument again, alongside this one.

---

## 6. ⚠️ The instrument failed twice while being built — root-cause classes

Recorded because the classes are the lesson, per `RETRACTION_LOG` doctrine. Both
happened **inside the instrument built to prevent exactly these failures**.

### 6.1 The input probe shipped INERT (C109 class)

`module_runner` perturbed an input, recorded it as the node's value, and then ran
`model(**batch)` — **the unperturbed batch**. Every input read as *unread*,
including `frames`. A vision-derived goal path that does not depend on vision is
impossible, and **that implausibility is what caught it** — not the positive
control, which covered only the submodule-intervention path.

⇒ **Root-cause class: a positive control powers ONE code path, not the module.**
Fixed at `goal_provenance.py:221` with the reason in-line, pinned by
`test_the_INPUT_probe_has_its_own_positive_control` — a deliberate tautology (a
head fed only `x` MUST depend on `x`, so `False` can only mean an inert probe).

### 6.2 The dead-node rule called DISJOINTNESS an artefact

The dead-node guard (§3) first judged deadness from the goal/situation matrix
alone. But **in a genuinely disjoint pair neither node moves the other by
definition**, so both read as constant and the instrument reported its own
success as UNPOWERED. It passed in isolation and failed in the full suite only
because the rule was added after the last green run.

⇒ **Root-cause class: a "nothing moved it" test is only meaningful against a
source that SHOULD move it.** Deadness is now judged against the shared-input
probe. Pinned in both directions:
`test_a_DISJOINT_pair_is_not_mistaken_for_a_DEAD_one` and
`test_a_genuinely_constant_goal_node_IS_flagged_dead`.

### 6.3 ⚠️ And the exit code lied, on cue

The first full-suite run was reported as **"exited with code 0" while one test
had FAILED** — the pipeline's exit status was `tail`'s, not `pytest`'s. This is
the standing *"exit codes are not evidence"* rule reproducing itself in the same
turn it was quoted. The suite was re-run writing pytest's own `$?` to the log
before any pipe. **Never read a piped suite's exit status.**

---

## 7. ⚠️ `goal_provenance.py` already existed — under `scripts/`

`stack/scripts/goal_provenance.py` (2026-07-26, 5,956 B) shares this module's
basename and answers a **different** question: *was the evaluated goal an oracle
read off the ego's own future, or produced from vision?* It is cited at
`GATE_PROTOCOL.md:186` and stamped into gate cards
(`Gates/flagship-v4-30k.card.json:15`).

I probed `tanitad/eval/` and concluded the name was free. **That is the
absence-at-one-location rule, and I broke it.** It surfaced only because a grep
for an unrelated purpose hit `GATE_PROTOCOL.md`.

**Resolution:** kept the name — it is accurate, the two live in different
namespaces, and this module is always imported fully qualified as
`tanitad.eval.goal_provenance`. Both files now carry a cross-reference. The real
hazard was in my test, which put `stack/scripts` on `sys.path[0]` — global
session pollution that would let any later bare `import goal_provenance` resolve
to the script. **Replaced with `importlib` load-by-path; `sys.path` is untouched.**

⇒ The two are complementary and the distinction is worth keeping straight:
`scripts/goal_provenance.py` asks **where the goal came from**;
`tanitad/eval/goal_provenance.py` asks **what the goal path is allowed to read**.

---

## 8. Two live defects found in passing

Neither is caused by this work; both are worth a work item. (Found by the
situation-path source sweep in §4.)

1. ⛔ **`sitclf_train.py` cannot read any banked substrate.**
   `load_substrate` requires `("img","y","valid","clip_id")`
   (`sitclf_train.py:106`, hard-fails `:107-108`), but the only real substrate
   builder emits `F=`, `Y=`, `V=`, `E=`, `clip_cluster=`
   (`…/2026-08-03-sitclf-matched-capacity/build_substrate.py:170-171`).
   `img ≠ F`, `clip_id ≠ clip_cluster`. An exhaustive grep for a producer of
   `img=`/`clip_id=` finds **only the test fixture**
   (`tests/test_sitclf_train.py:43`). **INFERRED consequence:** the promoted
   trainer has only ever run against synthetic fixtures.
2. ⚠️ **The deploy module's defaults name an inadmissible arm.**
   `DEPLOYED_ARM = "head_img_ego"` (`sitclf_deploy.py:59`) and `fused_name`
   default `"late_fuse(head_img, head_ego)"` (`:572`) both name **ego-reading**
   arms, which the same file's contract block (`:70-79`) declares closed under
   the vision-only ruling. `is_vision_only` (`:88`) enforces at runtime, so this
   is **naming drift, not a live leak** — but a caller using
   `four_family_report`'s defaults will label output with the forbidden arm.

---

## 9. What this does NOT establish

- It certifies **the graph built by these forwards on these batches** — the same
  evidence class `assert_isolation` states for itself, and stated in the same
  words on purpose. It is not a proof over all inputs.
- **Untrained weights.** Zero-init heads (`cond_tac_dyn`, `emission`'s final
  layer) are dead-valued and read as independent. The gate reports them
  UNPOWERED rather than clean; re-running against a trained checkpoint would
  convert those rows from UNPOWERED to MEASURED.
- The probed geometry is small (d_model 32, depth 1). **Wiring is topological**
  — a leak is an edge, and an edge is present or absent at any width — but this
  is an argument, not a measurement, and is labelled as such.
- **No claim about eval metrics.** No ADE, no four-family table: this is not an
  eval and does not pretend to be one.

---

## 10. Deliverable manifest

| artifact | path | state |
|---|---|---|
| **the instrument** | `stack/tanitad/eval/goal_provenance.py` | repo, **staged** |
| its unit tests (**17**) | `stack/tests/test_goal_provenance.py` | repo, **staged** |
| **the real-model GATE (11 tests)** | `stack/tests/test_goal_provenance_v6.py` | repo, **staged** |
| the CLI gate / runner | `stack/scripts/audit_goal_provenance.py` | repo, **staged** |
| measured artifact | `…/incoming/2026-08-18-goal-provenance-audit/goal_provenance_audit.json` | repo, **staged** |
| this report | `…/incoming/2026-08-18-goal-provenance-audit/GOAL_PROVENANCE_AUDIT.md` | repo, **staged** |

Nothing is on a pod, a worktree, or in an agent's context only.

**Reproduce:**
```
PYTHONUTF8=1 PYTHONPATH=stack python stack/scripts/audit_goal_provenance.py
PYTHONUTF8=1 PYTHONPATH=stack python -m pytest stack/tests/test_goal_provenance.py \
                                               stack/tests/test_goal_provenance_v6.py -q
```
Exit codes: **0** clean · **2** violation · **3** unpowered · **4** structural.

### Verification status — stated exactly

| what | result | class |
|---|---|---|
| `tests/test_goal_provenance.py` | **17 passed**, pytest's own `$?` = **0** | MEASURED |
| `tests/test_goal_provenance_v6.py` | **11 passed**, pytest's own `$?` = **0** | MEASURED |
| `scripts/audit_goal_provenance.py` | exit **3** (the known zero-init `cond_tac_dyn`), **0** with `--allow-unpowered` | MEASURED |
| **full `stack/` suite** | ⚠️ **NOT re-confirmed green in this turn** — see below | — |

⚠️ **The full suite did not complete.** An earlier full run finished in 152 s at
**839 passed / 1 failed**, the single failure being this work's own dead-node
regression (§6.2), which is now fixed and pinned. The re-run after the fix was
still at ~62 % when this turn ended: it was competing with another agent's live
2.5 GB job in the same tree (`stack/tanitad/models/v6.py` and
`train_v6_staged.py` are both modified-unstaged by a concurrent stream), i.e.
**exactly the CPU-contention condition the standing rule says not to run suites
under**. ⇒ **Whoever commits must re-run `pytest -q` on a quiet box first.** The
two new files were verified separately and are green; that is not the same claim
as the suite being green, and it is not presented as one.

⚠️ **And read the exit code from pytest, not from a pipe** — §6.3.

---

## 11. ⇒ ESCALATIONS (integration, not a "please merge" in a doc)

1. ⛔ **`goal_admissibility.py` still has zero call sites.** This turn adds a
   second instrument that *is* invoked; the older one is not. Either wire it
   into `taniteval` before publishing any goal-signal number, or record it as
   superseded for the provenance clause. **It should not sit at zero a 13th day.**
2. ⛔ **Wire the gate into the trainer preflight.** It is CPU-only and runs in
   ~6 s for 6 arms. The natural home is beside `assert_isolation` in
   `train_v6_staged.py`'s dry-run, so an inadmissible wiring fails in seconds
   rather than after a GPU-day. **This is the step that converts C108 into a
   mechanism, and I deliberately did NOT make that edit** — the trainer is owned
   by the training stream and a v6F S-W run is resuming; editing a live
   trainer's launch path without their sequencing is how a run gets lost.
   ⇒ **FILED as a work item** (`task_e8af17b7`), carrying the constraint that
   the two probes are complementary and both must be called.
3. **Re-run against a trained checkpoint** to convert `cond_tac_dyn` and
   `emission` from UNPOWERED to MEASURED. (Folded into the same work item.)
4. **The two defects in §8 have owners.** The substrate schema mismatch is the
   more serious — a trainer that has never run on real data —
   ⇒ **FILED** (`task_4f4fe064`), with the instruction to re-verify from source
   rather than trust this report's summary.

⚠️ Escalations 2–4 are **filed work items, not sentences in a document.** The
programme's own record is that an integration request living in a doc nobody
re-reads sits for 10–12 days; `goal_admissibility`'s zero-call-site state
(finding 1) is the same failure and is what this instrument was built to end.
