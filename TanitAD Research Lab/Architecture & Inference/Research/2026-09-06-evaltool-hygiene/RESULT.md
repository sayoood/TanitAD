# Eval-tool hygiene — `taniteval/tools/refcv3_arm.py`

**Date** 2026-09-06 · **Stream** eval-tool hygiene · **Compute** ZERO GPU (dev-box CPU only;
the A40 was not touched) · **Branch** `agent/arch-inf-20260803`

Two defects were escalated by the dump-banking stream, which owned no code file. Both are
fixed here, with pinning tests. A **third** defect — a live crash in the same file — was found
while establishing the test baseline and is fixed too.

---

## 0. The one-line answers

| question | answer |
|---|---|
| **Was any published NUMBER scored against 1/128 rather than 1/117?** | ⛔ **NO.** MEASURED over the whole repo: **0 mismatches** between a banked `chance` and its own `n_anchors`. Every computed denominator was already derived. |
| **Then what was wrong?** | ⚠️ **The wrong denominator IS PUBLISHED — as prose, INSIDE the banked artifacts.** `refcv4b_t1.json` carries `sidecar_schema.anchor_acc = "chance = 1/128 = 0.0078"` and `tactical_declared.anchor_selection.chance = 0.008547` **in the same file**. It is a documentation defect that reached the deliverables, not only the source. |
| **Does any conclusion flip?** | **No.** refcv4b's `anchor_acc` is **0.0993** — 11.6× the true chance 0.008547, 12.7× the false 0.0078. Both bars are cleared by >11×. |
| **Can a synthetic dump still be mistaken for a real arm?** | See §4. |

---

## 1. D1 — the hardcoded denominator (`D-EVALTOOL-ANCHOR-CHANCE`)

### 1.1 The defect, and its exact scope

`_SIDECAR_DOC["anchor_acc"]` shipped the literal string `"chance = 1/128 = 0.0078"` into
**every** manifest. **128** is `stack/tanitad/refs/refc.py:356`'s DEFAULT bank; refcv4b's
fitted bank is **117**, whose chance is **1/117 = 0.008547** — 9.4 % larger.

⛔ **The computed field was never wrong.** Two independent mechanisms, each with a same-breath
non-zero control:

* **Mechanism 1 — parsed-object walk** (`json.loads` on every JSON, inspecting the OBJECT, not
  the text): **3,480 JSON + 1,703 MD scanned, 0 unreadable**, control = **1,487** files
  containing `anchor` (non-zero, so the walk really read the corpus).
  Every dict carrying a `chance` key was checked against its sibling `n_anchors`:
  **0 mismatches.** The distinct refcv3-family pairs in the corpus are exactly
  **(117, 0.008547)**, **(128, 0.007812)** and **(20, 0.05)** — all correct derivations.
* **Mechanism 2 — literal-string walk** over `.json/.md/.py/.txt`: **8,879 files scanned,
  0 unreadable**, control = **2,293** files containing `anchor`. 47 hits on `1/128`, 41 on
  `its 128 anchors`, 38 on `chance = 1/`.

⚠️ *The first run of mechanism 1 died mid-write on a **cp1252 `UnicodeEncodeError`** in
`print()` — the exact truncated-artifact failure `CLAUDE.md` warns about. It was re-run with
ASCII-safe output to a UTF-8 file; the numbers above are from the completed runs.*

### 1.2 Where the stale string actually landed

At JSON path **`refcv3.manifest.sidecar_schema.anchor_acc`**, in every banked 117-anchor arm:

```
TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refcv4b-landing/raw/refcv4b_t1.json
                                                       .../raw/refcv4b_ego_zero.json
                                                       .../raw/refcv4b_frames_blind.json
.../Research/2026-09-06-refcv4b-navpred/raw/refcv4b_navpred.json
                                       .../raw/refcv4b_navpred_CORRECTED.json
                                       .../raw/refcv4b_navflip.json
.../Research/2026-09-06-selection-quality/raw/refcv4b_t1_s1_ARM.json
.../Research/2026-09-06-rollability/raw/c3_cost_probe.json
.../Research/2026-09-06-rollability/raw/refcv4b_c3_devbox.json
TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-05-nav-compliance-metric/raw/refcv4b-9500-openloop.json
```

**THE ARTIFACT THAT SETTLES IT** is
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-refcv4b-landing/raw/refcv4b_t1.json`,
which carries **both** readings and contradicts itself:

| path in that one file | value | correct? |
|---|---|---|
| `refcv3.manifest.model.n_anchors` | **117** | — |
| `refcv3.manifest.sidecar_schema.anchor_acc` | `"chance = 1/128 = 0.0078"` | ⛔ **WRONG** |
| `refcv3.tactical_declared.anchor_selection.chance` | **0.008547** | ✅ right |
| `refcv3.tactical_declared.anchor_selection.anchor_acc.mean` | **0.0993** | ✅ (T1, episode-cluster bootstrap, [0.0728, 0.1281]) |

### 1.3 ⭐ The prose claim that is NOT a retraction

`Project Steering/GOALS_AND_CLAIMS.md` L4957 states *"`eval_anchor_acc` 0.569, chance 1/128"*
under `D-REFCV3-EPOCH-READ`. **That row is about refcv3, not refcv4b, and refcv3's bank really
is 128** — MEASURED from the two checkpoints' own manifests:

| arm | ckpt | step | n_anchors | chance | anchor_acc (T1) |
|---|---|---|---|---|---|
| **refcv3** | `refcv3-b1-v72-30k/ckpt_40284_FINAL.pt` | 40,284 | **128** | 0.007812 | 0.5654 [0.532, 0.5992] |
| **refcv4b** | `refcv4b-b1-v72-40k/ckpt_40284_FINAL.pt` | 40,284 | **117** | 0.008547 | 0.0993 [0.0728, 0.1281] |

⇒ **the register does not need a correction**; the two arms simply have different banks, and
the confusion is entirely a property of the *tool's* hardcoded string. (Both arms are step
40,284 on the same 141-episode / 4,823-window grid, which is exactly why they are so easy to
conflate.)

### 1.4 The fix — derived, in one place

* **`anchor_chance(n_anchors)`** is now the only place `1/n` is computed. It returns `None` —
  never a fallback — when the artifact declares no bank size, and it refuses a value that is
  not a real number (a `"117"` string is an artifact whose schema is not what we think it is).
* **`sidecar_schema(n_anchors)`** fills the schema line from the run's own `model.n_anchors`,
  so the string and the computed field are **the same derivation** and cannot drift.
* ⛔ **Both `or 128` fallbacks are gone.** `_selection_profile` now **REFUSES** on a manifest
  with no bank size (it was silently publishing `n_anchors=128`, which deflates `entropy_ratio`
  by 1.85 % — `ln 117 / ln 128 = 4.7622 / 4.8520`), and the tactical family publishes
  `chance: null` with `chance_absent_because` rather than inventing a denominator.
* The manifest's `t1_definition._is` ("among its 128 anchors") and the module docstring are now
  per-run / scoped.

⛔ **NO COMPUTED METRIC MOVES.** `anchor_chance(117)` returns **0.008547**, bit-identical to the
value already in the landing JSON; the old expression was `round(1.0/max(1,n), 6)` and the new
one is `round(1.0/int(n), 6)` over the same `n`. Verified directly against the banked artifacts.

⚠️ **Substituting 117 for 128 would have been the same defect with a different number** — this
is the `df` / `step_s` / cylindrical-FOV family: a true value quoted where it does not apply.

### 1.5 The pinning tests

`stack/tests/test_refcv3_arm.py`, four tests:

* `test_anchor_chance_is_one_over_the_runs_own_bank` — 117/128/256 at the exact banked
  precision, and `None` for every non-declaring input.
* `test_the_schema_string_and_the_computed_chance_cannot_drift` — the schema line, the tactical
  `chance` and `model.n_anchors` are checked **against each other**, never against a literal.
* `test_the_denominator_follows_the_bank_size_by_mutation[117|128|256|20]` — ⭐ **the pin that
  would fail if the bank size changed**: parametrised over four bank sizes, each asserting the
  published denominator moved with it *and* that no other bank's number appears.
* `test_a_manifest_with_no_bank_size_is_refused_not_defaulted` — the `or 128` regression guard.

---

## 2. D2 — the stamp that could not discriminate (`D-EVALTOOL-STAMP-BLIND`)

### 2.1 The defect, MEASURED on live artifacts

`_UNVERIFIED_ON_REAL_CKPT` ("UNVERIFIED on a real checkpoint … validated on a random-init
RefCV3Model … synthetic 3-episode slice only") was emitted **unconditionally**, twice — into
every manifest (`run_dump`) and every analysis record (`main`).

⛔ **It reads identically on a fixture and on a real 40,284-step arm.** Applying the *old* and
the *new* logic to three artifacts that are actually on disk:

| banked artifact | step / eps / windows / anchors | OLD `_unverified` | NEW verdict |
|---|---|---|---|
| `…/2026-09-06-refcv4b-landing/raw/refcv4b_t1.json` | 40,284 / 141 / 4,823 / 117 | **present** | `REAL_CHECKPOINT_ON_REAL_CORPUS` |
| `…/2026-09-06-refcv4b-landing/raw/REFCV3_BASELINE_REANALYZED.json` | 40,284 / 141 / 4,823 / 128 | **present** | `REAL_CHECKPOINT_ON_REAL_CORPUS` |
| `…/Benchmarks & Evals/Research/2026-09-03-refcv3-arm/raw/fixture_dump/manifest.json` | 11 / 3 / 42 / 20 | **present** | `SYNTHETIC_FIXTURE` |

⇒ anyone using the old string as a fixture/real discriminator **misclassifies every real dump
the programme holds**, which is how it reached the PI as exactly such a discriminator.

⚠️ **Why the obvious discriminator does not work:** a clean strict load is NOT evidence of
trainedness. The fixture's `state_dict_load` reports `missing_keys: []` and
`unexpected_keys: []` — **identical to the real landing arm** — because a random-init model
saved and reloaded also loads cleanly. This is asserted in the suite
(`test_a_clean_strict_load_is_carried_but_never_read_as_a_discriminator`) so no future reader
rediscovers it, and it is named in the emitted block under `not_a_discriminator`.

### 2.2 The fix

`provenance_stamp(manifest)` derives, from the manifest's **own** `model` / `grid` blocks:

* `checkpoint_scale` ∈ {TRAINED, SMOKE, UNKNOWN} — from `model.step` (threshold 1,000;
  fixture 11 vs real 40,284, three orders of magnitude apart)
* `corpus_scale` ∈ {FULL, SMOKE} — from `grid.n_episodes` / `grid.n_windows` (10 / 500;
  fixture 3 / 42 vs real 141 / 4,823)
* `bank_scale` ∈ {REAL, SMOKE, UNKNOWN} — from `model.n_anchors` (64; fixture 20 vs real 117)
* `verdict`, `discriminators` (the four values that decided), `thresholds`,
  `not_a_discriminator`, `_reading`, `_not_what_this_proves`

⭐ **Three axes, reported separately, never pooled** — the same reason the four metric families
are never pooled into one score: a single label hides *which* axis is smoke-scale.

`provenance_keys()` emits `_provenance` **always** (so absence is never ambiguous with an old
record) and `_unverified` **only when the claim is true of that run**. `main()` no longer
re-asserts the constant — re-asserting it there is what made it unconditional.

⭐ Because the stamp is derived from the manifest, **`analyze_refcv3` classifies a dump banked
before the stamp existed** — retroactively, from `manifest.json` alone, no re-roll, no GPU.

### 2.3 ⛔ The two-sided mutation proof

`test_mutating_the_run_from_synthetic_to_real_flips_the_stamp` runs `run_dump` **twice** on the
same corpus with **one lever moved** — the checkpoint's own step count (11 → 40,284) — and
requires the stamp to change. Its assertions are deliberately ordered so that the discriminating
one is written in the **old vocabulary**, so the test fails on the *substance* against the
pre-fix file and not merely on a key that did not exist yet.

Against the pre-fix file (blob `5e1349a9a9874d83685572bca30ef9d69bfed91b`) it fails with:

```
E  AssertionError: the stamp still asserts UNVERIFIED-ON-A-REAL-CHECKPOINT about a
   40,284-step checkpoint - it cannot discriminate, which is D-EVALTOOL-STAMP-BLIND
E  assert '_unverified' not in {...}
```

and the two D1 tests fail the same way:

```
E  AssertionError: the schema publishes a 1/128 chance level for a 20-anchor bank
E  AssertionError: a manifest with no bank size still produced a profile: n_anchors=128
```

⛔ **A stamp that reads identically on both has measured nothing** — that was the defect, and
this test is what makes its return impossible.

---

## 3. D3 (found here, not in the brief) — `e7_off` crashed (`D-EVALTOOL-E7-CLOSURE-ARG`)

Establishing the pre-edit baseline surfaced **two live failures** in
`stack/tests/test_refcv3_ablations.py`, both from
`TypeError: 'NoneType' object is not callable` inside this file.

**Root cause.** The `e7_off` wrapper captured the original hook through a **default parameter** —
`def _hook_no_e7(cache, nav_cmd=None, ego_state=None, _o=_orig)` — which makes the closure cell
**positionally reachable from the call site**. When `RefCV3Model._hook` gained a 4th argument
(`nav_args`, `refc_v3.py:1031`, the strategic-bypass work), the model's own call at
`refc_v3.py:1373` passed it fourth, it bound to `_o`, and `_o(...)` was then `None`.

**Fix.** `_make_hook_no_e7(orig)` — a factory, so the original is captured as a real closure over
a parameter that no caller can reach, and every argument is forwarded verbatim so a 5th cannot
repeat this.

⛔ **No banked number moves:** the ablation *crashed*; it produced no metric to move.

---

## 4. Can a synthetic dump still be mistaken for a real arm?

**Not by reading the stamp, and not by accident — but the stamp reads self-reported fields, so
a manifest whose `model.step` is wrong is outside what any stamp can see.** Every dump now
carries `_provenance` with the four values that decided and the thresholds they were compared
against; `_unverified` appears only when it is true; and the three discriminators separate the
fixture from the real arm by two to three orders of magnitude each, so no single threshold is
load-bearing. This is stated in the emitted block itself under `_not_what_this_proves`.

---

## 5. Suite — CONTROLLED comparison, failure-ID diff in BOTH directions

Run off-Drive (`C:\Users\Admin\tanitad-evtool-20260906`, a fresh copy verified byte-identical to
the worktree by `git hash-object`; ⛔ **not** the `tanitad-wt` mirror, which re-syncs and drops
edits). `PYTHONPATH=<clone>/stack`, `OMP_NUM_THREADS=6`, CPU only.

### 5.1 The two files this work touches

| file | BEFORE (pre-edit tool, blob `5e1349a…`) | AFTER |
|---|---|---|
| `stack/tests/test_refcv3_arm.py` (20 pre-existing tests) | 20 passed | **20 passed** |
| `stack/tests/test_refcv3_arm.py` (+12 new) | **12 failed** / 20 passed | **32 passed** |
| `stack/tests/test_refcv3_ablations.py` | **2 failed** / 26 passed / 1 skipped | **28 passed** / 1 skipped |

**Failure-ID diff, both directions:**

*Removed (were failing, now pass) — 2, both pre-existing, both D3:*
```
test_refcv3_ablations.py::test_each_ablation_moves_the_record[e7_off]
test_refcv3_ablations.py::test_the_ablation_stamp_reaches_the_manifest_and_every_arm_block
```
*Added (now failing that were not before): **NONE.***

*The 12 new tests fail on the pre-edit tool by construction* — that is the mutation proof, not a
regression.

### 5.2 The other suites that import this tool

`test_hierarchy_panel_defect_gate` · `test_kingate_contract` · `test_navcomp_labels_shape` ·
`test_refcv3_ha0_ext_shared` · `test_refcv3_route_nav_alignment` · `test_refc_v3_rollability` ·
`taniteval/tests/test_refcv3_arm_astar_geometry` · `taniteval/tests/test_render_refcv3_video`
— **82 passed, 2 skipped, 0 failed.**

*(One initial failure, `test_kingate_contract::test_the_gate_result_is_stamped_T0_until_a_T1_leg_exists`,
was a `FileNotFoundError` for `Project Steering/Decisions/…` — an artifact of the partial
off-Drive copy. It passes once that tree is copied in. Environmental, not a regression.)*

### 5.3 ⭐ THE WHOLE `stack/tests` SUITE, RUN TWICE — the controlled comparison

The entire suite was run **with the pre-edit tool** and **with the post-edit tool**, and the
failure-ID sets diffed in both directions. (The pre-edit run ignores `test_refcv3_arm.py`,
whose 12 new tests fail there by construction; the post-edit set is compared with that file
excluded so the two are like-for-like.)

| | pre-edit tool (blob `5e1349a…`) | post-edit tool |
|---|---|---|
| totals | **55 failed**, 7,002 passed, 119 skipped, 2 xfailed, 29 errors (12m19s) | **53 failed**, 7,035 passed, 120 skipped, 2 xfailed, 29 errors (15m20s) |
| distinct failing IDs | **55** | **53** (excluding `test_refcv3_arm.py`) |

**⬇ ONLY IN PRE — fixed by this work (2):**
```
FAILED stack/tests/test_refcv3_ablations.py::test_each_ablation_moves_the_record[e7_off]
FAILED stack/tests/test_refcv3_ablations.py::test_the_ablation_stamp_reaches_the_manifest_and_every_arm_block
```
**⬆ ONLY IN POST — introduced by this work: NONE. Zero.**

The 53 shared failures/29 errors are identical in both runs and untouched by this stream —
`test_v6_chain`, `test_rl_*`, `test_secret_scan`, `test_text_encoding_is_explicit`,
`test_v6_st_launch_fixes` and similar, most of them artifacts of the partial off-Drive copy
(missing `.git` hooks, run directories, checkpoints). ⛔ They are reported as a **shared
baseline**, not claimed as clean, and not claimed as mine.

---

## 6. What was deliberately NOT touched

* ⛔ No computed metric. `anchor_chance(117)` = **0.008547** = the value already banked.
* ⛔ `oracle_sel` / `anchor_acc` / `sel_agrees_oracle` semantics untouched.
* ⭐ The sibling's `a_star` fix (per-window `_decoded_bank`, the two branches collapsed) is
  **built on, not reverted** — it is present in the committed file.
* Files owned by live siblings (`refc.py`, `refc_v3.py`, `refc_v3_train.py`, `taniteval/ci.py`,
  the paper, …) were read but never edited.

## 7. ⚠️ Escalation — the same defect, mirrored, in a file this stream does not own

`taniteval/tools/training_watch/build_watch.py:21` hardcodes `CHANCE_ANCHOR = 1.0 / 117.0` and
its caption says *"117-anchor fan · chance 0.85 %"*. That is **correct for refcv4b and wrong for
refcv3**, whose bank is 128 — the mirror image of D1, with the other number. It should read the
bank size from the run it is describing. **Not fixed here** (not this stream's file); escalated
rather than written into a doc nobody re-reads.

## 8. ⚠️ Index hygiene observed while banking (not caused here, not repaired here)

`git diff --cached --diff-filter=D` lists **85 staged deletions** in the shared index. A first
classification pass read **84 phantom** (present in `HEAD` *and* non-empty on disk) / 0 genuine;
a **second pass over the identical list disagreed**, reporting `in_head=False` for ten of them.
⛔ **Two passes of the same probe disagreeing means the probe is not answering** — the
`git ls-tree`/`cat-file` under-reporting documented in `CLAUDE.md`, so **no classification is
asserted here**, only that a pathspec-free commit in this state is unsafe. This commit therefore
names its two paths explicitly and seeds its tree from `HEAD`, so the shared index is never read
and no sibling's staged work is swept or deleted.

## 9. Deliverable manifest

| artifact | where it lives |
|---|---|
| `taniteval/tools/refcv3_arm.py` (the three fixes) | **repo**, committed |
| `stack/tests/test_refcv3_arm.py` (+12 tests) | **repo**, committed |
| this `RESULT.md` | **repo**, committed |
| `raw/SUITE_PRE_ids.txt`, `raw/SUITE_POST_ids.txt`, `raw/SUITE_DIFF.txt` | **repo**, committed |
| `raw/PROVENANCE_ON_BANKED_ARTIFACTS.txt` (the stamp applied to the 3 real manifests) | **repo**, committed |
| `raw/CHANCE_PAIRS.txt` (the 0-mismatch corpus sweep) | **repo**, committed |
| register entries `D-EVALTOOL-ANCHOR-CHANCE`, `D-EVALTOOL-STAMP-BLIND`, `D-EVALTOOL-E7-CLOSURE-ARG` | **appended** to `Project Steering/GOALS_AND_CLAIMS.md` |
| off-Drive test tree (throwaway) | `C:\Users\Admin\tanitad-evtool-20260906` — **not** the `tanitad-wt` mirror |
