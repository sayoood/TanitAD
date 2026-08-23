# `goal_evidence: grounded` retired, and the ladder-edge "flake" was a real defect landed 18 h earlier

**Package owner:** arch-inf agent, 2026-08-16 · branch `agent/arch-inf-20260803`
**Two independent 0-GPU defects.** Priority 1 gates the 4,472-clip Alpamayo build; priority 2 is an
intermittent test that had to be *diagnosed*, not retried into silence.
**STAGED, NOT COMMITTED, NOT PUSHED.**

---

## 0. TL;DR

| | verdict | class |
|---|---|---|
| **T1 — `goal_evidence: grounded`** | ⛔ **RETIRED.** So is `provisional`. The verdict is now always `not_computable` with the reason NAMED; the one measured fact survives as **`sign_like_object_present`**. Emission of `grounded` over aug120: **15/201 → 0/201**. | **MEASURED** (§1.4, re-run on real fused records) |
| **T2 — the S-J "flake"** | ⛔ **NOT a flake and NOT an RNG problem.** `KeyError: 'interp'`: the grad census dropped a zero-parameter group, and `any()` over a **hash-randomised `set`** short-circuited before or after the missing key. Rate **3/25 = 12.0 %**, predicted **10.75 %** from a 400-seed sweep, and **DETERMINISTIC** at `PYTHONHASHSEED=3`/`12`. | **MEASURED** (§2.1–2.3) |
| **T2 — does it pre-date today?** | ⛔ **NO — and this is the useful fact.** The test file is **byte-identical** at `ee02ff7` (16:55) and HEAD, and at `ee02ff7` it **passes at the exact seeds that fail deterministically at HEAD**. The break arrived at **20:57 today**, in `06b8782`. | **MEASURED** (§2.4) |
| ⭐ **T2 — a genuine MODEL defect found underneath** | With `agent_slots=True`, `interp` owns **62 parameters**, `apply_stage_freeze(stack, "S-J")` marks **all 62 TRAINABLE**, and the S-J loss reaches **0 of them**. That is verbatim the lie `v6.py`'s own `STAGE_GROUPS` docstring says it avoids. Latent only because the flag defaults False — **and the agent-slot decoder is an active workstream today.** | **MEASURED** (§2.6) |

⛔ **ESCALATION (integration, not documentation) — three items, §5.** The `interp`/S-J defect needs a
`v6.py` change I do **not** own; `NEXT_4472_BUILD_INPUTS.md` §3 still describes `goal_evidence` by
its retired name and by a corpus-mismatched reliability figure; and a `RETRACTION_LOG.md` class is
warranted that I deliberately did **not** write (that file carries another agent's staged work).

---

# PART 1 — `goal_evidence: grounded` is retired

## 1.1 The decision, and why RETIRE rather than rename in place

The brief offered two options: rename to something honest, or remove the class entirely.
**I did neither exactly — I retired the VERDICT and kept the MEASUREMENT under its own name**, which
is strictly stronger than either, and here is the argument for each half.

**Why not keep a positive verdict under a new name (e.g. `sign_present`).** A `verdict` field in
this fuser is read as *the corroboration outcome for this check*. Any token in that slot that is
not `not_computable` reads as "something was established". Nothing was: the sign's KIND is
unchecked, its TEXT is ungated at 0/31, and its FRAME is unchecked. A renamed positive verdict
would move the overstatement one word to the left.

**Why not delete the check.** Deleting it makes the gap invisible. The fuser's own house rule —
already applied to `scene_vs_situations` and to an absent SAM3 leg — is *"absence of an instrument,
recorded as absence"*. It also throws away a real, cheap measurement (how many `traffic sign`
tracks the clip carries), which the 4,472 build will want.

**⇒ The shape that satisfies both:**

| field | what it now says | provenance |
|---|---|---|
| `verdict` | **always `not_computable`** | — |
| `reason` | names all three gaps: KIND-blind / FRAME-blind / applies-to-ego-blind detector, and the TEXT gate CLOSED at 0/31 | — |
| `sign_like_object_present` | ⭐ **the only measured fact**: a sign-LIKE object was detected somewhere in this clip | `sam3` |
| `sam3_sign_tracks` | the raw count (unchanged name, unchanged semantics) | `sam3` |
| `evidence_sign_kind` | **NEW.** The kind the VLM itself recorded for the sign it cited — ⚠️ a **VLM self-report**, never corroboration | `vlm` |

`provisional` goes too, and this is not tidiness. It read as *"SAM3 looked and found no sign"*, but
the reliability study measures **NO RECALL for any concept** — zero sign tracks is not evidence of
no sign. It was the same overstatement with the sign flipped. On aug120 it never fired (0/201), so
retiring it costs nothing measurable.

⚠️ **Why `evidence_sign_kind` is worth adding rather than scope creep.** It is exactly the field a
future KIND check needs, it costs one dict lookup, and it turns the **24/31 non-nav gap** from a
number in a study into a per-clip auditable fact. It is recorded on **both** branches — including
SAM3-absent — because the KIND is a VLM-side fact; gating it on SAM3 coverage would have made the
gap measurable on only 15 of 31 clips, which is a coverage artifact dressed as a rate.

## 1.2 The evidence the retirement rests on — all of it PRIMARY

| fact | value | source (primary) |
|---|---|---|
| SAM3 `traffic sign` precision | **0.880 [0.795, 0.958]**, n=64 / 33 clips, episode-cluster bootstrap | `…/2026-08-16-sam3-concept-reliability/SAM3_CONCEPT_RELIABILITY.md` §3 |
| dominant FP mode | **sign-SHAPED non-signs** — pharmacy cross at **0.807**, the highest-scoring of the six sign FPs ⇒ **no threshold separates them** | ibid. §3.2 |
| sign TEXT gate | **CLOSED at 0/31** | `Project Steering/G1_RESULT.md` |
| cited sign is not `nav` | **24/31** (speed 15 · other 6 · yield 2 · stop 1; nav 7) | `…/2026-08-16-s2-strategic-gap/raw/aug120_analysis.json` → `route_to_sign_kinds` |
| verdict inventory before | `grounded` **15**, `not_computable` **16**, `provisional` **0** (n=201) | same file → `route_to_verdicts` |

⇒ **The inputs were never the problem.** ~88 % of the tracks are real signs. The name was the
problem, and no score cut repairs a name.

## 1.3 ⛔ The blast radius is ZERO downstream code — checked, not assumed

- `corroboration.goal_evidence` is read by **nothing** outside `ph1_fuse.py` and its test. Every
  other repo hit is `goal_evidence_sign`, the **VLM input field**, which is untouched
  (`ph0_v2.py`, `ph0_rich_overlay.py`, `ph0_v2_overlay.py`, `colab/s2_lab_lib.py`).
- The run summary counts **only** `verdict == "corroborated"` (`ph1_fuse.py`, `summ["corroborated"]
  += sum(1 for c in cor.values() if c.get("verdict") == "corroborated")`). `grounded` was **never**
  in that tally — aug120's `corroborated: 88` is exactly `speed_sign_vs_ego 83 + red_light 5`.
  ⇒ **no published corroboration number moves.** Pinned by
  `test_the_retirement_does_not_move_the_corroborated_TALLY`.
- Nothing was added to the `conflicts` stream. A retirement that converts a false positive into a
  false conflict is not a fix; asserted in the sweep test.

## 1.4 ⭐ The emission-rate change — MEASURED, by re-running the new predicate on real records

`code/g1_goal_evidence_rate.py` → `raw/goal_evidence_rate.json`. It does **not** re-fuse: the fused
records already carry every input this check reads (VLM symbols, VLM `signs`, SAM3 tracks, the
absence marker), so old-verdict-in-record vs new-verdict-recomputed is an exact A/B on the same
clip, zero GPU.

**Corpus (aug120, 201 clips)** — denominators from `aug120_analysis.json`, a primary raw JSON:

| | `grounded` | `provisional` | `not_computable` |
|---|---|---|---|
| **before** | **15/201 (7.46 %)** | 0/201 | 16/201 |
| **after** | ⛔ **0/201** | ⛔ 0/201 | **31/201 (15.42 %)** — every `route_to` clip |
| `sign_like_object_present: true` | — | — | **15/201**, i.e. exactly the clips `grounded` used to fire on |

**Sample re-run (30 fused records in-repo, MEASURED not arithmetic):** 8 `route_to` clips,
before `{grounded 4, not_computable 4}` → after `{not_computable 8}`; **4 changed**, 0 regressions.

⭐ **And the four retired `grounded` verdicts cited signs of kind `stop`, `speed`, `yield`, `speed`
— 0 of 4 were navigation signs.** A stop sign and a give-way triangle were grounding a
`route_to <place>` claim. Across all 8 sampled `route_to` clips the kinds are speed 4 · nav 2 ·
stop 1 · yield 1 = **2/8 nav**, consistent with the corpus 7/31 = 22.6 %.

⇒ At the 4,472-clip scale the retired predicate would have emitted roughly **330 `grounded`
verdicts** (15/201 × 4472, ESTIMATED by rate) as a supervision channel, of which — at the measured
kind distribution — about **77 % would not even have cited a navigation sign.**

## 1.5 The pins

`stack/tests/test_ph1_fuse.py`, five new test functions = **+28 collected cases**, all passing
(44 → 72 in the file):

1. `test_the_retired_goal_evidence_tokens_are_GONE` — **swept, not sampled**: 3 track counts × 2
   evidence indices × 4 sign kinds = 24 combinations. The retired predicate was a CONJUNCTION
   (`ev is not None and n_sign_frames > 0`); a single-case test would have left three of its four
   corners unpinned.
2. `test_the_retired_tokens_cannot_be_re_emitted_from_the_SOURCE` — an **AST** pin: any string
   literal *equal to* a retired token, anywhere in the module outside the one declaration, is a
   failure. Prose in the retirement note that merely mentions the word does not trip it.
   ⚠️ **Non-vacuity CONTROLLED**: injecting `"verdict": "grounded"` back into the emitter makes it
   report `[(574, 'grounded')]`.
3. `test_the_PRESENCE_fact_survives_under_an_honest_name`
4. `test_the_cited_sign_KIND_is_recorded_as_a_VLM_SELF_REPORT` — including that an unknown kind
   reads as `None` and is never fabricated (out-of-range index, dropped `signs` block, null index,
   `bool` index), and that the SAM3-absent branch carries the KIND but **not** the SAM3-side facts.
5. `test_the_retirement_does_not_move_the_corroborated_TALLY`

---

# PART 2 — the `[S-J]` "flake": a real defect, landed 18 hours earlier

⛔ **A test that passes on retry has not been fixed (C84).** So nothing below rests on a re-run.

## 2.1 The rate, with an n

`tests/test_v6_ladder_edges.py`, whole file, 25 separate processes, identical env
(`PYTHONUTF8=1 OMP_NUM_THREADS=6`, venv torch 2.11.0+cu128, pytest 9.1.1):

> **3 failed / 25 = 12.0 %.** All three are the *identical* signature —
> `KeyError: 'interp'` at `test_after_init_from_exactly_the_intended_groups_train[S-J]`,
> line 459. No other failure occurred in 25 × 26 = 650 test executions.

## 2.2 ⭐ The mechanism — and the plausible hypothesis is REFUTED

The failing line is the LAST assertion:

```python
assert any(census[g]["grad"] for g in want), \
    f"stage {stage} trained NOTHING — the freeze map and the loss disagree"
```

Three facts compose into the failure:

1. **`_grad_census` used `setdefault`**, so a group owning **zero parameters** was simply *absent*
   from the census dict rather than present-and-empty.
2. **`STAGE_GROUPS["S-J"] is MODULE_GROUPS`**, which includes **`interp`** — a group `v6.py`
   documents verbatim as *"EMPTY at the default build: `V6Config.agent_slots` is False, so this
   group holds ZERO parameters"*. So `want` contains a key `census` does not have.
   MEASURED params-per-group at the tiny default build: encoder 17 · readout 2 · predictor_op 35 ·
   layer_tac 67 · layer_str 54 · planner 9 · aux 30 · **interp 0**.
3. **`want` is a `set` of `str`, and `any()` SHORT-CIRCUITS.** Python randomises string hashing per
   process (`PYTHONHASHSEED`), so set iteration order changes run to run. All seven *built* groups
   have `grad > 0` at S-J, so the generator raises **iff `interp` is iterated first**.

**The prediction this makes is quantitative, and it holds.** Over 400 `PYTHONHASHSEED` values,
`next(iter({...the 8 group names...}))` is `interp` on **43/400 = 10.75 %** — against the observed
**3/25 = 12.0 %**.

⚠️ **The RNG hypothesis in the brief (*"seeds the model but draws its batch from global RNG"*) is
REFUTED, and it is worth saying why it was so plausible.** `_stage_batch` really does draw
`gt_wp = torch.randn(2, 10, 2)` from the global stream while `synthetic_train_batch` uses a local
`torch.Generator`. But (a) `mk()` calls `torch.manual_seed(seed)` and nothing between it and the
draw touches the global stream, so `gt_wp` is in fact deterministic; and (b) **the failure is a
`KeyError` on a structural dict lookup — no data value can reach it.** A real mechanism that is
adjacent to the failure is still not the mechanism.

## 2.3 DETERMINISTIC reproduction — the flake is a function of one environment variable

| `PYTHONHASHSEED` | first element of `want` | result |
|---|---|---|
| **3** | `interp` | ⛔ **1 failed, 3 passed** |
| **12** | `interp` | ⛔ **1 failed, 3 passed** |
| 0 | `aux` | 4 passed |
| 5 | `readout` | 4 passed |

Every one reproduces on demand. **There is no flakiness left to describe** — there is a defect with
a 10.75 % exposure rate.

## 2.4 ⛔ Does it pre-date today's changes? **NO** — and the A/B is clean

This was the single most useful open question, and it is now MEASURED rather than argued.

- The `interp` group entered `MODULE_GROUPS` in **`06b8782`, 2026-08-16 20:57:13 +0200**. Walking
  every commit that touched `v6.py`: `ee02ff7` (16:55) and all twelve before it contain **zero**
  occurrences of `interp` in the file.
- The test file was created in **`5725d95`, 2026-08-16 02:26:12 +0200** — **18.5 h earlier** — when
  `MODULE_GROUPS` held seven groups, *all* of which own parameters. `any(census[g] ...)` was total
  and safe as written.
- **The test file is byte-identical between `ee02ff7` and HEAD** (`git diff --stat` empty), so this
  is a clean single-variable A/B. Running that identical file against the `ee02ff7` tree
  (extracted with `git archive`, no checkout, no worktree):

| tree | seed 3 | seed 12 | seed 0 | seed 5 |
|---|---|---|---|---|
| `ee02ff7` (pre-`interp`) | **4 passed** | **4 passed** | 4 passed | 4 passed |
| HEAD | ⛔ **1 failed** | ⛔ **1 failed** | 4 passed | 4 passed |

⇒ **The test was correct when written and was broken by a model change 18.5 hours later.** Neither
"the test is flaky" nor "the test was always wrong" survives.

## 2.5 The fix — the mechanism, not the symptom

`stack/tests/test_v6_ladder_edges.py`:

1. **`_grad_census` is now TOTAL over `MODULE_GROUPS`** — an empty group reads `{"grad": 0,
   "none": 0}` instead of vanishing. *A census is a total function over the partition;* the
   `setdefault` form was silently a partial one.
2. **New `_built(census)`** — the groups that own ≥1 parameter *at this build*. "Declared in
   `MODULE_GROUPS`" and "built by this config" are different sets, and conflating them is what
   produced the flake.
3. The vacuity-control fixture now asserts **`set(MODULE_GROUPS) - built == {"interp"}`**, naming
   the expected empty group — so a future build that silently empties a *different* group is a
   failure, not a shrug — and then asserts every **built** group is reachable.
4. The final assertion iterates `want & built`, and **first** asserts that intersection is
   non-empty, so "this stage declares only unbuilt groups" reports itself rather than hiding inside
   a vacuously-false `any()`.

⚠️ **The bug also SUPPRESSED THE ASSERTION IT LIVED IN.** A genuine *"stage S-J trained nothing"*
would have surfaced as a bare `KeyError` with no message instead of the intended diagnosis. Fixing
the lookup restores the assertion's ability to report, which is the larger of the two wins.

**Anti-flake pin:** `test_the_grad_census_is_TOTAL_over_the_group_partition` asserts the census is
total, that `interp` is present-and-empty, and — the exact expression that raised — that **every
stage's declared groups are indexable in the census**.
⚠️ **Non-vacuity CONTROLLED**: monkeypatching `_grad_census` back to the pre-fix `setdefault` body
makes the pin fail with *"the census is not total over MODULE_GROUPS — the KeyError is back"*.

**Post-fix rate: 0 failures in 40 runs** (same harness, same env) — and, more decisively than any
rate, it now passes at seeds 3 and 12, which fail deterministically without the fix.

## 2.6 ⭐⭐ The serious finding underneath: S-J declares a group its loss never reaches

The empty `interp` group was hiding a real model defect. MEASURED on the tiny stack, both builds:

| build | `interp` params | marked **TRAINABLE** by `apply_stage_freeze(·, "S-J")` | params that **received grad** from the S-J loss |
|---|---|---|---|
| `agent_slots=False` (default) | 0 | 0 | 0 |
| **`agent_slots=True`** | **62** | ⛔ **62** | ⛔ **0** |

`v6.py`'s own `STAGE_GROUPS` docstring states the rule being broken:

> *"Listing `interp` as trainable in a stage whose loss never reaches it would make the freeze audit
> report a module as 'training' while it receives exactly zero gradient — the same lie
> `V6LossWeights.for_stage` zeroes its planner terms to avoid."*

…and then defines `"S-J": MODULE_GROUPS`, which does exactly that. The docstring even notices
(*"`interp` APPEARS IN NO STAGE BUT S-J (which is `MODULE_GROUPS` by definition)"*) and calls it
deliberate — which is defensible **only** while the group is empty.

⚠️ **It is latent, not harmless.** The agent-slot decoder is an active workstream *today*
(`…/2026-08-16-agent-slot-decoder`, `…/2026-08-16-slot-probe-run`). The first run with
`agent_slots=True` gets a freeze audit reporting 62 parameters as training while none of them moves
— the precise failure class `CLAUDE.md` calls *a probe that reports the wrong scope*.

**Pinned as `@pytest.mark.xfail(strict=True)`** on
`test_S_J_must_not_declare_trainable_a_group_its_loss_never_reaches`. The test states the *desired*
invariant, is marked as a known defect, and because `strict=True` an XPASS is a FAILURE — so the
moment `v6.py` is fixed the suite says so and tells the fixer to delete the marker.
⛔ **I did not fix `v6.py`: I do not own it.** §5, escalation 1.

---

## 3. Suite

Invocation, quoted in full as required:

```
cd stack && PYTHONUTF8=1 OMP_NUM_THREADS=6 pytest -q
```

| | passed | **failed** | skipped | xfailed | wall |
|---|---|---|---|---|---|
| stated baseline | 3689 | 0 | 7 | 2 | — |
| **after this package** | **3744** | ⭐ **0** | **7** | **3** | 399 s |

**My delta is +29 passed and +1 xfailed**, counted from `--collect-only`, not inferred:

| file | before | after | delta |
|---|---|---|---|
| `tests/test_ph1_fuse.py` | 44 | **72** | +28 (a 24-case sweep + 4 singles) |
| `tests/test_v6_ladder_edges.py` | 26 | **28** | +1 passed (the anti-flake pin) · +1 **xfailed** (§2.6) |

⚠️ **The remaining +26 passed are NOT mine.** Three sibling agents are editing `stack/`
concurrently, so the suite total moved for reasons outside this package; the honest statement is
*0 failed* plus my own counted delta, not ownership of the total. The **xfailed 2 → 3** is exactly
my one marker.

Both files I own were also run in isolation: `tests/test_ph1_fuse.py` **72 passed**,
`tests/test_v6_ladder_edges.py` **27 passed / 1 xfailed** — the latter at five different
`PYTHONHASHSEED` values (0, 3, 5, 7, 12), including the two that fail deterministically without
the fix.

⚠️ **One measurement trap hit and avoided while doing this, worth recording:** `grep -l failed` over
the run logs matched **35 of 35** post-fix runs — because pytest prints **`xfailed`**, which
*contains* `failed`. That is the `CLAUDE.md` self-matching-filter trap in a new costume, and it
would have reported a 100 % failure rate on a suite that was 40/40 green. The opaque
`ZZ<i>-<rc>-<pass>-<fail>ZZ` marker (rc parsed pod-/process-side, never grepped from the text) is
what kept the number right, and `g2_flake_diagnosis.py` parses **return codes only** for the same
reason.

## 4. What this package does NOT say

1. ⛔ **It does not measure whether any `route_to` label is CORRECT.** It removes a claim that was
   never established; it establishes no replacement. The strategic ROUTE_TO channel is exactly as
   supported as it was — the record now says so.
2. ⛔ **No recall figure for `traffic sign`**, so `sign_like_object_present: false` still is not
   evidence of "no sign". That is why `provisional` is gone rather than renamed.
3. ⚠️ **The 30-record A/B is a SAMPLE of the 201.** The corpus before/after uses
   `aug120_analysis.json`'s per-verdict counts; the *after* column is arithmetic on those
   denominators (the new predicate is unconditional), not a re-fuse of all 201. A full re-fuse
   needs the far-side v2+SAM3 inputs, which are not in the repo.
4. ⚠️ **The `interp`/S-J measurement is on the TINY test stack.** 62 parameters is that stack's
   figure, not the production model's. The *structure* — declared trainable, zero gradient — is
   config-independent; the count is not.
5. ⚠️ **Single machine.** The hash-order mechanism is interpreter-level and platform-independent by
   construction, but the 12.0 % rate was measured on this dev box only.

## 5. ⛔ Escalations — decisions and integration, not documentation

1. ⛔ **`v6.py` needs the S-J group tuple fixed** (§2.6). Either spell S-J's groups explicitly
   without `interp`, or give S-J a loss that reaches it. **Owner: whoever owns
   `tanitad/models/v6.py`** — I pinned it `xfail(strict=True)` rather than editing a file I do not
   own. ⚠️ This must land **before** the first `agent_slots=True` run, not after.
2. ⚠️ **`…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md` §3 is now stale in two ways:** it
   describes the channel as `goal_evidence: grounded` (retired here) and cites G1's *"⅔ of best sign
   crops contained no sign"* as a property of SAM3's sign class — which the reliability study
   REFUTES on `aug120` (0.880 [0.795, 0.958]) and which was measured on a *different corpus*.
   **Owner: the aug120-fusion package.** Both sentences should name their corpus.
3. ⚠️ **A `RETRACTION_LOG.md` class is warranted and I did NOT write it**, because that file
   currently carries another agent's staged changes and editing it would sweep their work into
   mine. **Proposed class, for whoever holds the log:**
   *"a proposed MECHANISM that is adjacent to the failure, adopted before it was tested against the
   failure's own shape."* The RNG hypothesis here was real (the batch draw genuinely is global) and
   still wrong (the failure is a structural `KeyError` no data value can reach). This is the
   sibling of C84 — C84 says *a pass on retry is not a fix*; this says *a plausible mechanism is
   not a diagnosis*. **The discriminator is cheap and should be the standing rule: make the
   mechanism PREDICT A RATE, then measure the rate.** Here: 10.75 % predicted from a 400-seed
   sweep, 12.0 % observed in 25 runs, and deterministic at two named seeds.

## 6. Deliverable manifest

⛔ Every row verified in the index: NEW files by `git ls-files --cached`, MODIFIED tracked files by
**blob comparison** (`git ls-files --stage` vs `git hash-object`) — `--cached` is not sufficient for
a modified tracked file. **STAGED, NOT COMMITTED, NOT PUSHED**, branch `agent/arch-inf-20260803`.

| artifact | path (repo-relative) | what it is |
|---|---|---|
| **T1 — the fix** | `stack/scripts/ph1_fuse.py` | `grounded`/`provisional` retired; `not_computable` + named reason; `sign_like_object_present`, `sam3_sign_tracks`, new `evidence_sign_kind()` helper |
| **T1 — the pins** | `stack/tests/test_ph1_fuse.py` | 5 new tests (24-case sweep + AST source pin + presence/KIND/tally pins) |
| **T2 — the fix + pins** | `stack/tests/test_v6_ladder_edges.py` | total `_grad_census`, `_built()`, built-intersected assertions, anti-flake pin, and the `xfail(strict=True)` model-defect pin |
| this report | `…/incoming/2026-08-16-evidence-and-flake/EVIDENCE_AND_FLAKE.md` | — |
| emission-rate A/B | `…/2026-08-16-evidence-and-flake/code/g1_goal_evidence_rate.py` → `raw/goal_evidence_rate.json` | §1.4, recomputed on real fused records |
| flake evidence | `…/2026-08-16-evidence-and-flake/raw/flake_diagnosis.json` | §2.1–2.4: the 25-run rate, the 400-seed sweep, the per-seed determinism table, the pre-date A/B |

**Read, not modified:** `stack/tanitad/models/v6.py`, `stack/scripts/train_v6_staged.py`,
`…/2026-08-16-sam3-concept-reliability/`, `…/2026-08-16-s2-strategic-gap/`,
`…/2026-08-15-aug120-fusion/`, `Project Steering/G1_RESULT.md`.

**Not touched, by instruction:** `stack/scripts/ph0_sam3.py`, `colab/`,
`…/2026-08-16-sam3-extraction-v2/`, `…/2026-08-16-slot-probe-run/`,
`Project Steering/MODEL_REGISTRY.md`, `Project Steering/RETRACTION_LOG.md`.

**Far side:** nothing. **No HF write, no pod, no GPU, no network.**
