# `trafficsim` reactivity (full-runtime) + the wheelbase option-B execution

**Date:** 2026-07-26 (Europe/Berlin) · **Host:** `tanitad-eval` only. pod1 / pod2 / pod3 never contacted.
**Author:** trafficsim+wheelbase agent
**Status:** PRE-REGISTERED before any run. Sections filled in order; banked incrementally.

**Evidence classes** (CLAUDE.md operating standard 1): `MEASURED` (ours + artifact path) ·
`PUBLISHED` (cited) · `INHERITED` (another doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.

**Estimator, named once and used throughout:** paired episode-cluster bootstrap
(`taniteval/ci.py`, B = 2000). Resampling unit stated per test. **`overlapping_holdout_se` is never used.**

---

## 0. PRE-REGISTRATION — written before any measurement

### 0.1 Task 1 — what I am actually testing, and why it is NOT what the brief says

The brief's premise is: *"`RUN_RECIPE.md:26` says trafficsim is disabled by default, so the reactive-agent
model may simply never have been enabled — in which case the tactical gate's `agents don't react` failure
is a configuration artefact and the negative result is void."*

⚠️ **I am recording, before running, that this premise appears to be circular, and that the check for it
is the first thing I will do.** The `[−0.21, +0.14] m against a 4.5 m noise floor` figure the brief quotes
as the gate's failure is — on its face — the figure published in
`…/incoming/2026-07-26-4brain-gates/GATE_RESULTS.md` §2.4, and that document's §2.1 states that the same
session *fetched the CATK weights, built the PyG extensions and ran the service*. If so, the number the
brief wants to void **was produced with trafficsim enabled**, and "it was never enabled" cannot explain it.

**Pre-registered outcome A (premise void):** trafficsim was already enabled when the gate-2 number was
produced. Then the brief's proposed mechanism is dead, and the honest report is *"the premise is
falsified, the gate stands on this ground"* — **not** a manufactured reason to re-run.

**Pre-registered outcome B (premise live):** trafficsim was genuinely off, or non-functional, during
gate 2. Then the gate-2 verdict is void and must be withdrawn loudly.

**Either way, one open question remains and it is named by the gate document itself** (§2.4, the
"one thing that could overturn this" box): gate 2 drove the trafficsim service **directly over its own
gRPC contract**, with a session hand-built after `runtime/services/traffic_service.py` — *faithful, but
not byte-identical to a full runtime integration*. Its own stated decisive follow-up is **one full
closed-loop rollout with `trafficsim=catk`**. **That is the experiment I will run**, because it is the
only remaining way the "agents don't react" verdict could still be a harness artefact.

### 0.2 The full-runtime reactivity test — design fixed in advance

Same contrast as gate 2, so the comparison is like-for-like, but through the **complete runtime**
(wizard-generated configs → renderer + physics + controller + trafficsim + driver → `alpasim_runtime.simulate`):

| arm | ego behaviour | how |
|---|---|---|
| **GO** | ego drives forward | constant-forward driver (the M2 `simple_driver.py` policy) |
| **STOP** | ego halts and stays put | same driver, zero-velocity trajectory |
| **GO2** | identical in construction to GO | ⭐ the **stochastic-floor control** |

Non-ego agent positions are read from the runtime's own `rollout.asl` log — the runtime's record, not a
hand-built session. Statistic, fixed in advance and identical in form to gate 2:

    Δ = between_arm_mean_pairwise_distance − within_arm_mean_pairwise_distance

with `between` = GO-vs-STOP pairs and `within` = GO-vs-GO2 pairs, **like-for-like and equal under the
null**. Reaction ⇒ Δ **positive and CI-separated**. Unit of resampling = **agent**.

**Pre-registered decision rule, both outcomes committed now:**

| result | verdict I will publish |
|---|---|
| Δ **positive and separated**, concentrated in the near-ego stratum | **The tactical gate's "agents don't react" verdict is VOID.** I withdraw it, loudly, in the headline — the direct-gRPC harness was the artefact. |
| Δ **not separated**, near-ego null | **The verdict SURVIVES, and is now stronger**, having been reproduced through a second, independent integration path. |
| Δ separated but **only far-field**, near-ego null | **Weakened, not void** — reported as a partial, with the multiplicity count stated. |

**MDE, stated before running** (this is the rule that killed a control today at MDE 2.8× the leak it
existed to catch). The effect this test exists to catch is *an ego-induced displacement of nearby
agents large enough to make `Y_outcome` a function of the policy's choice.* Gate 2's best-powered scene
bounded that at ±0.21 m against a 4.5 m floor. **I will report this test's own achieved CI half-width
and state explicitly whether it is tight enough to catch an effect of the size T1–T4 need** — and if the
full-runtime test is *less* powered than gate 2's, I will say so and will **not** claim it overturns anything.

**Proof the test CAN fail (both directions, required):**
1. **Fidelity direction** — the arms must actually differ at the ego: I will measure the GO-vs-STOP ego
   separation and require it to be large (gate 2 measured 19.94 m mean / 60.99 m max). If the ego does not
   differ, the test is void and I will say so rather than reporting a null.
2. **Deliberately-failing-input direction** — a **replay control**: compare returned agent positions
   against the agents' own logged tracks. If they match, `trafficsim` is replay and the whole construct
   collapses regardless of the arm contrast.

### 0.3 Licence constraint, acknowledged before touching anything

AlpaSim's NuRec/gsplat renderer is under **NGC-DL-CONTAINER-LICENSE, which forbids derivatives.**
I will **run and configure** it only. `trafficsim=catk` is an existing wizard config option
(`src/wizard/configs/trafficsim/catk.yaml`) — selecting it is configuration, not a derivative.
**If any step requires editing renderer code I stop and report.** Recorded in §4.

### 0.4 Task 2 — wheelbase option B, pre-registered scope

Option B is **fix-forward-only**. Before executing I commit to the parity rule: **if B would change
episode selection, I stop and escalate instead of proceeding.** The concrete B is whatever
`…/incoming/2026-07-26-wheelbase-impact/WHEELBASE_IMPACT.md` §5 specifies — read first, followed as
written, not as summarised to me. Its parity statement is reproduced and checked in §5.

---

*(Sections 1–6 are filled in as the work runs. Nothing below this line was written before its measurement.)*

---

## 1. TASK 1 STEP 1 — the premise, verified. **It is FALSE, and it was falsified by our own artifact.**

### 1.1 Is `trafficsim` present, importable, functional? — YES, on four independent probes

All `MEASURED` 2026-07-26 on `tanitad-eval`, tier **CONFIRMED**:

| # | probe | result |
|---|---|---|
| 1 | package present | `…/alpasim/src/trafficsim/alpasim_trafficsim/` with `catk/`, `grpc/`, `config/`, an `.egg-info` **and populated `__pycache__`** (i.e. it has been imported before) |
| 2 | importable | `import alpasim_trafficsim, alpasim_trafficsim.catk.model_adapter` → **OK** in the alpasim venv |
| 3 | weights real | `data/trafficsim-models/catk_v120/latest.ckpt` = **69,960,427 B**, sha256 `7c5a89bc6e876c025a82572b72f87ca97dd75fe5f57245dbf2b63fc3b3c4455e` — **byte-identical to the hash `GATE_RESULTS.md` §2.1 published**. Token vocabularies present. |
| 4 | runnable | PyG extensions import (`torch_cluster` 1.6.3); `catk_trafficsim_server` entrypoint exists in the venv |

**And a fifth, which is the one that matters:** `src/wizard/configs/trafficsim/catk.yaml` exists as a **stock config group**, selected with `trafficsim=catk`. Enabling it is a **configuration** act, not a code change.

### 1.2 ⛔ The brief's premise is circular — stated plainly

The brief asks me to test whether the tactical gate's `[−0.21, +0.14] m vs a 4.5 m noise floor` failure
"may be an artefact of configuration, because trafficsim was never enabled".

**That number was produced BY the run that enabled it.** `GATE_RESULTS.md` §2.1 (dated 2026-07-26,
same day) records that the gate-2 session **fetched the CATK weights via the LFS batch API**
(sha256-verified — the same hash I re-measured above), **built the PyG extensions from source**, and
**ran the CATK service**; §2.4 then reports agents deviating from their logged tracks by **8.37–78.17 m**
with **≥95.5 % of poses not the logged pose** — i.e. `IS_REPLAY: false`, CATK genuinely simulating.

The program harvest read `RUN_RECIPE.md:26` — *"trafficsim (disabled by default)"*, written **2026-07-22**
— and inferred "never enabled". That line is still **literally true and still the default**, and it is
**stale as evidence** about a run performed four days later.

⇒ **Pre-registered outcome A. The brief's proposed mechanism for voiding the tactical gate is dead.**
The gate did not fail because trafficsim was off. I am not going to manufacture a reason to re-run it.

### 1.3 ⭐ But the premise being wrong surfaced a DIFFERENT true finding, and it is material

`src/wizard/configs/trafficsim/disabled.yaml` sets `runtime.endpoints.trafficsim.skip: true`, and
`runtime/services/traffic_service.py:simulate_traffic` shows what `skip` means (`MEASURED`, source-read):

> `if self.skip: … return traffic positions from recorded trajectories`

**`skip` is literal REPLAY.** So every closed-loop number the program has published — the REF-C n=12
suite, the flagship-vs-REF-C n=12 suite, and the native-1080 re-run (`RUN_RECIPE.md` §12, §15, §16) —
was produced against **replayed, non-reactive traffic**, because they all used the default.
`MEASURED`, tier **CONFIRMED**. This does not invalidate those results (both arms saw identical
replayed traffic, so the *paired* comparisons stand) but it does bound what they can mean: **no
published TanitAD closed-loop number has ever involved a reactive agent.** That belongs in the record
and was not previously written down anywhere.

---

## 4. WHICH of the nine 4-brain problems `trafficsim` gates — **"four of nine" VERIFIED, not inherited**

Checked against the **primary source**, `…/2026-07-26-4brain-dominance-program/STRATEGIC_TACTICAL_PROBLEM_SPEC.md`
§8 "Summary table — the nine problems" (not against `GATE_RESULTS.md`, which is a secondary read of it).
The `Source` column names the corpus each problem requires:

| # | problem | horizon | source column says | needs `trafficsim`? |
|---|---|---|---|---|
| S1 | branch selection | 10–30 s | **AlpaSim**; Cosmos-DD/nuScenes w/ adapter | ❌ |
| S2 | lane for the manoeuvre | 10–25 s | **AlpaSim** | ❌ |
| S3 | manoeuvre timing | 5–25 s | **PhysicalAI today** (20.5 %/62.9 %) | ❌ |
| S4 | roundabout exit ordinal | 10–30 s | **AlpaSim (n=8)**, L2D (3,532 eps) | ❌ |
| **T1** | yield vs proceed | 5–15 s | **AlpaSim + trafficsim ON** | ✅ |
| **T2** | gap acceptance | 5–12 s | **AlpaSim + trafficsim** | ✅ |
| **T3** | overtake vs follow | 8–15 s | **AlpaSim + trafficsim** | ✅ |
| **T4** | right-of-way | 5–15 s | **AlpaSim + trafficsim only** | ✅ |
| OP | operative (§4 of the spec) | 0–2 s | largely validated | ❌ |

**⇒ `trafficsim` gates exactly FOUR of the nine: T1, T2, T3, T4. The brief's "four of nine" is CORRECT.**
`MEASURED` from the spec, tier **CONFIRMED**.

Three qualifications the bare figure hides, all worth more than the count:

1. **It is a *different* four from gate 1's four.** Gate 1 (VectorMap) unblocks **S1, S2, S4 + HP-4** —
   the strategic half. The two gates do **not** overlap, so "four of nine" is not a subset of an
   already-solved set; between them the two gates cover **seven** of the nine.
2. **T4 has no alternative source at all** — the spec says *"AlpaSim + trafficsim **only**"*. T1–T3
   list AlpaSim + trafficsim; T4 has no fallback corpus whatsoever. It is the most exposed of the four.
3. ⚠️ **The dependency is stronger than "needs traffic present".** All four define `Y_outcome` as a
   *simulated consequence*, and their whole admissibility argument is *"a consequence cannot be circular
   with any model input by construction"*. That argument needs the consequence to be **a function of the
   policy's choice** — not merely for traffic to exist. So the gate is not "is trafficsim installed"
   (it is, §1.1) but "does the simulated world respond to the ego", which is what §2 measures.

---

## 5. TASK 2 — wheelbase option B, EXECUTED

*(Numbered 5 to keep §2–4 for the trafficsim measurement, which runs longer. Task 2 is complete.)*

**Source followed:** `…/incoming/2026-07-26-wheelbase-impact/WHEELBASE_IMPACT.md` §5 "The concrete B",
read in full first. Its four numbered steps are implemented below in its order, plus the §6 corrections
that it marks as owed *regardless* of the decision.

### 5.1 🔴 The parity statement — what changes and what provably does not

| | changed? |
|---|---|
| **Episode selection** | ❌ **NO.** `cache_key` hashes the *ordered clip-id list* + build params. Option B alters label VALUES for future builds; it touches no clip list, no split, no skip set. **Nothing re-selects episodes, so nothing had to be refused or escalated.** |
| `physicalai-train-e438721ae894` (2,376 eps, skip-hash `f09e44db`) | ❌ **NO — byte-identical, enforced in code.** |
| Any existing cache, arm, checkpoint or published number | ❌ **NO.** Option B is fix-**forward** only. |
| Default behaviour of `build_episode` / `signals_at` | ❌ **NO.** Both default to the legacy 2.9 m constant. |
| Future builds launched with `--wheelbase-mode per_clip_v1` | ✅ **YES** — and they mint a **different cache key**, so they can never be compared against a legacy arm by accident. |

**Are previously-published numbers now incomparable?** **No — and that is the point of choosing B over
A.** Option A (re-mint labels + re-baseline) *would* have made all 27 recomputed arms incomparable, for
a measured **+0.0056 m** on a number whose own CI half-width is **0.060 m**. B introduces a **regime
boundary** instead: legacy numbers stay exactly comparable to each other, corrected numbers will be
comparable to each other, and the two sets are separated by a cache key rather than by a promise in
prose. **The one thing a future agent must not do is run a paired test across the boundary** — which is
why it is now registered in `MODEL_REGISTRY.md` §0.1.1 as a named regime boundary.

### 5.2 The mechanism — why parity is preserved *by construction*, not by care

The load-bearing line is `physicalai.label_params()`:

```python
if wheelbase_mode == WHEELBASE_MODE_LEGACY:
    return {}                                   # contributes NOTHING to the key
return {"wheelbase_mode": wheelbase_mode}       # a corrected cache CANNOT collide
```

Both build scripts splat it: `params = {..., "calib": "ftheta_v2", **label_params(args.wheelbase_mode)}`.
Under the legacy default the dict is **bit-identical to the one that minted `e438721ae894`**, so
`rebuild_pai_rolling.py`'s existing determinism oracle (`ABORT: key mismatch`) still passes unchanged —
and that oracle is now *also* the thing that makes crossing the regime boundary impossible by accident,
because a corrected build must be launched with its own `--expect-key`.

⚠️ **The trap this avoids, stated because it is the obvious wrong implementation:** adding
`"wheelbase": ...` to the params dict *unconditionally* — even set to the legacy value — re-keys **every
cache in the program** and silently voids the parity guarantee. `label_params` returns `{}` rather than
`{"wheelbase_mode": "const2p9"}` precisely to prevent that, and the test below pins it.

### 5.3 What was implemented

| # | §5 step | where |
|---|---|---|
| 1 | do not touch the constant for existing arms | `physicalai.py` — `WHEELBASE = 2.9` **unchanged in value**; only its comment was corrected (§6 row 1) |
| 2 | per-clip `vehicle_dimensions` join, resolved **exactly like `intrinsics_for_clip`** | `physicalai.wheelbase_for_clip()` — local CSV (`$TANITAD_PAI_WHEELBASE` or `<root>/calibration/physicalai_wheelbase.csv`) → per-chunk `calibration/vehicle_dimensions` parquet (downloaded on demand) → **loud** fallback, never silent. `strict=True` in a corrected build **refuses** rather than falling back. |
| 3 | extend the cache-key params with the label regime | `physicalai.label_params()`; wired into `build_pai_cache.py` and `rebuild_pai_rolling.py`, both with a new `--wheelbase-mode` |
| 4 | register the discontinuity | `MODEL_REGISTRY.md` **§0.1.1**, new subsection, citing the measurement |

A deliberate structural note: `signals_at` had **no clip in scope** (it takes only the egomotion frame
and query times), so the wheelbase is threaded as a defaulted parameter and resolved one level up in
`build_episode`, which does have `clip["clip_id"]` and can recover the corpus root from the mp4 path the
same way `_decode_mp4` already does for calibration. `_veh_rows` is a deliberate sibling of
`_front_wide_rows`, **not** a reuse: `vehicle_dimensions` is per-clip, has no `camera_name` column, and
filtering it through `_front_wide_rows` would silently return an empty frame.

### 5.4 The guard proves it can fail — both directions

`stack/tests/test_wheelbase_regime.py` — **14 tests, all passing** (`MEASURED`, tier **CONFIRMED**).
Existing suite unaffected: `test_physicalai`, `test_physicalai_signals`, `test_physicalai_rig`,
`test_epcache`, `test_epcache_key`, `test_parity_manifest`, `test_cosmos_drive` → **65 passed**.

| direction | test | what it would catch |
|---|---|---|
| **fidelity** | `test_legacy_regime_contributes_no_build_param`, `test_legacy_params_dict_is_bit_identical` | the exact failure that would void parity — a key that moves. Asserts `cache_key(srcs, merged) == cache_key(srcs, CANONICAL)`. |
| **fidelity** | `test_signals_at_default_is_the_legacy_constant` | a defaulted parameter that silently changes existing labels |
| **fidelity** | `test_steer_uses_the_supplied_wheelbase` (×5 real values), `test_gain_matches_the_measured_range` | that the correction has the **measured magnitude and direction** — gains pinned to 2.730/2.9 = 0.9414 and 3.216/2.9 = 1.1090, the envelope `WHEELBASE_IMPACT.md` §2.1 measured |
| **deliberately failing** | `test_per_clip_regime_changes_the_key` | two regimes sharing a key |
| **deliberately failing** | `test_unknown_regime_is_refused` | a typo'd regime silently building |
| **deliberately failing** | `test_strict_resolution_raises_when_unresolvable` | ⭐ **a half-corrected cache** — clips that fail to resolve silently reverting to 2.9. This is the one that matters most: without it a `per_clip_v1` build could mint a mixture and label it corrected. |
| **deliberately failing** | `test_nonstrict_resolution_falls_back_loudly` | a silent fallback (asserts the word `APPROXIMATION` is actually printed) |

**MDE, against the effect each guard exists to catch.** These are *exact* guards, not statistical ones —
the quantity under test is a hash equality and a floating-point identity, so the detectable effect is
**any** difference at all: one changed key, one changed label, one unresolved clip. There is no MDE gap
of the kind that killed today's controls, because there is no sampling involved. Stated explicitly
rather than left implicit, since "the guard can't fail" is the class this section exists to close.

### 5.5 §6 honesty corrections — done, and one deliberately NOT done

| # | correction | status |
|---|---|---|
| 1 | `physicalai.py:51` comment claimed a "Hyperion platform class" justification | ✅ **rewritten** as an explicit, dated, cited approximation carrying the five real values, the clip-mean 2.9568, and the measured impact |
| 2 | `cosmos_drive.py:63` justified 2.9 by cross-reference to that false claim | ✅ **rewritten**; Cosmos-DD's true wheelbase recorded as **UNKNOWN — not probed**, with a do-not-assume warning |
| 3 | `GEOMETRY_INTEGRITY_AUDIT.md` "real 2.85 (1.5 %)" | ✅ **corrected in place, BOTH sites** (`:40` and `:77–78`), citing the measurement and the C2 entry — see the self-correction below |
| 4 | `RETRACTION_LOG.md` entry, class **C2** | ✅ **appended** — plus the *reason* one chunk misled: `I(wheelbase; country) = 0.769/1.880` bits, so chunk shards are **systematically**, not noisily, biased for this variable |
| 5 | `l2d.py` / `nuscenes.py` are correct, do not touch | ✅ **untouched** (2.72 Kia Niro, 2.588 Renault Zoe — published specs for single-vehicle corpora) |
| 6 | `closedloop.py:99` 2.7 train/serve skew | ⚠️ **DOCUMENTED IN CODE, VALUE DELIBERATELY UNCHANGED — and escalated.** See below. |

### 5.6 ⚠️ A C2 I committed IN THIS DOCUMENT and caught before shipping it

I first wrote row 3 as ***"NOT DONE — file not found in this repo"***, on the strength of a repo-wide
`find` and a content grep run from the wrong starting directory. **The file exists**, at
`Benchmarks & Eval/GEOMETRY_INTEGRITY_AUDIT.md`, and carries the stale claim at **two** sites (`:40`
and `:77–78`), not the one the measurement listed. The second probe — `git ls-files | grep -i` — found
it immediately.

That is **exactly the C2 class** (*absence from a single probe*) that this same section was appending a
C2 retraction about, and I nearly published it inside the correction. Logged here rather than quietly
fixed, because the root-cause class is the reusable part: **`find` from an assumed subtree is one probe;
`git ls-files` is the tool that owns the fact.** Both sites are now corrected.

⚖️ **Why I did not change `closedloop.py`'s 2.7 to 2.9.** The measurement calls it "the cheapest thing to
fix in the whole report", and it is — but it is not *free*, and the report itself offers "align **or**
document". Aligning moves **every closed-loop number produced from here** and breaks comparability with
the published n=12 REF-C, flagship-vs-REF-C and native-1080 suites. That is a PI call about a published
series, not an agent's judgement call, so I documented the skew precisely at the constant (path
invariance ✅, action skew +7.41 %, open-loop cost +0.0026 [−0.0006, +0.0062] not separated) and
**escalated the value change** rather than making the record inconsistent without a decision.

### 5.7 Test-suite state

`MEASURED` 2026-07-26 after all edits: **`stack` 1119 passed, 7 skipped** · **`taniteval` 449 passed**.
Green before staging, as `CLAUDE.md` requires.

---

## 6. Deliverable manifest

**STAGED (`git add`), NOT committed, NOT pushed.** Branch `agent/benchmarks-eval-20260721`.

| artifact | what | where |
|---|---|---|
| `TRAFFICSIM_WHEELBASE.md` | this report | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-trafficsim-wheelbase/` |
| `ts_scene_pick.py` | USDZ→`obstacle.parquet` dynamic-agent census over all 51 scenes | same dir · `tanitad-eval:/workspace/ts_scene_pick.py` |
| `ts_wizard_gen.sh` | wizard config gen with **`trafficsim=catk`** (configuration only) | same dir · `tanitad-eval:/workspace/ts_wizard_gen.sh` |
| `ts_master.sh` | 3-arm × R-repeat full-runtime driver (GO / STOP / GO2) | same dir · `tanitad-eval:/workspace/ts_master.sh` |
| `ts_reactivity.py` | ASL parser + paired episode-cluster bootstrap + both controls | same dir · `tanitad-eval:/workspace/ts_reactivity.py` |
| `artifacts/ts_scene_pick.json` | per-scene agent census (51 scenes) | same dir |
| `artifacts/ts_reactivity.json` | **the reactivity result** — raw JSON per number | same dir |
| **code — per-clip wheelbase (option B)** | `label_params`, `wheelbase_for_clip`, `_veh_rows`, `_load_chunk_wheelbase`, `_wheelbase_from_parquet`, `_wheelbase_table`, `signals_at(wheelbase=)`, `build_episode(wheelbase_mode=)` | `repo:stack/tanitad/data/physicalai.py` |
| code — build wiring | `--wheelbase-mode` + `label_params` splat | `repo:stack/scripts/build_pai_cache.py`, `repo:stack/scripts/rebuild_pai_rolling.py` |
| code — tests | 14 tests, both directions | `repo:stack/tests/test_wheelbase_regime.py` |
| doc-in-code corrections | the 2.9 comment; the Cosmos cross-reference; the closedloop 2.7 skew | `repo:stack/tanitad/data/physicalai.py`, `repo:stack/tanitad/data/cosmos_drive.py`, `repo:taniteval/taniteval/closedloop.py` |
| registry | **§0.1.1 named regime boundary** | `repo:Project Steering/MODEL_REGISTRY.md` |
| retraction log | **two** new rows: the wheelbase **C2**, and the trafficsim **C1** circular-premise | `repo:Project Steering/RETRACTION_LOG.md` |
| audit correction | both stale wheelbase sites | `repo:Benchmarks & Eval/GEOMETRY_INTEGRITY_AUDIT.md` |
| **pod only (regenerable)** | `/workspace/tsreact/clipgt-41c06176…/{GO,STOP,GO2}_r{1,2,3}/` rollouts, `rollout.asl`, per-arm service logs, `/workspace/tsreact_{smoke,full}.log` | `tanitad-eval` — **regenerable via `ts_master.sh`**; the extracted numbers are in the repo JSON |

⚠️ **Nothing produced by this session lives in only one place** except the raw ASL rollouts, which are
regenerable from the staged scripts and whose every extracted number is in `artifacts/ts_reactivity.json`.

⚠️ **Concurrent staging note:** `MODEL_REGISTRY.md` and `RETRACTION_LOG.md` were **already modified by
sibling agents** when I edited them (`git status` showed them dirty before my first write). My additions
are confined to a new `§0.1.1` and two appended table rows respectively; whoever commits must expect
other agents' edits in those two files and should say so in the message rather than splitting them out.

## 7. Escalations — stated here, not left in a README

1. ⛔ **The brief's premise for voiding the tactical gate is FALSE and should stop propagating.** The
   program harvest's *"trafficsim was never enabled"* came from a doc **older than the run it impeaches**.
   Anything downstream that treats the tactical gate as "probably a config artefact" needs correcting.
2. 🔴 **`taniteval/closedloop.py` wheelbase 2.7 vs the 2.9 the models trained on — needs a PI decision.**
   Documented in code, value unchanged. Aligning it is one constant, no retrain, no parity impact, but
   it moves every future closed-loop number and breaks comparability with the published n=12 suites.
3. 🟡 **No published TanitAD closed-loop number has ever run against reactive traffic** (§1.3). If any
   downstream claim depends on agents responding to the ego, it is unsupported by those suites.
4. 🟡 **A corrected-regime cache does not exist yet.** Option B is *capability*, not a built artefact:
   the first `--wheelbase-mode per_clip_v1` build must be commissioned deliberately, with its own
   `--expect-key`, and must NOT be paired against any legacy arm.
