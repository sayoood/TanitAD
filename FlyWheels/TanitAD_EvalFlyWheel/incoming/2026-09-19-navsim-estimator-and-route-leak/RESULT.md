# RESULT — W2 (E3): the NavSim estimator is SETTLED, the route leak is SETTLED, and the four blocking NavSim gates are now MACHINERY

**Stream** E3 → **W2** of the eval-suite build (`…/2026-09-19-eval-suite-build/BUILD_PLAN.md`) ·
**2026-09-19/20** · **Package** = this folder · CPU only, no dataset downloaded by this stream.
Every number is **MEASURED by this stream** unless marked otherwise; each names its artifact.

## 0. Headline

1. ⭐ **The estimator is settled: a LOG-CLUSTER bootstrap** — clusters = the OpenScene
   `log_name`, B = 2000, n_clusters ≥ 8 (RG-14) — **of the devkit's OWN aggregate**, pre-registered
   in `SPEC.md` (sha256 `bc4d8967…`, written before any interval existed) and implemented in
   `taniteval/adapters/navsim_ci.py`. It reproduces the official value exactly on a real run
   (E1's CV warmup: **0.1853562745165113, |Δ| 0.0**) and matches the devkit's own aggregation
   function on four fixtures, including **physically duplicated logs** (12/12) — the property the
   bootstrap rests on.
2. ⛔ **`tools/criteria_check.py` evaluated NONE of `benchmarks.navsim`** (measured by E1 and
   independently by E2). It now evaluates the five criteria and all four BLOCKING gates, with
   **26 mutation arms that must go RED**, plus a protocol-tag union check. Registry → **2.10.0**.
3. ⛔ **`EPDMS_v2.multipliers` omitted TLC** while the same registry's `EPDMS_submetrics`
   multiplied by it. Fixed to the literal `["NC","DAC","DDC","TLC"]` (devkit `pdm_enums.py:165-171`
   @0a380a9), with a cross-block consistency test — the old test had PINNED the defect.
4. ⭐ **Route leak: PARTIAL — a ROUTE-LEVEL ORACLE.** `driving_command` is a function of (the ego
   pose NOW, the nuPlan route, the map) and nothing else — reproduced **1,902/1,902** with
   OpenScene's own function — but the route IS the expert's driven path at roadblock granularity
   (**79.8 %** of scenes vs a **34.6 %** no-future null control; it starts BEHIND the ego in
   **83.5 %**). It is **not** a trajectory leak (best future-path rule **94.2–94.7 %** < the 99 %
   bar). ⇒ a command-conditioned NavSim number is *driving with an oracle route*.
5. ⛔ **Stage 2's command and route are COPIED from the expert's own frame** at the same timestamp
   (**5,462/5,462** navhard, **204/204** warmup) — not recomputed at the perturbed start
   (recomputation agrees on only 83.9 %). A synthetic start displaced from the expert therefore
   receives the command for where the EXPERT was.
6. **Integration asks (§6):** E2's artifacts now FAIL the settled route-leak criterion (they carry
   the pre-settlement "UNVERIFIED") and E1's human artifact FAILED the ego gate on an incoherent
   `vision_only` — **both are fixed by rebuilding with the updated adapter** (verified: E1's own
   builder, re-run unchanged against the new adapter in a scratch copy, produces **0 violations**).
   W1: `navsim_ci.interval_from_run` is ready for navhard; it needs the **pre-CSV frame with its
   `weight` column** (the published CSV drops it).

---

## A. The estimator

### A.1 The census — what the cluster structure actually is (MEASURED)

`code/split_census.py` → `raw/split_census.json`. Read controls: **147/147** log pickles read, 0
failures, **75,122** key frames. Every stage-1 scene list is derived TWICE — the devkit's own
`filter_scenes` (its source text executed verbatim from the pinned blob) and an independent
re-derivation — and the two **agree token for token on all 7 local splits**.

| split | stage-1 | stage-2 | mapping keys | `log_name` | drives | cities | scenes/log (min·med·max) |
|---|---|---|---|---|---|---|---|
| **navhard_two_stage** | 450 | 5,462 | 225 | **76** | 34 | 4 | 2 · 4 · 30 |
| **navtest** | 12,146 | — | — | **136** | 44 | 4 | 8 · 70 · 568 |
| navtest_two_stage | 11,989 | — | — | 136 | 44 | 4 | 5 · 70 · 568 |
| warmup_two_stage | 16 | 204 | 8 | **7** | 7 | 3 | 2 · 2 · 4 |
| navsafe_two_stage | 164 | 978 (yaml) | — | 36 | 18 | — | — |
| test (OpenScene standard, CONTROL) | 5,044 | — | — | 138 | 44 | 4 | 3 · 27 · 223 |
| private_test_hard_two_stage | 140 (yaml) | 1,732 (yaml) | `null` (private) | **unobservable** | — | — | — |
| navtrain | 103,288 (yaml) | — | — | 1,192 | 162 | — | — |

**Overlap — where the dependence lives.**
* navtest scenes are 14-frame windows at a stride of ONE key frame: consecutive scenes share a
  median **13 of 14** frames, **92.6 %** of consecutive pairs share ≥ 1 frame, and each key frame
  sits in **5.55** scenes (frame-reuse factor). navhard: a key's `orig`/`prev` are adjacent frames
  (Δ = 1 frame = 0.5 s, **225/225**) sharing **11 of 12** frames, and the EC term couples them.
* **The control reads its known value:** the OpenScene standard `test` split (`frame_interval:
  null`) has frame-reuse factor **1.0000** and **0** of 158,789 within-log pairs sharing a frame.
* **Across logs nothing is shared**: 0 time-overlapping adjacent segment pairs, 0 shared tokens,
  minimum gap **11.5 s** (navhard median 109.5 s).
* **Stage 2 never leaves its log**: every synthetic scene's own `scene_metadata.log_name` equals its
  key's stage-1 log — **5,462/5,462** (navhard), **204/204** (warmup) — and its
  `corresponding_original_scene` is the stage-1 scene's final (t = +4 s) frame in every case
  (`code/two_stage_synthetic_census.py`, `raw/two_stage_*.json`).
* navhard keys within one log do not share frames (gap ≥ 16 frames = 8 s) but are close in time
  (median 40 s; **19.5 %** ≤ 12 s) — `raw/key_gap_within_log.json`.
* **Split nesting (MEASURED, `raw/stage2_command_provenance.json`):** warmup's 16 stage-1 tokens are
  ALL navhard tokens (16/16) and 11 are navtest tokens; **367 of navhard's 450** are navtest tokens.
  ⇒ warmup is a subset of navhard; navhard and navtest overlap heavily.
* **private_test_hard is unobservable locally**: 0 of its 140 tokens appear in the test metadata —
  beside the control that makes the 0 meaningful: **450/450** navhard tokens found.

### A.2 The rule (pre-registered in `SPEC.md`, hash `raw/SPEC_PREREG_HASH.txt`)

Cluster unit **`log_name`** (primary), **`nuplan_drive`** as a sensitivity arm; a paired lever claim
is `separated` only if separated under **both** units (no tunable threshold), and — per
`H-ESTIM-SEED-1` — a separated interval is **necessary, not sufficient**: the record carries the
sentence naming the two variances it is blind to. Floor **8** (RG-14). Ceiling: the split's log
count (more clusters than logs ⇒ a finer unit was resampled ⇒ FAIL). B = 2000, percentile, seed 0.
**Per split: navhard ✅ (76/34) · navtest ✅ (136/44) · ⛔ warmup NEVER (7 < 8) · ⛔ private_test_hard
NEVER (log identities not released).**

### A.3 The statistic is the devkit's own — and that is CHECKED, not assumed

`taniteval/adapters/navsim_ci.py` computes each unit's contribution exactly as
`run_pdm_score.py::calculate_individual_mapping_scores` (L242-290 @0a380a9) does, with its NaN
semantics (empty group / zero total weight / a NaN score all propagate; `isin` counts a row once);
the aggregate is then the skip-NaN mean the devkit takes. **Because a unit's contribution depends on
one log only (§A.1), resampling logs reproduces the official aggregation on every draw.**

| check | result | artifact |
|---|---|---|
| F1 — a key computed **on paper** | devkit 0.44, ours 0.44 (|Δ| < 1e-12) | `raw/devkit_aggregation_reference.json` |
| F2 — 40 keys / 10 logs with NaN, missing-row, zero-weight and duplicate cases | devkit **0.2886492611337752**, ours identical to 1e-12; 3 NaN key contributions reported, not hidden | same |
| F3 — dropping whole logs | **12/12** subsets agree | same |
| F4 — **physical bootstrap replicates** (logs drawn with replacement, a repeated log's tokens renamed so the devkit sees two copies) | **12/12** agree | same |
| version invariance | identical under pandas 3.0.3 / numpy 2.5.1 **and** the NavSim runtime's pandas 2.3.3 / numpy 1.23.4 | `raw/devkit_aggregation_reference_navsim_runtime.json` |
| **real run** — E1's CV warmup pre-CSV frame + the warmup mapping | ours **0.1853562745165113** = the devkit's `extended_pdm_score_combined`, **|Δ| 0.0** (E1's rerun R1: 2.8e-17, its own C7 float-order finding) | `taniteval/tests/test_navsim_ci.py::test_T_REPRO…` |

⛔ **The published CSV cannot carry an interval**: `run_pdm_score.py:422` keeps only
`token/valid/score_cols`, so the stage-2 Gaussian **weights** (and `log_name`) are gone. An interval
needs the pre-CSV frame — E1's `*_final_scores_frame.csv` has exactly the right columns.

### A.4 Analytic tests and the RED mutations (`taniteval/tests/test_navsim_ci.py`, 20 tests)

| id | expectation (a LITERAL, not an expression over the code) | measured |
|---|---|---|
| T-ZERO | identical contributions ⇒ lo = hi = 0.75, se = 0.0 **exactly** | ✅ |
| T-TWO | 2 clusters ⇒ draws ∈ {0.0, 0.5, 1.0}, E[θ*] = 0.5, lo 0.0 / hi 1.0 | ✅ (mean 0.4928) |
| T-SE | 10 logs × 20 identical-within-log scenes ⇒ SE = `√(0.0825/10)` = 0.09083 | **0.0902** (0.7 % off) |
| **T-MUT** | the SAME fixture with unit-level resampling inside the machine | **0.0205 — 4.4× narrower; the analytic target goes RED** |
| T-MUT (caller) | scene tokens passed as clusters | `n_clusters` 200 > the split's 10 logs ⇒ admissibility **FAIL** |
| T-FLOOR | 7 logs | UNAVAILABLE, n = 7, reason names the floor 8 |
| T-PAIR | a constant +0.1 offset | delta = lo = hi = 0.1, separated under BOTH units |
| T-PAIR (asymmetric NaN) | arms that failed on different units | REFUSED (RG-13) |

⭐ **Why both mutation arms exist.** The code mutation's record still reads `n_clusters = 10` — the
metadata is produced by the same code, so **only the independent analytic target catches it**. The
caller defect the metadata *can* see, and that is what the registry gate checks.

### A.5 Registry 2.10.0 + the checker

* `GATE_estimator_cluster_unit`: UNRESOLVED → **SETTLED**, with `cluster_unit`, `min_clusters` 8,
  `n_boot_min` 2000, `aggregation_by_protocol`, `max_clusters_by_protocol` (**76 / 7 / 136 / 136 /
  null**), `can_rule_by_protocol`, the census path and the SPEC path.
* `EPDMS_v2.multipliers` **+ TLC**; `PDMS_v1` gains its required pin.
* `GATE_no_cross_protocol_comparison` gains **`devkit_pins`** per protocol — a **PDMS_v1 number
  stamped with the v2 SHA now FAILS** ("v1 PDMS from `main` silently computes EPDMS").
* New `benchmarks.navsim_v1` block (formula `NC·DAC·(5EP+5TTC+2C)/12`, v1.1 SHA `3e8291b…`, the
  per-file sha256 of the five v1.1 sources read, aggregation `single_stage_token_mean` from
  `run_pdm_score.py:144`, split census).
* New `ROUTE_LEAK_VERDICT` (§B), consumed by `navsim.py::route_leak_check` (two-sided pin).
* **`protocol_tags`**: the union of three blocks that stay separate (NavSim's closed set ·
  nuScenes' two planning tags · the internal `TanitAD_T1_refc_physicalai`), computed by
  `criteria_check.registered_protocol_tags()`. A tag outside the union is a **violation**; a test
  pins the union equal to **W1's `summary.schema.json` enum** (8 tags, currently equal).
  ⚠️ A bare top-level `protocol` STRING counts as a tag only on a suite record (`schema:
  taniteval.bench.*`) — MEASURED: 24 banked artifacts carry a free-text `protocol` sentence, and
  reading those as tags would have manufactured violations.
* `tools/criteria_check.py` now evaluates the NavSim block (`check_navsim`, `check_benchmarks`,
  `registered_protocol_tags`, `artifact_protocol_tags`, `check_protocol_tag` are public for W1),
  renders a `BENCHMARK GATES` section, aggregates them in the census, and returns them under
  `result["benchmarks"]` (additive — W1's `--json` consumers keep working).

**Gate semantics** (each with its mutation arm; `tools/tests/test_criteria_check.py`):
PASS = the declared, admissible form · REFUSED = an honest `{UNAVAILABLE, reason, n}` (a work item)
· **FAIL** = anything else, reported in the ABSENT column with a `FAIL —` detail.
A headline that could not be computed (W1's navhard CV crash in the official aggregation) is a WORK
ITEM and **still passes the structural gates**; a missing estimator declaration in the same artifact
is a VIOLATION — pinned by `test_a_crashed_headline_still_passes_the_STRUCTURAL_gates`.

### A.7 Registry 2.10.2 — W6's nuScenes "not a criterion" guard, integrated

W6 (nuScenes) left `…/2026-09-19-nuscenes-planning-harness/code/criteria_guard.patch`. Its **code
and test hunks applied verbatim** (`git apply --include=tools/criteria_check.py
--include=tools/tests/test_criteria_check.py`, clean); its **registry hunk was refused** because the
registry had moved to 2.10.1 while W6 was writing, so the same gate object was landed here as
**2.10.2** (`code/registry_update_v2_10_2.py`), attributed to W6.

* `EXTERNAL_ONLY` scope: a nuScenes open-loop planning record declaring `claim_bearing: false` is
  **reported and never scored, never counted as compliant**; `claim_bearing` true or absent is a
  VIOLATION; and its numbers inside an in-scope TanitAD driving artifact are a VIOLATION **even when
  the flag says false** (H-EVAL-6 SUPPORTED, D-BENCH-PORT SKIP claim-bearing).
* ⭐ **W2 added a SECOND detection mechanism** — a protocol tag matching `nuScenes_OL_*` — because a
  suite record carries no planning block at all and one detector is one detector. Two new RED
  mutation arms (tag + `claim_bearing` true/absent → violation; the tag inside a driving artifact →
  violation) with a **NavSim-tag control** that must read `none`.
* Guard tests: **10 pass** (W6's 5 + W2's 2, plus the union/tag tests that touch them).

### A.6 Registry 2.10.1 — a modality claim inside the modality gate was WRONG (W4 flagged, W2 verified)

`GATE_modality_label.why` said *"Drive-JEPA's 93.7 PDMS headline uses 1024x256 stacking
FRONT+LEFT+RIGHT, while its perception-free row uses the front camera only at 512x256"*. I re-read
the banked primary rather than take the correction on trust — `TanitAD Research
Lab/Library/papers/2601.22032_…pdf`, sha256 `d88c053a67ac…` (**== `library.json`**):

* **Table 8, p.14** — Transfuser **1024×256** · HydraMDP++ 1024×256 · DriveSuprim 1024×256 ·
  GoalFlow 1024×256 · iPad **4×768×432** · **Ours 2×512×256**, with the text *"…follow Transfuser in
  using an input resolution of 1024×256, formed by stacking the front, left, and right camera
  images … Our setting uses only the front camera at 512×256. We include both I_t and I_t−1…"*;
  **§4.2, p.7**: *"We use only the front-view camera, resized to 512×256."*
  ⇒ the three-camera stack is **Transfuser's**; **Drive-JEPA is front-camera-only in every row**.
* **Table 2, p.8** — the perception-free block prints an `Inputs` column: LAW **C & L**,
  World4Drive **C & L**, Epona **Camera**, Drive-JEPA **Camera**. ⇒ the ladder is **not**
  front-camera-only; the key was renamed `comparable_ladder_perception_free` (the old name now
  holds a string, so a consumer indexing it fails loudly), each rung carries its printed inputs and
  backbone, and Epona's **86.2** (Table 2) vs **86.1** (Table 1, same paper) is recorded.
* `published_reference_numbers` gained the qualifiers W4 verified (INHERITED, their
  `published_results.json` is the primary): PDM-Closed **51.3 is pre-#151**, 56.6 post-fix; **DrivoR
  56.3 = DrivoR + 134k synthetic + TOAD search** (navtrain-only 48.3); the navtest ladder mixes
  ResNet34 and ViT/L backbones; Drive-JEPA 93.3 is its own checklist misquoting 93.7.
* ⚠️ **The superseded sentence is still quoted outside the registry** — MEASURED:
  `…/2026-09-19-leaderboard-currency/raw/external_field_sources.json`, and the orchestrator says it
  reached the W1/E2 briefs. Flagged, not edited (not my files).

Pinned by two tests: the ladder/inputs literals, and a regression arm that fails if the Drive-JEPA
attribution returns.

### A.8 Beyond NavSim — an UNDEFINED lateral term now REFUSES, and `cross_mae_m` says what it is

W1 found both while scoring the STOP floor; the fixes belong at the source, and they reach every
four-family table the programme publishes.

**(1) `None` is not a refusal — fixed in `taniteval/taniteval/four_families.py::lateral`.**
When every step of a plan is below `min_ds_m` (a stationary plan has no path tangent) the heading,
yaw-rate and curvature errors are genuinely UNDEFINED. The function returned `None`, and
`criteria_check.classify` read a null exactly as it reads a MISSING key -> **three binding LATERAL
criteria came out as silent-omission violations of an instrument that had honestly declined**
(MEASURED 2026-09-19 by W1 on the NavSim STOP floor). The four terms now emit
`{status: UNAVAILABLE, reason, n, n_steps_total, min_ds_m}` — never null, and never `0.0`, which
would read as *perfect lateral agreement from a car that never moved*. ⛔ **The defined branch is
byte-identical: no banked number moves.**
Checker half: `classify` now **diagnoses a present-but-null key separately** ("a null is not a
refusal: it carries no reason and no n") from a missing one, because the fixes differ — a missing
key needs an emitter, a null needs a refusal. W1's artifact-seam repair
(`bench/navsim/artifacts.py::refuse_undefined_lateral`) stays as the regression guard; it now also
reports terms already refused upstream, so its contract (and W1's test) still holds.

**(2) `cross_mae_m` is a lateral OFFSET, not a cross-track error — VERIFIED AT SOURCE.**
`_seq_geometry` returns `"cross": p[..., 1][:, 1:]` (**`four_families.py:187`**, identical at HEAD) — the ego-frame **y column** with the origin prepended — and `lateral()` differences it at `:776`, the line the orchestrator flagged, reporting `mean |y_pred - y_gt|` at **matched time index**. There is no
projection onto the GT path, no arc-length matching, and no rotation into the GT tangent frame.
=> **two arms whose plans have y == 0 score identically however differently they drive**: W1
measured CV and an all-zero STOP plan both at **1.0658 m** on warmup; I reproduced the mechanism
analytically (`taniteval/tests/test_four_families_lateral_undefined.py`) — on a curving GT the
stationary arm's `cross_mae_m` equals the **GT's own mean |y|** to 5e-5, i.e. it is a property of
the human's path, not of the arm. It is a lateral ERROR only while the two paths progress
together; a large along-track error makes it uninterpretable. Every lateral block now carries
`_cross_is` (the sentence), `_along_mae_m_for_context` (the number that says whether it is
informative) and `_projection_based_alternative` (`lateral.py::frenet_dense` ->
`headline.pathgeom_crosstrack_m`). ⛔ **No banked number was rewritten** — the deliverable is the
qualifier and the refusal, not a re-scoring.

**(1b) ⛔ The same defect had a WORSE half, and it was reachable — `yaw_rate_mae_degps` emitted
`NaN`.** A comment at `n_steps_yaw_rate` (2026-08-23) records that the term is masked by
`both_pair` while being emitted under the HEADING count, so an arm with valid single steps but no
valid PAIR emits NaN; the author named the right denominator and deliberately left the value alone.
**I probed whether that branch is reachable rather than arguing it: it is.** MEASURED 2026-09-20 on
a plan that advances on alternate steps only — `n_steps_heading` **24**, `n_steps_curvature` **0**,
`yaw_rate_mae_degps` = **nan**, and `json.dumps(..., allow_nan=False)` refuses it. ⚠️ **A NaN is
worse than the null this task is about**, because it is a `float` and therefore passes every
`isinstance(v, (int, float))` guard my audit found — `summarize.py` would have written it straight
into a published `metrics` block as a number. The guard is now `n_curv`, so that case refuses.
⛔ **Still no number is rewritten: `n_curv > 0` implies `n_head > 0`, so every case that produced a
real value produces the identical value** — the test recomputes it from the raw geometry to prove
it. Only the NaN and the old `None` became refusals.

**(3) The downstream audit the type change demanded — one renderer needed a 6-line guard.**
A refusal is a *block* where a *scalar* used to be, so I traced every live consumer of these four
keys rather than assuming: `bench/navsim/summarize.py` and `benchreport/render.py` already require
`isinstance(v, (int, float))` (and summarize even reads `v["reason"]`); `tools/release_gate.py`'s
`_num` finds no numeric sub-key and returns `None`, which routes the term into the RG-02 refusal
path it was built for — now WITH a reason, which is what RG-02 asks for; `stack/scripts/lan_probe.py`
does `abs(cv - fv)` unguarded, but that already raised `TypeError` on the old `None`, so nothing
regresses. **The one real defect** was `taniteval/tools/refav1_openloop_report.py::_f`, whose
`except (TypeError, ValueError): return str(x)` fallthrough would have printed a ~600-character
dict into a single markdown table cell. It now renders `REFUSED (n=0) — <reason…>`; `None`, floats,
bools and non-refusal dicts are byte-identical, and the test pins all four.

Tests: 13 new in `taniteval/tests/test_four_families_lateral_undefined.py` (refusal shape · the
never-null/never-zero arm · the defined branch unchanged · the one-moving-step boundary · the
blindness reproduction · the GT-excursion identity · the qualifier on a moving arm · and a
parametrised checker arm per criterion with the **null mutation** that must go RED and the
missing-key control · **the reachable-NaN arm** · **and the renderer arm**: a long refusal must come out as a short token with
its `n`, with `None` / float / bool / non-refusal-dict controls that must be unchanged).

### A.9 The gate did NOT block a correct artifact — it was read from a STALE COMMIT (registry 2.10.3)

E1 reported that `GATE_estimator_cluster_unit` "still reads *until the unit is settled and
PRE-REGISTERED, the ONLY admissible interval is {status: UNAVAILABLE}*" and was therefore forcing
its valid navhard interval (**76 log clusters, EPDMS ×100 11.4816, CI [8.25, 14.50]**) to declare
UNAVAILABLE. ⛔ **I checked at source before loosening anything, and the premise is false — which
matters, because the fix it implies would have weakened a gate that was already correct.**

| probe | result |
|---|---|
| working-tree registry version | **2.10.3** (was 2.10.2 when E1 wrote) — its `rule` already admits `navsim_log_cluster_bootstrap` with `8 ≤ n_clusters ≤ the split's log count` |
| `git show HEAD:products/P7-TanitEval/CRITERIA_REGISTRY.json` | **2.9.0** — and the quoted sentence is **exactly** its `admissible_until_settled` |
| the four gates run on **E1's real navhard CV artifact** | the estimator gate reads its OWN declarations, and those are malformed (below) |

⭐ **THE MECHANISM, and it generalises past NavSim:** the settled gate has lived in the working
tree since 2.10.0 — **staged and never committed**, because the operating standard tells agents to
stage and never commit. E1 read the registry the way a fresh process does and got **2.9.0**. ⇒
**A staged gate is not in force for anyone; an artifact is built against the registry a process can
READ.** "The working tree says otherwise" is not a defence a second agent can use. This is the
first measured cost of the stage-never-commit rule, and it is an escalation, not a fix I can make
(§F).

**What the artifact actually declares, and why both shapes passed as "an honest refusal".**
`estimator.cluster_unit = {status: UNAVAILABLE, n: 76, reason: "log_name"}` — the **unit name in
the reason field** — and `estimator.interval = {status: UNAVAILABLE, reason: "…the v2.9.0 registry
gate admits only UNAVAILABLE here until W2 updates it", summary_interval: {…the real interval…}}`.
⛔ **A FALSE REFUSAL reads exactly like an honest one**, so the gate would have filed E1's
measurement as a WORK ITEM and the number would have stayed invisible. Both are now caught:

* **FALSE REFUSAL** — a declined interval that holds an **admissible** one (searched to depth 3)
  **FAILS**, naming the key path to promote. ⭐ The nested block is validated by the **same**
  `_navsim_interval_ok` the declared interval goes through, so the gate cannot admit a block in one
  place and reject the identical block in the other.
* **A one-token refusal reason** on `cluster_unit` **FAILS** — a single token cannot explain
  anything, and the refusal shape was hiding a unit that is settled and correct.
* ⭐ **The discriminating control is the half that makes the detector meaningful:** a refusal that
  banks a **rejected** candidate (an episode-cluster CI) stays **REFUSED** — the detector fires on
  admissibility, not on the word "interval".

**Registry → 2.10.3** (`code/registry_update_v2_10_3.py`): the gate now **states its
post-settlement form** (PASS / REFUSED / the nine FAIL branches) and **supersedes the 2.9.0 text
by name**, so a reader cannot re-derive it. ⛔ **No rule was loosened.** The PASS branch was
already machinery: `test_a_fully_declared_navsim_artifact_passes_every_gate` and eleven estimator
mutation arms have pinned it since 2.10.0.

**Tests (+7, `tools/tests/test_criteria_check.py` → 185 passed):** a valid navhard interval is NOT
refused (E1's digits) · the false-refusal arm · the inadmissible-nested control · the one-token
reason arm + its real-reason control · warmup still refuses at 7 clusters and a 7-cluster interval
still FAILs · the registry-literal arm · and **the decisive one — E1's REAL banked artifact**:
as published it is **diagnosed**, and with its own nested `summary_interval.detail` promoted and
`cluster_unit` declared as the string, the gate **PASSES on 76 log_names** with `lo`/`hi` reading
back **8.25 / 14.50** ×100. That is the proof nothing in the gate blocked a correct artifact.

### A.10 E1's adapter gaps G1–G6 — 9/9 already closed (MEASURED, not asserted)

E1's `build_artifacts.py` docstring lists six gaps it fills locally. That docstring predates this
stream's adapter work, so I probed each one positively rather than inheriting the list
(`code/e1_adapter_gaps_G1_G6.py` → `raw/e1_adapter_gaps_G1_G6.json`, **9/9 CLOSED**):

| gap | closed by | evidence |
|---|---|---|
| **G1** long stage-suffixed devkit columns | `submetrics_from_row(..., stage="one"\|"two")` reads `<long>_stage_one\|_stage_two` | run against E1's **real** `CV.devkit.csv`: **0 missing terms**, TLC = 1.0 |
| **G2** `estimator.cluster_unit` | `build_artifact` always emits it | `"log_name"` |
| **G3** `protocol.ego_status_enforcement` | `ego_status_enforcement=` | mechanism + declared fields + evidence |
| **G4** `protocol.sensor_set` / `setting` | `sensor_set=` / `setting=` | both round-trip |
| **G5** `protocol.navsim_protocol` / `devkit_sha` | `navsim_protocol=` / `devkit_sha=` | `EPDMS_v2_navhard_two_stage` @ `0a380a90…` |
| **G6** `protocol.corpus` / `controls` | emitted from `SPLIT_CENSUS` / `controls=` | corpus names the devkit scene filter + token counts |

⇒ **E1 can delete its six local fillers and call the adapter.** (E1's own item 2, the missing
**TLC** multiplier, closed in registry **2.10.0** — not 2.10.1 as relayed — and is pinned twice:
a literal `["NC","DAC","DDC","TLC"]` and a cross-block consistency test against `EPDMS_submetrics`,
which is the check that would have caught the omission on the day.)

---

## B. The route leak — verdict: **PARTIAL — a ROUTE-LEVEL ORACLE**

**Evidence class: PUBLISHED-CODE (mechanism) + MEASURED (four probes, four different mechanisms).**

### B.1 At source (PUBLISHED-CODE, pinned)

| what | where |
|---|---|
| the route read straight from the nuPlan DB | OpenScene@`7286074` `create_openscene_metadata.py:95-99` — `roadblock_ids = lidar_pc.scene.roadblock_ids` |
| the command computed from (pose NOW, map, route) | `…:122-127` + `helpers/driving_command.py:40-100` (route correction → Dijkstra centreline → 20 m ahead → \|y\| ≥ 2 m). **No future input exists in the signature.** |
| what the nuPlan route IS | nuplan-devkit@`ce3c323` `docs/nuplan_schema.md:181-190` — a scene *"stores a goal … that is a future ego pose from beyond that scene"* and *"a sequence of road blocks to navigate towards the goal"* |
| nuPlan's own derivation | `nuplan_scenario.py:171-177` — `_route_roadblock_ids`: *"Route roadblock ids extracted from expert trajectory"*; `:238-242` reads `scene.roadblock_ids` (`nuplan_scenario_queries.py:307-325`) |
| the devkit never computes it | navsim@`0a380a9`: `driving_command` appears only at `dataclasses.py:144, 204, 429, 468, 588` — all READS (git grep over the pinned tree; control: `roadblock_ids` 5 hits in the same file) |

Banked sources + sha256: `raw/source_probes/` (`SHA256SUMS.txt`).

### B.2 Empirically (MEASURED — `code/route_leak_probe.py` → `raw/route_leak_probe.json`)

1,902 stage-1 frames (all 450 navhard + a 1,452 navtest sample), 1,921 nuPlan scenes, 778 synthetic
scenes, 4 cities, local nuPlan maps.

| probe | result |
|---|---|
| (a) **reproduction** — OpenScene's own function on (stored pose, stored route, map) | **1,902/1,902**; turning frames **739/739**; by split navhard 450/450, navtest 1,452/1,452 |
| (b) **route provenance** — is the route the ego's own path? | starts BEHIND the ego in **83.5 %** of scenes (median index 1); its forward blocks are visited by the logged ego, or lie beyond the recorded segment, in **79.8 %** — against a **34.6 %** NULL CONTROL (a no-future straight-ahead route); future visits in route order **96.4 %** |
| (d) **counterfactual** on turning commands | route rebuilt from the ego's FUTURE path → **710/739 = 96.1 %**; NO-FUTURE straight route → **491/739 = 66.4 %** (overall 97.5 % vs 84.6 %) |
| (c) **agreement with the raw future path** (n = 65,783) | best rule (lateral offset after 20 m of driven arc, \|y\| ≥ 2–3 m) **94.2 – 94.7 %**; 4 s-horizon rules ≤ 86 % (turning recall 0.35–0.49) |
| stage 2 | the synthetic command AND route are **identical to the original frame's** at the same timestamp: **5,462/5,462** navhard, **204/204** warmup; recomputing at the perturbed pose agrees on only **83.9 %** of 778 |

### B.3 Reading

* The command is **not** a function of the expert's future TRAJECTORY: no threshold rule on the
  future path reaches the 99 % bar, and the 4 s-horizon rules are far below it — it carries no
  speed, stop or lane-change information.
* The **route** is the expert's own driven path at roadblock granularity — the 34.6 % null control
  and the "starts behind the ego" signature are what separate that from "any route would look like
  this". ⇒ at inference the command reveals the expert's future **route choice** at the next
  junction.
* ⇒ **the same class as our own `refcv4b` nav token** (*"oracle, provenance ego-future"*): a
  command-conditioned NavSim number is **driving with an ORACLE ROUTE**. Admissible under the
  goal/situation-disjoint rule (it is not a situation-classifier output), **optimistic by
  construction**, comparable inside NavSim (every agent gets it), and **never** evidence of route or
  strategic skill. The command may never be a route-head LABEL (the flagship-v1 echo, 369/369).
* Consequence for stage 2: a synthetic start displaced from the expert carries the command computed
  for the EXPERT's pose — one more reason a stage-2-only statistic is not an EPDMS.
* **Machine-readable**: `navsim.py::ROUTE_LEAK_VERDICT` (returned by `route_leak_check()`), pinned
  equal to the registry's `benchmarks.navsim.ROUTE_LEAK_VERDICT`.

---

## C. Suites run (counts MEASURED; every skip REASON read, none counted blind)

### C.1 The files this work package owns

| suite | result | artifact |
|---|---|---|
| `tools/tests/test_criteria_check.py` | **178 passed · 0 failed · 0 skipped** (51 new cases in 21 functions by W2, 26 of them gate mutation arms in one parametrised test; + W6's 5 nuScenes guard tests and W2's 2 tag-detected arms) | `raw/pytest_tools_test_criteria_check.txt` |
| `taniteval/tests/test_navsim_adapter.py` + `test_navsim_ci.py` | **79 passed · 0 failed · 0 skipped** (then +1 test: 21 in `test_navsim_ci.py` alone) | `raw/pytest_taniteval_navsim_adapter_and_ci.txt`, `…_ci_final.txt` |
| E1's OWN artifact builder, re-run unchanged against the updated adapter (scratch copy) | exit 0, both arms rebuilt, **0 violations** each, its own 5 gate mutations still all RED | `raw/e1_builder_compat_stdout.txt` |

**Final combined run of the three owned files (`raw/pytest_final_all_owned_files.txt`):
`tools/tests/test_criteria_check.py` + `taniteval/tests/test_navsim_ci.py` +
`taniteval/tests/test_navsim_adapter.py` → **258 passed · 0 failed · 0 skipped**.**

### C.2 `pytest -q tools/tests` — per file (the directory run stalled on one file, so it was run file by file; `raw/pytest_tools_per_file.txt`)

**32 + 29 + 18 + 43 + 178 + 37 + 7 + 19 + 25 + 24 + 16 + 29 + 4 + 21 + 13 = 495 passed**, plus:

| file | result | mine? |
|---|---|---|
| `test_library.py` | **1 failed** / 21 passed — `library.json` entry `1407.7644` has an **empty `title`** (staged today by the Library stream, commit `34c808f` lineage) | ⛔ not W2's — Library owner's one-line fix |
| `test_library_tracking.py` | **TIMEOUT at 300 s** — it shells out `git cat-file -e HEAD:<path>` per banked paper (~437) on this mount | ⛔ not W2's — pre-existing cost |
| `test_release_gate.py` | **17 failed** / 21 passed | ⛔ **PRE-EXISTING, verified**: the identical counts (19 failed / 34 passed together with the file below) come out of a HEAD-only checkout (`git archive HEAD tools products/P7-TanitEval` into a scratch tree). Cause: the `strat.nav_compliance*` criteria (registry ≥ 2.6) are ABSENT in the gate's own synthetic fixtures, so RG-02 fires on every arm |
| `test_registry_paths_allow.py` | **2 failed** / 13 passed — dead/malformed citations in the real registry | ⛔ pre-existing, identical at HEAD |

### C.3 `pytest -q taniteval/tests` — whole directory

**1,558 passed · 17 failed · 13 errors · 24 skipped · 1 xfailed** (`raw/pytest_taniteval_full.txt`).
⛔ **None of the failures or errors is in a file this package touches**: every one is in a sibling
stream's in-flight code — `tests/test_leaderboard.py` (W4: `AttributeError: 'function' object has no
attribute 'render_section' / 'load_model'` at fixture setup), W1's GPU-gap watchdog test, and W5's
report/viz tests. My two files pass inside the same run.
**All 24 skips are NO_TREE-class** and were read: 13 "committed artifact/tensors not present"
(old `TanitAD Research Hub/…` paths), 7 "module not present" (`pc6_linear_readout.py`,
`ll1_ladder.py`, `v4_corridor_cl`), 1 "OpenCV not installed in this venv"
(`test_nuscenes_planning.py`, W6), 1 "val40 manifest not present", 1 "branch build needs a full
model", 1 "30 k corridor artifact absent". None is a disabled guard of W2's.

---

## D. Proposed register rows (verbatim — `GOALS_AND_CLAIMS.md` is the orchestrator's to edit)

| id | claim | status | evidence |
|---|---|---|---|
| **D-NAVSIM-ESTIMATOR-1** | ⭐ **THE NAVSIM INTERVAL IS A LOG-CLUSTER BOOTSTRAP OF THE DEVKIT'S OWN AGGREGATE — clusters = the OpenScene `log_name`, B = 2000, n_clusters ≥ 8 (RG-14).** MEASURED: scenes overlap heavily WITHIN a log (navtest consecutive scenes share a median 13 of 14 frames; each key frame sits in 5.55 scenes; navhard's orig/prev share 11 of 12 and are coupled by the EC term) and share NOTHING across logs (0 shared tokens, 0 time-overlapping segment pairs, min gap 11.5 s); every stage-2 scene lives in its key's log (5,462/5,462 navhard, 204/204 warmup), so a unit's contribution depends on ONE log and resampling logs reproduces the official aggregation exactly. CONTROL: the OpenScene standard `test` split reads frame-reuse 1.0000. Splits: navhard 76 logs / 34 drives ✅, navtest 136 / 44 ✅, ⛔ warmup 7 (never), ⛔ private_test_hard (log identities not released). Validated against the devkit's OWN aggregation on 4 fixtures incl. physically duplicated logs (12/12) and on E1's real CV warmup run (0.1853562745165113, \|Δ\| 0.0) | **SUPPORTED — pre-registered + implemented + gated** | `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/{SPEC.md,RESULT.md,raw/split_census.json,raw/devkit_aggregation_reference.json}`; `taniteval/adapters/navsim_ci.py`; `taniteval/tests/test_navsim_ci.py`; registry 2.10.0 `GATE_estimator_cluster_unit` |
| **D-NAVSIM-ROUTE-LEAK** | ⭐ **NAVSIM'S `driving_command` IS A ROUTE-LEVEL ORACLE — PARTIAL LEAK.** The command is a function of (the ego pose NOW, nuPlan's `route_roadblock_ids`, the map) and of nothing else (PUBLISHED-CODE OpenScene@7286074 `create_openscene_metadata.py:122-127` + `helpers/driving_command.py:40-100`; the devkit only READS it, `dataclasses.py:144/204/429/468/588`), reproduced on 1,902/1,902 local frames (turning 739/739). But the ROUTE is the expert's own driven path at roadblock granularity: nuPlan's schema defines the scene goal as *"a future ego pose from beyond that scene"* (`docs/nuplan_schema.md:181-190`) and its own `_route_roadblock_ids` is *"extracted from expert trajectory"* (`nuplan_scenario.py:171-177`); MEASURED the route starts BEHIND the ego in 83.5 % of 1,921 scenes and its forward blocks are the driven ones in 79.8 % against a 34.6 % no-future NULL CONTROL; a route rebuilt from the ego's FUTURE reproduces 96.1 % of turning commands vs 66.4 % from a no-future route. ⛔ It is NOT a trajectory leak: the best threshold rule on the future path reaches 94.2–94.7 % (n 65,783), below the 99 % bar, and 4 s-horizon rules ≤ 86 % — no speed, stop or lane-change information. ⇒ a command-conditioned NavSim number is DRIVING WITH AN ORACLE ROUTE (the refcv4b *"oracle, provenance ego-future"* class): comparable inside NavSim, never evidence of route/strategic skill, never a route-head label; pair it with a command-withheld arm. Stage 2 copies the command AND route from the expert's own frame (5,462/5,462 navhard, 204/204 warmup) rather than recomputing at the perturbed start (83.9 % agreement) | **SUPPORTED (PUBLISHED-CODE + MEASURED)** | this RESULT §B; `raw/route_leak_probe.json`; `raw/stage2_command_provenance.json`; `raw/source_probes/`; registry `ROUTE_LEAK_VERDICT`; `NAVSIM_PROTOCOL.md` §9 items 4 & 15 |
| **D-CRITERIA-NAVSIM-GATES** | ⛔ **THE FOUR BLOCKING NAVSIM GATES WERE DECORATIVE: `tools/criteria_check.py` EVALUATED NO PART OF `benchmarks.navsim`** — measured 2026-09-19 by E1 and independently by E2, both of whom had to hand-write their own evaluators, so every NavSim artifact passed. Registry 2.10.0 + the checker now evaluate the five criteria and all four gates (estimator unit · ego-status enforcement · modality label · cross-protocol, with per-protocol devkit SHA pins), refuse any protocol tag outside the registered union of the NavSim / nuScenes / internal blocks, and ship **26 mutation arms that must go RED**. ⛔ ALSO FIXED: `variants.EPDMS_v2.multipliers` read `[NC, DAC, DDC]` — TLC missing — while the same registry's `EPDMS_submetrics` multiplied by it (devkit `pdm_enums.py:165-171` has four); the previous test PINNED the defect, and a cross-block consistency test now prevents it | **SUPPORTED — landed in registry 2.10.0** | `tools/criteria_check.py`; `tools/tests/test_criteria_check.py`; `products/P7-TanitEval/CRITERIA_REGISTRY.json` 2.10.0 changelog |
| **D-FF-LATERAL-NULL** | ⛔ **AN UNDEFINED METRIC THAT RETURNS `None` IS READ AS A SILENT OMISSION.** MEASURED 2026-09-19 (W1, NavSim STOP floor): `four_families.lateral` returned `None` for heading / yaw-rate / curvature when every step was below `min_ds_m` — a stationary plan has no path tangent — and `tools/criteria_check.py` read the nulls exactly as MISSING keys, so three BINDING LATERAL criteria were reported as violations of an instrument that had honestly declined. FIXED AT SOURCE 2026-09-20 (W2): the four terms emit `{status: UNAVAILABLE, reason, n, n_steps_total, min_ds_m}` — never null, never `0.0` (which would read as perfect lateral agreement from a car that never moved). ⛔ **The same defect had a WORSE, REACHABLE half:** `yaw_rate_mae_degps` is masked by `both_pair` but was emitted under the HEADING count, so an arm with valid single steps and no valid PAIR emitted **NaN** — a `float`, which passes every `isinstance(v, (int, float))` guard downstream while strict JSON refuses it (MEASURED 2026-09-20: n_head 24, n_curv 0). It refuses now too. — with the DEFINED branch byte-identical throughout so no banked number moves; `criteria_check.classify` now diagnoses a present-but-null key separately from a missing one; W1's artifact seam stays as the regression guard | **SUPPORTED — fixed at source, gated, 13 tests incl. a null-mutation arm and the reachable-NaN arm** | `taniteval/taniteval/four_families.py::lateral`; `tools/criteria_check.py::classify`; `taniteval/tests/test_four_families_lateral_undefined.py`; `taniteval/tools/refav1_openloop_report.py::_f` |
| **D-LAT-CROSS-IS-AN-OFFSET** | ⚠️ **`four_families.lateral.cross_mae_m` IS A LATERAL OFFSET AT MATCHED TIME INDEX, NOT A CROSS-TRACK ERROR — so a LATERAL row quoted alone cannot distinguish a moving arm from a parked one.** VERIFIED AT SOURCE: `_seq_geometry` returns `"cross": p[..., 1][:, 1:]` (the ego-frame **y column**, origin prepended) and `lateral()` reports `mean \|y_pred - y_gt\|` over matched step indices — no projection onto the GT path, no arc-length matching, no rotation into the GT tangent frame. MEASURED 2026-09-19 (W1, NavSim warmup): a constant-velocity arm and an all-zero STOP plan both read **1.0658 m**; REPRODUCED analytically 2026-09-20 (W2), where the stationary arm's `cross_mae_m` equals the GT's OWN mean \|y\| to 5e-5 — a property of the human's path, not of the arm. ⛔ NOT a NavSim artefact: the same reducer produces the LATERAL rows in the refcv3 / refcv4b / refcv5-v2 T1 panels and in `MODEL_REGISTRY.md`. **No banked number is rewritten**; every existing LATERAL number stands with the QUALIFIER: *"`cross_mae_m` is a lateral offset at matched time index, informative only while the along-track error is small; read it with the LONGITUDINAL family, and use `headline.pathgeom_crosstrack_m` (`lateral.py::frenet_dense`) when a distance-to-path is meant."* The block now emits that sentence plus `_along_mae_m_for_context` | **SUPPORTED (source-verified + reproduced)** | `taniteval/taniteval/four_families.py` (`_seq_geometry` `"cross"`, `lateral`); `taniteval/tests/test_four_families_lateral_undefined.py`; W1's warmup run |
| **D-REGISTRY-STAGED-NOT-IN-FORCE** | ⛔ **A STAGED GATE IS NOT IN FORCE FOR ANYONE — AN ARTIFACT IS BUILT AGAINST THE REGISTRY A PROCESS CAN READ.** MEASURED 2026-09-20: `CRITERIA_REGISTRY.json` carried the SETTLED estimator gate in the working tree from registry 2.10.0, **staged and never committed** (the operating standard tells agents to stage and never commit). E1 read `HEAD` — version **2.9.0**, whose `admissible_until_settled` admits ONLY `{status: UNAVAILABLE}` — and therefore down-declared a VALID navhard interval (76 log clusters, EPDMS ×100 11.4816, CI [8.25, 14.50]) into `estimator.interval.summary_interval`, plus a `cluster_unit` refusal whose `reason` was the single token `log_name`. ⛔ The gate was never wrong; it was **unreadable**. ⭐ Both shapes read exactly like an honest refusal, so the measurement would have been filed as a WORK ITEM and stayed invisible — the checker now FAILS a false refusal (validating the nested block with the SAME validator the declared one uses) and a one-token refusal reason, with a discriminating control that keeps a banked REJECTED candidate an honest refusal | **SUPPORTED — measured at source (HEAD vs working tree), gated, 7 new arms incl. one on the REAL banked artifact** | `git show HEAD:products/P7-TanitEval/CRITERIA_REGISTRY.json` = 2.9.0 vs working tree 2.10.3; `tools/criteria_check.py::_navsim_interval_ok` / `_admissible_interval_hiding_in`; `tools/tests/test_criteria_check.py`; `taniteval/results/bench/navsim_v2/navhard_two_stage/20260920T082848Z-navsim_v2-none-06e257/artifacts/CV.json` |
| **D-NAVSIM-SPLIT-NESTING** | **NavSim's public splits are NESTED, MEASURED**: warmup's 16 stage-1 tokens are ALL navhard tokens (16/16; 11 also navtest), and 367 of navhard's 450 are navtest tokens. A warmup or navhard number is therefore not an independent sample from navtest, and no row may be presented as if it were | **SUPPORTED (MEASURED)** | `raw/split_census.json`, `raw/stage2_command_provenance.json` |

---

## E. Deliverable manifest

**Repo (`D:\Projects\TanitAD`), staged by this stream:**

| path | what | single location? |
|---|---|---|
| `taniteval/adapters/navsim_ci.py` | **NEW** — the log-cluster bootstrap, the paired form, the admissibility predicate, `interval_from_run` | repo |
| `taniteval/adapters/navsim.py` | settled estimator + route verdict, `log_names=`, `interval=`/`devkit_sha=`/`sensor_set=`/`setting=`/`ego_status_enforcement=`/`nav_compliance=`/`controls=`, devkit long-column sub-metrics, `SPLIT_CENSUS`, protocol tags; every existing signature preserved | repo |
| `taniteval/adapters/__init__.py` | module list + the settled-unit note | repo |
| `taniteval/tests/test_navsim_ci.py` | **NEW** — 20 tests: analytic literals, the devkit reference, the real-run reproduction, 2 RED mutations | repo |
| `taniteval/tests/test_navsim_adapter.py` | updated for the settled defaults + 6 new tests (end-to-end gate pass, its scene-token mutation, log_names, verdict pins, vision-only fix) | repo |
| `products/P7-TanitEval/CRITERIA_REGISTRY.json` | **2.10.3** — §A.5 (settled gate, TLC, protocol tags) + §A.9 (the post-settlement form, stated in the registry) | repo |
| `tools/criteria_check.py` | evaluates `benchmarks.navsim` + the protocol-tag union (+311 lines) | repo |
| `tools/tests/test_criteria_check.py` | +34 tests incl. 26 gate mutation arms and the 7 post-settlement arms of §A.9 (**185 passed**) | repo |
| `taniteval/taniteval/four_families.py` | **SOURCE FIX (§A.8)** — an UNDEFINED lateral term REFUSES with reason + n instead of returning `None` — and the reachable `yaw_rate_mae_degps` **NaN** branch refuses too (§A.8(1b)); every lateral block now states what `cross_*` is, with its along-track context and the projection-based alternative. **The defined branch is byte-identical** | repo |
| `taniteval/tests/test_four_families_lateral_undefined.py` | **NEW** — 13 tests: the refusal shape, the never-null/never-zero regression arm, the defined branch unchanged, the one-moving-step boundary, the `cross_mae_m` blindness reproduced from scratch, the GT-excursion identity, the qualifier on a moving arm, a checker arm per criterion (null mutation → ABSENT, missing-key control), and the renderer arm | repo |
| `taniteval/taniteval/bench/navsim/artifacts.py` | **W1's file, 4 lines** — `refuse_undefined_lateral` also reports terms already refused upstream, so their contract survives the source fix (COMMS row 11) | repo |
| `taniteval/tools/refav1_openloop_report.py` | **6 lines** — `_f` renders a refusal block as `REFUSED (n=…) — reason` instead of `str()`-ing a ~600-char dict into one markdown cell (§A.8(3)) | repo |
| `…/raw/pytest_four_families_lateral.txt` | the consumer sweep + checker suite after the source fix, with the file list it was derived from | repo |
| `products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md` | §8.5 and §9 items 4 + 15 marked SETTLED with their evidence | repo |
| `…/2026-09-19-navsim-estimator-and-route-leak/SPEC.md` + `raw/SPEC_PREREG_HASH.txt` | the pre-registration and its hash | repo |
| `…/PLAN.md`, `…/RESULT.md`, `…/COMMS.md` | this package | repo |
| `…/code/split_census.py`, `two_stage_synthetic_census.py`, `devkit_aggregation_crosscheck.py`, `route_leak_probe.py`, `registry_update_v2_10_0.py` | the probes and the registry migration | repo |
| `…/raw/split_census.json`, `two_stage_{warmup,navhard}_two_stage.json`, `key_gap_within_log.json`, `devkit_aggregation_reference*.json`, `route_leak_probe.json`, `stage2_command_provenance.json`, `cluster_maps/*.json`, `source_probes/*`, `pytest_*.txt` | the evidence | repo |

**Off-repo (named because it exists in ONE place):**
* `…/scratchpad/frames_cache.pkl` (26 MB — the compact per-frame table for the 147 test logs) and
  `synthetic_{warmup,navhard}.pkl`: **rebuildable** by `code/split_census.py` /
  `code/two_stage_synthetic_census.py` in ~50 min and ~40 s respectively. Not banked (size).
* The NavSim data and runtimes (`C:/Users/Admin/navsim-crun`, `D:/Archive/devbox-C/navsim`) are
  E1/W3's deliverables, untouched by this stream.

---

## F. Integration asks (escalated — not written into a README)

1. **E2 / W1 — rebuild the refcv4b artifacts with the updated adapter.** Their
   `route_leak_check` is the pre-settlement "UNVERIFIED" refusal on arms that CONSUME the command,
   which the settled criterion now FAILS; `build_artifact()` supplies the verdict by default.
2. **E1 / W1 — rebuild the reference artifacts** (their `vision_only` was `true` for a privileged
   log-replay arm; the adapter now requires cameras + nothing privileged). Verified: E1's own
   builder, re-run unchanged, yields **0 violations** on both arms with the new adapter.
3. **W1 — capture the PRE-CSV frame** (`pdm_score_df` with `weight` and `log_name`) in every NavSim
   run; without it no navhard interval can be computed. `navsim_ci.interval_from_run(protocol=…,
   rows=…, mapping=…, clusters_by_unit=…, official_value=…)` is the one call.
4. **W1 — the suite's protocol enum and the registry union are now pinned equal** by a test on my
   side too; if W1 adds a tag, add it to its own registry block (never to NavSim's closed set).
5. **W4 (leaderboard)** — every NavSim row must carry the protocol tag + devkit SHA, and the
   command-conditioned rows must be labelled **oracle route** (§B). warmup and private_test_hard
   rows can never carry a CI.
6. **W6 (nuScenes)** — the registry now holds the two nuScenes protocol tags in
   `benchmarks.nuscenes.tasks.planning_openloop.protocol_tags`; no `claim_bearing` guard patch was
   present in W6's package when this stream finished, so none was integrated.
7. **Release-gate owner — `tools/release_gate.py`'s RG-02 fires on EVERY arm today** (17 of its 38
   tests fail, identically at HEAD): the three `strat.nav_compliance*` criteria (registry ≥ 2.6)
   are silently ABSENT in its fixtures, so "four families present or refused" reads FAIL for any
   artifact of the pre-nav-compliance shape. ⛔ Not W2's file and NOT fixed here — but it means the
   release gate is currently reporting a FAIL that is about the fixture, not the model. The
   adapter-side half of that same gap IS fixed: `build_artifact` now DECLINES the three criteria
   with a reason, so NavSim artifacts read REFUSED (work item) instead of ABSENT.
8. **Library owner** — `library.json` entry `1407.7644` has an empty `title`, which fails
   `tools/tests/test_library.py::test_every_entry_has_the_fields_a_citation_needs`.
9. **Orchestrator / W1 / E2** — the corrected modality sentence (§A.6) must replace the old one
   wherever it was quoted: `…/2026-09-19-leaderboard-currency/raw/external_field_sources.json` and the
   W1/E2 briefs. The registry is now right; those copies are stale.
10. **Register owner** — the four rows in §D.
