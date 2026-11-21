# P8 — TanitDataSetCreator — BACKLOG

`Companion to products/P8-datasetcreator/SPEC.md. Prioritized. Every item carries WHY,
the MEASURED evidence of the gap, effort, dependencies, and a definition of done that is
checkable by CONTENT — never by exit code, file count, or filename
(TANITAD_PROGRAMME.md §6.4).`
`Owner: TanitAD_DataFlyWheel. Last updated 2026-08-23.`

**Evidence-class key** — `MEASURED` (ours, artifact path) · `PUBLISHED` (cited) ·
`INHERITED` (not re-verified here) · `ESTIMATED` · `HYPOTHESIS`.
**Effort** — `S` ≤ half a day · `M` 1–2 days · `L` 3–5 days · `XL` > 1 week. All ESTIMATED.

⚠️ **Read `SPEC.md §2` before pulling any item.** Several capabilities that look missing
already EXIST as libraries (`stack/tanitad/lake/`), and several that look present are
**stamps rather than checks** (`SPEC.md §2.2`). Re-deriving either wastes a day.

---

## Priority tiers

| Tier | Meaning |
|---|---|
| **P0** | The guard. Nothing may be built until these land — a configurator without them is the C113 defect with a CLI |
| **P1** | The minimum useful product: a spec that resolves, proves, and emits |
| **P2** | The central measured claim (data efficiency) and the integrations that make P8 load-bearing |
| **P3** | Surface, ergonomics, and the gaps P8 inherited but did not create |

---

## P0 — the guard

### P8-01 · Implement the four-relation parity classifier + naming firewall
**Tier** P0 · **Effort** M · **Depends on** — (nothing; all machinery exists)

**Why.** P8 is by definition an episode-re-selection machine, and re-selection is the one
thing `CLAUDE.md §Invariants` forbids. Every other P8 item is unsafe to build first.

**Measured evidence of the gap.**
- The parity key **is** a selection digest — `stack/tanitad/data/epcache.py:62-67`,
  `cache_key = sha1(json({"ids": ordered_source_ids, "params": params}, sort_keys=True))[:12]`.
  So a relation is *computable*, not merely declarable. Nothing computes it today.
- Corpus identity is a **path substring** — `stack/tanitad/data/parity.py:238-241`,
  longest-key-first `if key in s`. P8 is the first tool that lets a user choose that path.
- The **writer-side** firewall does not exist. `parity.py:1789-1801` `assert_not_parity`
  guards *readers* only (side models that must not read the parity corpus).
- All four verifications already exist and must be *called*, not rewritten:
  `check_uids` (`parity.py:325`), `verify_v2_membership` (`:1385`),
  `register_v2_geometry_sibling` (`:1577`), and the subset pattern
  `make_parity_clip_digests.py:155-229` `build_deployment`.

**Definition of done.**
1. `parity.assert_output_not_parity(path, *, label, relation, proof)` — or a P8-local
   equivalent — refuses when `parity.corpus_key_of(out_dir)` resolves to a registered key
   and the relation is not a *proved* `identity`/`geometry-sibling`.
2. The refusal prints the path, the matched key, and both remedies — and **prints no clip
   ids** (pinned by a test in the shape of `stack/tests/test_build_parity_guard.py:434`).
3. `relation_proof` is written into `PROVENANCE.json` for all four relations, including
   the refusal case.
4. ⭐ **Round-trip against a committed artifact**: a `recorded-view` spec for the 40-episode
   deployed val reproduces `deployed_val40_clip_digests.json`'s `digest_of_digests`
   (`e15eeab3eb25…`). A synthetic fixture does not satisfy this.
5. `git ls-files --cached` shows the test file staged.

---

### P8-02 · Wire the ingest gate, early, and pin the ordering by AST
**Tier** P0 · **Effort** S · **Depends on** P8-01

**Why.** A gate that runs after the download is a gate that costs money to trip, and a new
corpus door with no gate is exactly how C112 happened.

**Measured evidence.**
- `parity.py:2289-2290`, verbatim: *"C112's own launch-path defect died **after** paying
  for a 536 MB download; a gate that runs late is a gate that costs money to trip."*
- The gate exists and is wired at **6 doors** — `epcache.py:114-117`,
  `rebuild_pai_rolling.py:130-131`, `v2_to_pilot.py:95-96`, `v2_compressed.py:416,418`,
  `aug120_pipeline.py:79-80`, `slice_v2_cache.py:121-122`. **P8 would be a 7th, ungated.**
- MEASURED contamination that makes this concrete: of 4,729 Alpamayo clips, **201** are
  inside `physicalai-train-e438721ae894` and **6 of the 40** deployed-val episodes are in
  the record set — `test_build_parity_guard.py:339`, `DATA_STRATEGY.md:160-162`. The
  buildable rate, which `DATA_STRATEGY.md:161` marks **QUOTE THIS**, is **201/257 = 78.2101 %**.

**Definition of done.**
1. `require_ingest_gate("P8.resolve")` + `guard_corpus_build(...)` called before the first
   member is resolved and before any output directory is created.
2. `rec` embedded verbatim in `PROVENANCE.json` (required by `parity.py:2298-2300`).
3. `test_p8_gate_refuses_before_any_write` asserts **both** the raise **and**
   `not os.path.exists(out)` — the shape of `test_build_parity_guard.py:479`.
4. `test_p8_gate_precedes_the_expensive_step` compares **AST line numbers**
   (`min(gate) < min(spend)`) — the shape of `test_build_parity_guard.py:529`.
5. A typo in `corpus_role` **refuses** rather than weakening the check (delegated to
   `parity.py:2302-2312`; pinned by a P8 test so the delegation cannot silently break).

---

### P8-03 · Make P8 appear in the DERIVED corpus-writer population
**Tier** P0 · **Effort** S · **Depends on** P8-02

**Why.** `test_build_parity_guard.py` derives its door population by AST walk instead of
listing it, because *"the builder is one place"* was false. If P8's builder is invisible to
that derivation, the suite goes green over an ungated door — a guard reporting on a
population that excludes the thing being guarded.

**Measured evidence.**
- `test_build_parity_guard.py:153` `derive_corpus_writers(stack_root)`; the artifact regex
  is `:108-114` (`\.v2ep\.pt`, `ep_\{…\}\.pt`, `ep_%0\d*d\.pt`, `/videos/`, `/ego/`,
  `clips\.json`); `WRITE_CALLS` at `:117-121`.
- `:261` `test_every_derived_corpus_writer_is_gated_or_classified` requires every derived
  writer to be `gated` or explicitly in `NOT_AN_INGEST_DOOR`.
- ⚠️ The deriver has a **known blind spot**: `v2_compressed.build` passes its artifact path
  through a helper parameter and is caught by the MENTION census only
  (`build_doors.json._known_limitation`, `test_build_parity_guard.py:310-327`). P8 must not
  land in that blind spot.
- ⚠️ `stack/tanitad/lake/view.py` is currently classified `CANNOT_GATE`
  (`test_build_parity_guard.py:233-243`) **and was never edited** — its reason lives only
  in the test's classification map. P8 builds on `view.py`.

**Definition of done.**
1. Running `derive_corpus_writers` lists P8's builder with `gated: true`.
2. ⛔ It is **derived**, not hand-added to `KNOWN_DOORS`. Verified by deleting the P8 entry
   from any hand-list and re-running — it must still appear.
3. If the deriver cannot see the write, a **behavioural** test compensates (the
   `v2_compressed` precedent at `:479`) and the limitation is recorded in
   `build_doors.json._known_limitation` rather than left implicit.
4. `view.py`'s `CANNOT_GATE` classification is re-examined and either confirmed **in
   `view.py` itself** or changed.

---

### P8-04 · Bank the 7-case neuter matrix, RED, with a byte-compare restore
**Tier** P0 · **Effort** M · **Depends on** P8-01, P8-02

**Why.** `TANITAD_PROGRAMME.md:161-162`: *"A guard must be shown ABLE TO FAIL (the
deliberate-regression arm) before its PASS means anything."* P8's guard is the product's
entire safety claim; an unfalsified guard is a green suite over an untested rule.

**Measured evidence.** The precedent is banked and all-RED:
`TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-18-build-parity-guard/raw/neuter_matrix.txt`
— 8 cases, e.g. `RED 12  core guard_corpus_build -> no-op (12 failed, 12 passed)`,
`RED 1  require_ingest_gate -> no-op (1 failed, 23 passed)`.
⚠️ Its header records the operational lesson: an earlier attempt *"was killed by a 2-minute
tool timeout MID-CASE and left `aug120_pipeline.py` neutered on disk — a `try/finally` is
no protection against SIGKILL. That is why the byte-compare exists."*

**Definition of done.**
1. All **7** cases from `SPEC.md §5.7` (N1–N7) banked RED, each with failed/passed counts.
2. An md5 byte-compare shows every touched file identical to its pre-run state.
3. The runner is committed alongside the raw output — a matrix without its runner cannot be
   re-derived.
4. ⛔ **The deliverable is the banked RED result, not the runner.** A matrix that was
   "designed" is not a matrix that was run.

---

## P1 — the minimum useful product

### P8-05 · `DatasetSpec` schema + `tanitds spec lint` + `parity-check`
**Tier** P1 · **Effort** M · **Depends on** P8-01

**Why.** There is no dataset-config format in this repo at all. Selections live in shell
command strings, which cannot be diffed, reviewed, digested, or handed to a UI.

**Measured evidence.**
- **Zero** `.yaml`/`.yml` files under `stack/`; the only TOML is `stack/pyproject.toml`.
  No `dataset_config`, `DatasetConfig`, or `corpus_config` symbol anywhere.
- Today's substitutes: `stack/scripts/flagship_phase0.run.env:31-32`
  `TRAIN_CMD='… --data cached --cache-dirs …'` and
  `stack/ops/runs.d/flagship-v5f-w120-30k.env:10`, the latter annotated at `:2` as *"copied
  VERBATIM from `/proc/19412/cmdline` of the live trainer"* — i.e. the config format is a
  screenshot of a process.
- ⚠️ **Editing a manifest under a live supervisor changes nothing** (`CLAUDE.md`
  traps: `supervise_run.sh` sources its manifest once at startup). A declarative spec that
  is *resolved and digested* rather than *replayed as a string* removes this whole failure
  mode.

**Definition of done.**
1. `SPEC.md §3.2` schema implemented as `tanitad.datasetspec/1` with a JSON-schema or
   dataclass validator.
2. `tanitds spec lint` is **pure** — no filesystem reads beyond the spec, no network — so
   it is CI-safe.
3. Every rule in `SPEC.md §3.3` enforced, each **delegated** to the module that owns it
   (`SOURCE_REGISTRY`, `verify_license_scope`, `guard_corpus_build`). A test asserts the
   delegation by monkeypatching the owner and checking P8 propagates the refusal.
4. `tanitds parity-check <spec>` is read-only, prints relation + proof verdict +
   `comparability_key` + the ingest-gate record, exits `0`/`3`.
5. `select.scenario` non-empty is lint-**refused** with a message naming `P8-07` (P5 is not
   ready — `SPEC.md §1.3`).

---

### P8-06 · The determinism + provenance suite (12 tests) and `tanitds verify`
**Tier** P1 · **Effort** M · **Depends on** P8-05

**Why.** `SPEC.md §4.1`'s contract is worth nothing as prose. And P8 must not inherit the
repo's established stamp-don't-check pattern.

**Measured evidence of the pattern P8 must break.**
- ⛔ **`PARITY_SKIP_HASH` is never compared to anything in `parity.py`.** Five occurrences,
  none a check: definition `:80`, record stamps `:364` and `:1755`, prints `:406` and
  `:1782`. A cache with zero skip markers still gets `"skip_hash": "f09e44db"` written into
  its provenance and printed in the VERIFIED banner. The only real check is pod-side at
  `stack/scripts/pod_ops/compute_skipset.py:86-91`, which no trainer invokes.
- ⛔ `stack/scripts/parity_skipset.sh:36` **prints** the hash and never compares it, under
  `set +e` (`:2`).
- ⛔ `tools/corpus_census.py:140` carries `parity="physicalai-train-e438721ae894 /
  skip-hash f09e44db / 2376 eps"` as a **hand-written string**, copied verbatim to JSON at
  `:633`, never verified. It also *reads* LFS sha256 at `:440-443` and then **discards it**
  (`count_hf_members` at `:450-467` uses only `meta["size"]`).
- The canonical skip-hash form is **comma-joined** (`compute_skipset.py:3-4`,
  `parity_skipset.sh:34-36`) — deliberately *different* from `uid_digest`'s newline-join
  (`parity.py:127-137`). Getting this wrong produces a plausible-looking wrong hash.

**Definition of done.**
1. `stack/tests/test_p8_determinism.py` with **T1–T12** from `SPEC.md §4.5`.
2. ⭐ **T3 and T6 are the point.** T3 (perturbed predicate ⇒ digest changes) is the CAN-FAIL
   arm; T6 (mutated skipset ⇒ emitted `skip_hash` changes) proves the hash is computed and
   not copied. Without both, the suite proves only that a function is constant.
3. T1 uses **two separate subprocesses**. Same-process reuse would pass on a cached result.
4. `tanitds verify <dir>` recomputes every digest from the artifacts and exits non-zero on
   any mismatch.
5. `pytest -q` green (`CLAUDE.md §Invariants`).

---

### P8-07 · Resolver over the lake catalog + the six predicate families
**Tier** P1 · **Effort** L · **Depends on** P8-05

**Why.** Every primitive exists as a library; none has a caller. This is the item that
turns a pile of tested functions into a product.

**Measured evidence.**
- `lake/catalog.py:86` `resolve_view(...)`, `:105` `resolve_members(...)`; predicates in
  `lake/filtering.py` (`detect_corrupt` `:108`, `blur_band` `:184`, `exposure_band` `:196`,
  `truncation_frac` `:214`, `egomotion_sane` `:275`, `assign_rig` `:249`,
  `assess_quality` `:348`); dedup `lake/dedup.py:216`; curation `lake/curation.py:222`.
- ⛔ **Repo-wide, `curation`, `verify_license_scope` and `enrich_lake` have ZERO script
  consumers.** Every lake CLI is ingest or proof (`lake_ingest.py`, `lake_byteproof.py`,
  `build_tanitdataset.py`, `ingest_nuscenes.py`).
- ⛔ `lake/view.py` cannot serialise a selection: `filter_expr` is a live
  `pyarrow.dataset.Expression`; `signature()` (`:58-64`) is used only as a hydrate-cache
  tag (`:168`); the only persisted file is `DONE` = `{"episodes","hydrated","reused"}`
  (`:143-144`).
- `lake/__init__.py:60-75` exports **11 names**; `curation`, `view`, `catalog`,
  `license_guard`, `proof`, `dedup` are not among them.

**Definition of done.**
1. `tanitds resolve <spec>` compiles families 2–6 into a `pyarrow` predicate + a post-filter
   pass, emits `SELECTION.json` with all four digests and `per_predicate_attrition`.
2. ⭐ **`--format selection-only` works end-to-end first**, before any builder integration —
   it makes P8 useful on day one against existing pipelines.
3. `per_predicate_attrition` sums correctly (T9) and is rendered in the data card.
   `DATA_STRATEGY.md:834-847`: a census *"is a claim about the FILTER until proven otherwise"*.
4. `select.situation.classes_any` refuses `roundabout` as **UNPOWERED** — 26 clusters,
   PI-deferred (`stack/tanitad/data/situations.py:18`).
5. `selection_conditioning` (`SPEC.md §3.5`) emitted, listing the privileged channels each
   applied family read. MEASURED basis: situation labels are a pure function of ego poses —
   `situations.py:34-39`, `:62-66`; `scripts/emit_situation_labels.py:53-58` reads only
   `d["poses"]` and passes `cross=None`.
6. `lake/__init__.py` `__all__` widened, or P8 imports by full path and says why.

---

## P2 — the measured claim and the integrations

### P8-08 · ⭐ PREREG-P8-CUR1 — curated vs random at equal size
**Tier** P2 · **Effort** L (+ GPU) · **Depends on** P8-07 · **Blocked on** PI `SPEC.md §10 Q5`

**Why.** The DataFlyWheel's stated goal is *"best results with least data effort"*
(`TANITAD_PROGRAMME.md:91-93`). **That goal currently rests on an unmeasured assumption.**

**Measured evidence of the gap.**
- ⛔ A repo-wide search for a curated-vs-random comparison (`curated.*random`,
  `random.*curated`, `vs.random`, `baseline.*random` across `stack/**/*.py` and
  `stack/**/*.md`) returns **zero matches**. No A/B harness, no ablation script, no test.
- `lake/curation.py` computes weights (`inverse_frequency_weights` `:55`, `weakness_boost`
  `:110`, `safety_event` `:150`) and stores them in a `curation_weight` column — and
  **nothing in this repo consumes them for a comparative claim**.
- Every assertion in `stack/tests/test_lake_curation.py` is about the weights being
  *computed correctly* (rare > majority, singleton clamps to exactly `10.0`, holdout
  determinism). **None** is about downstream effect.

**Definition of done.**
1. `products/P8-datasetcreator/PREREG-P8-CUR1.md` committed **before launch**, carrying
   `SPEC.md §8.2` verbatim: arms, ≥3 random seeds, the fixed model, the shared eval set,
   the paired episode-cluster bootstrap, all four families, and **all four outcomes** —
   HELPS / DOES-NOT-HELP / HARMS / CANNOT-RULE.
2. The **zero-GPU pre-flight** runs first: stratum entropy + weakness share + speed
   histogram must show the arms actually differ. ⛔ If they do not, the experiment **cannot
   rule** and must not be launched.
3. Controls C1 (constant-only), C2 (raw-input floor), C3 (deliberate anti-curation
   regression) all run. ⛔ If C3 does not go the wrong way, report **CANNOT RULE** — never
   a null.
4. `RESULT.md` published **whatever the outcome**, `GOALS_AND_CLAIMS.md` H-P8-1 updated in
   the same turn, `MODEL_REGISTRY.md` rows added.
5. ⚠️ **Compute**: local (Thor / RTX 4060) is the default and is unmetered; the HF Pro quota
   is a hard ceiling and no metered job starts without a verified remaining quota
   (`TANITAD_PROGRAMME.md:22-28`).

---

### P8-09 · ESCALATION — `comparability_key` refusal in P7 TanitEval
**Tier** P2 · **Effort** M · **Depends on** P8-01 · **Owner** EvalFlyWheel (P7), **not** P8

**Why.** `SPEC.md §5.4` is the structural bar that makes a NON-PARITY declaration mean
something. **P8 cannot enforce it** — the comparison happens in P7.

**Measured evidence.**
- The precedent exists but is count-level and hand-maintained: `parity_manifest.json`
  splits `known_deployments` (600, 40) from `deployments_seen_but_NOT_admissible` (a
  12-episode pod1 partial whose `why_not` reads *"a decision-grade eval against it silently
  reports a different benchmark. **It already blocked a decision-grade run once.**"*),
  enforced only by a count comparison at `parity.py:692-698`.
- The programme has measured this integration failure twice: an orthogonality instrument
  unmerged **10 days**, a LAL-v2 implementation **12 days** — both because the request
  lived in a README nobody re-read.

**Definition of done.**
1. ⛔ **Raised as an escalation to the Master Mind in a report, with an owner and a date —
   not written into a doc and left** (`AGENT_OPERATING_STANDARD.md` rule 3).
2. P7 refuses a paired comparison across differing `comparability_key`s, naming both keys
   and both dataset ids.
3. A test proves the refusal fires on a real pair.
4. ⚠️ Until this lands, `SPEC.md §11` records the bar as a **convention, not enforcement**.
   That sentence is removed only when the test is green.

---

### P8-10 · P4 seam — `--dataset-spec` on the trainers
**Tier** P2 · **Effort** M · **Depends on** P8-06 · **Coordinate with** TrainingFlyWheel

**Why.** An arm's corpus identity is currently inferred by a reader from a directory name.

**Measured evidence.**
- ⛔ **No `--corpus` flag on any trainer and no `--parity-key` flag anywhere in the repo.**
- `stack/scripts/train_flagship_v4.py:1800` — `--train-cache` / `--val-cache`, **no help
  text**; `:1808` `--v2-train-cache`; `:1835` `--require-parity`; `:1851`
  `--parity-off-reason`.
- `stack/tanitad/train/train_worldmodel.py:482` `--data
  {toy,comma2k19,physicalai,realmix,mix,cached}` is the **only symbolic corpus selector in
  the repo**.
- The model to copy is `stack/scripts/v2_to_pilot.py:161,169,173,175` — `--corpus`,
  `--corpus-role`, `--exclude-parity-overlap`, `--sanctioned-audit`.

**Definition of done.**
1. `--dataset-spec <file>` on at least one live trainer.
2. `spec_sha256`, `selection_sha256`, `comparability_key`, `parity.relation` land in the
   run's `config.json` **and** its `MODEL_REGISTRY.md` row.
3. ⛔ No `--force`. The escape hatch is a **reason**, following
   `train_flagship_v4.py:1847-1851`, and it stamps `decision_grade: false`.
4. ⚠️ **Verify the target machine runs today's code by a real `import`, not by `git log`** —
   MEASURED 2026-07-27, pod2 sat at `0f93b98` while a gate fix was at HEAD, and a launch
   from it would have restored the crash the fix removed.

---

### P8-11 · P5 seam — `scena.resolve_tags(tags) -> list[ClipRef]`
**Tier** P2 · **Effort** L · **Depends on** — · **Owner** shared P5/P8

**Why.** `TANITAD_PROGRAMME.md:43` describes P8 as sitting *"on top of TanitScena"*, and
`SKILLS_SPECS.md:45` has `/TanitAD_DesignDataSet` *"query TanitScena (P5) → select
sources"*. **The wire does not exist.**

**Measured evidence.**
- `stack/tanitad/scena/README.md:1-6` — TanitScena turns the Opponent-Analyzer's
  `SCENARIO_DATABASE.md` **SC-01..SC-14** catalogue into a searchable app. The corpus is
  **14 markdown documents**.
- `stack/tanitad/scena/vector.py:163` builds the index over `s["id"]` (the strings
  `SC-xx`); `:37-41` `doc_text` embeds only `title`/`description`/`correct_behavior`/`tags`.
- `stack/tanitad/scena/parse.py:108-130` — the `data_sources[].link` fields resolve to
  public landing pages and HF **search** URLs, explicitly *not* real repo ids.
- ⇒ **There is no path from TanitScena to an `ep_*.pt`.**

**Definition of done.**
1. `scena.resolve_tags(tags: list[str]) -> list[ClipRef]` where
   `ClipRef = {source, clip_id | episode_uid}`.
2. ⛔ `ClipRef` **carries `source`**, so a tag match in one corpus can never be silently
   applied to another. That is the C112 shape — an overlap assumed from provenance rather
   than computed from ids.
3. `select.scenario` lint-refusal in P8 lifted, with a test that a scenario-only spec
   resolves to a non-empty, digested selection.
4. ⚠️ Gated clip ids are never returned to a hosted context (`parity.py:1829-1834`).

---

## P3 — surface, ergonomics, and inherited gaps

### P8-12 · `tanitds card --html` (UI tier T0)
**Tier** P3 · **Effort** S · **Depends on** P8-06

**Why.** A dataset whose provenance is only readable as JSON does not get read.

**Measured evidence.** No Gradio or Streamlit anywhere in the repo (two probes: `grep -rl
gradio --include=*.py`, and a second over `*.md`/`*.txt` — both empty). ⇒ a
zero-dependency static page is the only thing this repo can serve today.

**Definition of done.** Self-contained HTML (inline CSS/SVG, no JS required, no CDN):
parity verdict banner, the four digests, licence/tier breakdown, attrition waterfall,
stratum histogram, `selection_conditioning` disclosure. ⛔ Shows **counts and digests only**
for any `gated-confidential` source.

---

### P8-13 · Spec editor UI (tier T1)
**Tier** P3 · **Effort** L · **Depends on** P8-07, P8-12

**Why.** `TANITAD_PROGRAMME.md:43` says *"clickable + CLI"*.

**Measured evidence.** The pattern to copy exists and is proven in-repo:
`stack/scripts/scena_app.py:116` `build_app(db_md, static=None, cache_dir=None,
prefer="auto") -> FastAPI` with routes `/`, `/api/meta`, `/api/scenarios`,
`/api/scenario/{sid}`, `/api/search`, `/api/reindex`, serving three static files
(`index.html`, `app.js`, `style.css`) — **no build step, no CDN**.

**Definition of done.**
1. Predicate controls with a **live resolved count + stratum distribution + attrition**.
2. A *Parity* panel calling `parity-check`.
3. **Download spec** emits a valid `.tanitds.json` that `spec lint` accepts.
4. ⛔ ⭐ **The UI has no build button.** `SPEC.md §7.1`: the UI writes configs, never
   datasets — that is what makes it impossible for the clickable path to bypass a guard.
5. Read-only over the catalog. No registry writes from the browser.

---

### P8-14 · Skill `/TanitAD_DesignDataSet`
**Tier** P3 · **Effort** S · **Depends on** P8-05, P8-06

**Why.** `TANITAD_PROGRAMME.md:194` lists it; `SKILLS_SPECS.md:42-46` specs it; skills are
the programme's **conservation mechanism** (`TANITAD_PROGRAMME.md:186-189`) — a validated
procedure that stays in a transcript is lost.

**Definition of done.** The skill runs: query P5 (when `P8-11` lands) → licence-class check
→ `tanitds spec new`/`lint` → `tanitds parity-check` → `tanitds build` →
`tanitds verify` → register. ⭐ Its parity/provenance step is `parity-check` + `verify`,
**not** a prose reminder — that substitution is the entire lesson of
`BUILD_PARITY_GUARD.md:10-14`.

---

### P8-15 · Mint the parity VAL uid digest
**Tier** P3 · **Effort** S (blocked on host access) · **Depends on** — · **Blocked on** PI `SPEC.md §10 Q4`

**Why.** ⚠️ **The clean val split has no content check at all.** A *substituted* val set of
the right size passes today.

**Measured evidence.**
- `parity_manifest.json` → `corpora["physicalai-val-0c5f7dac3b11"].episode_uid_sha256` is
  `null`; `uid_source: "count-only-unrecorded"`.
- `check_uids` then takes the COUNT-ONLY branch (`parity.py:384-391`) and `_diff_lines`
  returns *"(manifest carries no uid list for this split — count only)"* (`parity.py:457`).
- The recorder exists and refuses to overwrite a good digest without `--force`:
  `make_parity_manifest.py:195-212`, with the procedure spelled out at `:29-45`.
- ⚠️ `parity.py:1846-1848` names the sibling gap: no val-side **clip** digest set exists
  either, *"which is not minted (this host has never held the 600 val clip ids)"*.

**Definition of done.**
1. On a host that holds the 600 val clip ids, after `pod_ops/compute_skipset.py` prints
   `VERDICT MATCH`: `make_parity_manifest.py --record --split val --cache-dir <…>`.
2. The changed `stack/tanitad/data/parity_manifest.json` **staged into the repo** — verified
   with `git ls-files --cached`, not with a `git add` exit code
   (`CLAUDE.md`: `git add` can silently no-op).
3. `SPEC.md §5.6`'s COUNT-ONLY caveat removed only once the digest is committed.

---

### P8-16 · Wire `corpus_census.py` into CI and make its parity string a check
**Tier** P3 · **Effort** S · **Depends on** — · **Owner** shared DataFlyWheel/DevEnv

**Why.** A durability census that nobody runs is not a census, and a parity field that is
copied prose is the same defect class as the skip-hash stamp.

**Measured evidence.**
- `tools/ci_gate.py` has **no** reference to `corpus_census`; the owning doc flags it open:
  `…/2026-08-03-corpus-durability/CORPUS_DURABILITY.md:316` — *"🟠 Wire
  `tools/corpus_census.py` into the nightly job."*
- `corpus_census.py:140` `parity="physicalai-train-e438721ae894 / skip-hash f09e44db /
  2376 eps"` is hand-written, copied to JSON at `:633`, **never verified**.
- It *reads* LFS sha256 at `:440-443` and **discards it** — `count_hf_members` (`:450-467`)
  uses only `meta["size"]`. The digest needed to make the parity field a real check is
  already fetched and thrown away.

**Definition of done.**
1. Census runs nightly; exit `2` (`ZERO_COPIES`/`UNRESOLVED`) fails the job, exit `1`
   (`SINGLE_COPY`) warns (`corpus_census.py:672-678`).
2. The `parity` field is **verified against `parity_manifest.json`**, not copied — or, if
   it cannot be verified from the census's own data, it is renamed to
   `parity_declared` so it cannot be read as a check.
3. ⚠️ Read the instrument's inclusion rule before quoting any count it produces
   (`DATA_STRATEGY.md:846-847`).

---

### P8-17 · Resolve the licence conflict blocking `licence.release`
**Tier** P3 · **Effort** S (PI decision) · **Blocked on** PI `SPEC.md §10 Q1`

**Why.** P8's `licence.release` field cannot be specified correctly while two committed
documents disagree about whether PhysicalAI-derived data may be published at all.

**Measured evidence.** `DataEng/DATA_STRATEGY.md:783-788` records the disagreement itself:
`TANITDATASET_V1_STRATEGY.md` classes PhysicalAI/Alpamayo as *"commercial-OK for internal
AV dev but no-derivatives → firewalled, recipe-only"*, while `ALPAMAYO2_SUPER_ANALYSIS.md`
cites an **OpenMDW-1.1 derivative permission** — *"These are not obviously the same reading,
and the disagreement is recorded here rather than resolved."* Already escalated as a PI
decision (§12 row 1, `:878`). Related: `:780-782` — *"⛔ The A2 augmentation set is INSIDE
the firewall, not outside it."*

**Definition of done.** PI ruling recorded in `DECISIONS.md` with an ID; `DATA_STRATEGY.md
§9` updated; `SOURCE_REGISTRY` amended if the ruling changes a class; `SPEC.md §10 Q1`
struck. **Until then P8 defaults to the stricter reading** (`release: "private"`).

---

## Dependency graph

```
P8-01 (relations + naming firewall)
  ├─> P8-02 (ingest gate, early)  ──> P8-03 (derived door population)
  │                                └─> P8-04 (neuter matrix, RED)      [P0 complete]
  ├─> P8-05 (schema + lint + parity-check)
  │      ├─> P8-06 (determinism suite + verify) ──> P8-10 (P4 seam)
  │      │                                     └─> P8-12 (HTML card) ─> P8-13 (spec editor UI)
  │      ├─> P8-07 (resolver + predicates) ─────> P8-08 (PREREG-P8-CUR1)   [PI Q5]
  │      └─> P8-14 (/TanitAD_DesignDataSet)
  └─> P8-09 (P7 comparability refusal)          [ESCALATION — P7 owns]

independent: P8-11 (P5 seam)  ·  P8-15 (val digest) [PI Q4]
             P8-16 (census in CI)  ·  P8-17 (licence ruling) [PI Q1]
```

**Critical path to a usable product:** `P8-01 → P8-02 → P8-05 → P8-07` (+ `P8-06` for the
contract). ⭐ `--format selection-only` (in `P8-07`) makes P8 useful **before** any builder
or UI integration.

**Unblocked right now, zero GPU, no PI decision:** `P8-01`, `P8-02`, `P8-03`, `P8-04`,
`P8-05`, `P8-06`, `P8-07`, `P8-12`, `P8-16`. ⇒ **A PI decision blocks at most `P8-08`,
`P8-15` and `P8-17` — never the programme** (`CLAUDE.md §NEVER IDLE`).
