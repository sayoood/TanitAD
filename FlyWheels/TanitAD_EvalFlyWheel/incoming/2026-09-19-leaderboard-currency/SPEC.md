# SPEC — E4: what "current" means for `Benchmarks & Eval/LEADERBOARD.md` (5 October package)

**Stream** E4, EvalFlyWheel session 2026-09-19. **Brief** `BRIEF.md` (this folder). **Page owner**
this stream, exclusively, for this session. CPU only, 0 GPU, no downloads except banking.

## 1. Definition of "current"

The page is **current** when, for every arm that has an admissible eval result, the page either
(a) carries that result as a row whose every cell traces to `Project Steering/MODEL_REGISTRY.md` or a
raw eval JSON, **or** (b) names the arm in §12 with the reason it has no row.

An eval result is **admissible on this page** only if it satisfies all of the following:

| # | condition | where the rule comes from |
|---|---|---|
| A1 | the number is read from the registry **or** a raw eval JSON, never a summary, changelog, commit message, RESULT.md prose, weekly report or `PROJECT_STATE.md` | page header; `CLAUDE.md` "Source of truth" |
| A2 | it carries its **tier** (T0 / T1 / T2) **and** its **loop** (OPEN / CLOSED, PI ruling 2026-09-02) | `CLAUDE.md` "EVERY NUMBER CARRIES ITS EVAL TIER"; page §0.8 |
| A3 | it carries its **corpus / grid** (n windows / n episodes) and never shares a comparison cell with another corpus | page §0.3 |
| A4 | the interval is the **episode-cluster bootstrap** (paired for two arms on the same windows); ⛔ never `overlapping_holdout_se` | page §0.4 |
| A5 | the four families are reported per family; a missing family is `NOT MEASURED — <reason>, n=<n>` | `CLAUDE.md` "EVERY EVAL REPORTS FOUR METRIC FAMILIES" |
| A6 | the floors it is read against are named (`ha` / `ha0` / `ha0_ext` at T1; `cl_ha0` / `cl_ha` / `cl_ha0_ext` closed-loop) where the artifact has them | page §0.6, §0.9 G-2 |
| A7 | an anchor-selecting model's absolute ADE is quoted **with its rig**, or the paired margin is quoted instead | registry `D-OS-ADE-CROSSRIG-RESOLVED` |

A conflict between the registry and the raw JSON is **reported, never silently fixed**: the raw JSON
wins on the page and the conflict is recorded in the audit and proposed as a registry change.

## 2. Audit scope

1. **Registry**: every section of `MODEL_REGISTRY.md` that carries an eval result (§1–§5, §12, §13).
2. **Raw results**: every file in `taniteval/results/` (138 entries), with emphasis on the ones added
   since 2026-09-03 (git `--diff-filter=A`).
3. **Eval packages** since 2026-09-03 under `TanitAD Research Lab/**/` and `FlyWheels/**/`
   (`git log --since=2026-09-03 --name-only` located 284 dated package directories).
4. **External field**: every external number the new section prints is re-read **from the banked
   PDF** (library key + table + page), sha256-verified against `library.json`.

## 3. External-field rules (new section)

| rule | consequence |
|---|---|
| one sub-table **per protocol** (`navsim.cross_protocol`, closed set in `CRITERIA_REGISTRY.json` v2.9.0) | navhard two-stage, navtest PDMS v1, navtest single-stage EPDMS, nuScenes OL, Bench2Drive never share a column |
| the NavSim **official column** is `EPDMS_v2_navhard_two_stage` with the harness stated | the papers state no devkit SHA; our pin is `autonomousvision/navsim@0a380a9` (2025-10-27), the pre/post human-filter-fix status of each external row is stated from the source table |
| every external row states modality + ego-status use **as the authors claim it** | the leaderboard server records no modality |
| a number that cannot be re-read from a banked primary stays **INHERITED** and out of every comparison cell | page §9 inherited rows are re-classified, not deleted |
| TanitAD's row in each table is `NOT MEASURED — pending <artifact>` | warmup_two_stage numbers (E1/E2) get their **own** row, never the navhard column |

## 3b. W4 — what "regenerates itself" means (added when the PI expanded the scope)

| # | contract | enforced by |
|---|---|---|
| R1 | `python -m taniteval.leaderboard build` (from `<repo>/taniteval`) writes ONE marker-delimited section of the page and `Benchmarks & Eval/leaderboard.html`; every byte outside `<!-- BENCH:BEGIN -->…<!-- BENCH:END -->` is copied verbatim | `build.splice`; `test_hand_written_bytes_outside_the_markers_are_untouched` |
| R2 | **Delete the section, run build, get it back byte-identical** — whether the deletion kept the markers or removed them | `test_delete_the_section_and_build_returns_it_byte_identical[content|markers]` |
| R3 | the host file's newline convention is preserved (this checkout is CRLF) | `test_crlf_is_preserved` |
| R4 | determinism: sorted inputs, fixed float formats, no wall-clock; the section ends with a digest of the canonical data model, so it changes only when an input value changes | `test_build_is_idempotent_and_reports_no_change` |
| R5 | no number is typed into the generator's config — values are JSON pointers into registry-anchored raw artifacts, W1 `summary.json` files, or `published_results.json` | `leaderboard_sources.json._rule`; `test_the_generator_reads_the_raw_json_and_does_not_cache` (mutation) |
| R6 | our benchmark rows come through **W1's contract** (`taniteval.bench.contract.validate_summary`); a pre-contract artifact is converted first and validated the same way, and says so in `provenance.not_produced_by_bench_cli` | `sources.w1_validate`; `test_fixture_is_valid_under_the_w1_contract_and_current` |
| R7 | one protocol per table; STOP and CV on every NavSim run; no interval on warmup; pre-#151 values never share a comparison table with post-fix ones; a refused headline is rendered **with its reason**, never omitted | `render.assert_one_protocol`, `sources.summary_rows`, and four mutation tests |
| R8 | every external number carries library key · table · PDF page, and is re-readable from the banked PDF | `code/verify_published_against_pdfs.py` (+ 2 deliberate-regression arms) |

## 4. Out of scope (deliberately)

* No re-measurement, no GPU, no new eval run. A missing number stays missing.
* No edit to `MODEL_REGISTRY.md`, `GOALS_AND_CLAIMS.md`, `CRITERIA_REGISTRY.json` (proposals go in
  `RESULT.md`).
* No rewrite of the 13 T1 "closed-loop" mislabel sites (§0.8) unless git shows no owner — and then
  only as a §12 work item, never a silent edit.
* No wholesale page rewrite: surgical Edit-tool changes; every pre-existing number is preserved
  unless a raw artifact contradicts it, and then the correction is visible.
