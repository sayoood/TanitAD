# CENSUS WIRING — `dump_exclusions.json` is now CONSUMED, not prose-guarded (2026-08-18)

**Task:** institutionalize C126's fix. The root cause of the "27 arms = 27 dumps over 25 distinct
arms" error class is that **censuses re-derive their arm list from `glob("windows_*.pt")`, a
surface no prose correction can reach** (RETRACTION_LOG.md C126; DUPLICATES.md). The
machine-readable exclusion list existed since the results-hygiene pass but *nothing consumed it*.
Now census code does, and a stale exclusion **fails loudly** instead of silently excluding the
wrong content.

**Evidence class throughout: MEASURED** (this clone, branch `agent/arch-inf-20260803`, base
`65211ecc`; every command output quoted below was produced on this dev box, 2026-08-18).

---

## 1. The shared loader — `taniteval/taniteval/dump_census.py` (NEW)

`list_dumps(results_dir, *, include_excluded=False) -> CensusResult` with:

| contract point | implementation |
|---|---|
| never silently drop | `.excluded` (dump name → reason) is part of the return value; `.summary()` / `__str__` renders `"N dumps = M distinct arms (K excluded: …)"` for reports |
| `.paths` | deduplicated census (or ALL files under `include_excluded=True` — for tools that legitimately operate per-dump; `counts["distinct_arms"]` still subtracts) |
| `.counts` | `{"dumps_found": N, "distinct_arms": M}` |
| missing `dump_exclusions.json` | loud, not fatal: everything returned, `exclusions_missing=True`, summary says `"DISTINCT-ARM COUNT NOT DEDUPLICATED — dump_exclusions.json missing"` (tests and scratch dirs must not be pushed back to the bare glob) |
| **stale exclusion** | `StaleExclusionError`: every entry's `sha256` (and `canonical_sha256`) is validated against the on-disk bytes; a re-banked/changed file **fails the census** — the recorded equality claim is about bytes that no longer exist, and silently excluding unverified content would be worse than double counting |
| canonical partner absent | the excluded dump is **kept** (dropping it would remove the MODEL from the census, not a duplicate) with a note in the summary |
| unparseable exclusions file | `ExclusionsError` (fatal) — a census cannot run against an unreadable guard |
| `check_explicit(paths)` | the duplicate-VALUE guard for tools whose dumps arrive as explicit CLI args (ff_rescore): classifies `pair_present` (both members passed — the C126 error) vs `excluded_passed` (excluded alone — legitimate, noted), and reports which exclusions files were consulted |
| join surface for derived artifacts | `.pairs_by_key` (excluded arm key → canonical arm key) lets `driving_<key>.json`-level censuses dedup too |

No torch, no GPU — names + bytes only, importable everywhere including pod-side scripts.

## 2. Surfaces found — the grep sweep (two+ patterns, per the absence rule)

Commands run at repo root (full outputs reproducible):

```
grep -rn 'windows_\*' --include='*.py' .                                   # pattern 1: literal glob star
grep -rniE 'glob.*windows|windows_.*\.pt' --include='*.py' .               # pattern 2: glob/pt reads
grep -rnE 'glob\(.*windows_|iglob\(.*windows_|listdir|fnmatch.*windows' taniteval tools stack/scripts stack/tanitad   # pattern 3: live dirs
grep -rnE 'windows_\*' --include='*.sh' --include='*.ps1' .                # pattern 4: shell — no hits
```

### 2a. LIVE surfaces — wired

| surface | was | now |
|---|---|---|
| `taniteval/taniteval/driving.py:873` `arms_with_windows()` — **the** tier-0 census; feeds `run_all()` | bare `Path(res_dir).glob("windows_*.pt")` | routes through `dump_census.list_dumps(res_dir).arm_keys`; `run_all()` prints `[driving-all] census: <summary>` before iterating |
| `taniteval/taniteval/driving.py:936` `_load_blocks()` — feeds `panel_rows()` (dashboard 04c) **and** `leaderboard_md()` (LEADERBOARD.md generator) | bare `glob("driving_*.json")` — **both C126 arms HAVE `driving_*.json` blocks, so the panel and the regenerated leaderboard double-counted one model per pair until today** | duplicate row dropped iff its canonical partner's block is present, never silently: prints `[driving-census] <summary>; dropped duplicate driving_*.json rows: [...]` |
| `taniteval/tools/ff_rescore.py` (`:574` name-dup check) | refused duplicate NAMES only — nothing checked VALUES | `check_explicit` runs on the passed `.pt` dumps **before the expensive part**: a known same-model pair is `⛔ REFUSED` (exit 2) unless `--include-excluded` (new flag — for when the point IS the pair, e.g. verifying equality); an excluded dump passed alone gets a printed `[census] NOTE`; no exclusions file beside the dumps gets a printed "check unavailable" line |
| `taniteval/recompute_ci.py` | fixed hand-curated `ARMS` list (no glob; contains no excluded key today) | prints `[census] <summary>` at start (its output gets quoted), and skips-with-print any `ARMS` key that is a recorded exclusion — the guard is **dormant by construction, not by luck**: a future edit adding a duplicate key gets caught |

### 2b. LIVE surface — declares it counts dumps (the `include_excluded` semantics)

| surface | disposition |
|---|---|
| `tools/corpus_census.py:264-274` (`evaldumps-windows-fan` artifact, `members=29`) | a **file-presence/banking probe across hosts** (`ls | wc -l` over ssh) — it counts DUMP FILES, which is correct for stranding detection and must NOT dedup. Its `desc` now says so explicitly and points at `dump_exclusions.json` / `taniteval.dump_census`. No behavior change; 43/43 tests green. |

### 2c. BANKED / ARCHIVED surfaces — found, deliberately NOT edited

Editing banked experiment code retroactively would falsify the provenance of the banked numbers
it produced. These are listed so the census is complete; the structural fix is that **future**
census code imports `taniteval.dump_census`:

- `stack/experiments/pod-rescue-20260802/eval/workspace/run_ctrv_readjudication.py:93` (pod filesystem snapshot)
- `…/Benchmarks & Eval/Implementation/incoming/2026-08-02-ctrv-floor/run_ctrv_readjudication.py:93`
- `…/incoming/2026-08-04-distance-keeping-arms/code/rank_all_arms.py:23`
- `…/incoming/2026-08-15-dir-yaw-gate-reread/tools/gate_reread.py:145`
- `…/Architecture & Inference/…/incoming/2026-08-17-diryaw-reread/code/gate_operating_point.py:178` (sibling-owned area — untouched per brief)
- `…/Benchmarks & Eval/Research/2026-08-03-sota-top3-executed/code/run_item2_progress_and_predictivity.py:132` (its RESULTS.md was already corrected by C126)
- `…/incoming/2026-07-25-jack-blast-radius/recompute_jack_fullset.py:123`
- `…/incoming/2026-08-18-val40-lead-join/code/score_val40_dumps.py:127` (the pass that re-discovered the pairs)
- `…/incoming/2026-07-26-program-harvest/artifacts/build_index.py:232` (a path string in an index, not a census)

Comment-only sites (counts of FILES, ruled "still true as written" by DUPLICATES.md):
`bench.py:197`, `runner.py:189`, `driving.py:564`, `gate_emitters.py:62`,
`test_run_gate_corridor.py:347`, `test_eid_normalisation.py:39`, `dump_lead_join.py:57`.

Targeted per-key loads (`windows_{key}.pt` f-strings) are not censuses and were left alone.
Shell scripts: no `windows_*` globbing found (pattern 4).

## 3. The real summary line (MEASURED, this clone)

```
27 dumps = 25 distinct arms (2 excluded: windows_overfit_refa-dynin-30k.pt -> windows_refa-dynin-30k.pt, windows_refc-v12-identity.pt -> windows_refc-xl-30k.pt)
```

Running it also **validated all four recorded sha256s** against the committed bytes (no
`StaleExclusionError`) — the exclusions file is fresh for this tree.

Real-bank behavior of each wired surface:

```
[driving-census] 27 dumps = 25 distinct arms (2 excluded: …); dropped duplicate driving_*.json rows: ['overfit_refa-dynin-30k', 'refc-v12-identity']
  → _load_blocks kept 25 blocks; panel_rows/leaderboard_md now count 25 distinct arms
arms_with_windows(results) → 25 keys

ff_rescore --dump refa=…windows_refa-dynin-30k.pt --dump overfit30k=…windows_overfit_refa-dynin-30k.pt
  → ⛔ REFUSED (exit 2): "the passed dumps include BOTH members of a known same-model pair …
     Drop one, or pass --include-excluded if the point IS the pair"

recompute_ci --results taniteval/results
  → first line: [census] 27 dumps = 25 distinct arms (2 excluded: …); exits 0, no ARMS key skipped
```

## 4. Tests — `taniteval/tests/test_dump_census.py` (NEW, 15 tests)

Exclusion honored AND reported; summary format; missing-json loud flag; stale excluded-sha fatal;
stale canonical-sha fatal; unparseable-json fatal; `include_excluded` returns files but still
reports; canonical-absent keeps the dump; `check_explicit` pair/alone/none classification;
`arms_with_windows` consumes exclusions; no-exclusions tmp-dir behavior unchanged (the old
contract verbatim); `_load_blocks` drops the duplicate row loudly / keeps it when the canonical
block is absent; **ff_rescore subprocess refusal** (fixture pair, exit 2, message pins
`--include-excluded`); and a **real-bank relationship pin**: `distinct = found − 2` with exactly
the two recorded names excluded — a relationship, not a pinned total (new dumps may land), and it
re-validates the four sha256s on every run, so **a re-banked dump fails the suite loudly instead
of silently re-entering censuses** (same mechanism as the pinned feature-count test,
2026-08-16).

Runs (dev box, venv python, PYTHONUTF8=1):

```
pytest taniteval/tests/test_dump_census.py -q                 → 15 passed
pytest taniteval/tests -q -k "census or rescore"              → 16 passed, 1169 deselected
pytest taniteval/tests/test_driving.py test_driving_gate_block.py test_runner_gate_print.py -q → 49 passed
pytest tools/tests/test_corpus_census.py -q                   → 43 passed
pytest taniteval/tests -q  (full suite)                       → 1185 passed, 1 warning in 129.57s
                                                                (warning pre-existing: ci.py:171 nanmean
                                                                 RuntimeWarning in test_clhorizon, untouched)
```

No dedicated ff_rescore test file existed before; the subprocess refusal test above is the first.

## 5. What this changes for published numbers

Nothing retroactive — C126 already applied every doc correction. Going forward: `panel_rows` and
`leaderboard_md` regenerate with 25 rows (previously 27 — the LEADERBOARD §1b "count of rows that
beat CV/CTRV" class of error can no longer be produced by regeneration), `driving.run_all`
backfills 25 arms, and any tool that tries to score a known same-model pair side by side must say
`--include-excluded` out loud.

## 6. Escalations (integration needs — named here, not buried)

1. **Future incoming/ census code**: the operating standard should carry one line — *"a census
   over `windows_*.pt` imports `taniteval.dump_census`; a bare glob is a defect"* — so agent
   briefs inherit it. `AGENT_OPERATING_STANDARD.md` / `EVAL_DOCTRINE.md` are orchestrator-owned;
   not edited by this pass.
2. **Pod-side censuses**: pods have stale checkouts and no git (CLAUDE.md); any pod-side census
   must FILE-SHIP `dump_census.py` + `dump_exclusions.json` alongside the tool (md5-verified),
   or the pod re-derives the duplicates the local code can no longer produce.
3. **`MODEL_REGISTRY.md`** is sibling-owned today and needs nothing from this pass (C126 already
   corrected it).
