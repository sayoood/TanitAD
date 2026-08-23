# REPO MIGRATION PLAN — Research Hub → Research Lab + products/

`Created 2026-08-22 from the full-repo audit (read-only agent, 6,215 tracked
files surveyed). This is the executable plan for TANITAD_PROGRAMME.md §2's
structure. Execute phases IN ORDER; each phase is one commit; verification
gates between phases are mandatory.`

## Ground truth from the audit

- **398 work packages** exist; naming is good (394/397 match `YYYY-MM-DD-slug`)
  but **0 of 398 conform** to the target internal schema (`SPEC.md` count: 0).
  Competing vocabularies: `scripts/`(27) vs `code/`(105); `artifacts/`(29) vs
  `results/`(6) vs `raw/`(131); `INTAKE.md`(48) vs `PRE_REGISTRATION.md`(31)
  vs `PREREG.md`(2); `MANIFEST.md`(24) vs `RESULTS.md`(3) vs `VERDICT.md`(2).
- **Blast radius of the rename**: literal `"TanitAD Research Hub"` in **517
  code+md files / 814 occurrences** (+11k occurrences inside raw JSON records,
  which stay as historical values).
- ⛔ **`tools/` CANNOT stay unchanged**: 12 files hardcode the Hub name
  (`registry_paths.py:80,236`, `repo_janitor.py:58`, `session_guard.py:62,278`,
  `kb_add.py:46`, `corpus_census.py` ×6, `safe_commit.py` ×2, 4 test files).
  These are the hygiene enforcers — edit them IN THE SAME COMMIT as the rename
  or every subsequent commit is blocked.
- ⛔ **`Project Steering/MODEL_REGISTRY.md` carries 24 hub paths** and is lint-
  guarded — rewrite in the rename commit.
- **Name collisions to resolve**: top-level `Benchmarks & Eval/` vs Hub area;
  `DataEng/` vs Hub `Data Engineering/`; Hub `Project Steering/` (3-file stub)
  vs the real one; Hub `Evaluation/` vs `Benchmarks & Eval/` (overlapping
  mandate, different schema).

## Safety actions ALREADY DONE (2026-08-22)

- ✅ 361 gitignored media files (119 mp4 + parquets + Videos/) backed up to
  `C:\Users\Admin\tanitad-media-backup\` — they existed only on the flapping G:.
- ✅ 127 `__pycache__/`, 20 `.pytest_cache/`, `colab/_smoke_work` deleted.
- ⏸ `_pod_backup/pod2-2026-08-03/ckpts/*.pt` (**22.6 GB**, untracked pod-rescue
  copies from 08-03) — **awaiting PI confirmation** before deletion. The 3
  tracked text files there (the irreplaceable diffs) are kept regardless.

## Field mapping (four Lab fields)

| current Hub area | → Lab field |
|---|---|
| `Data Engineering/` | Data Engineering (in place) |
| `Architecture & Inference/` | Architecture & Inference (in place) |
| `Production & Optimization/` + `Tools&DevEnv/` | **Deployment & Optimization** (merge) |
| `Benchmarks & Eval/` + `Opponent Analyzer/` + `Evaluation/` | **Opponent & Benchmarks** (merge) |
| `Library/` | stays at Lab root (kb_add.py:46 depends on `<root>/Library`) |
| `agents/` (rotation) | `archive/hub-agents-2026-08-03` (superseded) |

## Phases (each = one commit, verification between)

- **P0 remaining**: `git rm` the 3 duplicate PDFs in `Ressources/` (byte-identical
  to Library copies); PI call on the 22.6 GB.
- **P1 — THE RENAME** (atomic): `git mv "TanitAD Research Hub" "TanitAD Research
  Lab"` + literal rewrite in the 517 files + the 12 `tools/` constants + the 24
  MODEL_REGISTRY paths + README/CLAUDE.md. Gate: `tools/registry_paths.py`,
  `registry_lint.py`, all three test suites green, `rg -c "TanitAD Research
  Hub"` = 0 outside archive/ and raw records.
- **P2 — field merges** (mkdir 4 fields; mv P&O→Deploy&Opt, Tools&DevEnv→
  `_devenv`; B&E→Opponent&Benchmarks, Opponent Analyzer→`_opponent`,
  Evaluation→`_evaluation`). Fixes the 33 files / 43 refs that name merged areas.
- **P3 — schema flatten, SCOPED**: flatten `Implementation/incoming/` and
  normalise child names **only for WPs with post-2026-08-15 activity (~50)**;
  older WPs move by folder only. (Full retrofit = ~5,000 path changes + 145
  code-ref fixes for zero measurement value — audit recommendation adopted.)
  New WPs use the §3 schema from day one via `new_workpackage.py`.
- **P4 — orphans**: the wave1 ledger orphan → A&I; Hub `Project Steering/` stub
  → merge/archive; loose Hub-root docs → archive or dated WPs;
  `HYPOTHESIS_LEDGER.md` + `KNOWLEDGE_BASE.md` stay at Lab root explicitly.
- **P5 — 13 legacy undated Implementation dirs** (57 files) → `archive/legacy-implementation/<field>/`.
- **P6 — products/**: create `products/P1..P8-*/`; move the design docs listed
  in the audit (V3/V35/V4 designs → P1; DATA_STRATEGY etc → P2; IDM design →
  P3; SCENARIO_DATABASE.md → P5; THOR runbooks → P6; LEADERBOARD + top-level
  `Benchmarks & Eval/` → P7; TANITDATASET docs + builds → P8). Code bodies stay
  in `stack/`/`taniteval/` with pointer READMEs. ⚠️ Open decision: do the five
  IDM work packages follow P3 or stay Lab WPs?
- **P7 — stale root docs** → `archive/` (`PROJECT_STATE.md`, `DECISIONS.md`,
  `Ressources/` contents; the Deep Think analyses preserved).
- **P8 — verify** (the full gate list above) + a pinning test that the old name
  no longer appears in code.

## Preserve-list (from the audit's stranded-value section)

560 MB tracked eval media in `Evaluation/Videos/` (17 campaigns — the §7
showcase corpus) · the pod2 rescue diffs · top-level LEADERBOARD.md (P7) ·
Deep Think analyses · `HYPOTHESIS_LEDGER.md` / `KNOWLEDGE_BASE.md` (Lab root) ·
the O234 raw dir → its own WP · the flat-JSON gate results in Evaluation WPs
(wrap in `raw/` during P3) · 221 tracked logs (compress, never drop).

## Execution constraints

1. ⛔ **Do not run P1 while any agent is staging into old paths.** The label
   agent (own session) writes under `TanitAD Research Hub/…` — coordinate a
   window or land its work first.
2. The index must be EMPTY at P1 start (commit or stash in-flight work first).
3. Each phase's commit message names the phase; `archive/` is append-only.
4. G: flap risk: run P1 from a verified-fresh clone state, and re-verify staging
   (blob compare) after each phase — the index moves under concurrent sessions.
