# PLAN — E4 (leaderboard currency) → W4 (leaderboard generator), executed 2026-09-19/20, CPU only

Scope changed mid-session: the PI expanded E4 into **W4 — a leaderboard that regenerates itself**
(`…/2026-09-19-eval-suite-build/BUILD_PLAN.md`, row W4). The original brief stands inside it.

| # | step | method | status |
|---|---|---|---|
| 0 | **Stop-condition check** on `LEADERBOARD.md` | blob comparison HEAD / index / worktree, all three asserted 40 chars | ✅ at start: `97939e78…` everywhere → proceeded. ⛔ **Re-checked before writing: a sibling had appended and staged a correction (`bb1283d9…`) → STOPPED**, per the brief |
| 1 | **Currency audit** | registry result-bearing sections read in full; `taniteval/results/` census + `git --diff-filter=A` since 09-03; 284 dated packages located by `git log`; A&I 09-07…19 classified by one read-only sub-agent (INHERITED, banked verbatim); every number that reaches a row re-read from raw JSON by this stream | ✅ `raw/currency_audit.json` (17 items) → rendered as §0F |
| 2 | **External field** | 67 rows re-read from banked PDFs (library key · table · page); sha256 re-checked; two missing primaries banked (`2603.25946`, `2604.03581`) | ✅ `products/P7-TanitEval/benchmarks/published_results.json` + `raw/external_field_sources.json` |
| 3 | **Verify the external rows** | independent script: page-token presence per cited page + two deliberate-regression arms | ✅ 66/66 true arm, M1 66/66 RED (after adding row windows — first run 59/66, bar NOT moved), M2 56/66 RED → `raw/verify_published_against_pdfs.json` |
| 4 | **The generator** | `taniteval/taniteval/leaderboard/**`: pointer-only config, W1-contract reader (`contract.validate_summary`), legacy→W1 converter, markdown renderer with the rules as guards, marker splice preserving CRLF, HTML via W5's components | ✅ `python -m taniteval.leaderboard build` |
| 5 | **Acceptance** | delete the section → build → byte-identical, in both deletion modes; idempotent rebuild; hand-written bytes outside the markers unchanged | ✅ 25/25 in `taniteval/tests/test_leaderboard.py` |
| 6 | **Page content** | 5 October position · four families per arm · §0E per protocol · §0F audit · W-18 loop stamps · stale-claim annotations · W-23…W-26 | ✅ **in the working copy only** — `raw/LEADERBOARD.proposed.md` + `.diff` |
| 7 | **Stage + manifest** | exact paths, blob-verified, re-verified at the end of the turn | ✅ see RESULT.md manifest |

**Not done, deliberately:** no re-measurement, no GPU, no submission; no edit to `LEADERBOARD.md` (step 0),
`MODEL_REGISTRY.md`, `GOALS_AND_CLAIMS.md`, `CRITERIA_REGISTRY.json`, W1's `bench/**` or W5's `benchreport/**`;
the 13 T1 "closed-loop" mislabel sites are listed (W-23), not rewritten.
